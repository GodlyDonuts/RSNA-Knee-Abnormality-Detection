"""Prepared finite remote epoch publication phases. No daemon and no job launch."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

# Set this before executing any local helper. run_path reads this explicit source
# file, bypassing importlib's pre-existing bytecode cache for publication.py.
# load_workflow still rejects every unlisted file before any workflow import.
sys.dont_write_bytecode = True
p = SimpleNamespace(
    **runpy.run_path(str(Path(__file__).with_name("publication.py")), run_name="epoch_publication")
)


def load_workflow(projection, manifest_file, expected_sha):
    if p.digest(manifest_file) != expected_sha:
        raise ValueError("Projection manifest differs from root-reviewed hash")
    manifest = json.loads(manifest_file.read_text())
    for name, record in manifest["files"].items():
        p.confined(name)
        f = projection / name
        if (
            f.is_symlink()
            or not f.is_file()
            or f.stat().st_size != record["bytes"]
            or p.digest(f) != record["sha256"]
        ):
            raise ValueError("Exact Git projection changed")
    paths = list(projection.rglob("*"))
    if any(x.is_symlink() for x in paths):
        raise ValueError("Projection links are forbidden")
    actual = {str(x.relative_to(projection)) for x in paths if x.is_file()}
    if actual != set(manifest["files"]):
        raise ValueError("Projection contains extra/missing files, including cached or native code")
    # -B alone does not prevent reading existing bytecode. Exact membership above
    # rejects all unbound pyc/so; this prevents writing caches for later phases.
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "epoch_workflow", projection / "scripts/kaggle/workflow.py"
    )
    workflow = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(workflow)
    return workflow, manifest


def remote_from_stdin(workflow):
    import requests

    value = json.load(sys.stdin)
    if (
        set(value) != {"token"}
        or not isinstance(value["token"], str)
        or not value["token"].startswith("KGAT_")
    ):
        raise ValueError("Exactly one access token is required on stdin")
    token = value["token"]
    remote = workflow.Remote.__new__(workflow.Remote)
    remote.session = requests.Session()
    remote.session.headers["Authorization"] = "Bearer " + token
    identity = remote.call("security.OAuthService", "IntrospectToken", {"token": token})
    if identity.get("active") is not True or identity.get("username") != "godlydonuts":
        raise ValueError("Authorized account identity differs")
    remote.username = identity["username"]
    return remote, token


@contextlib.contextmanager
def single_coordinator(base):
    """Atomic shared-directory claim, deliberately no automatic stale lock stealing."""
    claim = base / "coordinator-claim"
    claim.mkdir(mode=0o700)
    try:
        p.exclusive_json(
            claim / "owner.json",
            {
                "pid": os.getpid(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "host": os.uname().nodename,
                "recovery": "manual after scheduler and intent reconciliation",
            },
        )
        yield
    finally:
        # Normal return releases claim. Killed process leaves it for explicit investigation.
        (claim / "owner.json").unlink()
        claim.rmdir()


def validate_operation(workflow, args):
    op = workflow.load_operation(args.operation)
    meta, _, _ = workflow.validate_snapshot(args.snapshot, args.projection)
    manifest = json.loads((args.operation.parent / "dataset/bundle-manifest.json").read_text())
    if (
        op["checkpoint_sha256"] != meta["checkpoint_sha256"]
        or op["epoch"] != meta["epoch"]
        or manifest["snapshot_sidecar_sha256"] != p.digest(args.snapshot)
        or manifest["training_binding_sha256"] != meta["binding_sha256"]
        or op.get("shared_wheelhouse") is None
    ):
        raise ValueError("Operation/snapshot binding differs or shared runtime absent")
    kernel_meta = json.loads((args.operation.parent / "kernel/kernel-metadata.json").read_text())
    if kernel_meta["title"] != op["kernel_slug"].split("/")[1].replace("-", " "):
        raise ValueError("Kernel title must reproduce its intended dedicated slug")
    files = p.expected_files(args.operation.parent / "dataset", manifest)
    return op, meta, files


def publication_journal(base, meta, op, revision):
    original = base / "dataset-publications" / meta["checkpoint_sha256"]
    if not revision:
        return original
    proof_file = original / "verified.json"
    if not proof_file.is_file():
        raise ValueError("The original Dataset must be fully verified before correction")
    proof = json.loads(proof_file.read_text())
    if (
        proof.get("status") != "all_payload_bytes_verified"
        or proof.get("slug") == op["dataset_slug"]
    ):
        raise ValueError("Correction must use a distinct Dataset after full original verification")
    return original.with_name(original.name + "-" + op["manifest_sha256"][:12])


def supersede_unsubmitted_kernel(workflow, remote, db, op, expected_old_sha):
    """Preserve an old notebook before one corrected push attempt."""
    db.execute("BEGIN IMMEDIATE")
    workflow.verify_account(db, remote)
    entry = workflow.read_entry(db, op["checkpoint_sha256"])
    old = entry.get("operation", {})
    if (
        entry.get("state") != "pushed"
        or entry.get("submission_id") is not None
        or entry.get("operation_sha256") != expected_old_sha
        or old.get("kernel_slug") == op["kernel_slug"]
        or old.get("dataset_slug") == op["dataset_slug"]
        or not entry.get("kernel_version")
    ):
        raise ValueError("Only an exact unsubmitted old notebook can be superseded")
    state = workflow.check_kernel(remote, entry, entry["kernel_version"])
    if state.get("status") != "COMPLETE":
        raise ValueError("Old notebook state differs from the verified public run")
    history = entry.get("superseded_operations", [])
    if not isinstance(history, list) or any(
        item.get("operation_sha256") == expected_old_sha for item in history
    ):
        raise ValueError("Prior operation history is invalid or already preserved")
    entry["superseded_operations"] = [
        *history,
        {
            "operation": old,
            "operation_sha256": expected_old_sha,
            "kernel_version": entry["kernel_version"],
            "dataset": entry["dataset"],
            "state": "pushed_unsubmitted",
        },
    ]
    entry.pop("kernel_version")
    entry.pop("kernel_absence_proof", None)
    entry.pop("dataset", None)
    entry.pop("operation", None)
    entry.pop("operation_sha256", None)
    entry["state"] = "queued"
    workflow.save_entry(db, entry, "preserved exact unsubmitted notebook for corrected push")


def management_create(python):
    def create(token, slug, title, upload):
        request = {
            "token": token,
            "slug": slug,
            "title": title,
            "upload": str(upload),
            "archive_sha256": p.digest(upload / "bundle.zip"),
        }
        process = subprocess.run(
            [str(python), str(Path(p.__file__)), "--sdk-worker"],
            input=p.canonical(request),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=3600,
            check=False,
        )
        if process.returncode or len(process.stdout) > 8192:
            raise ValueError("Management process uncertain; inspect only sanitized journal")
        result = json.loads(process.stdout)
        if result.get("returned") is not True:
            raise ValueError("Dataset mutation result uncertain")
        return result

    return create


def run(args):
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("A bounded CPU-only allocation is required")
    base = Path.home() / "rsna-knee/kaggle"
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    workflow, projection = load_workflow(
        args.projection, args.projection_manifest, args.projection_sha256
    )
    op, meta, files = validate_operation(workflow, args)
    journal = publication_journal(base, meta, op, args.publication_revision)
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    with single_coordinator(base):
        if args.action == "prepare-transport":
            upload = journal / "upload"
            upload.mkdir(mode=0o700)
            archive = p.build_zip(args.operation.parent / "dataset", files, upload / "bundle.zip")
            record = {
                "status": "prepared_not_published",
                "archive_sha256": archive["sha256"],
                "archive_bytes": archive["bytes"],
                "checkpoint_sha256": meta["checkpoint_sha256"],
                "operation_sha256": p.digest(args.operation),
                "projection_sha256": args.projection_sha256,
                "package_git_commit": projection["package_git_commit"],
                "training_git_commit": projection["training_git_commit"],
            }
            p.exclusive_json(journal / "transport.json", record)
            return record
        transport = json.loads((journal / "transport.json").read_text())
        if (
            transport["operation_sha256"] != p.digest(args.operation)
            or transport["projection_sha256"] != args.projection_sha256
        ):
            raise ValueError("Transport identity changed")
        remote, token = remote_from_stdin(workflow)
        if args.action == "create-dataset":
            if args.management_python is None:
                raise ValueError("Separate pinned SDK management interpreter is required")
            # No new mutations while any other dataset create remains unverified.
            for other in (base / "dataset-publications").iterdir():
                if (
                    other != journal
                    and (other / "intent.json").exists()
                    and not (other / "verified.json").exists()
                ):
                    raise ValueError("Another dataset mutation awaits read-only reconciliation")
            workflow.verify_shared_wheelhouse(remote, op["shared_wheelhouse"])
            return p.create_once(
                remote,
                token,
                op["dataset_slug"],
                files,
                journal / "upload",
                journal,
                transport,
                create=management_create(args.management_python),
            )
        intent = json.loads((journal / "intent.json").read_text())
        if (
            intent["files"] != files
            or intent["context"] != transport
            or intent["slug"] != op["dataset_slug"]
        ):
            raise ValueError("Durable mutation intent differs")
        if args.action == "reconcile-dataset":
            result = p.reconcile(remote, op["dataset_slug"], files)
            result.update(
                operation_sha256=p.digest(args.operation),
                intent_sha256=p.digest(journal / "intent.json"),
            )
            existing = journal / "verified.json"
            if existing.exists():
                if json.loads(existing.read_text()) != result:
                    raise ValueError("Previously frozen verification differs")
            else:
                p.exclusive_json(existing, result)
            return result
        proof = json.loads((journal / "verified.json").read_text())
        if (
            proof.get("status") != "all_payload_bytes_verified"
            or proof.get("files") != files
            or proof.get("operation_sha256") != p.digest(args.operation)
            or proof.get("intent_sha256") != p.digest(journal / "intent.json")
        ):
            raise ValueError("Full byte verification is missing or mismatched")
        # Recheck current Dataset identity; the notebook rehashes every mounted file.
        if p.metadata(remote, op["dataset_slug"]) != {
            k: proof[k] for k in ("id", "version", "slug", "private")
        }:
            raise ValueError("Dataset version or identity drifted")
        with workflow.ledger(base / "account-ledger.sqlite3", remote.username) as db:
            workflow.enqueue(db, args.snapshot, repo=args.projection)
            if args.action == "push-corrected-kernel":
                if not args.publication_revision or not args.superseded_operation_sha256:
                    raise ValueError("Corrected push requires a revision and exact old operation")
                supersede_unsubmitted_kernel(
                    workflow, remote, db, op, args.superseded_operation_sha256
                )
                return workflow.push(remote, db, args.operation)
            if args.action == "push-kernel":
                return workflow.push(remote, db, args.operation)
            if args.action == "submit":
                entry = workflow.read_entry(db, meta["checkpoint_sha256"])
                if entry.get("operation_sha256") != p.digest(args.operation):
                    raise ValueError(
                        "Submission operation differs from the exact current ledger entry"
                    )
                return workflow.submit(remote, db, meta["checkpoint_sha256"])
            raise ValueError("Unsupported phase")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=[
            "prepare-transport",
            "create-dataset",
            "reconcile-dataset",
            "push-kernel",
            "push-corrected-kernel",
            "submit",
        ],
    )
    for name in ("projection", "projection-manifest", "operation", "snapshot"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--projection-sha256", required=True)
    parser.add_argument("--publication-revision", action="store_true")
    parser.add_argument("--superseded-operation-sha256")
    parser.add_argument("--management-python", type=Path)
    args = parser.parse_args()
    try:
        result = run(args)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        print(
            json.dumps(
                {"status": "blocked", "error_type": type(exc).__name__, "automatic_retry": False}
            )
        )
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
