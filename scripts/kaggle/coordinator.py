"""Prepared finite remote epoch publication phases. No daemon and no job launch."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import json
import os
import runpy
import stat
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
NODE_SCRATCH = Path("/tmp")


def upload_location(args, journal, checkpoint_sha, transport=None):
    """Keep the journal/ledger shared; bind a private upload ZIP to its owning node."""
    root = getattr(args, "node_local_upload_root", None)
    previous = (transport or {}).get("node_local_upload")
    if root is None:
        if previous is not None:
            raise ValueError("Node-local transport cannot fall back to shared storage")
        return journal / "upload", None
    root = Path(root)
    if (
        not root.is_absolute()
        or root.parent.resolve() != NODE_SCRATCH.resolve()
        or root.name != "rsna-kaggle-upload-" + checkpoint_sha[:12]
        or root.is_symlink()
    ):
        raise ValueError("Upload root must be the exact private checkpoint directory in /tmp")
    if args.action == "prepare-transport":
        root.mkdir(mode=0o700, exist_ok=False)
    info = root.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Node-local upload directory ownership/permissions differ")
    binding = {
        "root": str(root),
        "host": os.uname().nodename,
        "uid": info.st_uid,
        "device": info.st_dev,
        "inode": info.st_ino,
    }
    if args.action != "prepare-transport" and previous != binding:
        raise ValueError("Node-local upload moved or node/identity changed")
    return root / "upload", binding


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
    manifest = json.loads((args.operation.parent / "dataset/bundle-manifest.json").read_text())
    admission = None
    if op.get("fitting_admission_sha256") is not None:
        admission = args.operation.parent / "dataset/fitting-admission.json"
        if (
            p.digest(admission) != op["fitting_admission_sha256"]
            or manifest.get("fitting_admission_sha256") != op["fitting_admission_sha256"]
        ):
            raise ValueError("Reviewed fitting admission changed")
    elif manifest.get("fitting_admission_sha256") is not None:
        raise ValueError("Reviewed fitting admission missing from operation")
    meta, _, _ = (
        workflow.validate_snapshot(args.snapshot, args.projection)
        if admission is None
        else workflow.validate_snapshot(args.snapshot, args.projection, fitting_admission=admission)
    )
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


def supersede_failed_submission_kernel(workflow, remote, db, op, expected_old_sha):
    """Archive one scored-as-failed code attempt before a distinct correction."""
    db.execute("BEGIN IMMEDIATE")
    workflow.verify_account(db, remote)
    entry = workflow.read_entry(db, op["checkpoint_sha256"])
    old = entry.get("operation", {})
    if (
        entry.get("state") != "submitted"
        or entry.get("operation_sha256") != expected_old_sha
        or not entry.get("submission_id")
        or not entry.get("kernel_version")
        or old.get("kernel_slug") == op["kernel_slug"]
        or old.get("dataset_slug") == op["dataset_slug"]
    ):
        raise ValueError("Only an exact, distinct failed code submission can be corrected")
    found = [
        row
        for row in remote.submissions()
        if row.get("ref") == entry["submission_id"]
        and row.get("description") == entry.get("description")
    ]
    if (
        len(found) != 1
        or workflow.submission_outcome(found[0]) != "failed"
        or found[0].get("status") != "COMPLETE"
        or found[0].get("totalBytes") != 0
        or found[0].get("publicScore") is not None
        or workflow.check_kernel(remote, entry, entry["kernel_version"]).get("status") != "COMPLETE"
    ):
        raise ValueError("Prior hidden failure or exact public notebook is unverified")
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
            "state": "failed_hidden_submission",
            "submission_id": entry["submission_id"],
            "description": entry["description"],
            "attempted_at": entry["attempted_at"],
            "remote_status": found[0]["status"],
            "remote_total_bytes": found[0]["totalBytes"],
            "remote_outcome": "failed",
        },
    ]
    for key in (
        "kernel_version",
        "kernel_absence_proof",
        "dataset",
        "operation",
        "operation_sha256",
        "submission_id",
        "description",
        "attempted_at",
        "submission_identity",
        "submitted_at",
        "remote_status",
        "remote_error_description",
        "remote_total_bytes",
        "remote_outcome",
        "public_score",
        "inference_receipt",
        "inference_receipt_sha256",
    ):
        entry.pop(key, None)
    entry["state"] = "queued"
    workflow.save_entry(db, entry, "preserved exact failed hidden submission for corrected push")


def dataset_publication_binding(op, operation_file, original_file):
    """Reuse one byte-verified Dataset only for a runtime/identity-only notebook correction."""
    expected = op.get("dataset_publication_operation_sha256")
    if expected is None:
        if original_file is not None:
            raise ValueError("Unexpected original Dataset operation")
        return p.digest(operation_file)
    if original_file is None or p.digest(original_file) != expected:
        raise ValueError("Original Dataset operation proof is missing or changed")
    original = json.loads(original_file.read_text())
    extras = {
        "runtime_image",
        "runtime_reference_checkpoint_sha256",
        "dataset_publication_operation_sha256",
    }
    if (
        set(op) - extras != set(original)
        or any(
            op[k] != original[k]
            for k in original
            if k not in {"kernel_slug", "kernel_metadata_sha256"}
        )
        or op["kernel_slug"] == original["kernel_slug"]
    ):
        raise ValueError("Runtime correction changed the byte-verified Dataset or inference source")
    return expected


def supersede_uncertain_private_placeholder(workflow, remote, db, op, expected_old_sha):
    """Preserve an unresolved placeholder; authorize one distinct notebook identity."""
    db.execute("BEGIN IMMEDIATE")
    workflow.verify_account(db, remote)
    entry = workflow.read_entry(db, op["checkpoint_sha256"])
    old = entry.get("operation", {})
    if (
        entry.get("state") != "push_uncertain"
        or entry.get("kernel_version") is not None
        or entry.get("submission_id") is not None
        or entry.get("description") is not None
        or entry.get("operation_sha256") != expected_old_sha
        or op.get("dataset_publication_operation_sha256") != expected_old_sha
        or old.get("kernel_slug") == op["kernel_slug"]
        or old.get("dataset_slug") != op.get("dataset_slug")
        or old.get("manifest_sha256") != op.get("manifest_sha256")
        or old.get("kernel_source_sha256") != op.get("kernel_source_sha256")
        or old.get("fitting_admission_sha256") != op.get("fitting_admission_sha256")
    ):
        raise ValueError("Only an exact unsubmitted uncertain identity can be preserved")
    owner, slug = old["kernel_slug"].split("/")
    try:
        remote.call(
            "kernels.KernelsApiService", "GetKernel", {"userName": owner, "kernelSlug": slug}
        )
    except workflow.RemoteError as error:
        if str(error) != "Kaggle GetKernel HTTP 403":
            raise
    else:
        raise ValueError("Original notebook is accessible; reconcile its exact version instead")
    result = remote.call(
        "kernels.KernelsApiService",
        "ListKernels",
        {
            "group": "PROFILE",
            "page": 1,
            "pageSize": 100,
            "language": "all",
            "kernelType": "all",
            "outputType": "all",
            "search": slug,
        },
    )
    placeholder = {"title": "[Private Notebook]", "lastRunTime": "2010-04-01T00:00:00Z"}
    if result != {"kernels": [placeholder]}:
        raise ValueError("Original private placeholder observation differs")
    # This does not classify the original push as failed or retry its slug.
    # Dataset/source/runtime-only identity binding is separately checked by run().
    history = entry.get("superseded_operations", [])
    if not isinstance(history, list) or any(
        item.get("operation_sha256") == expected_old_sha for item in history
    ):
        raise ValueError("Original uncertain identity already preserved")
    entry["superseded_operations"] = [
        *history,
        {
            "operation": old,
            "operation_sha256": expected_old_sha,
            "state": "push_uncertain_never_retry",
            "kernel_version": None,
            "dataset": entry["dataset"],
            "original_absence_proof": entry["kernel_absence_proof"],
            "readonly_placeholder_observation": result,
            "checked_at": workflow.utcnow(),
        },
    ]
    for key in ("kernel_absence_proof", "dataset", "operation", "operation_sha256"):
        entry.pop(key, None)
    entry["state"] = "queued"
    workflow.save_entry(db, entry, "preserved uncertain slug before one distinct identity attempt")


def supersede_failed_public_runtime(workflow, remote, db, op, expected_old_sha):
    """Archive exact unsubmitted public ERROR; pin the runtime of an accepted reference."""
    db.execute("BEGIN IMMEDIATE")
    workflow.verify_account(db, remote)
    entry = workflow.read_entry(db, op["checkpoint_sha256"])
    old = entry.get("operation", {})
    if (
        entry.get("state") != "pushed"
        or entry.get("submission_id") is not None
        or entry.get("operation_sha256") != expected_old_sha
        or op.get("dataset_publication_operation_sha256") != expected_old_sha
        or not entry.get("kernel_version")
        or old.get("kernel_slug") == op["kernel_slug"]
    ):
        raise ValueError("Only the exact unsubmitted failed public runtime can be superseded")
    if workflow.check_kernel(remote, entry, entry["kernel_version"]).get("status") != "ERROR":
        raise ValueError("Original public notebook is not terminal ERROR")
    reference = workflow.read_entry(db, op["runtime_reference_checkpoint_sha256"])
    if (
        workflow.check_kernel(remote, reference, reference["kernel_version"]).get("status")
        != "COMPLETE"
    ):
        raise ValueError("Reference runtime has no completed exact public execution")
    reference_op = reference["operation"]
    metadata = remote.kernel("GetKernel", reference_op["kernel_slug"], reference["kernel_version"])[
        "metadata"
    ]
    receipt = remote.output_json(
        reference_op["kernel_slug"], reference["kernel_version"], "kaggle-inference-receipt.json"
    )
    if (
        metadata.get("dockerImage") != op.get("runtime_image")
        or reference_op.get("shared_wheelhouse") != op.get("shared_wheelhouse")
        or receipt.get("status") != "offline_inference_complete"
        or receipt.get("checkpoint_sha256") != reference["checkpoint_sha256"]
        or receipt.get("bundle_manifest_sha256") != reference_op["manifest_sha256"]
    ):
        raise ValueError("Runtime pin does not bind the accepted offline reference")
    history = entry.get("superseded_operations", [])
    if not isinstance(history, list) or any(
        x.get("operation_sha256") == expected_old_sha for x in history
    ):
        raise ValueError("Original runtime failure is already preserved or ambiguous")
    entry["superseded_operations"] = [
        *history,
        {
            "operation": old,
            "operation_sha256": expected_old_sha,
            "kernel_version": entry["kernel_version"],
            "dataset": entry["dataset"],
            "state": "failed_public_runtime_unsubmitted",
            "remote_status": "ERROR",
            "accepted_runtime_reference_checkpoint_sha256": reference["checkpoint_sha256"],
            "accepted_runtime_image": op["runtime_image"],
        },
    ]
    for key in (
        "kernel_version",
        "kernel_absence_proof",
        "dataset",
        "operation",
        "operation_sha256",
    ):
        entry.pop(key, None)
    entry["state"] = "queued"
    workflow.save_entry(
        db, entry, "preserved exact failed public runtime before distinct pinned-image push"
    )


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
    if op.get("dataset_publication_operation_sha256") is not None and args.action not in {
        "push-runtime-corrected-kernel",
        "submit",
    }:
        raise ValueError(
            "Verified Dataset reuse is limited to a runtime correction and its submission"
        )
    publication_sha = dataset_publication_binding(
        op, args.operation, getattr(args, "dataset_publication_operation", None)
    )
    journal = publication_journal(base, meta, op, args.publication_revision)
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    with single_coordinator(base):
        if args.action == "prepare-transport":
            upload, node_binding = upload_location(args, journal, meta["checkpoint_sha256"])
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
            if node_binding is not None:
                record["node_local_upload"] = node_binding
            p.exclusive_json(journal / "transport.json", record)
            return record
        transport = json.loads((journal / "transport.json").read_text())
        if transport["operation_sha256"] != publication_sha or (
            op.get("dataset_publication_operation_sha256") is None
            and transport["projection_sha256"] != args.projection_sha256
        ):
            raise ValueError("Transport identity changed")
        upload, _ = upload_location(args, journal, meta["checkpoint_sha256"], transport)
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
                upload,
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
            or proof.get("operation_sha256") != publication_sha
            or proof.get("intent_sha256") != p.digest(journal / "intent.json")
        ):
            raise ValueError("Full byte verification is missing or mismatched")
        # Recheck current Dataset identity; the notebook rehashes every mounted file.
        if p.metadata(remote, op["dataset_slug"]) != {
            k: proof[k] for k in ("id", "version", "slug", "private")
        }:
            raise ValueError("Dataset version or identity drifted")
        with workflow.ledger(base / "account-ledger.sqlite3", remote.username) as db:
            if op.get("fitting_admission_sha256") is not None:
                workflow.enqueue(
                    db,
                    args.snapshot,
                    repo=args.projection,
                    fitting_admission=args.operation.parent / "dataset/fitting-admission.json",
                )
            else:
                workflow.enqueue(db, args.snapshot, repo=args.projection)
            if args.action == "push-corrected-kernel":
                if not args.publication_revision or not args.superseded_operation_sha256:
                    raise ValueError("Corrected push requires a revision and exact old operation")
                supersede_unsubmitted_kernel(
                    workflow, remote, db, op, args.superseded_operation_sha256
                )
                return workflow.push(remote, db, args.operation)
            if args.action == "push-after-failed-submission":
                if not args.publication_revision or not args.superseded_operation_sha256:
                    raise ValueError("Failed-submission correction needs exact prior operation")
                supersede_failed_submission_kernel(
                    workflow, remote, db, op, args.superseded_operation_sha256
                )
                return workflow.push(remote, db, args.operation)
            if args.action == "push-runtime-corrected-kernel":
                if not args.superseded_operation_sha256:
                    raise ValueError("Runtime correction requires the exact prior operation")
                workflow.kernel_request(args.operation, op)
                supersede_failed_public_runtime(
                    workflow, remote, db, op, args.superseded_operation_sha256
                )
                return workflow.push(remote, db, args.operation)
            if args.action == "push-after-uncertain-placeholder":
                if not args.superseded_operation_sha256:
                    raise ValueError("Distinct identity needs the exact uncertain operation")
                supersede_uncertain_private_placeholder(
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
            "push-after-failed-submission",
            "push-runtime-corrected-kernel",
            "push-after-uncertain-placeholder",
            "submit",
        ],
    )
    for name in ("projection", "projection-manifest", "operation", "snapshot"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--projection-sha256", required=True)
    parser.add_argument("--publication-revision", action="store_true")
    parser.add_argument("--superseded-operation-sha256")
    parser.add_argument("--dataset-publication-operation", type=Path)
    parser.add_argument("--management-python", type=Path)
    parser.add_argument("--node-local-upload-root", type=Path)
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
