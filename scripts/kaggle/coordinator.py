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


def node_journal(args, base, meta, op):
    """Opt-in node-local journal for new components or exact daily-queue epochs.

    The account ledger and coordinator claim stay shared. Two small shared
    markers are the durable cross-node record: an exclusive intent (written
    before the single create RPC) and the full-byte verification. Existing
    shared journals are never moved, and a marked operation cannot fall back.
    """
    root = getattr(args, "node_local_journal_root", None)
    shared = base / "dataset-publications"
    key = meta["checkpoint_sha256"] + "-" + op["manifest_sha256"][:12]
    intent = shared / (key + ".node-intent.json")
    verified = shared / (key + ".node-verified.json")
    if root is None:
        if intent.exists() or verified.exists():
            raise ValueError("Node-local journal cannot fall back to shared storage")
        return None
    root = Path(root)
    checkpoint = meta["checkpoint_sha256"]
    admission_sha = op.get("fitting_admission_sha256")
    daily_epoch = (
        op.get("common_bundle") is None
        and isinstance(admission_sha, str)
        and len(admission_sha) == 64
        and all(c in "0123456789abcdef" for c in admission_sha)
        and any(
            op.get("dataset_slug") == f"godlydonuts/rsna-alldev-e{epoch:03d}-{checkpoint[:12]}"
            and op.get("kernel_slug")
            == f"godlydonuts/rsna-alldev-offline-e{epoch:03d}-{checkpoint[:12]}"
            for epoch in range(1, 11)
        )
    )
    if (
        (op.get("common_bundle") is None and not daily_epoch)
        or getattr(args, "publication_revision", False)
        or (shared / meta["checkpoint_sha256"]).exists()
        or (shared / key).exists()
    ):
        raise ValueError("Node-local journals are limited to new component operations")
    if (
        not root.is_absolute()
        or root.parent.resolve() != NODE_SCRATCH.resolve()
        or root.name != "rsna-kaggle-journal-" + key[:12] + key[64:]
        or root.is_symlink()
    ):
        raise ValueError("Journal root must be the exact private operation directory in /tmp")
    if args.action == "prepare-transport":
        if intent.exists() or verified.exists():
            raise ValueError("Operation already has a durable shared intent")
        shared.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.mkdir(mode=0o700, exist_ok=False)
    elif not root.exists():
        if args.action == "create-dataset" or not intent.exists():
            raise ValueError("Node-local journal is missing; only marker reconciliation remains")
        return SimpleNamespace(root=None, binding=None, intent=intent, verified=verified)
    info = root.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Node-local journal ownership/permissions differ")
    binding = {
        "root": str(root),
        "host": os.uname().nodename,
        "uid": info.st_uid,
        "device": info.st_dev,
        "inode": info.st_ino,
    }
    return SimpleNamespace(root=root, binding=binding, intent=intent, verified=verified)


def pending_dataset_mutation(base, journal, own_marker=None):
    """Any unverified shared journal intent or node-local intent marker blocks creates."""
    for other in (base / "dataset-publications").iterdir():
        if other.name.endswith(".node-intent.json"):
            done = other.with_name(other.name[: -len(".node-intent.json")] + ".node-verified.json")
            if other != own_marker and not done.exists():
                return True
        elif (
            other != journal
            and other.is_dir()
            and (other / "intent.json").exists()
            and not (other / "verified.json").exists()
        ):
            return True
    return False


COMMON_ACTIONS = ("prepare-common-transport", "create-common-dataset", "reconcile-common-dataset")
COMPONENT_PROJECTION = (
    "scripts/kaggle/components.py",
    "scripts/kaggle/coordinator.py",
    "scripts/kaggle/publication.py",
    "scripts/kaggle/workflow.py",
    "scripts/kaggle/notebook_entry.py",
    "scripts/kaggle/chunked_infer.py",
    "scripts/kaggle/project_code.py",
)


def require_component_projection(projection):
    """Component phases run only from one projection holding this coordinator's own bytes."""
    files = projection.get("files", {})
    if any(name not in files for name in COMPONENT_PROJECTION) or any(
        p.digest(Path(__file__).with_name(Path(name).name)) != files[name]["sha256"]
        for name in ("scripts/kaggle/coordinator.py", "scripts/kaggle/publication.py")
    ):
        raise ValueError("Component phases need one projection with every coordinator dependency")


def common_markers(base, op):
    shared = base / "dataset-publications"
    key = "common-" + op["manifest_sha256"]
    return shared, shared / (key + ".node-intent.json"), shared / (key + ".node-verified.json")


def common_journal(args, base, op):
    """Node-local journal for one common Dataset create; two small shared markers."""
    shared, intent, verified = common_markers(base, op)
    root = getattr(args, "node_local_journal_root", None)
    if root is None:
        raise ValueError("Common Dataset phases require their node-local journal root")
    root = Path(root)
    if (
        not root.is_absolute()
        or root.parent.resolve() != NODE_SCRATCH.resolve()
        or root.name != "rsna-kaggle-common-journal-" + op["manifest_sha256"][:12]
        or root.is_symlink()
    ):
        raise ValueError("Journal root must be the exact private common directory in /tmp")
    if args.action == "prepare-common-transport":
        if intent.exists() or verified.exists():
            raise ValueError("Common Dataset already has a durable shared intent")
        shared.mkdir(mode=0o700, parents=True, exist_ok=True)
        root.mkdir(mode=0o700, exist_ok=False)
    elif not root.exists():
        raise ValueError("Node-local journal is missing; only marker reconciliation remains")
    info = root.stat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise ValueError("Node-local journal ownership/permissions differ")
    binding = {
        "root": str(root),
        "host": os.uname().nodename,
        "uid": info.st_uid,
        "device": info.st_dev,
        "inode": info.st_ino,
    }
    return SimpleNamespace(root=root, binding=binding, intent=intent, verified=verified)


def run_common(args):
    """Bounded once-only private common Dataset create and full-byte reconciliation.

    No checkpoint ledger entry, kernel or submission. Any intent marker makes
    the create permanently read-only: unknown outcomes are reconciled, never
    re-created, and a different node cannot create the same common Dataset.
    """
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("A bounded CPU-only allocation is required")
    base = Path.home() / "rsna-knee/kaggle"
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    workflow, projection = load_workflow(
        args.projection, args.projection_manifest, args.projection_sha256
    )
    require_component_projection(projection)
    components = workflow.components
    if args.action == "reconcile-common-dataset":
        return reconcile_common(args, base, workflow)
    op, manifest = components.load_common_operation(args.common_operation)
    tree = args.common_operation.parent / op["bundle_path"]
    files = p.expected_files(tree, manifest, components.COMMON_MANIFEST)
    if files[components.COMMON_MANIFEST] != {
        "sha256": op["manifest_sha256"],
        "bytes": op["manifest_bytes"],
    }:
        raise ValueError("Common manifest bytes differ from the operation")
    operation_sha = p.digest(args.common_operation)
    local = common_journal(args, base, op)
    with single_coordinator(base):
        if args.action == "prepare-common-transport":
            upload = local.root / "upload"
            upload.mkdir(mode=0o700)
            archive = p.build_zip(tree, files, upload / "bundle.zip")
            tree_bytes = sum(row["bytes"] for row in files.values()) + (
                (tree / "dataset-metadata.json").stat().st_size
            )
            record = {
                "status": "prepared_not_published",
                "kind": "common_dataset_transport",
                "archive_sha256": archive["sha256"],
                "archive_bytes": archive["bytes"],
                "common_manifest_sha256": op["manifest_sha256"],
                "common_operation_sha256": operation_sha,
                "projection_sha256": args.projection_sha256,
                "package_git_commit": projection["package_git_commit"],
                "training_git_commit": projection["training_git_commit"],
                "node_local_journal": local.binding,
                "common_tree_node": op["node"],
                "footprint": {
                    "node_local_common_tree_bytes": tree_bytes,
                    "node_local_upload_zip_bytes": archive["bytes"],
                    "shared_marker_files": 2,
                    "shared_marker_names": [local.intent.name, local.verified.name],
                    "deletion_credit": False,
                },
                "checkpoint_entry": None,
                "competition_submission": False,
            }
            p.exclusive_json(local.root / "transport.json", record)
            return record
        if args.action != "create-common-dataset":
            raise ValueError("Unsupported common phase")
        transport = json.loads((local.root / "transport.json").read_text())
        if (
            transport.get("node_local_journal") != local.binding
            or transport.get("common_operation_sha256") != operation_sha
            or transport.get("projection_sha256") != args.projection_sha256
        ):
            raise ValueError("Node-local journal moved or node/identity changed")
        remote, token = remote_from_stdin(workflow)
        if op["account"] != remote.username:
            raise ValueError("Common Dataset belongs to another account")
        ledger_file = base / "account-ledger.sqlite3"
        if not ledger_file.is_file():
            raise ValueError("The canonical shared account ledger must already exist")
        with workflow.ledger(ledger_file, remote.username) as db:
            workflow.verify_account(db, remote)
        if args.management_python is None:
            raise ValueError("Separate pinned SDK management interpreter is required")
        if local.intent.exists() or local.verified.exists():
            raise ValueError("Common create already has a durable intent; reconcile only")
        if pending_dataset_mutation(base, None, local.intent):
            raise ValueError("Another dataset mutation awaits read-only reconciliation")
        workflow.verify_shared_wheelhouse(remote, op["shared_wheelhouse"])
        # Exclusive shared intent precedes the one create; never retried. It
        # embeds the full original operation so reconciliation needs no node.
        p.exclusive_json(
            local.intent,
            {
                "schema_version": 1,
                "kind": COMMON_INTENT_KIND,
                "account": op["account"],
                "slug": op["dataset_slug"],
                "dataset_version": 1,
                "private": True,
                "manifest_name": components.COMMON_MANIFEST,
                "files": files,
                "operation": op,
                "transport": transport,
                "node_local_journal": local.binding,
                "common_operation_sha256": operation_sha,
                "retry_prohibited": True,
                "checkpoint_entry": None,
                "competition_submission": False,
            },
        )
        return p.create_once(
            remote,
            token,
            op["dataset_slug"],
            files,
            local.root / "upload",
            local.root,
            transport,
            create=management_create(args.management_python),
        )


COMMON_INTENT_KIND = "node_local_common_dataset_create_intent"
COMMON_INTENT_KEYS = {
    "schema_version",
    "kind",
    "account",
    "slug",
    "dataset_version",
    "private",
    "manifest_name",
    "files",
    "operation",
    "transport",
    "node_local_journal",
    "common_operation_sha256",
    "retry_prohibited",
    "checkpoint_entry",
    "competition_submission",
}


def common_intent_files(components, marker, op, operation_sha):
    """Bind a shared create intent to the exact original operation; no local tree access."""
    files = marker.get("files") if isinstance(marker, dict) else None
    if (
        not isinstance(marker, dict)
        or set(marker) != COMMON_INTENT_KEYS
        or marker["schema_version"] != 1
        or marker["kind"] != COMMON_INTENT_KIND
        or marker["common_operation_sha256"] != operation_sha
        or marker["operation"] != op
        or marker["account"] != op["account"]
        or marker["slug"] != op["dataset_slug"]
        or marker["dataset_version"] != 1
        or marker["private"] is not True
        or marker["manifest_name"] != components.COMMON_MANIFEST
        or marker["retry_prohibited"] is not True
        or marker["checkpoint_entry"] is not None
        or marker["competition_submission"] is not False
        or not isinstance(files, dict)
        or files.get(components.COMMON_MANIFEST)
        != {"sha256": op["manifest_sha256"], "bytes": op["manifest_bytes"]}
    ):
        raise ValueError("Durable shared common intent differs from the original operation")
    rows = {k: v for k, v in files.items() if k != components.COMMON_MANIFEST}
    components.validate_table(rows)
    if (
        components.object_hash(rows) != op["files_sha256"]
        or len(rows) != op["member_count"]
        or sum(row["bytes"] for row in rows.values()) != op["total_bytes"]
        or rows.get("wheelhouse.json", {}).get("sha256") != op["wheelhouse_manifest_sha256"]
    ):
        raise ValueError("Durable shared common intent member table differs")
    return files


def reconcile_common(args, base, workflow):
    """Strictly read-only marker reconciliation, independent of node-local state.

    Reads only the reviewed operation file (any path; its bytes must hash to the
    intent's ``common_operation_sha256``), the shared intent marker, the shared
    ledger and the remote. It never reads the common tree, the node-local
    journal or the host identity, so it works after complete loss of the
    creating node and from any host. It never creates.
    """
    components = workflow.components
    operation_sha = p.digest(args.common_operation)
    op = components.validate_common_operation(json.loads(args.common_operation.read_text()))
    _, intent, verified = common_markers(base, op)
    with single_coordinator(base):
        if not intent.is_file():
            raise ValueError("No durable common create intent exists to reconcile")
        intent_sha = p.digest(intent)
        files = common_intent_files(components, json.loads(intent.read_text()), op, operation_sha)
        remote, _ = remote_from_stdin(workflow)
        if op["account"] != remote.username:
            raise ValueError("Common Dataset belongs to another account")
        ledger_file = base / "account-ledger.sqlite3"
        if not ledger_file.is_file():
            raise ValueError("The canonical shared account ledger must already exist")
        with workflow.ledger(ledger_file, remote.username) as db:
            workflow.verify_account(db, remote)
        result = p.reconcile(
            remote, op["dataset_slug"], files, manifest_name=components.COMMON_MANIFEST
        )
        if (
            result.get("files") != files
            or result.get("slug") != op["dataset_slug"]
            or result.get("version") != 1
            or result.get("private") is not True
            or result.get("downloaded_all_bytes") is not True
            or p.digest(intent) != intent_sha
        ):
            raise ValueError("Remote common Dataset differs from the durable intent")
        result.update(
            common_operation_sha256=operation_sha,
            intent_sha256=intent_sha,
            checkpoint_entry=None,
        )
        if verified.exists():
            if json.loads(verified.read_text()) != result:
                raise ValueError("Previously frozen verification differs")
        else:
            p.exclusive_json(verified, result)
        return result


def require_common_proof(base, op):
    """A component checkpoint binds the coordinator's own frozen common verification."""
    record = op.get("common_bundle")
    if record is None:
        return
    _, _, verified = common_markers(base, record)
    if not verified.is_file() or p.digest(verified) != record.get("verification_sha256"):
        raise ValueError("Common Dataset lacks its coordinator full-byte verification marker")


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


def inference_only_dataset_reuse(original, corrected, expected_old_sha):
    """Allow new inference orchestration around the exact previously verified Dataset."""
    expected = {
        "original_operation_sha256": expected_old_sha,
        "engine": "two_device_exact_FP32_prefetch_v1",
        "precision": "fp32",
        "original_dataset_bytes_preserved": True,
        "original_model_source_and_weights_preserved": True,
    }
    if corrected.get("inference_runtime_correction") != expected:
        raise ValueError("Explicit exact-FP32 Dataset reuse binding required")
    changed = {
        "kernel_slug",
        "kernel_source_sha256",
        "kernel_metadata_sha256",
        "prepared_at",
        "dataset_publication_operation_sha256",
        "inference_runtime_correction",
        "reused_dataset_verification_sha256",
        "reused_dataset_id",
    }
    if (
        corrected.get("dataset_publication_operation_sha256") != expected_old_sha
        or set(corrected) - changed != set(original) - changed
        or any(corrected[k] != original[k] for k in original if k not in changed)
        or corrected["kernel_slug"] == original["kernel_slug"]
        or corrected["kernel_source_sha256"] == original["kernel_source_sha256"]
    ):
        raise ValueError("Inference correction changed the original Dataset/model binding")


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
    ):
        raise ValueError("Only an exact, distinct failed code submission can be corrected")
    if old.get("dataset_slug") == op["dataset_slug"]:
        inference_only_dataset_reuse(old, op, expected_old_sha)
        if op.get("reused_dataset_id") != entry["dataset"].get("id"):
            raise ValueError("Original verified Dataset identity must remain unchanged")
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
            "remote_error_description": found[0].get("errorDescription"),
            "public_score": found[0].get("publicScore"),
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
    if op.get("inference_runtime_correction") is not None:
        inference_only_dataset_reuse(original, op, expected)
        return expected
    extras = {
        "runtime_image",
        "runtime_reference_checkpoint_sha256",
        "dataset_publication_operation_sha256",
    }
    if (
        set(op) - extras != set(original) - extras
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


DRAFT_ATTESTATION_KEYS = {
    "schema_version",
    "kind",
    "account",
    "kernel_slug",
    "observed_at",
    "editor_opened",
    "version_history",
    "inputs_match_operation",
    "code_matches_operation",
    "ui_save_version_performed",
    "save_kernel_retry_performed",
}
DRAFT_ATTESTATION_MAX_AGE_SECONDS = 900
PLACEHOLDER = {"title": "[Private Notebook]", "lastRunTime": "2010-04-01T00:00:00Z"}
DRAFT_CLASSIFICATIONS = {"private_draft_owner_attested", "private_draft_accessible_owner_attested"}


def draft_attestation_valid(attestation, account, slug, now):
    """Owner's authenticated-UI evidence of an unversioned draft at the exact slug."""
    from datetime import datetime

    if not isinstance(attestation, dict) or set(attestation) != DRAFT_ATTESTATION_KEYS:
        return False
    try:
        observed = datetime.fromisoformat(attestation["observed_at"])
        age = (now - observed).total_seconds()
    except (TypeError, ValueError):
        return False
    return (
        attestation["schema_version"] == 1
        and attestation["kind"] == "owner_authenticated_ui_private_draft_observation"
        and attestation["account"] == account
        and attestation["kernel_slug"] == slug
        and 0 <= age <= DRAFT_ATTESTATION_MAX_AGE_SECONDS
        and attestation["editor_opened"] is True
        and attestation["version_history"] == "starting_fresh_no_saved_versions"
        and attestation["inputs_match_operation"] is True
        and attestation["code_matches_operation"] is True
        and attestation["ui_save_version_performed"] is False
        and attestation["save_kernel_retry_performed"] is False
    )


DRAFT_EXPORT_MARKER = "# %% [code]\n"


def accessible_draft_exact(workflow, found, operation, slug):
    """Exact private offline T4 draft metadata and source, minus only one generated marker."""
    import hashlib

    meta = found.get("metadata", {})
    source = found.get("blob", {}).get("source")
    if not isinstance(source, str):
        return False, None
    marker = source.startswith(DRAFT_EXPORT_MARKER)
    body = source[len(DRAFT_EXPORT_MARKER) :] if marker else source
    runtime = operation.get("runtime_image")
    exact = (
        operation.get("kernel_slug") == slug
        and meta.get("ref") == slug
        and meta.get("isPrivate") is True
        and meta.get("enableInternet") is False
        and meta.get("machineShape") == "NvidiaTeslaT4"
        and meta.get("competitionDataSources") == [workflow.COMPETITION]
        and sorted(meta.get("datasetDataSources") or [])
        == sorted(workflow.dataset_sources(operation))
        and (runtime is None or meta.get("dockerImage") == runtime)
        and hashlib.sha256(body.encode()).hexdigest() == operation["kernel_source_sha256"]
    )
    return exact, marker


def observe_uncertain_push(workflow, remote, slug, attestation=None, now=None, operation=None):
    """Read-only. Classify committed version / owner-attested draft / unresolved.

    The private placeholder alone proves neither absence nor a draft. An accessible
    kernel without a version is a draft only with exact metadata/source and owner evidence.
    """
    from datetime import datetime, timezone

    now = now or datetime.now(timezone.utc)
    owner, name = slug.split("/")
    try:
        found = remote.call(
            "kernels.KernelsApiService", "GetKernel", {"userName": owner, "kernelSlug": name}
        )
    except workflow.RemoteError as error:
        get_kernel = str(error)
    else:
        version = found.get("metadata", {}).get("currentVersionNumber")
        if type(version) is int and version >= 1:
            return {
                "classification": "committed_version",
                "get_kernel": "accessible",
                "current_version": version,
            }
        accessible = {
            "classification": "unresolved",
            "get_kernel": "accessible",
            "current_version": version,
            "checked_at": now.isoformat(),
        }
        if version is not None or operation is None:
            return accessible
        exact, marker = accessible_draft_exact(workflow, found, operation, slug)
        accessible.update(
            draft_metadata_and_source_exact=exact,
            draft_export_marker_removed=marker,
            owner_attestation_valid=draft_attestation_valid(
                attestation, remote.username, slug, now
            ),
        )
        if attestation is not None:
            accessible["owner_attestation_sha256"] = workflow.object_hash(attestation)
        if exact and accessible["owner_attestation_valid"]:
            accessible["classification"] = "private_draft_accessible_owner_attested"
        return accessible
    listing = remote.call(
        "kernels.KernelsApiService",
        "ListKernels",
        {
            "group": "PROFILE",
            "user": owner,
            "page": 1,
            "pageSize": 100,
            "language": "all",
            "kernelType": "all",
            "outputType": "all",
            "search": name,
        },
    )
    result = {
        "get_kernel": get_kernel,
        "placeholder_only": listing == {"kernels": [PLACEHOLDER]},
        "listing_sha256": workflow.object_hash(listing),
        "owner_attestation_valid": draft_attestation_valid(attestation, remote.username, slug, now),
        "checked_at": now.isoformat(),
    }
    if attestation is not None:
        result["owner_attestation_sha256"] = workflow.object_hash(attestation)
    draft = (
        get_kernel == "Kaggle GetKernel HTTP 403"
        and result["placeholder_only"]
        and result["owner_attestation_valid"]
    )
    result["classification"] = "private_draft_owner_attested" if draft else "unresolved"
    return result


def commit_private_draft_once(workflow, remote, db, operation_path, expected_sha, attestation):
    """One first-commit SaveKernel for the exact null-identity draft; never a new slug."""
    import hashlib

    operation = workflow.load_operation(operation_path)
    slug = operation["kernel_slug"]
    db.execute("BEGIN IMMEDIATE")
    try:
        workflow.verify_account(db, remote)
        entry = workflow.read_entry(db, operation["checkpoint_sha256"])
        response = entry.get("kernel_create_response")
        history = entry.get("superseded_operations", [])
        if (
            entry.get("state") != "push_uncertain"
            or entry.get("kernel_version") is not None
            or entry.get("submission_id") is not None
            or entry.get("description") is not None
            or "draft_commit" in entry
            or entry.get("operation_sha256") != expected_sha
            or workflow.digest(operation_path) != expected_sha
            or entry.get("operation") != operation
            or not isinstance(entry.get("kernel_absence_proof"), dict)
            or not isinstance(entry.get("dataset"), dict)
            or not isinstance(response, dict)
            or any(response.get(key) is not None for key in ("kernel_id", "ref", "version"))
            or any(item.get("operation_sha256") == expected_sha for item in history)
            or slug.split("/")[0] != remote.username
        ):
            raise ValueError("Only the exact unsubmitted null-identity push may commit its draft")
        request = workflow.kernel_request(operation_path, operation)
        if (
            request["slug"] != slug
            or hashlib.sha256(request["text"].encode()).hexdigest()
            != operation["kernel_source_sha256"]
        ):
            raise ValueError("First-commit request differs from the exact prepared source")
        observation = observe_uncertain_push(
            workflow, remote, slug, attestation, operation=operation
        )
        if observation["classification"] not in DRAFT_CLASSIFICATIONS:
            raise ValueError("No fresh exact owner-attested unversioned draft")
        if workflow.verify_remote_dataset(remote, operation) != entry["dataset"]:
            raise ValueError("Verified Dataset identity drifted")
    except Exception:
        db.rollback()
        raise
    entry["draft_commit"] = {
        "schema_version": 1,
        "kind": "first_commit_of_owner_attested_private_draft",
        "operation_sha256": expected_sha,
        "original_kernel_create_response_sha256": workflow.object_hash(response),
        "request_sha256": workflow.object_hash(request),
        "owner_attestation": attestation,
        "observation": observation,
        "expected_version": 1,
        "attempted_at": workflow.utcnow(),
        "retry_prohibited": True,
    }
    entry["state"] = "draft_commit_intent"
    workflow.save_entry(db, entry, "durable intent before one first-commit SaveKernel of draft")
    db.execute("BEGIN IMMEDIATE")
    try:
        result = remote.call("kernels.KernelsApiService", "SaveKernel", request)
        entry["draft_commit"]["response"] = workflow.save_kernel_response(result)
        if (
            result.get("versionNumber") != 1
            or type(result.get("versionNumber")) is not int
            or result.get("error")
            or result.get("ref") != "/code/" + slug
        ):
            raise workflow.RemoteError("Draft first commit needs read-only reconciliation")
        entry.update(state="pushed", kernel_version=1)
        workflow.save_entry(db, entry, "exact draft first committed as version 1")
    except Exception as exc:
        entry["draft_commit"]["exception"] = workflow.bounded_exception(exc)
        entry["state"] = "draft_commit_uncertain"
        workflow.save_entry(db, entry, "draft commit uncertain; retries prohibited")
        raise
    return entry


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
    if args.action in COMMON_ACTIONS:
        return run_common(args)
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("A bounded CPU-only allocation is required")
    base = Path.home() / "rsna-knee/kaggle"
    base.mkdir(mode=0o700, parents=True, exist_ok=True)
    workflow, projection = load_workflow(
        args.projection, args.projection_manifest, args.projection_sha256
    )
    op, meta, files = validate_operation(workflow, args)
    if op.get("common_bundle") is not None:
        require_component_projection(projection)
        require_common_proof(base, op)
    if op.get("dataset_publication_operation_sha256") is not None and args.action not in {
        "push-runtime-corrected-kernel",
        "push-after-uncertain-placeholder",
        "submit",
        "push-after-failed-submission",
    }:
        raise ValueError(
            "Verified Dataset reuse is limited to a bound notebook correction and its submission"
        )
    publication_sha = dataset_publication_binding(
        op, args.operation, getattr(args, "dataset_publication_operation", None)
    )
    local = node_journal(args, base, meta, op)
    if local is None:
        journal = publication_journal(base, meta, op, args.publication_revision)
        journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    else:
        journal = local.root
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
            if local is not None:
                record["node_local_journal"] = local.binding
            p.exclusive_json(journal / "transport.json", record)
            return record
        marker = None
        if local is not None and local.intent.exists():
            marker = json.loads(local.intent.read_text())
        if journal is None:
            transport = marker["transport"]
        else:
            transport = json.loads((journal / "transport.json").read_text())
            if local is not None and (
                transport.get("node_local_journal") != local.binding
                or (marker is not None and marker["transport"] != transport)
            ):
                raise ValueError("Node-local journal moved or node/identity changed")
        if transport["operation_sha256"] != publication_sha or (
            op.get("dataset_publication_operation_sha256") is None
            and transport["projection_sha256"] != args.projection_sha256
        ):
            raise ValueError("Transport identity changed")
        if journal is not None:
            upload, _ = upload_location(args, journal, meta["checkpoint_sha256"], transport)
        remote, token = remote_from_stdin(workflow)
        if args.action == "create-dataset":
            if args.management_python is None:
                raise ValueError("Separate pinned SDK management interpreter is required")
            # No new mutations while any other dataset create remains unverified.
            if pending_dataset_mutation(base, journal, local and local.intent):
                raise ValueError("Another dataset mutation awaits read-only reconciliation")
            workflow.verify_shared_wheelhouse(remote, op["shared_wheelhouse"])
            if op.get("common_bundle") is not None:
                workflow.verify_common_bundle(remote, op["common_bundle"])
            if local is not None:
                # Exclusive shared intent precedes the one create; never retried.
                p.exclusive_json(
                    local.intent,
                    {
                        "schema_version": 1,
                        "kind": "node_local_dataset_create_intent",
                        "slug": op["dataset_slug"],
                        "files": files,
                        "transport": transport,
                        "node_local_journal": local.binding,
                        "operation_sha256": p.digest(args.operation),
                        "retry_prohibited": True,
                        "competition_submission": False,
                    },
                )
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
        if local is not None:
            if (
                marker is None
                or marker["files"] != files
                or marker["slug"] != op["dataset_slug"]
                or marker["operation_sha256"] != p.digest(args.operation)
            ):
                raise ValueError("Durable shared node-local intent differs")
            intent_file = local.intent
        else:
            intent_file = journal / "intent.json"
        if journal is not None:
            intent = json.loads((journal / "intent.json").read_text())
            if (
                intent["files"] != files
                or intent["context"] != transport
                or intent["slug"] != op["dataset_slug"]
            ):
                raise ValueError("Durable mutation intent differs")
        verified_file = journal / "verified.json" if local is None else local.verified
        if args.action == "reconcile-dataset":
            result = p.reconcile(remote, op["dataset_slug"], files)
            result.update(
                operation_sha256=p.digest(args.operation),
                intent_sha256=p.digest(intent_file),
            )
            existing = verified_file
            if existing.exists():
                if json.loads(existing.read_text()) != result:
                    raise ValueError("Previously frozen verification differs")
            else:
                p.exclusive_json(existing, result)
            return result
        proof = json.loads(verified_file.read_text())
        if (
            proof.get("status") != "all_payload_bytes_verified"
            or proof.get("files") != files
            or proof.get("operation_sha256") != publication_sha
            or proof.get("intent_sha256") != p.digest(intent_file)
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
            if args.action == "commit-private-draft":
                if not args.uncertain_operation_sha256 or args.draft_attestation is None:
                    raise ValueError("Draft commit needs the exact operation and owner attestation")
                return commit_private_draft_once(
                    workflow,
                    remote,
                    db,
                    args.operation,
                    args.uncertain_operation_sha256,
                    json.loads(args.draft_attestation.read_text()),
                )
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
            "commit-private-draft",
            "submit",
            *COMMON_ACTIONS,
        ],
    )
    for name in ("projection", "projection-manifest"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("operation", "snapshot", "common-operation"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument("--projection-sha256", required=True)
    parser.add_argument("--publication-revision", action="store_true")
    parser.add_argument("--superseded-operation-sha256")
    parser.add_argument("--uncertain-operation-sha256")
    parser.add_argument("--draft-attestation", type=Path)
    parser.add_argument("--dataset-publication-operation", type=Path)
    parser.add_argument("--management-python", type=Path)
    parser.add_argument("--node-local-upload-root", type=Path)
    parser.add_argument("--node-local-journal-root", type=Path)
    args = parser.parse_args()
    if args.action in COMMON_ACTIONS:
        if (
            args.common_operation is None
            or (args.node_local_journal_root is not None)
            != (args.action != "reconcile-common-dataset")
            or args.operation is not None
            or args.snapshot is not None
            or args.publication_revision
            or args.superseded_operation_sha256
            or args.dataset_publication_operation
            or args.node_local_upload_root
        ):
            parser.error(
                "common phases take only --common-operation; prepare/create also take"
                " their journal root, marker-only reconcile does not"
            )
    elif args.operation is None or args.snapshot is None or args.common_operation is not None:
        parser.error("checkpoint phases require --operation and --snapshot")
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
