"""One guarded epoch-10 notebook push after an unrelated unnamed profile stub.

Run only on a CPU-only Newton allocation. The old immutable package projection
and Dataset are verified again; the separately pinned workflow changes only the
private-notebook absence guard. The Kaggle token arrives on stdin.
"""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace

R = Path("/home/sa305415/rsna-knee")
PROJECTION = R / "kaggle/code-projections/2bc2695-b740-epoch1-v1"
SOURCE = PROJECTION / "source"
PACKAGE_SHA = "098a3b351305af4166fd7fe57f33bdf8387d3f42c3b255210985f25205e6459b"
WORKFLOW_PATH = R / "kaggle/recovery-code/epoch10-searchguard-v1/workflow.py"
WORKFLOW_SHA = "87252a0149f371e0b9fbed62ae41efde1a7f9f0ee2d0aec06c9ed66003ab8dc6"
OPERATION = R / "kaggle/bundles/epoch010-1464f64063d2-2bc2695-v1/operation.json"
OPERATION_SHA = "376c04a36792b6f2b4e9516a7e8ef7bdf93677076be6a734938eb194a2663e67"
SNAPSHOT = R / "runs/dinov3-l16-production-fold0-b74060b-v1/epoch-checkpoints/epoch-010.json"
CHECKPOINT_SHA = "1464f64063d2ac8729452f0f3edc112a21516ce51ad3b8718a176a599dc3f28e"
DATASET_SLUG = "godlydonuts/rsna-knee-e010-1464f64063d2"
KERNEL_SLUG = "godlydonuts/rsna-knee-epoch10-1464f64063d2"
DATASET_ID = 12146648


def load(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ValueError("Pinned source module missing")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("CPU-only Slurm allocation required")
    coordinator = load(SOURCE / "scripts/kaggle/coordinator.py", "pinned_coordinator")
    _, projection = coordinator.load_workflow(SOURCE, PROJECTION / "projection.json", PACKAGE_SHA)
    if projection["package_git_commit"] != "2bc26959c4b7d3c82a7cba41f8b8f662970a91ac":
        raise ValueError("Immutable inference source projection differs")
    if coordinator.p.digest(WORKFLOW_PATH) != WORKFLOW_SHA:
        raise ValueError("Reviewed search guard source differs")
    workflow = load(WORKFLOW_PATH, "reviewed_search_workflow")
    if workflow.digest(OPERATION) != OPERATION_SHA:
        raise ValueError("Immutable epoch-10 operation differs")
    op, meta, files = coordinator.validate_operation(
        workflow,
        SimpleNamespace(operation=OPERATION, snapshot=SNAPSHOT, projection=SOURCE),
    )
    if (
        meta["checkpoint_sha256"] != CHECKPOINT_SHA
        or op["dataset_slug"] != DATASET_SLUG
        or op["kernel_slug"] != KERNEL_SLUG
        or op["dataset_version"] != 1
        or op["epoch"] != 10
    ):
        raise ValueError("Epoch-10 identity differs")
    base = R / "kaggle"
    journal = base / "dataset-publications" / CHECKPOINT_SHA
    verified_path = journal / "verified.json"
    verified = json.loads(verified_path.read_text())
    if (
        verified.get("status") != "all_payload_bytes_verified"
        or verified.get("files") != files
        or verified.get("operation_sha256") != OPERATION_SHA
        or verified.get("intent_sha256") != workflow.digest(journal / "intent.json")
        or verified.get("id") != DATASET_ID
        or verified.get("version") != 1
        or verified.get("slug") != DATASET_SLUG
        or verified.get("private") is not True
    ):
        raise ValueError("Exact original private Dataset byte proof differs")
    remote, _ = coordinator.remote_from_stdin(workflow)
    with coordinator.single_coordinator(base):
        if coordinator.p.metadata(remote, DATASET_SLUG) != {
            k: verified[k] for k in ("id", "version", "slug", "private")
        }:
            raise ValueError("Private Dataset identity drifted")
        with workflow.ledger(base / "account-ledger.sqlite3", remote.username) as db:
            workflow.verify_account(db, remote)
            entry = workflow.read_entry(db, CHECKPOINT_SHA)
            if (
                entry.get("state") != "queued"
                or entry.get("kernel_version") is not None
                or entry.get("operation") is not None
                or entry.get("submitted") is not False
            ):
                raise ValueError("Epoch-10 ledger no longer has an untouched push state")
            absence = remote.require_absent_kernel(KERNEL_SLUG)
            if absence.get("method") != "complete_authenticated_profile_search_after403":
                raise ValueError("Expected exact-search absence proof was not reproduced")
            recovery = base / "kernel-push-recoveries" / CHECKPOINT_SHA
            recovery.mkdir(mode=0o700)
            coordinator.p.exclusive_json(
                recovery / "searchguard-intent.json",
                {
                    "schema_version": 1,
                    "status": "one_epoch10_searchguard_push_intent",
                    "checkpoint_sha256": CHECKPOINT_SHA,
                    "operation_sha256": OPERATION_SHA,
                    "dataset_verified_sha256": workflow.digest(verified_path),
                    "original_projection_sha256": PACKAGE_SHA,
                    "reviewed_workflow_sha256": WORKFLOW_SHA,
                    "recovery_script_sha256": workflow.digest(Path(__file__)),
                    "initial_ledger_entry_sha256": workflow.object_hash(entry),
                    "absence_proof": absence,
                    "dataset_id": DATASET_ID,
                    "no_new_dataset_or_checkpoint_bytes": True,
                },
            )
            try:
                result = workflow.push(remote, db, OPERATION)
                if result.get("state") != "pushed" or result.get("kernel_version") != 1:
                    raise ValueError("Notebook version is not exact private version 1")
                coordinator.p.exclusive_json(
                    recovery / "searchguard-result.json",
                    {
                        "status": "pushed",
                        "kernel_slug": KERNEL_SLUG,
                        "version": 1,
                        "notebook_source_sha256": op["kernel_source_sha256"],
                        "ledger_entry_sha256": workflow.object_hash(result),
                    },
                )
                print(json.dumps({"status": "pushed", "kernel_slug": KERNEL_SLUG, "version": 1}))
            except Exception as exc:
                coordinator.p.exclusive_json(
                    recovery / "searchguard-uncertain.json",
                    {
                        "status": "push_uncertain",
                        "error_type": type(exc).__name__,
                        "automatic_retry": False,
                    },
                )
                raise


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__}))
        raise SystemExit(1) from None
