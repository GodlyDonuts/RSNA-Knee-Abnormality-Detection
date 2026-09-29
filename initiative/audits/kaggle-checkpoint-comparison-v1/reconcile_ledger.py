"""Preserve an absent uncertain notebook; never repeat its Kaggle push."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/sa305415/rsna-knee")
PROJECTION = ROOT / "kaggle/code-projections/f4f90bc-b740-inputguard-v1"
PROJECTION_SHA = "f42e065ea2e3b99a31336511911663ed90389eb0506822885e5dc9ad075f5236"
LATEST_SHA = "5e400e82c4b50e94a27efcb250fc86d911b2191b6ed04792276fe51d9e093734"
OLD_OPERATION_SHA = "50a9cd3c2d390e0788c0c907945dab27e55ba3d71214c8b6c2af1b31cbef4deb"
NEW_KERNEL = "godlydonuts/rsna-knee-epoch10-f2-inputguard-5e400e82c4b5"
FAILED_SHA = "eb280864bd1c2f896210ce360e1ae36d59649c854a9051523c2454a34d08c7b1"


def archive_absent_push(workflow, remote, db):
    db.execute("BEGIN IMMEDIATE")
    workflow.verify_account(db, remote)
    entry = workflow.read_entry(db, LATEST_SHA)
    old = entry.get("operation", {})
    if (
        entry.get("state") != "push_uncertain"
        or entry.get("operation_sha256") != OLD_OPERATION_SHA
        or entry.get("kernel_version") is not None
        or entry.get("submission_id") is not None
        or entry.get("submitted") is not False
        or entry.get("attempted_at") is not None
        or entry.get("description") is not None
        or old.get("kernel_slug") != "godlydonuts/rsna-knee-f2e010-pixelfix-5e400e82c4b5"
        or old.get("checkpoint_sha256") != LATEST_SHA
        or old.get("kernel_slug") == NEW_KERNEL
    ):
        raise ValueError("The exact unsubmitted uncertain push changed; reconcile first")
    old_absence = remote.require_absent_kernel(old["kernel_slug"])
    new_absence = remote.require_absent_kernel(NEW_KERNEL)
    allowed = {
        "GetKernel404",
        "complete_authenticated_profile_after403",
        "complete_authenticated_profile_search_after403",
    }
    if old_absence.get("method") not in allowed or new_absence.get("method") not in allowed:
        raise ValueError("Exact authenticated absence proof required for both slugs")
    history = entry.get("superseded_operations", [])
    if not isinstance(history, list) or any(
        x.get("operation_sha256") == OLD_OPERATION_SHA for x in history
    ):
        raise ValueError("Uncertain attempt is already preserved or history is malformed")
    entry["superseded_operations"] = [
        *history,
        {
            "state": "absent_uncertain_push_preserved_never_retry",
            "operation": old,
            "operation_sha256": OLD_OPERATION_SHA,
            "dataset": entry["dataset"],
            "old_kernel_absence": old_absence,
            "new_kernel_absence": new_absence,
            "replacement_kernel_slug": NEW_KERNEL,
            "reconciled_at": workflow.utcnow(),
        },
    ]
    for name in (
        "operation",
        "operation_sha256",
        "dataset",
        "kernel_absence_proof",
        "kernel_version",
    ):
        entry.pop(name, None)
    entry["state"] = "queued"
    workflow.save_entry(
        db,
        entry,
        "Preserved absent uncertain push before distinct tested correction; old slug never retried",
    )
    return entry


def main():
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("CPU-only Slurm allocation required")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "pinned_coordinator", PROJECTION / "code/scripts/kaggle/coordinator.py"
    )
    coordinator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coordinator)
    workflow, _ = coordinator.load_workflow(
        PROJECTION / "code", PROJECTION / "projection.json", PROJECTION_SHA
    )
    remote, _ = coordinator.remote_from_stdin(workflow)
    base = ROOT / "kaggle"
    journal = base / "kernel-push-migrations" / "fold2-e010-inputguard-v1"
    with (
        coordinator.single_coordinator(base),
        workflow.ledger(base / "account-ledger.sqlite3", remote.username) as db,
    ):
        failed = workflow.reconcile_submission(remote, db, FAILED_SHA)
        if failed.get("submission_id") != 56550973 or failed.get("remote_outcome") != "failed":
            raise ValueError("Prior terminal result differs")
        initial = workflow.read_entry(db, LATEST_SHA)
        journal.mkdir(mode=0o700)
        coordinator.p.exclusive_json(
            journal / "intent.json",
            {
                "status": "read_only_remote_reconciliation_before_distinct_replacement",
                "initial_entry_sha256": workflow.object_hash(initial),
                "checkpoint_sha256": LATEST_SHA,
                "old_operation_sha256": OLD_OPERATION_SHA,
                "replacement_kernel_slug": NEW_KERNEL,
                "script_sha256": coordinator.p.digest(Path(__file__)),
                "retry_old_kernel": False,
                "competition_submission": False,
            },
        )
        try:
            entry = archive_absent_push(workflow, remote, db)
        except Exception:
            db.rollback()
            raise
        coordinator.p.exclusive_json(
            journal / "result.json",
            {
                "status": "old_uncertain_operation_preserved_new_candidate_queued",
                "entry_sha256": workflow.object_hash(entry),
                "failed_entry8_reconciled": failed["remote_outcome"],
            },
        )
        print(
            json.dumps(
                {
                    "status": "reconciled",
                    "epoch10_state": entry["state"],
                    "prior_epoch8_outcome": failed["remote_outcome"],
                    "kaggle_mutation": False,
                }
            )
        )


if __name__ == "__main__":
    main()
