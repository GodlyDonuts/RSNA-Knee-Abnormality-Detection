"""Read five exact submission outcomes into the one canonical ledger."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

ROOT = Path("/home/sa305415/rsna-knee")
STAGE = ROOT / "kaggle/launches/checkpoint-comparison-inputguard-v1"
PROJECTION = ROOT / "kaggle/code-projections/f4f90bc-b740-inputguard-v1"
PROJECTION_SHA = "f42e065ea2e3b99a31336511911663ed90389eb0506822885e5dc9ad075f5236"


def main():
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("One-CPU zero-GPU Slurm allocation required")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(
        "pinned_coordinator", PROJECTION / "code/scripts/kaggle/coordinator.py"
    )
    coordinator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(coordinator)
    workflow, _ = coordinator.load_workflow(
        PROJECTION / "code", PROJECTION / "projection.json", PROJECTION_SHA
    )
    bindings_file = STAGE / "final-campaign-bindings.json"
    if len(sys.argv) != 2 or coordinator.p.digest(bindings_file) != sys.argv[1]:
        raise ValueError("Reviewed five-submission bindings differ")
    bindings = json.loads(bindings_file.read_text())
    if sorted(x["epoch"] for x in bindings) != [2, 4, 6, 8, 10]:
        raise ValueError("Five matched checkpoint epochs required")
    remote, _ = coordinator.remote_from_stdin(workflow)
    base = ROOT / "kaggle"
    outcomes = []
    with (
        coordinator.single_coordinator(base),
        workflow.ledger(base / "account-ledger.sqlite3", remote.username) as db,
    ):
        for bound in bindings:
            entry = workflow.read_entry(db, bound["checkpoint_sha256"])
            if (
                entry["state"] != "submitted"
                or entry["submission_id"] != bound["submission_id"]
                or entry["kernel_version"] != 1
                or entry["operation"]["kernel_slug"] != bound["kernel_slug"]
                or entry["operation"]["manifest_sha256"] != bound["manifest_sha256"]
            ):
                raise ValueError("Canonical accepted identity differs; never retry")
        for bound in bindings:
            entry = workflow.reconcile_submission(remote, db, bound["checkpoint_sha256"])
            outcomes.append(
                {
                    "epoch": bound["epoch"],
                    "checkpoint_sha256": bound["checkpoint_sha256"],
                    "submission_id": entry["submission_id"],
                    "kernel_slug": entry["operation"]["kernel_slug"],
                    "kernel_version": entry["kernel_version"],
                    "state": entry["state"],
                    "hidden_status": entry["remote_status"],
                    "hidden_error_description": entry["remote_error_description"],
                    "hidden_total_bytes": entry["remote_total_bytes"],
                    "hidden_outcome": entry["remote_outcome"],
                    "public_score": entry["public_score"],
                    "reconciled_at": entry["reconciled_at"],
                }
            )
        account = remote.account()
    print(
        json.dumps(
            {
                "checked_at": workflow.utcnow(),
                "status": "five_exact_accepted_identities_reconciled",
                "canonical_ledger": str(base / "account-ledger.sqlite3"),
                "read_only_kaggle_queries": True,
                "competition_mutation": False,
                "python_version": sys.version,
                "slurm_job_id": os.environ["SLURM_JOB_ID"],
                "account": account,
                "outcomes": outcomes,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
