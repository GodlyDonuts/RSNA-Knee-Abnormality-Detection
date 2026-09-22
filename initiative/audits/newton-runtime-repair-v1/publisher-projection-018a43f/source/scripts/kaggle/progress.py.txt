"""Read-only operational projection of the canonical ledger and bound train histories."""

import argparse
import csv
import hashlib
import json
import math
import sqlite3
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def finite(value):
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        raise ValueError("Boolean is not a numeric observation")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Nonfinite observation")
    return number


def project(records, histories):
    rows = []
    for entry in records:
        epoch = entry["epoch"]
        step = entry["optimizer_steps"]
        binding = entry["binding_sha256"]
        matches = []
        history_hashes = []
        for history, sha in histories:
            if history.get("binding_sha256") != binding:
                continue
            for update in history.get("updates", []):
                # Trainer epochs are zero-based; snapshots are completed-epoch counts.
                if update.get("epoch") == epoch - 1 and update.get("optimizer_steps") == step:
                    matches.append(finite(update["loss"]))
                    history_hashes.append(sha)
        if len(set(matches)) > 1:
            raise ValueError("Conflicting training loss at the exact completed boundary")
        complete = entry.get("remote_status") == "COMPLETE"
        rows.append(
            {
                "epoch": epoch,
                "fold": entry["fold"],
                "seed": entry["seed"],
                "run_binding_sha256": binding,
                "checkpoint_sha256": entry["checkpoint_sha256"],
                "kernel_slug": entry.get("operation", {}).get("kernel_slug"),
                "kernel_version": entry.get("kernel_version"),
                "submission_id": entry.get("submission_id"),
                "queue_state": entry["state"],
                "submission_status": entry.get("remote_status"),
                "public_score": finite(entry.get("public_score")) if complete else None,
                "score_checked_at": entry.get("reconciled_at"),
                "optimizer_steps": step,
                "training_loss_last_update": matches[-1] if matches else None,
                "training_loss_history_sha256": sorted(set(history_hashes)),
            }
        )
    return sorted(rows, key=lambda r: (r["run_binding_sha256"], r["epoch"], r["checkpoint_sha256"]))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ledger", required=True, type=Path)
    p.add_argument("--history", action="append", type=Path, default=[])
    p.add_argument("--output", required=True, type=Path)
    args = p.parse_args()
    if args.output.exists():
        raise ValueError("Use a new projection directory; preserved snapshots are immutable")
    # mode=ro prohibits ledger mutation. Snapshot the logical rows consistently,
    # rather than hashing the database file while the coordinator can update it.
    with sqlite3.connect(args.ledger.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("BEGIN")
        meta = dict(db.execute("SELECT key,value FROM meta"))
        records = [
            json.loads(row[0]) for row in db.execute("SELECT record FROM entries ORDER BY sha")
        ]
        db.rollback()
    histories = []
    for path in args.history:
        value = json.loads(path.read_text())
        if not isinstance(value.get("binding_sha256"), str) or not isinstance(
            value.get("updates"), list
        ):
            raise ValueError("Only binding-keyed training update histories accepted")
        histories.append((value, digest(path)))
    rows = project(records, histories)
    logical = json.dumps(
        {"meta": meta, "records": records}, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    result = {
        "schema_version": 1,
        "purpose": "observational_fixed_epoch_trajectory",
        "selection_performed": False,
        "labels_or_confirmation_accessed": False,
        "ledger_logical_snapshot_sha256": hashlib.sha256(logical).hexdigest(),
        "projector_sha256": digest(Path(__file__)),
        "history_sha256": sorted(sha for _, sha in histories),
        "loss_semantics": "Last training update at exact snapshot optimizer step, not epoch mean; null if unavailable.",
        "score_semantics": "Only COMPLETE submission public scores; pending/unknown null. No ranking or checkpoint selection.",
        "rows": rows,
    }
    args.output.mkdir(parents=True)
    (args.output / "progress.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n"
    )
    if rows:
        with (args.output / "progress.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow(
                    {
                        **row,
                        "training_loss_history_sha256": ";".join(
                            row["training_loss_history_sha256"]
                        ),
                    }
                )
    print(json.dumps({"status": "projected", "rows": len(rows), "ledger_mutated": False}))


if __name__ == "__main__":
    main()
