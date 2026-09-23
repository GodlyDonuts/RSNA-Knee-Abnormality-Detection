"""Preserve unscheduled completed-epoch recovery files during the active run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import torch

ODD_EPOCHS = (3, 5, 7, 9)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def capture(run: Path, epoch: int, job_id: str) -> dict:
    if epoch not in ODD_EPOCHS:
        raise ValueError("Only unscheduled odd training epochs need capture")
    source = run / "checkpoint.pt"
    output = run / "epoch-checkpoints"
    target = output / f"epoch-{epoch:03d}.pt"
    sidecar = target.with_suffix(".json")
    if target.exists() or sidecar.exists():
        if not target.is_file() or not sidecar.is_file():
            raise ValueError("Incomplete prior epoch capture")
        record = json.loads(sidecar.read_text())
        if record.get("checkpoint_sha256") != sha(target) or record.get("epoch") != epoch:
            raise ValueError("Prior odd-epoch capture changed")
        return record
    if source.is_symlink() or not source.is_file():
        raise ValueError("Recovery checkpoint is unavailable")
    os.link(source, target)
    state = torch.load(target, map_location="cpu", weights_only=True)
    run_record = json.loads((run / "run.json").read_text())
    binding = state.get("binding")
    if (
        state.get("schema_version") != 1
        or state.get("optimizer_boundary") is not True
        or type(state.get("epoch")) is not int
        or type(state.get("next_update")) is not int
        or type(state.get("step")) is not int
        or (state["epoch"], state["next_update"], state["step"]) != (epoch, 0, epoch * 281)
        or state.get("scheduled_epoch") is not False
        or binding != run_record["binding"]
        or binding["config"].get("submission_every_epochs") != 2
        or binding["config"].get("diagnostic_study_limit") is not None
        or run_record.get("training_studies") != 2247
    ):
        target.unlink()
        raise ValueError("Odd-epoch recovery file is not the completed full-data epoch")
    record = {
        "schema_version": 1,
        "status": "verified_unscheduled_epoch_archive",
        "epoch": epoch,
        "optimizer_steps": state["step"],
        "checkpoint_file": target.name,
        "checkpoint_sha256": sha(target),
        "checkpoint_bytes": target.stat().st_size,
        "binding_sha256": hashlib.sha256(
            json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "release_binding_sha256": binding["release"]["binding_sha256"],
        "fold": binding["config"]["fold"],
        "seed": binding["config"]["seed"],
        "submission_eligible": False,
        "capture_job_id": job_id,
        "source_run": str(run),
    }
    with sidecar.open("x") as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record


def run(log: Path, output: Path, primary_job: str, interval: float) -> None:
    if interval <= 0 or not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("A CPU-only Slurm watcher with positive polling interval is required")
    seen: set[int] = set()
    while seen != set(ODD_EPOCHS):
        for line in log.read_text(errors="replace").splitlines():
            if not line.startswith('{"epoch": '):
                continue
            event = json.loads(line)
            completed = event.get("epoch", -1) + 1
            if completed in ODD_EPOCHS and event.get("update") == 281 and completed not in seen:
                record = capture(output, completed, os.environ["SLURM_JOB_ID"])
                seen.add(completed)
                print(json.dumps(record, sort_keys=True), flush=True)
        if seen == set(ODD_EPOCHS):
            break
        p = subprocess.run(
            ["squeue", "-h", "-j", primary_job, "-o", "%T"],
            capture_output=True,
            text=True,
            check=True,
        )
        if p.stdout.strip() != "RUNNING":
            raise ValueError("Primary training ended before all odd epochs were archived")
        time.sleep(interval)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--primary-job", required=True)
    parser.add_argument("--interval", type=float, default=2.0)
    args = parser.parse_args()
    run(args.log, args.output, args.primary_job, args.interval)


if __name__ == "__main__":
    main()
