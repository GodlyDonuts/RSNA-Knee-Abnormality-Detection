"""Preserve fold-1 completed odd epochs before the rolling checkpoint advances."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch

ODD_EPOCHS = (1, 3, 5, 7, 9)


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def capture(run: Path, epoch: int, job_id: str, event: dict) -> dict | None:
    source = run / "checkpoint.pt"
    output = run / "epoch-checkpoints"
    output.mkdir(mode=0o700, exist_ok=True)
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
        return None
    os.link(source, target)
    try:
        state = torch.load(target, map_location="cpu", weights_only=True)
        run_record = json.loads((run / "run.json").read_text())
        binding = state.get("binding")
        config = run_record["binding"]["config"]
        updates = math.ceil(
            run_record["training_studies"]
            / (
                config["batch_size"]
                * config["accumulation_steps"]
                * run_record["binding"]["world_size"]
            )
        )
        if (
            state.get("schema_version") != 1
            or state.get("optimizer_boundary") is not True
            or (state.get("epoch"), state.get("next_update"), state.get("step"))
            != (epoch, 0, event["optimizer_steps"])
            or state.get("scheduled_epoch") is not False
            or binding != run_record["binding"]
            or config.get("fold") != 2
            or config.get("submission_every_epochs") != 2
            or config.get("diagnostic_study_limit") is not None
            or event["update"] != updates
            or event["epoch"] + 1 != epoch
        ):
            return None
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
            "fold": 2,
            "seed": config["seed"],
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
    finally:
        if not sidecar.exists():
            target.unlink()


def run(log: Path, output: Path, primary_job: str, interval: float) -> None:
    if interval <= 0 or not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("A CPU-only Slurm watcher with positive polling interval is required")
    seen: set[int] = set()
    while seen != set(ODD_EPOCHS):
        if log.is_file() and (output / "run.json").is_file():
            run_record = json.loads((output / "run.json").read_text())
            binding = run_record["binding"]
            config = binding["config"]
            updates = math.ceil(
                run_record["training_studies"]
                / (config["batch_size"] * config["accumulation_steps"] * binding["world_size"])
            )
            for line in log.read_text(errors="replace").splitlines():
                if not line.startswith('{"epoch": '):
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                epoch = event.get("epoch", -1) + 1
                if epoch in ODD_EPOCHS and epoch not in seen and event.get("update") == updates:
                    record = capture(output, epoch, os.environ["SLURM_JOB_ID"], event)
                    if record is not None:
                        seen.add(epoch)
                        print(json.dumps(record, sort_keys=True), flush=True)
        if seen == set(ODD_EPOCHS):
            break
        state = subprocess.run(
            ["squeue", "-h", "-j", primary_job, "-o", "%T"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if state not in {"RUNNING", "PENDING"}:
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
