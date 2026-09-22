"""Immutable completed-epoch artifacts for the authorized Kaggle cadence."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from rsna.competitive.checkpoint import atomic_json, file_sha256


def scheduled_epoch(epoch: int, next_update: int, config: dict) -> bool:
    interval = config.get("submission_every_epochs", 0)
    return (
        type(epoch) is int
        and 0 < epoch <= config["epochs"]
        and type(next_update) is int
        and next_update == 0
        and type(interval) is int
        and interval > 0
        and epoch % interval == 0
    )


def write_epoch_snapshot(
    checkpoint: Path, output: Path, binding: dict, epoch: int, step: int
) -> dict:
    if not scheduled_epoch(epoch, 0, binding["config"]):
        raise ValueError("Epoch is outside the explicitly configured submission cadence")
    output.mkdir(parents=True, exist_ok=True)
    path = output / f"epoch-{epoch:03d}.pt"
    sha = file_sha256(checkpoint)
    record = {
        "schema_version": 1,
        "status": "scheduled_epoch_snapshot",
        "epoch": epoch,
        "optimizer_steps": step,
        "checkpoint_file": path.name,
        "checkpoint_sha256": sha,
        "checkpoint_bytes": checkpoint.stat().st_size,
        "binding_sha256": hashlib.sha256(
            json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest(),
        "release_binding_sha256": binding["release"]["binding_sha256"],
        "fold": binding["config"]["fold"],
        "seed": binding["config"]["seed"],
        "submission_every_epochs": binding["config"]["submission_every_epochs"],
        "diagnostic_only": binding["config"].get("diagnostic_study_limit") is not None,
        "submission_eligible": binding["config"].get("diagnostic_study_limit") is None,
        "submitted": False,
    }
    if path.exists():
        if path.is_symlink() or file_sha256(path) != sha:
            raise ValueError("Refusing to replace an existing epoch snapshot")
    else:
        # The next recovery checkpoint is atomically replaced; this inode remains
        # immutable without writing a second complete weight file now.
        os.link(checkpoint, path)
    metadata = path.with_suffix(".json")
    if metadata.exists():
        if metadata.is_symlink() or json.loads(metadata.read_text()) != record:
            raise ValueError("Refusing to replace existing epoch snapshot metadata")
    else:
        atomic_json(metadata, record)
    return record
