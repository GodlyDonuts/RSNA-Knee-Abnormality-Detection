"""Validate and preserve the bounded full-data pilot before exact resume."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from rsna.competitive.checkpoint import file_sha256, load_checkpoint

ROOT = Path("/home/sa305415/rsna-knee")
SOURCE = ROOT / "code-stopfix-1f486765"
RUN = ROOT / "runs/dinov3-l16-production-fold0-b74060b-v1"
PRESERVED = RUN / "pilot-checkpoints"
COMMIT = "b74060b98f01bff1b924070ccba5ce44dd002f40"


def canonical_hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def main() -> None:
    if PRESERVED.exists() or PRESERVED.is_symlink():
        raise ValueError("Pilot checkpoint preservation already attempted")
    assert (
        subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"]).decode().strip()
        == COMMIT
    )
    config = ROOT / "encoder-stage/run-configs/production-fold0.json"
    assert file_sha256(config) == "9c69708be0379d34c0ec4e998023dbf1c149f13d66732b8be22055293822f8fd"
    run_file, invocation_file, checkpoint_file = (
        RUN / "run.json",
        RUN / "invocation.json",
        RUN / "checkpoint.pt",
    )
    for path in (run_file, invocation_file, checkpoint_file):
        assert path.is_file() and not path.is_symlink()
    run = json.loads(run_file.read_text())
    invocation = json.loads(invocation_file.read_text())
    binding = run["binding"]
    assert run["status"] == "running" and not run["diagnostic_run"]
    assert run["training_studies"] == 2247 and run["outer_studies"] == 1156
    assert binding["world_size"] == 2 and binding["device_type"] == "cuda"
    assert binding["config"]["epochs"] == 10
    assert binding["config"]["fold"] == 0
    assert binding["config"]["diagnostic_study_limit"] is None
    assert (
        binding["code_sha256"]["competitive/data.py"]
        == "04f9b0e3c6e70c6a462775f130d8ec58c8d476233a690762bb2ea26097085aec"
    )
    assert invocation["status"] == "checkpointed_for_resume" and invocation["exit_code"] == 75
    assert invocation["binding_sha256"] == canonical_hash(binding)
    assert len(invocation["updates"]) == 26
    assert [row["update"] for row in invocation["updates"]] == list(range(1, 27))
    assert all(
        row["epoch"] == 0 and row["optimizer_steps"] == row["update"]
        for row in invocation["updates"]
    )
    checkpoint_sha = file_sha256(checkpoint_file)
    assert invocation["checkpoint_sha256"] == checkpoint_sha
    state = load_checkpoint(checkpoint_file, binding)
    assert (state["epoch"], state["next_update"], state["step"]) == (0, 26, 26)
    assert not state["scheduled_epoch"] and len(state["rng_by_rank"]) == 2
    assert len(state["model"]) > 0 and len(state["optimizer"]["state"]) > 0
    assert not (RUN / "epoch-checkpoints").exists()

    PRESERVED.mkdir(mode=0o700)
    copies = {
        "update-026.pt": checkpoint_file,
        "run-update-026.json": run_file,
        "invocation-update-026.json": invocation_file,
    }
    for name, source in copies.items():
        os.link(source, PRESERVED / name)
        assert (PRESERVED / name).stat().st_ino == source.stat().st_ino
    proof = {
        "status": "pilot_update26_checkpoint_preserved",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "pilot_job": 840991,
        "source_git": COMMIT,
        "config_sha256": file_sha256(config),
        "binding_sha256": invocation["binding_sha256"],
        "checkpoint_sha256": checkpoint_sha,
        "checkpoint_bytes": checkpoint_file.stat().st_size,
        "run_sha256": file_sha256(run_file),
        "invocation_sha256": file_sha256(invocation_file),
        "epoch": state["epoch"],
        "next_update": state["next_update"],
        "optimizer_steps": state["step"],
        "world_size": binding["world_size"],
        "diagnostic_run": False,
    }
    with (PRESERVED / "update-026-proof.json").open("x") as writer:
        json.dump(proof, writer, sort_keys=True, indent=2)
        writer.write("\n")
        writer.flush()
        os.fsync(writer.fileno())
    print(json.dumps(proof, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
