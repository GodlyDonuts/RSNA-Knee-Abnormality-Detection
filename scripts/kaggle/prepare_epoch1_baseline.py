"""Freeze the owner's one-off completed epoch-1 replay without editing weights."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import torch
from baseline_infer import digest, validate_baseline


def object_hash(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def run(checkpoint: Path, proof_file: Path, repo: Path, output: Path) -> dict:
    if output.exists() or output.is_symlink():
        raise ValueError("Epoch-1 baseline snapshot already exists")
    proof = json.loads(proof_file.read_text())
    if (
        proof.get("status") != "completed_epoch1_replay"
        or proof.get("checkpoint_sha256") != digest(checkpoint)
        or proof.get("checkpoint_bytes") != checkpoint.stat().st_size
        or proof.get("optimizer_steps") != 281
        or proof.get("source_checkpoint_sha256")
        != "b50ae5591b7b5b02f36423676f1f7fcbedacd69c8d848172bee459e4ae8d044a"
    ):
        raise ValueError("Completed replay proof does not bind checkpoint")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    binding = validate_baseline(state)
    code = {
        str(p.relative_to(repo / "rsna")): digest(p) for p in sorted((repo / "rsna").rglob("*.py"))
    }
    if binding.get("code_sha256") != code:
        raise ValueError("Replay checkpoint does not bind exact trained source")
    output.mkdir(mode=0o700, parents=True)
    target = output / "epoch-001.pt"
    os.link(checkpoint, target)
    copied = output / "baseline-proof.json"
    shutil.copyfile(proof_file, copied)
    record = {
        "schema_version": 1,
        "status": "explicit_epoch1_baseline_snapshot",
        "epoch": 1,
        "optimizer_steps": 281,
        "checkpoint_file": target.name,
        "checkpoint_sha256": proof["checkpoint_sha256"],
        "checkpoint_bytes": proof["checkpoint_bytes"],
        "binding_sha256": object_hash(binding),
        "release_binding_sha256": binding["release"]["binding_sha256"],
        "fold": binding["config"]["fold"],
        "seed": binding["config"]["seed"],
        "submission_every_epochs": binding["config"]["submission_every_epochs"],
        "diagnostic_only": False,
        "submission_eligible": True,
        "submitted": False,
        "baseline_proof_file": copied.name,
        "baseline_proof_sha256": digest(copied),
        "authorization": "owner_explicit_epoch1_baseline_request_2026-09-22",
    }
    sidecar = target.with_suffix(".json")
    with sidecar.open("x") as stream:
        json.dump(record, stream, sort_keys=True, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "proof-file", "repo", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(run(args.checkpoint, args.proof_file, args.repo, args.output), sort_keys=True),
        flush=True,
    )


if __name__ == "__main__":
    main()
