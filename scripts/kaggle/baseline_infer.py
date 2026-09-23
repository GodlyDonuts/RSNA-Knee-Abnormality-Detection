"""Explicit one-off epoch-1 inference using the byte-exact trained inference code.

The trainer's two-epoch submission cadence remains unchanged. This adapter only
authorizes the owner's requested completed epoch-1 replay checkpoint; all model,
preprocessing, code-hash and output operations run through the trained infer.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from rsna.competitive import infer
from rsna.release import validate_checkpoint_binding
from rsna.schema import CACHE_VERSION, TARGETS


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def validate_baseline(checkpoint: dict) -> dict:
    if checkpoint.get("schema_version") != 1 or checkpoint.get("optimizer_boundary") is not True:
        raise ValueError("Baseline requires an optimizer-boundary checkpoint")
    binding = checkpoint["binding"]
    validate_checkpoint_binding(binding["release"])
    config = binding["config"]
    if (
        type(checkpoint.get("epoch")) is not int
        or type(checkpoint.get("next_update")) is not int
        or type(checkpoint.get("step")) is not int
        or type(config.get("fold")) is not int
        or type(config.get("seed")) is not int
        or type(config.get("epochs")) is not int
        or type(config.get("submission_every_epochs")) is not int
        or checkpoint.get("epoch") != 1
        or checkpoint.get("next_update") != 0
        or checkpoint.get("step") != 281
        or checkpoint.get("scheduled_epoch") is not False
        or config.get("epochs") != 10
        or config.get("submission_every_epochs") != 2
        or config.get("diagnostic_study_limit") is not None
        or config.get("fold") != 0
        or config.get("seed") != 20260922
        or config.get("encoder", {}).get("architecture") != "dinov3_l16"
        or binding.get("world_size") != 2
        or binding.get("targets") != list(TARGETS)
        or binding.get("cache_version") != CACHE_VERSION
        or binding.get("source_revision") != binding["release"]["source_revision"]
    ):
        raise ValueError("Only the exact completed fold-0 epoch-1 baseline is authorized")
    return binding


def main() -> None:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-sidecar", type=Path, required=True)
    parser.add_argument("--baseline-proof", type=Path, required=True)
    args, remaining = parser.parse_known_args()
    sidecar = json.loads(args.baseline_sidecar.read_text())
    proof = json.loads(args.baseline_proof.read_text())
    if (
        sidecar.get("status") != "explicit_epoch1_baseline_snapshot"
        or sidecar.get("epoch") != 1
        or sidecar.get("checkpoint_sha256") != digest(args.checkpoint)
        or sidecar.get("baseline_proof_sha256") != digest(args.baseline_proof)
        or proof.get("status") != "completed_epoch1_replay"
        or proof.get("checkpoint_sha256") != sidecar["checkpoint_sha256"]
        or proof.get("optimizer_steps") != sidecar.get("optimizer_steps")
    ):
        raise ValueError("Epoch-1 baseline proof or checkpoint differs")
    original = infer.validate_inference_checkpoint

    def explicitly_authorized(checkpoint: dict, **kwargs: object) -> dict:
        if kwargs.get("allow_diagnostic") or kwargs.get("allow_scheduled_epoch"):
            raise ValueError("Baseline adapter cannot broaden other inference modes")
        binding = validate_baseline(checkpoint)
        if (
            infer.file_sha256(args.checkpoint) != sidecar["checkpoint_sha256"]
            or checkpoint["step"] != sidecar["optimizer_steps"]
        ):
            raise ValueError("Baseline loaded weights differ from proof")
        return binding

    try:
        infer.validate_inference_checkpoint = explicitly_authorized
        sys.argv = [sys.argv[0], "--checkpoint", str(args.checkpoint), *remaining]
        infer.main()
    finally:
        infer.validate_inference_checkpoint = original


if __name__ == "__main__":
    main()
