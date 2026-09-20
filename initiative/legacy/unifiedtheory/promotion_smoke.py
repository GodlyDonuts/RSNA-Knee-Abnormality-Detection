#!/usr/bin/env python3
"""Smoke-test the fail-closed fixed-family promotion gate."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace

import torch

try:
    from .constants import TARGETS
    from .run_promoted_family_inference import (
        AUDIT_CONTRACT,
        audited_checkpoints,
        validate_audit,
    )
except ImportError:
    from constants import TARGETS
    from run_promoted_family_inference import (
        AUDIT_CONTRACT,
        audited_checkpoints,
        validate_audit,
    )


def write_checkpoint(path: Path, model_type: str, seed: int, fold: int) -> None:
    config = {"aggregator": "mean", "num_targets": len(TARGETS)}
    if model_type == "patch":
        config["token_adapter_bottleneck"] = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_type": model_type,
            "seed": seed,
            "fold": fold,
            "model_config": config,
        },
        path,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        audit_path = root / "audit.json"
        audit_path.write_text(
            json.dumps(
                {
                    "contract": AUDIT_CONTRACT,
                    "paired_fold_target_cells": len(TARGETS),
                    "combined_gain": 0.011,
                    "minimum_preregistered_gain": 0.01,
                    "promote_combined": True,
                }
            )
        )
        audit = validate_audit(audit_path, folds=1)
        if not audit["promote_combined"]:
            raise AssertionError("valid audit did not promote")

        summary = root / "summary"
        patch = root / "patch"
        write_checkpoint(summary / "seed7" / "mean_fold0.pt", "summary", 7, 0)
        write_checkpoint(patch / "seed7" / "patch_mean_fold0.pt", "patch", 7, 0)
        args = SimpleNamespace(
            seeds=[7], folds=1, summary_runs=summary, patch_runs=patch
        )
        paths = audited_checkpoints(args)
        if len(paths) != 2:
            raise AssertionError("complete one-fold family did not resolve to two checkpoints")

        bad = torch.load(paths[1], map_location="cpu", weights_only=True)
        bad["model_config"]["token_adapter_bottleneck"] = 64
        torch.save(bad, paths[1])
        try:
            audited_checkpoints(args)
        except ValueError as error:
            if "zero-adapter" not in str(error):
                raise
        else:
            raise AssertionError("wrong patch family passed checkpoint validation")

    print("promotion gate smoke test passed")


if __name__ == "__main__":
    main()
