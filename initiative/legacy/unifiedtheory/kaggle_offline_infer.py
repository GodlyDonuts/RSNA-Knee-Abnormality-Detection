#!/usr/bin/env python3
"""Offline, time-budgeted Kaggle inference for the hidden RSNA test set.

The competition requires internet-disabled notebook inference in at most nine
hours.  This entry point loads a locally attached DINO model, extracts hidden
test features, ensembles locally attached checkpoints, validates the exact
submission schema, and records wall-clock provenance.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

try:
    from .constants import TARGETS
except ImportError:
    from constants import TARGETS


def checkpoint_paths(patterns: list[str]) -> list[Path]:
    paths = sorted({Path(path) for pattern in patterns for path in glob.glob(pattern)})
    if not paths:
        raise ValueError("checkpoint patterns resolved to no files")
    return paths


def needs_patch_grid(paths: list[Path]) -> bool:
    for path in paths:
        try:
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            checkpoint = torch.load(path, map_location="cpu")
        if checkpoint.get("model_type", "summary") == "patch":
            return True
    return False


def run(command: list[str], deadline: float) -> None:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Kaggle inference time budget exhausted")
    print("+ " + " ".join(command), flush=True)
    subprocess.run(command, check=True, timeout=remaining)


def validate_submission(sample_path: Path, submission_path: Path) -> pd.DataFrame:
    sample = pd.read_csv(sample_path)
    submission = pd.read_csv(submission_path)
    if list(submission.columns) != list(sample.columns):
        raise ValueError("submission columns do not exactly match sample_submission.csv")
    if len(submission) != len(sample):
        raise ValueError("submission row count does not match the hidden test set")
    if not submission.iloc[:, 0].astype(str).equals(sample.iloc[:, 0].astype(str)):
        raise ValueError("submission study identifiers or order do not match the sample")
    predictions = submission[TARGETS].to_numpy(dtype=float)
    if not np.isfinite(predictions).all() or not (
        (predictions >= 0) & (predictions <= 1)
    ).all():
        raise ValueError("submission predictions must be finite probabilities in [0, 1]")
    return submission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dino-model", type=Path, required=True)
    parser.add_argument("--checkpoint-glob", action="append", required=True)
    parser.add_argument("--work-dir", type=Path, default=Path("/kaggle/working/rsna_cache"))
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/submission.csv"))
    parser.add_argument("--runtime-json", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-slices", type=int, default=64)
    parser.add_argument("--time-budget-hours", type=float, default=8.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    deadline = started + args.time_budget_hours * 3600
    here = Path(__file__).resolve().parent
    paths = checkpoint_paths(args.checkpoint_glob)
    patch_grid = 4 if needs_patch_grid(paths) else 0
    if not args.dino_model.is_dir():
        raise FileNotFoundError(f"local DINO model directory not found: {args.dino_model}")

    # Enforce the same environment used by the notebook rule: no model hub or
    # telemetry fallback is permitted during hidden-test inference.
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )
    args.work_dir.mkdir(parents=True, exist_ok=True)
    test_cache = args.work_dir / "test"
    cache_index = test_cache / "test_cache_index.csv"
    run(
        [
            sys.executable,
            str(here / "extract_features.py"),
            "--data-root",
            str(args.data_root),
            "--split",
            "test",
            "--output",
            str(test_cache),
            "--model-name",
            str(args.dino_model),
            "--patch-grid",
            str(patch_grid),
            "--max-slices",
            str(args.max_slices),
            "--batch-size",
            str(args.batch_size),
            "--local-files-only",
        ],
        deadline,
    )
    run(
        [
            sys.executable,
            str(here / "infer.py"),
            "--cache-index",
            str(cache_index),
            "--sample-submission",
            str(args.data_root / "sample_submission.csv"),
            "--checkpoints",
            *[str(path) for path in paths],
            "--ensemble",
            "rank",
            "--batch-size",
            str(args.inference_batch_size),
            "--workers",
            str(args.workers),
            "--output",
            str(args.output),
        ],
        deadline,
    )

    submission = validate_submission(args.data_root / "sample_submission.csv", args.output)

    elapsed = time.monotonic() - started
    record = {
        "elapsed_seconds": elapsed,
        "budget_seconds": args.time_budget_hours * 3600,
        "studies": len(submission),
        "checkpoints": [str(path) for path in paths],
        "patch_grid": patch_grid,
        "internet_required": False,
        "reports_used": False,
    }
    runtime_path = args.runtime_json or args.output.with_suffix(".runtime.json")
    runtime_path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"validated {args.output}; elapsed={elapsed / 3600:.3f} hours", flush=True)


if __name__ == "__main__":
    main()
