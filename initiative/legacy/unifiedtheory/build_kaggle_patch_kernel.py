#!/usr/bin/env python3
"""Build, but do not publish, the fixed patch-only Kaggle Code kernel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


CELL = r'''
from pathlib import Path
import json
import subprocess
import sys

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")

code_hits = list(INPUT.rglob("kaggle_offline_infer.py"))
if len(code_hits) != 1:
    raise RuntimeError(f"expected exactly one inference entry point, found {code_hits}")
code_dir = code_hits[0].parent

checkpoints = sorted(INPUT.rglob("patch_mean_fold*.pt"))
if len(checkpoints) != 15:
    raise RuntimeError(f"expected exactly 15 patch checkpoints, found {len(checkpoints)}")

model_candidates = []
for config in INPUT.rglob("config.json"):
    try:
        payload = json.loads(config.read_text())
    except Exception:
        continue
    if payload.get("model_type") == "dinov2" and int(payload.get("hidden_size", 0)) == 768:
        model_candidates.append(config.parent)
if len(model_candidates) != 1:
    raise RuntimeError(f"expected one local DINOv2-base model, found {model_candidates}")

command = [
    sys.executable, str(code_dir / "kaggle_offline_infer.py"),
    "--data-root", str(INPUT / "rsna-knee-abnormality-detection"),
    "--dino-model", str(model_candidates[0]),
    "--work-dir", str(WORKING / "patch_only_cache"),
    "--output", str(WORKING / "submission.csv"),
    "--runtime-json", str(WORKING / "runtime.json"),
    "--batch-size", "64", "--inference-batch-size", "3",
    "--workers", "4", "--max-slices", "64", "--time-budget-hours", "8.5",
]
for checkpoint in checkpoints:
    command.extend(["--checkpoint-glob", str(checkpoint)])
print("+", " ".join(command), flush=True)
subprocess.run(command, check=True)
if not (WORKING / "submission.csv").is_file():
    raise RuntimeError("inference completed without submission.csv")
'''.strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", default="tomdifiore")
    parser.add_argument("--slug", default="rsna-knee-fixed-patch-rank")
    parser.add_argument("--code-dataset", default="tomdifiore/rsna-knee-unifiedtheory-code")
    parser.add_argument("--checkpoint-dataset", default="tomdifiore/rsna-knee-fixed-patch-checkpoints")
    parser.add_argument("--dino-model-source", default="metaresearch/dinov2/pyTorch/base/1")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    notebook_path = args.output / "patch_only_inference.ipynb"
    notebook = {
        "cells": [
            {
                "cell_type": "markdown", "metadata": {},
                "source": ["# RSNA knee fixed patch-only rank ensemble\n", "Offline image-only inference from 15 fixed fold/seed checkpoints.\n"],
            },
            {
                "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
                "source": [line + "\n" for line in CELL.splitlines()],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4, "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    metadata = {
        "id": f"{args.owner}/{args.slug}",
        "title": "RSNA Knee Fixed Patch Rank Ensemble",
        "code_file": notebook_path.name,
        "language": "python", "kernel_type": "notebook",
        "is_private": False, "enable_gpu": True, "enable_internet": False,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [args.code_dataset, args.checkpoint_dataset],
        "model_sources": [args.dino_model_source],
    }
    (args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"notebook": str(notebook_path), "metadata": metadata}, indent=2))


if __name__ == "__main__":
    main()
