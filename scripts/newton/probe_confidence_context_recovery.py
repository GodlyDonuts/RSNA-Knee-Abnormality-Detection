"""Seven fresh CPU processes: both objective recoveries and frozen-engine parity.

Engineering only; 32 already-admitted fitting MRI feature records, no clinical
evaluation, acquisition, Kaggle mutation or encoder optimization.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from rsna.competitive.confidence_context_train import VERSION
from rsna.competitive.released_context_train import digest


def same(a, b):
    if isinstance(a, torch.Tensor):
        if not torch.equal(a, b):
            raise ValueError("Decoded tensor differs")
    elif isinstance(a, np.ndarray):
        if not np.array_equal(a, b):
            raise ValueError("Decoded RNG array differs")
    elif isinstance(a, dict):
        if a.keys() != b.keys():
            raise ValueError("Complete state keys differ")
        for key in a:
            same(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        if type(a) is not type(b) or len(a) != len(b):
            raise ValueError("Decoded state sequence differs")
        for x, y in zip(a, b, strict=True):
            same(x, y)
    elif a != b:
        raise ValueError("Decoded numerical state differs")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--uniform-inputs", type=Path, required=True)
    parser.add_argument("--confidence-inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("Fresh CPU-only recovery output required")
    args.output.mkdir(mode=0o700, parents=True)
    source = Path(__file__).resolve().parents[2]
    script = source / "scripts/newton/run_confidence_context_fit.py"
    results = {}
    for role, manifest in (
        ("uniform", args.uniform_inputs),
        ("source_confidence", args.confidence_inputs),
    ):
        original = json.loads(manifest.read_text())
        if original.get("take") != 32 or len(original["labels"]) != 32:
            raise ValueError("Exactly32 admitted engineering feature records required")
        recipe = args.output / (role + "-recipe.json")
        recipe.write_text(json.dumps({"epochs": 2, "objective": role}, sort_keys=True))
        base = [
            sys.executable,
            "-B",
            str(script),
            "--inputs",
            str(manifest),
            "--inputs-sha256",
            digest(manifest),
            "--recipe",
            str(recipe),
        ]
        for name, extra in (
            ("full", []),
            ("stop", ["--stop-after", "1"]),
            (
                "resume",
                ["--resume", str(args.output / (role + "-stop") / "checkpoint.pt")],
            ),
        ):
            with (args.output / (role + "-" + name + ".private.log")).open("x") as log:
                subprocess.run(
                    base + ["--output", str(args.output / (role + "-" + name))] + extra,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
        full, resumed = (args.output / (role + "-" + x) for x in ("full", "resume"))
        state = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=False)
        same(
            state,
            torch.load(resumed / "checkpoint.pt", map_location="cpu", weights_only=False),
        )
        if (
            state["version"] != VERSION
            or state["completed"] != 4
            or state["epoch"] != 2
            or not state["model"]["output"].count_nonzero()
            or (full / "predictions.private.npy").read_bytes()
            != (resumed / "predictions.private.npy").read_bytes()
        ):
            raise ValueError("Actual recovery state or all32 prediction bytes differ")
        results[role] = {
            "fresh_processes": 3,
            "updates": 4,
            "complete_model_AdamW_LR_order_counters_allRNG_history_equal": True,
            "all32x12_prediction_bytes_equal": True,
            "checkpoint_sha256": digest(full / "checkpoint.pt"),
        }

    # Same features/labels/masks/seed and complete old numerical engine. Only
    # version/recipe objective metadata differs; all numerical state must match.
    old_recipe = args.output / "legacy-recipe.json"
    old_recipe.write_text(json.dumps({"epochs": 2}))
    legacy = args.output / "legacy-full"
    with (args.output / "legacy.private.log").open("x") as log:
        subprocess.run(
            [
                sys.executable,
                "-B",
                str(source / "scripts/newton/run_released_context_fit.py"),
                "--inputs",
                str(args.uniform_inputs),
                "--inputs-sha256",
                digest(args.uniform_inputs),
                "--recipe",
                str(old_recipe),
                "--output",
                str(legacy),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    old = torch.load(legacy / "checkpoint.pt", map_location="cpu", weights_only=False)
    new_path = args.output / "uniform-full"
    new = torch.load(new_path / "checkpoint.pt", map_location="cpu", weights_only=False)
    new["version"] = old["version"]
    del new["original_trainer_sha256"]
    del new["recipe"]["objective"]
    same(old, new)
    if (legacy / "predictions.private.npy").read_bytes() != (
        new_path / "predictions.private.npy"
    ).read_bytes():
        raise ValueError("Original engine prediction bytes changed")
    receipt = {
        "status": "actual32_source_confidence_engine_recovery_and_uniform_parity_accepted",
        "CPU_job": os.environ.get("SLURM_JOB_ID"),
        "roles": results,
        "fresh_processes": 7,
        "uniform_complete_numerical_state_and_predictions_equal_frozen_original": True,
        "clinical_or_competition_accuracy_evaluated": False,
        "protected_values_read": False,
    }
    (args.output / "accepted.json").write_text(json.dumps(receipt, sort_keys=True, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
