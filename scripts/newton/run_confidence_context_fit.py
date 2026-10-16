"""Fit one explicit source-confidence arm; private CPU-only immutable outputs."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from rsna.competitive.confidence_context_train import VERSION, Recipe, Trainer
from rsna.competitive.released_context_train import digest
from scripts.newton.run_released_context_fit import load_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("Confidence-head fitting uses one CPU and no GPU")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.output.exists():
        raise ValueError("Fresh immutable private output required")
    data, inputs = load_data(args.inputs, args.inputs_sha256)
    recipe = Recipe(**json.loads(args.recipe.read_text()))
    if (
        recipe.embedding_dim != 1024
        or recipe.targets != 12
        or inputs["binding"].get("confidence_engine") != VERSION
        or inputs["binding"].get("objective") != recipe.objective
    ):
        raise ValueError("Exact parent graph and objective admission required")
    args.output.mkdir(mode=0o700, parents=True)
    started = time.monotonic()
    trainer = Trainer(data, recipe)
    if args.resume is not None:
        trainer.load(args.resume)

    def progress(record):
        if record["update"] % 100 == 0 or trainer.cursor == 0:
            print(json.dumps({"objective": recipe.objective, "optimizer": record}), flush=True)

    trainer.run(
        stop_after=args.stop_after, epoch_directory=args.output / "epochs", progress=progress
    )
    checkpoint = args.output / "checkpoint.pt"
    trainer.save(checkpoint)
    prediction = args.output / "predictions.private.npy"
    with prediction.open("xb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        np.save(handle, trainer.predictions(), allow_pickle=False)
    result = {
        "status": "fixed_final_complete"
        if trainer.epoch == recipe.epochs
        else "stopped_at_update_boundary",
        "engine": VERSION,
        "objective": recipe.objective,
        "recipe": vars(recipe),
        "studies": len(data),
        "completed_updates": trainer.completed,
        "epochs": trainer.epoch,
        "trainable_parameters": sum(p.numel() for p in trainer.model.parameters()),
        "checkpoint_sha256": digest(checkpoint),
        "prediction_sha256": digest(prediction),
        "inputs_sha256": args.inputs_sha256,
        "binding": inputs["binding"],
        "elapsed_seconds": time.monotonic() - started,
        "clinical_accuracy_evaluated": False,
        "independent_internal_validation": False,
        "CNN_parameters_fitted": 0,
        "competition_submissions": 0,
    }
    (args.output / "terminal.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
