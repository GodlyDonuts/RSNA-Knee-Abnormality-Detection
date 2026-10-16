"""Run the exact CPU FP32 licensed context engine, including engineering recovery.

Private input manifests bind only admitted fitting features/source probabilities.
No CNN, protected evaluation, disease AUC or competition actions run here.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from rsna.competitive.released_context_train import (
    FrozenFeatures,
    Recipe,
    Trainer,
    digest,
)


def load_data(path, sha):
    if digest(path) != sha:
        raise ValueError("Exact independently admitted private input manifest required")
    value = json.loads(Path(path).read_text())
    index = Path(value["index"])
    if digest(index) != value["index_sha256"]:
        raise ValueError("Original private feature index changed")
    index_value = json.loads(index.read_text())
    rows = [
        {**r, "source_kernel": r.get("source_kernel", value.get("default_source_kernel"))}
        for r in index_value["records"]
    ]
    if value.get("take") is not None:
        rows = rows[: value["take"]]
    if (
        value["uids_sha256"]
        != __import__("hashlib")
        .sha256(json.dumps([r["study_uid"] for r in rows], separators=(",", ":")).encode())
        .hexdigest()
    ):
        raise ValueError("Exact ordered fitting membership required")
    if (
        value["clinical_evaluation_used"] is not False
        or value["source_rows_filtered_before_numeric_read"] is not True
        or value["independent_internal_validation"] is not False
    ):
        raise ValueError("Fixed fitting-only and nonindependent-parent boundaries required")
    for name, expected in value["source_files"].items():
        if digest(name) != expected:
            raise ValueError("Frozen numerical source changed")
    data = FrozenFeatures(rows, value["roots"], value["labels"], value["masks"], value["binding"])
    return data, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipe", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("This accepted fitter uses CPU FP32 only")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if args.output.exists():
        raise ValueError("Fresh private engine output required; never overwrite a run")
    data, inputs = load_data(args.inputs, args.inputs_sha256)
    recipe = Recipe(**json.loads(args.recipe.read_text()))
    if recipe.embedding_dim != 1024 or recipe.targets != 12:
        raise ValueError("Exact real parent embedding and finding graph required")
    args.output.mkdir(mode=0o700, parents=True)
    started = time.monotonic()
    trainer = Trainer(data, recipe)
    if args.resume is not None:
        trainer.load(args.resume)

    def progress(record):
        if record["update"] % 25 == 0 or trainer.cursor == 0:
            print(
                json.dumps(
                    {"context_optimizer": record, "elapsed_seconds": time.monotonic() - started}
                ),
                flush=True,
            )

    trainer.run(
        stop_after=args.stop_after, epoch_directory=args.output / "epochs", progress=progress
    )
    trainer.save(args.output / "checkpoint.pt")
    prediction = args.output / "predictions.private.npy"
    with prediction.open("xb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        np.save(handle, trainer.predictions(), allow_pickle=False)
    result = {
        "status": "fixed_final_complete"
        if trainer.epoch == recipe.epochs
        else "stopped_at_update_boundary",
        "completed_updates": trainer.completed,
        "epochs": trainer.epoch,
        "studies": len(data),
        "trainable_parameters": sum(p.numel() for p in trainer.model.parameters()),
        "checkpoint_sha256": digest(args.output / "checkpoint.pt"),
        "prediction_sha256": digest(prediction),
        "inputs_sha256": args.inputs_sha256,
        "binding": inputs["binding"],
        "elapsed_seconds": time.monotonic() - started,
        "clinical_accuracy_evaluated": False,
        "independent_internal_validation": False,
        "competition_submissions": 0,
        "CNN_parameters_fitted": 0,
    }
    (args.output / "terminal.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
