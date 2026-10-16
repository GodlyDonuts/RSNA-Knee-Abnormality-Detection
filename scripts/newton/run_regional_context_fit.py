"""Fit a typed coherent spatial-input contrast with the frozen generic CPU engine."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

import rsna.competitive.regional_context_train as regional_module
import rsna.competitive.released_context as context_module
import rsna.competitive.released_context_train as numerical_module
from rsna.competitive.regional_context_train import (
    ENGINE,
    GENERIC_TRAINER_SHA,
    PARENT_SOURCE,
    POPULATIONS,
    REGIONAL_HELPER_SHA,
    REPRESENTATION,
    ROLES,
    RegionalFeatures,
)
from rsna.competitive.released_context_train import Recipe, Trainer, digest
from rsna.schema import TARGETS


def load_data(path, expected_sha, role):
    if digest(path) != expected_sha:
        raise ValueError("Exact private independently admitted regional inputs required")
    value = json.loads(Path(path).read_text())
    if (
        value.get("engine") != ENGINE
        or value.get("representation") != REPRESENTATION
        or value.get("population_stage") not in ("engineering32", "full3887")
        or value.get("clinical_evaluation_used") is not False
        or value.get("independent_internal_validation") is not False
        or value.get("source_rows_filtered_before_numeric_read") is not True
        or value.get("complete_coherent_cache_verified") is not True
        or value.get("targets") != list(TARGETS)
    ):
        raise ValueError("Explicit coherent fitting-only source and population admission required")
    index = Path(value["index"])
    if digest(index) != value["index_sha256"]:
        raise ValueError("Original coherent feature index changed")
    contents = json.loads(index.read_text())
    if (
        contents.get("representation") != REPRESENTATION
        or contents.get("parent_source_sha256") != PARENT_SOURCE
        or contents.get("source_sha256") != REGIONAL_HELPER_SHA
    ):
        raise ValueError("Unmixed same-forward feature index required")
    rows = [
        dict(r, source_key=r.get("source_key", value.get("default_source_key")))
        for r in contents["records"]
    ]
    expected_count, expected_uids_sha = POPULATIONS[value["population_stage"]]
    if value["uids_sha256"] != expected_uids_sha:
        raise ValueError("Original accepted fitting-only cohort hash required")
    if (
        len(rows) != expected_count
        or hashlib.sha256(
            json.dumps([r["study_uid"] for r in rows], separators=(",", ":")).encode()
        ).hexdigest()
        != value["uids_sha256"]
    ):
        raise ValueError("Exact admitted complete ordered fitting membership required")
    if (
        not value["source_files"]
        or value["binding"].get("numerical_trainer_sha256") != GENERIC_TRAINER_SHA
    ):
        raise ValueError("Unchanged numerical trainer must be bound explicitly")
    for name, sha in value["source_files"].items():
        if digest(name) != sha:
            raise ValueError("Frozen admitted numerical source changed")
    required_files = [Path(__file__).resolve()]
    for module in (regional_module, context_module, numerical_module):
        if module.__file__ is None:
            raise ValueError("Real immutable source files required")
        required_files.append(Path(module.__file__).resolve())
    if digest(numerical_module.__file__) != GENERIC_TRAINER_SHA or any(
        str(p) not in value["source_files"] for p in required_files
    ):
        raise ValueError("All actually imported frozen engine modules must be admitted")
    data = RegionalFeatures(
        rows, value["roots"], value["labels"], value["masks"], value["binding"], role
    )
    return data, value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("inputs", "recipe", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--inputs-sha256", required=True)
    parser.add_argument("--role", choices=ROLES, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--resume-sha256")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("This numerical engine is CPU FP32 only")
    if args.output.exists():
        raise ValueError("Original fitting output cannot be overwritten")
    if bool(args.resume) != bool(args.resume_sha256) or (
        args.resume is not None and digest(args.resume) != args.resume_sha256
    ):
        raise ValueError("Exact trusted original recovery checkpoint hash required")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    data, inputs = load_data(args.inputs, args.inputs_sha256, args.role)
    recipe = Recipe(**json.loads(args.recipe.read_text()))
    if (
        recipe.embedding_dim != 8192
        or recipe.targets != 12
        or recipe.width != 128
        or recipe.layers != 2
    ):
        raise ValueError("Exact matched8192input/128width/12target/twolayer graph required")
    if inputs["population_stage"] == "full3887" and recipe != Recipe(embedding_dim=8192):
        raise ValueError("Predeclared complete fixed10-epoch production recipe required")
    args.output.mkdir(mode=0o700, parents=True)
    started = time.monotonic()
    trainer = Trainer(data, recipe)
    if sum(p.numel() for p in trainer.model.parameters()) != 1334796:
        raise ValueError("Exact identical role graph and parameter count required")
    if args.resume is not None:
        trainer.load(args.resume)

    def progress(record):
        if trainer.cursor == 0 and inputs["population_stage"] == "full3887":
            snapshot = args.output / "epochs" / f"epoch-{trainer.epoch:03d}.pt"
            if snapshot.stat().st_size > 17 * 2**20:
                raise ValueError("Original raw head snapshot exceeds the admitted17MiB bound")
        if record["update"] % 25 == 0 or trainer.cursor == 0:
            print(json.dumps({"role": args.role, "regional_optimizer": record}), flush=True)

    trainer.run(
        stop_after=args.stop_after,
        epoch_directory=args.output / "epochs"
        if inputs["population_stage"] == "full3887"
        else None,
        progress=progress,
    )
    trainer.save(args.output / "checkpoint.pt")
    if (args.output / "checkpoint.pt").stat().st_size > 17 * 2**20:
        raise ValueError("Original raw head checkpoint exceeds the admitted17MiB bound")
    prediction = args.output / "predictions.private.npy"
    with prediction.open("xb") as handle:
        os.fchmod(handle.fileno(), 0o600)
        np.save(handle, trainer.predictions(), allow_pickle=False)
    terminal = {
        "engine": ENGINE,
        "role": args.role,
        "status": "fixed_final_complete"
        if trainer.epoch == recipe.epochs
        else "stopped_at_update_boundary",
        "diagnostic_only": inputs["population_stage"] == "engineering32",
        "immutable_production_epoch_snapshots": inputs["population_stage"] == "full3887",
        "completed_updates": trainer.completed,
        "epochs": trainer.epoch,
        "studies": len(data),
        "trainable_parameters": 1334796,
        "checkpoint_sha256": digest(args.output / "checkpoint.pt"),
        "prediction_sha256": digest(prediction),
        "inputs_sha256": args.inputs_sha256,
        "recipe_sha256": digest(args.recipe),
        "numerical_trainer_sha256": GENERIC_TRAINER_SHA,
        "binding": data.binding,
        "elapsed_seconds": time.monotonic() - started,
        "fitting_source_probabilities_used": True,
        "clinical_accuracy_evaluated": False,
        "independent_internal_validation": False,
        "competition_submissions": 0,
        "CNN_parameters_fitted": 0,
    }
    (args.output / "terminal.json").write_text(
        json.dumps(terminal, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps(terminal), flush=True)


if __name__ == "__main__":
    main()
