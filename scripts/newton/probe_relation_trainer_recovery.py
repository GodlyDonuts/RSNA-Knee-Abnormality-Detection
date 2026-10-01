"""Matched real-MRI production-trainer recovery pilot; never a submitted model.

The caller must supply independently materialized original cache members for
sixteen training and sixteen outer-fold studies. This script does not stage or
dispatch Slurm work. Its checkpoints are diagnostic, not production candidates.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.delta_checkpoint import _tensor_record
from rsna.competitive.folds import load_folds
from rsna.competitive.train import _uid_hash, select_development
from rsna.release import load_verified_release
from rsna.schema import TARGETS
from scripts.newton.audit_relation_geometry import digest
from scripts.newton.measure_relation_gradients import (
    admit_cpu_receipt,
    admitted_batches,
    matched_configs,
    validate_cache,
)


def state_record(value):
    """Hash decoded values, including every optimizer and RNG tensor."""
    if isinstance(value, torch.Tensor):
        if not torch.isfinite(value).all():
            raise ValueError("Nonfinite decoded checkpoint tensor")
        return {"type": "tensor", "value": _tensor_record(value)}
    if isinstance(value, dict):
        return {
            "type": "dict",
            "value": [
                [state_record(key), state_record(item)]
                for key, item in sorted(value.items(), key=lambda row: repr(row[0]))
            ],
        }
    if isinstance(value, (tuple, list)):
        return {"type": type(value).__name__, "value": [state_record(item) for item in value]}
    if value is None or type(value) in (str, int, bool):
        return {"type": type(value).__name__, "value": value}
    if type(value) is float and math.isfinite(value):
        return {"type": "float", "value": value}
    raise ValueError("Unsupported or nonfinite decoded checkpoint value")


def admit_gradient(receipt, decision, cpu_sha, input_binding, configs):
    expected = {name: asdict(config) for name, config in configs.items()}
    if (
        receipt.get("status")
        != "actual_official_weight_full224_BF16_training_gradient_measurement_complete"
        or receipt.get("optimizer_updates") != 0
        or receipt.get("clinical_accuracy_evidence") is not False
        or receipt.get("input_binding") != input_binding
        or receipt.get("CPU_acceptance_sha256") != cpu_sha
        or receipt.get("configs") != expected
        or receipt.get("same_classification_gradients_logits_pairs_across_arms") is not True
        or receipt.get("MRI_only_12_output_inference") is not True
        or decision.get("status")
        != "training_only_gradient_scale_review_accepted_for_bounded_pilot"
        or decision.get("configs") != expected
        or not isinstance(decision.get("reason"), str)
        or not decision["reason"].strip()
    ):
        raise ValueError(
            "Actual gradient measurement and explicit training-only scale review required"
        )
    results = receipt.get("results", {})
    if set(results) != set(configs) or any(len(rows) != 2 for rows in results.values()):
        raise ValueError("Exactly two actual effective batches per matched gradient arm required")
    for name, rows in results.items():
        for reference, row in zip(results["zero"], rows):
            if any(
                row.get(key) != reference.get(key)
                for key in (
                    "classification_gradient_sha256",
                    "classification_logits_sha256",
                    "pair_roster_sha256",
                    "observed_target_counts",
                )
            ):
                raise ValueError(
                    "Actual gradient controls differ in classification or observations"
                )
            gradient = row.get("encoder_gradients", {})
            for key in ("classification_norm", "unweighted_relation_norm", "combined_norm"):
                value = gradient.get(key)
                if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                    raise ValueError(
                        "Finite nonzero real encoder gradients required for recovery pilot"
                    )
            if (
                row.get("microbatches") != 8
                or not 0 < row.get("represented_studies", 0) <= 8
                or row.get("relation_pairs", 0) < 1
                or not math.isclose(
                    row.get("weighted_relation_loss", -1),
                    configs[name].relation["weight"] * row.get("relation_loss", -1),
                    rel_tol=1e-6,
                    abs_tol=1e-9,
                )
            ):
                raise ValueError(
                    "Gradient roster or weighted objective differs from the exact recipe"
                )


def invocation_values(record):
    rows = record.get("updates", [])
    keys = (
        "epoch",
        "update",
        "optimizer_steps",
        "loss",
        "gradient_norm",
        "observed_cells",
        "classification_loss",
        "relation_loss",
        "relation_studies",
        "relation_weight",
    )
    if not rows or any(any(key not in row for key in keys) for row in rows):
        raise ValueError("Actual production-trainer relation update records required")
    if any(
        type(row[key]) not in (int, float) or not math.isfinite(row[key])
        for row in rows
        for key in keys
    ):
        raise ValueError("Nonfinite actual production-trainer update")
    return [{key: row[key] for key in keys} for row in rows]


def compare_completed(
    full, resumed, partial, full_updates, first, last, predictions, replay_predictions, policy
):
    """Exact payload/trajectory comparison, independent of pickle file hashes."""
    for checkpoint in (full, resumed):
        if (
            checkpoint.get("schema_version") != 2
            or checkpoint.get("optimizer_boundary") is not True
            or (checkpoint.get("epoch"), checkpoint.get("next_update"), checkpoint.get("step"))
            != (2, 0, 4)
            or checkpoint.get("binding", {}).get("world_size") != 1
            or checkpoint["binding"].get("config", {}).get("relation") != policy
            or len(checkpoint.get("rng_by_rank", [])) != 1
            or not checkpoint.get("model")
            or not checkpoint.get("optimizer", {}).get("state")
            or checkpoint.get("model_storage", {}).get("format") != "frozen_base_delta_v1"
        ):
            raise ValueError("Exact two-epoch/four-update delta boundary required")
    for invocation, status, exit_code in (
        (full_updates, "completed", 0),
        (first, "checkpointed_for_resume", 75),
        (last, "completed", 0),
    ):
        if invocation.get("status") != status or invocation.get("exit_code") != exit_code:
            raise ValueError("Actual full/stop/resume invocation status differs")
    updates = invocation_values(full_updates)
    if [(row["epoch"], row["update"], row["optimizer_steps"]) for row in updates] != [
        (0, 1, 1),
        (0, 2, 2),
        (1, 1, 3),
        (1, 2, 4),
    ]:
        raise ValueError("Exact four production-trainer update boundaries required")
    if (
        (partial.get("epoch"), partial.get("next_update"), partial.get("step")) != (0, 1, 1)
        or partial.get("binding") != full["binding"]
        or state_record(full) != state_record(resumed)
        or invocation_values(first) + invocation_values(last) != invocation_values(full_updates)
        or len(invocation_values(first)) != 1
        or len(invocation_values(last)) != 3
        or predictions != replay_predictions
    ):
        raise ValueError(
            "Actual trainer interruption/resume changed state, updates or MRI predictions"
        )
    return digest(state_record(full))


def execute(command, output, expected_exit):
    """Run a fresh production CLI process and preserve its exact small logs."""
    with (
        output.with_suffix(".stdout.txt").open("x") as out,
        output.with_suffix(".stderr.txt").open("x") as err,
    ):
        completed = subprocess.run(command, stdout=out, stderr=err, timeout=1200)
    if completed.returncode != expected_exit:
        raise ValueError(
            f"Production trainer returned{completed.returncode}; expected{expected_exit}"
        )


def validate_predictions(checkpoint, frame, train, outer, source, encoder_provenance):
    binding = checkpoint["binding"]
    expected_code = {
        name.removeprefix("rsna/"): row["sha256"]
        for name, row in source["files"].items()
        if name.startswith("rsna/") and name.endswith(".py")
    }
    if (
        not expected_code
        or binding["training_uids_sha256"] != _uid_hash(train)
        or binding["outer_uids_sha256"] != _uid_hash(outer)
        or binding["code_sha256"] != expected_code
        or binding["config"]["encoder"] != encoder_provenance
        or checkpoint["model_storage"]["encoder_provenance"] != encoder_provenance
        or binding.get("device_type") != "cuda"
        or binding["config"].get("precision") != "bf16"
        or set(frame.study_uid) != set(outer.study_uid)
        or frame.study_uid.duplicated().any()
        or frame[list(TARGETS)].shape != (16, 12)
        or not np.isfinite(frame[list(TARGETS)].to_numpy()).all()
        or not ((frame[list(TARGETS)] >= 0) & (frame[list(TARGETS)] <= 1)).to_numpy().all()
    ):
        raise ValueError("Actual trainer input bindings or MRI-only predictions differ")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config-root",
        "proof",
        "request",
        "run",
        "studies",
        "artifact-root",
        "folds",
        "series",
        "cache",
        "archive",
        "output",
        "source-manifest",
        "cpu-acceptance",
        "gradient-acceptance",
        "gradient-decision",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID", "").isdecimal()
        or os.environ.get("SLURM_CPUS_PER_TASK") != "2"
        or not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or not torch.cuda.is_bf16_supported()
        or "H100" not in torch.cuda.get_device_name(0)
        or int(os.environ.get("WORLD_SIZE", "1")) != 1
    ):
        raise ValueError("Actual one-H100/two-CPU/world1 BF16 pilot allocation required")
    if args.output.exists():
        raise FileExistsError("Pilot output exists; reconcile original attempt")
    configs = matched_configs(args.config_root)
    if any(
        config.epochs != 2 or config.submission_every_epochs != 1 for config in configs.values()
    ):
        raise ValueError("This engineering pilot requires the exact prepared two-epoch recipe")
    torch.set_num_threads(2)
    batches, input_binding = admitted_batches(args, configs["physical"])
    admit_cpu_receipt(json.loads(args.cpu_acceptance.read_text()), input_binding, configs)
    gradient, decision = (
        json.loads(path.read_text()) for path in (args.gradient_acceptance, args.gradient_decision)
    )
    if decision.get("gradient_receipt_sha256") != file_sha256(args.gradient_acceptance):
        raise ValueError("Training-only scale decision differs from the actual measurement")
    admit_gradient(gradient, decision, file_sha256(args.cpu_acceptance), input_binding, configs)
    del batches  # Original inputs were verified; real CLI loaders will decode them again.
    manifest, release = load_verified_release(args.studies, args.artifact_root)
    train, outer = select_development(
        manifest, load_folds(args.folds, manifest, release), configs["physical"].fold
    )
    train, outer = train.head(16), outer.head(16)
    if len(outer) != 16 or set(train.study_uid) & set(outer.study_uid):
        raise ValueError(
            "Exactly sixteen distinct admitted outer-fold engineering studies required"
        )
    outer_binding = validate_cache(
        args.cache,
        args.archive,
        json.loads(args.proof.read_text()),
        outer.study_uid.tolist(),
        pd.read_csv(args.series),
    )
    args.output.mkdir(mode=0o700)
    results, started = {}, time.monotonic()
    for name, config in configs.items():
        full, replay = args.output / (name + "-full"), args.output / (name + "-replay")
        base = [sys.executable, "-B", "-m", "rsna.competitive.train"]
        paths = {
            "config": args.config_root / f"{name}-pilot-config.json",
            **{key: getattr(args, key) for key in ("studies", "series", "cache", "folds")},
            "artifact-root": args.artifact_root,
        }
        for key, path in paths.items():
            base += ["--" + key, str(path)]
        base += ["--device", "cuda"]
        execute(base + ["--output", str(full)], args.output / (name + "-full"), 0)
        execute(
            base + ["--output", str(replay), "--stop-after-updates", "1"],
            args.output / (name + "-stop"),
            75,
        )
        partial = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
        first = json.loads((replay / "invocation.json").read_text())
        execute(
            base + ["--output", str(replay), "--resume", str(replay / "checkpoint.pt")],
            args.output / (name + "-resume"),
            0,
        )
        checkpoint = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=True)
        restored = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
        full_updates = json.loads((full / "invocation.json").read_text())
        replay_updates = json.loads((replay / "invocation.json").read_text())
        predicted = (full / "outer_predictions.csv").read_bytes()
        replay_predicted = (replay / "outer_predictions.csv").read_bytes()
        state_sha = compare_completed(
            checkpoint,
            restored,
            partial,
            full_updates,
            first,
            replay_updates,
            predicted,
            replay_predicted,
            config.relation,
        )
        if checkpoint["binding"]["config"] != {
            **asdict(config),
            "encoder": gradient["encoder_provenance"],
        }:
            raise ValueError("Actual production-trainer numerical recipe differs")
        frame = pd.read_csv(full / "outer_predictions.csv")
        validate_predictions(
            checkpoint,
            frame,
            train,
            outer,
            json.loads(args.source_manifest.read_text()),
            gradient["encoder_provenance"],
        )
        results[name] = {
            "decoded_complete_checkpoint_sha256": state_sha,
            "checkpoint_sha256": file_sha256(full / "checkpoint.pt"),
            "checkpoint_bytes": (full / "checkpoint.pt").stat().st_size,
            "predictions_sha256": file_sha256(full / "outer_predictions.csv"),
            "actual_optimizer_updates_per_trajectory": 4,
            "interrupted_boundary": {"epoch": 0, "next_update": 1, "step": 1},
            "exact_complete_model_optimizer_RNG_and_predictions_recovery": True,
            "updates": invocation_values(full_updates),
            "training_elapsed_seconds_before_final_inference": full_updates["updates"][-1][
                "elapsed_seconds"
            ],
            "peak_GPU_allocated_bytes": json.loads((full / "result.json").read_text())[
                "cuda_peak_memory_bytes"
            ],
        }
        del partial, checkpoint, restored
    record = {
        "status": "actual_full224_BF16_matched_production_trainer_recovery_passed",
        "diagnostic_only": True,
        "clinical_accuracy_evidence": False,
        "production_candidate": False,
        "GPU": torch.cuda.get_device_name(0),
        "torch_version": str(torch.__version__),
        "input_binding": input_binding,
        "outer_engineering_cache_binding": outer_binding,
        "outer_studies": 16,
        "outer_predictions_used_for_model_or_recipe_selection": False,
        "configs": {name: asdict(config) for name, config in configs.items()},
        "source_probe_sha256": file_sha256(Path(__file__)),
        "CPU_acceptance_sha256": file_sha256(args.cpu_acceptance),
        "gradient_acceptance_sha256": file_sha256(args.gradient_acceptance),
        "gradient_decision_sha256": file_sha256(args.gradient_decision),
        "results": results,
        "elapsed_seconds_includes_recovery_and_inference_not_production_throughput": time.monotonic()
        - started,
    }
    with (args.output / "recovery-verified.json").open("x") as handle:
        json.dump(record, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": record["status"], "production_candidate": False}), flush=True)


if __name__ == "__main__":
    main()
