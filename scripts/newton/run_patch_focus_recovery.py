"""Finite full/stop1/resume recovery for the matched source patch-focus arms.

PREPARED ONLY: this issue performs no GPU reservation, dispatch, transfer, upload or
competition submission. Root funds and runs it once on one assigned GPU.

The UNCHANGED canonical trainer ``rsna.competitive.train`` runs in distinct
processes from one verified source tree: one uninterrupted engineering fit of the
reviewed source-slot r32 recipe (resource and first-update reference only), then for
each patch-focus arm (``control`` then ``candidate``) a full engineering fit, a fresh
replay stopped after one optimizer update (exit 75) and its resume. Acceptance
requires complete learned/frozen-base/AdamW/LR/counter/RNG/scaler/history/prediction
equality, the source-slot graph plus only ``patch_query``, a never-moving control
query, a moved candidate query, first-update losses equal to the source-slot fit and
the matched binding. Added elapsed time and peak GPU memory versus the source-slot fit
are measured. Engineering evidence only: never accuracy, a recipe choice or a submission.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsna.competitive.train import TrainConfig, learning_rate_multiplier
from rsna.schema import TARGETS
from scripts.newton import run_dense_suffix_recovery as generic
from scripts.newton.probe_patch_witness_trainer_recovery import decoded, digest, numerical_updates

RECIPE = Path("initiative/experiments/competitive-v1/source-patch-focus-v1")
REFERENCE = Path("initiative/experiments/competitive-v1/source-matched-slot-v1/r32-config.json")
ARMS = {"control": "control-r32-config.json", "candidate": "candidate-r32-config.json"}
FOCUS = {"control": "disabled", "candidate": "query_residual"}
PATCH_FIELDS = ("kind", "patch_focus", "patch_pool")
# Frozen capacity factor: DINOv3-L rank-32 residuals on all 120 targets
# (96 attention projections in 24 layers plus 24 FFN projections in the last 12).
ADAPTER = {"rank": 32, "ffn_last_layers": 12}
TARGET_MODULES = 120
ENCODER_TRAINABLE = 10_223_616
EMBEDDING_DIM = 1024
QUERY = "patch_query"
ENGINEERING = generic.ENGINEERING
RESERVE = generic.RESERVE
OUTPUT_CAP_BYTES = 12 * 2**30
sha = generic.sha
verify_source = generic.verify_source
subprocess_executor = generic.subprocess_executor


def load_recipe(source: Path) -> dict[str, dict]:
    """Reference plus both arms; arms equal the reference except the classifier policy."""
    reference = json.loads((source / REFERENCE).read_text())
    configs = {arm: json.loads((source / RECIPE / name).read_text()) for arm, name in ARMS.items()}
    slot = reference["classifier_strategy"]
    for arm, values in configs.items():
        strategy = values["classifier_strategy"]
        if (
            {k: v for k, v in values.items() if k != "classifier_strategy"}
            != {k: v for k, v in reference.items() if k != "classifier_strategy"}
            or {k: v for k, v in strategy.items() if k not in PATCH_FIELDS}
            != {k: v for k, v in slot.items() if k != "kind"}
            or strategy["kind"] != "source_patch_focus_v1"
            or strategy["patch_focus"] != FOCUS[arm]
            or slot["kind"] != "source_matched_slot_v1"
        ):
            raise ValueError("Arm differs from the source-slot recipe beyond its focus policy")
    for values in (reference, *configs.values()):
        TrainConfig(**values).validate()
        adapter = values["encoder"].get("adapter", {})
        if (
            any(adapter.get(key) != value for key, value in ADAPTER.items())
            or "suffix_adapter" in values["encoder"]
            or values["batch_size"] * values["accumulation_steps"] != 8
            or values["checkpoint_format"] != "frozen_base_delta"
            or values["public_input"] is None
            or values["freeze_encoder"] is not True
            or values["epochs"] != 10
            or values.get("diagnostic_study_limit") is not None
        ):
            raise ValueError("Recipe differs from the frozen rank32/all120 source-slot factor")
    return {"reference": reference, **configs}


def engineering_config(recipe: Mapping, spec: Mapping) -> dict:
    """Mounted official spec plus the recipe's adapter and the fixed engineering scope."""
    encoder = dict(recipe["encoder"])
    adapter = encoder.pop("adapter")
    if set(spec) != set(encoder) or any(
        spec[key] != encoder[key] for key in encoder if key not in {"weights_path", "config_path"}
    ):
        raise ValueError("Mounted encoder spec differs from the pinned official source")
    values = dict(recipe, encoder=dict(spec, adapter=adapter), **ENGINEERING)
    TrainConfig(**values).validate()
    return values


def check_graph(full: Mapping, reference: Mapping) -> dict:
    """Source-slot learned graph plus only the zero-initialized ``[12,1024]`` query."""
    model, base = full["model"], reference["model"]
    encoder = {k for k in model if k.startswith("encoder.")}
    if (
        set(model) - set(base) != {QUERY}
        or set(base) - set(model)
        or any(model[k].shape != base[k].shape for k in base)
        or tuple(model[QUERY].shape) != (len(TARGETS), EMBEDDING_DIM)
        or len(encoder) != 2 * TARGET_MODULES
        or not all(k.endswith((".lora_A", ".lora_B")) for k in encoder)
        or sum(model[k].numel() for k in encoder) != ENCODER_TRAINABLE
        or full["model_storage"]["frozen_tensors"] != reference["model_storage"]["frozen_tensors"]
    ):
        raise ValueError("Learned graph differs from source-slot plus only the patch query")
    return {
        "learned_tensors": len(model),
        "encoder_learned_tensors": len(encoder),
        "encoder_learned_parameters": ENCODER_TRAINABLE,
        "patch_query_parameters": model[QUERY].numel(),
    }


def check_query(arm: str, full: Mapping) -> dict:
    """Control's gate-0 query never moves (zero gradient, zero AdamW step); candidate's does."""
    query = full["model"][QUERY]
    if not torch.isfinite(query).all():
        raise ValueError("Patch query is nonfinite")
    moved = bool(query.count_nonzero())
    if arm == "control" and moved:
        raise ValueError("Disabled control query received a nonzero gradient")
    if arm == "candidate" and not moved:
        raise ValueError("Candidate query never received an informative gradient")
    rows = int((query.abs().sum(1) > 0).sum())
    return {"patch_query_rows_moved": rows, "patch_query_abs_max": float(query.abs().max())}


def resources(root: Path) -> dict:
    result = json.loads((root / "full" / "result.json").read_text())
    peak, elapsed = result["cuda_peak_memory_bytes"], result["elapsed_seconds"]
    if not (isinstance(elapsed, float) and math.isfinite(elapsed) and elapsed > 0):
        raise ValueError("Full fit elapsed time must be finite and positive")
    if peak is not None and (type(peak) is not int or peak <= 0):
        raise ValueError("Peak GPU memory must be a positive byte count")
    return {"elapsed_seconds": elapsed, "cuda_peak_memory_bytes": peak}


def added(arm: Mapping, reference: Mapping) -> dict:
    peak = (
        None
        if arm["cuda_peak_memory_bytes"] is None
        else arm["cuda_peak_memory_bytes"] - reference["cuda_peak_memory_bytes"]
    )
    return {
        "added_elapsed_seconds": arm["elapsed_seconds"] - reference["elapsed_seconds"],
        "elapsed_ratio": arm["elapsed_seconds"] / reference["elapsed_seconds"],
        "added_cuda_peak_memory_bytes": peak,
    }


def compare_first_update(histories: Mapping[str, list]) -> dict:
    """Zero query: every arm's first full-batch loss equals the source-slot fit bitwise."""
    first = {name: rows[0] for name, rows in histories.items()}
    if len({row["loss"] for row in first.values()}) != 1:
        raise ValueError("Zero-query first-update loss differs from the source-slot fit")
    if any(not row.get("encoder_gradient_norm", 0) > 0 for row in first.values()):
        raise ValueError("Adapter gradients were absent at the first update")
    norms = {name: row["gradient_norm"] for name, row in first.items()}
    control = abs(norms["control"] - norms["reference"]) / norms["reference"]
    if not control <= 1e-6:
        raise ValueError("Control first-update gradient norm differs from the source-slot fit")
    return {
        "first_update_loss_equal": True,
        "control_gradient_norm_bitwise_reference": norms["control"] == norms["reference"],
        "control_gradient_norm_relative_difference": control,
        "control_encoder_gradient_norm_bitwise_reference": (
            first["control"]["encoder_gradient_norm"] == first["reference"]["encoder_gradient_norm"]
        ),
    }


def compare_matched(bindings: Mapping[str, dict]) -> None:
    """Same data, roster, folds, release, source and settings; only the focus policy differs."""
    values = {name: json.loads(json.dumps(binding)) for name, binding in bindings.items()}
    for name, value in values.items():
        strategy = value["config"]["classifier_strategy"]
        for field in PATCH_FIELDS:
            strategy.pop(field, None)
    if not values["reference"] == values["control"] == values["candidate"]:
        raise ValueError("Matched runs differ beyond the classifier focus policy")


def run(args: argparse.Namespace, executor: generic.Executor | None = None) -> dict:
    source = args.source.resolve()
    code = verify_source(source, json.loads(args.source_manifest.read_text()))
    recipe = load_recipe(source)
    spec = json.loads(args.encoder_spec.read_text())
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("Exactly one visible assigned GPU required for sequential runs")
    if args.max_output_bytes > OUTPUT_CAP_BYTES:
        raise ValueError("Recovery output cap exceeds its explicit bound")
    if shutil.disk_usage(args.output.parent).free < args.max_output_bytes + RESERVE:
        raise ValueError("Admitted output cap plus twenty GiB reserve unavailable")
    args.output.mkdir(mode=0o700)
    execute = executor or subprocess_executor(args, source)
    results, bindings, histories, measured, walls = {}, {}, {}, {}, {}
    checkpoints: dict[str, dict] = {}
    for name in ("reference", *ARMS):
        values = engineering_config(recipe[name], spec)
        root = args.output / name
        root.mkdir(mode=0o700)
        config = root / "config.json"
        with config.open("x") as handle:
            handle.write(json.dumps(values, sort_keys=True, indent=2) + "\n")
        walls[name] = {}

        def phase(label: str, extra: list[str], expected: int) -> None:
            used = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
            if used > args.max_output_bytes:
                raise ValueError("Admitted recovery output cap exceeded")
            print(json.dumps({"run": name, "phase": label}), flush=True)
            started = time.monotonic()
            if execute(root, label, config, extra) != expected:
                raise ValueError("Canonical trainer failed; original phase logs retained")
            walls[name][label] = time.monotonic() - started

        full, replay = root / "full", root / "replay"
        phase("full", ["--output", str(full)], 0)
        checkpoint = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=True)
        checkpoints[name] = checkpoint
        histories[name] = numerical_updates(json.loads((full / "invocation.json").read_text()))
        measured[name] = resources(root)
        binding = checkpoint["binding"]
        if name != "reference":
            phase("stop", ["--output", str(replay), "--stop-after-updates", "1"], 75)
            partial = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
            first = json.loads((replay / "invocation.json").read_text())
            phase("resume", ["--output", str(replay), "--resume", str(replay / "checkpoint.pt")], 0)
            results[name] = recover(root, partial, first)
            results[name] |= check_graph(checkpoint, checkpoints["reference"])
            results[name] |= check_query(name, checkpoint)
            binding = partial["binding"]
        if binding["code_sha256"] != code or binding["device_type"] != args.device:
            raise ValueError("Trainer ran a different numerical source or device")
        bindings[name] = binding
    compare_matched(bindings)
    first_update = compare_first_update(histories)
    result = {
        "status": "actual_matched_patch_focus_full_stop1_resume_recovery_accepted",
        "arms": results,
        "first_update_versus_source_slot": first_update,
        "resources": {
            "full_fit": measured,
            "phase_wall_seconds": walls,
            "added_versus_source_slot": {
                arm: added(measured[arm], measured["reference"]) for arm in ARMS
            },
        },
        "source_manifest_sha256": sha(args.source_manifest),
        "recipe_sha256": {
            "reference": sha(source / REFERENCE),
            **{arm: sha(source / RECIPE / name) for arm, name in ARMS.items()},
        },
        "device": args.device,
        "GPUs_used_concurrently": 1 if args.device == "cuda" else 0,
        "diagnostic_only": True,
        "recipe_selection_evidence": False,
        "clinical_accuracy_evidence": False,
        "competition_submission": False,
    }
    with (args.output / "accepted.json").open("x") as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return result


def recover(root: Path, partial: dict, first: dict) -> dict:
    """Complete-state equality of an uninterrupted fit and its stop1/resume replay."""
    full_dir, replay_dir = root / "full", root / "replay"
    full = torch.load(full_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    resumed = torch.load(replay_dir / "checkpoint.pt", map_location="cpu", weights_only=True)
    history = numerical_updates(json.loads((full_dir / "invocation.json").read_text()))
    last = json.loads((replay_dir / "invocation.json").read_text())
    config = full["binding"]["config"]
    total = len(history)
    if (
        total < 2 * config["epochs"]
        or total % config["epochs"]
        or [row["optimizer_steps"] for row in history] != list(range(1, total + 1))
    ):
        raise ValueError("Every engineering update must be a finite optimizer step")
    for checkpoint in (full, resumed):
        if (
            (checkpoint.get("epoch"), checkpoint.get("next_update"), checkpoint.get("step"))
            != (config["epochs"], 0, total)
            or checkpoint.get("schema_version") != 2
            or checkpoint.get("optimizer_boundary") is not True
            or checkpoint.get("model_storage", {}).get("format") != "frozen_base_delta_v1"
            or len(checkpoint.get("rng_by_rank", [])) != 1
            or set(checkpoint["rng_by_rank"][0]) != {"torch", "cuda", "python", "numpy"}
            or (config["precision"] == "fp16") != ("amp_scaler" in checkpoint)
        ):
            raise ValueError("Complete learned/base/AdamW/RNG/scaler delta state required")
    if (partial.get("epoch"), partial.get("next_update"), partial.get("step")) != (0, 1, 1):
        raise ValueError("Exact interrupted first optimizer boundary required")
    if (
        partial["binding"] != full["binding"]
        or partial["model_storage"] != full["model_storage"]
        or decoded(full) != decoded(resumed)
    ):
        raise ValueError("Complete model/base/AdamW/LR/counter/RNG/scaler recovery differs")
    if numerical_updates(first) + numerical_updates(last) != history:
        raise ValueError("Interrupted plus resumed numerical history differs from the full fit")
    model = full["model"]
    if set(full["model_storage"]["learned_tensors"]) != set(model) or set(
        full["model_storage"]["frozen_tensors"]
    ) & set(model):
        raise ValueError("Learned and frozen-base tensor partitions differ")
    multiplier = learning_rate_multiplier(
        total - 1, total, config["warmup_updates"], config["lr_schedule"]
    )
    groups = full["optimizer"]["param_groups"]
    if (
        [group["lr"] for group in groups]
        != [config["head_lr"] * multiplier, config["encoder_lr"] * multiplier]
        or len(full["optimizer"]["state"]) != len(model)
        or any(float(state["step"]) != total for state in full["optimizer"]["state"].values())
    ):
        raise ValueError("AdamW groups/LR schedule/state differ from the declared recipe")
    csv, replayed = (d / "outer_predictions.csv" for d in (full_dir, replay_dir))
    results = [json.loads((d / "result.json").read_text()) for d in (full_dir, replay_dir)]
    if csv.read_bytes() != replayed.read_bytes() or results[0]["metrics"] != results[1]["metrics"]:
        raise ValueError("Final prediction bytes or metrics differ after recovery")
    prediction = pd.read_csv(csv)
    scores = prediction[list(TARGETS)].to_numpy()
    if (
        list(prediction.columns) != ["study_uid", "fold", "seed", *TARGETS]
        or prediction.study_uid.duplicated().any()
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
    ):
        raise ValueError("Exact finite bounded outer predictions required")
    return {
        "complete_state_sha256": digest(decoded(full)),
        "checkpoint_sha256": sha(full_dir / "checkpoint.pt"),
        "prediction_sha256": sha(csv),
        "finite_updates": history,
        "frozen_base_tensors": len(full["model_storage"]["frozen_tensors"]),
        "exact_complete_state_history_and_prediction_recovery": True,
    }


def parser() -> argparse.ArgumentParser:
    value = generic.parser()
    value.description = __doc__
    value.set_defaults(max_output_bytes=OUTPUT_CAP_BYTES)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    if not math.isfinite(arguments.remaining_seconds) or arguments.remaining_seconds <= 0:
        raise SystemExit("--remaining-seconds must be positive")
    print(json.dumps(run(arguments), sort_keys=True), flush=True)
