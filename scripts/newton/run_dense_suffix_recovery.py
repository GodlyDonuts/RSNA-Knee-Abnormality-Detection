"""Immutable sequential full/stop1/resume recovery for the matched suffix-capacity arms.

PREPARED ONLY: this issue performs no GPU reservation, dispatch, transfer, upload or
competition submission. Root admits fresh funding and runs it once on an owning node.

For each arm (``suffix_r32`` then ``dense``) the UNCHANGED canonical trainer
``rsna.competitive.train`` runs in three distinct processes from one verified source
tree: a full two-epoch engineering fit, a fresh replay stopped after one optimizer
update (exit 75), and its resume. Acceptance requires complete equality of learned
model, frozen-base records, AdamW state/LR, counters, every RNG stream, AMP scaler,
numerical history and prediction bytes, the declared pinned trainable graph and
the matched-arm binding. Engineering evidence only, never accuracy or a candidate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsna.competitive.train import TrainConfig, learning_rate_multiplier
from rsna.schema import TARGETS
from scripts.newton.probe_patch_witness_trainer_recovery import decoded, digest, numerical_updates

RECIPE = Path("initiative/experiments/competitive-v1/dense-suffix-capacity-v1")
ARMS = {"suffix_r32": "suffix-r32-config.json", "dense": "dense-config.json"}
# Pinned DINOv3-L, frozen prefix 12: 72 suffix projections.
PINNED = {
    "suffix_r32": {"method": "suffix_rs_lora_v1", "trainable_parameters": 7_077_888, "lr": 1e-4},
    "dense": {"method": "dense_suffix_delta_v1", "trainable_parameters": 150_994_944, "lr": 1e-5},
}
ENGINEERING = dict(
    epochs=2,
    diagnostic_study_limit=16,
    warmup_updates=2,
    submission_every_epochs=0,
    min_free_gib=20,
)
RESERVE = 20 * 2**30
Executor = Callable[[Path, str, Path, list[str]], int]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_recipe(recipe: Path) -> dict[str, dict]:
    """Both predeclared arms; they may differ only in suffix policy and encoder LR."""
    configs = {arm: json.loads((recipe / name).read_text()) for arm, name in ARMS.items()}
    first, second = configs["suffix_r32"], configs["dense"]
    if {k for k in first.keys() | second.keys() if first.get(k) != second.get(k)} != {
        "encoder",
        "encoder_lr",
    } or {
        k
        for k in first["encoder"].keys() | second["encoder"].keys()
        if first["encoder"].get(k) != second["encoder"].get(k)
    } != {"suffix_adapter"}:
        raise ValueError("Matched arms may differ only in suffix policy and predeclared LR")
    for arm, values in configs.items():
        TrainConfig(**values).validate()
        if (
            values["encoder"]["suffix_adapter"]["method"] != PINNED[arm]["method"]
            or values["encoder_lr"] != PINNED[arm]["lr"]
            or values["batch_size"] * values["accumulation_steps"] != 8
            or values["checkpoint_format"] != "frozen_base_delta"
            or values["public_input"] is None
        ):
            raise ValueError("Arm differs from its predeclared matched recipe")
    return configs


def engineering_config(recipe: Mapping, spec: Mapping) -> dict:
    """Mounted official spec plus the recipe's suffix policy and the fixed engineering scope."""
    encoder = dict(recipe["encoder"])
    policy = encoder.pop("suffix_adapter")
    if set(spec) != set(encoder) or any(
        spec[key] != encoder[key] for key in encoder if key not in {"weights_path", "config_path"}
    ):
        raise ValueError("Mounted encoder spec differs from the pinned official source")
    values = dict(recipe, encoder=dict(spec, suffix_adapter=policy), **ENGINEERING)
    TrainConfig(**values).validate()
    return values


def compare_recovery(arm_root: Path, partial: dict, first: dict, expected: Mapping) -> dict:
    """Complete-state equality of an uninterrupted fit and its stop1/resume replay."""
    full_dir, replay_dir = arm_root / "full", arm_root / "replay"
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
    if not history[0].get("encoder_gradient_norm", 0) > 0:
        raise ValueError("Suffix residual gradients were absent at the first update")
    provenance = config["encoder"]["suffix_adapter"]
    suffixes = (
        (".dense_delta",)
        if provenance["method"] == "dense_suffix_delta_v1"
        else (".lora_A", ".lora_B")
    )
    learned_encoder = {
        f"encoder.backbone.{module}{suffix}"
        for module in provenance["target_modules"]
        for suffix in suffixes
    }
    model = full["model"]
    if (
        provenance["method"] != expected["method"]
        or provenance["trainable_parameters"] != expected["trainable_parameters"]
        or config["encoder_lr"] != expected["lr"]
        or {k for k in model if k.startswith("encoder.")} != learned_encoder
        or sum(model[k].numel() for k in learned_encoder) != expected["trainable_parameters"]
        or set(full["model_storage"]["learned_tensors"]) != set(model)
        or set(full["model_storage"]["frozen_tensors"]) & set(model)
    ):
        raise ValueError("Trainable graph differs from the declared suffix arm")
    changed = [k for k in learned_encoder if not k.endswith(".lora_A")]
    if not all(bool(model[k].count_nonzero()) for k in changed):
        raise ValueError("A declared zero-initialized residual never changed")
    multiplier = learning_rate_multiplier(
        total - 1, total, config["warmup_updates"], config["lr_schedule"]
    )
    groups = full["optimizer"]["param_groups"]
    expected_lrs = [config["head_lr"] * multiplier, config["encoder_lr"] * multiplier]
    if (
        [group["lr"] for group in groups] != expected_lrs
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
        "trainable_encoder_parameters": expected["trainable_parameters"],
        "frozen_base_tensors": len(full["model_storage"]["frozen_tensors"]),
        "exact_complete_state_history_and_prediction_recovery": True,
    }


def compare_matched(bindings: Mapping[str, dict]) -> None:
    """Both arms share data, roster, folds, release, source, code and every other setting."""
    left, right = (json.loads(json.dumps(bindings[arm])) for arm in ARMS)
    for value in (left, right):
        value["config"]["encoder"].pop("suffix_adapter")
        value["config"].pop("encoder_lr")
    if left != right:
        raise ValueError("Matched arms differ beyond suffix policy and predeclared LR")


def subprocess_executor(args: argparse.Namespace, source: Path) -> Executor:
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(source),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="1",
        MKL_NUM_THREADS="1",
        OPENBLAS_NUM_THREADS="1",
        WORLD_SIZE="1",
        RANK="0",
        LOCAL_RANK="0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    deadline = time.monotonic() + args.remaining_seconds
    artifact = args.artifact_root
    base = [
        sys.executable,
        "-B",
        str(source / "scripts/newton/enforce_deterministic_runtime.py"),
        "--module",
        "rsna.competitive.train",
        "--device",
        args.device,
        "--fitting-supervision",
        str(args.overlay),
        "--studies",
        str(artifact / "initiative/releases/report-supervision-v1/studies.parquet"),
        "--artifact-root",
        str(artifact),
        "--folds",
        str(artifact / "initiative/experiments/competitive-v1/folds"),
        "--series",
        str(
            artifact
            / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv"
        ),
        "--cache",
        str(args.cache),
    ]

    def execute(root: Path, phase: str, config: Path, extra: list[str]) -> int:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Whole recovery deadline exhausted")
        with (
            (root / f"{phase}.stdout").open("x") as out,
            (root / f"{phase}.stderr").open("x") as err,
        ):
            return subprocess.run(
                [*base, "--config", str(config), *extra],
                env=env,
                cwd=source,
                stdout=out,
                stderr=err,
                timeout=remaining,
            ).returncode

    return execute


def verify_source(source: Path, manifest: Mapping[str, str]) -> dict[str, str]:
    package = {str(p.relative_to(source)) for p in (source / "rsna").rglob("*.py") if p.is_file()}
    if not package or package != {n for n in manifest if n.startswith("rsna/")}:
        raise ValueError("Source manifest must bind exactly every canonical rsna module")
    for name, value in manifest.items():
        if sha(source / name) != value:
            raise ValueError("Immutable numerical source differs from its manifest")
    return {n.removeprefix("rsna/"): h for n, h in manifest.items() if n.startswith("rsna/")}


def run(
    args: argparse.Namespace,
    executor: Executor | None = None,
    expected: Mapping[str, Mapping] = PINNED,
) -> dict:
    source = args.source.resolve()
    code = verify_source(source, json.loads(args.source_manifest.read_text()))
    recipe = load_recipe(source / RECIPE)
    spec = json.loads(args.encoder_spec.read_text())
    if args.device == "cuda" and torch.cuda.device_count() != 1:
        raise ValueError("Exactly one visible assigned GPU required for sequential arms")
    if shutil.disk_usage(args.output.parent).free < args.max_output_bytes + RESERVE:
        raise ValueError("Admitted output cap plus twenty GiB reserve unavailable")
    args.output.mkdir(mode=0o700)
    execute = executor or subprocess_executor(args, source)
    results, bindings = {}, {}
    for arm in ARMS:
        values = engineering_config(recipe[arm], spec)
        root = args.output / arm
        root.mkdir(mode=0o700)
        config = root / "config.json"
        with config.open("x") as handle:
            handle.write(json.dumps(values, sort_keys=True, indent=2) + "\n")

        def phase(name: str, extra: list[str], code_expected: int) -> None:
            used = sum(p.stat().st_size for p in args.output.rglob("*") if p.is_file())
            if used > args.max_output_bytes:
                raise ValueError("Admitted recovery output cap exceeded")
            print(json.dumps({"arm": arm, "phase": name}), flush=True)
            if execute(root, name, config, extra) != code_expected:
                raise ValueError("Canonical trainer failed; original phase logs retained")

        full, replay = root / "full", root / "replay"
        phase("full", ["--output", str(full)], 0)
        phase("stop", ["--output", str(replay), "--stop-after-updates", "1"], 75)
        partial = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
        first = json.loads((replay / "invocation.json").read_text())
        phase("resume", ["--output", str(replay), "--resume", str(replay / "checkpoint.pt")], 0)
        results[arm] = compare_recovery(root, partial, first, expected[arm])
        binding = partial["binding"]
        if binding["code_sha256"] != code or binding["device_type"] != args.device:
            raise ValueError("Trainer ran a different numerical source or device")
        bindings[arm] = binding
    compare_matched(bindings)
    result = {
        "status": "actual_matched_dense_suffix_full_stop1_resume_recovery_accepted",
        "arms": results,
        "source_manifest_sha256": sha(args.source_manifest),
        "recipe_sha256": {arm: sha(source / RECIPE / name) for arm, name in ARMS.items()},
        "device": args.device,
        "GPUs_used_concurrently": 1 if args.device == "cuda" else 0,
        "diagnostic_only": True,
        "clinical_accuracy_evidence": False,
        "competition_submission": False,
    }
    with (args.output / "accepted.json").open("x") as handle:
        handle.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source",
        "source-manifest",
        "encoder-spec",
        "artifact-root",
        "cache",
        "overlay",
        "output",
    ):
        value.add_argument("--" + name, type=Path, required=True)
    value.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    value.add_argument("--max-output-bytes", type=int, default=12 * 2**30)
    value.add_argument("--remaining-seconds", type=float, required=True)
    return value


if __name__ == "__main__":
    arguments = parser().parse_args()
    if not math.isfinite(arguments.remaining_seconds) or arguments.remaining_seconds <= 0:
        raise SystemExit("--remaining-seconds must be positive")
    print(json.dumps(run(arguments), sort_keys=True), flush=True)
