"""Immutable one-H100 canonical-trainer recovery for both matched public-input recipe arms.

Each arm runs the unchanged ``rsna.competitive.train`` under the strict deterministic
runtime in fresh full, stop-after-one-update and resume processes, then the existing
``compare_recovery`` proves complete decoded state and prediction-byte equality.
Diagnostic engineering only: no accuracy, production fitting or competition claim.
"""

import argparse
import gc
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from scripts.newton.probe_patch_witness_trainer_recovery import (
    compare_recovery,
    digest,
    numerical_updates,
)
from scripts.newton.probe_public_input_recipe_inputs import (
    ARM_FIELDS,
    ARMS,
    CHECKPOINT_CAP_BYTES,
    CONFIG_SHA256,
    CPU_RECEIPT_CAP_BYTES,
    ENGINEERING_STUDIES,
    MERGED_RECEIPT,
    NATIVE_CACHE,
    NATIVE_TENSOR_SHA256,
    NODE,
    PUBLIC_CACHE,
    STATUS,
    canonical,
    load_configs,
    sha,
    validate_funding,
    verify_source,
)

RECOVERY_STATUS = "public_input_recipe_one_H100_canonical_trainer_recovery_accepted"
PHASE_TIMEOUT_SECONDS = 3 * 3600
RNG_KEYS = {"torch", "cuda", "python", "numpy"}


def engineering_config(base: dict, arm: str) -> dict:
    """Shorten only the schedule of an admitted arm: 2 epochs, 16 studies, 4 updates."""
    if arm not in ARMS:
        raise ValueError("Unknown public-input recipe arm")
    from rsna.competitive.train import TrainConfig

    values = dict(
        base,
        epochs=2,
        diagnostic_study_limit=ENGINEERING_STUDIES,
        warmup_updates=2,
        submission_every_epochs=0,
        min_free_gib=3,
    )
    TrainConfig(**values).validate()
    if (
        values["precision"],
        values["amp_init_scale"],
        values["encode_chunk_size"],
        values["batch_size"],
        values["accumulation_steps"],
        values["encoder"]["adapter"]["rank"],
        values["freeze_encoder"],
        values["checkpoint_format"],
        (values["side"], values["slices"]),
        "public_input" in values,
    ) != (
        "fp16",
        128,
        8,
        1,
        8,
        32,
        True,
        "frozen_base_delta",
        (336, 10) if arm == "public" else (384, 16),
        arm == "public",
    ):
        raise ValueError("Explicit plain rank32 FP16 effective8 input-arm recipe required")
    return values


def validate_cpu_receipt(cpu: dict, manifest_sha: str, funding_sha: str, code: dict, configs):
    """Accept only the independent CPU admission of exactly this source/funding/config."""
    if (
        cpu.get("status") != STATUS
        or cpu.get("source_manifest_sha256") != manifest_sha
        or cpu.get("funding_sha256") != funding_sha
        or cpu.get("code_sha256") != code
        or cpu.get("config_sha256") != CONFIG_SHA256
        or cpu.get("configs") != configs
        or cpu.get("caches") != {"native": NATIVE_CACHE, "public": PUBLIC_CACHE}
        or cpu["initial_learned_state"]["native"] != cpu["initial_learned_state"]["public"]
        or cpu["GPU_or_optimizer_or_clinical_inference_executed"] is not False
        or cpu["merge_render_download_or_transfer_executed"] is not False
        or cpu["protected_labels_used_for_recipe_or_metrics"] is not False
        or cpu["original_large_base_proof_and_stats_reused"] is not True
        or cpu["public_cohort"]["members"] != cpu["full_fitting_studies"] + cpu["full_outer_studies"]
        or any(
            cpu["tensor_sha256"][f"native/{role}"][kind] != NATIVE_TENSOR_SHA256[kind][role]
            for kind in ("images", "other")
            for role in ("fitting", "development")
        )
        or any(
            cpu["tensor_sha256"][f"native/{role}"]["labels"]
            != cpu["tensor_sha256"][f"public/{role}"]["labels"]
            for role in ("fitting", "development")
        )
    ):
        raise ValueError("Independent public-input recipe CPU admission required")
    return cpu


def output_bytes(root: Path) -> int:
    return sum(path.stat().st_size for path in root.rglob("*") if path.is_file())


def check_cap(root: Path) -> None:
    if output_bytes(root) > CHECKPOINT_CAP_BYTES:
        raise ValueError("Funded recovery checkpoint/output cap exceeded")


def validate_state(whole: dict, values: dict, arm: str, cpu: dict, code: dict) -> None:
    """Bind the recovered checkpoint to the admitted source, rosters, base, scaler and RNG."""
    binding = whole["binding"]
    bound = binding["config"]
    expected = {key: value for key, value in values.items() if key != "encoder"}
    storage = whole["model_storage"]
    optimizer = whole["optimizer"]
    rng = whole["rng_by_rank"][0]
    if (
        binding["training_uids_sha256"] != cpu["training_16_sha256"]
        or binding["outer_uids_sha256"] != cpu["outer_16_sha256"]
        or binding["code_sha256"] != code
        or binding["device_type"] != "cuda"
        or binding["world_size"] != 1
        or {key: bound.get(key) for key in expected} != expected
        or bound["encoder"] != cpu["encoder_provenance"][arm]
        or storage["encoder_provenance"] != cpu["encoder_provenance"][arm]
        or canonical(storage["frozen_tensors"])
        != cpu["initial_learned_state"][arm]["frozen_base_sha256"]
        or binding["experimental_fitting_supervision"] != cpu["supervision"]
        or (
            binding.get("public_input_cache") != cpu["public_cohort"]
            or binding.get("private_research_only") is not True
            if arm == "public"
            else "public_input_cache" in binding
        )
    ):
        raise ValueError("Recovered source/recipe/roster/frozen-base binding differs")
    scaler = whole.get("amp_scaler", {})
    if (
        scaler.get("_growth_tracker") != 4
        or not math.isfinite(scaler.get("scale", math.nan))
        or scaler["scale"] <= 0
    ):
        raise ValueError("Recovered FP16 scaler differs from four finite updates")
    if (
        len(optimizer["param_groups"]) < 1
        or any(
            not math.isfinite(group["lr"]) or group["lr"] <= 0
            for group in optimizer["param_groups"]
        )
        or any(float(state["step"]) != 4 for state in optimizer["state"].values())
    ):
        raise ValueError("Recovered AdamW learning rate/counters differ")
    if set(rng) != RNG_KEYS or rng["cuda"] is None:
        raise ValueError("Recovered checkpoint lacks complete CPU/CUDA/Python/NumPy RNG")


def validate_predictions(path: Path, outer_sha: str) -> str:
    from rsna.schema import TARGETS

    payload = path.read_bytes()
    prediction = pd.read_csv(path)
    scores = prediction[list(TARGETS)].to_numpy() if set(TARGETS) <= set(prediction) else None
    if (
        list(prediction.columns) != ["study_uid", "fold", "seed", *TARGETS]
        or len(prediction) != ENGINEERING_STUDIES
        or scores is None
        or scores.shape != (ENGINEERING_STUDIES, 12)
        or digest(sorted(prediction.study_uid.tolist())) != outer_sha
        or not np.isfinite(scores).all()
        or (scores < 0).any()
        or (scores > 1).any()
    ):
        raise ValueError("Exact finite development16x12 predictions required")
    return hashlib.sha256(payload).hexdigest()


def check_placement() -> None:
    if (
        os.uname().nodename != NODE
        or not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "2"
        or torch.cuda.device_count() != 1
        or "H100" not in torch.cuda.get_device_name(0)
    ):
        raise ValueError("Original owning-node one-H100/two-CPU allocation required")


def trainer_command(code: Path, config: Path, cpu: dict, arm: str) -> list[str]:
    inputs = cpu["trainer_inputs"]
    return [
        sys.executable,
        "-B",
        str(code / "scripts/newton/enforce_deterministic_runtime.py"),
        "--module",
        "rsna.competitive.train",
        "--config",
        str(config),
        "--device",
        "cuda",
        "--fitting-supervision",
        inputs["overlay"],
        "--studies",
        inputs["studies"],
        "--artifact-root",
        inputs["artifact-root"],
        "--folds",
        inputs["folds"],
        "--series",
        inputs["series"],
        "--cache",
        cpu["caches"][arm],
    ]


def recover_arm(code: Path, output: Path, root: Path, values: dict, cpu: dict, arm: str, deadline):
    root.mkdir(mode=0o700)
    config = root / "config.json"
    config.write_text(json.dumps(values, sort_keys=True, indent=2) + "\n")
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(code),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        WORLD_SIZE="1",
        RANK="0",
        LOCAL_RANK="0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )
    command = trainer_command(code, config, cpu, arm)
    full, replay = root / "full", root / "replay"

    def execute(extra: list[str], phase: str, expected: int) -> None:
        check_cap(output)
        remaining = min(PHASE_TIMEOUT_SECONDS, deadline - time.monotonic())
        if remaining <= 0:
            raise TimeoutError("Funded recovery wall time exhausted")
        print(json.dumps({"arm": arm, "phase": phase}), flush=True)
        with (
            (root / f"{phase}.stdout").open("x") as out,
            (root / f"{phase}.stderr").open("x") as err,
        ):
            result = subprocess.run(
                command + extra, env=env, cwd=code, stdout=out, stderr=err, timeout=remaining
            )
        check_cap(output)
        if result.returncode != expected:
            raise ValueError("Canonical trainer phase failed; original logs retained")

    execute(["--output", str(full)], "full", 0)
    execute(["--output", str(replay), "--stop-after-updates", "1"], "stop", 75)
    partial = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
    first = json.loads((replay / "invocation.json").read_text())
    execute(["--output", str(replay), "--resume", str(replay / "checkpoint.pt")], "resume", 0)
    whole = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=True)
    resumed = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
    history = json.loads((full / "invocation.json").read_text())
    last = json.loads((replay / "invocation.json").read_text())
    state = compare_recovery(
        whole,
        resumed,
        partial,
        history,
        first,
        last,
        (full / "outer_predictions.csv").read_bytes(),
        (replay / "outer_predictions.csv").read_bytes(),
    )
    return whole, state, numerical_updates(history), full


def run(root: Path, wall_seconds: float) -> dict:
    deadline = time.monotonic() + wall_seconds
    check_placement()
    torch.set_num_threads(2)
    code = root / "code"
    manifest_path, funding_path, cpu_path = (
        root / "source.json",
        root / "funding.json",
        root / "CPU-accepted.json",
    )
    code_sha256 = verify_source(code, json.loads(manifest_path.read_text()))
    funding = validate_funding(json.loads(funding_path.read_text()))
    if cpu_path.stat().st_size > CPU_RECEIPT_CAP_BYTES:
        raise ValueError("CPU receipt exceeds its explicit cap")
    configs = load_configs(code)
    cpu = validate_cpu_receipt(
        json.loads(cpu_path.read_text()), sha(manifest_path), sha(funding_path), code_sha256, configs
    )
    if sha(MERGED_RECEIPT) != funding["merged_cache_receipt_sha256"]:
        raise ValueError("Root complete-cache verification receipt changed")
    output = root / "recovery"
    if shutil.disk_usage(root).free < CHECKPOINT_CAP_BYTES + funding["reserve_bytes"]:
        raise ValueError("Funded recovery space plus reserve is unavailable")
    output.mkdir(mode=0o700)
    engineering = {arm: engineering_config(configs[arm], arm) for arm in ARMS}
    differing = {
        key
        for key in set(engineering["native"]) | set(engineering["public"])
        if engineering["native"].get(key) != engineering["public"].get(key)
    }
    if differing != ARM_FIELDS:
        raise ValueError("Engineering arms must differ only in public input/side/slices")
    results = {}
    for arm in ARMS:
        whole, state, updates, full = recover_arm(
            code, output, output / arm, engineering[arm], cpu, arm, deadline
        )
        validate_state(whole, engineering[arm], arm, cpu, code_sha256)
        results[arm] = {
            "complete_state_sha256": state,
            "checkpoint_sha256": sha(full / "checkpoint.pt"),
            "prediction_sha256": validate_predictions(
                full / "outer_predictions.csv", cpu["outer_16_sha256"]
            ),
            "finite_updates": updates,
            "amp_scaler": whole["amp_scaler"],
            "exact_complete_state_scaler_RNG_and_prediction_recovery": True,
        }
        del whole
        gc.collect()
        torch.cuda.empty_cache()
    value = {
        "status": RECOVERY_STATUS,
        "arms": results,
        "source_git": cpu["source_git"],
        "CPU_receipt_sha256": sha(cpu_path),
        "funding_sha256": sha(funding_path),
        "recovery_output_bytes": output_bytes(output),
        "GPUs_used": 1,
        "diagnostic_only": True,
        "clinical_accuracy_evidence": False,
        "production_fitting": False,
        "competition_submission": False,
    }
    with (root / "recovery-accepted.json").open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"status": RECOVERY_STATUS}), flush=True)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--wall-seconds", type=float, required=True)
    arguments = parser.parse_args()
    run(arguments.root, arguments.wall_seconds)
