"""Own CPU initialization and actual canonical GPU recovery for the R32 slot recipe.

Reuses the completed original input admission without rerendering or transferring
MRI. It does not reuse a different classifier's optimizer/recovery acceptance.
"""

import argparse
import gc
import json
import shutil
import time
from pathlib import Path

import torch

from scripts.newton.probe_public_input_recipe_inputs import (
    STATUS,
    initialize,
    sha,
    verify_source,
)
from scripts.newton.probe_public_input_recipe_recovery import (
    check_placement,
    output_bytes,
    recover_arm,
    validate_predictions,
    validate_state,
)

RECIPE = "initiative/experiments/competitive-v1/source-matched-slot-v1/r32-config.json"
RESERVE = 20 * 2**30
CAP = 6 * 2**30
CPU_STATUS = "source_slot_r32_original_inputs_and_own_initialization_CPU_accepted"


def recipe(code: Path, original: dict) -> dict:
    from rsna.competitive.source_slot import AUGMENT, STRATEGY_FIXED
    from rsna.competitive.train import TrainConfig

    values = json.loads((code / RECIPE).read_text())
    plain = original["configs"]["public"]
    if {k for k in values.keys() | plain.keys() if values.get(k) != plain.get(k)} != {
        "classifier_strategy"
    } or values["classifier_strategy"] != dict(STRATEGY_FIXED, augmentation=AUGMENT):
        raise ValueError("Only the declared source-slot classifier recipe may change")
    TrainConfig(**values).validate()
    return values


def prerequisites(root: Path):
    source = json.loads((root / "source.json").read_text())
    code = root / "code"
    code_hashes = verify_source(code, source)
    if RECIPE not in source["files"] or (
        "scripts/newton/probe_source_slot_recovery.py" not in source["files"]
    ):
        raise ValueError("Own wrapper and recipe source must be immutable")
    funding = json.loads((root / "funding.json").read_text())
    original_path = Path(funding["original_CPU_receipt"])
    if sha(original_path) != funding["original_CPU_sha256"]:
        raise ValueError("Original completed input receipt changed")
    original = json.loads(original_path.read_text())
    if (
        original["status"] != STATUS
        or original["code_sha256"] != code_hashes
        or (original["full_fitting_studies"], original["full_outer_studies"]) != (2247, 1156)
        or funding["checkpoint_cap_bytes"] != CAP
        or funding["reserve_bytes"] != RESERVE
        or funding["required_free_bytes"] < CAP + RESERVE
        or shutil.disk_usage(root).free < funding["required_free_bytes"]
    ):
        raise ValueError("Original input scope or fresh funded reserve unavailable")
    return code, code_hashes, original, recipe(code, original)


def cpu_admission(root: Path) -> dict:
    import os

    from rsna.competitive.backbones import create_encoder
    from rsna.competitive.source_slot import SourceSlotStudyNet

    if (
        os.uname().nodename != "evc31"
        or not os.environ.get("SLURM_JOB_ID")
        or torch.cuda.device_count()
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
    ):
        raise ValueError("Genuine one-CPU zero-GPU admission required")
    torch.set_num_threads(1)
    code, hashes, original, values = prerequisites(root)

    def factory(encoder, dim, **kwargs):
        return SourceSlotStudyNet(
            encoder,
            dim,
            values["classifier_strategy"],
            encode_chunk_size=values["encode_chunk_size"],
            input_channels=3,
        )

    states = []
    for _ in range(2):
        model, provenance, state = initialize(values, "public", create_encoder, factory)
        if provenance != original["encoder_provenance"]["public"] or (
            state["frozen_base_sha256"]
            != original["initial_learned_state"]["public"]["frozen_base_sha256"]
        ):
            raise ValueError("Official encoder provenance or frozen base changed")
        states.append(state)
        del model
        gc.collect()
    if states[0] != states[1]:
        raise ValueError("Own slot model initialization/RNG is not reproducible")
    result = {
        "status": CPU_STATUS,
        "source_git": json.loads((root / "source.json").read_text())["source_git"],
        "code_sha256": hashes,
        "config": values,
        "config_sha256": sha(code / RECIPE),
        "original_CPU_sha256": sha(
            Path(json.loads((root / "funding.json").read_text())["original_CPU_receipt"])
        ),
        "funding_sha256": sha(root / "funding.json"),
        "initialization": states[0],
        "source_inputs_reused_without_pixel_transfer": True,
        "GPU_or_optimizer_executed": False,
    }
    with (root / "CPU-accepted.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    return result


def gpu_recovery(root: Path, wall_seconds: float) -> dict:
    if not 0 < wall_seconds <= 6900:
        raise ValueError("Positive bounded GPU recovery wall time required")
    check_placement()
    code, hashes, original, values = prerequisites(root)
    own = json.loads((root / "CPU-accepted.json").read_text())
    if (
        own["status"] != CPU_STATUS
        or own["code_sha256"] != hashes
        or own["config"] != values
        or own["config_sha256"] != sha(code / RECIPE)
        or own["funding_sha256"] != sha(root / "funding.json")
    ):
        raise ValueError("This exact slot model's CPU initialization admission required")
    engineering = dict(
        values,
        epochs=2,
        diagnostic_study_limit=16,
        warmup_updates=2,
        submission_every_epochs=0,
        min_free_gib=20,
    )
    output = root / "recovery"
    output.mkdir(mode=0o700)
    whole, state, updates, full = recover_arm(
        code,
        output,
        output / "public",
        engineering,
        original,
        "public",
        time.monotonic() + wall_seconds,
    )
    validate_state(whole, engineering, "public", original, hashes)
    if len(updates) != 4 or output_bytes(output) > CAP:
        raise ValueError("Four actual finite updates within funded checkpoint cap required")
    result = {
        "status": "source_slot_r32_actual_canonical_GPU_recovery_accepted",
        "source_git": own["source_git"],
        "CPU_receipt_sha256": sha(root / "CPU-accepted.json"),
        "complete_state_sha256": state,
        "checkpoint_sha256": sha(full / "checkpoint.pt"),
        "prediction_sha256": validate_predictions(
            full / "outer_predictions.csv", original["outer_16_sha256"]
        ),
        "finite_updates": updates,
        "recovery_output_bytes": output_bytes(output),
        "production_fitting": False,
        "clinical_accuracy_evidence": False,
    }
    with (root / "recovery-accepted.json").open("x") as stream:
        json.dump(result, stream, indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--phase", choices=("cpu", "gpu"), required=True)
    parser.add_argument("--wall-seconds", type=float, default=6900)
    args = parser.parse_args()
    receipt = (
        cpu_admission(args.root)
        if args.phase == "cpu"
        else gpu_recovery(args.root, args.wall_seconds)
    )
    print(json.dumps({"status": receipt["status"]}), flush=True)
