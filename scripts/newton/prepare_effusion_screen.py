"""Bind the fixed paired screen to actual original MRI recovery evidence.

This is a preparation gate, never a GPU dispatch, funding decision or accuracy
result. It cannot replace the original remote source, dataset or checkpoints.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import asdict
from pathlib import Path

from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.train import TrainConfig

SCIENTIFIC_GIT = "665c0d5"
WEIGHTS_SHA = "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
ENCODER_CONFIG_SHA = "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5"
OVERLAY_SHA = "936e3e2af058726fe3572b7e4ad1680c44f2fd82ad6e9840e2607a1b090ff018"


def checked_sha(value):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError("Original whole-byte SHA required")
    return value


def accept(cpu, gpu, source, source_sha, cpu_sha, configs):
    """Fail closed on mismatched evidence and undeclared numerical changes."""
    checked_sha(source_sha)
    checked_sha(cpu_sha)
    if (
        cpu.get("status") != "actual_native384_effusion_inputs_passed"
        or cpu.get("job_id") != "869076"
        or cpu.get("source_manifest_sha256") != source_sha
        or cpu.get("changed_engineering_cells") != 1
        or cpu.get("observed_engineering_cells") != {"control": 86, "candidate": 87}
        or (cpu.get("full_training_studies"), cpu.get("full_outer_studies")) != (2247, 1156)
        or cpu.get("clinical_accuracy_evidence") is not False
        or cpu.get("original_assets_verified") is not True
        or not source.get("source_git", "").startswith(SCIENTIFIC_GIT)
    ):
        raise ValueError("Exact original complete-roster MRI CPU acceptance required")
    if (
        gpu.get("status") != "actual_native384_effusion_trainer_recovery_passed"
        or gpu.get("job_id") != "869077"
        or gpu.get("node", "").split(".")[0] != "evc31"
        or gpu.get("input_acceptance_sha256") != cpu_sha
        or gpu.get("clinical_accuracy_evidence") is not False
        or gpu.get("diagnostic_only") is not True
        or set(gpu.get("results", {})) != {"control", "candidate"}
    ):
        raise ValueError("Actual original H100 recovery required, never a pending job")
    initial = gpu.get("matched_initial_model_and_RNG", {})
    for key in ("complete_model_sha256", "all_RNG_sha256"):
        checked_sha(initial.get(key))
    encoder = initial.get("encoder_provenance", {})
    if (
        encoder.get("weights_sha256") != WEIGHTS_SHA
        or encoder.get("config_sha256") != ENCODER_CONFIG_SHA
        or encoder.get("architecture") != "dinov3_l16"
        or encoder.get("pretrained") is not True
    ):
        raise ValueError("Original pinned official base and matched initialization required")
    for arm, enabled in (("control", False), ("candidate", True)):
        result = gpu["results"][arm]
        for key in ("decoded_checkpoint_sha256", "checkpoint_sha256", "predictions_sha256"):
            checked_sha(result.get(key))
        if result.get("exact_complete_state_and_prediction_recovery") is not True:
            raise ValueError("Complete original numerical recovery must actually pass")
        updates = result.get("updates", [])
        if [(r.get("epoch"), r.get("update"), r.get("optimizer_steps")) for r in updates] != [
            (0, 1, 1),
            (0, 2, 2),
            (1, 1, 3),
            (1, 2, 4),
        ]:
            raise ValueError("Actual original four-update trajectories required")
        if any(
            type(r.get(k)) not in (int, float) or not math.isfinite(r[k]) or r[k] < 0
            for r in updates
            for k in ("loss", "gradient_norm", "observed_cells")
        ):
            raise ValueError("Actual finite original numerical updates required")
        config = TrainConfig(**configs[arm])
        config.validate()
        pilot = TrainConfig(**cpu["configs"][arm])
        pilot.validate()
        bound = asdict(config)
        expected = asdict(pilot)
        expected.update(
            epochs=3, diagnostic_study_limit=None, warmup_updates=141, submission_every_epochs=1
        )
        if bound != expected or config.fitting_supervision != {
            "kind": "effusion_review_v1",
            "sha256": OVERLAY_SHA,
            "enabled": enabled,
        }:
            raise ValueError("Only predeclared pilot-to-screen schedule and scope may change")
        receipt = cpu["supervision_receipts"][arm]
        if (
            receipt.get("overlay_sha256") != OVERLAY_SHA
            or receipt.get("applied_cells") != (31 if enabled else 0)
            or receipt.get("evaluation_masks_changed") is not False
        ):
            raise ValueError("Exact fitting-only arm and unchanged evaluation masks required")
    a, b = (asdict(TrainConfig(**configs[k])) for k in ("control", "candidate"))
    b["fitting_supervision"] = a["fitting_supervision"]
    if a != b:
        raise ValueError("Only enabled supervision may differ between screen arms")
    return {
        "status": "original_MRI_recovery_admitted_for_fixed_screen_preparation",
        "scientific_source_git": source["source_git"],
        "source_manifest_sha256": source_sha,
        "CPU_receipt_sha256": cpu_sha,
        "actual_engineering_GPU_job": "869077",
        "training_studies": 2247,
        "outer_studies": 1156,
        "optimizer_updates_per_arm": 843,
        "fixed_final_epoch": 3,
        "primary_outcome": "Effusion_outer_AUC",
        "secondary_outcomes": "all12_outer_AUC_and_macro",
        "paired_uncertainty_unit": "whole_leakage_component",
        "best_epoch_selection": False,
        "protected_evaluation_used": False,
        "clinical_accuracy_evidence": False,
        "production_funding_or_dispatch": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("cpu", "gpu", "source", "control-config", "candidate-config", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    paths = (args.cpu, args.gpu, args.source, args.control_config, args.candidate_config)
    if any(p.is_symlink() or not p.is_file() or p.stat().st_size > 2**20 for p in paths):
        raise ValueError("Original bounded regular receipts/configs required")
    cpu, gpu, source, control, candidate = (json.loads(p.read_text()) for p in paths)
    result = accept(
        cpu,
        gpu,
        source,
        file_sha256(args.source),
        file_sha256(args.cpu),
        {"control": control, "candidate": candidate},
    )
    result["GPU_receipt_sha256"] = file_sha256(args.gpu)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")


if __name__ == "__main__":
    main()
