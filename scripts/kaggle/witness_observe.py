"""Read original T4 checkpoints after a terminal observer error; never train."""

from __future__ import annotations

import argparse
import gc
import json
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

if __package__:
    from .witness_recovery import validate_predictions
else:
    from witness_recovery import validate_predictions

from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.train import TrainConfig
from scripts.newton.probe_patch_witness_trainer_recovery import decoded, digest, numerical_updates


def update_log(path):
    rows = []
    for line in path.read_text().splitlines():
        if line.startswith("{"):
            record = json.loads(line)
            if "optimizer_steps" in record:
                rows.append(record)
    return numerical_updates({"updates": rows})


def validate_bound_config(bound, original, encoder_sha):
    """Saved config binds derived encoder provenance, not input mount paths."""
    if digest(bound["encoder"]) != encoder_sha:
        raise ValueError("Original complete encoder provenance differs")
    expected = asdict(
        TrainConfig(**dict(original, precision="fp32", encode_chunk_size=1, min_free_gib=3))
    )
    expected.pop("amp_init_scale", None)
    for name in ("relation", "fitting_supervision", "neighbor_channels", "witness"):
        if expected[name] is None:
            expected.pop(name)
    expected["encoder"] = bound["encoder"]
    if bound != expected:
        raise ValueError("Original numerical recipe differs")


def observe(control, outputs, cpu_receipt, receipt_sha, destination):
    started = time.monotonic()
    torch.set_num_threads(1)
    if file_sha256(cpu_receipt) != receipt_sha:
        raise ValueError("Original CPU input receipt differs")
    accepted = json.loads(cpu_receipt.read_text())
    if (
        accepted["source_git"] != "d3cf58242455bf96e46b0c2071e8e9fe16cf4aaf"
        or accepted["input_tensors_equal_to_original"] is not True
        or accepted["labels_masks_equal_to_original"] is not True
        or accepted["initial_learned_state"]["control"]
        != accepted["initial_learned_state"]["candidate"]
    ):
        raise ValueError("Actual original MRI and initialization admission required")
    proof = json.loads((control / "original-input-proof.json").read_text())
    for name, sha in proof["source_sha256"].items():
        if file_sha256(control / "code" / name) != sha:
            raise ValueError("Original scientific source differs")
    results = {}
    for role in ("control", "candidate"):
        root = outputs / role
        full = torch.load(root / "full/checkpoint.pt", map_location="cpu", weights_only=True)
        replay = torch.load(root / "replay/checkpoint.pt", map_location="cpu", weights_only=True)
        for state in (full, replay):
            if (
                (state.get("epoch"), state.get("next_update"), state.get("step")) != (2, 0, 4)
                or state.get("schema_version") != 2
                or state.get("optimizer_boundary") is not True
                or not state.get("model")
                or not state.get("optimizer", {}).get("state")
                or len(state.get("rng_by_rank", [])) != 1
                or state.get("model_storage", {}).get("format") != "frozen_base_delta_v1"
            ):
                raise ValueError("Complete original four-update terminal state required")
        original = proof["expected_config"][role]
        binding = full["binding"]
        validate_bound_config(
            binding["config"],
            original,
            accepted["initial_learned_state"][role]["encoder_provenance_sha256"],
        )
        if (
            binding["training_uids_sha256"] != proof["training_16_sha256"]
            or binding["outer_uids_sha256"] != proof["outer_16_sha256"]
            or binding["release"] != accepted["release_binding"]
            or binding["code_sha256"]
            != {
                name.removeprefix("rsna/"): sha
                for name, sha in proof["source_sha256"].items()
                if name.startswith("rsna/")
            }
        ):
            raise ValueError("Original input/roster/release bindings differ")
        state = decoded(full)
        if state != decoded(replay):
            raise ValueError("Original learned/base/AdamW/LR/counters/all-RNG states differ")
        updates = numerical_updates(json.loads((root / "full/invocation.json").read_text()))
        interrupted = update_log(root / "stop.stdout")
        resumed = update_log(root / "resume.stdout")
        if (
            len(updates) != 4
            or len(interrupted) != 1
            or interrupted + resumed != updates
            or [(r["epoch"], r["update"], r["optimizer_steps"]) for r in updates]
            != [(0, 1, 1), (0, 2, 2), (1, 1, 3), (1, 2, 4)]
        ):
            raise ValueError("Original uninterrupted/stop/resume numerical trajectories differ")
        csv = (root / "full/outer_predictions.csv").read_bytes()
        if csv != (root / "replay/outer_predictions.csv").read_bytes():
            raise ValueError("Original prediction bytes differ")
        validate_predictions(
            pd.read_csv(root / "full/outer_predictions.csv"),
            proof["outer_16_sha256"],
            original["fold"],
            original["seed"],
        )
        results[role] = {
            "complete_decoded_state_sha256": digest(state),
            "checkpoint_sha256": file_sha256(root / "full/checkpoint.pt"),
            "predictions_sha256": file_sha256(root / "full/outer_predictions.csv"),
            "optimizer_updates": 4,
            "updates": updates,
            "complete_terminal_state_and_trajectory_and_CSV_equal": True,
            "interrupted_checkpoint_bytes_retained": False,
            "interruption_evidence": "original one-update stop log plus canonical resumed updates2–4; intermediate checkpoint was overwritten by resume",
        }
        del full, replay, state
        gc.collect()
    record = {
        "status": "actual_original_T4_terminal_states_trajectories_and_CSV_verified_on_CPU",
        "source_git": accepted["source_git"],
        "CPU_receipt_sha256": receipt_sha,
        "roles": results,
        "original_GPU_notebook_terminal": "ERROR",
        "observer_schema_error_corrected": True,
        "GPU_or_optimizer_executed_in_this_observer": False,
        "clinical_accuracy_evidence": False,
        "competition_submission": False,
        "elapsed_seconds": time.monotonic() - started,
    }
    with destination.open("x") as handle:
        json.dump(record, handle, sort_keys=True, indent=2)
        handle.write("\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("control", "outputs", "cpu_receipt", "destination"):
        p.add_argument("--" + name.replace("_", "-"), type=Path, required=True)
    p.add_argument("--receipt-sha", required=True)
    a = p.parse_args()
    observe(a.control, a.outputs, a.cpu_receipt, a.receipt_sha, a.destination)
