"""Bounded actual-GPU recovery for final public-source fitting; no deployment."""

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from rsna.competitive.train import TrainConfig
from scripts.newton.probe_patch_witness_trainer_recovery import compare_recovery, numerical_updates


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_placement(manifest, actual_node):
    expected = manifest.get("owning_node", "evc31")
    if expected not in {"evc31", "evc26", "evc41"} or actual_node != expected:
        raise ValueError("Exact independently admitted recovery node required")


def engineering_config(values):
    """The final recipe is immutable; only the bounded recovery budget changes."""
    TrainConfig(**values).validate()
    required = {
        "fit_scope": "all_development",
        "epochs": 10,
        "warmup_updates": 426,
        "diagnostic_study_limit": None,
        "seed": 20260922,
        "side": 384,
        "slices": 16,
        "max_series": 6,
        "batch_size": 1,
        "accumulation_steps": 8,
        "precision": "fp16",
        "amp_init_scale": 128,
        "encode_chunk_size": 8,
        "pooling": "attention",
        "slice_context": "none",
        "slice_sampling": "uniform",
        "series_sampling": "selected",
        "augment": False,
        "freeze_encoder": True,
        "checkpoint_format": "frozen_base_delta",
    }
    if any(values.get(k) != v for k, v in required.items()):
        raise ValueError("Exact CPU-admitted final public-source recipe required")
    policy = values.get("fitting_supervision", {})
    if (
        policy.get("kind") != "label_source_final_v1"
        or policy.get("enabled") is not True
        or values.get("encoder", {}).get("architecture") != "dinov3_l16"
        or values["encoder"].get("adapter", {}).get("rank") != 32
        or values["encoder"].get("weights_sha256")
        != "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
        or any(
            values.get(k) is not None
            for k in (
                "relation",
                "witness",
                "plane_memory",
                "neighbor_channels",
                "patch_summary",
                "cache_contrast",
            )
        )
    ):
        raise ValueError("Isolated original-base public-source final fit required")
    result = dict(
        values, epochs=2, diagnostic_study_limit=16, warmup_updates=2, submission_every_epochs=0
    )
    TrainConfig(**result).validate()
    return result


def validate_final_checkpoint(full, resumed, partial, history, first, last):
    for checkpoint in (full, resumed):
        config = checkpoint.get("binding", {}).get("config", {})
        if (
            config.get("fit_scope") != "all_development"
            or config.get("fitting_supervision", {}).get("kind") != "label_source_final_v1"
            or checkpoint.get("amp_scaler", {}).get("_growth_tracker") != 4
        ):
            raise ValueError("Final-fit scope/source/scaler recovery differs")
    # No prediction file or held-out metric exists in this fitting scope. None
    # is intentional, not an invented empty prediction receipt.
    return compare_recovery(full, resumed, partial, history, first, last, None, None)


def run(root):
    manifest = json.loads((root / "source.json").read_text())
    accepted = json.loads((root / "input-accepted.json").read_text())
    funding = json.loads((root / "funding.json").read_text())
    validate_placement(manifest, os.uname().nodename)
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "2"
        or accepted["status"] != "actual_final_public_source_CPU_roster_and_16MRI_inputs_verified"
        or (
            accepted["training_studies"],
            accepted["outer_studies"],
            accepted["masked_out_source_studies"],
        )
        != (3403, 0, 0)
        or funding.get("new_recovery_bytes") != 2**30
        or funding.get("new_final_fit_and_publication_bytes") != 6 * 2**30
        or funding.get("new_shared_bytes_files") != 0
        or sha(root / "input-accepted.json") != manifest["input_receipt_sha256"]
    ):
        raise ValueError("Exact independently funded CPU admission and placement required")
    for name, digest in manifest["files"].items():
        path = root / name
        if (
            path.is_symlink()
            or not path.resolve().is_relative_to(root.resolve())
            or sha(path) != digest
        ):
            raise ValueError("Original frozen recovery source changed")
    if shutil.disk_usage(root).free < funding["required_free_bytes"]:
        raise ValueError("Existing liabilities plus recovery/final fit/reserve must remain funded")
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or "H100" not in torch.cuda.get_device_name(0)
    ):
        raise ValueError("Allocated original one-H100 CUDA context required")
    torch.ones(1, device="cuda").sum().item()
    values = json.loads((root / "final-config.json").read_text())
    if sha(root / "final-config.json") != accepted["config_sha256"]:
        raise ValueError("Original CPU final configuration changed")
    config = root / "recovery-config.json"
    save(config, engineering_config(values))
    full, replay = root / "full", root / "replay"
    inp = json.loads((root / "trainer-inputs.json").read_text())
    command = [
        sys.executable,
        "-B",
        str(root / "code/scripts/newton/enforce_deterministic_runtime.py"),
        "--module",
        "rsna.competitive.train",
        "--config",
        str(config),
        "--device",
        "cuda",
        "--fitting-supervision",
        str(root / "overlay.json"),
    ]
    for key, value in inp.items():
        command.extend(("--" + key, value))
    records = {}
    for phase, extra, expected in (
        ("full", ["--output", str(full)], 0),
        ("stop", ["--output", str(replay), "--stop-after-updates", "1"], 75),
        ("resume", ["--output", str(replay), "--resume", str(replay / "checkpoint.pt")], 0),
    ):
        with (
            (root / (phase + ".stdout")).open("x") as out,
            (root / (phase + ".stderr")).open("x") as err,
        ):
            result = subprocess.run(
                command + extra, cwd=root / "code", stdout=out, stderr=err, timeout=1200
            )
        if result.returncode != expected:
            raise ValueError("Original GPU recovery failed; preserve without retry")
        output = full if phase == "full" else replay
        records[phase] = json.loads((output / "invocation.json").read_text())
        if phase == "stop":
            partial = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
    whole = torch.load(full / "checkpoint.pt", map_location="cpu", weights_only=True)
    resumed = torch.load(replay / "checkpoint.pt", map_location="cpu", weights_only=True)
    for output in (full, replay):
        result = json.loads((output / "result.json").read_text())
        if (
            result["metrics"] is not None
            or result["outer_studies"] != 0
            or (output / "outer_predictions.csv").exists()
        ):
            raise ValueError("Final-fit recovery cannot claim held-out predictions")
    state = validate_final_checkpoint(
        whole, resumed, partial, records["full"], records["stop"], records["resume"]
    )
    bound = whole["binding"]
    expected_modules = {
        k.removeprefix("code/rsna/"): v
        for k, v in accepted["source_modules"].items()
        if k.startswith("code/rsna/")
    }
    if (
        bound["code_sha256"] != expected_modules
        or bound["config"]["fitting_supervision"] != values["fitting_supervision"]
    ):
        raise ValueError("Actual learned source or label policy differs from CPU acceptance")
    receipt = {
        "status": "actual_final_public_source_H100_complete_state_recovery_accepted",
        "job_id": os.environ["SLURM_JOB_ID"],
        "scientific_source_git": accepted["source_git"],
        "CPU_receipt_sha256": sha(root / "input-accepted.json"),
        "complete_state_sha256": state,
        "checkpoint_sha256": sha(full / "checkpoint.pt"),
        "finite_updates": numerical_updates(records["full"]),
        "engineering_training_studies": 16,
        "outer_studies": 0,
        "predictions_available": False,
        "clinical_or_Kaggle_accuracy": False,
        "production_dispatched": False,
    }
    save(root / "recovery-verified.json", receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
