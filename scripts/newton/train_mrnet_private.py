"""Full Stanford research fitting and a recovery exercise of that exact engine.

No raw Stanford material or learned weights are published. The full-source lane
remains prepared until its own actual GPU recovery, source rights and resource
acceptance have been reviewed; the earlier pilot is not silently substituted.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from rsna.competitive.adapters import configure_encoder_training
from rsna.competitive.backbones import create_encoder
from rsna.competitive.checkpoint import atomic_json, file_sha256
from rsna.competitive.mrnet import MRNET_TARGETS, MRNetAdapterNet, MRNetStudyDataset
from rsna.competitive.mrnet_training import MRNetRecipe, fit_source, predict_source
from scripts.newton.probe_mrnet_adapter_recovery import same


def engine_hash() -> str:
    return file_sha256(Path(__file__).parents[2] / "rsna/competitive/mrnet_training.py")


def read_binding(args):
    config = json.loads(args.config.read_text())
    input_receipt = json.loads(args.input_receipt.read_text())
    if (
        input_receipt["status"] != "actual_Stanford_three_plane_exam_tensors_verified"
        or input_receipt["source_train_exams"] != 1129
        or input_receipt["validation_training_permitted"] is not False
        or config.get("private_noncommercial_research_only") is not True
        or config.get("external_weight_publication") is not False
        or config.get("encoder", {}).get("architecture") != "dinov3_l16"
    ):
        raise ValueError(
            "Accepted actual source inputs and private official-encoder research required"
        )
    binding = {
        "source_input_receipt_sha256": file_sha256(args.input_receipt),
        "acquisition_receipt_sha256": file_sha256(args.acquisition_receipt),
        "config_sha256": file_sha256(args.config),
        "engine_sha256": engine_hash(),
        "entrypoint_sha256": file_sha256(Path(__file__)),
        "source_git": config["source_git"],
        "private_research_only": True,
        "external_weight_publication": False,
        "source_labels_never_mapped_to_RSNA_targets": True,
    }
    return config, binding


def worker(args):
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_capability() != (9, 0)
    ):
        raise ValueError("One allocated H100 required")
    if not os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_CPUS_PER_TASK") != "2":
        raise ValueError("Actual one-GPU two-CPU Slurm allocation required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(20261001)
    np.random.seed(20261001)
    torch.manual_seed(20261001)
    torch.cuda.manual_seed_all(20261001)
    config, binding = read_binding(args)
    train = MRNetStudyDataset(args.root, args.acquisition_receipt, "train")
    valid = MRNetStudyDataset(args.root, args.acquisition_receipt, "valid")
    if len(train) != 1129 or len(valid) != 119 or set(train.uids) & set(valid.uids):
        raise ValueError("Exact original supervised source memberships required")
    recipe = MRNetRecipe()
    if args.mode == "recovery-worker":
        train, valid = copy.copy(train), copy.copy(valid)
        train.uids = tuple(train.uids[i] for i in np.linspace(0, 1128, 16).round().astype(int))
        valid.uids = tuple(valid.uids[i] for i in np.linspace(0, 118, 4).round().astype(int))
        recipe = MRNetRecipe(epochs=2)
    else:
        if args.stop_after is not None or args.recovery_receipt is None:
            raise ValueError("Full fitting requires accepted exact-engine GPU recovery")
        recovery = json.loads(args.recovery_receipt.read_text())
        if (
            recovery.get("status") != "actual_full_source_MRNet_engine_GPU_recovery_passed"
            or recovery.get("binding") != binding
            or recovery.get("complete_checkpoint_and_prediction_bytes_equal") is not True
        ):
            raise ValueError("The older adapter pilot cannot accept a different full-source engine")
    if (
        not args.root.is_absolute()
        or not args.root.as_posix().startswith("/tmp/")
        or not args.output.is_absolute()
        or not args.output.as_posix().startswith("/tmp/")
        or args.output.parent.stat().st_dev == args.acquisition_receipt.stat().st_dev
        or shutil.disk_usage(args.output.parent).free < 23 * 1024**3
    ):
        raise ValueError(
            "Private node-local data/checkpoints with funded 3GiB and 20GiB reserve required"
        )
    encoder, width, provenance = create_encoder(config["encoder"])
    configure_encoder_training(encoder, True)
    model = MRNetAdapterNet(encoder, width, hidden_dim=128, chunk=16)
    device = torch.device("cuda", 0)
    result = fit_source(
        model,
        train,
        provenance,
        binding,
        recipe,
        args.output,
        device,
        resume=args.resume,
        stop_after=args.stop_after,
    )
    if result["status"] == "checkpointed_for_resume":
        raise SystemExit(75)
    predictions = predict_source(model, valid, device)
    atomic_json(args.output / "source-predictions.json", {"predictions": predictions})
    terminal = {
        "status": "private_full_Stanford_fitting_complete"
        if args.mode == "full"
        else "engineering_trajectory_complete",
        "binding": binding,
        "source_training_exams": len(train),
        "source_validation_exams": len(valid),
        "step": result["step"],
        "checkpoint_sha256": file_sha256(args.output / "last.pt"),
        "no_competition_submission_or_weight_publication": True,
    }
    if args.mode == "full":
        # Fixed final epoch only. These source tasks/splits cannot be compared
        # numerically to RSNA macro AUC or used as a replacement RSNA holdout.
        scores = np.asarray([row["probability"] for row in predictions])
        labels = np.asarray([[valid.labels[t][uid] for t in MRNET_TARGETS] for uid in valid.uids])
        terminal["fixed_final_source_validation_AUC"] = {
            target: float(roc_auc_score(labels[:, i], scores[:, i]))
            for i, target in enumerate(MRNET_TARGETS)
        }
        terminal["RSNA_or_Kaggle_accuracy_measured"] = False
    atomic_json(args.output / "terminal.json", terminal)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("full", "recovery", "recovery-worker"), default="full")
    parser.add_argument("--recovery-receipt", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.mode != "recovery":
        worker(args)
        return
    _, binding = read_binding(args)
    if args.output.exists() or shutil.disk_usage(args.output.parent).free < 23 * 1024**3:
        raise ValueError("Fresh funded private recovery directory required")
    args.output.mkdir()
    command = [
        sys.executable,
        "-B",
        str(Path(__file__)),
        "--mode",
        "recovery-worker",
        "--root",
        str(args.root),
        "--acquisition-receipt",
        str(args.acquisition_receipt),
        "--input-receipt",
        str(args.input_receipt),
        "--config",
        str(args.config),
    ]
    full, resumed = args.output / "full", args.output / "interrupted-resumed"
    for extra, expected in (
        (["--output", str(full)], 0),
        (["--output", str(resumed), "--stop-after", "1"], 75),
        (["--output", str(resumed), "--resume"], 0),
    ):
        if subprocess.run(command + extra, check=False).returncode != expected:
            raise ValueError("Actual full-source engine recovery trajectory failed")
    left = torch.load(full / "last.pt", map_location="cpu", weights_only=True)
    right = torch.load(resumed / "last.pt", map_location="cpu", weights_only=True)
    if (
        not same(left, right)
        or (full / "source-predictions.json").read_bytes()
        != (resumed / "source-predictions.json").read_bytes()
    ):
        raise ValueError(
            "Actual complete AdamW model/base LR counter RNG or prediction recovery differs"
        )
    atomic_json(
        args.output / "recovery-verified.json",
        {
            "status": "actual_full_source_MRNet_engine_GPU_recovery_passed",
            "binding": binding,
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "node": os.uname().nodename,
            "training_exams": 16,
            "engineering_validation_exams": 4,
            "optimizer_updates_per_full_trajectory": 4,
            "complete_checkpoint_and_prediction_bytes_equal": True,
            "clinical_accuracy_measured": False,
        },
    )


if __name__ == "__main__":
    main()
