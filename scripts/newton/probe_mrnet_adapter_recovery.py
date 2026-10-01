"""Bounded Stanford adapter training and exact fresh-process recovery check.

Sixteen source-training exams, four validation exams for engineering predictions,
two epochs/four effective-eight updates. No clinical selection or submission.
All learned states stay on the owning private node; no external-derived weights
are published while downstream source permissions remain unresolved.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

from rsna.competitive.adapters import configure_encoder_training
from rsna.competitive.backbones import create_encoder
from rsna.competitive.checkpoint import (
    atomic_json,
    capture_rng,
    file_sha256,
    restore_rng,
    save_checkpoint,
)
from rsna.competitive.delta_checkpoint import FrozenBaseDelta
from rsna.competitive.mrnet import MRNetAdapterNet, MRNetStudyDataset


def same(left, right):
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.shape == right.shape and left.dtype == right.dtype and torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right, strict=True))
    return left == right


def worker(args):
    if (
        not torch.cuda.is_available()
        or torch.cuda.device_count() != 1
        or torch.cuda.get_device_capability() != (9, 0)
    ):
        raise ValueError("One allocated H100 required")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    random.seed(20261001)
    np.random.seed(20261001)
    torch.manual_seed(20261001)
    torch.cuda.manual_seed_all(20261001)
    config = json.loads(args.config.read_text())
    source = json.loads(args.input_receipt.read_text())
    if (
        source["status"] != "actual_Stanford_three_plane_exam_tensors_verified"
        or source["source_train_exams"] != 1129
        or source["validation_training_permitted"] is not False
    ):
        raise ValueError("Actual source input admission required")
    train = MRNetStudyDataset(args.root, args.acquisition_receipt, "train")
    valid = MRNetStudyDataset(args.root, args.acquisition_receipt, "valid")
    chosen = np.linspace(0, len(train) - 1, 16).round().astype(int).tolist()
    encoder, width, provenance = create_encoder(config["encoder"])
    configure_encoder_training(encoder, True)
    model = MRNetAdapterNet(encoder, width, hidden_dim=128, chunk=16)
    guard = FrozenBaseDelta(model, provenance)
    device = torch.device("cuda", 0)
    model.to(device)
    base_parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    head_parameters = [
        p
        for name, p in model.named_parameters()
        if p.requires_grad and not name.startswith("encoder.")
    ]
    optimizer = torch.optim.AdamW(
        [{"params": base_parameters, "lr": 1e-4}, {"params": head_parameters, "lr": 3e-4}],
        weight_decay=1e-4,
    )
    binding = {
        "source_input_receipt_sha256": file_sha256(args.input_receipt),
        "acquisition_receipt_sha256": file_sha256(args.acquisition_receipt),
        "encoder_provenance": provenance,
        "training_cases": [train.uids[i] for i in chosen],
        "source_target_order": ["acl", "meniscus", "abnormal"],
        "epochs": 2,
        "effective_batch": 8,
        "optimizer_updates": 4,
        "source_git": config["source_git"],
        "private_research_only": True,
        "clinical_selection": False,
    }
    step, updates = 0, []
    if args.resume:
        saved = torch.load(args.output / "last.pt", map_location="cpu", weights_only=False)
        if saved["binding"] != binding:
            raise ValueError("Exact private Stanford recovery binding required")
        guard.restore(model, saved)
        optimizer.load_state_dict(saved["optimizer"])
        restore_rng(saved["rng"], device)
        step, updates = saved["step"], saved["updates"]
    elif args.output.exists():
        raise FileExistsError("Existing Stanford training trajectory requires read-only review")
    else:
        args.output.mkdir()
    start = time.monotonic()
    while step < 4:
        epoch, batch = divmod(step, 2)
        generator = torch.Generator().manual_seed(20261001 + epoch)
        order = torch.randperm(16, generator=generator).tolist()
        multiplier = 1.0 if step == 0 else (1 + math.cos(math.pi * step / 4)) / 2
        optimizer.param_groups[0]["lr"] = 1e-4 * multiplier
        optimizer.param_groups[1]["lr"] = 3e-4 * multiplier
        optimizer.zero_grad(set_to_none=True)
        model.train()
        losses = []
        for offset in range(batch * 8, (batch + 1) * 8):
            item = train[chosen[order[offset]]]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(
                    item["images"][None].to(device),
                    item["slice_mask"][None].to(device),
                    item["ordinal"][None].to(device),
                )
                loss = nn.functional.binary_cross_entropy_with_logits(
                    logits.float(), item["labels"][None].to(device)
                )
            if not torch.isfinite(loss):
                raise ValueError("Nonfinite actual Stanford loss")
            (loss / 8).backward()
            losses.append(float(loss.detach()))
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
            raise ValueError("Nonfinite actual Stanford adapter gradient")
        gradient = nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if not torch.isfinite(gradient) or gradient <= 0:
            raise ValueError("Actual finite nonzero optimizer gradient required")
        optimizer.step()
        if any(not torch.isfinite(p).all() for p in model.parameters() if p.requires_grad):
            raise ValueError("Nonfinite actual Stanford learned tensor")
        step += 1
        updates.append(
            {
                "step": step,
                "mean_exam_loss": sum(losses) / 8,
                "unclipped_gradient_norm": float(gradient),
                "encoder_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
            }
        )
        state = {
            **guard.checkpoint_model(model),
            "binding": binding,
            "step": step,
            "optimizer": optimizer.state_dict(),
            "rng": capture_rng(device),
            "updates": updates,
        }
        save_checkpoint(args.output / "last.pt", state)
        if step % 2 == 0:
            os.link(args.output / "last.pt", args.output / f"epoch-{step // 2:03d}.pt")
        print(
            json.dumps(
                {"actual_MRNet_optimizer_update": updates[-1], "trajectory": args.output.name}
            ),
            flush=True,
        )
        if args.stop_after == step:
            os.link(args.output / "last.pt", args.output / "preserved-interrupted-step.pt")
            raise SystemExit(75)
    model.eval()
    predictions = []
    with torch.no_grad():
        for index in np.linspace(0, len(valid) - 1, 4).round().astype(int).tolist():
            item = valid[index]
            with torch.autocast("cuda", dtype=torch.bfloat16):
                probability = (
                    model(
                        item["images"][None].to(device),
                        item["slice_mask"][None].to(device),
                        item["ordinal"][None].to(device),
                    )
                    .float()
                    .sigmoid()
                )
            if not torch.isfinite(probability).all():
                raise ValueError("Nonfinite source engineering prediction")
            predictions.append(
                {
                    "source_case": item["source_case_id"],
                    "probability": probability[0].cpu().tolist(),
                }
            )
    atomic_json(
        args.output / "engineering-predictions.json",
        {"clinical_selection": False, "predictions": predictions},
    )
    atomic_json(
        args.output / "terminal.json",
        {
            "status": "private_MRNet_adapter_training_trajectory_complete",
            "step": step,
            "elapsed_seconds": time.monotonic() - start,
            "binding": binding,
            "checkpoint_sha256": file_sha256(args.output / "last.pt"),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--input-receipt", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after", type=int)
    args = parser.parse_args()
    if args.worker:
        worker(args)
        return
    if args.output.exists() or shutil.disk_usage(args.output.parent).free < 22 * 1024**3:
        raise ValueError(
            "Fresh private node-local trajectory and funded2GiB checkpoint budget required"
        )
    args.output.mkdir()
    command = [
        sys.executable,
        "-B",
        str(Path(__file__)),
        "--worker",
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
        result = subprocess.run(command + extra, check=False)
        if result.returncode != expected:
            raise ValueError("Actual Stanford trainer trajectory failed")
    left = torch.load(full / "last.pt", map_location="cpu", weights_only=False)
    right = torch.load(resumed / "last.pt", map_location="cpu", weights_only=False)
    if (
        not same(left, right)
        or (full / "engineering-predictions.json").read_bytes()
        != (resumed / "engineering-predictions.json").read_bytes()
    ):
        raise ValueError(
            "Actual model optimizer LR counter RNG or predictions differ after fresh-process recovery"
        )
    result = {
        "status": "actual_Stanford_adapter_four_update_fresh_process_recovery_passed",
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.uname().nodename,
        "training_exams": 16,
        "engineering_validation_exams": 4,
        "epochs": 2,
        "optimizer_updates_per_full_trajectory": 4,
        "complete_learned_model_base_AdamW_LR_counters_all_RNG_and_predictions_equal": True,
        "clinical_accuracy_claimed": False,
        "production_training": False,
        "weights_published": False,
        "source_training_labels_never_mapped_to_RSNA_targets": True,
    }
    atomic_json(args.output / "recovery-verified.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
