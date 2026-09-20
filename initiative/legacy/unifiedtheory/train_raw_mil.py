#!/usr/bin/env python3
"""Train one high-resolution raw-DICOM max-MIL specialist fold."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .constants import TARGETS
    from .raw_mil import (
        RawStudyDataset,
        RawStudyMILModel,
        SliceFeatureBackbone,
        build_study_manifest,
        collate_raw_studies,
        normalize_pixels,
    )
    from .train import (
        assign_folds,
        compute_pos_weight,
        macro_auc,
        masked_bce,
        pairwise_auc_loss,
        seed_everything,
    )
except ImportError:
    from constants import TARGETS
    from raw_mil import (
        RawStudyDataset,
        RawStudyMILModel,
        SliceFeatureBackbone,
        build_study_manifest,
        collate_raw_studies,
        normalize_pixels,
    )
    from train import (
        assign_folds,
        compute_pos_weight,
        macro_auc,
        masked_bce,
        pairwise_auc_loss,
        seed_everything,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--backbone",
        choices=("dinov2", "efficientnet_b3", "radimagenet_resnet50"),
        required=True,
    )
    parser.add_argument("--model-name")
    parser.add_argument("--backbone-checkpoint", type=Path)
    parser.add_argument("--no-pretrained", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--image-size", type=int, default=336)
    parser.add_argument("--train-slices", type=int, default=24)
    parser.add_argument("--val-slices", type=int, default=32)
    parser.add_argument("--crop-mm", type=float, default=160.0)
    parser.add_argument("--trainable-blocks", type=int, default=2)
    parser.add_argument("--encoder-batch-size", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--backbone-lr", type=float, default=1e-5)
    parser.add_argument("--head-lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--gold-weight", type=float, default=8.0)
    parser.add_argument("--rank-weight", type=float, default=0.0)
    parser.add_argument("--pool", choices=("max", "topk", "logmeanexp"), default="max")
    parser.add_argument("--topk", type=int, default=3)
    parser.add_argument("--max-series-per-plane", type=int, default=1)
    parser.add_argument("--report-embeddings", type=Path)
    parser.add_argument("--report-weight", type=float, default=0.0)
    parser.add_argument("--limit-studies", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def move(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def model_forward(
    model: RawStudyMILModel, batch: dict[str, Any], return_aux: bool = False
) -> torch.Tensor | dict[str, torch.Tensor]:
    return model(
        normalize_pixels(batch["pixels"]),
        batch["plane"],
        batch["fluid"],
        batch["fatsat"],
        batch["study_index"],
        batch["num_studies"],
        return_aux=return_aux,
    )


def load_report_embeddings(path: Path | None) -> tuple[dict[str, torch.Tensor] | None, int]:
    if path is None:
        return None, 0
    payload = np.load(path, allow_pickle=False)
    uids = payload["uids"].astype(str)
    values = payload["embeddings"].astype(np.float32)
    if values.ndim != 2 or len(uids) != len(values):
        raise ValueError("report embedding archive has incompatible arrays")
    if len(set(uids)) != len(uids):
        raise ValueError("report embedding archive contains duplicate studies")
    return {
        uid: torch.from_numpy(value) for uid, value in zip(uids.tolist(), values)
    }, int(values.shape[1])


@torch.inference_mode()
def evaluate(model: RawStudyMILModel, loader: DataLoader, device: torch.device) -> dict[str, Any]:
    model.eval()
    logits, labels, masks, gold_masks, uids = [], [], [], [], []
    for batch in loader:
        uids.extend(batch["uid"])
        batch = move(batch, device)
        output = model_forward(model, batch)
        assert torch.is_tensor(output)
        logits.append(output.float().cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
        masks.append(batch["label_mask"].cpu().numpy())
        gold_masks.append(batch["gold_mask"].cpu().numpy())
    logits_array = np.concatenate(logits)
    labels_array = np.concatenate(labels)
    masks_array = np.concatenate(masks).astype(bool)
    gold_array = np.concatenate(gold_masks).astype(bool)
    probabilities = 1 / (1 + np.exp(-np.clip(logits_array, -30, 30)))
    evaluation_mask = masks_array & gold_array if np.any(masks_array & ~gold_array) else masks_array
    score, per_target = macro_auc(labels_array, probabilities, evaluation_mask)
    return {
        "uids": uids,
        "probabilities": probabilities,
        "macro_auc": score,
        "per_target": dict(zip(TARGETS, per_target)),
    }


def make_loader(
    frame: pd.DataFrame,
    manifest: dict[str, Any],
    args: argparse.Namespace,
    training: bool,
    report_embeddings: dict[str, torch.Tensor] | None,
) -> DataLoader:
    dataset = RawStudyDataset(
        frame,
        manifest,
        TARGETS,
        args.image_size,
        args.train_slices if training else args.val_slices,
        args.crop_mm,
        training,
        args.max_series_per_plane,
        report_embeddings,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=training,
        num_workers=args.workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.workers > 0,
        collate_fn=collate_raw_studies,
    )


def main() -> None:
    args = parse_args()
    if args.accumulate < 1:
        raise ValueError("--accumulate must be positive")
    if args.backbone == "radimagenet_resnet50" and args.backbone_checkpoint is None:
        raise ValueError("radimagenet_resnet50 training requires --backbone-checkpoint")
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    args.output.mkdir(parents=True, exist_ok=True)
    labels = pd.read_csv(args.labels_csv, dtype={"StudyInstanceUID": str})
    labels = assign_folds(labels, args.folds, args.seed, "fold", "scanner_group")
    if args.limit_studies:
        labels = labels.groupby("_fold", group_keys=False).head(args.limit_studies)
    train_frame = labels[labels["_fold"] != args.fold].copy()
    val_frame = labels[labels["_fold"] == args.fold].copy()
    report_embeddings, report_dim = load_report_embeddings(args.report_embeddings)
    if args.report_weight > 0 and report_embeddings is None:
        raise ValueError("--report-weight requires --report-embeddings")
    manifest = build_study_manifest(args.data_root / "train_series.csv", args.data_root, "train")
    train_loader = make_loader(train_frame, manifest, args, True, report_embeddings)
    val_loader = make_loader(val_frame, manifest, args, False, report_embeddings)

    backbone = SliceFeatureBackbone(
        args.backbone,
        args.model_name,
        args.trainable_blocks,
        args.local_files_only,
        not args.no_pretrained,
        args.backbone_checkpoint,
    )
    model = RawStudyMILModel(
        backbone,
        len(TARGETS),
        pool=args.pool,
        encoder_batch_size=args.encoder_batch_size,
        topk=args.topk,
        report_dim=report_dim,
    )
    device = torch.device(args.device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameter_groups(args.backbone_lr, args.head_lr),
        weight_decay=args.weight_decay,
    )
    updates_per_epoch = max(1, int(np.ceil(len(train_loader) / args.accumulate)))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs * updates_per_epoch)
    )
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    pos_weight = compute_pos_weight(train_frame, TARGETS).to(device)
    best, stale, history = -float("inf"), 0, []
    run_name = f"{args.backbone}_{args.pool}_{args.image_size}"
    checkpoint = args.output / f"{run_name}_fold{args.fold}.pt"

    for epoch in range(1, args.epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        running = 0.0
        for step, batch in enumerate(train_loader, start=1):
            batch = move(batch, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
                output = model_forward(model, batch, return_aux=args.report_weight > 0)
                prediction = output["logits"] if isinstance(output, dict) else output
                confidence = batch["confidence"] * torch.where(
                    batch["gold_mask"],
                    torch.full_like(batch["confidence"], args.gold_weight),
                    torch.ones_like(batch["confidence"]),
                )
                loss = masked_bce(
                    prediction,
                    batch["labels"],
                    batch["label_mask"],
                    confidence,
                    pos_weight,
                )
                if args.rank_weight:
                    loss = loss + args.rank_weight * pairwise_auc_loss(
                        prediction,
                        batch["labels"],
                        batch["label_mask"],
                        confidence,
                    )
                if args.report_weight:
                    assert isinstance(output, dict)
                    predicted_report = F.normalize(output["report_embedding"], dim=-1)
                    target_report = F.normalize(batch["report_embedding"], dim=-1)
                    loss = loss + args.report_weight * (
                        1 - (predicted_report * target_report).sum(dim=-1)
                    ).mean()
                scaled_loss = loss / args.accumulate
            scaler.scale(scaled_loss).backward()
            running += float(loss.detach())
            if step % args.accumulate == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
        train_seconds = time.monotonic() - epoch_started
        evaluation_started = time.monotonic()
        metrics = evaluate(model, val_loader, device)
        evaluation_seconds = time.monotonic() - evaluation_started
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, len(train_loader)),
            "macro_auc": metrics["macro_auc"],
            "per_target_auc": metrics["per_target"],
            "train_seconds": train_seconds,
            "evaluation_seconds": evaluation_seconds,
            "peak_cuda_gib": (
                torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0.0
            ),
        }
        history.append(record)
        print(json.dumps(record, allow_nan=True), flush=True)
        score = metrics["macro_auc"]
        if np.isfinite(score) and score > best:
            best, stale = score, 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "args": {
                        key: str(value) if isinstance(value, Path) else value
                        for key, value in vars(args).items()
                    }
                    | {"report_dim": report_dim},
                    "targets": TARGETS,
                    "fold": args.fold,
                    "score": score,
                    "backbone_load_report": backbone.load_report,
                },
                checkpoint,
            )
            oof = pd.DataFrame({"StudyInstanceUID": metrics["uids"]})
            for index, target in enumerate(TARGETS):
                oof[target] = metrics["probabilities"][:, index]
            oof.to_csv(args.output / f"{run_name}_fold{args.fold}_oof.csv", index=False)
        else:
            stale += 1
        if stale >= args.patience:
            break
    (args.output / f"{run_name}_fold{args.fold}_history.json").write_text(
        json.dumps(history, indent=2, allow_nan=True, default=str) + "\n"
    )
    print(f"best macro AUC={best:.6f}; checkpoint={checkpoint}")


if __name__ == "__main__":
    main()
