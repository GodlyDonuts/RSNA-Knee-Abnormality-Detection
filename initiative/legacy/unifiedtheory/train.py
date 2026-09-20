#!/usr/bin/env python3
"""Train one scanner-grouped fold of the DINO + ALiBi study model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import pandas as pd
import torch
from torch import Tensor
import torch.nn.functional as F
from torch.utils.data import DataLoader

try:
    from .constants import TARGETS
    from .data import (
        FeatureStudyDataset,
        collate_studies,
        load_feature_cache,
        merge_cache_and_labels,
        model_inputs,
        move_batch,
        patch_model_inputs,
    )
    from .model import KneeAlibiModel, KneeModelConfig
    from .patch_model import PatchKneeAlibiModel, PatchKneeModelConfig
except ImportError:
    from constants import TARGETS
    from data import (
        FeatureStudyDataset,
        collate_studies,
        load_feature_cache,
        merge_cache_and_labels,
        model_inputs,
        move_batch,
        patch_model_inputs,
    )
    from model import KneeAlibiModel, KneeModelConfig
    from patch_model import PatchKneeAlibiModel, PatchKneeModelConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-index", type=Path, required=True)
    parser.add_argument("--labels-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--aggregator", choices=("mean", "index_alibi", "physical_alibi"), default="physical_alibi"
    )
    parser.add_argument("--model-type", choices=("summary", "patch"), default="summary")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold-column", default="fold")
    parser.add_argument("--group-column", default="scanner_group")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--series-depth", type=int, default=2)
    parser.add_argument("--study-depth", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--rank-weight", type=float, default=0.1)
    parser.add_argument("--gold-weight", type=float, default=8.0)
    parser.add_argument("--series-dropout", type=float, default=0.15)
    parser.add_argument(
        "--token-adapter-bottleneck",
        type=int,
        default=0,
        help="cached DINO token-space adapter; 64 is the stage-two default experiment",
    )
    parser.add_argument("--report-weight", type=float, default=0.0)
    parser.add_argument("--report-embeddings", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def assign_folds(
    frame: pd.DataFrame,
    n_folds: int,
    seed: int,
    fold_column: str,
    group_column: str,
) -> pd.DataFrame:
    """Respect supplied folds; otherwise split by scanner group when possible."""

    frame = frame.copy()
    if fold_column in frame and frame[fold_column].notna().all():
        frame["_fold"] = frame[fold_column].astype(int)
        return frame
    if group_column in frame and frame[group_column].nunique() >= n_folds:
        from sklearn.model_selection import GroupKFold

        splitter = GroupKFold(n_splits=n_folds)
        frame["_fold"] = -1
        for fold, (_, val) in enumerate(splitter.split(frame, groups=frame[group_column])):
            frame.loc[frame.index[val], "_fold"] = fold
        frame.attrs["split_warning"] = "scanner-grouped"
        return frame
    from sklearn.model_selection import KFold

    splitter = KFold(n_splits=n_folds, shuffle=True, random_state=seed)
    frame["_fold"] = -1
    for fold, (_, val) in enumerate(splitter.split(frame)):
        frame.loc[frame.index[val], "_fold"] = fold
    frame.attrs["split_warning"] = (
        "random-fold fallback: not leakage-safe; provide scanner groups or precomputed folds"
    )
    return frame


def masked_bce(
    logits: Tensor,
    labels: Tensor,
    mask: Tensor,
    confidence: Tensor,
    pos_weight: Tensor,
) -> Tensor:
    values = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none", pos_weight=pos_weight
    )
    weights = mask.to(values.dtype) * confidence
    return (values * weights).sum() / weights.sum().clamp_min(1.0)


def pairwise_auc_loss(
    logits: Tensor, labels: Tensor, mask: Tensor, confidence: Tensor
) -> Tensor:
    losses: list[Tensor] = []
    for target in range(logits.shape[1]):
        valid = mask[:, target]
        positive = valid & (labels[:, target] > 0.5)
        negative = valid & (labels[:, target] <= 0.5)
        if not positive.any() or not negative.any():
            continue
        differences = logits[positive, target][:, None] - logits[negative, target][None, :]
        pair_weight = (
            confidence[positive, target][:, None] * confidence[negative, target][None, :]
        )
        losses.append((F.softplus(-differences) * pair_weight).sum() / pair_weight.sum().clamp_min(1))
    return torch.stack(losses).mean() if losses else logits.sum() * 0


def report_contrastive_loss(predicted: Tensor, report: Tensor, present: Tensor) -> Tensor:
    predicted = predicted[present]
    report = report[present]
    if predicted.shape[0] < 2:
        return predicted.sum() * 0
    predicted = F.normalize(predicted, dim=-1)
    report = F.normalize(report, dim=-1)
    similarities = predicted @ report.T / 0.07
    target = torch.arange(predicted.shape[0], device=predicted.device)
    return (F.cross_entropy(similarities, target) + F.cross_entropy(similarities.T, target)) / 2


def macro_auc(labels: np.ndarray, probabilities: np.ndarray, mask: np.ndarray) -> tuple[float, list[float]]:
    from sklearn.metrics import roc_auc_score

    scores: list[float] = []
    for target in range(labels.shape[1]):
        valid = mask[:, target].astype(bool)
        y = labels[valid, target]
        if len(y) == 0 or np.unique(y).size < 2:
            scores.append(float("nan"))
        else:
            scores.append(float(roc_auc_score(y, probabilities[valid, target])))
    finite = [score for score in scores if np.isfinite(score)]
    return (float(np.mean(finite)) if finite else float("nan"), scores)


def compute_pos_weight(frame: pd.DataFrame, targets: list[str]) -> Tensor:
    values = []
    for target in targets:
        y = pd.to_numeric(frame[target], errors="coerce") if target in frame else pd.Series(dtype=float)
        positive = float((y == 1).sum())
        negative = float((y == 0).sum())
        values.append(min(10.0, max(0.1, negative / max(positive, 1.0))))
    return torch.tensor(values, dtype=torch.float32)


def make_loader(
    frame: pd.DataFrame,
    batch_size: int,
    workers: int,
    shuffle: bool,
    report_embeddings: Path | None,
    include_patch_features: bool,
) -> DataLoader:
    dataset = FeatureStudyDataset(
        frame,
        TARGETS,
        report_embeddings,
        include_patch_features=include_patch_features,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
        collate_fn=collate_studies,
        drop_last=False,
    )


def inputs_for_model(batch: Dict[str, Any], model_type: str) -> Dict[str, Tensor]:
    return patch_model_inputs(batch) if model_type == "patch" else model_inputs(batch)


@torch.inference_mode()
def evaluate(
    model: KneeAlibiModel | PatchKneeAlibiModel,
    loader: DataLoader,
    device: torch.device,
    model_type: str,
) -> Dict[str, Any]:
    model.eval()
    logits_all: list[np.ndarray] = []
    labels_all: list[np.ndarray] = []
    masks_all: list[np.ndarray] = []
    gold_masks_all: list[np.ndarray] = []
    uids: list[str] = []
    for batch in loader:
        uids.extend(batch["uid"])
        batch = move_batch(batch, device)
        logits = model(**inputs_for_model(batch, model_type))
        logits_all.append(logits.float().cpu().numpy())
        labels_all.append(batch["labels"].cpu().numpy())
        masks_all.append(batch["label_mask"].cpu().numpy())
        gold_masks_all.append(batch["gold_mask"].cpu().numpy())
    logits = np.concatenate(logits_all)
    labels = np.concatenate(labels_all)
    masks = np.concatenate(masks_all)
    gold_masks = np.concatenate(gold_masks_all)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    # Weak report targets are useful training supervision but cannot be used to
    # select checkpoints. If weak labels are present anywhere, every target is
    # scored only on its expert subset; a one-class gold fold becomes NaN and is
    # omitted from the macro average rather than falling back to its teacher.
    evaluation_mask = masks.copy()
    if np.any(masks.astype(bool) & ~gold_masks.astype(bool)):
        evaluation_mask = masks.astype(bool) & gold_masks.astype(bool)
    score, per_target = macro_auc(labels, probabilities, evaluation_mask)
    return {
        "macro_auc": score,
        "per_target_auc": dict(zip(TARGETS, per_target)),
        "uids": uids,
        "logits": logits,
        "probabilities": probabilities,
        "labels": labels,
        "masks": masks,
        "evaluation_masks": evaluation_mask,
    }


def main() -> None:
    args = parse_args()
    seed_everything(args.seed)
    torch.set_float32_matmul_precision("high")
    args.output.mkdir(parents=True, exist_ok=True)
    frame = merge_cache_and_labels(args.cache_index, args.labels_csv)
    missing_targets = [target for target in TARGETS if target not in frame]
    if missing_targets:
        raise ValueError(
            "labels CSV must contain all 12 target columns; missing " + ", ".join(missing_targets)
        )
    frame = assign_folds(frame, args.folds, args.seed, args.fold_column, args.group_column)
    print(frame.attrs.get("split_warning", "using supplied fold assignments"), flush=True)
    train_frame = frame[frame["_fold"] != args.fold].copy()
    val_frame = frame[frame["_fold"] == args.fold].copy()
    if len(train_frame) == 0 or len(val_frame) == 0:
        raise ValueError(f"fold {args.fold} produced an empty train or validation split")

    example = load_feature_cache(train_frame.iloc[0]["cache_file"])
    feature_dim = int(example["features"].shape[-1])
    report_dim = 0
    if args.report_weight > 0:
        if args.report_embeddings is None:
            raise ValueError("--report-weight requires --report-embeddings")
        candidates = sorted(args.report_embeddings.glob("*.pt"))
        if not candidates:
            raise ValueError("report embedding directory contains no .pt files")
        try:
            report = torch.load(candidates[0], map_location="cpu", weights_only=True)
        except TypeError:
            report = torch.load(candidates[0], map_location="cpu")
        if isinstance(report, dict):
            report = report["embedding"]
        report_dim = int(torch.as_tensor(report).numel())

    if args.model_type == "patch":
        if "patch_features" not in example:
            raise ValueError("--model-type patch requires caches extracted with --patch-grid")
        if args.report_weight:
            raise ValueError("report embedding contrastive loss is currently summary-model only")
        config = PatchKneeModelConfig(
            feature_dim=feature_dim,
            patch_dim=int(example["patch_features"].shape[-1]),
            hidden_dim=args.hidden_dim,
            n_heads=args.heads,
            series_depth=args.series_depth,
            study_depth=args.study_depth,
            dropout=args.dropout,
            aggregator=args.aggregator,
            num_targets=len(TARGETS),
            series_dropout=args.series_dropout,
            token_adapter_bottleneck=args.token_adapter_bottleneck,
        )
        model = PatchKneeAlibiModel(config)
    else:
        config = KneeModelConfig(
            feature_dim=feature_dim,
            hidden_dim=args.hidden_dim,
            n_heads=args.heads,
            series_depth=args.series_depth,
            study_depth=args.study_depth,
            dropout=args.dropout,
            aggregator=args.aggregator,
            num_targets=len(TARGETS),
            report_dim=report_dim,
        )
        model = KneeAlibiModel(config)
    device = torch.device(args.device)
    model = model.to(device)
    train_loader = make_loader(
        train_frame,
        args.batch_size,
        args.workers,
        True,
        args.report_embeddings,
        include_patch_features=args.model_type == "patch",
    )
    val_loader = make_loader(
        val_frame,
        args.batch_size,
        args.workers,
        False,
        None,
        include_patch_features=args.model_type == "patch",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    amp_enabled = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    pos_weight = compute_pos_weight(train_frame, TARGETS).to(device)

    best_auc = -float("inf")
    stale = 0
    history: list[dict[str, Any]] = []
    run_name = args.aggregator if args.model_type == "summary" else f"patch_{args.aggregator}"
    checkpoint_path = args.output / f"{run_name}_fold{args.fold}.pt"
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                output = model(
                    **inputs_for_model(batch, args.model_type),
                    return_aux=args.report_weight > 0,
                )
                logits = output["logits"] if isinstance(output, dict) else output
                confidence = batch["confidence"] * torch.where(
                    batch["gold_mask"],
                    torch.full_like(batch["confidence"], args.gold_weight),
                    torch.ones_like(batch["confidence"]),
                )
                loss = masked_bce(
                    logits,
                    batch["labels"],
                    batch["label_mask"],
                    confidence,
                    pos_weight,
                )
                if args.rank_weight:
                    loss = loss + args.rank_weight * pairwise_auc_loss(
                        logits, batch["labels"], batch["label_mask"], confidence
                    )
                if args.report_weight and "report_embedding" in batch:
                    loss = loss + args.report_weight * report_contrastive_loss(
                        output["report_embedding"],
                        batch["report_embedding"],
                        batch["report_mask"],
                    )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running += float(loss.detach())

        metrics = evaluate(model, val_loader, device, args.model_type)
        record = {
            "epoch": epoch,
            "train_loss": running / max(1, len(train_loader)),
            "macro_auc": metrics["macro_auc"],
            "per_target_auc": metrics["per_target_auc"],
        }
        history.append(record)
        print(json.dumps(record, allow_nan=True), flush=True)
        score = metrics["macro_auc"]
        if np.isfinite(score) and score > best_auc:
            best_auc = score
            stale = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "model_config": config.to_dict(),
                    "model_type": args.model_type,
                    "targets": TARGETS,
                    "fold": args.fold,
                    "score": score,
                    "seed": args.seed,
                },
                checkpoint_path,
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
        json.dumps(history, indent=2, allow_nan=True) + "\n"
    )
    frame[["StudyInstanceUID", "_fold"]].to_csv(
        args.output / f"fold_assignments_seed{args.seed}.csv", index=False
    )
    print(f"best macro AUC={best_auc:.6f}; checkpoint={checkpoint_path}")


if __name__ == "__main__":
    main()
