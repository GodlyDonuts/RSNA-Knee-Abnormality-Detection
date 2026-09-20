from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import Tensor, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from rsna.data import StudyDataset
from rsna.model import StudyNet, auc_metrics, masked_bce
from rsna.schema import TARGETS


def seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)


def tensors(batch: dict, device: torch.device) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    return tuple(
        batch[name].to(device, non_blocking=True)
        for name in ("images", "features", "series_mask", "labels", "label_mask")
    )


@torch.inference_mode()
def predict(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> tuple[pd.DataFrame, dict]:
    model.eval()
    identifiers, probabilities, labels, masks = [], [], [], []
    for batch in loader:
        images, features, series_mask, y, mask = tensors(batch, device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            logits = model(images, features, series_mask)
        probabilities.append(logits.float().sigmoid().cpu().numpy())
        identifiers.extend(batch["study_uid"])
        labels.append(y.cpu().numpy())
        masks.append(mask.cpu().numpy())
    if not probabilities:
        raise ValueError("No studies to predict")
    scores = np.concatenate(probabilities)
    predictions = pd.DataFrame(scores, columns=TARGETS)
    predictions.insert(0, "StudyInstanceUID", identifiers)
    return predictions, auc_metrics(np.concatenate(labels), scores, np.concatenate(masks))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--slices", type=int, default=24)
    parser.add_argument("--side", type=int, default=224)
    parser.add_argument("--max-series", type=int, default=8)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.workers < 0 or args.lr <= 0:
        raise ValueError("Invalid training configuration")
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    torch.set_num_threads(2)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    manifest = pd.read_parquet(args.studies)
    series = pd.read_csv(args.series)
    train = manifest[manifest.split == "train"]
    train = train[train[[f"{t}__mask" for t in TARGETS]].sum(axis=1) > 0]
    validation = manifest[manifest.split == "weak_validation"]
    if train.empty or validation.empty:
        raise ValueError("Training and weak validation studies are both required")
    if set(train.report_group) & set(validation.report_group):
        raise ValueError("Duplicate reports cross train/validation")
    if set(train.leakage_group) & set(validation.leakage_group):
        raise ValueError("Duplicate images cross train/validation")
    heldout = manifest[manifest.split == "gold_evaluation"]
    if set(train.leakage_group) & set(heldout.leakage_group):
        raise ValueError("Training studies overlap gold evaluation groups")
    if any(train[f"{t}__provenance"].eq("official_gold").any() for t in TARGETS):
        raise ValueError("Gold labels cannot enter weak training")
    config = {"slices": args.slices, "side": args.side, "max_series": args.max_series}
    run_config = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "workers": args.workers,
        "world_size": world,
        "lr": args.lr,
    }
    training_data = StudyDataset(train, series, args.cache, augment=True, **config)
    validation_data = StudyDataset(validation, series, args.cache, **config)
    if (training_data.cache_side, training_data.source_revision) != (
        validation_data.cache_side,
        validation_data.source_revision,
    ):
        raise ValueError("Train/validation cache preprocessing differs")
    sampler: DistributedSampler | None = (
        DistributedSampler(training_data, seed=args.seed) if world > 1 else None
    )
    generator = torch.Generator().manual_seed(args.seed + rank)
    loader = DataLoader(
        training_data,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=sampler is None,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=seed_worker,
        generator=generator,
    )
    val_loader = DataLoader(validation_data, batch_size=args.batch_size, num_workers=args.workers)
    model = StudyNet(args.width).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scaler = torch.amp.GradScaler(device.type, enabled=device.type == "cuda")
    start_epoch, step = 0, 0
    manifest_hash = hashlib.sha256(args.studies.read_bytes()).hexdigest()
    series_hash = hashlib.sha256(args.series.read_bytes()).hexdigest()
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=True)
        if not checkpoint["epoch_complete"]:
            raise ValueError("Cannot resume a partial smoke-test epoch; start a fresh run")
        if (
            checkpoint["manifest_sha256"] != manifest_hash
            or checkpoint["series_sha256"] != series_hash
        ):
            raise ValueError("Resume data differs from checkpoint")
        if checkpoint["config"] != config or checkpoint["width"] != args.width:
            raise ValueError("Resume model/data configuration differs")
        if checkpoint["targets"] != list(TARGETS) or checkpoint["run_config"] != run_config:
            raise ValueError("Resume targets or training settings differ")
        if (checkpoint["cache_side"], checkpoint["cache_source_revision"]) != (
            training_data.cache_side,
            training_data.source_revision,
        ):
            raise ValueError("Resume cache preprocessing differs")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch, step = checkpoint["epoch"] + 1, checkpoint["step"]
        torch.set_rng_state(checkpoint["torch_rng"].cpu())
        generator.set_state(checkpoint["loader_rng"].cpu())
        if device.type == "cuda":
            torch.cuda.set_rng_state_all([state.cpu() for state in checkpoint["cuda_rng"]])
    wrapped: nn.Module = model
    if world > 1:
        wrapped = DistributedDataParallel(
            model, device_ids=[local_rank] if device.type == "cuda" else None
        )
    args.output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for epoch in range(start_epoch, args.epochs):
        torch.manual_seed(args.seed + rank + epoch)
        generator.manual_seed(args.seed + rank + epoch)
        if sampler is not None:
            sampler.set_epoch(epoch)
        wrapped.train()
        loss_sum, batches = 0.0, 0
        for batch in loader:
            images, features, series_mask, labels, mask = tensors(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device.type, enabled=device.type == "cuda"):
                logits = wrapped(images, features, series_mask)
                loss = masked_bce(logits, labels, mask)
            if not torch.isfinite(loss):
                raise ValueError("Non-finite training loss")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach())
            batches += 1
            step += 1
            if args.max_steps is not None and step >= args.max_steps:
                break
        if rank == 0:
            predictions, metrics = predict(model, val_loader, device)
            metrics.update(
                {
                    "epoch": epoch,
                    "step": step,
                    "training_loss": loss_sum / max(1, batches),
                    "elapsed_seconds": time.monotonic() - started,
                    "evaluation_source": "weak_report_validation",
                }
            )
            predictions.to_csv(args.output / "weak_validation_predictions.csv", index=False)
            with (args.output / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(metrics) + "\n")
            checkpoint = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "epoch": epoch,
                "step": step,
                "width": args.width,
                "config": config,
                "run_config": run_config,
                "targets": list(TARGETS),
                "manifest_sha256": manifest_hash,
                "series_sha256": series_hash,
                "seed": args.seed,
                "torch_rng": torch.get_rng_state(),
                "loader_rng": generator.get_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else [],
                "epoch_complete": batches == len(loader),
                "cache_side": training_data.cache_side,
                "cache_source_revision": training_data.source_revision,
            }
            temporary = args.output / "checkpoint.partial"
            torch.save(checkpoint, temporary)
            temporary.replace(args.output / "checkpoint.pt")
            print(json.dumps(metrics), flush=True)
        if world > 1:
            dist.barrier()
        if args.max_steps is not None and step >= args.max_steps:
            break
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
