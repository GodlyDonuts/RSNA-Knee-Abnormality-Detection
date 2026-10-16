"""Short end-to-end adaptation of an externally supplied, exact MRI foundation.

No external code, weights, clinical labels, image paths or network operations are
embedded here. Actual source/image/weight/GPU admissions belong to the wrapper.
All parameters train; BatchNorm running statistics stay at their accepted values.
One or two fixed ranks accumulate a complete effective batch and checkpoint only
at synchronized optimizer boundaries. This module makes no accuracy claim.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from rsna.competitive.checkpoint import (
    atomic_json,
    capture_rng,
    restore_rng,
    run_lock,
    save_checkpoint,
)
from rsna.competitive.released_context import anchored_loss

VERSION = "licensed_MRI_foundation_end_to_end_uniform12_v1"
PARENT_SHA = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"


@dataclass(frozen=True)
class Recipe:
    epochs: int = 3
    effective_batch: int = 8
    windows: int = 12
    seed: int = 20261016
    encoder_lr: float = 3e-6
    head_lr: float = 3e-5
    weight_decay: float = 0.02
    warmup_fraction: float = 0.1
    final_lr_fraction: float = 0.01
    source_weight: float = 0.25
    clip: float = 1.0

    def __post_init__(self):
        if (
            any(type(v) is not int or v < 1 for v in (self.epochs, self.effective_batch))
            or type(self.windows) is not int
            or not 1 <= self.windows <= 94
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**63
            or any(
                type(v) not in (int, float) or not math.isfinite(v) or v <= 0
                for v in (
                    self.encoder_lr,
                    self.head_lr,
                    self.weight_decay,
                    self.clip,
                    self.source_weight,
                    self.warmup_fraction,
                    self.final_lr_fraction,
                )
            )
            or self.source_weight > 1
            or self.warmup_fraction >= 1
            or self.final_lr_fraction > 1
        ):
            raise ValueError("Finite positive fixed end-to-end recipe required")


def sample(uid, recipe, epoch):
    """Sample from the original 94 centers without changing their donor slices."""
    if not isinstance(uid, str) or not uid or type(epoch) is not int or epoch < 0:
        raise ValueError("Original identity and zero-based epoch required")
    h = hashlib.sha256(f"{VERSION}/{recipe.seed}/{epoch}/{uid}".encode()).digest()
    g = torch.Generator().manual_seed(int.from_bytes(h[:8], "little") % (2**63))
    positions = torch.randperm(94, generator=g)[: recipe.windows].tolist()
    mirror = bool(torch.randint(0, 2, (), generator=g).item())
    return positions, mirror


def lr_factor(completed, total, recipe):
    if type(completed) is not int or not 0 <= completed < total:
        raise ValueError("Future optimizer boundary required")
    warm = max(1, math.ceil(total * recipe.warmup_fraction))
    if completed < warm:
        return (completed + 1) / warm
    fraction = (completed - warm) / max(1, total - warm - 1)
    return (
        recipe.final_lr_fraction
        + (1 - recipe.final_lr_fraction) * (1 + math.cos(math.pi * fraction)) / 2
    )


def train_with_frozen_bn(model):
    model.train()
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()


def fit(model, data, output, binding, recipe, *, device, resume=False, stop_after=None):
    """Fit a source-injected model exposing ``backbone`` and B,K,C,H,W forward.

    Dataset ``uids`` must be unique. ``case(index, positions, mirror)`` returns
    images plus aligned labels/bool mask and *view-specific* full-parent logits.
    Missing slots in the final batch use zero-loss dummy forwards on both ranks;
    normalization remains over actual studies and 12 targets, never dummy rows.
    """
    if not isinstance(recipe, Recipe) or not len(data) or len(data.uids) != len(data):
        raise ValueError("Versioned recipe and admitted nonempty original population required")
    if len(set(data.uids)) != len(data) or binding.get("parent_checkpoint_sha256") != PARENT_SHA:
        raise ValueError("Unique fitting identities and exact parent binding required")
    rank = dist.get_rank() if dist.is_initialized() else 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    if world not in (1, 2) or recipe.effective_batch % world:
        raise ValueError("Fixed one/two-rank divisible effective batch required")
    if device.type == "cuda" and torch.cuda.get_device_capability(device) < (7, 0):
        raise ValueError("Supported real CUDA GPU required")
    if not all(p.requires_grad for p in model.parameters()):
        raise ValueError("Every encoder and head parameter must train")
    batches = math.ceil(len(data) / recipe.effective_batch)
    total = recipe.epochs * batches
    if stop_after is not None and (type(stop_after) is not int or not 1 <= stop_after < total):
        raise ValueError("Strict intermediate optimizer boundary required")
    expected = {
        **binding,
        "version": VERSION,
        "recipe": asdict(recipe),
        "population": len(data),
        "population_sha256": hashlib.sha256(
            json.dumps(data.uids, separators=(",", ":")).encode()
        ).hexdigest(),
        "world_size": world,
        "torch_version": str(torch.__version__),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "precision": "FP16_autocast_FP32_loss_dynamic_scaler" if device.type == "cuda" else "FP32",
        "BN_running_statistics": "frozen_at_accepted_parent",
        "total_steps": total,
    }
    output = Path(output)
    model.to(device)
    encoder = list(model.backbone.parameters())
    head = [p for name, p in model.named_parameters() if not name.startswith("backbone.")]
    if not encoder or not head:
        raise ValueError("Distinct complete encoder and MIL-head parameter groups required")
    optimizer = torch.optim.AdamW(
        [{"params": encoder, "lr": recipe.encoder_lr}, {"params": head, "lr": recipe.head_lr}],
        weight_decay=recipe.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda", init_scale=1024.0)
    ddp = (
        DistributedDataParallel(
            model,
            device_ids=[device.index] if device.type == "cuda" else None,
            broadcast_buffers=False,
        )
        if world > 1
        else model
    )
    checkpoint = output / "checkpoint.pt"
    epoch = cursor = completed = 0
    history = []
    guard = run_lock(output) if rank == 0 else contextlib.nullcontext()
    with guard:
        if resume:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if state["binding"] != expected or state["optimizer_boundary"] is not True:
                raise ValueError(
                    "Exact source/population/recipe/runtime/optimizer boundary required"
                )
            epoch, cursor, completed = state["epoch"], state["cursor"], state["completed"]
            if (
                type(epoch) is not int
                or not 0 <= epoch <= recipe.epochs
                or type(cursor) is not int
                or not 0 <= cursor < batches
                or completed != epoch * batches + cursor
                or (epoch == recipe.epochs and cursor)
                or len(state["history"]) != completed
                or len(state["rng_by_rank"]) != world
                or len(state["scaler_by_rank"]) != world
                or state["schedule"]
                != {"completed": completed, "total": total, "kind": "warmup_cosine"}
            ):
                raise ValueError("Complete checkpoint cursors/history/RNG/scaler/schedule required")
            if stop_after is not None and stop_after <= completed:
                raise ValueError("Stop boundary must follow the resumed update")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scaler.load_state_dict(state["scaler_by_rank"][rank])
            history = state["history"]
            restore_rng(state["rng_by_rank"][rank], device)
        elif rank == 0:
            if checkpoint.exists() or (output / "run.json").exists():
                raise ValueError("Existing fitting output requires explicit verified resume")
            atomic_json(output / "run.json", {"binding": expected})
        if world > 1:
            dist.barrier()

        def save():
            rng = capture_rng(device)
            scale = scaler.state_dict()
            rngs: list = [rng]
            scales: list = [scale]
            if world > 1:
                rngs, scales = [None] * world, [None] * world
                dist.all_gather_object(rngs, rng)
                dist.all_gather_object(scales, scale)
                if any(s != scales[0] for s in scales):
                    raise ValueError("Mixed-precision scale diverged across synchronized ranks")
            if rank == 0:
                save_checkpoint(
                    checkpoint,
                    {
                        "binding": expected,
                        "optimizer_boundary": True,
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "scaler_by_rank": scales,
                        "rng_by_rank": rngs,
                        "schedule": {
                            "completed": completed,
                            "total": total,
                            "kind": "warmup_cosine",
                        },
                        "epoch": epoch,
                        "cursor": cursor,
                        "completed": completed,
                        "history": history,
                    },
                    min_free_gib=2.0,
                )
            if world > 1:
                dist.barrier()

        train_with_frozen_bn(model)
        while epoch < recipe.epochs:
            order = torch.randperm(
                len(data), generator=torch.Generator().manual_seed(recipe.seed + epoch)
            ).tolist()
            indices = order[cursor * recipe.effective_batch : (cursor + 1) * recipe.effective_batch]
            factor = lr_factor(completed, total, recipe)
            for group, peak in zip(
                optimizer.param_groups, (recipe.encoder_lr, recipe.head_lr), strict=True
            ):
                group["lr"] = peak * factor
            optimizer.zero_grad(set_to_none=True)
            local_loss = torch.zeros((), device=device, dtype=torch.float64)
            micros = math.ceil(len(indices) / world)
            for micro in range(micros):
                position = micro * world + rank
                real = position < len(indices)
                index = indices[position] if real else indices[0]
                selected, mirror = sample(data.uids[index], recipe, epoch)
                case = data.case(index, selected, mirror)
                images = case["images"].to(device)
                if (
                    images.ndim != 4
                    or images.shape[0] != recipe.windows
                    or not torch.isfinite(images).all()
                ):
                    raise ValueError("Finite original selected-window tensor required")
                sync = (
                    ddp.no_sync() if world > 1 and micro + 1 < micros else contextlib.nullcontext()
                )
                with sync:
                    with torch.autocast(
                        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
                    ):
                        logits = ddp(images.unsqueeze(0)).float()
                    loss = anchored_loss(
                        logits,
                        case["parent_logits"].to(device).reshape(1, 12).float(),
                        case["labels"].to(device).reshape(1, 12).float(),
                        case["mask"].to(device).reshape(1, 12),
                        source_weight=recipe.source_weight,
                    )
                    weighted = loss * (world / len(indices) if real else 0.0)
                    scaler.scale(weighted).backward()
                    if real:
                        local_loss += loss.detach().double() / len(indices)
            scaler.unscale_(optimizer)
            valid = all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
            )
            ok = torch.tensor(int(valid), device=device)
            if world > 1:
                dist.all_reduce(ok, op=dist.ReduceOp.MIN)
            if not ok.item():
                raise ValueError("Missing/nonfinite actual encoder or MIL-head gradient")
            norm = nn.utils.clip_grad_norm_(
                model.parameters(), recipe.clip, error_if_nonfinite=True
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            if world > 1:
                dist.all_reduce(local_loss)
            completed += 1
            cursor += 1
            history.append(
                {
                    "update": completed,
                    "epoch": epoch + 1,
                    "batch": cursor,
                    "studies": len(indices),
                    "loss": float(local_loss),
                    "gradient_norm": float(norm),
                    "encoder_lr": optimizer.param_groups[0]["lr"],
                    "head_lr": optimizer.param_groups[1]["lr"],
                    "scale": scaler.get_scale(),
                }
            )
            if rank == 0:
                atomic_json(
                    output / "progress.json",
                    {"completed": completed, "total": total, "last": history[-1]},
                )
            finished = cursor == batches
            if finished:
                if rank == 0:
                    snapshot = output / "epoch-checkpoints" / f"epoch-{epoch + 1:03d}.pt"
                    if snapshot.exists():
                        raise ValueError("Completed immutable epoch already exists")
                    save_checkpoint(
                        snapshot,
                        {
                            "binding": expected,
                            "epoch": epoch + 1,
                            "completed": completed,
                            "model": model.state_dict(),
                        },
                        min_free_gib=2.0,
                    )
                epoch += 1
                cursor = 0
            stopped = stop_after == completed
            if finished or stopped:
                save()
            if stopped:
                break
        return {
            "status": "fixed_final_fit_complete"
            if completed == total
            else "stopped_optimizer_boundary",
            "completed": completed,
            "total": total,
            "world_size": world,
        }
