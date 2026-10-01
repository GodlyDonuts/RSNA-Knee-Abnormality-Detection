"""Private source-specific exam fitting, with complete optimizer-boundary recovery.

This engine does not map source labels to RSNA targets or publish learned weights.
It is shared by the planned full-source run and its bounded recovery exercise.
"""

from __future__ import annotations

import math
import os
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from rsna.competitive.checkpoint import (
    atomic_json,
    capture_rng,
    file_sha256,
    restore_rng,
    run_lock,
    save_checkpoint,
)
from rsna.competitive.delta_checkpoint import FrozenBaseDelta
from rsna.competitive.mrnet import MRNET_TARGETS, MRNetAdapterNet, MRNetStudyDataset


@dataclass(frozen=True)
class MRNetRecipe:
    epochs: int = 10
    effective_batch: int = 8
    seed: int = 20261001
    encoder_lr: float = 1e-4
    head_lr: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1

    def __post_init__(self) -> None:
        if (
            self.epochs < 1
            or self.effective_batch < 1
            or not 0 <= self.warmup_epochs < self.epochs
            or any(not math.isfinite(v) or v <= 0 for v in (self.encoder_lr, self.head_lr))
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid private Stanford fitting recipe")


def _multiplier(step: int, total: int, warmup: int) -> float:
    if step < warmup:
        return (step + 1) / warmup
    return (1 + math.cos(math.pi * (step - warmup) / (total - warmup))) / 2


def fit_source(
    model: MRNetAdapterNet,
    train: MRNetStudyDataset,
    provenance: dict[str, Any],
    binding: dict[str, Any],
    recipe: MRNetRecipe,
    output: Path,
    device: torch.device,
    *,
    resume: bool = False,
    stop_after: int | None = None,
) -> dict[str, Any]:
    """Accumulate an exam-level loss over every training exam once per epoch.

    The last smaller batch uses its actual size. Validation cannot enter fitting.
    Snapshots retain full AdamW/counters/RNG plus all learned tensors, with the
    unchanged frozen official base reconstructed from its exact provenance.
    """
    if (
        train.split != "train"
        or train.training_permitted is not True
        or not len(train)
        or len(set(train.uids)) != len(train)
        or binding.get("private_research_only") is not True
        or binding.get("external_weight_publication") is not False
    ):
        raise ValueError("Unique source training exams and private research scope required")
    steps_per_epoch = math.ceil(len(train) / recipe.effective_batch)
    total = steps_per_epoch * recipe.epochs
    if stop_after is not None and not 1 <= stop_after < total:
        raise ValueError("Interruption must be a nonterminal optimizer boundary")
    bound = {
        **binding,
        "recipe": asdict(recipe),
        "source_target_order": list(MRNET_TARGETS),
        "training_cases": list(train.uids),
        "steps_per_epoch": steps_per_epoch,
        "device_type": device.type,
        "precision": "bf16_autocast_fp32_loss" if device.type == "cuda" else "fp32",
    }
    if output.exists() and not resume:
        raise FileExistsError("Existing training output requires read-only reconciliation")
    if resume and not (output / "last.pt").is_file():
        raise ValueError("Recovery needs an existing complete optimizer checkpoint")
    guard = FrozenBaseDelta(model, provenance)
    model.to(device)
    optimizer = torch.optim.AdamW(
        [
            {
                "params": [p for p in model.encoder.parameters() if p.requires_grad],
                "lr": recipe.encoder_lr,
            },
            {
                "params": [
                    p
                    for name, p in model.named_parameters()
                    if p.requires_grad and not name.startswith("encoder.")
                ],
                "lr": recipe.head_lr,
            },
        ],
        weight_decay=recipe.weight_decay,
    )
    with run_lock(output):
        step, updates = 0, []
        if resume:
            saved = torch.load(output / "last.pt", map_location="cpu", weights_only=True)
            if saved.get("binding") != bound or saved.get("optimizer_boundary") is not True:
                raise ValueError("Exact source, recipe and optimizer boundary required")
            step, updates = saved["step"], saved["updates"]
            if (
                not isinstance(step, int)
                or not 1 <= step <= total
                or len(updates) != step
                or [row["step"] for row in updates] != list(range(1, step + 1))
            ):
                raise ValueError("Incomplete Stanford optimizer history")
            guard.restore(model, saved)
            optimizer.load_state_dict(saved["optimizer"])
            restore_rng(saved["rng"], device)
        while step < total:
            epoch, batch = divmod(step, steps_per_epoch)
            generator = torch.Generator().manual_seed(recipe.seed + epoch)
            order = torch.randperm(len(train), generator=generator).tolist()
            indices = order[batch * recipe.effective_batch : (batch + 1) * recipe.effective_batch]
            multiplier = _multiplier(step, total, recipe.warmup_epochs * steps_per_epoch)
            optimizer.param_groups[0]["lr"] = recipe.encoder_lr * multiplier
            optimizer.param_groups[1]["lr"] = recipe.head_lr * multiplier
            optimizer.zero_grad(set_to_none=True)
            model.train()
            losses = []
            for index in indices:
                item = train[index]
                if item["source_split"] != "train" or item["labels"].shape != (3,):
                    raise ValueError("Source-specific training exam and three labels required")
                context = (
                    torch.autocast("cuda", dtype=torch.bfloat16)
                    if device.type == "cuda"
                    else nullcontext()
                )
                with context:
                    logits = model(
                        item["images"][None].to(device),
                        item["slice_mask"][None].to(device),
                        item["ordinal"][None].to(device),
                    )
                    labels = item["labels"][None].to(device)
                    if (
                        not torch.isfinite(labels).all()
                        or not ((labels == 0) | (labels == 1)).all()
                    ):
                        raise ValueError("Only observed binary source labels may supervise fitting")
                    loss = nn.functional.binary_cross_entropy_with_logits(logits.float(), labels)
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite Stanford exam loss")
                (loss / len(indices)).backward()
                losses.append(float(loss.detach()))
            learned = [p for p in model.parameters() if p.requires_grad]
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in learned):
                raise ValueError("Nonfinite Stanford adapter gradient")
            gradient = nn.utils.clip_grad_norm_(learned, 1.0)
            if not torch.isfinite(gradient) or gradient <= 0:
                raise ValueError("Finite nonzero real optimizer gradient required")
            optimizer.step()
            if any(not torch.isfinite(p).all() for p in learned):
                raise ValueError("Nonfinite Stanford learned tensor")
            step += 1
            row = {
                "step": step,
                "epoch": epoch + 1,
                "source_cases": [train.uids[index] for index in indices],
                "exams": len(indices),
                "mean_exam_loss": sum(losses) / len(indices),
                "unclipped_gradient_norm": float(gradient),
                "encoder_lr": optimizer.param_groups[0]["lr"],
                "head_lr": optimizer.param_groups[1]["lr"],
            }
            updates.append(row)
            print({"actual_private_MRNet_update": row}, flush=True)
            # Keep the next-batch cursor and every RNG stream at an effective
            # batch boundary; wall time and resource measurements aren't science.
            if step % steps_per_epoch == 0 or stop_after == step:
                state = {
                    **guard.checkpoint_model(model),
                    "binding": bound,
                    "optimizer_boundary": True,
                    "step": step,
                    "optimizer": optimizer.state_dict(),
                    "rng": capture_rng(device),
                    "updates": updates,
                }
                save_checkpoint(output / "last.pt", state)
                if step % steps_per_epoch == 0:
                    snapshot = output / f"epoch-{epoch + 1:03d}.pt"
                    if snapshot.exists():
                        raise FileExistsError("An immutable epoch snapshot already exists")
                    os.link(output / "last.pt", snapshot)
                    atomic_json(
                        snapshot.with_suffix(".json"),
                        {
                            "epoch": epoch + 1,
                            "step": step,
                            "bytes": snapshot.stat().st_size,
                            "sha256": file_sha256(snapshot),
                            "binding": bound,
                            "private_research_only": True,
                        },
                    )
            if stop_after == step:
                return {"status": "checkpointed_for_resume", "step": step, "updates": updates}
        result = {"status": "private_source_fitting_complete", "step": step, "updates": updates}
        atomic_json(output / "training-result.json", result)
        return result


@torch.no_grad()
def predict_source(
    model: MRNetAdapterNet, valid: MRNetStudyDataset, device: torch.device
) -> list[dict[str, Any]]:
    if valid.split != "valid" or valid.training_permitted is not False:
        raise ValueError("Only the untouched original source validation split is evaluated")
    model.eval()
    predictions = []
    for index, uid in enumerate(valid.uids):
        item = valid[index]
        context = (
            torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        )
        with context:
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
            raise ValueError("Nonfinite Stanford source prediction")
        predictions.append({"source_case": uid, "probability": probability[0].cpu().tolist()})
    return predictions
