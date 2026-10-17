"""One fixed-final numerical engine for cached/online spatial-reader fitting.

The encoder is an immutable external parent; only the reader belongs to the
optimizer. Diagnostic cases/weights cannot become a competition candidate.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsna.competitive.checkpoint import (
    atomic_json,
    capture_rng,
    restore_rng,
    run_lock,
    save_checkpoint,
)
from rsna.competitive.released_context import anchored_loss
from rsna.competitive.spatial_reader import VERSION, SpatialReader
from rsna.schema import TARGETS


@dataclass(frozen=True)
class SpatialRecipe:
    epochs: int = 3
    effective_batch: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    warmup_epochs: int = 1
    clip: float = 1.0
    teacher_fraction: float = 0.25
    seed: int = 20261017
    channels: int = 1024
    width: int = 64

    def __post_init__(self):
        for name in ("epochs", "effective_batch", "channels", "width"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive fixed reader schedule and dimensions required")
        if (
            type(self.seed) is not int
            or not 0 <= self.seed < 2**63
            or type(self.warmup_epochs) is not int
            or not 0 <= self.warmup_epochs < self.epochs
            or not all(math.isfinite(x) and x > 0 for x in (self.learning_rate, self.clip))
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
            or not 0 <= self.teacher_fraction < 1
        ):
            raise ValueError("Invalid reader optimizer or source-anchoring recipe")


def learning_rate(recipe: SpatialRecipe, completed: int, batches: int) -> float:
    total, warmup = recipe.epochs * batches, recipe.warmup_epochs * batches
    if not 0 <= completed < total:
        raise ValueError("Update is outside fixed reader schedule")
    if completed < warmup:
        return recipe.learning_rate * (completed + 1) / warmup
    return (
        recipe.learning_rate
        * 0.5
        * (1 + math.cos(math.pi * (completed - warmup) / max(1, total - warmup - 1)))
    )


def epoch_order(count: int, seed: int, epoch: int) -> list[int]:
    raw = json.dumps([VERSION, count, seed, epoch], separators=(",", ":")).encode()
    number = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") % (2**63)
    return torch.randperm(count, generator=torch.Generator().manual_seed(number)).tolist()


class SpatialTrainer:
    def __init__(
        self, data, recipe: SpatialRecipe, *, arm: str, device: torch.device, binding: dict
    ):
        if arm not in {"aligned", "null"} or not len(data) or not data.binding or not binding:
            raise ValueError("Complete admitted reader inputs, binding and matched arm required")
        if type(binding.get("diagnostic_only")) is not bool:
            raise ValueError("Explicit diagnostic/production eligibility required")
        if not binding["diagnostic_only"] and (
            len(data) != 2247
            or recipe.epochs != 3
            or recipe.effective_batch != 8
            or recipe.channels != 1024
            or recipe.width != 64
        ):
            raise ValueError("Production reader requires the admitted fixed fold0 schedule")
        self.data, self.recipe, self.arm, self.device = data, recipe, arm, device
        random.seed(recipe.seed)
        np.random.seed(recipe.seed % (2**32))
        torch.manual_seed(recipe.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(recipe.seed)
        self.model = SpatialReader(recipe.channels, recipe.width, seed=recipe.seed).to(device)
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=recipe.learning_rate,
            weight_decay=recipe.weight_decay,
            foreach=False,
        )
        self.batches = math.ceil(len(data) / recipe.effective_batch)
        self.total = recipe.epochs * self.batches
        self.binding = {
            **binding,
            "engine": VERSION,
            "recipe": asdict(recipe),
            "arm": arm,
            "data": data.binding,
            "fitting_count": len(data),
            "total_steps": self.total,
            "targets": list(TARGETS),
            "torch_version": str(torch.__version__),
            "runtime": {
                "device_type": device.type,
                "deterministic": torch.are_deterministic_algorithms_enabled(),
                "matmul_TF32": torch.backends.cuda.matmul.allow_tf32,
                "cudnn_benchmark": torch.backends.cudnn.benchmark,
                "cudnn_deterministic": torch.backends.cudnn.deterministic,
                "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            },
        }
        self.epoch, self.cursor, self.completed = 0, 0, 0
        self.history: list[dict] = []

    def state(self):
        return {
            "schema_version": 1,
            "binding": self.binding,
            "optimizer_boundary": True,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "epoch": self.epoch,
            "cursor": self.cursor,
            "completed": self.completed,
            "order": epoch_order(len(self.data), self.recipe.seed, self.epoch)
            if self.epoch < self.recipe.epochs
            else [],
            "scheduler": {
                "completed": self.completed,
                "next_lr": learning_rate(self.recipe, self.completed, self.batches)
                if self.completed < self.total
                else 0.0,
            },
            "rng": capture_rng(self.device),
            "history": self.history,
        }

    def save(self, path: Path):
        save_checkpoint(path, self.state())

    def load(self, path: Path):
        value = torch.load(path, map_location="cpu", weights_only=True)
        if value.get("binding") != self.binding or value.get("optimizer_boundary") is not True:
            raise ValueError("Reader source, data, schedule or arm changed")
        epoch, cursor, completed = value["epoch"], value["cursor"], value["completed"]
        if (
            any(type(x) is not int for x in (epoch, cursor, completed))
            or not 0 <= epoch <= self.recipe.epochs
            or not 0 <= cursor < self.batches
            or completed != epoch * self.batches + cursor
            or completed > self.total
            or (epoch == self.recipe.epochs and cursor != 0)
            or len(value["history"]) != completed
            or value["order"]
            != (
                epoch_order(len(self.data), self.recipe.seed, epoch)
                if epoch < self.recipe.epochs
                else []
            )
            or value["scheduler"]
            != {
                "completed": completed,
                "next_lr": learning_rate(self.recipe, completed, self.batches)
                if completed < self.total
                else 0.0,
            }
        ):
            raise ValueError("Reader optimizer-boundary cursor/order/LR state differs")
        self.model.load_state_dict(value["model"], strict=True)
        self.optimizer.load_state_dict(value["optimizer"])
        self.epoch, self.cursor, self.completed, self.history = (
            epoch,
            cursor,
            completed,
            value["history"],
        )
        restore_rng(value["rng"], self.device)

    def fit(self, output: Path, *, resume=False, stop_after=None, should_stop=None):
        checkpoint = output / "checkpoint.pt"
        if stop_after is not None and (
            type(stop_after) is not int or not 0 < stop_after <= self.total
        ):
            raise ValueError("Bounded complete future update required")
        with run_lock(output):
            if resume:
                self.load(checkpoint)
            elif checkpoint.exists() or (output / "run.json").exists():
                raise ValueError("Existing original reader requires explicit resume")
            else:
                atomic_json(output / "run.json", {"binding": self.binding})
            if stop_after is not None and stop_after <= self.completed:
                raise ValueError("Stop boundary must be in the future")
            self.model.train()
            while self.completed < self.total:
                order = epoch_order(len(self.data), self.recipe.seed, self.epoch)
                indices = order[
                    self.cursor * self.recipe.effective_batch : (self.cursor + 1)
                    * self.recipe.effective_batch
                ]
                self.optimizer.zero_grad(set_to_none=True)
                rate = learning_rate(self.recipe, self.completed, self.batches)
                for group in self.optimizer.param_groups:
                    group["lr"] = rate
                loss_value, maximum, relations = 0.0, 0.0, 0
                for index in indices:
                    sample = self.data[index]
                    base = sample["base_logits"].to(self.device)
                    logits, correction = self.model(
                        sample["maps"].to(self.device), base, sample["plan"], arm=self.arm
                    )
                    # Explicit naming: 75% observed source +25% parent on observed
                    # cells. Missing cells retain the detached parent soft target.
                    loss = anchored_loss(
                        logits[None],
                        base[None],
                        sample["labels"].to(self.device)[None],
                        sample["label_mask"].to(self.device)[None],
                        source_weight=1 - self.recipe.teacher_fraction,
                    ) / len(indices)
                    loss.backward()
                    loss_value += float(loss.detach())
                    maximum = max(maximum, float(correction.detach().abs().max()))
                    relations += len(sample["plan"].groups)
                if any(
                    p.grad is None or not torch.isfinite(p.grad).all()
                    for p in self.model.parameters()
                ):
                    raise ValueError("Missing or nonfinite actual reader gradients")
                projection_square = torch.zeros((), dtype=torch.float64, device=self.device)
                for parameter in self.model.project.parameters():
                    if parameter.grad is None:
                        raise ValueError("Missing actual projection gradient")
                    projection_square += parameter.grad.detach().double().square().sum()
                projection_norm = float(projection_square.sqrt())
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.recipe.clip, error_if_nonfinite=True
                )
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
                if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                    raise ValueError("Nonfinite learned reader parameter")
                self.completed += 1
                self.cursor += 1
                self.history.append(
                    {
                        "step": self.completed,
                        "epoch": self.epoch + 1,
                        "studies": len(indices),
                        "loss": loss_value,
                        "lr": rate,
                        "gradient_norm": float(norm),
                        "projection_gradient_norm": projection_norm,
                        "max_abs_correction": maximum,
                        "qualified_relation_groups": relations,
                    }
                )
                finished_epoch = self.cursor == self.batches
                if finished_epoch:
                    self.epoch += 1
                    self.cursor = 0
                    target = output / "epoch-checkpoints" / f"epoch-{self.epoch:03d}.pt"
                    if target.exists():
                        raise ValueError("Immutable completed reader epoch already exists")
                    self.save(target)
                atomic_json(
                    output / "progress.json",
                    {
                        "completed": self.completed,
                        "total": self.total,
                        "last_update": self.history[-1],
                    },
                )
                stopped = self.completed == stop_after or (
                    should_stop is not None and should_stop()
                )
                if finished_epoch or stopped:
                    self.save(checkpoint)
                if stopped:
                    break
        return {
            "status": "fixed_final_spatial_fit_complete"
            if self.completed == self.total
            else "stopped_at_complete_update",
            "completed": self.completed,
            "total": self.total,
        }

    @torch.no_grad()
    def predictions(self, data):
        previous = self.model.training
        self.model.eval()
        rows = []
        try:
            for index in range(len(data)):
                sample = data[index]
                logits, _ = self.model(
                    sample["maps"].to(self.device),
                    sample["base_logits"].to(self.device),
                    sample["plan"],
                    arm=self.arm,
                )
                probability = logits.sigmoid().float().cpu()
                if not torch.isfinite(probability).all():
                    raise ValueError("Nonfinite spatial-reader prediction")
                rows.append([sample["study_uid"], *probability.tolist()])
        finally:
            self.model.train(previous)
        return pd.DataFrame(rows, columns=["study_uid", *TARGETS])
