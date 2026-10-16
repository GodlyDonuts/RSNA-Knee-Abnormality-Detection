"""Prospective source-confidence contrast on the original frozen MRI features.

The source calls p=.5 unaddressed. This experiment varies only its influence
relative to the detached MRI teacher, not the stored targets or observation mask.
There is no independent validation claim for the exposed licensed parent.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch.nn import functional as F

from rsna.competitive.released_context import anchored_loss
from rsna.competitive.released_context_train import Recipe as OriginalRecipe
from rsna.competitive.released_context_train import Trainer as OriginalTrainer
from rsna.competitive.released_context_train import batch, learning_rate, restore_rng

VERSION = "licensed_source_confidence_context_trainer_v1"
OBJECTIVES = ("uniform", "source_confidence")
ORIGINAL_TRAINER_SHA = "5addea8212501ba31021c739543a7fa627cf70066ad439fa0889ba5c5b3fa59f"


@dataclass(frozen=True)
class Recipe(OriginalRecipe):
    objective: str = "source_confidence"

    def __post_init__(self):
        super().__post_init__()
        if self.objective not in OBJECTIVES:
            raise ValueError("Explicit uniform or source_confidence objective required")


def confidence_loss(logits, parent_logits, labels, mask, *, source_weight):
    """Keep the old all-cell mean; return uncertain source influence to teacher.

    alpha=w*2*abs(y-.5) for observed cells, otherwise zero. The target is
    alpha*y+(1-alpha)*teacher. No sum-of-confidence normalization, class weighting,
    threshold or relabeling. This changes source gradient scale intentionally.
    """
    if (
        logits.shape != labels.shape
        or parent_logits.shape != logits.shape
        or mask.shape != logits.shape
        or logits.ndim != 2
        or min(logits.shape) < 1
        or mask.dtype != torch.bool
        or any(x.dtype != torch.float32 for x in (logits, parent_logits, labels))
        or any(x.device != logits.device for x in (parent_logits, labels, mask))
        or not torch.isfinite(logits).all()
        or not torch.isfinite(parent_logits).all()
        or not torch.isfinite(labels[mask]).all()
        or ((labels[mask] < 0) | (labels[mask] > 1)).any()
        or type(source_weight) not in (float, int)
        or not math.isfinite(source_weight)
        or not 0 < source_weight <= 1
    ):
        raise ValueError(
            "Aligned finite FP32 probabilities, masks and valid source weight required"
        )
    teacher = parent_logits.detach().sigmoid()
    # Remove unobserved NaNs before arithmetic, not after multiplying by zero.
    observed = torch.where(mask, labels.detach(), teacher)
    confidence = torch.where(mask, 2 * (observed - 0.5).abs(), 0.0)
    alpha = source_weight * confidence
    target = alpha * observed + (1 - alpha) * teacher
    return F.binary_cross_entropy_with_logits(logits, target.detach(), reduction="mean")


class Trainer(OriginalTrainer):
    """Original model/optimizer/ordering; explicit versioned objective and loop.

    The original engine is frozen. This subclass changes no module globals and
    rejects its checkpoints; uniform numerical parity is checked independently.
    """

    def __init__(self, dataset, recipe):
        if not isinstance(recipe, Recipe):
            raise ValueError("Versioned confidence-experiment recipe required")
        super().__init__(dataset, recipe)

    def state(self):
        value = super().state()
        value["version"] = VERSION
        value["original_trainer_sha256"] = ORIGINAL_TRAINER_SHA
        return value

    def load(self, path):
        value = torch.load(path, map_location="cpu", weights_only=False)
        if (
            value["version"] != VERSION
            or value["original_trainer_sha256"] != ORIGINAL_TRAINER_SHA
            or value["recipe"] != asdict(self.recipe)
            or value["binding"] != self.data.binding
            or value["population"] != len(self.data)
            or value["scheduler"]
            != {
                "kind": "linear_warmup_cosine_v1",
                "batches_per_epoch": self.steps,
                "completed": value["completed"],
            }
            or value["completed"] != value["epoch"] * self.steps + value["cursor"]
            or not 0 <= value["epoch"] <= self.recipe.epochs
            or not 0 <= value["cursor"] < self.steps
            or len(value["history"]) != value["completed"]
        ):
            raise ValueError("Checkpoint objective, recipe, input binding or update cursor changed")
        order = value["order"]
        if (value["cursor"] and order is None) or (
            order is not None and sorted(order) != list(range(len(self.data)))
        ):
            raise ValueError("Complete original epoch permutation required")
        self.model.load_state_dict(value["model"], strict=True)
        self.optimizer.load_state_dict(value["optimizer"])
        self.order_rng.set_state(value["order_rng"])
        self.epoch, self.cursor, self.completed = (
            value["epoch"],
            value["cursor"],
            value["completed"],
        )
        self.order, self.history = order, value["history"]
        restore_rng(value["rng"])

    def run(self, *, stop_after=None, epoch_directory=None, progress=None):
        if (
            stop_after is not None
            and not self.completed < stop_after <= self.steps * self.recipe.epochs
        ):
            raise ValueError("Future fixed update boundary required")
        self.model.train()
        while self.epoch < self.recipe.epochs:
            if self.order is None:
                self.order = torch.randperm(len(self.data), generator=self.order_rng).tolist()
            start = self.cursor * self.recipe.batch_size
            indices = self.order[start : start + self.recipe.batch_size]
            x, labels, masks = batch(self.data, indices)
            self.optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(self.recipe, self.completed, self.steps)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            logits, correction = self.model(*x)
            objective = anchored_loss if self.recipe.objective == "uniform" else confidence_loss
            loss = objective(logits, x[1], labels, masks, source_weight=self.recipe.source_weight)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe.clip)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError("Nonfinite confidence-context optimizer update")
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                raise ValueError("Nonfinite learned confidence-context parameter")
            self.completed += 1
            self.cursor += 1
            record = {
                "update": self.completed,
                "epoch": self.epoch + 1,
                "batch": self.cursor,
                "studies": len(indices),
                "lr": lr,
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "max_abs_correction": float(correction.detach().abs().max()),
            }
            self.history.append(record)
            if self.cursor == self.steps:
                self.epoch += 1
                self.cursor, self.order = 0, None
                if epoch_directory is not None:
                    self.save(Path(epoch_directory) / f"epoch-{self.epoch:03d}.pt")
            if progress is not None:
                progress(record)
            if stop_after == self.completed:
                break
