"""Study-pair ranking contrast, retaining the frozen MRI teacher's anchor.

Only the fitting objective changes. The parent's exposed training cohort cannot
provide independent validation. This is a weighted RankNet-style surrogate, not
the LibAUC min-max algorithm or an assurance of better competition accuracy.
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

VERSION = "licensed_study_pair_ranking_context_trainer_v1"
ORIGINAL_TRAINER_SHA = "5addea8212501ba31021c739543a7fa627cf70066ad439fa0889ba5c5b3fa59f"


@dataclass(frozen=True)
class Recipe(OriginalRecipe):
    objective: str = "source_rank"
    rank_fraction: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if self.objective not in ("uniform", "source_rank") or (
            type(self.rank_fraction) not in (int, float)
            or not math.isfinite(self.rank_fraction)
            or not 0 <= self.rank_fraction <= 1
        ):
            raise ValueError("Explicit objective and finite 0..1 ranking fraction required")


def ranking_loss(logits, parent_logits, labels, mask, *, source_weight, rank_fraction=0.5):
    """Replace half the observed source BCE by a per-target study-pair risk.

    Consecutive rows are the two anatomical views of a distinct study. Rank the
    inference-time mean of sigmoid probabilities, with stable log-odds scores.
    Pair weight is max(y_i-y_j,0); exact source ties do not order studies. Masks
    exclude pairs with either source target absent. Normalize each target by its
    total positive pair weight, then average targets, preserving the old source
    observation coverage. No pairs (including a singleton final batch) keeps the
    exact old loss for that target. The teacher contribution remains unchanged.

    Fractional labels supply ordering evidence, not calibrated probabilities of
    clinical truth. Pairwise gradients can amplify incorrect weak supervision.
    """
    if (
        logits.ndim != 2
        or not logits.shape[0]
        or logits.shape[0] % 2
        or not logits.shape[1]
        or any(x.shape != logits.shape for x in (parent_logits, labels, mask))
        or any(x.dtype != torch.float32 for x in (logits, parent_logits, labels))
        or mask.dtype != torch.bool
        or any(x.device != logits.device for x in (parent_logits, labels, mask))
        or type(rank_fraction) not in (int, float)
        or not math.isfinite(rank_fraction)
        or not 0 <= rank_fraction <= 1
    ):
        raise ValueError("Aligned FP32 paired views and finite 0..1 ranking fraction required")
    base = anchored_loss(logits, parent_logits, labels, mask, source_weight=source_weight)
    y = torch.where(mask, labels.detach(), 0.0).reshape(-1, 2, logits.shape[1])
    m = mask.reshape(-1, 2, logits.shape[1])
    if not torch.equal(m[:, 0], m[:, 1]) or not torch.equal(y[:, 0], y[:, 1]):
        raise ValueError("Both views must carry the same original study labels and masks")
    if rank_fraction == 0 or len(y) == 1:
        return base
    z = logits.reshape(-1, 2, logits.shape[1])
    # log(mean(sigmoid(z))) - log(mean(1-sigmoid(z))); log(2) cancels.
    scores = torch.logsumexp(F.logsigmoid(z), 1) - torch.logsumexp(F.logsigmoid(-z), 1)
    y, m = y[:, 0], m[:, 0]
    pair_weight = (y[:, None] - y[None, :]).clamp_min(0)
    pair_weight = pair_weight * (m[:, None] & m[None, :])
    mass = pair_weight.sum((0, 1))
    active = mass > 0
    if not active.any():
        return base
    pair_risk = (pair_weight * F.softplus(-(scores[:, None] - scores[None, :]))).sum(
        (0, 1)
    ) / mass.clamp_min(torch.finfo(logits.dtype).tiny)
    observed = torch.where(mask, labels.detach(), 0.0)
    point_risk = (
        F.binary_cross_entropy_with_logits(logits, observed, reduction="none") * mask
    ).mean(0)
    coverage = mask.float().mean(0)
    replacement = torch.where(active, coverage * pair_risk - point_risk, 0.0).mean()
    return base + source_weight * rank_fraction * replacement


class Trainer(OriginalTrainer):
    """Frozen model/optimizer/order; separate versioned ranking loss and recovery."""

    def __init__(self, dataset, recipe):
        if not isinstance(recipe, Recipe):
            raise ValueError("Versioned ranking-experiment recipe required")
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
            if self.recipe.objective == "uniform":
                loss = anchored_loss(
                    logits, x[1], labels, masks, source_weight=self.recipe.source_weight
                )
            else:
                loss = ranking_loss(
                    logits,
                    x[1],
                    labels,
                    masks,
                    source_weight=self.recipe.source_weight,
                    rank_fraction=self.recipe.rank_fraction,
                )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe.clip)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError("Nonfinite ranking-context optimizer update")
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                raise ValueError("Nonfinite learned ranking-context parameter")
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
