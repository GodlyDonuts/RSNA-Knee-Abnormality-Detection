"""Paired-view, between-study teacher ranking for a complete trainable encoder.

This is a weighted RankNet contrast on detached teacher order, not clinical
truth, an unbiased AUC estimator or LibAUC's min-max algorithm. Original observed
source labels/masks retain their pointwise contribution. Live trainers are not
modified; a separate source-bound wrapper must admit any actual experiment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from rsna.competitive.licensed_importance import Recipe as BaseRecipe
from rsna.competitive.licensed_importance import sample
from rsna.competitive.released_context import anchored_loss

VERSION = "full_encoder_two_view_teacher_pair_risk_v1"


@dataclass(frozen=True)
class Recipe(BaseRecipe):
    role: str = "influence"
    objective: str = "pointwise"
    rank_fraction: float = 0.5

    def __post_init__(self):
        super().__post_init__()
        if (
            self.role != "influence"
            or self.objective not in ("pointwise", "teacher_rank")
            or type(self.rank_fraction) is not float
            or self.rank_fraction != 0.5
        ):
            raise ValueError(
                "Matched fixed16 influence and predeclared half-teacher ranking required"
            )


def ranking_loss(logits, teacher_logits, labels, masks, *, source_weight=0.25, fraction=0.5):
    """Replace a fixed fraction of teacher BCE by per-target teacher-pair risk.

    Rows are consecutive opposite anatomical views of each distinct study.
    Teacher pair weights are max(q_i-q_j,0), where q is mean view probability.
    Student order uses stable log-odds of its mean view probability. Pair risk is
    normalized per target, weighted by that target's original teacher coverage,
    and averaged across all targets. Original observed-source BCE stays intact.
    No teacher pairs means exact original BCE for that target; no invented order
    is assigned to ties. Detached teacher scores are not diagnostic labels.
    """
    if (
        logits.ndim != 2
        or not logits.shape[0]
        or logits.shape[0] % 2
        or not logits.shape[1]
        or any(x.shape != logits.shape for x in (teacher_logits, labels, masks))
        or any(x.dtype != torch.float32 for x in (logits, teacher_logits, labels))
        or masks.dtype != torch.bool
        or any(x.device != logits.device for x in (teacher_logits, labels, masks))
        or type(fraction) not in (int, float)
        or not math.isfinite(fraction)
        or not 0 <= fraction <= 1
    ):
        raise ValueError("Finite aligned FP32 paired views and a 0..1 ranking fraction required")
    base = anchored_loss(logits, teacher_logits, labels, masks, source_weight=source_weight)
    width = logits.shape[1]
    m = masks.reshape(-1, 2, width)
    observed = torch.where(masks, labels.detach(), 0).reshape(-1, 2, width)
    if not torch.equal(m[:, 0], m[:, 1]) or not torch.equal(observed[:, 0], observed[:, 1]):
        raise ValueError("Two views must retain identical original labels and masks")
    if fraction == 0 or len(m) == 1:
        return base
    teacher = teacher_logits.detach().sigmoid()
    q = teacher.reshape(-1, 2, width).mean(1)
    pair_weight = (q[:, None] - q[None, :]).clamp_min(0)
    mass = pair_weight.sum((0, 1))
    active = mass > 0
    if not active.any():
        return base
    z = logits.reshape(-1, 2, width)
    score = torch.logsumexp(F.logsigmoid(z), 1) - torch.logsumexp(F.logsigmoid(-z), 1)
    risk = (pair_weight * F.softplus(-(score[:, None] - score[None, :]))).sum((0, 1))
    risk = risk / mass.clamp_min(torch.finfo(logits.dtype).tiny)
    coefficient = 1 - source_weight * masks.float()
    teacher_point = (
        coefficient * F.binary_cross_entropy_with_logits(logits, teacher, reduction="none")
    ).mean(0)
    replacement = torch.where(active, coefficient.mean(0) * risk - teacher_point, 0).mean()
    return base + fraction * replacement


def coupled_batch(model, data, indices, recipe, epoch, device):
    """Retain only checkpointed per-case graphs until the complete study-batch loss.

    Uses existing IID16 draws, the same positions for both anatomical views, and
    the original sampled view first. No second cache/replay forward is added.
    The real wrapper must instantiate the unchanged checkpointed encoder and
    measure actual peak memory; this helper cannot establish GPU feasibility.
    """
    if (
        not isinstance(recipe, Recipe)
        or not indices
        or len(indices) > recipe.effective_batch
        or len(indices) != len(set(indices))
        or any(type(i) is not int or not 0 <= i < len(data) for i in indices)
    ):
        raise ValueError("Nonempty distinct original studies within the effective batch required")
    outputs, teachers, labels, masks = [], [], [], []
    image_bytes = 0
    for index in indices:
        positions, first_view, logq = sample(data.uids[index], recipe, epoch, data.proposal(index))
        for mirror in (first_view, not first_view):
            case = data.case(index, positions, mirror)
            images = case["images"]
            if (
                not isinstance(images, torch.Tensor)
                or images.dtype != torch.float32
                or images.ndim != 4
                or images.shape[0] != 16
                or not torch.isfinite(images).all()
            ):
                raise ValueError("Original finite FP32 selected16-window tensors required")
            image_bytes += images.numel() * images.element_size()
            if image_bytes > 1024**3:
                raise ValueError("Explicit one-GiB total retained image input bound exceeded")
            with torch.autocast(
                device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
            ):
                logits = model(images.to(device).unsqueeze(0), logq.to(device).unsqueeze(0)).float()
            if logits.shape != (1, 12) or not torch.isfinite(logits).all():
                raise ValueError("Finite original one-study twelve-target logits required")
            outputs.append(logits)
            teachers.append(case["parent_logits"].to(device).reshape(1, 12).float())
            labels.append(case["labels"].to(device).reshape(1, 12).float())
            masks.append(case["mask"].to(device).reshape(1, 12))
    args = tuple(torch.cat(group) for group in (outputs, teachers, labels, masks))
    loss = (
        anchored_loss(*args, source_weight=recipe.source_weight)
        if recipe.objective == "pointwise"
        else ranking_loss(*args, source_weight=recipe.source_weight, fraction=recipe.rank_fraction)
    )
    return loss, {
        "studies": len(indices),
        "anatomical_views": 2 * len(indices),
        "retained_image_input_bytes": image_bytes,
    }
