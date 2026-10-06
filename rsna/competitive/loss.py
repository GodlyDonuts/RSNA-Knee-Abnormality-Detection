"""Normalize sparse supervision over a whole effective batch, including DDP."""

from __future__ import annotations

import torch
import torch.distributed as dist
from torch import Tensor
from torch.nn import functional as F


def observed_counts(masks: list[Tensor]) -> Tensor:
    """Collect once per optimizer update, before any microbatch backward pass."""
    if not masks:
        raise ValueError("An effective batch must contain a microbatch")
    for mask in masks:
        if mask.ndim != 2 or not torch.isfinite(mask).all() or (mask < 0).any():
            raise ValueError("Supervision weights must be finite and nonnegative")
    counts = torch.stack([mask.double().sum(dim=0) for mask in masks]).sum(dim=0)
    if dist.is_initialized():
        dist.all_reduce(counts)
    return counts


def effective_batch_loss(logits: Tensor, labels: Tensor, mask: Tensor, counts: Tensor) -> Tensor:
    """One microbatch's contribution to the global per-target mean BCE.

    Sum these losses across accumulation steps, without dividing again by the
    accumulation count. DDP averages gradients, hence the world-size multiplier.
    All-masked global batches must skip the optimizer too (including weight decay).
    """
    if logits.shape != labels.shape or logits.shape != mask.shape or logits.ndim != 2:
        raise ValueError("Expected aligned [study,target] logits, labels and masks")
    if counts.shape != logits.shape[1:] or not torch.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("Invalid effective-batch observed counts")
    if not torch.isfinite(mask).all() or (mask < 0).any():
        raise ValueError("Invalid supervision weights")
    active = mask > 0
    if (
        not torch.isfinite(labels[active]).all()
        or ((labels[active] < 0) | (labels[active] > 1)).any()
    ):
        raise ValueError("Observed labels must be probabilities")
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits")
    # Unobserved labels have no mathematical role, even if their placeholder is NaN.
    safe_labels = torch.where(active, labels, torch.zeros_like(labels))
    sums = (
        F.binary_cross_entropy_with_logits(logits.float(), safe_labels.float(), reduction="none")
        * mask
    ).sum(dim=0)
    included = counts > 0
    if not included.any():
        return logits.sum() * 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    return world * (sums[included] / counts[included].to(sums.dtype)).mean()


def masked_cell_entropy(logits: Tensor, mask: Tensor, masked_counts: Tensor) -> Tensor:
    """One microbatch's contribution to the global per-target mean binary
    entropy over unobserved cells (mask == 0), mirroring effective_batch_loss
    normalization: per-target sums are divided by the effective-batch masked
    counts collected once per optimizer update."""
    if logits.shape != mask.shape or logits.ndim != 2:
        raise ValueError("Expected aligned [study,target] logits and masks")
    if (
        masked_counts.shape != logits.shape[1:]
        or not torch.isfinite(masked_counts).all()
        or (masked_counts < 0).any()
    ):
        raise ValueError("Invalid effective-batch masked counts")
    if not torch.isfinite(mask).all() or (mask < 0).any():
        raise ValueError("Invalid supervision weights")
    if not torch.isfinite(logits).all():
        raise ValueError("Non-finite logits")
    unobserved = (mask == 0).to(mask.dtype)
    sums = (
        F.binary_cross_entropy_with_logits(
            logits.float(), torch.full_like(logits, 0.5).float(), reduction="none"
        )
        * unobserved
    ).sum(dim=0)
    included = masked_counts > 0
    if not included.any():
        return logits.sum() * 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    return world * (sums[included] / masked_counts[included].to(sums.dtype)).mean()


def masked_cell_consistency(
    student_logits: Tensor, teacher_logits: Tensor, mask: Tensor, masked_counts: Tensor
) -> Tensor:
    """One microbatch's contribution to the global per-target mean squared
    error between student and EMA-teacher logits over unobserved cells
    (mask == 0), normalized like masked_cell_entropy."""
    if (
        student_logits.shape != teacher_logits.shape
        or student_logits.shape != mask.shape
        or student_logits.ndim != 2
    ):
        raise ValueError("Expected aligned [study,target] student/teacher logits and masks")
    if (
        masked_counts.shape != student_logits.shape[1:]
        or not torch.isfinite(masked_counts).all()
        or (masked_counts < 0).any()
    ):
        raise ValueError("Invalid effective-batch masked counts")
    if not torch.isfinite(mask).all() or (mask < 0).any():
        raise ValueError("Invalid supervision weights")
    if not torch.isfinite(student_logits).all() or not torch.isfinite(teacher_logits).all():
        raise ValueError("Non-finite logits")
    unobserved = (mask == 0).to(mask.dtype)
    sums = ((student_logits.float() - teacher_logits.float()).pow(2) * unobserved).sum(dim=0)
    included = masked_counts > 0
    if not included.any():
        return student_logits.sum() * 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    return world * (sums[included] / masked_counts[included].to(sums.dtype)).mean()


def adjacent_slice_pairs(slice_masks: list[Tensor]) -> Tensor:
    """Collect once per optimizer update: total consecutive valid-slice pairs.

    Slices are stored in strictly increasing physical order along each series
    axis, so the k-index adjacency of two valid slices is physical adjacency.
    """
    if not slice_masks:
        raise ValueError("An effective batch must contain a microbatch")
    pairs = torch.zeros((), dtype=torch.float64, device=slice_masks[0].device)
    for mask in slice_masks:
        if mask.ndim != 3 or mask.dtype != torch.bool:
            raise ValueError("Slice masks must be boolean [study,series,slice]")
        pairs = pairs + (mask[:, :, 1:] & mask[:, :, :-1]).sum()
    if dist.is_initialized():
        dist.all_reduce(pairs)
    return pairs


def slice_score_tv(slice_scores: Tensor, slice_mask: Tensor, pair_count: Tensor) -> Tensor:
    """One microbatch's contribution to the global per-target mean squared
    difference between the gated attention scores of physically adjacent
    slices (SmMIL-style smoothness), mirroring effective_batch_loss
    normalization: per-target sums are divided by the effective-batch
    adjacent-pair count collected once per optimizer update."""
    if (
        slice_scores.ndim != 4
        or slice_mask.shape != slice_scores.shape[:3]
        or slice_mask.dtype != torch.bool
    ):
        raise ValueError("Expected aligned [study,series,slice,target] scores and bool mask")
    if pair_count.ndim != 0 or not torch.isfinite(pair_count) or (pair_count < 0).any():
        raise ValueError("Invalid effective-batch adjacent-pair count")
    if not torch.isfinite(slice_scores.float()).all():
        raise ValueError("Non-finite slice scores")
    pairs = (slice_mask[:, :, 1:] & slice_mask[:, :, :-1]).to(slice_scores.dtype)
    diffs = (slice_scores[:, :, 1:, :].float() - slice_scores[:, :, :-1, :].float()).pow(2)
    sums = (diffs * pairs[..., None]).sum(dim=(0, 1, 2))
    if pair_count.item() <= 0:
        return slice_scores.sum() * 0
    world = dist.get_world_size() if dist.is_initialized() else 1
    return world * (sums / pair_count.to(sums.dtype)).mean()


def validate_masked_cell(policy: object) -> None:
    """Masked-cell objective contract: a config-bound arm, not an artifact."""
    if not isinstance(policy, dict):
        raise ValueError("Masked-cell policy must be an object")
    if type(policy.get("enabled")) is not bool:
        raise ValueError("Masked-cell enabled flag must be a boolean")
    kind = policy.get("kind")
    if kind == "entropy_max":
        if sorted(policy) != ["enabled", "kind", "weight"]:
            raise ValueError("Masked-cell policy fields are enabled, kind and weight")
    elif kind == "ema_consistency":
        if sorted(policy) != ["alpha", "enabled", "kind", "ramp_fraction", "weight"]:
            raise ValueError(
                "Masked-cell ema_consistency fields are alpha, enabled, kind, ramp_fraction and weight"
            )
        alpha = policy["alpha"]
        if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not (0 < alpha < 1):
            raise ValueError("Masked-cell alpha must be in (0, 1)")
        ramp = policy["ramp_fraction"]
        if isinstance(ramp, bool) or not isinstance(ramp, (int, float)) or not (0 < ramp <= 1):
            raise ValueError("Masked-cell ramp_fraction must be in (0, 1]")
    else:
        raise ValueError("Unknown masked-cell objective kind")
    weight = policy["weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not (0 < weight <= 1):
        raise ValueError("Masked-cell weight must be in (0, 1]")


def validate_slice_smooth(policy: object) -> None:
    """Slice-smoothness objective contract: a config-bound arm, not an artifact."""
    if not isinstance(policy, dict) or sorted(policy) != ["enabled", "kind", "weight"]:
        raise ValueError("Slice-smooth policy fields are enabled, kind and weight")
    if type(policy["enabled"]) is not bool:
        raise ValueError("Slice-smooth enabled flag must be a boolean")
    if policy["kind"] != "attention_score_tv":
        raise ValueError("Unknown slice-smooth objective kind")
    weight = policy["weight"]
    if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not (0 < weight <= 1):
        raise ValueError("Slice-smooth weight must be in (0, 1]")
