"""Gap-conditioned, variable-length latent paths; mathematical prototype only.

Inputs are latent evidence, not clinical segmentations. A path may stop after
one window. Separate chains must later have disjoint original source support;
mixture over chains must never become independent voting by overlapping views.
No image loader, encoder, optimizer, checkpoint or competition runner lives here.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import functional as F

# Unit chosen from fitting-only metadata, not a clinical lesion-size threshold.
# CPU951202: median dense disjoint-source repeat center gap in2247 fitting exams.
REFERENCE_GAP_MM = 9.900010275666745


def _joint(joint: Tensor) -> tuple[Tensor, Tensor]:
    if (
        joint.ndim != 2
        or joint.shape[0] != joint.shape[1]
        or not len(joint)
        or joint.dtype not in (torch.float32, torch.float64)
        or not torch.isfinite(joint).all()
        or (joint < 0).any()
        or joint.requires_grad
    ):
        raise ValueError("A finite fixed nonnegative square spatial joint is required")
    tolerance = 1e-6 if joint.dtype == torch.float32 else 1e-12
    if not torch.allclose(joint, joint.T, atol=tolerance, rtol=0):
        raise ValueError(
            "Reversible spatial prior required; arbitrary directed transport is separate"
        )
    total = joint.sum()
    if total <= 0:
        raise ValueError("Spatial prior cannot be empty")
    normalized = joint / total
    marginal = normalized.sum(1)
    if (marginal <= 0).any():
        raise ValueError("Remove padding/zero-mass tokens before constructing a path")
    return normalized, marginal


def factorized_joint(joint: Tensor) -> Tensor:
    """Same stationary weighted token marginal, independently redrawn each step."""
    _, marginal = _joint(joint)
    return marginal[:, None] * marginal[None, :]


def log_path_partition(
    unary: Tensor,
    joint: Tensor,
    gaps_mm: Tensor,
    continuation_logits: Tensor,
    *,
    reference_gap_mm: float = REFERENCE_GAP_MM,
) -> Tensor:
    """Log E_prior exp(sum of evidence along one possibly focal latent path).

    Start window is uniform; start token follows the spatial stationary measure.
    An edge continues with sigmoid(kappa - log(gap/reference_gap)). A last window
    forces termination; every earlier window can terminate. Thus zero evidence
    has total prior mass1 for any sequence length, gap, or continuation setting.
    The fixed transition is joint[p,q]/marginal[p]. It remains stationary under
    both the physical joint and its rank-one matched control.
    """
    normalized, marginal = _joint(joint)
    if (
        unary.ndim != 3
        or not unary.shape[0]
        or not unary.shape[2]
        or unary.shape[1] != len(marginal)
        or unary.dtype != joint.dtype
        or unary.device != joint.device
        or not torch.isfinite(unary).all()
        or gaps_mm.shape != (unary.shape[0] - 1,)
        or gaps_mm.dtype != unary.dtype
        or gaps_mm.device != unary.device
        or not torch.isfinite(gaps_mm).all()
        or (gaps_mm <= 0).any()
        or continuation_logits.shape != (unary.shape[2],)
        or continuation_logits.dtype != unary.dtype
        or continuation_logits.device != unary.device
        or not torch.isfinite(continuation_logits).all()
        or not math.isfinite(reference_gap_mm)
        or reference_gap_mm <= 0
    ):
        raise ValueError("Complete finite window/token/target evidence and physical gaps required")
    tiny = torch.finfo(unary.dtype).tiny
    log_transition = normalized.clamp_min(tiny).log() - marginal.log()[:, None]
    log_transition = log_transition.masked_fill(normalized == 0, -torch.inf)
    log_start = marginal.log()[:, None]
    value = unary[-1]
    starts = [torch.logsumexp(log_start + value, dim=0)]
    for t in range(len(unary) - 2, -1, -1):
        logit = continuation_logits - (gaps_mm[t] / reference_gap_mm).log()
        future = torch.logsumexp(log_transition[:, :, None] + value[None, :, :], dim=1)
        value = unary[t] + torch.logaddexp(F.logsigmoid(-logit), F.logsigmoid(logit) + future)
        starts.append(torch.logsumexp(log_start + value, dim=0))
    return (
        torch.logsumexp(torch.stack(starts), dim=0) - math.log(len(unary)) + continuation_logits * 0
    )


def witness_score(unary, joint, gaps_mm, continuation_logits, *, reference_gap_mm=REFERENCE_GAP_MM):
    """Subtract the numerical zero-evidence baseline; logits remain unbounded."""
    kwargs = {"reference_gap_mm": reference_gap_mm}
    actual = log_path_partition(unary, joint, gaps_mm, continuation_logits, **kwargs)
    baseline = log_path_partition(
        torch.zeros_like(unary), joint, gaps_mm, continuation_logits, **kwargs
    )
    return actual - baseline


def acquisition_mixture(chains, continuation_logits, *, reference_gap_mm=REFERENCE_GAP_MM):
    """A mixture over possible starts; overlapping phases are never multiplied.

    Caller must separately prove original source membership, chronological order
    and non-overlap WITHIN each chain. This numerical layer does not inspect SOPs
    and cannot accept an MRI input policy or provide clinical localization.
    """
    if not chains:
        raise ValueError("At least one source-qualified chain is required")
    values, lengths = [], []
    for unary, joint, gaps in chains:
        values.append(
            witness_score(
                unary, joint, gaps, continuation_logits, reference_gap_mm=reference_gap_mm
            )
            + math.log(len(unary))
        )
        lengths.append(len(unary))
    return torch.logsumexp(torch.stack(values), dim=0) - math.log(sum(lengths))
