"""Nonhomogeneous latent paths and an exact start-conditioned marginal null.

Metadata transitions may change and need not be reversible. The independent
control preserves each position's physical-prior marginal CONDITIONAL ON EACH
POSSIBLE START, rather than assuming stationarity or changing image evidence.
This is a mathematical layer, not a qualified MRI predictor or registration.
"""

from __future__ import annotations

import math

import torch
from prototype import REFERENCE_GAP_MM
from torch import Tensor
from torch.nn import functional as F


def inputs(unary, joints, starts, gaps, kappa, reference_gap_mm):
    if unary.ndim != 3 or min(unary.shape) < 1:
        raise ValueError("Nonempty window/token/target evidence required")
    length, tokens, diagnoses = unary.shape
    if (
        unary.dtype not in (torch.float32, torch.float64)
        or not torch.isfinite(unary).all()
        or joints.shape != (length - 1, tokens, tokens)
        or starts.shape != (length, tokens)
        or gaps.shape != (length - 1,)
        or kappa.shape != (diagnoses,)
        or any(
            x.device != unary.device or x.dtype != unary.dtype
            for x in (joints, starts, gaps, kappa)
        )
        or any(x.requires_grad for x in (joints, starts, gaps))
        or not all(torch.isfinite(x).all() for x in (joints, starts, gaps, kappa))
        or (joints < 0).any()
        or (starts < 0).any()
        or (gaps <= 0).any()
        or not math.isfinite(reference_gap_mm)
        or reference_gap_mm <= 0
    ):
        raise ValueError("Fixed finite source priors and positive physical gaps required")
    tolerance = 1e-6 if unary.dtype == torch.float32 else 1e-12
    if not torch.allclose(starts.sum(1), torch.ones_like(starts[:, 0]), atol=tolerance, rtol=0):
        raise ValueError("Every original start needs its complete token measure")
    row_mass = joints.sum(2, keepdim=True)
    if not torch.isfinite(row_mass).all() or (row_mass <= 0).any():
        raise ValueError("Unobserved outgoing states need an explicit separate policy")
    transitions = joints / row_mass
    tiny = torch.finfo(unary.dtype).tiny
    log_transitions = transitions.clamp_min(tiny).log().masked_fill(transitions == 0, -torch.inf)
    log_starts = starts.clamp_min(tiny).log().masked_fill(starts == 0, -torch.inf)
    edge_logits = kappa[None, :] - (gaps[:, None] / reference_gap_mm).log()
    return (
        transitions,
        log_transitions,
        log_starts,
        F.logsigmoid(edge_logits),
        F.logsigmoid(-edge_logits),
    )


def conditional_marginals(transitions: Tensor, starts: Tensor) -> Tensor:
    """m[s,t,p]: exact token measure after continuing from original start s.

    No observed evidence/labels enter these measures. Invalid t<s slots are0.
    This transparent reference uses quadratic length storage/computation; actual
    MRI/dense-window throughput must be measured before any full fitting.
    """
    length, tokens = starts.shape
    if transitions.shape != (length - 1, tokens, tokens):
        raise ValueError("Complete original varying transition sequence required")
    if starts.requires_grad or transitions.requires_grad:
        raise ValueError("Conditional null is derived only from fixed source metadata")
    rows = []
    for first in range(length):
        marginal = starts[first]
        values = [torch.zeros_like(marginal) for _ in range(first)] + [marginal]
        for edge in range(first, length - 1):
            marginal = marginal @ transitions[edge]
            values.append(marginal)
        rows.append(torch.stack(values))
    return torch.stack(rows)


def log_partition(
    unary: Tensor,
    joints: Tensor,
    starts: Tensor,
    gaps: Tensor,
    kappa: Tensor,
    *,
    arm: str,
    reference_gap_mm: float = REFERENCE_GAP_MM,
) -> Tensor:
    transitions, log_transitions, log_starts, log_continue, log_stop = inputs(
        unary, joints, starts, gaps, kappa, reference_gap_mm
    )
    length = len(unary)
    if arm == "physical":
        value = unary[-1]
        values = [torch.logsumexp(log_starts[-1, :, None] + value, dim=0)]
        for t in range(length - 2, -1, -1):
            future = torch.logsumexp(log_transitions[t, :, :, None] + value[None, :, :], dim=1)
            value = unary[t] + torch.logaddexp(log_stop[t], log_continue[t] + future)
            values.append(torch.logsumexp(log_starts[t, :, None] + value, dim=0))
    elif arm == "independent":
        measures = conditional_marginals(transitions, starts)
        tiny = torch.finfo(unary.dtype).tiny
        values = []
        for first in range(length):
            marginal = measures[first, first:]
            log_marginal = marginal.clamp_min(tiny).log().masked_fill(marginal == 0, -torch.inf)
            emission = torch.logsumexp(log_marginal[:, :, None] + unary[first:], dim=1)
            value = emission[-1]
            for t in range(length - 2, first - 1, -1):
                value = emission[t - first] + torch.logaddexp(log_stop[t], log_continue[t] + value)
            values.append(value)
    else:
        raise ValueError("An explicit physical or start-conditioned independent arm is required")
    return torch.logsumexp(torch.stack(values), dim=0) - math.log(length) + kappa * 0


def witness_score(unary, joints, starts, gaps, kappa, *, arm, reference_gap_mm=REFERENCE_GAP_MM):
    options = {"arm": arm, "reference_gap_mm": reference_gap_mm}
    value = log_partition(unary, joints, starts, gaps, kappa, **options)
    baseline = log_partition(torch.zeros_like(unary), joints, starts, gaps, kappa, **options)
    return value - baseline
