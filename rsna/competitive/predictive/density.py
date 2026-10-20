"""Sparse complete-roster support evidence for the user's predictive proposal.

The supplied dense mathematical prototype is retained as an independent oracle.
This implementation never forms support-by-region dense tensors. Fixed priors
retain all unobserved alternatives. No claim of calibrated clinical likelihood.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch.nn import functional as F


@dataclass(frozen=True)
class Supports:
    indptr: torch.Tensor
    regions: torch.Tensor
    log_prior: torch.Tensor
    region_count: int

    def validate(self, device):
        p, i, w = self.indptr, self.regions, self.log_prior
        if (
            type(self.region_count) is not int
            or self.region_count < 1
            or p.ndim != 1
            or len(p) < 2
            or p.dtype != torch.int64
            or i.ndim != 1
            or i.dtype != torch.int64
            or w.ndim != 2
            or w.shape[1] != len(p) - 1
            or w.shape[0] < 1
            or w.requires_grad
            or any(x.device != device for x in (p, i, w))
            or int(p[0]) != 0
            or int(p[-1]) != len(i)
            or (p[1:] <= p[:-1]).any()
            or (i < 0).any()
            or (i >= self.region_count).any()
            or not torch.isfinite(w).all()
        ):
            raise ValueError(
                "Finite fixed complete-roster CSR supports and geometry-only priors required"
            )
        owners = torch.repeat_interleave(torch.arange(len(p) - 1, device=device), p[1:] - p[:-1])
        if ((owners[1:] == owners[:-1]) & (i[1:] <= i[:-1])).any():
            raise ValueError(
                "Every support must have sorted unique regions, not repeated pixel evidence"
            )
        return owners


@dataclass(frozen=True)
class Evidence:
    log_p0: torch.Tensor
    log_p1: torch.Tensor
    log_bayes_factor: torch.Tensor
    local_log_ratio: torch.Tensor
    no_observation_prior_mass: torch.Tensor


def evidence(
    targets,
    negative_mean,
    finding_shift,
    fixed_log_variance,
    supports,
    observed,
    *,
    edge_chunk=16384,
):
    if (
        not isinstance(supports, Supports)
        or targets.ndim != 3
        or negative_mean.ndim != 4
        or finding_shift.shape != negative_mean.shape
        or targets.requires_grad
        or targets.dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError("Detached descriptor targets and FP32/64 conditional predictions required")
    B, J, D = targets.shape
    C = negative_mean.shape[1]
    if (
        min(B, J, D, C) < 1
        or negative_mean.shape != (B, C, J, D)
        or fixed_log_variance.shape != (C, D)
        or fixed_log_variance.requires_grad
        or supports.region_count != J
        or supports.log_prior.shape[0] != C
        or observed.shape != (B, J)
        or observed.dtype != torch.bool
        or type(edge_chunk) is not int
        or edge_chunk < 1
    ):
        raise ValueError("Aligned complete-roster distributions and fixed shared variance required")
    values = (negative_mean, finding_shift, fixed_log_variance, supports.log_prior)
    if (
        any(
            v.dtype != targets.dtype or v.device != targets.device or not torch.isfinite(v).all()
            for v in values
        )
        or observed.device != targets.device
        or not torch.isfinite(targets[observed]).all()
        or (fixed_log_variance.abs() > 20).any()
    ):
        raise ValueError("Aligned finite masked descriptor arithmetic required")
    owners = supports.validate(targets.device)
    residual = targets.masked_fill(~observed[..., None], 0)[:, None] - negative_mean
    inv = (-fixed_log_variance).exp()[None, :, None]
    local = (
        (finding_shift * residual * inv - 0.5 * finding_shift.square() * inv)
        .sum(-1)
        .masked_fill(~observed[:, None], 0)
    )
    logp0 = (
        (
            -0.5
            * (
                residual.square() * inv + fixed_log_variance[None, :, None] + math.log(2 * math.pi)
            ).sum(-1)
        )
        .masked_fill(~observed[:, None], 0)
        .sum(-1)
    )
    K = supports.log_prior.shape[1]
    scores = local.new_zeros(B, C, K)
    touched = torch.zeros(B, K, device=targets.device, dtype=torch.int64)
    for start in range(0, len(owners), edge_chunk):
        region = supports.regions[start : start + edge_chunk]
        owner = owners[start : start + edge_chunk]
        scores = scores.index_add(2, owner, local.index_select(2, region))
        touched.index_add_(1, owner, observed.index_select(1, region).long())
    prior = supports.log_prior.log_softmax(-1)
    logbf = (scores + prior[None]).logsumexp(-1)
    missing = (prior.exp()[None] * (touched == 0)[:, None]).sum(-1)
    if any(not torch.isfinite(x).all() for x in (logp0, logbf)):
        raise FloatingPointError("Nonfinite support-mixture density; no clipping or fallback")
    return Evidence(logp0, logp0 + logbf, logbf, local, missing)


def two_mask_loss(
    evidences,
    context_logits,
    targets,
    label_mask,
    observations,
    descriptor_width=64,
    density_weight=0.1,
):
    """BCE of the MEAN posterior, not mean per-context BCE or multiplied odds."""
    if (
        context_logits.ndim != 3
        or context_logits.shape[1] != 2
        or len(evidences) != 2
        or len(observations) != 2
        or targets.shape != (context_logits.shape[0], context_logits.shape[2])
        or label_mask.shape != targets.shape
        or label_mask.dtype != torch.bool
        or type(descriptor_width) is not int
        or descriptor_width < 1
        or not math.isfinite(density_weight)
        or density_weight < 0
    ):
        raise ValueError(
            "Two complementary contexts, unchanged weak labels/masks and a fixed objective required"
        )
    if (
        not torch.isfinite(context_logits).all()
        or not torch.isfinite(targets[label_mask]).all()
        or ((targets[label_mask] < 0) | (targets[label_mask] > 1)).any()
    ):
        raise ValueError("Finite context predictions and observed weak probabilities required")
    z = context_logits + torch.stack([e.log_bayes_factor for e in evidences], 1)
    q = torch.where(label_mask, targets.detach(), 0)
    logp = torch.logsumexp(F.logsigmoid(z), 1) - math.log(2)
    log1p = torch.logsumexp(F.logsigmoid(-z), 1) - math.log(2)
    exam = (-(q * logp + (1 - q) * log1p)).masked_fill(
        ~label_mask, 0
    ).sum() / label_mask.sum().clamp_min(1)
    density = z.new_zeros(())
    valid = z.new_zeros(())
    for e, observed in zip(evidences, observations):
        count = observed.sum(1, keepdim=True) * descriptor_width
        active = label_mask & (count > 0)
        nll = -(q * e.log_p1 + (1 - q) * e.log_p0) / count.clamp_min(1)
        density = density + nll.masked_fill(~active, 0).sum()
        valid = valid + active.sum()
    density = density / valid.clamp_min(1)
    return exam + density_weight * density, {
        "exam_loss": exam,
        "density_loss": density,
        "probabilities": logp.exp(),
    }
