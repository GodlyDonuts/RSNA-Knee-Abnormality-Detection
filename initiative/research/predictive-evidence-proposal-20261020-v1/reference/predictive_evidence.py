"""CPU-tested mathematical core for a proposed knee-MRI research experiment.

Not a trained MRI model, production loader, or clinically validated classifier.
Assumption: non-overlapping descriptor regions are conditionally independent
within each proposed lesion support. The model marginalizes support location;
it does NOT relabel unobserved regions as healthy or renormalize support priors
to the observed subset. Predictive means must not see held-out target pixels.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F


@dataclass(frozen=True)
class EvidenceOutput:
    log_p0: Tensor              # [batch, class], descriptor density, not pixel density
    log_p1: Tensor              # [batch, class]
    log_bayes_factor: Tensor    # [batch, class]
    local_log_ratio: Tensor    # [batch, class, region]
    support_log_posterior: Tensor  # [batch, class, support], conditional on Y=1
    no_observation_prior_mass: Tensor  # [batch, class]


def predictive_evidence(
    targets: Tensor,           # [B,J,D], fixed observed descriptors
    negative_mean: Tensor,     # [B,C,J,D], predicted from visible context only
    finding_shift: Tensor,     # [B,C,J,D], positive-minus-negative predictive mean
    shared_log_variance: Tensor,  # [C,D], SAME variance under both hypotheses
    supports: Tensor,          # bool [C,K,J], defined over complete region roster
    support_logits: Tensor,    # [C,K], independent of the observation mask/targets
    observed: Tensor,          # bool [B,J]
) -> EvidenceOutput:
    """Marginal likelihood ratio with coverage-aware latent supports.

    For local evidence ell_j = log f1(t_j) - log f0(t_j),
      BF_O = sum_Z pi_Z exp(sum_{j in Z intersect O} ell_j).
    Supports outside O remain in the sum with their original prior mass.
    This identity is exact for this model; the model's MRI assumptions are unproven.
    """
    if targets.ndim != 3 or negative_mean.ndim != 4:
        raise ValueError("Expected targets [B,J,D] and means [B,C,J,D]")
    B, J, D = targets.shape
    if min(B, J, D) <= 0:
        raise ValueError("Non-empty batch, region roster, and descriptors required")
    if targets.requires_grad:
        raise ValueError("Targets must be fixed/detached, not a collapsing learned target")
    if negative_mean.shape[0] != B or negative_mean.shape[2:] != (J, D):
        raise ValueError("Predictive mean shape does not match target roster")
    C = negative_mean.shape[1]
    if C < 1 or finding_shift.shape != negative_mean.shape:
        raise ValueError("Aligned non-empty class hypotheses required")
    if shared_log_variance.shape != (C, D):
        raise ValueError("Shared class/descriptor variance must have shape [C,D]")
    if supports.ndim != 3 or supports.shape[0] != C or supports.shape[2] != J:
        raise ValueError("Support masks must have shape [C,K,J]")
    K = supports.shape[1]
    if K < 1 or supports.dtype != torch.bool or not supports.any(-1).all():
        raise ValueError("Each complete-roster support must be non-empty and boolean")
    if support_logits.shape != (C, K):
        raise ValueError("One prior log-weight per complete-roster support required")
    if observed.shape != (B, J) or observed.dtype != torch.bool:
        raise ValueError("Boolean [B,J] observation mask required")
    floats = (targets, negative_mean, finding_shift, shared_log_variance, support_logits)
    if targets.dtype not in (torch.float32, torch.float64):
        raise ValueError("Use FP32/FP64 for likelihood arithmetic, not FP16")
    if any(v.dtype != targets.dtype or v.device != targets.device for v in floats):
        raise ValueError("All floating tensors must share dtype and device")
    if supports.device != targets.device or observed.device != targets.device:
        raise ValueError("Masks must be on the same device")
    if not torch.isfinite(targets[observed]).all():
        raise ValueError("Observed target descriptors must be finite")
    if not all(torch.isfinite(v).all() for v in floats[1:]):
        raise ValueError("Predictive parameters must be finite")
    if (shared_log_variance.abs() > 20).any():
        raise ValueError("Unreasonable variance; calibrate scale on fitting data")

    # Missing entries are neutral even when padding contains NaN.
    safe_targets = targets.masked_fill(~observed[..., None], 0)
    residual = safe_targets[:, None] - negative_mean
    invvar = (-shared_log_variance).exp()[None, :, None, :]
    ell = (finding_shift * residual * invvar
           - 0.5 * finding_shift.square() * invvar).sum(-1)
    ell = ell.masked_fill(~observed[:, None, :], 0)

    lp0_local = -0.5 * (
        residual.square() * invvar
        + shared_log_variance[None, :, None, :]
        + math.log(2 * math.pi)
    ).sum(-1)
    lp0 = lp0_local.masked_fill(~observed[:, None, :], 0).sum(-1)
    log_pi = support_logits.log_softmax(-1)
    support_evidence = torch.einsum("bcj,ckj->bck", ell, supports.to(ell.dtype))
    terms = log_pi[None] + support_evidence
    log_bf = terms.logsumexp(-1)
    support_log_posterior = terms - log_bf[..., None]
    no_overlap = ~(supports[None] & observed[:, None, None, :]).any(-1)
    missing_mass = (log_pi.exp()[None] * no_overlap).sum(-1)
    out = EvidenceOutput(lp0, lp0 + log_bf, log_bf, ell,
                         support_log_posterior, missing_mass)
    if not all(torch.isfinite(v).all() for v in
               (out.log_p0, out.log_p1, out.log_bayes_factor)):
        raise FloatingPointError("Non-finite density calculation")
    return out


def weak_supervised_loss(
    evidence: EvidenceOutput,
    context_logits: Tensor,    # [B,C], context ONLY; not an inherited teacher
    soft_targets: Tensor,      # [B,C], accepted weak labels, not ground truth
    label_mask: Tensor,        # bool [B,C]
    observed: Tensor,          # bool [B,J]
    descriptor_width: int,
    density_weight: float = 0.1,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Expected conditional NLL + exam loss, with unchanged weak-label masks.

    q*log p1 + (1-q)*log p0 treats q as a fractional weak label.
    It does not infer a new 'true' label and does not establish q's calibration.
    Loss normalization is per observed descriptor; BF inference is not normalized.
    """
    shape = evidence.log_bayes_factor.shape
    if any(v.shape != shape for v in (context_logits, soft_targets, label_mask)):
        raise ValueError("Aligned [B,C] predictions, source labels, and masks required")
    if label_mask.dtype != torch.bool or observed.dtype != torch.bool:
        raise ValueError("Masks must be boolean")
    if type(descriptor_width) is not int or descriptor_width < 1:
        raise ValueError("Positive descriptor width required")
    if not math.isfinite(density_weight) or density_weight < 0:
        raise ValueError("Nonnegative finite density weight required")
    if not torch.isfinite(context_logits).all():
        raise ValueError("Context logits must be finite")
    if not torch.isfinite(soft_targets[label_mask]).all():
        raise ValueError("Observed source targets must be finite")
    if ((soft_targets[label_mask] < 0) | (soft_targets[label_mask] > 1)).any():
        raise ValueError("Source targets must lie in [0,1]")
    q = torch.where(label_mask, soft_targets.detach(), torch.zeros_like(soft_targets))
    logits = context_logits + evidence.log_bayes_factor
    exam_cells = F.binary_cross_entropy_with_logits(logits, q, reduction="none")
    count = label_mask.sum().clamp_min(1)
    exam_loss = exam_cells.masked_fill(~label_mask, 0).sum() / count
    nll = -(q * evidence.log_p1 + (1 - q) * evidence.log_p0)
    scalar_count = observed.sum(-1, keepdim=True) * descriptor_width
    active = label_mask & (scalar_count > 0)
    density_loss = (nll / scalar_count.clamp_min(1)).masked_fill(~active, 0).sum()
    density_loss = density_loss / active.sum().clamp_min(1)
    return exam_loss + density_weight * density_loss, {
        "exam_loss": exam_loss, "density_loss": density_loss, "logits": logits,
    }


def average_mask_posteriors(context_logits: Tensor, log_bayes_factors: Tensor) -> Tensor:
    """[B,M,C] -> [B,C]. Average posteriors; do not multiply correlated masks."""
    if (context_logits.ndim != 3 or context_logits.shape != log_bayes_factors.shape
            or context_logits.shape[1] < 1):
        raise ValueError("Aligned non-empty [B,mask,class] arrays required")
    if not torch.isfinite(context_logits).all() or not torch.isfinite(log_bayes_factors).all():
        raise ValueError("Finite masked predictions required")
    return (context_logits + log_bayes_factors).sigmoid().mean(1)
