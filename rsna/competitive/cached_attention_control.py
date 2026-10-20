"""Research-only cached-moment control variate for sampled MIL attention.

A fixed full reference supplies numerator/denominator moments. Samples estimate
only the student's change from that reference. The moments are unbiased under
IID sampling; their ratio and nonlinear-loss gradient generally are not. At an
exact deterministic reference, logits are exact for every sample and expected
first derivatives match the full bag. No production trainer is changed here.
"""

from __future__ import annotations

import torch


def _inputs(reference_weights, reference_votes, indices, log_proposal, score_delta, vote_delta):
    """Return sampled numerator/normalizer; expose invalid ratio states explicitly.

    Arrays are BxNxT, BxK, BxK and BxKxT respectively. Fixed reference weights
    sum to one over N. Delta scores subtract reference *unnormalized* scores;
    delta votes subtract the corresponding fixed reference class projection.
    Duplicate IID indices are valid and count separately. No RNG is consumed.
    """
    w, v, i, q, ds, dv = (
        reference_weights,
        reference_votes,
        indices,
        log_proposal,
        score_delta,
        vote_delta,
    )
    if (
        not isinstance(w, torch.Tensor)
        or w.ndim != 3
        or not w.numel()
        or w.dtype not in (torch.float32, torch.float64)
        or any(not isinstance(x, torch.Tensor) for x in (v, i, q, ds, dv))
        or v.shape != w.shape
        or i.ndim != 2
        or i.shape[0] != w.shape[0]
        or not i.numel()
        or i.dtype != torch.int64
        or q.shape != i.shape
        or ds.shape != (*i.shape, w.shape[2])
        or dv.shape != ds.shape
        or any(x.dtype != w.dtype or x.device != w.device for x in (v, q, ds, dv))
        or i.device != w.device
        or any(x.requires_grad for x in (w, v, q))
        or any(not torch.isfinite(x).all() for x in (w, v, q, ds, dv))
        or (w <= 0).any()
        or (q > 0).any()
        or (i < 0).any()
        or (i >= w.shape[1]).any()
    ):
        raise ValueError(
            "Aligned finite fixed moments, IID indices/probabilities and FP32/64 deltas required"
        )
    tolerance = 1e-6 if w.dtype == torch.float32 else 1e-12
    if not torch.allclose(w.sum(1), torch.ones_like(w.sum(1)), atol=tolerance, rtol=0):
        raise ValueError("Fixed full reference attention must have unit class mass")
    return w, v, i, q, ds, dv


def moments(reference_weights, reference_votes, indices, log_proposal, score_delta, vote_delta):
    """Unbiased change moments; the ratio may have a nonpositive denominator."""
    w, v, i, q, ds, dv = _inputs(
        reference_weights, reference_votes, indices, log_proposal, score_delta, vote_delta
    )
    selected = i.unsqueeze(-1).expand_as(ds)
    a0 = w.gather(1, selected)
    v0 = v.gather(1, selected)
    # q is the exact probability of each draw, not a probability over unique indices.
    invq = (-q).exp().unsqueeze(-1)
    a = a0 * ds.exp()
    numerator = (w * v).sum(1) + ((a * (v0 + dv) - a0 * v0) * invq).mean(1)
    denominator = 1 + ((a - a0) * invq).mean(1)
    if not torch.isfinite(numerator).all() or not torch.isfinite(denominator).all():
        raise ValueError("Nonfinite cached-moment estimate; do not silently clip or resample")
    return numerator, denominator


def readout(
    reference_weights, reference_votes, indices, log_proposal, score_delta, vote_delta, bias
):
    """A research estimator, not a replacement for the deployed full-bag graph."""
    numerator, denominator = moments(
        reference_weights, reference_votes, indices, log_proposal, score_delta, vote_delta
    )
    if (
        not isinstance(bias, torch.Tensor)
        or bias.shape not in (numerator.shape, numerator.shape[1:])
        or bias.dtype != numerator.dtype
        or bias.device != numerator.device
        or not torch.isfinite(bias).all()
        or (denominator <= 0).any()
    ):
        raise ValueError(
            "Positive sampled denominator and finite current bias required; no fallback"
        )
    result = numerator / denominator + bias
    if not torch.isfinite(result).all():
        raise ValueError("Finite sampled logits required")
    return result


def linear_readout(
    reference_weights, reference_votes, indices, log_proposal, score_delta, vote_delta, bias
):
    """Sample the full-attention first-order change, with no random denominator.

    This is unbiased for a fixed reference's linearized logit, not for the exact
    nonlinear current model. Reference logits/expected first derivatives are
    exact; finite drift introduces deterministic Taylor error. No clipping,
    automatic refresh, fallback or production-engine substitution occurs.
    """
    w, v, i, q, ds, dv = _inputs(
        reference_weights, reference_votes, indices, log_proposal, score_delta, vote_delta
    )
    center = (w * v).sum(1)
    if (
        not isinstance(bias, torch.Tensor)
        or bias.shape not in (center.shape, center.shape[1:])
        or bias.dtype != center.dtype
        or bias.device != center.device
        or not torch.isfinite(bias).all()
    ):
        raise ValueError("Finite aligned current class bias required")
    selected = i.unsqueeze(-1).expand_as(ds)
    a0, v0 = w.gather(1, selected), v.gather(1, selected)
    delta = (a0 * (-q).exp().unsqueeze(-1) * (dv + (v0 - center.unsqueeze(1)) * ds)).mean(1)
    logits = center + delta + bias
    if not torch.isfinite(logits).all():
        raise ValueError("Finite linearized logits required; no silent clipping")
    return logits
