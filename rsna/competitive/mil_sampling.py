"""Label-free importance-sampling fidelity of a frozen finding-specific MIL head.

This measures approximation to a model, never clinical evidence or accuracy.
Self-normalized finite-bag estimates may be biased. No sampling policy is promoted
from this module alone, and no fitted/evaluation labels are consumed.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

from rsna.competitive.released_context_infer import PARENT_SOURCE
from rsna.competitive.released_context_train import PARENT_SHA


@torch.no_grad()
def readout_moments(value, features):
    """Exact original FP32 head plus a FP64 moment representation for analysis."""
    if (
        value.get("format") != "licensed_parent_MIL_readout_v1"
        or value.get("parent_checkpoint_sha256") != PARENT_SHA
        or value.get("source_notebook_sha256") != PARENT_SOURCE
        or features.shape != (2, 94, 1024)
        or features.dtype != torch.float32
        or not torch.isfinite(features).all()
    ):
        raise ValueError("Original licensed readout and two real FP32 feature views required")
    state = value["model"]
    shapes = {
        "norm.weight": (1024,),
        "norm.bias": (1024,),
        "att.0.weight": (256, 1024),
        "att.0.bias": (256,),
        "att.3.weight": (12, 256),
        "att.3.bias": (12,),
        "clsW": (12, 1024),
        "clsb": (12,),
    }
    if set(state) != set(shapes) or any(
        not isinstance(state[k], torch.Tensor)
        or tuple(state[k].shape) != shape
        or state[k].dtype != torch.float32
        or not torch.isfinite(state[k]).all()
        for k, shape in shapes.items()
    ):
        raise ValueError("Exact complete finite FP32 parent MIL parameter graph required")
    h = F.layer_norm(features, (1024,), state["norm.weight"], state["norm.bias"], eps=1e-5)
    scores = F.linear(
        F.linear(h, state["att.0.weight"], state["att.0.bias"]).tanh(),
        state["att.3.weight"],
        state["att.3.bias"],
    )
    weights = scores.softmax(dim=1)
    pooled = torch.einsum("bkn,bkf->bnf", weights, h)
    logits = (pooled * state["clsW"]).sum(-1) + state["clsb"]
    # High precision separates estimator noise from reduction-order error. These
    # are still moments of this trained teacher, not independent target labels.
    votes = torch.einsum("bkd,td->bkt", h.double(), state["clsW"].double())
    return (
        scores.double().softmax(1).numpy(),
        votes.numpy(),
        state["clsb"].double().numpy(),
        logits.sigmoid().mean(0).numpy(),
    )


def probability(logits):
    return np.exp(-np.logaddexp(0.0, -logits))


def validate(weights, votes, bias):
    if (
        weights.ndim != 3
        or weights.shape != votes.shape
        or weights.shape[0] != 2
        or min(weights.shape[1:]) < 1
        or bias.shape != (weights.shape[2],)
        or not all(np.isfinite(x).all() for x in (weights, votes, bias))
        or (weights <= 0).any()
        or not np.allclose(weights.sum(1), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Finite two-view normalized positive teacher moments required")


def proposal(weights, votes, bias, *, uniform_fraction=0.25):
    validate(weights, votes, bias)
    if type(uniform_fraction) not in (float, int) or not 0 < uniform_fraction <= 1:
        raise ValueError("Positive uniform coverage floor required")
    center = (weights * votes).sum(1, keepdims=True)
    sensitivity = (weights * np.abs(votes - center)).mean(0)
    normalizer = sensitivity.sum(0, keepdims=True)
    scaled = np.divide(
        sensitivity,
        normalizer,
        out=np.full_like(sensitivity, 1 / len(sensitivity)),
        where=normalizer > 0,
    )
    q = uniform_fraction / len(sensitivity) + (1 - uniform_fraction) * scaled.mean(1)
    return q / q.sum()


def sampled_probabilities(weights, votes, bias, indices, q):
    validate(weights, votes, bias)
    if (
        indices.ndim != 2
        or indices.dtype != np.int64
        or not indices.size
        or (indices < 0).any()
        or (indices >= weights.shape[1]).any()
        or q.shape != (weights.shape[1],)
        or not np.isfinite(q).all()
        or (q <= 0).any()
        or not np.isclose(q.sum(), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Valid iid draws and complete positive proposal required")
    corrected = weights[:, indices] / q[indices][None, :, :, None]
    mu = (corrected * votes[:, indices]).sum(2) / corrected.sum(2)
    return probability(mu + bias).mean(0)


def benchmark(weights, votes, bias, *, draws=128, bag=16, seed=20261016):
    validate(weights, votes, bias)
    if any(type(x) is not int or x < 1 for x in (draws, bag)) or type(seed) is not int:
        raise ValueError("Fixed positive draw/bag counts and seed required")
    full = probability((weights * votes).sum(1) + bias).mean(0)
    q = proposal(weights, votes, bias)
    u = np.random.default_rng(seed).random((draws, bag))
    uniform = np.full(weights.shape[1], 1 / weights.shape[1])
    indices = np.floor(u * len(uniform)).astype(np.int64)
    other = np.searchsorted(np.cumsum(q), u, side="right").clip(max=len(q) - 1).astype(np.int64)
    left = sampled_probabilities(weights, votes, bias, indices, uniform)
    right = sampled_probabilities(weights, votes, bias, other, q)
    errors = [(p - full) ** 2 for p in (left, right)]
    return {
        "uniform_per_target_MSE": errors[0].mean(0),
        "influence_per_target_MSE": errors[1].mean(0),
        "uniform_mean_unique_windows": float(np.mean([len(set(x)) for x in indices])),
        "influence_mean_unique_windows": float(np.mean([len(set(x)) for x in other])),
        "proposal_min_probability": float(q.min()),
        "proposal_max_probability": float(q.max()),
        "draws": draws,
        "draws_per_view": bag,
        "mean_view_full_probability": full,
        "clinical_accuracy_evaluated": False,
    }
