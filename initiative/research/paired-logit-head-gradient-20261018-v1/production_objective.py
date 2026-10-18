"""Paired-logit sampling under actual MIL head drift on frozen teacher features.

No encoder gradients, stochastic training layers, clinical targets or disease
accuracy are represented here. This is a conditional head-only mechanism screen.
"""

from __future__ import annotations

import time

import torch
from torch.nn import functional as F

from rsna.competitive.released_context import anchored_loss

SHAPES = {
    "norm.weight": (1024,),
    "norm.bias": (1024,),
    "att.0.weight": (256, 1024),
    "att.0.bias": (256,),
    "att.3.weight": (12, 256),
    "att.3.bias": (12,),
    "clsW": (12, 1024),
    "clsb": (12,),
}


def parameters(state):
    if set(state) != set(SHAPES) or any(
        not isinstance(state[n], torch.Tensor)
        or state[n].shape != shape
        or state[n].dtype != torch.float32
        or not torch.isfinite(state[n]).all()
        for n, shape in SHAPES.items()
    ):
        raise ValueError("Complete finite FP32 MIL head required")
    return {n: t.detach().clone().requires_grad_() for n, t in state.items()}


def logits(p, x, log_q=None):
    """Same deterministic LayerNorm/tanh attention and finding-specific votes."""
    if x.ndim != 3 or x.shape[-1] != 1024 or x.dtype != torch.float32:
        raise ValueError("Complete FP32 frozen features required")
    h = F.layer_norm(x, (1024,), p["norm.weight"], p["norm.bias"], eps=1e-5)
    scores = F.linear(
        F.linear(h, p["att.0.weight"], p["att.0.bias"]).tanh(), p["att.3.weight"], p["att.3.bias"]
    )
    if log_q is not None:
        if log_q.shape != (x.shape[1],) or not torch.isfinite(log_q).all():
            raise ValueError("One finite original log proposal per IID draw required")
        scores = scores - log_q[None, :, None]
    weights = scores.softmax(1)
    pooled = torch.einsum("bkn,bkf->bnf", weights, h)
    return (pooled * p["clsW"]).sum(-1) + p["clsb"]


def gradient(p, z, target):
    value = F.binary_cross_entropy_with_logits(z, target, reduction="mean")
    tensors = torch.autograd.grad(value, tuple(p.values()))
    vector = torch.cat([x.detach().flatten() for x in tensors]).double()
    if not torch.isfinite(vector).all():
        raise ValueError("Finite complete learned-head gradient required")
    return vector


def paired_logits(parent_full, student_sample, parent_sample):
    if parent_full.requires_grad or parent_sample.requires_grad:
        raise ValueError("Original parent control variates must be detached")
    if student_sample.shape != parent_sample.shape or parent_full.shape != student_sample.shape:
        raise ValueError("Matching complete view/finding logits required")
    return parent_full + student_sample - parent_sample


def production_target(parent_logits, labels, mask, source_weight):
    # Validate with the actual original objective, preserving masked NaNs.
    anchored_loss(
        torch.zeros_like(parent_logits), parent_logits, labels, mask, source_weight=source_weight
    )
    teacher = parent_logits.detach().sigmoid()
    observed = torch.where(mask, labels, teacher)
    return (source_weight * observed + (1 - source_weight) * teacher).detach()


def benchmark(
    parent_state,
    student_state,
    features,
    q,
    *,
    parent_logits,
    labels,
    mask,
    source_weight=0.25,
    draws=64,
    seed=20261018,
):
    if (
        type(draws) is not int
        or draws < 2
        or type(seed) is not int
        or features.shape != (2, 94, 1024)
        or features.dtype != torch.float32
        or features.requires_grad
        or not torch.isfinite(features).all()
        or q.shape != (94,)
        or q.dtype != torch.float64
        or q.requires_grad
        or not torch.isfinite(q).all()
        or (q <= 0).any()
        or abs(float(q.sum()) - 1) > 1e-12
    ):
        raise ValueError("Fixed complete teacher features/proposal and paired draw count required")
    parent = {n: t.detach() for n, t in parameters(parent_state).items()}
    student = parameters(student_state)
    parent_start = time.perf_counter()
    with torch.no_grad():
        parent_full = logits(parent, features)
        parent_setup_seconds = time.perf_counter() - parent_start
        # New experiment: exact production view-specific masked target mixture.
        targets = production_target(parent_logits, labels, mask, source_weight)
        dense_logits = logits(student, features)
        dense_probability = dense_logits.sigmoid().mean(0)
    dense_gradient = gradient(student, logits(student, features), targets)
    generator = torch.Generator().manual_seed(seed)
    selected = torch.searchsorted(
        q.cumsum(0), torch.rand(draws, 16, generator=generator, dtype=torch.float64), right=True
    ).clamp_max(93)
    stats = {
        name: {
            "sum": torch.zeros_like(dense_gradient),
            "squared_error": 0.0,
            "probability_error": torch.zeros(12, dtype=torch.float64),
            "elapsed_seconds": 0.0,
        }
        for name in ("standard", "paired")
    }
    max_initial_forward_error = 0.0
    for draw_index, indices in enumerate(selected):
        x = features[:, indices]
        logq = q[indices].log().float()
        baseline_start = time.perf_counter()
        with torch.no_grad():
            parent_sample = logits(parent, x, logq)
        baseline_seconds = time.perf_counter() - baseline_start
        for name in ("standard", "paired") if draw_index % 2 == 0 else ("paired", "standard"):
            start = time.perf_counter()
            z = logits(student, x, logq)
            if name == "paired":
                z = paired_logits(parent_full, z, parent_sample)
            error = (z.detach().sigmoid().mean(0) - dense_probability).double()
            g = gradient(student, z, targets)
            stat = stats[name]
            stat["sum"] += g
            stat["squared_error"] += float((g - dense_gradient).square().sum())
            stat["probability_error"] += error.square()
            stat["elapsed_seconds"] += time.perf_counter() - start
            if name == "paired":
                stat["elapsed_seconds"] += baseline_seconds + parent_setup_seconds / draws
            if name == "paired":
                max_initial_forward_error = max(
                    max_initial_forward_error, float((z.detach() - dense_logits).abs().max())
                )
    output = {}
    for name, stat in stats.items():
        mean = stat["sum"] / draws
        bias = float((mean - dense_gradient).square().sum())
        mse = stat["squared_error"] / draws
        output[name] = {
            "gradient_squared_error_mean": mse,
            "empirical_mean_gradient_squared_error": bias,
            "sample_gradient_variance_trace": max(0.0, mse - bias),
            "probability_MSE_all12": (stat["probability_error"] / draws).tolist(),
            "measured_head_gradient_plus_estimator_overhead_seconds": stat["elapsed_seconds"],
        }
    return {
        "draws": draws,
        "seed": seed,
        "views": 2,
        "windows": 94,
        "sampled_windows": 16,
        "dense_gradient_squared_norm": float(dense_gradient.square().sum()),
        "head_parameters": len(dense_gradient),
        "estimators": output,
        "maximum_paired_vs_dense_logit_error": max_initial_forward_error,
        "empirical_bias_is_finite_draw_diagnostic_not_unbiasedness_proof": True,
        "encoder_feature_drift_and_gradients_measured": False,
        "head_dropout_enabled": False,
        "actual_training_precision_reproduced": False,
        "target_is_actual_view_specific_masked_production_mixture": True,
        "production_source_weight": source_weight,
        "observed_view_target_cells": int(mask.sum()),
    }
