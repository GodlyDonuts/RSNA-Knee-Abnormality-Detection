"""Frozen-parent ratio control variate; engineering surrogate, not disease evidence."""

from __future__ import annotations

import numpy as np


def moments(weights, votes, delta_scores, delta_votes):
    """Teacher weights normalize by frozen Z0; no full student normalizer is used."""
    if (
        weights.shape != votes.shape
        or weights.shape != delta_scores.shape
        or weights.shape != delta_votes.shape
        or weights.ndim != 3
        or weights.shape[0] != 2
        or weights.shape[2] != 12
        or not all(np.isfinite(x).all() for x in (weights, votes, delta_scores, delta_votes))
        or (weights <= 0).any()
        or not np.allclose(weights.sum(1), 1, atol=1e-12, rtol=0)
    ):
        raise ValueError("Complete finite two-view frozen teacher and simulated drift required")
    student_mass = weights * np.exp(delta_scores)
    student_votes = votes + delta_votes
    if not np.isfinite(student_mass).all() or not np.isfinite(student_votes).all():
        raise ValueError("Finite drift mass and votes required; no silent clipping")
    return student_mass, student_votes


def estimates(weights, votes, delta_scores, delta_votes, indices, q):
    mass, values = moments(weights, votes, delta_scores, delta_votes)
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
        raise ValueError("Bound positive proposal and original sampled windows required")
    density = q[indices][None, :, :, None]
    teacher_mass = weights[:, indices]
    teacher_values = votes[:, indices]
    sampled_mass = mass[:, indices]
    sampled_values = values[:, indices]
    standard_D = (sampled_mass / density).mean(2)
    standard_N = (sampled_mass * sampled_values / density).mean(2)
    delta_D = 1 + ((sampled_mass - teacher_mass) / density).mean(2)
    delta_N = (weights * votes).sum(1)[:, None, :] + (
        (sampled_mass * sampled_values - teacher_mass * teacher_values) / density
    ).mean(2)
    return standard_N, standard_D, delta_N, delta_D


def benchmark(weights, votes, bias, q, *, draws=64, seed=20261018):
    if type(draws) is not int or draws < 1 or bias.shape != (12,):
        raise ValueError("Fixed draw count and twelve teacher biases required")
    rng = np.random.default_rng(seed)
    u = rng.random((draws, 16))
    indices = np.searchsorted(q.cumsum(), u, side="right").clip(max=len(q) - 1).astype(np.int64)
    score_field = rng.standard_normal(weights.shape)
    vote_field = rng.standard_normal(weights.shape)
    center = (weights * votes).sum(1, keepdims=True)
    spread = np.sqrt((weights * (votes - center) ** 2).sum(1, keepdims=True))

    def probability(z):
        return np.exp(-np.logaddexp(0, -z))

    zero = np.zeros_like(weights)
    parent_N, parent_D, _, _ = estimates(weights, votes, zero, zero, indices, q)
    parent_full = (weights * votes).sum(1)[:, None, :]
    result = []
    for scale in (0.0, 0.05, 0.2, 0.5, 1.0):
        ds, dv = scale * score_field, scale * spread * vote_field
        mass, value = moments(weights, votes, ds, dv)
        full = probability((mass * value).sum(1) / mass.sum(1) + bias).mean(0)
        N, D, cvN, cvD = estimates(weights, votes, ds, dv, indices, q)
        valid = np.isfinite(cvD).all((0, 2)) & (cvD > 0).all((0, 2))
        # Never clamp, discard silently, or call failed denominators valid.
        base = probability(N / D + bias).mean(0)
        paired_logits = parent_full + N / D - parent_N / parent_D
        paired = probability(paired_logits + bias).mean(0)
        cv = np.full_like(base, np.nan)
        cv[valid] = probability(cvN[:, valid] / cvD[:, valid] + bias).mean(0)
        total = np.sum((base[valid] - full) ** 2, axis=0)
        other = np.sum((cv[valid] - full) ** 2, axis=0)
        result.append(
            {
                "scale": scale,
                "valid_paired_draws": int(valid.sum()),
                "invalid_draws": int((~valid).sum()),
                "draws": draws,
                "standard_paired_squared_error_sum": total,
                "standard_all_squared_error_sum": ((base - full) ** 2).sum(0),
                "stable_logit_delta_all_squared_error_sum": ((paired - full) ** 2).sum(0),
                "maximum_stable_zero_drift_error": float(np.max(np.abs(paired - full)))
                if scale == 0
                else None,
                "delta_paired_squared_error_sum": other,
                "minimum_delta_denominator": float(cvD.min()),
                "maximum_delta_zero_drift_error": float(np.nanmax(np.abs(cv - full)))
                if scale == 0
                else None,
            }
        )
    return result
