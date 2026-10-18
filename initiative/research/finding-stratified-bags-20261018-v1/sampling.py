"""Fixed-count finding proposals at the unchanged sixteen-window mixture density.

This addresses estimator coverage, not clinical lesion localization. The teacher
and its moments are frozen. No label, report, target weighting or encoder changes.
"""

from __future__ import annotations

import numpy as np

from rsna.competitive.mil_sampling import proposal, sampled_probabilities, validate


def components(weights, votes, bias):
    validate(weights, votes, bias)
    if weights.shape[2] != 12:
        raise ValueError("Twelve original finding proposals required")
    center = (weights * votes).sum(1, keepdims=True)
    sensitivity = (weights * np.abs(votes - center)).mean(0)
    normalizer = sensitivity.sum(0, keepdims=True)
    distributions = np.divide(
        sensitivity,
        normalizer,
        out=np.full_like(sensitivity, 1 / len(sensitivity)),
        where=normalizer > 0,
    )
    q = proposal(weights, votes, bias, uniform_fraction=0.25)
    mixture = (distributions.sum(1) + 4 / len(distributions)) / 16
    if not np.allclose(mixture, q, rtol=0, atol=1e-15):
        raise ValueError("Exact original mixture density must remain unchanged")
    return distributions, q


def draw(distributions, q, uniforms):
    k = len(q)
    if (
        distributions.shape != (k, 12)
        or uniforms.ndim != 2
        or uniforms.shape[1] != 16
        or not len(uniforms)
        or not all(np.isfinite(x).all() for x in (distributions, q, uniforms))
        or (distributions < 0).any()
        or not np.allclose(distributions.sum(0), 1, rtol=0, atol=1e-12)
        or not np.allclose((distributions.sum(1) + 4 / k) / 16, q, rtol=0, atol=1e-15)
        or (uniforms < 0).any()
        or (uniforms >= 1).any()
    ):
        raise ValueError("Fixed twelve-plus-four strata and half-open uniform draws required")
    iid = np.searchsorted(q.cumsum(), uniforms, side="right").clip(max=k - 1)
    strata = [
        np.searchsorted(distributions[:, c].cumsum(), uniforms[:, c], side="right").clip(max=k - 1)
        for c in range(12)
    ] + [(uniforms[:, c] * k).astype(np.int64) for c in range(12, 16)]
    return iid.astype(np.int64), np.stack(strata, axis=1).astype(np.int64)


def miss_probabilities(distributions, q, selected):
    """Exact no-hit probabilities for a specified teacher-support set, not a lesion."""
    if (
        distributions.shape != (len(q), 12)
        or selected.shape != q.shape
        or selected.dtype != np.bool_
        or not selected.any()
        or not np.allclose(distributions.sum(0), 1, rtol=0, atol=1e-12)
        or not np.allclose((distributions.sum(1) + 4 / len(q)) / 16, q, atol=1e-15, rtol=0)
    ):
        raise ValueError("Bound original mixture and nonempty support set required")
    a = np.clip(1 - q[selected].sum(), 0, 1) ** 16
    b = np.prod(np.clip(1 - distributions[selected].sum(0), 0, 1))
    b *= (1 - selected.mean()) ** 4
    return float(a), float(b)


def benchmark(weights, votes, bias, *, draws=128, seed=20261018):
    if type(draws) is not int or draws < 1 or type(seed) is not int:
        raise ValueError("Fixed positive draw count and integer seed required")
    distributions, q = components(weights, votes, bias)
    u = np.random.default_rng(seed).random((draws, 16))
    left, right = draw(distributions, q, u)
    full = np.exp(-np.logaddexp(0.0, -((weights * votes).sum(1) + bias))).mean(0)
    errors = [sampled_probabilities(weights, votes, bias, x, q) - full for x in (left, right)]
    misses = []
    for c in range(12):
        order = np.argsort(-distributions[:, c], kind="stable")
        count = int(np.searchsorted(distributions[order, c].cumsum(), 0.8) + 1)
        selected = np.zeros(len(q), dtype=bool)
        selected[order[:count]] = True
        misses.append(miss_probabilities(distributions, q, selected))
    return {
        "iid_MSE": (errors[0] ** 2).mean(0),
        "stratified_MSE": (errors[1] ** 2).mean(0),
        "iid_bias": errors[0].mean(0),
        "stratified_bias": errors[1].mean(0),
        "iid_support_miss_probability": np.array(misses)[:, 0],
        "stratified_support_miss_probability": np.array(misses)[:, 1],
        "iid_mean_unique": np.mean([len(set(x)) for x in left]),
        "stratified_mean_unique": np.mean([len(set(x)) for x in right]),
        "maximum_mixture_density_difference": float(
            np.max(np.abs((distributions.sum(1) + 4 / len(q)) / 16 - q))
        ),
    }
