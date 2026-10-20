"""Readout-only fidelity analysis after real encoder drift, never clinical scoring.

The linear estimator's IID logit bias/variance is computed analytically in FP64.
Ordinary sampled normalized attention is measured on common finite draws. This
cannot establish encoder-gradient quality or predictive accuracy. Inputs stay on
an owning node; returned aggregates contain no studies, labels or image records.
"""

from __future__ import annotations

import torch


def fidelity(
    reference_scores,
    reference_votes,
    current_scores,
    current_votes,
    proposals,
    *,
    draws=128,
    windows=16,
    seed=20261020,
):
    arrays = (reference_scores, reference_votes, current_scores, current_votes)
    if (
        any(not isinstance(x, torch.Tensor) for x in (*arrays, proposals))
        or reference_scores.ndim != 3
        or min(reference_scores.shape) < 1
        or any(x.shape != reference_scores.shape for x in arrays)
        or proposals.shape != reference_scores.shape[:2]
        or any(
            x.device.type != "cpu"
            or x.requires_grad
            or not x.is_floating_point()
            or not torch.isfinite(x).all()
            for x in (*arrays, proposals)
        )
        or (proposals <= 0).any()
        or not torch.allclose(
            proposals.double().sum(1),
            torch.ones(len(proposals), dtype=torch.float64),
            atol=1e-6,
            rtol=0,
        )
        or any(type(x) is not int or x < 1 for x in (draws, windows))
        or type(seed) is not int
    ):
        raise ValueError(
            "Detached finite CPU BxNxT readouts, positive normalized BxN proposals and explicit finite draw budget required"
        )
    r, v, s, u = (x.double() for x in arrays)
    q = proposals.double()
    # Normalize only FP32 serialization rounding in the diagnostic oracle.
    q = q / q.sum(1, keepdim=True)
    w = r.softmax(1)
    center = (w * v).sum(1)
    change = (u - v) + (v - center[:, None]) * (s - r)
    per_draw = w * change / q[..., None]
    mean_change = (q[..., None] * per_draw).sum(1)
    linear_mean = center + mean_change
    exact = (s.softmax(1) * u).sum(1)
    squared_bias = (linear_mean - exact).square()
    variance = ((q[..., None] * (per_draw - mean_change[:, None]).square()).sum(1)) / windows
    assert (variance >= 0).all() and torch.isfinite(variance).all()
    linear_mse = squared_bias + variance
    generator = torch.Generator(device="cpu").manual_seed(seed)
    empirical_linear = []
    empirical_sni = []
    for b in range(len(q)):
        i = torch.multinomial(q[b], draws * windows, replacement=True, generator=generator).reshape(
            draws, windows
        )
        sampled = ((s[b][i] - q[b][i].log()[..., None]).softmax(1) * u[b][i]).sum(1)
        linear = center[b] + per_draw[b][i].mean(1)
        empirical_linear.append((linear - exact[b]).square().mean(0))
        empirical_sni.append((sampled - exact[b]).square().mean(0))
    sampled_mse = torch.stack(empirical_sni).mean(0)
    oracle_mse = linear_mse.mean(0)
    result = {
        "status": "readout_fidelity_only_not_accuracy_or_encoder_gradient_acceptance",
        "observations": len(q),
        "windows_in_full_bag": q.shape[1],
        "targets": r.shape[2],
        "draws_each": draws,
        "windows_each": windows,
        "seed": seed,
        "diagnostic_precision": "float64; not bitwise deployed FP32 arithmetic",
        "linear_exact_logit_squared_bias_by_target": squared_bias.mean(0).tolist(),
        "linear_exact_IID_logit_variance_by_target": variance.mean(0).tolist(),
        "linear_exact_logit_MSE_by_target": oracle_mse.tolist(),
        "linear_common_draw_empirical_logit_MSE_by_target": torch.stack(empirical_linear)
        .mean(0)
        .tolist(),
        "SNI_common_draw_empirical_logit_MSE_by_target": sampled_mse.tolist(),
        "linear_exact_to_SNI_empirical_total_MSE_ratio": float(oracle_mse.sum() / sampled_mse.sum())
        if sampled_mse.sum() > 0
        else None,
        "clinical_labels_or_scores_used": False,
        "automatic_recipe_selection": False,
    }
    return result
