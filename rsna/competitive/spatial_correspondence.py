"""Joint spatial evidence with an explicit marginal-preserving control.

A coupling is supplied by a separately qualified geometry policy. These
operators cannot register images, infer a frame of reference or establish a
clinical finding. Never substitute window ordinals for that policy.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import Tensor


def _coupling(pi: Tensor) -> Tensor:
    if (
        not isinstance(pi, Tensor)
        or pi.ndim != 2
        or min(pi.shape) < 1
        or not pi.is_floating_point()
        or not torch.isfinite(pi).all()
        or (pi < 0).any()
        or not torch.isclose(pi.sum(), pi.new_tensor(1), atol=1e-6, rtol=0)
    ):
        raise ValueError("Finite nonnegative unit-mass qualified coupling required")
    return pi


def centered_cosupport(first: Tensor, second: Tensor, pi: Tensor) -> Tensor:
    """One weighted covariance per target, without a dense centered matrix.

    C = diag(E_first.T @ (Pi - a b.T) @ E_second). Inputs are model evidence,
    not probabilities or independent confirmations. Zero marginal evidence
    dependence does not imply absence of disease.
    """
    _coupling(pi)
    if (
        first.ndim != 2
        or second.ndim != 2
        or first.shape[0] != pi.shape[0]
        or second.shape[0] != pi.shape[1]
        or first.shape[1] != second.shape[1]
        or first.shape[1] < 1
        or first.dtype != pi.dtype
        or second.dtype != pi.dtype
        or first.device != pi.device
        or second.device != pi.device
        or not torch.isfinite(first).all()
        or not torch.isfinite(second).all()
    ):
        raise ValueError("Finite aligned token/target evidence and coupling required")
    a, b = pi.sum(1), pi.sum(0)
    return (first * (pi @ second)).sum(0) - (a @ first) * (b @ second)


def marginal_preserving_null(pi: Tensor, *, seed: int, attempts: int = 256) -> tuple[Tensor, dict]:
    """Bounded four-cycle mass swaps preserve both marginals and ALL support.

    Only already positive entries participate, so structural missingness,
    nonzero edge count and covered tokens remain identical. Sparse couplings
    without four-cycles are explicitly untestable by this null. This is a
    controlled counterfactual, not a sample claimed independent of geometry.
    """
    _coupling(pi)
    if type(seed) is not int or type(attempts) is not int or not 1 <= attempts <= 4096:
        raise ValueError("Explicit seed and bounded positive attempt count required")
    if pi.requires_grad:
        raise ValueError("Null geometry must be fixed independently of learned evidence")
    original = pi.detach().cpu().double().numpy()
    value = original.copy()
    rng = np.random.default_rng(seed)
    swaps = 0
    if min(value.shape) >= 2:
        for _ in range(attempts):
            i, k = rng.choice(value.shape[0], 2, replace=False)
            j, last = rng.choice(value.shape[1], 2, replace=False)
            entries = value[np.ix_([i, k], [j, last])]
            if not (entries > 0).all():
                continue
            # An interior move never creates/deletes an edge. Symmetric choice
            # includes either cycle direction; do not inspect evidence/labels.
            lower = -min(value[i, j], value[k, last])
            upper = min(value[i, last], value[k, j])
            delta = rng.uniform(lower * 0.75, upper * 0.75)
            value[i, j] += delta
            value[k, last] += delta
            value[i, last] -= delta
            value[k, j] -= delta
            swaps += 1
    result = torch.as_tensor(value, dtype=pi.dtype, device=pi.device)
    if (
        not torch.equal(result > 0, pi > 0)
        or not torch.allclose(result.sum(0), pi.sum(0), atol=1e-6, rtol=1e-6)
        or not torch.allclose(result.sum(1), pi.sum(1), atol=1e-6, rtol=1e-6)
    ):
        raise ValueError("Counterfactual failed marginal/support invariants")
    difference = float(torch.abs(result - pi).sum())
    return result, {
        "seed": seed,
        "attempts": attempts,
        "four_cycle_swaps": swaps,
        "L1_coupling_change": difference,
        "testable": swaps > 0 and difference > 1e-6,
        "same_positive_edge_support": True,
        "same_both_weighted_marginals": True,
        "clinical_or_registration_acceptance": False,
    }
