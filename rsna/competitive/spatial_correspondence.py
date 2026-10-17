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
        # Qualified local kernels are sparse. Uniformly guessing four indices
        # almost never finds a usable cycle. Enumerate row pairs sharing two
        # positive columns, then sample within that unchanged support.
        support = value > 0
        common = support.astype(np.int32) @ support.astype(np.int32).T
        row_pairs = np.argwhere(np.triu(common >= 2, k=1))
        for _ in range(attempts):
            if not len(row_pairs):
                break
            i, k = row_pairs[rng.integers(len(row_pairs))]
            columns = np.flatnonzero(support[i] & support[k])
            j, last = rng.choice(columns, 2, replace=False)
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


def randomized_correspondence_null(
    pi: Tensor, allowed: Tensor, *, seed: int, iterations: int = 256
) -> tuple[Tensor, dict]:
    """Randomize links while retaining weighted degrees and observed coverage.

    ``allowed`` describes observation/structural constraints, NOT the geometric
    neighborhood being tested. E.g. padding is unavailable, and local texture
    excludes self-links in BOTH arms. Keeping the geometric neighborhood itself
    would only test its distance weights. No evidence or labels select the null.
    """
    _coupling(pi)
    if (
        type(seed) is not int
        or type(iterations) is not int
        or not 1 <= iterations <= 1024
        or pi.requires_grad
        or allowed.shape != pi.shape
        or allowed.dtype != torch.bool
        or allowed.device != pi.device
        or ((pi > 0) & ~allowed).any()
    ):
        raise ValueError("Fixed geometry and complete common observation mask required")
    original = pi.detach().cpu().double().numpy()
    permitted = allowed.cpu().numpy()
    a, b = original.sum(1), original.sum(0)
    active = (a[:, None] > 0) & (b[None] > 0) & permitted
    rng = np.random.default_rng(seed)
    value = original[rng.permutation(len(a))][:, rng.permutation(len(b))].copy()
    value = np.where(active, value + 0.25 * np.outer(a, b), 0)
    # IPF is feasible because the original coupling supplies a witness under
    # the SAME allowed mask. All active allowed cells receive a positive floor. The 0.25 independent
    # mass prevents nearly disconnected permuted blocks from exhausting IPF.
    # This engineering choice uses geometry alone, never images or outcomes.
    converged = False
    for step in range(iterations):
        value *= np.divide(a, value.sum(1), out=np.ones_like(a), where=a > 0)[:, None]
        value *= np.divide(b, value.sum(0), out=np.ones_like(b), where=b > 0)[None]
        error = max(np.max(np.abs(value.sum(1) - a)), np.max(np.abs(value.sum(0) - b)))
        if error < 1e-10:
            converged = True
            break
    result = torch.as_tensor(value, dtype=pi.dtype, device=pi.device)
    same = (
        converged
        and torch.equal(result.sum(0) > 0, pi.sum(0) > 0)
        and torch.equal(result.sum(1) > 0, pi.sum(1) > 0)
        and not ((result > 0) & ~allowed).any()
        and torch.allclose(result.sum(0), pi.sum(0), atol=1e-6, rtol=1e-6)
        and torch.allclose(result.sum(1), pi.sum(1), atol=1e-6, rtol=1e-6)
    )
    centered = original - np.outer(a, b)
    denominator = float(np.linalg.norm(centered))
    ratio = float(np.linalg.norm(value - np.outer(a, b)) / max(denominator, 1e-30))
    difference = float(np.abs(value - original).sum())
    testable = bool(same and difference > 1e-6 and denominator > 1e-8 and 0.5 <= ratio <= 2)
    return result, {
        "seed": seed,
        "iterations": step + 1,
        "converged": converged,
        "testable": testable,
        "same_both_weighted_marginals_and_observed_tokens": bool(same),
        "same_structural_observation_mask": True,
        "geometric_positive_edges_preserved": False,
        "centered_operator_Frobenius_ratio": ratio,
        "L1_coupling_change": difference,
        "labels_or_evidence_used": False,
    }
