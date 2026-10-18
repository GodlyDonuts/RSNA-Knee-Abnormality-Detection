"""Exact small-state tests; no actual student or clinical acceptance."""

import itertools

import numpy as np
import pytest
from estimator import benchmark, estimates, moments


def fixture():
    rng = np.random.default_rng(42)
    w = rng.uniform(0.2, 1, (2, 3, 12))
    w /= w.sum(1, keepdims=True)
    v = rng.normal(size=w.shape)
    ds = rng.normal(0, 0.1, w.shape)
    dv = rng.normal(0, 0.1, w.shape)
    return w, v, ds, dv, np.array([0.2, 0.3, 0.5])


def test_zero_drift_forward_is_exact_full94_target_for_every_bag():
    w, v, _, _, q = fixture()
    ix = np.array(list(itertools.product(range(3), repeat=2)), dtype=np.int64)
    _, _, N, D = estimates(w, v, np.zeros_like(w), np.zeros_like(w), ix, q)
    np.testing.assert_array_equal(D, np.ones_like(D))
    np.testing.assert_allclose(N / D, (w * v).sum(1)[:, None, :] + np.zeros_like(N), atol=0, rtol=0)


def test_numerator_and_denominator_means_are_unbiased_not_ratio():
    w, v, ds, dv, q = fixture()
    ix = np.array(list(itertools.product(range(3), repeat=2)), dtype=np.int64)
    p = q[ix].prod(1)
    _, _, N, D = estimates(w, v, ds, dv, ix, q)
    mass, value = moments(w, v, ds, dv)
    np.testing.assert_allclose(np.einsum("s,bst->bt", p, N), (mass * value).sum(1), atol=1e-14)
    np.testing.assert_allclose(np.einsum("s,bst->bt", p, D), mass.sum(1), atol=1e-14)
    assert (
        np.max(np.abs(np.einsum("s,bst->bt", p, N / D) - (mass * value).sum(1) / mass.sum(1)))
        > 1e-8
    )


def test_negative_denominator_is_preserved_not_clipped():
    w, v, _, _, q = fixture()
    q = np.array([0.01, 0.49, 0.5])
    ix = np.zeros((1, 16), dtype=np.int64)
    ds = np.full_like(w, -10)
    _, _, _, D = estimates(w, v, ds, np.zeros_like(v), ix, q)
    assert (D < 0).any()


def test_no_student_normalizer_and_original_standard_estimator():
    w, v, ds, dv, q = fixture()
    ix = np.array([[0, 2, 1]], dtype=np.int64)
    N, D, _, _ = estimates(w, v, ds, dv, ix, q)
    m, x = moments(w, v, ds, dv)
    a = m[:, ix] / q[ix][None, :, :, None]
    np.testing.assert_allclose(N / D, (a * x[:, ix]).sum(2) / a.sum(2), atol=1e-14)


def test_predeclared_drift_grid_deterministic_and_zero_error_at_parent():
    w, v, _, _, q = fixture()
    a = benchmark(w, v, np.zeros(12), q, draws=16)
    b = benchmark(w, v, np.zeros(12), q, draws=16)
    assert [x["scale"] for x in a] == [0, 0.05, 0.2, 0.5, 1]
    for x, y in zip(a, b):
        for k in x:
            np.testing.assert_equal(x[k], y[k])
    assert a[0]["maximum_stable_zero_drift_error"] < 1e-14
    assert a[0]["invalid_draws"] == 0 and a[0]["maximum_delta_zero_drift_error"] < 1e-14


@pytest.mark.parametrize("damage", ["shape", "weights", "q", "indices"])
def test_invalid_bound_inputs_rejected(damage):
    w, v, ds, dv, q = fixture()
    ix = np.zeros((1, 16), dtype=np.int64)
    if damage == "shape":
        ds = ds[:1]
    elif damage == "weights":
        w[0, 0, 0] = np.nan
    elif damage == "q":
        q[0] = 0
    else:
        ix[0, 0] = 3
    with pytest.raises(ValueError):
        estimates(w, v, ds, dv, ix, q)


@pytest.mark.parametrize("drift", [0.0, 0.2])
def test_expected_fullbag_BCE_gradient_exact_only_at_parent(drift):
    import torch

    w, v, ds0, dv0, q = fixture()
    w = torch.tensor(w, dtype=torch.float64)
    v = torch.tensor(v, dtype=torch.float64)
    ds = torch.tensor(drift * ds0, dtype=torch.float64, requires_grad=True)
    dv = torch.tensor(drift * dv0, dtype=torch.float64, requires_grad=True)
    mass = w * ds.exp()
    values = v + dv
    dense = (mass * values).sum(1) / mass.sum(1)
    target = torch.full_like(dense, 0.3)
    dense_loss = torch.nn.functional.binary_cross_entropy_with_logits(dense, target)
    dense_grad = torch.autograd.grad(dense_loss, (ds, dv), retain_graph=True)
    total = 0
    for pair in itertools.product(range(3), repeat=2):
        ix = list(pair)
        density = torch.tensor(q[ix], dtype=torch.float64)[None, :, None]
        N = (w * v).sum(1) + ((mass[:, ix] * values[:, ix] - w[:, ix] * v[:, ix]) / density).mean(1)
        D = 1 + ((mass[:, ix] - w[:, ix]) / density).mean(1)
        total = total + float(q[ix].prod()) * torch.nn.functional.binary_cross_entropy_with_logits(
            N / D, target
        )
    sampled_grad = torch.autograd.grad(total, (ds, dv))
    maximum = max(float((a - b).abs().max()) for a, b in zip(dense_grad, sampled_grad))
    if drift == 0:
        assert maximum < 1e-14
    else:
        assert maximum > 1e-8  # Finite-bag ratio bias is not magically removed after drift.
