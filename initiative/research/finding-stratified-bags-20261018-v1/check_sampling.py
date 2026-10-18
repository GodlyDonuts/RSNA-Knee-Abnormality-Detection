"""Exact probability calculations and reference draws, without MRI or labels."""

import numpy as np
import pytest
from sampling import benchmark, components, draw, miss_probabilities

from rsna.competitive.mil_sampling import proposal


def fixture(k=7):
    rng = np.random.default_rng(29)
    w = rng.uniform(0.1, 1, (2, k, 12))
    w /= w.sum(1, keepdims=True)
    return w, rng.normal(size=w.shape), np.zeros(12)


def test_original_density_unchanged_and_zero_sensitivity_fallback():
    for constant in (False, True):
        w, x, b = fixture()
        if constant:
            x.fill(0)
        p, q = components(w, x, b)
        np.testing.assert_array_equal(q, proposal(w, x, b))
        np.testing.assert_allclose(p.sum(1) + 4 / len(q), 16 * q, atol=1e-14)
        if constant:
            np.testing.assert_allclose(p, 1 / len(q))


def test_draws_equal_independent_inverse_cdf_reference():
    p, q = components(*fixture())
    u = np.random.default_rng(30).random((40, 16))
    u[0, 0] = 0
    u[0, 15] = np.nextafter(1.0, 0.0)
    iid, stratified = draw(p, q, u)
    for r in range(len(u)):
        for c in range(16):
            assert iid[r, c] == sum(q.cumsum() <= u[r, c])
            expected = sum(p[:, c].cumsum() <= u[r, c]) if c < 12 else int(u[r, c] * len(q))
            assert stratified[r, c] == expected


def test_exact_all_subset_no_hit_dominance():
    p, q = components(*fixture(4))
    for mask in range(1, 16):
        selected = np.array([bool(mask & (1 << k)) for k in range(4)])
        iid, stratified = miss_probabilities(p, q, selected)
        individual = [1 - column[selected].sum() for column in p.T]
        individual += [1 - selected.mean()] * 4
        assert stratified == pytest.approx(np.prod(individual), abs=1e-15)
        assert 0 <= stratified <= iid + 1e-14 <= 1 + 1e-14


def test_enumerated_bernoulli_count_distribution_same_mean_lower_linear_variance():
    p, q = components(*fixture(2))
    iid = np.array([1.0])
    stratified = np.array([1.0])
    for probability in [q[1]] * 16:
        iid = np.convolve(iid, [1 - probability, probability])
    for probability in list(p[1]) + [0.5] * 4:
        stratified = np.convolve(stratified, [1 - probability, probability])
    counts = np.arange(17)
    mean = 16 * q[1]
    assert iid @ counts == pytest.approx(mean, abs=1e-13)
    assert stratified @ counts == pytest.approx(mean, abs=1e-13)
    assert stratified @ (counts - mean) ** 2 <= iid @ (counts - mean) ** 2 + 1e-13


def test_benchmark_deterministic_finite_all_findings():
    args = fixture()
    left, right = benchmark(*args, draws=17), benchmark(*args, draws=17)
    for name in left:
        np.testing.assert_array_equal(left[name], right[name])
        assert np.isfinite(left[name]).all()
    assert left["iid_MSE"].shape == left["stratified_MSE"].shape == (12,)
    assert np.all(
        left["stratified_support_miss_probability"] <= left["iid_support_miss_probability"] + 1e-14
    )


@pytest.mark.parametrize("value", [-0.01, 1.0, np.nan])
def test_invalid_uniforms_refused(value):
    p, q = components(*fixture())
    u = np.zeros((2, 16))
    u[0, 0] = value
    with pytest.raises(ValueError):
        draw(p, q, u)


def test_changed_mixture_refused():
    p, q = components(*fixture())
    with pytest.raises(ValueError):
        draw(p, np.roll(q, 1), np.zeros((2, 16)))
