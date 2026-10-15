import math

import pytest
import torch

from rsna.competitive.adapter_soup import average_adapter_updates


def factors(n=3, rank=2, dtype=torch.float64):
    g = torch.Generator().manual_seed(19)
    return [
        (
            torch.randn(rank, 7, generator=g, dtype=dtype),
            torch.randn(8, rank, generator=g, dtype=dtype),
        )
        for _ in range(n)
    ]


def test_matches_independent_dense_optimal_rank_projection_and_norm():
    parents = factors()
    dense = sum(b @ a for a, b in parents) / len(parents)
    u, s, vh = torch.linalg.svd(dense, full_matrices=False)
    expected = (u[:, :2] * s[:2]) @ vh[:2]
    a, b, d = average_adapter_updates(parents, scaling=3.5)
    torch.testing.assert_close(b @ a, expected, atol=1e-12, rtol=1e-12)
    assert d["postcast_residual_frobenius_norm"] == pytest.approx(
        float((dense - b @ a).norm()) * 3.5
    )
    assert d["optimal_truncation_frobenius_norm"] == pytest.approx(float(s[2:].norm()) * 3.5)


def test_factor_gauge_invariance_and_naive_factor_average_is_wrong():
    a, b = factors(n=1)[0]
    g = torch.tensor([[2.0, 1.0], [0.0, 0.5]], dtype=torch.float64)
    changed = (g @ a, b @ torch.linalg.inv(g))
    out_a, out_b, _ = average_adapter_updates([(a, b), changed], scaling=4 / math.sqrt(2))
    torch.testing.assert_close(out_b @ out_a, b @ a, atol=1e-12, rtol=1e-12)
    assert not torch.allclose((b + changed[1]) / 2 @ ((a + changed[0]) / 2), b @ a)


def test_identical_parents_exact_factors_are_copied_without_alias_or_rng():
    a, b = factors(n=1, dtype=torch.float32)[0]
    before = torch.get_rng_state().clone()
    x, y, _ = average_adapter_updates([(a, b)] * 3, scaling=1.0)
    assert torch.equal(x, a) and torch.equal(y, b)
    assert x.data_ptr() != a.data_ptr() and y.data_ptr() != b.data_ptr()
    assert torch.equal(before, torch.get_rng_state())


def test_known_rank_one_truncation_and_zero_delta():
    parents = [
        (torch.tensor([[1.0, 0.0]]), torch.tensor([[1.0], [0.0]])),
        (torch.tensor([[0.0, 1.0]]), torch.tensor([[0.0], [0.5]])),
    ]
    a, b, d = average_adapter_updates(parents, scaling=2.0)
    torch.testing.assert_close(b @ a, torch.diag(torch.tensor([0.5, 0.0])))
    assert d["optimal_truncation_frobenius_norm"] == pytest.approx(0.5)
    zero = [(a, torch.zeros_like(b)) for a, b in parents]
    a, b, d = average_adapter_updates(zero, scaling=2.0)
    assert not (b @ a).count_nonzero()
    assert d["mean_delta_frobenius_norm"] == 0 and d["postcast_relative_error"] == 0


def test_order_invariance_and_float32_cast_error_is_measured():
    parents = factors(dtype=torch.float32)
    a, b, d = average_adapter_updates(parents, scaling=1.0)
    x, y, _ = average_adapter_updates(list(reversed(parents)), scaling=1.0)
    torch.testing.assert_close(
        b.double() @ a.double(), y.double() @ x.double(), atol=2e-6, rtol=2e-6
    )
    dense = sum(b.double() @ a.double() for a, b in parents) / len(parents)
    assert d["postcast_residual_frobenius_norm"] == pytest.approx(
        float((dense - b.double() @ a.double()).norm())
    )


@pytest.mark.parametrize("scaling", [0, -1, float("nan"), float("inf"), True])
def test_refuses_invalid_scaling(scaling):
    with pytest.raises(ValueError):
        average_adapter_updates(factors(), scaling=scaling)


@pytest.mark.parametrize("defect", ["nan", "shape", "dtype", "integer", "rank"])
def test_refuses_incompatible_factors(defect):
    parents = factors()
    a, b = parents[1]
    if defect == "nan":
        a[0, 0] = float("nan")
    if defect == "shape":
        a = a[:, :3]
    if defect == "dtype":
        a = a.float()
    if defect == "integer":
        a = a.long()
    if defect == "rank":
        b = b[:, :1]
    parents[1] = a, b
    with pytest.raises(ValueError):
        average_adapter_updates(parents, scaling=1.0)


def test_svd_uses_small_core_not_dense_update(monkeypatch):
    seen = []
    original = torch.linalg.svd

    def observe(x, **kwargs):
        seen.append(x.shape)
        return original(x, **kwargs)

    monkeypatch.setattr(torch.linalg, "svd", observe)
    average_adapter_updates(factors(), scaling=1.0)
    assert seen == [torch.Size([6, 6])]


@pytest.mark.parametrize("count", [0, 11])
def test_parent_count_is_explicitly_bounded(count):
    with pytest.raises(ValueError):
        average_adapter_updates(factors(n=count), scaling=1.0)


def test_finite_but_overflowing_factors_refused():
    a = torch.full((1, 2), 1e200, dtype=torch.float64)
    b = torch.full((2, 1), 1e200, dtype=torch.float64)
    with pytest.raises(ValueError, match="overflow"):
        average_adapter_updates([(a, b)], scaling=1.0)
