"""Factorized norm agrees with an independent dense residual on small matrices."""

import pytest
import torch

from scripts.research.adapter_update_geometry import effective_update_norm


@pytest.mark.parametrize("shape", [(7, 3, 5), (13, 4, 9)])
def test_factorized_norm_matches_dense_residual(shape):
    out, rank, inputs = shape
    torch.manual_seed(23)
    a, b, base = torch.randn(rank, inputs), torch.randn(out, rank), torch.randn(out, inputs)
    got = effective_update_norm(a, b, base, 1.4)
    expected = float(torch.linalg.vector_norm(1.4 * (b.double() @ a.double())))
    assert got["effective_update_Frobenius_norm"] == pytest.approx(expected, rel=1e-12)


def test_exact_zero_start_remains_zero():
    result = effective_update_norm(torch.ones(2, 3), torch.zeros(4, 2), torch.ones(4, 3), 1.4)
    assert (
        result["effective_update_Frobenius_norm"]
        == result["relative_effective_update_Frobenius_norm"]
        == 0
    )


@pytest.mark.parametrize("bad", ["shape", "nonfinite", "base_zero", "scaling"])
def test_invalid_projection_diagnostic_is_rejected(bad):
    a, b, base = torch.ones(2, 3), torch.ones(4, 2), torch.ones(4, 3)
    scale = 1.4
    if bad == "shape":
        b = torch.ones(4, 3)
    if bad == "nonfinite":
        a[0, 0] = float("nan")
    if bad == "base_zero":
        base.zero_()
    if bad == "scaling":
        scale = -1
    with pytest.raises(ValueError):
        effective_update_norm(a, b, base, scale)
