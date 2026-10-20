import itertools

import pytest
import torch

from rsna.competitive.cached_attention_diagnostics import fidelity


def inputs():
    generator = torch.Generator().manual_seed(9)
    r = torch.randn(1, 3, 2, dtype=torch.float64, generator=generator)
    v = torch.randn(1, 3, 2, dtype=torch.float64, generator=generator)
    s = r + torch.randn(r.shape, dtype=r.dtype, generator=generator) * 0.3
    u = v + torch.randn(v.shape, dtype=v.dtype, generator=generator) * 0.2
    q = torch.tensor([[0.1, 0.3, 0.6]], dtype=torch.float64)
    return r, v, s, u, q


def test_analytical_linear_mse_matches_all_ordered_iid_draws():
    r, v, s, u, q = inputs()
    result = fidelity(r, v, s, u, q, windows=2)
    w = r.softmax(1)
    center = (w * v).sum(1)
    truth = (s.softmax(1) * u).sum(1)
    delta = w / q[..., None] * (u - v + (v - center[:, None]) * (s - r))
    errors = torch.zeros_like(center)
    for a, b in itertools.product(range(3), repeat=2):
        estimate = center + (delta[:, a] + delta[:, b]) / 2
        errors += q[0, a] * q[0, b] * (estimate - truth).square()
    torch.testing.assert_close(
        torch.tensor(result["linear_exact_logit_MSE_by_target"], dtype=torch.float64),
        errors[0],
        atol=1e-15,
        rtol=1e-12,
    )


def test_reference_has_zero_linear_error_but_sampled_attention_need_not():
    r, v, _, _, q = inputs()
    result = fidelity(r, v, r, v, q)
    assert result["linear_exact_logit_MSE_by_target"] == [0.0, 0.0]
    assert sum(result["SNI_common_draw_empirical_logit_MSE_by_target"]) > 0


def test_global_rng_untouched_and_identical_common_draw_results():
    args = inputs()
    before = torch.get_rng_state().clone()
    result = fidelity(*args)
    assert torch.equal(before, torch.get_rng_state())
    assert result == fidelity(*args)


@pytest.mark.parametrize("field", ["bad_q", "zero_q", "grad", "nan", "budget"])
def test_refuses_invalid_inputs(field):
    r, v, s, u, q = inputs()
    kwargs = {}
    if field == "bad_q":
        q *= 2
    if field == "zero_q":
        q[0, 0] = 0
    if field == "grad":
        s.requires_grad_()
    if field == "nan":
        u[0, 0, 0] = float("nan")
    if field == "budget":
        kwargs["draws"] = 0
    with pytest.raises(ValueError):
        fidelity(r, v, s, u, q, **kwargs)
