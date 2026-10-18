"""New estimator statistics only; synthetic vectors do not accept actual GPU behavior."""

import pytest
import torch

from rsna.competitive.paired_encoder_noise import GradientReference, PairedGradientError


def grad(x, y):
    return {
        "backbone.w": torch.tensor([x], dtype=torch.float32),
        "head.w": torch.tensor([y], dtype=torch.float32),
    }


def reference(*values):
    r = GradientReference()
    for x, y in values:
        r.add(grad(x, y))
    return r


def test_exact_dense_reference_reports_all_encoder_and_head_parameters():
    r = reference((1, 2), (1, 2))
    screen = PairedGradientError(r)
    for sign in (-1, 1):
        screen.add(grad(1 + sign * 2, 2 + sign * 4), grad(1 + sign, 2 + sign * 2))
    out = screen.result()
    for name in ("all", "encoder", "head"):
        assert out["groups"][name]["MSE_ratio"] == pytest.approx(0.25)
        assert out["groups"][name]["reference_mean_squared_standard_error"] == 0
    assert out["groups"]["all"]["parameters"] == 2
    assert out["gradient_bias_proven"] is False


def test_stochastic_dense_reference_uncertainty_is_visible_and_never_clipped():
    r = reference((0, 0), (2, 4))
    assert r.standard_error_squared() == {"backbone.w": 1, "head.w": 4}
    screen = PairedGradientError(r)
    screen.add(grad(1, 2), grad(1, 2))
    screen.add(grad(1, 2), grad(1, 2))
    out = screen.result()["groups"]["all"]
    assert out["ordinary"]["MSE_against_estimated_dense_gradient"] == 0
    assert out["ordinary"]["MSE_minus_estimated_reference_noise"] == -5
    assert out["MSE_ratio"] is None


def test_bad_or_partial_controlled_draw_cannot_change_ordinary_statistics():
    screen = PairedGradientError(reference((1, 2), (1, 2)))
    with pytest.raises(ValueError):
        screen.add(grad(2, 3), {"head.w": torch.ones(1)})
    assert screen.count == 0
    assert sum(screen.errors["ordinary"].values()) == 0


def test_reference_is_frozen_and_statistics_consume_no_RNG():
    r = reference((1, 2), (1, 2))
    state = torch.get_rng_state().clone()
    screen = PairedGradientError(r)
    r.add(grad(99, 99))
    for _ in range(2):
        screen.add(grad(2, 4), grad(1.5, 3))
    assert screen.result()["groups"]["all"]["MSE_ratio"] == pytest.approx(0.25)
    assert torch.equal(state, torch.get_rng_state())


@pytest.mark.parametrize("kind", ["nan", "missing", "shape", "scaled_dtype", "autograd"])
def test_reference_refuses_corrupted_gradient_contract(kind):
    r = reference((1, 2), (1, 2))
    x = grad(2, 3)
    if kind == "nan":
        x["head.w"].fill_(float("nan"))
    elif kind == "missing":
        x.pop("head.w")
    elif kind == "shape":
        x["head.w"] = torch.ones(2)
    elif kind == "scaled_dtype":
        x["head.w"] = x["head.w"].half()
    else:
        x["head.w"].requires_grad_()
    with pytest.raises(ValueError):
        r.add(x)
    assert r.count == 2
