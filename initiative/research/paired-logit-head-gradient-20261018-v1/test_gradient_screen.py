"""New estimator-gradient mechanism only; no old MRI/model replay."""

import importlib.util
from pathlib import Path

import pytest
import torch

spec = importlib.util.spec_from_file_location(
    "new_paired_logit_gradient", Path(__file__).with_name("gradient_screen.py")
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def state():
    generator = torch.Generator().manual_seed(117)
    p = {n: torch.randn(shape, generator=generator) * 0.03 for n, shape in m.SHAPES.items()}
    p["norm.weight"] += 1
    return p


def test_paired_control_cancels_forward_at_identical_parent():
    parent = torch.tensor([[0.2, -0.3]], dtype=torch.float64)
    sampled = torch.tensor([[0.5, -0.1]], dtype=torch.float64, requires_grad=True)
    z = m.paired_logits(parent, sampled, sampled.detach())
    torch.testing.assert_close(z, parent, rtol=0, atol=1e-16)
    z.sum().backward()
    assert sampled.grad.tolist() == [[1.0, 1.0]]


def test_control_variate_must_not_backpropagate_into_teacher():
    x = torch.ones(2, 12, requires_grad=True)
    with pytest.raises(ValueError):
        m.paired_logits(x, x, x.detach())
    with pytest.raises(ValueError):
        m.paired_logits(x.detach(), x, x)


def test_actual_head_graph_has_finite_complete_dense_gradients():
    torch.set_num_threads(1)
    p = m.parameters(state())
    x = torch.randn(2, 94, 1024)
    g = m.gradient(p, m.logits(p, x), torch.full((2, 12), 0.5))
    assert g.shape == (279832,) and torch.isfinite(g).all() and g.square().sum() > 0


def test_new_screen_reports_prediction_and_gradient_errors_separately():
    torch.set_num_threads(1)
    p = state()
    x = torch.randn(2, 94, 1024)
    a = m.benchmark(p, p, x, torch.full((94,), 1 / 94, dtype=torch.float64), draws=4)
    assert a["head_parameters"] == 279832
    assert a["maximum_paired_vs_dense_logit_error"] < 1e-5
    standard = a["estimators"]["standard"]
    paired = a["estimators"]["paired"]
    assert sum(paired["probability_MSE_all12"]) < sum(standard["probability_MSE_all12"])
    for row in (standard, paired):
        assert (
            row["gradient_squared_error_mean"]
            >= row["empirical_mean_gradient_squared_error"] - 1e-10
        )
    assert a["encoder_feature_drift_and_gradients_measured"] is False
    assert a["actual_training_precision_reproduced"] is False


@pytest.mark.parametrize("draws", [0, 1, True])
def test_invalid_draw_budget_refused(draws):
    p = state()
    with pytest.raises(ValueError):
        m.benchmark(
            p,
            p,
            torch.zeros(2, 94, 1024),
            torch.full((94,), 1 / 94, dtype=torch.float64),
            draws=draws,
        )


def test_broken_head_and_proposal_refused():
    p = state()
    broken = dict(p)
    broken.pop("clsb")
    with pytest.raises(ValueError):
        m.parameters(broken)
    with pytest.raises(ValueError):
        m.benchmark(p, p, torch.zeros(2, 94, 1024), torch.zeros(94, dtype=torch.float64), draws=2)
