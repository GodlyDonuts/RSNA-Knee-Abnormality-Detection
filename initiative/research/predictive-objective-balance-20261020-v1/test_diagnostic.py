"""Small independent objective identities; no clinical or throughput acceptance."""

import importlib.util
import math
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("objective_diagnostic", HERE / "diagnostic.py")
D = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(D)


def test_exact_vector_decomposition_and_opposition():
    x = torch.tensor([2.0, 3.0], dtype=torch.float64, requires_grad=True)
    unused = torch.tensor([4.0], dtype=torch.float64, requires_grad=True)
    exam = 3 * x[0] + 4 * x[1]
    weighted = -4 * x[0] + 3 * x[1]
    result = D.gradient_balance(exam, weighted, exam + weighted, [("prior.x", x), ("negative.x", unused)])
    all_heads = result["all_nonencoder_parameters"]
    assert all_heads["exam_gradient_L2"] == 5
    assert all_heads["weighted_density_gradient_L2"] == 5
    assert all_heads["cosine"] == 0
    assert all_heads["total_gradient_L2"] == pytest.approx(math.sqrt(50))
    assert all_heads["sum_identity_max_abs_error"] == 0
    assert result["negative"]["weighted_density_over_exam_L2"] is None
    assert x.grad is None and unused.grad is None
    opposed = D.gradient_balance(exam, -exam, exam - exam, [("prior.x", x)])
    assert opposed["prior"]["cosine"] == -1
    assert opposed["prior"]["total_gradient_L2"] == 0


@pytest.mark.parametrize("label", [0.0, 0.3, 0.5, 1.0])
def test_supported_wrong_label_density_relation(label):
    q = torch.tensor([[label]], dtype=torch.float64)
    logp0 = torch.tensor([[-4.25]], dtype=torch.float64)
    logp1 = torch.tensor([[-2.5]], dtype=torch.float64)
    supported = -(q * logp1 + (1 - q) * logp0) / 128
    wrong = -((1 - q) * logp1 + q * logp0) / 128
    assert D.supported_margin(logp1 - logp0, q, torch.ones_like(q, dtype=torch.bool), 2, 64) \
        == pytest.approx(float(wrong - supported))
    assert D.supported_margin(logp1 - logp0, q, torch.zeros_like(q, dtype=torch.bool), 2, 64) is None
    assert D.supported_margin(logp1 - logp0, q, torch.ones_like(q, dtype=torch.bool), 0, 64) is None


def test_prior_loss_is_bce_of_mean_and_ignores_missing_values():
    logits = [torch.tensor([2.0, -1.0]), torch.tensor([-0.5, 3.0])]
    labels = torch.tensor([[0.25, float("nan")]])
    mask = torch.tensor([[True, False]])
    loss, probabilities = D.prior_exam(logits, labels, mask)
    p = (torch.sigmoid(logits[0][0]) + torch.sigmoid(logits[1][0])) / 2
    expected = -0.25 * torch.log(p) - 0.75 * torch.log1p(-p)
    assert float(loss) == pytest.approx(float(expected), abs=1e-7)
    assert float(probabilities[0, 0]) == pytest.approx(float(p))


def test_unit_variance_gaussian_constant_has_zero_gradient():
    mean = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    target = torch.tensor(0.04, dtype=torch.float64)
    density = 0.5 * ((target - mean).square() + math.log(2 * math.pi))
    gradient, = torch.autograd.grad(0.1 * density, [mean])
    assert float(density.detach()) - D.GAUSSIAN_CONSTANT == pytest.approx(0.0008)
    assert float(gradient) == pytest.approx(-0.004)


def test_neutral_decoder_density_can_supply_a_gradient_absent_from_bce():
    # Independent one-region, one-finding, one-descriptor Gaussian calculation.
    mu = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    shift = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    y = torch.tensor(0.04, dtype=torch.float64)
    bf = shift * (y - mu) - 0.5 * shift.square()
    exam = torch.nn.functional.softplus(-bf)
    density = 0.5 * (y - mu - shift).square() + D.GAUSSIAN_CONSTANT
    result = D.gradient_balance(
        exam, 0.1 * density, exam + 0.1 * density,
        [("negative.mu", mu), ("shift.delta", shift)],
    )
    assert result["negative"]["exam_gradient_L2"] == 0
    assert result["negative"]["weighted_density_gradient_L2"] == pytest.approx(0.004)
    assert result["shift"]["exam_gradient_L2"] == pytest.approx(0.02)
    assert result["shift"]["weighted_density_over_exam_L2"] == pytest.approx(0.2)


def test_measurement_runs_real_predictive_graph_without_parameter_or_gradient_writes():
    from rsna.competitive.predictive.density import Supports, two_mask_loss
    from rsna.competitive.predictive.model import Predictive

    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.project = torch.nn.Linear(3, 8)

        def forward(self, images):
            return self.project(images.mean((2, 3)))

    with torch.random.fork_rng():
        torch.manual_seed(381)
        model = Predictive(TinyEncoder(), 8, hidden=8).eval()
        for p in model.encoder.parameters():
            p.requires_grad_(False)
        with torch.no_grad():
            model.shift.weight.fill_(0.0001)
        original = {n: value.clone() for n, value in model.state_dict().items()}
        contexts = [
            {
                "images": torch.randn(2, 3, 2, 2),
                "token_metadata": torch.randn(2, 8),
                "region_metadata": torch.randn(3, 8),
                "targets": torch.randn(3, 64) * 0.04,
                "observed": torch.tensor([True, True, False]),
                "supports": Supports(
                    torch.tensor([0, 1, 3]), torch.tensor([0, 1, 2]), torch.zeros(12, 2), 3,
                ),
            }
            for _ in range(2)
        ]
        result = D.diagnose_case(
            model, contexts, torch.zeros(1, 12), torch.ones(1, 12, dtype=torch.bool),
            0, two_mask_loss,
        )
    assert set(result["gradient_balance"]) == {
        "all_nonencoder_parameters", "project", "position", "attention",
        "norm", "prior", "negative", "shift", "h",
    }
    assert result["gradient_balance"]["prior"]["weighted_density_gradient_L2"] == 0
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(value, original[n]) for n, value in model.state_dict().items())
    assert "probabilities" not in result and "labels" not in result
