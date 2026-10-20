"""Independent math and frozen-state tests; no MRI, account or remote execution."""

import copy
import importlib.util
import math
from pathlib import Path

import pytest
import torch

from rsna.competitive.predictive.density import Supports, evidence, two_mask_loss
from rsna.competitive.predictive.model import Predictive

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = load("density_decomposition_test", HERE / "diagnostic.py")
P = load("density_decomposition_prepare_test", HERE / "prepare.py")


@pytest.mark.parametrize("q", [0.0, 0.3, 0.5, 1.0])
def test_exact_loss_components_signed_margin_and_analytic_output_gradients(q):
    # Independent scalar one-support Gaussian oracle, not the decomposition implementation.
    mu = torch.tensor([[[[0.02], [-0.01]]]], dtype=torch.float64, requires_grad=True)
    delta = torch.tensor([[[[0.04], [0.03]]]], dtype=torch.float64, requires_grad=True)
    target = torch.tensor([[[0.11], [0.07]]], dtype=torch.float64)
    prior = torch.tensor([[[0.2], [-0.1]]], dtype=torch.float64, requires_grad=True)
    observed = torch.tensor([[True, True]])
    supports = Supports(torch.tensor([0, 2]), torch.tensor([0, 1]), torch.zeros(1, 1).double(), 2)
    e = evidence(target, mu, delta, torch.zeros(1, 1).double(), supports, observed)
    labels, mask = torch.tensor([[q]], dtype=torch.float64), torch.tensor([[True]])
    _, metrics = two_mask_loss([e, e], prior, labels, mask, [observed, observed],
                               descriptor_width=1, density_weight=100)
    a, d, summaries = D.components([e, e], labels, mask, [observed, observed], 1)
    residual = [0.09, 0.08]
    s = sum(dd * rr - .5 * dd * dd for dd, rr in zip([.04, .03], residual))
    assert float(a.detach()) == pytest.approx(
        CONSTANT := .5 * math.log(2 * math.pi) + sum(r * r for r in residual) / 4)
    assert CONSTANT > 0
    assert float(d.detach()) == pytest.approx(-q * s / 2)
    assert float((a + d).detach()) == pytest.approx(float(metrics["density_loss"].detach()))
    assert summaries[0]["signed_BF_margin_unnormalized"] == pytest.approx((2*q - 1) * s)
    assert summaries[0]["wrong_minus_supported_density_NLL"] == pytest.approx((2*q - 1) * s/2)
    gmu, gdelta = torch.autograd.grad(a + d, [mu, delta], retain_graph=True)
    assert torch.allclose(gmu.flatten(), torch.tensor(
        [(-r + q * dd)/2 for r, dd in zip(residual, [.04, .03])], dtype=torch.float64))
    assert torch.allclose(gdelta.flatten(), torch.tensor(
        [-q*(r - dd)/2 for r, dd in zip(residual, [.04, .03])], dtype=torch.float64))
    measured = D.gradient_components(metrics["exam_loss"], a, d, metrics["density_loss"],
                                     [("negative.mu", mu), ("shift.delta", delta), ("prior.a", prior)])
    assert measured["all_nonencoder_parameters"]["gradient_sum_absolute_L2_error"] < 1e-15
    assert measured["shift"]["common_A_L2"] == 0
    if q == 0:
        assert measured["shift"]["label_weighted_D_L2"] == 0
    assert all(x.grad is None for x in (mu, delta, prior))


def test_zero_observation_context_and_missing_label_have_original_active_denominator():
    target = torch.full((1, 2, 1), .1, dtype=torch.float64)
    mu = torch.zeros(1, 2, 2, 1, dtype=torch.float64, requires_grad=True)
    delta = torch.full_like(mu, .03, requires_grad=True)
    s = Supports(torch.tensor([0, 2]), torch.tensor([0, 1]), torch.zeros(2, 1).double(), 2)
    observations = [torch.tensor([[True, True]]), torch.tensor([[False, False]])]
    ev = [evidence(target, mu, delta, torch.zeros(2, 1).double(), s, o) for o in observations]
    labels = torch.tensor([[.2, float("nan")]], dtype=torch.float64)
    mask = torch.tensor([[True, False]])
    _, metrics = two_mask_loss(ev, torch.zeros(1, 2, 2).double(), labels, mask, observations,
                               descriptor_width=1)
    a, d, rows = D.components(ev, labels, mask, observations, 1)
    assert [r["active_labels"] for r in rows] == [1, 0]
    assert [r["density_denominator"] for r in rows] == [2, 0]
    assert float((a+d).detach()) == pytest.approx(float(metrics["density_loss"].detach()))
    assert float(a.detach()) == pytest.approx(.5 * (.01 + math.log(2 * math.pi)))


def toy():
    class Encoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.project = torch.nn.Linear(3, 8)
            self.calls = 0

        def forward(self, images):
            self.calls += 1
            return self.project(images.mean((2, 3)))

    model = Predictive(Encoder(), 8, hidden=8).eval()
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    with torch.no_grad():
        model.shift.weight.normal_(std=.001)
        model.negative.weight.normal_(std=.001)
    contexts = [{
        "images": torch.rand(2, 3, 2, 2), "token_metadata": torch.rand(2, 8),
        "region_metadata": torch.rand(3, 8), "targets": torch.rand(3, 64) * .04,
        "observed": torch.tensor([True, True, False]),
        "supports": Supports(torch.tensor([0, 1, 3]), torch.tensor([0, 1, 2]),
                             torch.zeros(12, 2), 3),
    } for _ in range(2)]
    return model, contexts


def test_matched_original_graph_preserves_inputs_rng_state_grad_and_uses_only_four_forwards():
    with torch.random.fork_rng():
        torch.manual_seed(567)
        model, contexts = toy()
        original = copy.deepcopy(model.state_dict())
        other = copy.deepcopy(original)
        other["negative.bias"] += .002
        other["shift.bias"] -= .001
        inputs = copy.deepcopy(contexts)
        rng = torch.random.get_rng_state().clone()
        labels, mask = torch.full((1, 12), .3), torch.ones(1, 12, dtype=torch.bool)
        out = D.matched_case(model, contexts, labels, mask, 0, two_mask_loss, other)
        assert model.encoder.calls == 4  # two contexts for each actual model; intervention reuses heads.
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert all(torch.equal(v, original[n]) for n, v in model.state_dict().items())
        assert all(p.grad is None for p in model.parameters())
        for now, before in zip(contexts, inputs):
            for name in ("images", "token_metadata", "region_metadata", "targets", "observed"):
                assert torch.equal(now[name], before[name])
        assert out["C24"]["density_weight"] == .1 and out["C10024"]["density_weight"] == 100
        assert out["C24"]["common_A"] != out["C10024"]["common_A"]
        for name in ("C24", "C10024"):
            result = out[name]
            assert result["A_plus_D_loss_absolute_error"] < 2e-6
            assert result["regional_mean_intervention_probability_mean_absolute_change"] > 0
            assert result["raw_labels_descriptors_predictions_exported"] is False
            assert "labels" not in result and "probabilities" not in result


def test_failed_matched_load_restores_original_values():
    model, contexts = toy()
    original = copy.deepcopy(model.state_dict())
    incompatible = copy.deepcopy(original)
    incompatible["negative.bias"] += .2
    incompatible["shift.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError):
        D.matched_case(model, contexts, torch.ones(1, 12), torch.ones(1, 12, dtype=torch.bool),
                       0, two_mask_loss, incompatible)
    assert all(torch.equal(v, original[n]) for n, v in model.state_dict().items())


def pilot_fixture():
    accepted = {
        "status": "actual_C100_full_stop_resume_engineering_complete",
        "GPU_job": "981620", "source_git": D.PILOT_SOURCE,
        "binding_sha256": D.PILOT_BINDING, "branch": "C100",
        "full_stop_resume_complete_state_and_prediction_bytes_equal": True,
        "objective_measurement_sha256": D.PRIOR_SHA,
        "terminal": {"density_weight": 100, "steps": 24},
        "real_optimizer_history": [{"step": n} for n in range(1, 25)],
    }
    original = {"source_git": "original_science", "code_sha256": {"rsna/a.py": "exact"}}
    binding = {
        "source_git": D.PILOT_SOURCE, "branch": "C100",
        "original_numerical_source_git": "original_science",
        "objective_measurement_sha256": D.PRIOR_SHA,
        "original_module_sha256": {"rsna/a.py": "exact"},
    }
    for name in ("targets16", "context_index"):
        for suffix in ("_private_path", "_sha256"):
            original[name + suffix] = binding[name + suffix] = name + suffix
    scheduler = {"GPU_job": "981620", "state": "COMPLETED", "exit_code": "0:0", "node": "evc27"}
    return accepted, scheduler, binding, original


@pytest.mark.parametrize("bad", ["pending", "different_inputs", "different_modules", "wrong_checkpoint"])
def test_actual_pilot_binding_rejects_pending_or_nonmatched_evidence(bad):
    accepted, scheduler, binding, original = pilot_fixture()
    D.validate_pilot(accepted, scheduler, binding, original)
    if bad == "pending":
        scheduler["state"] = "PENDING"
    elif bad == "different_inputs":
        binding["targets16_sha256"] = "other"
    elif bad == "different_modules":
        binding["original_module_sha256"] = {"rsna/a.py": "changed"}
    else:
        accepted["terminal"]["steps"] = 23
    with pytest.raises(ValueError):
        D.validate_pilot(accepted, scheduler, binding, original)


def test_preparation_refuses_pending_before_source_access_or_any_execution(monkeypatch):
    calls = []
    monkeypatch.setattr(P, "git_file", lambda *a: calls.append(a))
    reference = {
        "acceptance_sha256": "a" * 64, "terminal_sha256": "b" * 64,
        "checkpoint_stat": [1, 2, 3, 4], "scheduler": {
            "GPU_job": "981620", "state": "PENDING", "exit_code": "0:0", "node": "evc27",
        },
    }
    with pytest.raises(ValueError, match="Pending"):
        P.prepare("c" * 40, reference)
    assert calls == []


def test_baseline_capsule_has_only_exact_original_receipts_and_no_C100_claim(monkeypatch):
    paths = []

    def local_source(_pin, path):
        paths.append(path)
        return path.read_bytes()

    monkeypatch.setattr(P, "git_file", local_source)
    capsule = P.prepare_baseline("a" * 40)
    payload = capsule["payload"]
    assert payload["execution_mode"] == "C24_only"
    assert payload["fitting_indices"] == [0, 1]
    assert not any(k.startswith("C100") for k in payload)
    assert capsule["execution"]["comparison_scope"] == "C24_only_no_C100_model_loaded"
    assert capsule["execution"]["CPUs"] == 1 and capsule["execution"]["GPUs"] == 0
    assert P.digest(payload["prior_result_text"].encode()) == D.PRIOR_SHA
    assert P.digest(payload["original_loader_source"].encode()) == D.LOADER_SHA
    assert len(paths) == 3  # two new helper sources plus the pinned existing loader.


def test_baseline_model_has_two_forwards_no_new_state_load_and_no_C100_result(monkeypatch):
    model, contexts = toy()
    original = copy.deepcopy(model.state_dict())

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Baseline must not load any second model")

    monkeypatch.setattr(model, "load_state_dict", forbidden)
    result = D.baseline_case(model, contexts, torch.full((1, 12), .3),
                             torch.ones(1, 12, dtype=torch.bool), 0, two_mask_loss)
    assert model.encoder.calls == 2
    assert set(result) == {"fitting_ordinal", "C24"}
    assert result["C24"]["density_weight"] == .1
    assert all(p.grad is None for p in model.parameters())
    assert all(torch.equal(v, original[n]) for n, v in model.state_dict().items())


def test_baseline_rejects_C100_reference_or_unknown_scope_before_any_loader(monkeypatch):
    with pytest.raises(ValueError, match="without C100"):
        D.run_baseline({"execution_mode": "C24_only", "C100_reference": {}})
    with pytest.raises(ValueError, match="Explicit C24-only"):
        D.run_baseline({})
    with pytest.raises(ValueError, match="Unknown"):
        D.execute({"execution_mode": "C100_only"})
    calls = []
    monkeypatch.setattr(D, "run", lambda payload: calls.append("matched"))
    monkeypatch.setattr(D, "run_baseline", lambda payload: calls.append("baseline"))
    D.execute({"C100_reference": {}})
    D.execute({"execution_mode": "C24_only"})
    assert calls == ["matched", "baseline"]


def test_cached_image_exact_fill_summary_is_descriptive_and_preserves_source():
    values = torch.tensor([.5, .5, 0., .25, .75, 1.]).reshape(1, 1, 2, 3)
    original = values.clone()
    result = D.cached_image_statistics(values)
    assert result["fraction_elements_exact_fill_0.5"] == pytest.approx(1/3)
    assert result["nonfill_elements"] == 4
    s = result["nonfill_summary"]
    assert s["mean"] == .5 and s["minimum"] == 0 and s["maximum"] == 1
    assert s["std_population"] == pytest.approx(math.sqrt(.15625))
    assert [s[f"quantile_{q}"] for q in ("0.01", "0.5", "0.99")] \
        == pytest.approx([.0075, .5, .9925])
    assert torch.equal(values, original)
    assert "not an anatomical visibility mask" in result["interpretation"]
    assert result["raw_image_values_exported"] is False
    all_fill = D.cached_image_statistics(torch.full_like(values, .5))
    assert all_fill["fraction_elements_exact_fill_0.5"] == 1
    assert all_fill["nonfill_summary"] is None and all_fill["nonfill_elements"] == 0
