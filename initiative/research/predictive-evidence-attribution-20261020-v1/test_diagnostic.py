"""Synthetic numerical and one-use runner checks; no actual MRI or remote calls."""

import copy
import importlib.util
import json
import math
import os
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

HERE = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, HERE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = load("attribution_test_diagnostic", "diagnostic.py")
R = load("attribution_test_runner", "run_once.py")


def supports(classes=2):
    from rsna.competitive.predictive.density import Supports
    return Supports(torch.tensor([0, 1, 3, 4, 7]),
                    torch.tensor([0, 0, 1, 2, 0, 1, 2]),
                    torch.tensor([[-0.2, -1.0, -2.0, -0.4]], dtype=torch.float64)
                    .expand(classes, -1).clone(), 3)


def oracle_log_mixture(scores, log_prior, beta):
    # Independent scalar Python oracle, no tensor aggregation or analytic derivative.
    pmax = max(log_prior)
    denominator = pmax + math.log(sum(math.exp(x - pmax) for x in log_prior))
    terms = [p - denominator + beta * s for p, s in zip(log_prior, scores)]
    maximum = max(terms)
    return maximum + math.log(sum(math.exp(x - maximum) for x in terms))


def test_sparse_scores_and_analytic_beta_derivatives_against_independent_oracle():
    local = torch.tensor([[0.4, -0.2, 999.0], [-0.7, 0.3, -999.0]], dtype=torch.float64)
    s, observed = supports(), torch.tensor([True, True, False])
    scores, touched = D.support_scores(local, s, observed)
    assert touched.tolist() == [1, 2, 0, 2]
    assert torch.allclose(scores, torch.tensor([[0.4, 0.2, 0, 0.2],
                                               [-0.7, -0.4, 0, -0.4]], dtype=torch.float64),
                          atol=1e-15, rtol=0)
    logbf, prior, posterior, derivative, curvature = D.mixture_at_unit_precision(
        scores, s.log_prior,
    )
    assert torch.allclose(prior.sum(1), torch.ones(2, dtype=torch.float64))
    assert torch.allclose(posterior.sum(1), torch.ones(2, dtype=torch.float64))
    for i in range(2):
        def value(beta):
            return oracle_log_mixture(scores[i].tolist(), s.log_prior[i].tolist(), beta)
        h = 1e-4
        assert float(logbf[i]) == pytest.approx(value(1), abs=2e-15)
        assert float(derivative[i]) == pytest.approx((value(1+h)-value(1-h))/(2*h), abs=2e-10)
        assert float(curvature[i]) == pytest.approx((value(1+h)-2*value(1)+value(1-h))/h**2,
                                                  abs=1e-7)


def test_unobserved_alternatives_keep_prior_mass_and_zero_beta_derivative():
    s = supports()
    scores, touched = D.support_scores(torch.ones(2, 3, dtype=torch.float64), s,
                                      torch.zeros(3, dtype=torch.bool))
    logbf, prior, posterior, derivative, curvature = D.mixture_at_unit_precision(
        scores, s.log_prior,
    )
    assert not touched.any()
    assert torch.allclose(prior, posterior)
    assert torch.allclose(logbf, torch.zeros_like(logbf), atol=1e-15)
    assert torch.equal(derivative, torch.zeros_like(derivative))
    assert torch.equal(curvature, torch.zeros_like(curvature))


def test_coordinate_statistics_and_mean_intervention_preserve_originals():
    targets = torch.tensor([0.01, 0.03, 0.9], dtype=torch.float32)[:, None].expand(3, 64).clone()
    observed = torch.tensor([True, True, False])
    original = targets.clone()
    mu = torch.full((2, 3, 64), 0.01)
    delta = torch.full_like(mu, 0.01)
    result = D.descriptor_statistics(targets, mu, delta, observed)
    for block in result["blocks"].values():
        assert block["target_mean"] == pytest.approx(0.02)
        assert block["target_variance_across_observed_regions"] == pytest.approx(0.0001)
        assert block["negative_residual_MSE_across_classes_and_regions"] == pytest.approx(0.0002)
        assert block["positive_residual_MSE_across_classes_and_regions"] == pytest.approx(0.0001)
    changed = D.mean_intervention(targets, observed)
    assert torch.equal(targets, original) and changed.data_ptr() != targets.data_ptr()
    assert torch.equal(changed[~observed], targets[~observed])
    assert torch.equal(changed[observed], targets[observed].mean(0).expand(2, -1))


def test_full_callback_uses_frozen_predictions_once_and_never_mutates_model_or_inputs():
    from rsna.competitive.predictive.density import two_mask_loss
    from rsna.competitive.predictive.model import Predictive

    class TinyEncoder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.project = torch.nn.Linear(3, 8)
            self.calls = 0

        def forward(self, images):
            self.calls += 1
            return self.project(images.mean((2, 3)))

    with torch.random.fork_rng():
        torch.manual_seed(391)
        model = Predictive(TinyEncoder(), 8, hidden=8).eval()
        with torch.no_grad():
            model.shift.weight.normal_(std=0.001)
            model.negative.weight.normal_(std=0.001)
        original = {name: value.clone() for name, value in model.state_dict().items()}
        contexts = [{
            "images": torch.rand(2, 3, 2, 2), "token_metadata": torch.rand(2, 8),
            "region_metadata": torch.rand(3, 8), "targets": torch.rand(3, 64) * 0.04,
            "observed": torch.tensor([True, True, False]), "supports": supports(12),
        } for _ in range(2)]
        # Original forward requires support priors and descriptors to share FP32 dtype.
        contexts = [{**c, "supports": type(c["supports"])(
            c["supports"].indptr, c["supports"].regions, c["supports"].log_prior.float(), 3,
        )} for c in contexts]
        saved_inputs = copy.deepcopy(contexts)
        rng = torch.random.get_rng_state().clone()
        labels, mask = torch.full((1, 12), 0.3), torch.ones(1, 12, dtype=torch.bool)
        result = D.diagnose_case(model, contexts, labels, mask, 0, two_mask_loss)
        assert torch.equal(torch.random.get_rng_state(), rng)
        assert model.encoder.calls == 2
        assert all(p.grad is None for p in model.parameters())
        assert all(torch.equal(value, original[name]) for name, value in model.state_dict().items())
        for current, saved in zip(contexts, saved_inputs):
            for name in ("images", "token_metadata", "region_metadata", "targets", "observed"):
                assert torch.equal(current[name], saved[name])
    assert len(result["contexts"]) == 2
    assert result["contexts"][0]["density_normalization_N_times_64"] == 128
    for context in result["contexts"]:
        decomposition = context["support_statistics"]["support_score_decomposition"]
        assert decomposition["maximum_sum_identity_error"] <= 1e-10
        assert decomposition["support_terms_are_additive_but_mixture_logBF_terms_are_not"] is True
    assert result["readout_ablations"]["full"]["mean_absolute_change_from_full"] == 0
    assert result["readout_ablations"]["within_case_observed_coordinate_mean"][
        "mean_absolute_change_from_full"] > 0
    assert result["raw_labels_descriptors_or_prediction_rows_exported"] is False
    assert "labels" not in result and "probabilities" not in result
    json.dumps(result, allow_nan=False)
    with pytest.raises(ValueError, match="two fitting"):
        D.diagnose_case(model, contexts, labels, mask, 12, two_mask_loss)


def test_loader_derivation_keeps_all_original_admission_and_state_checks():
    original = (HERE.parent / "predictive-objective-balance-20261020-v1/diagnostic.py").read_text()
    changed = D.derive_loader(original)
    assert changed.replace('os.environ["SLURM_CPUS_PER_TASK"] == "1"',
                           'os.environ["SLURM_CPUS_PER_TASK"] == "2"').replace(
                               "torch.set_num_threads(1)", "torch.set_num_threads(2)") == original
    assert "for ordinal in (0, 1)" in changed
    assert "torch.equal(v, original_state[n])" in changed
    assert "stat(checkpoint) == before" in changed
    with pytest.raises(ValueError, match="CPU981494"):
        D.derive_loader(original + "\n")


def test_runner_is_one_cpu_no_gpu_no_shared_logs_and_source_uses_stdin():
    command = R.command("/tmp/synthetic.sock", "SOURCE_FROM_STDIN")
    remote = shlex.split(command[-1])
    assert command[:2] == ["ssh", "-S"]
    assert "--cpus-per-task=1" in remote and "--gres=none" in remote
    assert "--mem=12G" in remote and "--time=00:30:00" in remote
    assert "--chdir=/tmp" in remote and "OMP_NUM_THREADS=1" in remote
    assert "CUDA_VISIBLE_DEVICES=" in remote
    assert not any(x.startswith(("--output", "--error")) for x in remote)
    assert remote[-3:] == ["-B", "-c", "SOURCE_FROM_STDIN"]


def test_failed_single_invocation_cannot_allocate_again(monkeypatch, tmp_path):
    monkeypatch.setattr(R, "ROOT", tmp_path)
    (tmp_path / "artifacts").mkdir()
    monkeypatch.setattr(sys, "argv", ["run_once.py", "--source-git", "a" * 40])
    monkeypatch.setattr(R, "prepare", lambda pin: (
        {"diagnostic.py": b"pass", "run_once.py": b"pass"}, {"fitting_indices": [0, 1]},
        SimpleNamespace(COMMAND="stdin_only", envelope=lambda *args: "{}"),
    ))
    calls = []

    def run(*args, **kwargs):
        calls.append(args)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(R.subprocess, "run", run)
    previous = os.umask(0o077)
    try:
        with pytest.raises(RuntimeError, match="no retry"):
            R.main()
        with pytest.raises(FileExistsError):
            R.main()
    finally:
        os.umask(previous)
    assert len(calls) == 1
    journal = tmp_path / "artifacts" / HERE.name
    intent = json.loads((journal / "intent.json").read_bytes())
    assert intent["automatic_retry"] is False
    assert intent["fitting_indices"] == [0, 1]
    assert json.loads((journal / "result.json").read_bytes())["exit_code"] == 1
