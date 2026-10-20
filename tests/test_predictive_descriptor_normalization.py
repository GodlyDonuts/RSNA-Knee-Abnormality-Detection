"""Standalone descriptor candidate mathematics; no fitting-data or pipeline changes."""

import hashlib
import json
import math

import pytest
import torch

from rsna.competitive.predictive.density import Supports, evidence
from rsna.competitive.predictive.descriptor_normalization import (
    STD_FLOOR,
    FrozenDescriptorNormalizer,
    StudyBalancedMoments,
)


def binding(n=2):
    return {"split": "fitting", "studies": n, "fitting_roster_sha256": "a"*64,
            "source_manifest_sha256": "b"*64, "descriptor_source_sha256": "c"*64}


def fit(studies):
    moments = StudyBalancedMoments(binding(len(studies)))
    for i, values in enumerate(studies):
        moments.add_study(i, values, torch.arange(len(values)), source_receipt_sha256=f"{i+1:064x}")
    return moments.finalize()


def test_equal_study_weight_population_variance_includes_between_study_variation():
    first, second = torch.zeros(16, 64), torch.full((64, 64), 4.)
    artifact = fit([first, second])
    assert artifact.mean == (2.,) * 64
    assert artifact.variance == (4.,) * 64
    assert artifact.scale == (2.,) * 64
    assert artifact.regions == 80
    # Region-weighted pooling would instead give3.2: longer exams do not dominate.
    assert torch.cat([first, second]).mean() != artifact.mean[0]


def test_streamed_FP64_moments_match_independent_population_mixture_without_cancellation():
    with torch.random.fork_rng():
        torch.manual_seed(721)
        studies = [1000 + torch.randn(n, 64) * .001 for n in (16, 48, 32)]
    artifact = fit(studies)
    means = torch.stack([x.double().mean(0) for x in studies])
    mean = means.mean(0)
    variance = torch.stack([((x.double() - mean)**2).mean(0) for x in studies]).mean(0)
    assert torch.allclose(torch.tensor(artifact.mean).double(), mean, atol=6e-5, rtol=0)
    assert torch.allclose(torch.tensor(artifact.variance).double(), variance, atol=1e-12, rtol=1e-6)
    # Verify stored FP64 values themselves, rather than FP32 tensor conversion.
    assert max(abs(a-b) for a, b in zip(artifact.mean, mean.tolist())) < 2e-13
    assert max(abs(a-b) for a, b in zip(artifact.variance, variance.tolist())) < 1e-16


def test_zero_variance_floor_finite_transform_and_immutable_private_roundtrip():
    values = torch.full((16, 64), .25)
    artifact = fit([values])
    assert artifact.scale == (STD_FLOOR,) * 64
    assert artifact.variance == (0.,) * 64
    normalized = artifact.transform_targets(values)
    assert torch.equal(normalized, torch.zeros_like(values))
    assert normalized.data_ptr() != values.data_ptr()
    body = artifact.to_bytes()
    digest = hashlib.sha256(body).hexdigest()
    restored = FrozenDescriptorNormalizer.from_bytes(body, expected_sha256=digest,
                                                     expected_binding=binding(1))
    assert restored == artifact
    assert restored.summary()["floored_coordinates"] == 64
    assert restored.summary()["artifact_sha256"] == digest
    assert not any(k in restored.summary() for k in ("mean", "variance", "scale"))
    copied_binding = artifact.binding
    copied_binding["studies"] = 99
    assert artifact.binding["studies"] == 1


@pytest.mark.parametrize("bad", ["unfinished", "duplicate_study", "out_of_order", "region_alias",
                                "missing_tile", "nan", "inf", "wrong_width", "requires_grad",
                                "heldout", "after_finalize"])
def test_refuse_partial_noncanonical_nonfinite_or_nonfitting_inputs(bad):
    values = torch.zeros(16, 64)
    if bad == "heldout":
        with pytest.raises(ValueError, match="fitting"):
            StudyBalancedMoments({**binding(1), "split": "validation"})
        return
    moments = StudyBalancedMoments(binding(2))
    ordinals = torch.arange(16)
    if bad == "unfinished":
        moments.add_study(0, values, ordinals, source_receipt_sha256="d"*64)
        with pytest.raises(ValueError, match="Complete declared"):
            moments.finalize()
        return
    if bad == "duplicate_study":
        moments.add_study(0, values, ordinals, source_receipt_sha256="d"*64)
    if bad == "region_alias":
        ordinals[-1] = 0
    if bad == "missing_tile":
        values, ordinals = values[:-1], ordinals[:-1]
    if bad in ("nan", "inf"):
        values[0, 0] = float(bad)
    if bad == "wrong_width":
        values = values[:, :63]
    if bad == "requires_grad":
        values.requires_grad_()
    if bad == "after_finalize":
        moments = StudyBalancedMoments(binding(1))
        moments.add_study(0, values, ordinals, source_receipt_sha256="d"*64)
        moments.finalize()
    with pytest.raises(ValueError):
        moments.add_study(1 if bad == "out_of_order" else 0, values, ordinals,
                          source_receipt_sha256="d"*64)


def test_rejected_study_does_not_partially_advance_stream():
    moments = StudyBalancedMoments(binding(1))
    invalid = torch.zeros(16, 64)
    invalid[2, 4] = float("nan")
    with pytest.raises(ValueError):
        moments.add_study(0, invalid, torch.arange(16), source_receipt_sha256="d"*64)
    moments.add_study(0, torch.zeros(16, 64), torch.arange(16), source_receipt_sha256="d"*64)
    assert moments.finalize().binding["studies"] == 1


@pytest.mark.parametrize("bad", ["hash", "full_cohort", "source", "floor", "scale", "unknown", "nan"])
def test_artifact_fails_closed_on_hash_scope_source_or_scale_changes(bad):
    artifact = fit([torch.zeros(16, 64)])
    body, expected = artifact.to_bytes(), binding(1)
    if bad == "hash":
        with pytest.raises(ValueError, match="SHA"):
            FrozenDescriptorNormalizer.from_bytes(body, expected_sha256="0"*64,
                                                   expected_binding=expected)
        return
    if bad == "full_cohort":
        expected = {**expected, "studies": 3887}
    elif bad == "source":
        expected = {**expected, "source_manifest_sha256": "f"*64}
    else:
        obj = json.loads(body)
        if bad == "floor":
            obj["std_floor"] = .01
        elif bad == "scale":
            obj["scale"][0] = .01
        elif bad == "unknown":
            obj["new_field"] = True
        elif bad == "nan":
            obj["mean"][0] = float("nan")
        body = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    with pytest.raises(ValueError):
        FrozenDescriptorNormalizer.from_bytes(body, expected_sha256=hashlib.sha256(body).hexdigest(),
                                               expected_binding=expected)


def test_masked_FP32_transform_preserves_censoring_and_ignores_only_unobserved_nan():
    artifact = fit([torch.zeros(16, 64)])
    values = torch.full((3, 64), .002)
    values[1] = float("nan")
    values[2] = .007
    observed = torch.tensor([True, False, False])
    original_mask = observed.clone()
    transformed = artifact.transform_targets(values, observed)
    assert torch.allclose(transformed[0], torch.full((64,), 2.))
    assert torch.isnan(transformed[1]).all() and torch.equal(transformed[2], values[2])
    assert torch.equal(observed, original_mask)
    with pytest.raises(ValueError, match="observed"):
        artifact.transform_targets(values)
    with pytest.raises(ValueError, match="FP32"):
        artifact.transform_targets(values.double(), observed)


def test_unit_normalized_gaussian_equals_raw_diagonal_variance_and_jacobian_cancels():
    with torch.random.fork_rng():
        torch.manual_seed(831)
        artifact = fit([torch.randn(16, 64) * .02 + .03, torch.randn(32, 64) * .01 + .04])
        raw_targets = torch.randn(1, 3, 64) * .02 + .03
        observed = torch.tensor([[True, False, True]])
        raw_targets[:, 1] = float("nan")
        z = artifact.transform_targets(raw_targets, observed)
        mu = (torch.randn(1, 2, 3, 64) * .1).requires_grad_()
        delta = (torch.randn_like(mu) * .03).requires_grad_()
        raw_mu, raw_delta = artifact.inverse_hypotheses(mu, delta)
        s = Supports(torch.tensor([0, 1, 2, 4]), torch.tensor([0, 1, 0, 2]),
                     torch.tensor([[-.5, -.7, -.3], [-.2, -.3, -.6]]), 3)
        normalized = evidence(z, mu, delta, torch.zeros(2, 64), s, observed)
        raw = evidence(raw_targets, raw_mu, raw_delta,
                       artifact.raw_log_variance().expand(2, -1), s, observed)
        assert torch.allclose(normalized.local_log_ratio, raw.local_log_ratio, atol=2e-6, rtol=1e-5)
        assert torch.allclose(normalized.log_bayes_factor, raw.log_bayes_factor, atol=2e-6, rtol=1e-5)
        jacobian = int(observed.sum()) * sum(math.log(scale) for scale in artifact.scale)
        assert torch.allclose(normalized.log_p0 - jacobian, raw.log_p0, atol=6e-5, rtol=1e-6)
        assert torch.allclose(normalized.log_p1 - jacobian, raw.log_p1, atol=6e-5, rtol=1e-6)
        assert torch.equal(normalized.no_observation_prior_mass, raw.no_observation_prior_mass)
        # The inverse map remains differentiable for hypothetical normalized prediction heads.
        g_z = torch.autograd.grad(normalized.log_bayes_factor.sum(), (mu, delta), retain_graph=True)
        g_raw = torch.autograd.grad(raw.log_bayes_factor.sum(), (mu, delta))
        assert all(torch.allclose(a, b, atol=4e-6, rtol=1e-5) for a, b in zip(g_z, g_raw))
        assert mu.grad is None and delta.grad is None


@pytest.mark.parametrize("no_observation", [False, True])
def test_neutral_shift_or_no_observations_remain_zero_evidence(no_observation):
    artifact = fit([torch.full((16, 64), .02)])
    targets = torch.full((1, 2, 64), .021)
    mask = torch.zeros(1, 2, dtype=torch.bool) if no_observation else torch.ones(1, 2, dtype=torch.bool)
    if no_observation:
        targets[:] = float("nan")
    mu = torch.ones(1, 1, 2, 64)
    delta = torch.ones_like(mu) if no_observation else torch.zeros_like(mu)
    s = Supports(torch.tensor([0, 1, 2]), torch.tensor([0, 1]), torch.zeros(1, 2), 2)
    raw_mu, raw_delta = artifact.inverse_hypotheses(mu, delta)
    for x, m, d, variance in (
        (artifact.transform_targets(targets, mask), mu, delta, torch.zeros(1, 64)),
        (targets, raw_mu, raw_delta, artifact.raw_log_variance()[None]),
    ):
        e = evidence(x, m, d, variance, s, mask)
        assert torch.equal(e.log_bayes_factor, torch.zeros_like(e.log_bayes_factor))
        assert torch.equal(e.local_log_ratio, torch.zeros_like(e.local_log_ratio))
