"""Source-pixel causal interventions and independent dense-oracle comparisons."""

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from rsna.competitive.predictive.density import Supports, evidence, two_mask_loss
from rsna.competitive.predictive.inputs import fixed_metadata_scale, masked_study, triplets

P = Path(
    "initiative/research/predictive-evidence-proposal-20261020-v1/reference/predictive_evidence.py"
)
spec = importlib.util.spec_from_file_location("provided_predictive_dense_oracle", P)
oracle = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oracle
spec.loader.exec_module(oracle)


def test_source_target_intervention_cannot_change_any_overlapping_context_channel():
    g = torch.Generator().manual_seed(20)
    first = torch.randint(0, 4096, (64, 64), generator=g, dtype=torch.int32)
    other = torch.randint(0, 4096, (64, 64), generator=g, dtype=torch.int32)
    raw = torch.stack((first, other, first))
    frames = torch.stack([fixed_metadata_scale(x, bits_stored=12) for x in raw])
    alias = torch.tensor([0, 1, 0])
    acq = torch.zeros(3, dtype=torch.int64)
    order = torch.tensor([0, 1, 0])
    a = masked_study(frames, alias, acq, order, pattern=0, halo=2)
    perturbed = raw.clone()
    perturbed[a.raw_hidden] = 4095 - perturbed[a.raw_hidden]
    changed = torch.stack([fixed_metadata_scale(x, bits_stored=12) for x in perturbed])
    b = masked_study(changed, alias, acq, order, pattern=0, halo=2)
    assert torch.equal(a.context_frames, b.context_frames)
    assert not torch.equal(a.descriptors[a.observed], b.descriptors[b.observed])
    centers = torch.tensor([0, 1, 2])
    neighbors = torch.tensor([[0, 0, 1], [0, 1, 2], [1, 2, 2]])
    assert torch.equal(
        triplets(a, centers, neighbors, side=32), triplets(b, centers, neighbors, side=32)
    )
    assert len(a.descriptors) == 32  # Duplicate source frame contributes no extra region.
    complement = masked_study(frames, alias, acq, order, pattern=1, halo=2)
    assert torch.equal(a.observed, ~complement.observed)
    assert (~a.raw_hidden).any() and (~complement.raw_hidden).any()


def test_raw_pixels_cannot_replace_masked_input_and_cross_series_neighbors_refused():
    x = torch.rand(2, 64, 64)
    a = masked_study(
        x, torch.arange(2), torch.arange(2), torch.zeros(2, dtype=torch.int64), pattern=0, halo=2
    )
    with pytest.raises(ValueError):
        triplets(x, torch.tensor([0]), torch.tensor([[0, 0, 0]]))
    with pytest.raises(ValueError):
        triplets(a, torch.tensor([0]), torch.tensor([[0, 0, 1]]))


def fixture():
    g = torch.Generator().manual_seed(33)
    t = torch.randn(2, 5, 4, generator=g, dtype=torch.float64)
    mu = torch.randn(2, 2, 5, 4, generator=g, dtype=torch.float64, requires_grad=True)
    delta = torch.randn(2, 2, 5, 4, generator=g, dtype=torch.float64, requires_grad=True) * 0.1
    var = torch.zeros(2, 4, dtype=torch.float64)
    observed = torch.tensor([[1, 0, 1, 0, 1], [0, 0, 0, 0, 0]], dtype=torch.bool)
    supports = Supports(torch.arange(6), torch.arange(5), torch.zeros(2, 5, dtype=torch.float64), 5)
    return t, mu, delta, var, supports, observed


def test_sparse_complete_roster_density_and_gradient_equal_independent_provided_oracle():
    t, mu, delta, var, s, o = fixture()
    a = evidence(t, mu, delta, var, s, o, edge_chunk=2)
    dense = torch.eye(5, dtype=torch.bool)[None].expand(2, -1, -1)
    b = oracle.predictive_evidence(t, mu, delta, var, dense, s.log_prior, o)
    for key in (
        "log_p0",
        "log_p1",
        "log_bayes_factor",
        "local_log_ratio",
        "no_observation_prior_mass",
    ):
        torch.testing.assert_close(getattr(a, key), getattr(b, key), atol=1e-12, rtol=1e-12)
    left = torch.autograd.grad(a.log_p1.sum(), (mu, delta), retain_graph=True)
    right = torch.autograd.grad(b.log_p1.sum(), (mu, delta))
    for x, y in zip(left, right):
        torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-12)
    assert a.log_bayes_factor[1].abs().max() < 1e-12
    assert (a.log_bayes_factor[0] >= a.no_observation_prior_mass[0].log() - 1e-12).all()


def test_repeated_evidence_inside_support_refused_and_alternative_split_invariant():
    t, mu, d, var, s, o = fixture()
    bad = Supports(torch.tensor([0, 2]), torch.tensor([0, 0]), torch.zeros(2, 1, dtype=t.dtype), 5)
    with pytest.raises(ValueError):
        evidence(t, mu, d, var, bad, o)
    split = Supports(
        torch.arange(11), torch.arange(5).repeat_interleave(2), torch.zeros(2, 10, dtype=t.dtype), 5
    )
    torch.testing.assert_close(
        evidence(t, mu, d, var, s, o).log_bayes_factor,
        evidence(t, mu, d, var, split, o).log_bayes_factor,
    )


def test_training_uses_BCE_of_mean_posterior_not_mean_BCE():
    t, mu, d, var, s, o = fixture()
    a = evidence(t, mu, d, var, s, o)
    z = torch.tensor(
        [[[2.0, -1.0], [-2.0, 3.0]], [[0.0, 1.0], [2.0, -3.0]]], dtype=t.dtype, requires_grad=True
    )
    q = torch.tensor([[0.2, 0.8], [0.4, 0.6]], dtype=t.dtype)
    m = torch.ones_like(q, dtype=torch.bool)
    loss, out = two_mask_loss([a, a], z, q, m, [o, o], descriptor_width=4, density_weight=0)
    p = (z + torch.stack([a.log_bayes_factor, a.log_bayes_factor], 1)).sigmoid().mean(1)
    torch.testing.assert_close(loss, torch.nn.functional.binary_cross_entropy(p, q))
    loss.backward()
    assert torch.isfinite(z.grad).all()
    torch.testing.assert_close(out["probabilities"], p)
