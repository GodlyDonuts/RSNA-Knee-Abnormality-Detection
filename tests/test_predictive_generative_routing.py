"""Numerical routing tests, including original evidence and actual shared encoder graphs."""

import copy
import csv
import io

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from rsna.competitive.predictive.density import Supports, evidence, two_mask_loss
from rsna.competitive.predictive.generative_routing import generative_two_mask_loss
from rsna.competitive.predictive.model import Predictive


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def example(dtype=torch.float64, q=0.35, no_observations=False):
    generator = torch.Generator().manual_seed(173)
    target = torch.randn(1, 4, 2, generator=generator, dtype=dtype) * 0.2
    means = [torch.randn(1, 3, 4, 2, generator=generator, dtype=dtype).requires_grad_()
             for _ in range(2)]
    shifts = [(torch.randn(1, 3, 4, 2, generator=generator, dtype=dtype) * 0.1)
              .requires_grad_() for _ in range(2)]
    observations = [torch.tensor([[True, False, True, False]]),
                    torch.tensor([[False, True, False, True]])]
    if no_observations:
        observations = [torch.zeros_like(o) for o in observations]
        target.fill_(float("nan"))  # Unobserved targets must not enter either objective.
    supports = Supports(torch.tensor([0, 2, 4, 8]),
                        torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
                        torch.zeros(3, 3, dtype=dtype), 4)
    ev = [evidence(target, mu, delta, torch.zeros(3, 2, dtype=dtype), supports, observed)
          for mu, delta, observed in zip(means, shifts, observations)]
    prior = torch.tensor([[[0.8, -0.7, 0.1], [-0.5, 0.6, 0.2]]],
                         dtype=dtype, requires_grad=True)
    labels = torch.tensor([[q, 1 - q, float("nan")]], dtype=dtype)
    mask = torch.tensor([[True, True, False]])
    return ev, prior, labels, mask, observations, means + shifts


def loss(fn, data, weight):
    return fn(*data[:5], descriptor_width=2, density_weight=weight)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("weight", [0.1, 100.0])
def test_forward_values_are_bitwise_original_and_evidence_graphs_untouched(dtype, weight):
    data = example(dtype)
    ev = data[0]
    original_refs = [(e.log_p0, e.log_p1, e.log_bayes_factor) for e in ev]
    original, om = loss(two_mask_loss, data, weight)
    routed, rm = loss(generative_two_mask_loss, data, weight)
    assert torch.equal(original, routed)
    assert om.keys() == rm.keys()
    assert all(torch.equal(om[k], rm[k]) for k in om)
    for e, refs in zip(ev, original_refs):
        assert all(now is before for now, before in zip(
            (e.log_p0, e.log_p1, e.log_bayes_factor), refs))
        assert all(t.requires_grad for t in refs)
    # Original inference still propagates through the original Bayes factors.
    grads = torch.autograd.grad(sum(e.log_bayes_factor.sum() for e in ev), data[-1])
    assert all(torch.count_nonzero(g) > 0 for g in grads)


@pytest.mark.parametrize("q", [0.0, 0.35, 1.0])
def test_mean_posterior_soft_label_oracle_and_density_only_hypothesis_gradients(q):
    data = example(q=q)
    _, om = loss(two_mask_loss, data, 100.0)
    total, rm = loss(generative_two_mask_loss, data, 100.0)
    ev, prior, labels, mask, _, heads = data
    z = prior + torch.stack([e.log_bayes_factor for e in ev], 1)
    probability = z.sigmoid().mean(1)
    expected = F.binary_cross_entropy(probability[mask], labels[mask])
    torch.testing.assert_close(rm["exam_loss"], expected, rtol=1e-14, atol=1e-14)
    wrong = sum(F.binary_cross_entropy(z[:, i].sigmoid()[mask], labels[mask])
                for i in (0, 1)) / 2
    assert abs(float((expected - wrong).detach())) > 1e-4
    assert all(g is None for g in torch.autograd.grad(
        rm["exam_loss"], heads, allow_unused=True, retain_graph=True))
    total_grad = torch.autograd.grad(total, heads, retain_graph=True)
    density_grad = torch.autograd.grad(100 * om["density_loss"], heads, retain_graph=True)
    for actual, expected_grad in zip(total_grad, density_grad):
        assert torch.equal(actual, expected_grad)
        assert torch.count_nonzero(actual[:, 2]) == 0  # Missing class contributes nothing.
    original_prior = torch.autograd.grad(om["exam_loss"], prior, retain_graph=True)[0]
    routed_prior = torch.autograd.grad(total, prior, retain_graph=True)[0]
    assert torch.equal(original_prior, routed_prior)
    # Independent two-mask derivative, including soft q and label averaging.
    safe_q = torch.where(mask, labels, 0)
    expected_prior = ((probability - safe_q) / (probability * (1 - probability)))[:, None]
    expected_prior = expected_prior * z.sigmoid() * (1 - z.sigmoid()) / (2 * mask.sum())
    expected_prior = expected_prior.masked_fill(~mask[:, None], 0)
    torch.testing.assert_close(routed_prior, expected_prior, rtol=1e-13, atol=1e-13)


@pytest.mark.parametrize("empty", ["one_context", "both_contexts", "all_labels"])
def test_no_observations_and_missing_labels_keep_original_zero_density_semantics(empty):
    data = example(no_observations=empty == "both_contexts")
    ev, prior, labels, mask, observations, heads = data
    if empty == "one_context":
        observations[1].zero_()
        ev[1] = evidence(torch.full((1, 4, 2), float("nan"), dtype=torch.float64),
                         heads[1], heads[3], torch.zeros(3, 2).double(),
                         Supports(torch.tensor([0, 4]), torch.arange(4),
                                  torch.zeros(3, 1).double(), 4), observations[1])
    if empty == "all_labels":
        mask.zero_()
        labels.fill_(float("nan"))
    total, measured = loss(generative_two_mask_loss, data, 100.0)
    reference, rm = loss(two_mask_loss, data, 100.0)
    assert torch.equal(total, reference) and torch.isfinite(total)
    assert all(torch.equal(measured[k], rm[k]) for k in rm)
    if empty != "one_context":
        assert measured["density_loss"] == 0
        head_grad = torch.autograd.grad(total, heads, allow_unused=True)
        assert all(g is None or torch.count_nonzero(g) == 0 for g in head_grad)
    if empty == "both_contexts":
        torch.testing.assert_close(measured["probabilities"], prior.sigmoid().mean(1))
    if empty == "all_labels":
        assert total == 0


def csv_bytes(probability):
    handle = io.StringIO(newline="")
    writer = csv.writer(handle)
    writer.writerow([f"target{i}" for i in range(12)])
    writer.writerow([format(x, ".9g") for x in probability.detach().flatten().tolist()])
    return handle.getvalue().encode()


def test_real_predictive_graph_keeps_encoder_both_paths_prior_and_exact_csv():
    with torch.random.fork_rng():
        torch.manual_seed(71)
        model = Predictive(nn.Sequential(nn.Flatten(), nn.Linear(12, 8)), 8,
                           hidden=8, descriptor_width=2).train()
        with torch.no_grad():
            # Nonneutral learned state exercises both paths, unlike zero-head startup.
            model.negative.weight.normal_(std=0.03)
            model.shift.weight.normal_(std=0.04)
        original_state = copy.deepcopy(model.state_dict())
        supports = Supports(torch.tensor([0, 2, 4]), torch.tensor([0, 1, 2, 3]),
                            torch.zeros(12, 2), 4)
        contexts = [dict(images=torch.rand(2, 3, 2, 2), token_metadata=torch.rand(2, 8),
                         region_metadata=torch.rand(4, 8), targets=torch.rand(4, 2),
                         observed=torch.tensor(o), supports=supports)
                    for o in ([True, False, True, False], [False, True, False, True])]
        output = [model(**c) for c in contexts]
        ev = [o["evidence"] for o in output]
        prior = torch.stack([o["context_logits"] for o in output])[None]
        labels, mask = torch.full((1, 12), 0.3), torch.ones(1, 12, dtype=torch.bool)
        args = (ev, prior, labels, mask, [c["observed"][None] for c in contexts])
        total, routed = generative_two_mask_loss(*args, descriptor_width=2, density_weight=100)
        reference, original = two_mask_loss(*args, descriptor_width=2, density_weight=100)
        assert torch.equal(total, reference)
        assert csv_bytes(routed["probabilities"]) == csv_bytes(original["probabilities"])
        parameters = tuple(model.parameters())
        gradients = [torch.autograd.grad(term, parameters, allow_unused=True, retain_graph=True)
                     for term in (total, routed["exam_loss"], 100 * routed["density_loss"],
                                  original["exam_loss"])]
        for i, (name, p) in enumerate(model.named_parameters()):
            gt, ge, gd, go = [torch.zeros_like(p) if g[i] is None else g[i] for g in gradients]
            torch.testing.assert_close(gt, ge + gd, rtol=3e-5, atol=3e-6)
            if name.startswith(("negative.", "shift.")):
                assert torch.count_nonzero(ge) == 0
                assert torch.equal(gt, gd)
            if name.startswith("prior."):
                assert torch.equal(ge, go) and torch.count_nonzero(gd) == 0
            if name == "encoder.1.weight":
                assert all(torch.count_nonzero(g) > 0 for g in (ge, gd))
                assert not torch.allclose(ge, go)
        assert all(p.grad is None for p in parameters)
        assert all(torch.equal(v, original_state[k]) for k, v in model.state_dict().items())


@pytest.mark.parametrize("weight", [-1, float("nan"), float("inf")])
def test_inherits_original_invalid_coefficient_rejection(weight):
    with pytest.raises(ValueError):
        loss(generative_two_mask_loss, example(), weight)


def test_routing_and_coefficient_must_be_explicit():
    data = example()
    with pytest.raises(TypeError, match="density_weight"):
        generative_two_mask_loss(*data[:5], descriptor_width=2)
    with pytest.raises(ValueError, match="Two original Evidence"):
        generative_two_mask_loss(data[0][:1], *data[1:5], density_weight=100)
