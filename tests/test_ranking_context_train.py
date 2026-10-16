import copy
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from test_released_context_train import dataset, same
from torch.nn import functional as F

from rsna.competitive.ranking_context_train import Recipe, Trainer, ranking_loss
from rsna.competitive.released_context import anchored_loss
from rsna.competitive.released_context_train import Recipe as OriginalRecipe
from rsna.competitive.released_context_train import Trainer as OriginalTrainer


def loss(z, y, *, mask=None, teacher=None, fraction=0.5):
    return ranking_loss(
        z,
        torch.zeros_like(z) if teacher is None else teacher,
        y,
        torch.ones_like(y, dtype=torch.bool) if mask is None else mask,
        source_weight=0.25,
        rank_fraction=fraction,
    )


def test_binary_two_study_closed_form_teacher_weight_and_gradient_direction():
    z = torch.tensor([[0.8], [0.8], [-0.2], [-0.2]], requires_grad=True)
    y = torch.tensor([[1.0], [1.0], [0.0], [0.0]], requires_grad=True)
    teacher = torch.zeros_like(z, requires_grad=True)
    actual = loss(z, y, teacher=teacher)
    point = F.binary_cross_entropy_with_logits(z, y.detach())
    expected = (
        0.75 * F.binary_cross_entropy_with_logits(z, torch.full_like(z, 0.5))
        + 0.125 * point
        + 0.125 * F.softplus(torch.tensor(-1.0))
    )
    assert torch.allclose(actual, expected)
    actual.backward()
    assert teacher.grad is None and y.grad is None
    # Isolate pair term: raising positive/lowering negative reduces pair risk.
    a = loss(z.detach(), y.detach(), fraction=1) - loss(z.detach(), y.detach(), fraction=0)
    assert torch.isfinite(a)


def test_pairwise_uses_mean_probability_tta_not_mean_logit_or_cross_view_pairs():
    z = torch.tensor([[8.0], [-2.0], [0.0], [0.0]], requires_grad=True)
    y = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    p = z.sigmoid().view(2, 2, 1).mean(1)
    score = torch.logit(p)
    pair = F.softplus(-(score[0] - score[1])).mean()
    point = F.binary_cross_entropy_with_logits(z, y)
    base = anchored_loss(z, torch.zeros_like(z), y, torch.ones_like(y, dtype=torch.bool))
    assert torch.allclose(loss(z, y), base + 0.125 * (pair - point), atol=1e-7)
    assert not torch.allclose(pair, F.softplus(-(z[:2].mean() - z[2:].mean())))


@pytest.mark.parametrize("kind", ["singleton", "all_ties", "all_masked", "no_joint_pair"])
def test_no_source_pairs_exact_old_loss_and_complete_gradient(kind):
    z = torch.tensor([[0.4, -0.2], [0.9, -0.5], [-0.6, 0.7], [-0.8, 1.1]])
    y = torch.tensor([[0.8, 0.3], [0.8, 0.3], [0.8, 0.3], [0.8, 0.3]])
    m = torch.ones_like(y, dtype=torch.bool)
    if kind == "singleton":
        z, y, m = z[:2], y[:2], m[:2]
    elif kind == "all_masked":
        m[:] = False
        y[:] = float("nan")
    elif kind == "no_joint_pair":
        m[:2, 1], m[2:, 0] = False, False
        y[2:] = torch.tensor([0.1, 0.9])
        y[~m] = float("nan")
    a, b = z.clone().requires_grad_(), z.clone().requires_grad_()
    l1, l2 = loss(a, y, mask=m), anchored_loss(b, torch.zeros_like(b), y, m)
    assert torch.equal(l1, l2)
    l1.backward()
    l2.backward()
    assert torch.equal(a.grad, b.grad)


def test_masked_nan_excluded_and_class_specific_replacement():
    z = torch.tensor([[0.5, 2.0], [0.5, 2.0], [-0.1, -2.0], [-0.1, -2.0]])
    y = torch.tensor([[1.0, float("nan")], [1.0, float("nan")], [0.0, 0.8], [0.0, 0.8]])
    m = torch.isfinite(y)
    a = loss(z, y, mask=m)
    expected = (loss(z[:, :1], y[:, :1]) + loss(z[:, 1:], y[:, 1:], mask=m[:, 1:])) / 2
    assert torch.allclose(a, expected)


def test_fractional_gaps_match_independent_directed_pair_loop_and_permutation():
    z = torch.tensor([[0.5], [0.2], [1.0], [-0.1], [-0.8], [-0.4]])
    y = torch.tensor([[0.9], [0.9], [0.6], [0.6], [0.1], [0.1]])
    scores = torch.logit(z.sigmoid().reshape(3, 2).mean(1))
    weight, total = 0.0, torch.tensor(0.0)
    for i in range(3):
        for j in range(3):
            w = max(0.0, float(y[2 * i] - y[2 * j]))
            weight += w
            total += w * F.softplus(-(scores[i] - scores[j]))
    base = anchored_loss(z, torch.zeros_like(z), y, torch.ones_like(y, dtype=torch.bool))
    expected = base + 0.125 * (total / weight - F.binary_cross_entropy_with_logits(z, y))
    assert torch.allclose(loss(z, y), expected, atol=1e-7)
    permutation = torch.tensor([4, 5, 0, 1, 2, 3])
    assert torch.allclose(loss(z[permutation], y[permutation]), expected, atol=1e-7)


def test_extreme_logits_finite_and_teacher_and_labels_detached():
    z = torch.tensor([[1000.0], [-1000.0], [-900.0], [-800.0]], requires_grad=True)
    y = torch.tensor([[0.0], [0.0], [1.0], [1.0]], requires_grad=True)
    teacher = torch.zeros_like(z, requires_grad=True)
    x = loss(z, y, teacher=teacher)
    x.backward()
    assert torch.isfinite(x) and torch.isfinite(z.grad).all()
    assert y.grad is None and teacher.grad is None


@pytest.mark.parametrize("bad", [-0.1, 1.1, True, float("nan")])
def test_reject_bad_fraction_and_misaligned_views(bad):
    z = torch.zeros(4, 2)
    with pytest.raises(ValueError):
        loss(z, z, fraction=bad)
    y = z.clone()
    y[1, 0] = 1
    with pytest.raises(ValueError, match="Both views"):
        loss(z, y)


def recipe(objective):
    return Recipe(objective=objective, epochs=2, batch_size=2, embedding_dim=8, targets=3, width=8)


@pytest.mark.parametrize("objective", ["uniform", "source_rank"])
@pytest.mark.parametrize("stop", [1, 3])
def test_complete_recovery_numerics_and_uneven_final_batch(tmp_path, objective, stop):
    d = dataset(tmp_path / "data", archive=True)
    for i in range(len(d)):
        d.labels[i] = torch.tensor([i / 5, 1 - i / 5, (i % 2) * 1.0])
    r = recipe(objective)
    full = Trainer(d, r)
    full.run(epoch_directory=tmp_path / "epochs")
    state, prediction = copy.deepcopy(full.state()), full.predictions()
    stopped = Trainer(d, r)
    stopped.run(stop_after=stop)
    stopped.save(tmp_path / "stop.pt")
    resumed = Trainer(d, r)
    resumed.load(tmp_path / "stop.pt")
    resumed.run()
    same(state, resumed.state())
    assert np.array_equal(prediction, resumed.predictions())
    assert [x["studies"] for x in resumed.history] == [2, 2, 1, 2, 2, 1]
    assert state["model"]["output"].count_nonzero()
    with pytest.raises(ValueError, match="objective"):
        Trainer(d, replace(r, rank_fraction=0.25)).load(tmp_path / "stop.pt")


def test_uniform_complete_numerics_and_prediction_bytes_equal_frozen_engine(tmp_path):
    d = dataset(tmp_path / "data")
    r = recipe("uniform")
    old_recipe = asdict(r)
    del old_recipe["objective"]
    del old_recipe["rank_fraction"]
    old = OriginalTrainer(d, OriginalRecipe(**old_recipe))
    old.run()
    state, prediction = copy.deepcopy(old.state()), old.predictions()
    new = Trainer(d, r)
    new.run()
    value = copy.deepcopy(new.state())
    value["version"] = state["version"]
    del value["original_trainer_sha256"]
    del value["recipe"]["objective"]
    del value["recipe"]["rank_fraction"]
    same(state, value)
    assert np.array_equal(prediction, new.predictions())
