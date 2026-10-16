import copy
from dataclasses import asdict, replace

import numpy as np
import pytest
import torch
from test_released_context_train import dataset, same
from torch.nn import functional as F

from rsna.competitive.confidence_context_train import Recipe, Trainer, confidence_loss
from rsna.competitive.released_context import anchored_loss
from rsna.competitive.released_context_train import Recipe as OriginalRecipe
from rsna.competitive.released_context_train import Trainer as OriginalTrainer


def small_recipe(objective):
    return Recipe(objective=objective, epochs=2, batch_size=2, embedding_dim=8, targets=3, width=8)


def test_uncertain_or_missing_targets_return_to_teacher_without_nan_or_label_gradients():
    parent = torch.tensor([[2.0, -1.0, 0.3]], requires_grad=True)
    labels = torch.tensor([[0.5, float("nan"), 1.0]], requires_grad=True)
    mask = torch.tensor([[True, False, True]])
    logits = torch.tensor([[1.7, -0.8, 0.4]], requires_grad=True)
    loss = confidence_loss(logits, parent, labels, mask, source_weight=0.25)
    target = parent.detach().sigmoid()
    target[0, 2] = 0.25 + 0.75 * target[0, 2]
    expected = F.binary_cross_entropy_with_logits(logits, target)
    assert torch.equal(loss, expected)
    loss.backward()
    assert parent.grad is None and labels.grad is None
    assert torch.allclose(logits.grad, (logits.detach().sigmoid() - target) / 3)


def test_confident_binary_source_equals_old_objective_and_fractional_scales_explicitly():
    logits = torch.tensor([[-0.7, 0.2, 1.4]])
    parent = torch.tensor([[0.4, -0.9, 1.2]])
    y = torch.tensor([[0.0, 1.0, 0.0]])
    mask = torch.ones_like(y, dtype=torch.bool)
    assert torch.equal(
        confidence_loss(logits, parent, y, mask, source_weight=0.25),
        anchored_loss(logits, parent, y, mask, source_weight=0.25),
    )
    y = torch.tensor([[0.5, 0.6, 0.9]])
    alpha = 0.25 * torch.tensor([[0.0, 0.2, 0.8]])
    target = alpha * y + (1 - alpha) * parent.sigmoid()
    assert torch.allclose(
        confidence_loss(logits, parent, y, mask, source_weight=0.25),
        F.binary_cross_entropy_with_logits(logits, target),
    )


@pytest.mark.parametrize("objective", ["uniform", "source_confidence"])
@pytest.mark.parametrize("stop", [1, 3])
def test_complete_recovery_objective_bound_with_uneven_batch(tmp_path, objective, stop):
    d = dataset(tmp_path / "data", archive=True)
    d.labels[1] = torch.tensor([0.5, 0.6, 0.9])
    r = small_recipe(objective)
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
    assert state["model"]["output"].abs().sum() > 0
    other = "uniform" if objective == "source_confidence" else "source_confidence"
    with pytest.raises(ValueError, match="objective"):
        Trainer(d, replace(r, objective=other)).load(tmp_path / "stop.pt")


def test_uniform_full_numerics_equal_original_engine(tmp_path):
    d = dataset(tmp_path / "data")
    recipe = small_recipe("uniform")
    old_recipe = asdict(recipe)
    del old_recipe["objective"]
    old = OriginalTrainer(d, OriginalRecipe(**old_recipe))
    old.run()
    original, predictions = copy.deepcopy(old.state()), old.predictions()
    new = Trainer(d, recipe)
    new.run()
    current = copy.deepcopy(new.state())
    del current["original_trainer_sha256"]
    del current["recipe"]["objective"]
    current["version"] = original["version"]
    same(original, current)
    assert np.array_equal(predictions, new.predictions())
    new.save(tmp_path / "new.pt")
    with pytest.raises(ValueError):
        OriginalTrainer(d, OriginalRecipe(**old_recipe)).load(tmp_path / "new.pt")


@pytest.mark.parametrize("bad", ["", "class_weight", "confidence_normalized"])
def test_reject_unknown_objective(bad):
    with pytest.raises(ValueError):
        Recipe(objective=bad)


@pytest.mark.parametrize("weight", [0, -1, 1.1, float("nan"), True])
def test_reject_invalid_source_weight(weight):
    z = torch.zeros(2, 3)
    with pytest.raises(ValueError):
        confidence_loss(z, z, z, torch.ones_like(z, dtype=torch.bool), source_weight=weight)


def test_observed_invalid_label_rejected_missing_all_teacher_only():
    z = torch.zeros(2, 3)
    mask = torch.ones_like(z, dtype=torch.bool)
    for bad in [float("nan"), -0.1, 1.1]:
        with pytest.raises(ValueError):
            confidence_loss(z, z, torch.full_like(z, bad), mask, source_weight=0.25)
    loss = confidence_loss(z, z, torch.full_like(z, float("nan")), ~mask, source_weight=0.25)
    assert torch.isfinite(loss)
