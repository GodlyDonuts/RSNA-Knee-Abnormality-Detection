from dataclasses import asdict

import pytest
import torch
from torch.nn import functional as F

from rsna.competitive import teacher_only as candidate
from rsna.competitive import teacher_pair


def test_recipe_changes_only_source_weight_and_preserves_reference_rejection():
    original = teacher_pair.Recipe(epochs=3, objective="pointwise")
    ablation = candidate.Recipe(epochs=3)
    assert asdict(ablation) == {**asdict(original), "source_weight": 0.0}
    with pytest.raises(ValueError):
        teacher_pair.Recipe(source_weight=0.0)
    with pytest.raises(ValueError):
        candidate.Recipe(objective="teacher_rank")
    with pytest.raises(ValueError):
        candidate.Recipe(source_weight=False)


def test_teacher_only_loss_has_exact_detached_teacher_gradient_and_ignores_source_values():
    z = torch.tensor([[0.2, -0.8], [1.5, -2.0]], requires_grad=True)
    q = torch.tensor([[-1.0, 0.5], [0.9, -0.2]], requires_grad=True)
    labels = torch.tensor([[0.0, 0.5], [1.0, float("nan")]])
    masks = torch.tensor([[True, True], [True, False]])
    old_labels, old_masks = labels.clone(), masks.clone()
    loss = candidate.teacher_loss(z, q, labels, masks, source_weight=0.0)
    expected = F.binary_cross_entropy_with_logits(z, q.detach().sigmoid())
    assert torch.equal(loss, expected)
    gradient, teacher_gradient = torch.autograd.grad(loss, (z, q), allow_unused=True)
    assert torch.allclose(
        gradient, (z.detach().sigmoid() - q.detach().sigmoid()) / z.numel(), atol=1e-8, rtol=0
    )
    assert teacher_gradient is None
    assert torch.equal(masks, old_masks) and torch.allclose(labels, old_labels, equal_nan=True)
    changed = torch.where(masks, 1 - labels, labels)
    assert torch.equal(candidate.teacher_loss(z, q, changed, masks, source_weight=0.0), loss)


def test_objective_is_restored_after_failure_and_no_nested_substitution():
    original = teacher_pair.anchored_loss
    with pytest.raises(RuntimeError):
        with candidate._isolated_objective():
            assert teacher_pair.anchored_loss is candidate.teacher_loss
            with pytest.raises(ValueError):
                with candidate._isolated_objective():
                    pass
            raise RuntimeError("deliberate stopped worker")
    assert teacher_pair.anchored_loss is original


@pytest.mark.parametrize("bad", [-0.1, 0.25, False, float("nan")])
def test_nonzero_or_nonfinite_source_setting_is_rejected(bad):
    z = torch.zeros(1, 12)
    y = z.clone()
    mask = torch.ones_like(z, dtype=torch.bool)
    with pytest.raises(ValueError):
        candidate.teacher_loss(z, z, y, mask, source_weight=bad)


def test_unbound_engine_cannot_start_any_training():
    called = []

    def fake(*args, **kwargs):
        called.append(1)

    with pytest.raises(ValueError):
        candidate.fit(None, None, None, {}, candidate.Recipe(), engine=fake)
    assert not called
