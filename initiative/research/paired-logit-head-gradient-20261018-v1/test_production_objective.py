"""New actual-loss variant, not a rerun of the earlier surrogate screen."""

import importlib.util
from pathlib import Path

import pytest
import torch

from rsna.competitive.released_context import anchored_loss

spec = importlib.util.spec_from_file_location(
    "new_production_gradient", Path(__file__).with_name("production_objective.py")
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_explicit_target_loss_and_gradient_equal_actual_masked_production_loss():
    teacher = torch.arange(24, dtype=torch.float32).reshape(2, 12) / 7 - 1
    labels = torch.ones(2, 12)
    mask = torch.ones(2, 12, dtype=torch.bool)
    mask[:, 3] = False
    labels[:, 3] = float("nan")
    logits = torch.zeros(2, 12, requires_grad=True)
    target = m.production_target(teacher, labels, mask, 0.25)
    actual = anchored_loss(logits, teacher, labels, mask, source_weight=0.25)
    candidate = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
    assert torch.equal(actual, candidate)
    a = torch.autograd.grad(actual, logits, retain_graph=True)[0]
    b = torch.autograd.grad(candidate, logits)[0]
    assert torch.equal(a, b) and torch.isfinite(b).all()
    assert not torch.equal(target[0], target[1])


def test_unobserved_targets_use_corresponding_view_teacher_without_mean_TTA():
    teacher = torch.stack([torch.full((12,), -3.0), torch.full((12,), 3.0)])
    target = m.production_target(
        teacher, torch.full((2, 12), float("nan")), torch.zeros(2, 12, dtype=torch.bool), 0.25
    )
    assert torch.equal(target, teacher.sigmoid())


def test_new_benchmark_requires_actual_objective_inputs():
    with pytest.raises(TypeError):
        m.benchmark({}, {}, torch.zeros(2, 94, 1024), torch.zeros(94))
