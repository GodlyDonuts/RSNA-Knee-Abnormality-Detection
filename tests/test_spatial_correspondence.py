"""The new relation must distinguish what marginal summaries cannot retain."""

import pytest
import torch

from rsna.competitive.regional_features import regional_summary
from rsna.competitive.spatial_correspondence import (
    centered_cosupport,
    marginal_preserving_null,
)


def test_actual_regional_operator_has_location_collision():
    maps = torch.zeros(1, 1024, 10, 10)
    vector = torch.arange(1024) / 1024
    maps[0, :, 0, 0], maps[0, :, 0, 1] = vector, -vector
    changed = maps.clone()
    changed[0, :, 0, 0], changed[0, :, 0, 1] = -vector, vector
    assert not torch.equal(maps, changed)
    assert torch.equal(maps.mean((2, 3)), changed.mean((2, 3)))
    assert torch.equal(regional_summary(maps), regional_summary(changed))


def test_joint_correspondence_changes_while_marginals_do_not():
    evidence = torch.tensor([[1.0], [0.0]])
    aligned = torch.eye(2) / 2
    broken = aligned.flip(1)
    assert torch.equal(aligned.sum(0), broken.sum(0))
    assert torch.equal(aligned.sum(1), broken.sum(1))
    assert centered_cosupport(evidence, evidence, aligned).item() == 0.25
    assert centered_cosupport(evidence, evidence, broken).item() == -0.25


def test_independent_coupling_and_constant_evidence_have_no_joint_signal():
    a, b = torch.tensor([0.25, 0.75]), torch.tensor([0.5, 0.25, 0.25])
    first, second = torch.randn(2, 12), torch.randn(3, 12)
    assert torch.allclose(
        centered_cosupport(first, second, a[:, None] * b), torch.zeros(12), atol=1e-6
    )
    pi = torch.tensor([[0.125, 0.125, 0.0], [0.375, 0.125, 0.25]])
    assert torch.allclose(
        centered_cosupport(torch.ones(2, 12), second, pi), torch.zeros(12), atol=1e-6
    )


def test_null_preserves_nonuniform_marginals_and_exact_missingness():
    pi = torch.tensor([[0.25, 0.10, 0.0], [0.05, 0.20, 0.0], [0.05, 0.10, 0.25]])
    control, receipt = marginal_preserving_null(pi, seed=711)
    assert receipt["testable"]
    assert torch.equal(control > 0, pi > 0)
    torch.testing.assert_close(control.sum(0), pi.sum(0))
    torch.testing.assert_close(control.sum(1), pi.sum(1))
    repeated, _ = marginal_preserving_null(pi, seed=711)
    assert torch.equal(control, repeated)
    assert not torch.equal(control, pi)


def test_sparse_null_abstains_instead_of_inventing_support():
    pi = torch.eye(3) / 3
    control, receipt = marginal_preserving_null(pi, seed=2)
    assert not receipt["testable"]
    assert torch.equal(control, pi)


def test_centered_evidence_has_finite_learnable_gradient():
    first = torch.randn(3, 12, requires_grad=True)
    second = torch.randn(4, 12, requires_grad=True)
    pi = torch.rand(3, 4)
    pi /= pi.sum()
    centered_cosupport(first, second, pi).square().sum().backward()
    assert torch.isfinite(first.grad).all() and first.grad.abs().sum() > 0
    assert torch.isfinite(second.grad).all() and second.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "pi", [torch.ones(2, 2), torch.tensor([[1.1, -0.1]]), torch.tensor([[float("nan")]])]
)
def test_reject_invalid_geometry_coupling(pi):
    with pytest.raises(ValueError, match="coupling"):
        centered_cosupport(torch.zeros(pi.shape[0], 12), torch.zeros(pi.shape[1], 12), pi)
