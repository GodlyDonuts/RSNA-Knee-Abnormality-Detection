"""Masked-cell entropy objective (ESA arm) tests."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from rsna.competitive.loss import (
    masked_cell_entropy,
    observed_counts,
    validate_masked_cell,
)
from rsna.competitive.train import TrainConfig


def _batch(studies: int = 4, targets: int = 12, seed: int = 0):
    torch.manual_seed(seed)
    logits = torch.randn(studies, targets)
    mask = (torch.rand(studies, targets) > 0.5).float()
    return logits, mask


def test_masked_entropy_equals_bce_against_point_five_on_masked_cells():
    logits, mask = _batch()
    counts = observed_counts([(mask == 0).float()])
    term = masked_cell_entropy(logits, mask, counts)
    unobserved = (mask == 0).float()
    sums = (
        F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, 0.5), reduction="none")
        * unobserved
    ).sum(dim=0)
    included = counts > 0
    expected = (sums[included] / counts[included].to(sums.dtype)).mean()
    torch.testing.assert_close(term, expected)


def test_masked_entropy_is_zero_when_everything_is_supervised():
    logits, mask = _batch()
    mask = torch.ones_like(mask)
    counts = observed_counts([(mask == 0).float()])
    term = masked_cell_entropy(logits, mask, counts)
    assert float(term) == 0.0


def test_masked_entropy_global_normalization_across_accumulation():
    logits, mask = _batch(studies=6)
    pieces = [(logits[:2], mask[:2]), (logits[2:4], mask[2:4]), (logits[4:], mask[4:])]
    counts = observed_counts([(m == 0).float() for _, m in pieces])
    whole = masked_cell_entropy(logits, mask, counts)
    summed = sum(masked_cell_entropy(p_l, p_m, counts) for p_l, p_m in pieces)
    torch.testing.assert_close(whole, summed)


def test_masked_entropy_gradient_reaches_only_masked_logits():
    logits, mask = _batch()
    logits.requires_grad_(True)
    counts = observed_counts([(mask == 0).float()])
    masked_cell_entropy(logits, mask, counts).backward()
    assert logits.grad is not None
    observed_grad = logits.grad[mask == 1]
    masked_grad = logits.grad[mask == 0]
    assert observed_grad.abs().max() == 0
    assert (masked_grad.abs() > 0).any()


def test_masked_entropy_rejects_invalid_shapes():
    logits, mask = _batch()
    counts = observed_counts([(mask == 0).float()])
    with pytest.raises(ValueError, match="aligned"):
        masked_cell_entropy(logits, mask[:, :4], counts)
    with pytest.raises(ValueError, match="masked counts"):
        masked_cell_entropy(logits, mask, torch.zeros(3))
    with pytest.raises(ValueError, match="Non-finite"):
        bad = logits.clone()
        bad[0, 0] = float("nan")
        masked_cell_entropy(bad, mask, counts)


def test_validate_masked_cell_accepts_valid_policy():
    validate_masked_cell({"enabled": True, "kind": "entropy_max", "weight": 0.1})
    validate_masked_cell({"enabled": False, "kind": "entropy_max", "weight": 1})


@pytest.mark.parametrize(
    "policy",
    [
        "entropy",
        {"enabled": True, "kind": "entropy_max"},
        {"enabled": True, "kind": "entropy_max", "weight": 0.5, "extra": 1},
        {"enabled": "yes", "kind": "entropy_max", "weight": 0.5},
        {"enabled": True, "kind": "entropy_min", "weight": 0.5},
        {"enabled": True, "kind": "entropy_max", "weight": 0},
        {"enabled": True, "kind": "entropy_max", "weight": 1.5},
        {"enabled": True, "kind": "entropy_max", "weight": "0.5"},
    ],
)
def test_validate_masked_cell_rejects_invalid_policies(policy):
    with pytest.raises(ValueError, match="[Mm]asked-cell"):
        validate_masked_cell(policy)


def _config(**changes):
    return TrainConfig(
        diagnostic_study_limit=4,
        slices=2,
        side=32,
        max_series=3,
        **changes,
    )


def test_masked_cell_config_must_be_an_isolated_arm():
    arm = {"enabled": True, "kind": "entropy_max", "weight": 0.05}
    _config(masked_cell=arm).validate()
    _config(masked_cell={**arm, "enabled": False}).validate()
    with pytest.raises(ValueError, match="isolated outer-fold"):
        _config(
            masked_cell=arm,
            fitting_supervision={
                "kind": "label_source_v1",
                "path": "overlay.json",
                "sha256": "a" * 64,
            },
        ).validate()
    with pytest.raises(ValueError, match="isolated outer-fold"):
        _config(masked_cell=arm, fit_scope="all_development").validate()
