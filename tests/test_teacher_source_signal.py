"""Exact mixture derivative and conditional report/teacher disagreement counts."""

import numpy as np
import pytest
import torch

from rsna.competitive.released_context import anchored_loss
from rsna.competitive.teacher_source_signal import audit, mixture_arrays


def inputs():
    logits = np.zeros((2, 2, 12), np.float32)
    labels = np.zeros((2, 12), np.float32)
    mask = np.zeros((2, 12), bool)
    logits[0, :, 0] = np.log(9)
    labels[0, 0] = 0
    logits[1, :, 0] = -np.log(9)
    labels[1, 0] = 1
    mask[:, 0] = True
    labels[~mask] = np.nan
    return logits, labels, mask


@pytest.mark.parametrize("weight", [0.25, 0.5, 0.75, 1.0])
def test_derivative_equals_actual_original_loss_with_masked_nans(weight):
    logits, labels, mask = inputs()
    _, target, g = mixture_arrays(logits, labels, mask, weight)
    z = torch.tensor(logits.reshape(-1, 12), dtype=torch.float64, requires_grad=True)
    teacher = z.detach().clone()
    y = torch.tensor(np.repeat(labels[:, None, :], 2, axis=1).reshape(-1, 12), dtype=torch.float64)
    m = torch.tensor(np.repeat(mask[:, None, :], 2, axis=1).reshape(-1, 12))
    loss = anchored_loss(z, teacher, y, m, source_weight=weight)
    observed = torch.autograd.grad(loss, z)[0].numpy().reshape(logits.shape)
    np.testing.assert_allclose(g, observed, rtol=1e-12, atol=1e-17)
    assert np.isfinite(target).all()


def test_two_views_are_not_counted_as_independent_studies():
    result = audit(*inputs())
    first = result["per_target"][0]
    assert first["observed_studies"] == 2 and first["binary_observed_view_cells"] == 4
    assert first["teacher_weak_label_discordant_view_cells"] == 4
    assert first["studies_teacher_discordant_in_both_views"] == 2
    assert first["quarter_mixture_target_still_teacher_side_view_cells"] == 4
    assert (
        result["predeclared_source_weight_sensitivity_not_a_recipe_selection"][1][
            "discordant_view_targets_still_teacher_side"
        ]
        == 0
    )
    assert result["masked_targets_equal_teacher_exactly"] is True
    assert result["per_target"][1]["fraction_discordant_view_targets_still_teacher_side"] is None


def test_soft_labels_are_not_forced_into_binary_disagreement():
    z, y, m = inputs()
    y[:, 0] = 0.4
    result = audit(z, y, m)
    assert result["per_target"][0]["observed_soft_label_studies"] == 2
    assert result["teacher_weak_label_discordant_view_cells"] == 0


@pytest.mark.parametrize("weight", [0, True, float("nan"), 1.1])
def test_invalid_objective_weight_refused(weight):
    with pytest.raises(ValueError):
        mixture_arrays(*inputs(), weight)


def test_observed_nan_and_wrong_mask_dtype_refused():
    z, y, m = inputs()
    y[0, 0] = np.nan
    with pytest.raises(ValueError):
        mixture_arrays(z, y, m, 0.25)
    z, y, m = inputs()
    with pytest.raises(ValueError):
        mixture_arrays(z, y, m.astype(np.int8), 0.25)
