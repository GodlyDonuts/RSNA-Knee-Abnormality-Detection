import numpy as np
import pytest

from rsna.competitive.supervision_gradient import audit


def test_identical_teacher_has_no_teacher_update_and_masked_source_is_zero():
    p = np.zeros((3, 2, 12))
    labels = np.ones((3, 12))
    mask = np.zeros((3, 12), dtype=bool)
    z = audit(p, p, labels, mask)
    assert z["aggregate"]["source_L2"] == z["aggregate"]["teacher_L2"] == 0
    assert z["aggregate"]["source_teacher_cosine"] is None
    mask[:] = True
    z = audit(p, p, labels, mask)
    assert z["aggregate"]["source_L2"] > 0 and z["aggregate"]["teacher_L2"] == 0


def test_known_opposing_soft_source_and_new_teacher_derivatives():
    p = np.zeros((2, 2, 12))
    q = np.full_like(p, 2.0)
    y = np.full((2, 12), 0.2)
    mask = np.ones((2, 12), dtype=bool)
    result = audit(p, q, y, mask)
    assert result["aggregate"]["source_teacher_cosine"] == pytest.approx(-1)
    assert result["aggregate"]["opposing_nonzero_derivative_cells"] == 48
    assert result["soft_observed_cells"] == 24
    assert result["source_labels_masks_or_targets_changed"] is False
    assert np.array_equal(y, np.full_like(y, 0.2))
