"""Anonymous synthetic source-state and derivative tests, not clinical evidence."""

import numpy as np
import pytest
import torch
from attribution import assess


def fixture():
    g = np.random.default_rng(4)
    y = g.random((5, 12)).astype(np.float32)
    m = np.ones_like(y, dtype=bool)
    t = g.normal(size=(5, 2, 12)).astype(np.float32)
    states = np.zeros((5, 12, 2), dtype=np.uint8)
    states[1, :, 0] = 1
    states[2, :, 1] = 1
    states[3] = 1
    states[4] = 2
    return y, m, t, states, np.zeros_like(m)


def test_categories_partition_all_cells_and_gradient_energy():
    result = assess(*fixture())
    groups = result["groups"]
    assert sum(sum(v["cells_all12"]) for v in groups.values()) == 60
    np.testing.assert_allclose(
        np.sum([v["source_gradient_energy_fraction_all12"] for v in groups.values()], 0),
        1,
        atol=1e-14,
    )
    assert result["not_clinical_error_or_accuracy"]


def test_matches_actual_BCE_gradient_at_dense_teacher_not_sampled_student():
    y, m, t, states, preserve = fixture()
    x = torch.tensor(t.astype(np.float64), requires_grad=True)
    p = x.detach().sigmoid()
    target = 0.25 * torch.tensor(y)[:, None, :] + 0.75 * p
    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        x, target.expand_as(x), reduction="sum"
    )
    loss.backward()
    actual = np.sqrt(x.grad.numpy().__pow__(2).mean((0, 1)))
    np.testing.assert_allclose(
        assess(y, m, t, states, preserve)["total_unnormalized_source_logit_gradient_RMS_all12"],
        actual,
        atol=1e-14,
    )


def test_zero_weighted_source_gradient_is_explicit_not_nan():
    y, m, t, s, p = fixture()
    m.fill(False)
    result = assess(y, m, t, s, p)
    assert result["zero_energy_targets_all12"] == [True] * 12
    assert all(g["macro_cell_energy_fraction"] is None for g in result["groups"].values())


@pytest.mark.parametrize("damage", ["shape", "state", "nan", "mask", "weight"])
def test_invalid_or_unidentified_states_refused(damage):
    y, m, t, s, p = fixture()
    kwargs = {}
    if damage == "shape":
        s = s[:1]
    elif damage == "state":
        s[0, 0, 0] = 3
    elif damage == "nan":
        t[0, 0, 0] = np.nan
    elif damage == "mask":
        m = m.astype(np.uint8)
    else:
        kwargs["source_weight"] = 0.5
    with pytest.raises(ValueError):
        assess(y, m, t, s, p, **kwargs)
