"""Decision-audit AUC weights agree with an independent sklearn reference."""

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from scripts.rank_learning import auc_plan, component_weights, interval, weighted_auc


@pytest.mark.parametrize("seed", [17, 20261001])
def test_ties_and_component_multiplicity_match_sklearn(seed):
    rng = np.random.default_rng(seed)
    truth = rng.integers(0, 2, size=100)
    scores = rng.integers(0, 5, size=100) / 4
    plan = auc_plan(truth, scores)
    groups = np.repeat(np.arange(50), 2)
    for _ in range(20):
        weights = component_weights(groups, rng)
        assert weighted_auc(plan, weights) == pytest.approx(
            roc_auc_score(truth, scores, sample_weight=weights), abs=1e-12
        )
        np.testing.assert_array_equal(weights[::2], weights[1::2])
        assert weights.sum() == 100


def test_component_draws_preserve_unequal_group_sizes():
    groups = ["a", "a", "a", "b", "c", "c"]
    rng = np.random.default_rng(11)
    varied = False
    for _ in range(20):
        weights = component_weights(groups, rng)
        assert weights[0] == weights[1] == weights[2]
        assert weights[4] == weights[5]
        assert weights[0] + weights[3] + weights[4] == 3
        varied |= weights.sum() != len(groups)
    assert varied


def test_undefined_class_is_recorded_without_reweighting_macro():
    plan = auc_plan([0, 1], [0.2, 0.9])
    assert weighted_auc(plan, [1, 0]) is None
    assert weighted_auc(plan, [0, 0]) is None
    assert interval([0.1, None, -0.2], 3)["undefined_draws"] == 1
    assert interval([None, None], 2)["percentile_95"] is None


@pytest.mark.parametrize(
    "truth,scores", [([0, 2], [0.1, 0.2]), ([0, 1], [0.1, np.nan]), ([0], [0.1, 0.2])]
)
def test_invalid_auc_observations_rejected(truth, scores):
    with pytest.raises(ValueError):
        auc_plan(truth, scores)


@pytest.mark.parametrize("weights", [[1, -1], [1, np.inf], [1]])
def test_invalid_auc_weights_rejected(weights):
    with pytest.raises(ValueError):
        weighted_auc(auc_plan([0, 1], [0.1, 0.8]), weights)
