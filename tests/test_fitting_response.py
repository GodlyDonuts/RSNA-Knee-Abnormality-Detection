import numpy as np
import pytest

from rsna.competitive.fitting_response import pair_response, response_summary


def test_actual_correction_and_new_damage_are_separate():
    labels = np.array([1.0, 1.0, 0.0, 0.0])
    teacher = np.array([0.2, 0.8, 0.4, 0.6])
    learned = np.array([0.9, 0.3, 0.4, 0.6])
    result = pair_response(labels, np.ones(4, bool), teacher, learned, labels)
    assert result["dependent_source_pairs"] == 4
    assert result["teacher_opposed"] == 2
    assert result["opposed_corrected"] == 2
    assert result["ordered_newly_inverted"] == 2
    assert result["strong_source_pair_agreement_before"] == 0.5
    assert result["strong_source_pair_agreement_after"] == 0.5


def test_ties_are_not_strict_corrections():
    result = pair_response(
        np.array([1.0, 0.0]),
        np.ones(2, bool),
        np.array([0.1, 0.9]),
        np.full(2, 0.5),
        np.array([1, 0]),
    )
    assert result["opposed_corrected"] == 0
    assert result["opposed_candidate_tied"] == 1
    assert result["strong_source_pair_agreement_after"] == 0.5


def test_mask_and_intermediate_labels_excluded_from_pair_evidence():
    labels = np.array([1.0, 0.0, 1.0, 0.5])
    mask = np.array([True, True, False, True])
    result = pair_response(labels, mask, labels, labels, labels)
    assert result["dependent_source_pairs"] == 1
    assert result["strong_source_pair_agreement_after"] == 1


def test_correctable_denominator_excludes_unattainable_pointwise_orderings():
    labels = np.array([1.0, 0.0, 0.0])
    result = pair_response(
        labels,
        np.ones(3, bool),
        np.array([0.1, 0.6, 0.9]),
        labels,
        np.array([0.5, 0.4, 0.8]),
    )
    assert result["teacher_opposed"] == 2
    assert result["pointwise_correctable_opposed"] == 1
    assert result["correctable_opposed_actually_corrected"] == 1
    assert result["correctable_opposed_achieved_fraction"] == 1


def test_missing_class_is_undefined_not_perfect():
    p = np.array([0.2, 0.8])
    result = pair_response(np.ones(2), np.ones(2, bool), p, p, p)
    assert result["dependent_source_pairs"] == 0
    assert result["strong_source_pair_agreement_after"] is None
    assert result["opposed_correction_fraction"] is None


def test_unconstrained_optimum_reduces_target_error_and_cross_entropy():
    labels = np.array([1.0, 0.0])
    teacher = np.array([0.4, 0.6])
    optimum = 0.25 * labels + 0.75 * teacher
    result = response_summary(labels, np.ones(2, bool), teacher, teacher, optimum)
    assert result["candidate"]["pointwise_target_RMSE"] == 0
    assert (
        result["candidate"]["full_bag_pseudo_target_cross_entropy"]
        < result["initial"]["full_bag_pseudo_target_cross_entropy"]
    )
    assert result["teacher_pairs"]["opposed_corrected"] == 1
    assert result["changes_in_pointwise_desired_direction_fraction"] == 1
    assert result["clinical_or_held_out_accuracy_claim"] is False


def test_unchanged_model_with_no_desired_change_is_explicit():
    p = np.array([0.0, 1.0])
    result = response_summary(p, np.ones(2, bool), p, p, p)
    assert result["nonzero_desired_changes"] == 0
    assert result["changes_in_pointwise_desired_direction_fraction"] is None
    assert result["absolute_probability_change_quantiles"]["maximum"] == 0
    assert np.isfinite(result["candidate"]["full_bag_pseudo_target_cross_entropy"])


@pytest.mark.parametrize("bad", [np.nan, np.inf, -0.1, 1.1])
def test_invalid_probabilities_rejected(bad):
    with pytest.raises(ValueError):
        response_summary(np.array([bad]), np.ones(1, bool), np.ones(1), np.ones(1), np.ones(1))


@pytest.mark.parametrize("mask", [np.ones(2), np.ones(3, bool), np.zeros(2, bool)])
def test_misaligned_nonboolean_or_empty_observation_mask_rejected(mask):
    p = np.array([0.3, 0.7])
    with pytest.raises(ValueError):
        response_summary(p, mask, p, p, p)


def test_large_pair_count_is_block_exact():
    rng = np.random.default_rng(75)
    labels = np.r_[np.ones(150), np.zeros(170)]
    before, after, q = rng.random((3, 320))
    result = pair_response(labels, np.ones(320, bool), before, after, q)
    inverted = before[:150, None] < before[150:][None, :]
    corrected = after[:150, None] > after[150:][None, :]
    assert result["teacher_opposed"] == int(inverted.sum())
    assert result["opposed_corrected"] == int((inverted & corrected).sum())
