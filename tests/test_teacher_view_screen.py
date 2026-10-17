import numpy as np
import pytest

from scripts.newton.screen_teacher_views import summarize


def test_identical_views_have_no_disagreement_and_report_undefined_strict_pairs():
    out = summarize(np.zeros((4, 2, 12)))
    assert all(r["mean_absolute_view_probability_gap"] == 0 for r in out["targets"])
    assert all(
        r["discordance_fraction_conditional_on_strict_pairs"] is None for r in out["targets"]
    )
    assert out["clinical_or_competition_accuracy_claim"] is False


def test_shifted_logits_change_probabilities_without_reversing_order():
    x = np.broadcast_to(np.arange(4)[:, None, None], (4, 2, 12)).copy().astype(float)
    x[:, 1] += 1
    rows = summarize(x)["targets"]
    assert all(r["mean_absolute_view_probability_gap"] > 0 for r in rows)
    assert all(
        r["strict_pairs_in_both_views"] == 6 and r["discordant_view_pairs"] == 0 for r in rows
    )


def test_reversed_views_reverse_all_unique_pairs_without_counting_diagonal_or_twice():
    x = np.broadcast_to(np.arange(4)[:, None, None], (4, 2, 12)).copy().astype(float)
    x[:, 1] *= -1
    assert all(
        r["discordant_view_pairs"] == r["all_study_pairs"] == 6 for r in summarize(x)["targets"]
    )
    assert all(
        r["discordance_fraction_conditional_on_strict_pairs"] == 1 for r in summarize(x)["targets"]
    )


def test_nonfinite_teacher_values_are_refused():
    x = np.zeros((4, 2, 12))
    x[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        summarize(x)
