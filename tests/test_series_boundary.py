import numpy as np
import pytest

from rsna.competitive.series_boundary import same_series_windows, series_donor_indices

SLOTS = [26, 22, 18, 12, 18]


def test_full_reference_has_eight_mixed_windows_and_both_policies_preserve_other_values():
    centers = np.arange(1, 95)
    old = series_donor_indices(centers, SLOTS)
    repaired = series_donor_indices(centers, SLOTS, mode="edge-repeat")
    keep = same_series_windows(centers, SLOTS)
    np.testing.assert_array_equal(old, np.stack([centers - 1, centers, centers + 1], axis=1))
    assert keep.sum() == 86 and (old != repaired).sum() == 8
    np.testing.assert_array_equal(repaired[:, 1], centers)
    np.testing.assert_array_equal(repaired[keep], old[keep])
    bounds = np.cumsum([0, *SLOTS])
    slots = np.searchsorted(bounds[1:], centers, side="right")
    for row, slot in zip(repaired, slots, strict=True):
        assert ((bounds[slot] <= row) & (row < bounds[slot + 1])).all()


def test_series_edges_repeat_center_without_borrowing_the_other_plane():
    old = series_donor_indices([25, 26], SLOTS)
    new = series_donor_indices([25, 26], SLOTS, mode="edge-repeat")
    np.testing.assert_array_equal(old, [[24, 25, 26], [25, 26, 27]])
    np.testing.assert_array_equal(new, [[24, 25, 25], [26, 26, 27]])


def test_repeated_centers_and_short_series_remain_defined_and_do_not_mutate_inputs():
    centers = np.array([1, 1, 2, 3])
    before = centers.copy()
    repaired = series_donor_indices(centers, [1, 1, 3], mode="edge-repeat")
    np.testing.assert_array_equal(centers, before)
    np.testing.assert_array_equal(repaired, [[1, 1, 1], [1, 1, 1], [2, 2, 3], [2, 3, 4]])
    np.testing.assert_array_equal(
        same_series_windows(centers, [1, 1, 3]), [False, False, False, True]
    )


@pytest.mark.parametrize(
    "centers,sizes",
    [
        ([], SLOTS),
        ([1.0], SLOTS),
        ([True], SLOTS),
        ([0], SLOTS),
        ([95], SLOTS),
        ([1], []),
        ([1], [2]),
        ([1], [True, 5]),
        ([[1]], SLOTS),
    ],
)
def test_invalid_or_out_of_volume_indices_are_rejected(centers, sizes):
    with pytest.raises(ValueError):
        series_donor_indices(centers, sizes)


def test_an_unknown_policy_is_not_silently_treated_as_legacy():
    with pytest.raises(ValueError, match="policy"):
        series_donor_indices([1], SLOTS, mode="interpolate")
