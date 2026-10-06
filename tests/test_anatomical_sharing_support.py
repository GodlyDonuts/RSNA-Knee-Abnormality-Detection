"""Unknown labels cannot become negatives or fabricated rare-disease positives."""

import numpy as np
import pandas as pd
import pytest

from scripts.research.anatomical_sharing_support import GROUPS, logical_union, summarize


def test_positive_union_and_negative_intersection_preserve_unknowns():
    labels = np.array([[1, np.nan], [0, np.nan], [0, 0], [np.nan, np.nan], [0, 1]])
    masks = np.array([[1, 0], [1, 0], [1, 1], [0, 0], [1, 1]])
    original = labels.copy()
    values, observed = logical_union(labels, masks)
    assert values.tolist() == [1, 0, 0, 0, 1]
    assert observed.tolist() == [1, 0, 1, 0, 1]
    np.testing.assert_array_equal(labels, original)


@pytest.mark.parametrize(
    "labels,masks", [([[1, 2]], [[1, 1]]), ([[0, 1]], [[1, 0.5]]), ([[0, np.nan]], [[1, 1]])]
)
def test_unadmitted_labels_or_confidence_weights_are_rejected(labels, masks):
    with pytest.raises(ValueError):
        logical_union(labels, masks)


def test_support_counts_components_and_observed_sibling_negatives_without_case_records():
    frame = pd.DataFrame({"leakage_group": ["a", "a", "b", "c"]})
    for members in GROUPS.values():
        for i, target in enumerate(members):
            frame[target] = [1, 0, 0, np.nan] if i == 0 else [0, 1, 0, np.nan]
            frame[f"{target}__mask"] = [1, 1, 1, 0]
    original = frame.copy(deep=True)
    for row in summarize(frame):
        assert row["positive_studies"] == 2
        assert row["positive_components"] == 1
        assert row["negative_studies"] == 1
        assert row["unknown_studies"] == 1
        assert all(
            member["observed_negative_with_positive_sibling_studies"] == 1
            for member in row["members_support"]
        )
    pd.testing.assert_frame_equal(frame, original)
