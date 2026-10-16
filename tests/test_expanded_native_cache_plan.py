"""The additional population cannot reclaim held-out or correspondence-excluded rows."""

import pandas as pd
import pytest

from rsna.schema import TARGETS
from scripts.newton.expanded_native_cache_plan import select_additional


def inputs():
    rows = []
    for uid, split, allowed, mask in [
        ("new", "train", True, 0),
        ("unsafe", "train", False, 0),
        ("old", "train", True, 1),
        ("unobserved_outer", "train", True, 0),
        ("observed_outer", "train", True, 1),
        ("gold", "gold_evaluation", True, 0),
    ]:
        rows.append(
            {
                "study_uid": uid,
                "split": split,
                "leakage_group": uid,
                "report_supervision_allowed": allowed,
                **{target + "__mask": mask for target in TARGETS},
            }
        )
    studies = pd.DataFrame(rows)
    folds = pd.DataFrame({"study_uid": [r["study_uid"] for r in rows[:5]], "fold": [1, 1, 1, 0, 0]})
    return studies, folds


def test_unobserved_outer_and_unsafe_reports_do_not_enter_additional_fit():
    studies, folds = inputs()
    selected, outer = select_additional(studies, folds)
    assert selected.study_uid.tolist() == ["new"]
    assert set(outer.study_uid) == {"unobserved_outer", "observed_outer"}


def test_group_crossing_or_protected_overlap_is_rejected():
    studies, folds = inputs()
    studies.loc[studies.study_uid.eq("new"), "leakage_group"] = "unobserved_outer"
    with pytest.raises(ValueError, match="cross folds"):
        select_additional(studies, folds)
    studies, folds = inputs()
    studies.loc[studies.study_uid.eq("new"), "leakage_group"] = "gold"
    with pytest.raises(ValueError, match="protected"):
        select_additional(studies, folds)


def test_missing_fold_and_nonbinary_masks_are_rejected():
    studies, folds = inputs()
    with pytest.raises(ValueError, match="membership"):
        select_additional(studies, folds.iloc[1:])
    studies.loc[0, TARGETS[0] + "__mask"] = 2
    with pytest.raises(ValueError, match="Binary"):
        select_additional(studies, folds)
