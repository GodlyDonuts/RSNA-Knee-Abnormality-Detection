import copy
import hashlib

import pandas as pd
import pytest

from rsna.competitive.compact_expanded_train import (
    INDEX_SHA,
    OVERLAY_SHA,
    engineering_rows,
    validate_admission,
)


def admission():
    return {
        "status": "actual_expanded2592_source_and_native_inputs_CPU_verified",
        "node": "evc33",
        "population": {
            "expanded_fitting_count": 2592,
            "baseline_fitting_count": 2247,
            "additional_fitting_count": 345,
            "outer_endpoint_count": 1156,
            "excluded_all_fold_studies": 1301,
        },
        "overlay": {"overlay_sha256": OVERLAY_SHA},
        "metadata_union_index_sha256": INDEX_SHA,
        "old_image_tensor_parity": True,
        "all_old_labels_masks_and_outer_unchanged": True,
        "encoder_optimizer_GPU_or_accuracy_executed": False,
        "source_MRI_payload_bytes_copied": 0,
    }


def test_exact_actual_population_input_receipt_required():
    validate_admission(admission())


@pytest.mark.parametrize(
    "field",
    [
        "old_image_tensor_parity",
        "all_old_labels_masks_and_outer_unchanged",
        "encoder_optimizer_GPU_or_accuracy_executed",
        "source_MRI_payload_bytes_copied",
    ],
)
def test_cannot_replace_actual_image_parity_with_a_looser_claim(field):
    value = admission()
    value[field] = not value[field] if isinstance(value[field], bool) else 1
    with pytest.raises(ValueError):
        validate_admission(value)


@pytest.mark.parametrize(
    "field",
    [
        "expanded_fitting_count",
        "baseline_fitting_count",
        "additional_fitting_count",
        "outer_endpoint_count",
        "excluded_all_fold_studies",
    ],
)
def test_roster_sizes_are_not_interchangeable(field):
    value = admission()
    value["population"][field] += 1
    with pytest.raises(ValueError):
        validate_admission(value)


def test_engineering_mix_includes_actual_additional_cases_without_using_target_values():
    frame = pd.DataFrame(
        {
            "study_uid": [f"old{i}" for i in range(20)] + [f"added{i}" for i in range(20)],
            "target": [0] * 40,
        }
    )
    original = frame.copy(deep=True)
    old = set(frame.study_uid[:20])
    selected = engineering_rows(frame, old)
    assert len(selected) == 16 and sum(selected.study_uid.isin(old)) == 8
    expected = sorted(frame.study_uid[20:], key=lambda x: hashlib.sha256(x.encode()).hexdigest())[
        :8
    ]
    assert set(selected.study_uid) - old == set(expected)
    changed = copy.deepcopy(frame)
    changed["target"] = float("nan")
    assert engineering_rows(changed, old).study_uid.tolist() == selected.study_uid.tolist()
    pd.testing.assert_frame_equal(frame, original)
    with pytest.raises(ValueError):
        engineering_rows(frame, old - set(frame.study_uid[:18]))
