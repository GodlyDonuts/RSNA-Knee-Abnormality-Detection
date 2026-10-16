import pytest

from scripts.newton.run_compact_neighbor_recovery import validate_admission


def admitted():
    return {
        "status": "matched_source_neighbor_inputs_admitted",
        "source_git": "3803b717b69db6c5e89e11724c4daee15490cbe2",
        "node": "evc33",
        "fitting_count": 2247,
        "outer_count": 1156,
        "center_channel_bit_equal": True,
        "original_fitting_outer_release_masks_unchanged": True,
        "donor_windows_changed": 1,
        "GPU_used": False,
    }


def test_admission_is_specific_to_real_matched_center_input_evidence():
    validate_admission(admitted())
    for key, value in [
        ("center_channel_bit_equal", False),
        ("fitting_count", 2603),
        ("donor_windows_changed", 0),
        ("source_git", "dcfff319cb7bd7bbae679a20da2b1b9268e8d680"),
        ("status", "native_input_gate_passed"),
        ("GPU_used", True),
    ]:
        with pytest.raises(ValueError, match="admission"):
            validate_admission({**admitted(), key: value})
