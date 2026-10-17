import pytest

from scripts.kaggle.reference_capacity import OLD_SHA, OLD_SLUG, acknowledged_reference_capacity


def original():
    reference = {
        "slug": OLD_SLUG,
        "version": 2,
        "kernel_id": 139199164,
        "main": False,
        "status": "CANCEL_ACKNOWLEDGED",
    }
    other = {**reference, "slug": "godlydonuts/other", "kernel_id": 123, "status": None}
    raw = {
        "maximum_batch_GPU_sessions": 2,
        "reserved_main_lanes": 0,
        "effective_occupied_lanes": 2,
        "actual_active_registered_sessions": [reference, other],
        "available": False,
    }
    proof = {
        "kernel_id": 139199164,
        "version": 2,
        "actual_session": {"status": "CANCEL_ACKNOWLEDGED"},
        "fresh_source_sha256": OLD_SHA,
        "original_source_preserved": True,
    }
    return raw, proof


def test_only_actual_exact_terminal_retires_unknown_other_stays_charged():
    raw, proof = original()
    out = acknowledged_reference_capacity(raw, proof)
    assert out["available"] and out["effective_occupied_lanes"] == 1
    assert len(out["actual_active_registered_sessions"]) == 1
    assert out["actual_active_registered_sessions"][0]["status"] is None
    assert raw["available"] is False and raw["effective_occupied_lanes"] == 2


@pytest.mark.parametrize("status", [None, "QUEUED", "RUNNING", "CANCEL_REQUESTED"])
def test_never_retires_nonterminal_or_missing_status(status):
    raw, proof = original()
    raw["actual_active_registered_sessions"][0]["status"] = status
    with pytest.raises(ValueError):
        acknowledged_reference_capacity(raw, proof)


@pytest.mark.parametrize(
    "field,value", [("kernel_id", 1), ("version", 1), ("fresh_source_sha256", "0" * 64)]
)
def test_exact_source_identity_required(field, value):
    raw, proof = original()
    proof[field] = value
    with pytest.raises(ValueError):
        acknowledged_reference_capacity(raw, proof)


def test_reserved_main_lane_is_preserved():
    raw, proof = original()
    raw.update(reserved_main_lanes=1, effective_occupied_lanes=3)
    out = acknowledged_reference_capacity(raw, proof)
    assert out["available"] is False and out["effective_occupied_lanes"] == 2


def test_inconsistent_original_accounting_rejected():
    raw, proof = original()
    raw["effective_occupied_lanes"] = 1
    with pytest.raises(ValueError):
        acknowledged_reference_capacity(raw, proof)
