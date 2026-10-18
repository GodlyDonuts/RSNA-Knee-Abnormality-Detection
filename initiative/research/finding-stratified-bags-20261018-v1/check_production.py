"""Refuse incomplete actual-recovery contracts; real source closure is checked on owner."""

import copy

import production as p
import pytest


def fixture():
    row = {
        "finite_updates": 4,
        "complete_model_AdamW_LR_cursors_allrank_RNG_scaler_history_and_epoch_snapshots_equal": True,
        "all16_full94_two_view_prediction_bytes_equal": True,
    }
    return {
        "status": "COMPLETE",
        "kernel_id": 139349540,
        "version": 1,
        "source_sha256": p.RECOVERY,
        "receipt": {
            "status": "actual_finding_strata_bothT4_original_control_and_recovery_accepted",
            "worker_source_sha256": p.RECOVERY_WORKER,
            "source_module_sha256": {},
            "original_influence_complete_control_state_and_prediction_bytes_equal": True,
            "clinical_or_competition_accuracy_claim": False,
            "roles": {k: dict(row) for k in ("iid", "finding_strata")},
        },
    }


def test_complete_contract_and_original_assignment():
    actual = fixture()
    assert p.accept_recovery(actual, {}) == actual["receipt"]
    assert p.assignments("A=1\nB=2\n", {"A": 3}) == "A=3\nB=2\n"
    compile(p.HELPER, "actual_full_tensor_loader", "exec")


@pytest.mark.parametrize(
    "key,value",
    [("status", "RUNNING"), ("kernel_id", 139130771), ("version", 2), ("source_sha256", "0" * 64)],
)
def test_wrong_actual_recovery(key, value):
    actual = fixture()
    actual[key] = value
    with pytest.raises(ValueError):
        p.accept_recovery(actual, {})


@pytest.mark.parametrize(
    "key,value",
    [
        ("worker_source_sha256", "0" * 64),
        ("status", "pending"),
        ("source_module_sha256", {"extra": "0" * 64}),
        ("original_influence_complete_control_state_and_prediction_bytes_equal", False),
        ("clinical_or_competition_accuracy_claim", True),
    ],
)
def test_incomplete_numerical_control(key, value):
    actual = fixture()
    actual["receipt"][key] = value
    with pytest.raises(ValueError):
        p.accept_recovery(actual, {})


@pytest.mark.parametrize("role", ["iid", "finding_strata"])
@pytest.mark.parametrize(
    "key,value",
    [
        ("finite_updates", 0),
        (
            "complete_model_AdamW_LR_cursors_allrank_RNG_scaler_history_and_epoch_snapshots_equal",
            False,
        ),
        ("all16_full94_two_view_prediction_bytes_equal", False),
    ],
)
def test_both_allocations_required(role, key, value):
    actual = copy.deepcopy(fixture())
    actual["receipt"]["roles"][role][key] = value
    with pytest.raises(ValueError):
        p.accept_recovery(actual, {})


def test_source_ambiguity_and_unaccepted_control():
    for source in ("A=1\nA=2\n", "B=2\n"):
        with pytest.raises(ValueError):
            p.assignments(source, {"A": 3})
    with pytest.raises(ValueError):
        p.prepare("unaccepted", "unaccepted", "unaccepted")
    with pytest.raises(ValueError):
        p.one("same same", "same", "changed")
