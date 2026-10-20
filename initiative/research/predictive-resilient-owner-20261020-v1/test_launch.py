"""Original-owner exclusivity and exact capsule checks without remote operations."""

import importlib.util
from datetime import datetime, timezone
from pathlib import Path

import pytest

PATH = Path(__file__).with_name("launch.py")
SPEC = importlib.util.spec_from_file_location("resilient_launch_tests", PATH)
M = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(M)
NOW = datetime(2026, 10, 20, 16, tzinfo=timezone.utc).timestamp()


def valid():
    return {
        "status": "actual_idle_91865_stopped_before_any_phase",
        "pid": 91865, "source_git": M.OPERATIONAL, "children_before_stop": [],
        "children_after_stop": [], "owner_process_absent_after_stop": True,
        "phase_intents": [], "remote_jobs_cancelled": [],
        "observed_at": datetime.fromtimestamp(NOW, timezone.utc).isoformat(),
    }


def check(value=None, owner=None, files=None, live=""):
    M.validate_handoff(
        value or valid(), owner or {"pid": 91865, "source_git": M.OPERATIONAL},
        files or M.PREPHASE, live, NOW,
    )


def test_actual_prephase_stop_is_admitted():
    check()


@pytest.mark.parametrize("key,value", [
    ("pid", 91866), ("source_git", "a" * 40), ("children_before_stop", [4]),
    ("children_after_stop", [4]), ("owner_process_absent_after_stop", False),
    ("phase_intents", ["dispatch-BC-intent.json"]), ("remote_jobs_cancelled", ["980960"]),
    ("observed_at", "2026-10-20T15:40:00+00:00"),
    ("observed_at", "2026-10-20T16:01:00+00:00"),
])
def test_inexact_or_stale_handoff_rejected(key, value):
    handoff = valid()
    handoff[key] = value
    with pytest.raises(ValueError):
        check(value=handoff)


def test_any_original_phase_disallows_successor():
    with pytest.raises(ValueError):
        check(files=M.PREPHASE | {"contexts-C0-intent.json"})


def test_live_original_process_disallows_successor():
    with pytest.raises(ValueError):
        check(live="python original launch.py fitting")


def test_original_owner_intent_identity_is_fixed():
    with pytest.raises(ValueError):
        check(owner={"pid": 4, "source_git": M.OPERATIONAL})


def test_import_does_not_launch_or_mutate():
    assert M.OLD_PID == 91865
    assert set(M.PATHS) == {"evc33", "evc27"}
    assert len(M.PATHS["evc33"]) == 6 and len(M.PATHS["evc27"]) == 2
