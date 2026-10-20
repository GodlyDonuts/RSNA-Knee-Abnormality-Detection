"""Conditional forecasts from original counters; no network or scheduler mutation."""

import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "initiative/research/predictive-deadline-observer-20261020-v1"
spec = importlib.util.spec_from_file_location("predictive_deadline_analysis", DIR / "analyze.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def snapshot(completed, elapsed, observed, *, context=None):
    return {
        "observed_at": observed,
        "roles": {
            role: {
                "CPU_job": job,
                "source_git": m.SOURCE,
                "intent_sha256": m.INTENTS[role],
                "progress": {
                    "completed": completed if role == "raw" or context is None else context,
                    "total": m.TOTAL,
                    "elapsed_seconds": elapsed,
                    **({"reused_verified_originals": m.PREFIX} if role == "raw" else {}),
                },
            }
            for role, job in m.JOBS.items()
        },
    }


def schedule():
    return {
        "observed_at": "2026-10-20T13:00:00Z",
        "timezone": "UTC",
        "jobs": {
            job: {
                "JobId": job,
                "JobName": f"rsna-predictive-{role}-alias-recovery-v3",
                "JobState": "RUNNING",
                "NodeList": "evc33",
                "StartTime": "2026-10-20T11:00:00",
                "EndTime": "2026-10-21T07:00:00",
                "TimeLimit": "20:00:00",
            }
            for role, job in m.JOBS.items()
        },
    }


def test_raw_cumulative_excludes522_and_latest_rate_detects_slowdown():
    receipts = [
        snapshot(582, 600, "2026-10-20T11:10:00Z"),
        snapshot(702, 1800, "2026-10-20T11:30:00Z"),
        snapshot(732, 7200, "2026-10-20T13:00:00Z"),
    ]
    result = m.analyze(receipts, schedule(), as_of="2026-10-20T13:00:00Z")
    raw = result["roles"]["raw"]
    assert raw["scenarios"]["cumulative"]["studies_per_minute"] == pytest.approx(210 / 120)
    assert raw["scenarios"]["latest_interval"]["studies_per_minute"] == pytest.approx(30 / 90)
    assert raw["scenarios"]["fastest_observed_pace"]["studies_per_minute"] == 6
    assert raw["risk"] == "wall_limit_at_risk_under_slower_or_stalled_observed_pace"
    assert result["mutations"] == 0 and result["restart_authorized"] is False


def test_zero_progress_has_no_finite_upper_time_and_stale_progress_is_flagged():
    receipts = [
        snapshot(1500, 3600, "2026-10-20T12:00:00Z"),
        snapshot(1500, 7200, "2026-10-20T13:00:00Z"),
    ]
    result = m.analyze(receipts, schedule(), as_of="2026-10-20T13:00:00Z")
    raw = result["roles"]["raw"]
    assert raw["scenarios"]["slowest_observed_pace"]["conditional_finish_UTC"] is None
    assert raw["risk"] == "wall_limit_at_risk_under_slower_or_stalled_observed_pace"
    result = m.analyze(receipts, schedule(), as_of="2026-10-20T14:00:00Z")
    assert result["roles"]["raw"]["progress_stale_or_undated"] is True
    assert result["scheduler_observation_stale"] is True
    assert result["roles"]["raw"]["risk"] == "scheduler_unavailable_or_not_running"


@pytest.mark.parametrize(
    "field,value", [("CPU_job", "979000"), ("source_git", "f" * 40), ("intent_sha256", "e" * 64)]
)
def test_different_original_work_cannot_be_combined(field, value):
    value_receipt = snapshot(700, 3600, "2026-10-20T12:00:00Z")
    value_receipt["roles"]["raw"][field] = value
    with pytest.raises(ValueError, match="Different job"):
        m.analyze([value_receipt], as_of="2026-10-20T13:00:00Z")


@pytest.mark.parametrize("timezone", [None, "EDT", "America/New_York"])
def test_scheduler_requires_explicit_UTC_capture_not_local_EDT(timezone):
    value = schedule()
    if timezone is None:
        value.pop("timezone")
    else:
        value["timezone"] = timezone
    with pytest.raises(ValueError, match="explicitly be UTC"):
        m.analyze(
            [snapshot(700, 3600, "2026-10-20T12:00:00Z")], value, as_of="2026-10-20T13:00:00Z"
        )


def test_missing_scheduler_never_invents20hour_end():
    result = m.analyze([snapshot(700, 3600, "2026-10-20T12:00:00Z")], as_of="2026-10-20T12:05:00Z")
    assert result["roles"]["raw"]["scheduler"] is None
    assert result["roles"]["raw"]["scenarios"]["cumulative"]["scheduler_slack_seconds"] is None
    assert result["roles"]["raw"]["risk"] == "scheduler_unavailable_or_not_running"


def test_duplicate_sidecar_is_not_new_work_and_does_not_erase_staleness():
    first = snapshot(700, 3600, "2026-10-20T12:00:00Z")
    again = copy.deepcopy(first)
    again["observed_at"] = "2026-10-20T13:00:00Z"
    result = m.analyze([first, again], schedule(), as_of="2026-10-20T13:00:00Z")
    assert result["roles"]["raw"]["sample_count"] == 1
    assert result["roles"]["raw"]["sidecar_age_lower_bound_seconds"] == 3600
    again["roles"]["raw"]["progress"]["completed"] += 1
    with pytest.raises(ValueError, match="Conflicting count"):
        m.analyze([first, again], as_of="2026-10-20T13:00:00Z")


def test_regressed_count_nonfinite_elapsed_and_excess_input_are_rejected():
    first = snapshot(700, 3600, "2026-10-20T12:00:00Z")
    later = snapshot(699, 7200, "2026-10-20T13:00:00Z")
    with pytest.raises(ValueError, match="regressed"):
        m.analyze([first, later], as_of="2026-10-20T13:00:00Z")
    later["roles"]["raw"]["progress"]["elapsed_seconds"] = float("nan")
    with pytest.raises(ValueError, match="Finite"):
        m.analyze([later], as_of="2026-10-20T13:00:00Z")
    with pytest.raises(ValueError, match="Bounded"):
        m.analyze([first] * 65, as_of="2026-10-20T13:00:00Z")


def test_count_complete_still_requires_terminal_receipt():
    result = m.analyze(
        [snapshot(3887, 7200, "2026-10-20T13:00:00Z")], schedule(), as_of="2026-10-20T13:00:00Z"
    )
    assert result["roles"]["raw"]["risk"] == "counts_complete_terminal_verification_still_required"
    assert (
        result["roles"]["raw"]["scenarios"]["latest_interval"]["remaining_seconds_at_observation"]
        == 0
    )
    assert result["restart_authorized"] is False


def test_actual_original_receipts_match_wall_limit_and_ignore_EDT_archive():
    base = ROOT / "initiative/research/parallel-serving-account-20261020-v1"
    paths = [
        base / name
        for name in (
            "actual-input-progress.json",
            "actual-input-progress-1246.json",
            "actual-input-progress-1315.json",
            "actual-input-progress-1332.json",
            "actual-input-progress-1351.json",
        )
    ]
    records = [m.read_small(path)[0] for path in paths]
    scheduler = m.read_small(base / "actual-input-schedule-1351.json")[0]
    result = m.analyze(records, scheduler, as_of="2026-10-20T13:52:31Z")
    raw = result["roles"]["raw"]
    assert raw["completed"] == 919 and result["raw_minus_context_completed"] == 2
    assert raw["scheduler"]["end_UTC"] == "2026-10-21T07:57:20Z"
    assert raw["scenarios"]["cumulative"]["studies_per_minute"] == pytest.approx(
        397 / 6839.263745398028 * 60
    )
    assert raw["scenarios"]["latest_interval"]["studies_per_minute"] == pytest.approx(
        77 / (6839.263745398028 - 5677.531184692052) * 60
    )
    assert raw["risk"] == "observed_pace_scenarios_fit_current_wall_limit_conditionally"
    assert (
        result["pipeline_conditional_finish_UTC"]["slowest_observed_pace"] < "2026-10-21T07:57:20Z"
    )
    json.dumps(result, allow_nan=False)


def test_small_receipt_boundary_and_no_overwrites(tmp_path):
    path = tmp_path / "huge.json"
    path.write_bytes(b" " * (m.MAX_BYTES + 1))
    with pytest.raises(ValueError, match="bounded"):
        m.read_small(path)
    path.write_text('{"aggregate":1}')
    before = path.read_bytes()
    value, proof = m.read_small(path)
    assert value == {"aggregate": 1} and proof["bytes"] == len(before)
    assert path.read_bytes() == before
