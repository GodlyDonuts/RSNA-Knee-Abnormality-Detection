import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("predictive_deadline_policy", Path(__file__).with_name("policy.py"))
policy = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = policy
spec.loader.exec_module(policy)


def budget():
    return policy.Budget(
        deployment_wall_seconds=3 * 3600,
        scheduler_wait_seconds=2 * 3600,
        publication_return_seconds=3600,
        cache_admission_seconds=90 * 60,
        c0_transfer_seconds=150 * 60,
        c0_dispatch_control_seconds=45 * 60,
    )


@pytest.mark.parametrize("stage", policy.STAGES)
def test_boundary_rejects_only_new_stage_after_cutoff(stage):
    b = budget()
    cutoff = b.final_ready_by - timedelta(seconds=b.remaining_seconds(stage))
    at = policy.assess_dispatch(cutoff, stage, b, preservation_owner_expires_at=b.final_ready_by)
    after = policy.assess_dispatch(
        cutoff + timedelta(microseconds=1), stage, b,
        preservation_owner_expires_at=b.final_ready_by,
    )
    assert at["deadline_allows_new_stage"]
    assert not after["deadline_allows_new_stage"]
    assert after["already_submitted_jobs_must_still_be_observed_and_preserved"]
    assert not at["authorizes_remote_mutation"]


def test_current_nine_am_expiry_is_not_a_safe_nine_am_dispatch_cutoff():
    b = budget()
    result = policy.assess_dispatch(
        datetime(2026, 10, 22, 8, tzinfo=timezone.utc), "bc_dispatch", b,
        preservation_owner_expires_at=datetime(2026, 10, 22, 9, tzinfo=timezone.utc),
    )
    assert result["preservation_owner_covers_final_ready_by"]
    assert not result["new_stage_has_time"]
    assert result["stop_new_stage_after_utc"] == "2026-10-21T22:00:00+00:00"


def test_independent_concurrent_paths_and_portable_critical_path():
    b = budget()
    assert b.remaining_seconds("bc_dispatch") == 11 * 3600
    assert b.remaining_seconds("c0_dispatch") == 11 * 3600
    assert b.remaining_seconds("c0_transfer") == 14 * 3600 + 15 * 60
    assert b.remaining_seconds("cache_admission") == 15 * 3600 + 45 * 60
    # Completed transfer changes the remaining work without weakening the final deadline.
    now = datetime(2026, 10, 21, 20, tzinfo=timezone.utc)
    args = {"preservation_owner_expires_at": b.final_ready_by}
    assert policy.assess_dispatch(now, "c0_dispatch", b, **args)["deadline_allows_new_stage"]
    assert not policy.assess_dispatch(now, "c0_transfer", b, **args)["deadline_allows_new_stage"]


def test_old_owner_expiration_can_independently_block_a_new_dispatch():
    b = budget()
    result = policy.assess_dispatch(
        datetime(2026, 10, 20, 15, tzinfo=timezone.utc), "bc_dispatch", b,
        preservation_owner_expires_at=b.final_ready_by - timedelta(seconds=1),
    )
    assert result["new_stage_has_time"]
    assert not result["preservation_owner_covers_final_ready_by"]
    assert not result["deadline_allows_new_stage"]


@pytest.mark.parametrize("state", ["PENDING", "RUNNING", "COMPLETING"])
def test_launch_deadline_does_not_terminate_existing_job_retention(state):
    assert policy.retention_action(was_submitted=True, state=state, final_sealed=False) == (
        "continue_original_observation_and_preserve_each_sealed_epoch"
    )


def test_timeout_preserves_completed_epochs_without_silent_retraining():
    assert policy.retention_action(was_submitted=True, state="TIMEOUT", final_sealed=False) == (
        "preserve_completed_epochs_and_record_failure_no_automatic_restart"
    )
    assert policy.retention_action(was_submitted=True, state="TIMEOUT", final_sealed=True) == (
        "preserve_original_final_and_reconcile_actual_terminal"
    )


def test_requires_explicit_timezone_and_separate_runtime_accounting():
    with pytest.raises(ValueError, match="timezone"):
        policy.assess_dispatch(datetime(2026, 10, 20), "bc_dispatch", budget(),
                               preservation_owner_expires_at=budget().final_ready_by)
    values = vars(budget()).copy()
    values["deployment_wall_seconds"] = 3000
    with pytest.raises(ValueError, match="VM allocation"):
        policy.Budget(**values)


def test_unrecognized_stage_refuses_to_guess():
    with pytest.raises(ValueError, match="launch stage"):
        budget().remaining_seconds("restart_failed_job")
