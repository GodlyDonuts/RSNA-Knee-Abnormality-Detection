"""Pure prospective deadline decisions. No remote calls or authority for dispatch."""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

LATEST_ENTRY = datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
COMPETITION_END = datetime(2026, 10, 22, 23, 59, tzinfo=timezone.utc)
TRAINING_SECONDS = 8 * 3600
STAGES = ("cache_admission", "bc_dispatch", "c0_transfer", "c0_dispatch")


def aware(value):
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("An explicit timezone is required")
    return value.astimezone(timezone.utc)


@dataclass(frozen=True)
class Budget:
    """Explicit planning allowances, not promises about scheduler/service latency."""

    deployment_wall_seconds: int
    scheduler_wait_seconds: int
    publication_return_seconds: int
    cache_admission_seconds: int
    c0_transfer_seconds: int
    c0_dispatch_control_seconds: int

    def __post_init__(self):
        values = vars(self)
        if any(type(value) is not int or value < 0 for value in values.values()):
            raise ValueError("Nonnegative integer seconds required")
        if self.deployment_wall_seconds < 3360:
            raise ValueError("Deployment wall allowance cannot be below its VM allocation")
        if self.publication_return_seconds == 0:
            raise ValueError("Explicit final preservation/return allowance required")

    @property
    def final_ready_by(self):
        return LATEST_ENTRY - timedelta(seconds=self.deployment_wall_seconds)

    def remaining_seconds(self, stage):
        if stage not in STAGES:
            raise ValueError("Unknown exact launch stage")
        training = self.scheduler_wait_seconds + TRAINING_SECONDS
        final = training + self.publication_return_seconds
        if stage in ("bc_dispatch", "c0_dispatch"):
            return final
        portable = self.c0_transfer_seconds + self.c0_dispatch_control_seconds + final
        if stage == "c0_transfer":
            return portable
        # B/C and C0 train concurrently; do not charge three serial eight-hour fits.
        return self.cache_admission_seconds + max(final, portable)


def assess_dispatch(now, stage, budget, *, preservation_owner_expires_at):
    """Return a deadline-only prospective decision; never authorize a mutation.

    The caller must retain its exact-once journal/source/funding/runtime checks.
    A false result forbids a *new* stage, not observation of already submitted work.
    """
    now, expires = aware(now), aware(preservation_owner_expires_at)
    seconds = budget.remaining_seconds(stage)
    cutoff = budget.final_ready_by - timedelta(seconds=seconds)
    enough_time = now <= cutoff
    owner_covers_plan = expires >= budget.final_ready_by
    return {
        "status": "prospective_deadline_assessment_no_mutation_authorized",
        "stage": stage,
        "observed_at": now.isoformat(),
        "latest_entry_utc": LATEST_ENTRY.isoformat(),
        "final_ready_by_utc": budget.final_ready_by.isoformat(),
        "stop_new_stage_after_utc": cutoff.isoformat(),
        "remaining_planned_seconds": seconds,
        "training_wall_seconds_per_job": TRAINING_SECONDS,
        "planning_allowances_seconds": vars(budget),
        "deadline_allows_new_stage": enough_time and owner_covers_plan,
        "new_stage_has_time": enough_time,
        "preservation_owner_covers_final_ready_by": owner_covers_plan,
        "preservation_owner_expires_at": expires.isoformat(),
        "already_submitted_jobs_must_still_be_observed_and_preserved": True,
        "queue_delay_is_not_guaranteed_or_bounded_by_this_policy": True,
        "GPU_VM_budget_is_separate_from_wall_clock_budget": True,
        "authorizes_remote_mutation": False,
    }


def retention_action(*, was_submitted, state, final_sealed):
    """Separate preservation from launch cutoffs and unsuccessful terminals."""
    if type(was_submitted) is not bool or type(final_sealed) is not bool:
        raise ValueError("Explicit original state required")
    if not was_submitted:
        return "no_original_job_to_observe"
    if final_sealed:
        return "preserve_original_final_and_reconcile_actual_terminal"
    if state in {"PENDING", "CONFIGURING", "RUNNING", "COMPLETING", "SUSPENDED"}:
        return "continue_original_observation_and_preserve_each_sealed_epoch"
    if state in {"TIMEOUT", "FAILED", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "PREEMPTED"}:
        return "preserve_completed_epochs_and_record_failure_no_automatic_restart"
    return "reconcile_original_state_read_only"
