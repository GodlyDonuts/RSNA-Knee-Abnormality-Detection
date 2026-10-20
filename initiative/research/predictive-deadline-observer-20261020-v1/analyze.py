"""Local conditional deadline analysis of already collected, original aggregate receipts.

No network, scheduler commands, private cache reads, or training/control mutations.
"""

import argparse
import hashlib
import json
import math
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

TOTAL = 3887
SOURCE = "91224c242c5758fcb25b71c27d0a0b920458ee6b"
JOBS = {"raw": "979664", "context": "979665"}
INTENTS = {
    "raw": "aa9e9876f21a5bc1c1754c9c7050c30e4a481ce725b5ce30c804f10a775e1e58",
    "context": "810ccba69551b436c3397c4bdcbbf756523e190f7f459b2416e74c1c92d2376a",
}
PREFIX = 522
MAX_RECEIPTS = 64
MAX_BYTES = 256 * 1024


def require(condition, message):
    if not condition:
        raise ValueError(message)


def utc(value, *, scheduler=False):
    result = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if scheduler and result.tzinfo is None:
        result = result.replace(tzinfo=timezone.utc)
    require(result.tzinfo is not None, "Explicit timestamp timezone required")
    return result.astimezone(timezone.utc)


def timestamp(value):
    return value.isoformat().replace("+00:00", "Z") if value is not None else None


def read_small(path):
    path = Path(path)
    require(
        path.is_file() and not path.is_symlink() and path.stat().st_size <= MAX_BYTES,
        "Only bounded existing regular receipt files are accepted",
    )
    data = path.read_bytes()
    require(len(data) <= MAX_BYTES, "Receipt grew beyond bound")
    return json.loads(data), {
        "file": path.name,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
    }


def progress_samples(receipts, as_of):
    require(1 <= len(receipts) <= MAX_RECEIPTS, "Bounded nonempty receipt selection required")
    roles = {role: {} for role in JOBS}
    for receipt in receipts:
        observed = utc(receipt["observed_at"]) if receipt.get("observed_at") else None
        require(observed is None or observed <= as_of, "Observation is later than analysis time")
        require(set(receipt["roles"]) == set(JOBS), "Both original producer roles required")
        for role, expected_job in JOBS.items():
            value = receipt["roles"][role]
            require(
                value["CPU_job"] == expected_job
                and value["source_git"] == SOURCE
                and value["intent_sha256"] == INTENTS[role],
                "Different job/source/intent cannot be combined into this rate series",
            )
            progress = value["progress"]
            require(
                isinstance(progress, dict)
                and set(progress)
                <= {"completed", "total", "elapsed_seconds", "reused_verified_originals"},
                "Only original aggregate progress counters accepted",
            )
            completed, elapsed = progress["completed"], progress["elapsed_seconds"]
            require(
                type(completed) is int and 0 <= completed <= TOTAL and progress["total"] == TOTAL,
                "Original whole-study counters required",
            )
            require(
                type(elapsed) in (int, float) and math.isfinite(elapsed) and elapsed >= 0,
                "Finite monotonic worker elapsed time required",
            )
            if role == "raw":
                require(
                    progress["reused_verified_originals"] == PREFIX and completed >= PREFIX,
                    "Raw cumulative pace must exclude the retained522-case prefix",
                )
            sample = {
                "completed": completed,
                "elapsed_seconds": float(elapsed),
                "observed_at": observed,
            }
            previous = roles[role].get(float(elapsed))
            if previous:
                require(
                    previous["completed"] == completed, "Conflicting count for same worker time"
                )
                # A repeat read is not new work. Keep the first timestamp as a
                # lower bound on sidecar age; never re-date old work as progress.
                dates = [v for v in (previous["observed_at"], observed) if v is not None]
                sample["observed_at"] = min(dates) if dates else None
            roles[role][float(elapsed)] = sample
    result = {}
    for role, by_elapsed in roles.items():
        samples = [by_elapsed[key] for key in sorted(by_elapsed)]
        require(
            all(b["completed"] >= a["completed"] for a, b in zip(samples, samples[1:])),
            "Original completed counter regressed",
        )
        known = [s["observed_at"] for s in samples if s["observed_at"] is not None]
        require(known == sorted(known), "Observation order conflicts with original worker elapsed")
        result[role] = samples
    return result


def parse_scheduler(receipt, as_of):
    if receipt is None:
        return {}, None
    require(receipt.get("timezone") == "UTC", "Scheduler timestamp timezone must explicitly be UTC")
    observed = utc(receipt["observed_at"])
    require(observed <= as_of, "Scheduler observation is in the future")
    require(
        isinstance(receipt.get("jobs"), dict) and set(receipt["jobs"]) == set(JOBS.values()),
        "Exactly the two original sanitized scheduler rows required",
    )
    jobs = {}
    for job, fields in receipt["jobs"].items():
        require(fields["JobId"] == job, "Scheduler job key/body differ")
        role = next(role for role, value in JOBS.items() if value == job)
        require(
            fields["JobName"] == f"rsna-predictive-{role}-alias-recovery-v3"
            and fields["NodeList"] == "evc33",
            "Original job name/node binding changed",
        )
        start, end = fields["StartTime"], fields["EndTime"]
        require(
            start not in ("Unknown", "N/A", "None") and end not in ("Unknown", "N/A", "None"),
            "Actual scheduler start/end required, not an estimated replacement",
        )
        start, end = utc(start, scheduler=True), utc(end, scheduler=True)
        require(start <= observed and start < end, "Scheduler start/end chronology invalid")
        state = fields["JobState"]
        require(re.fullmatch(r"[A-Z_]+", state), "Explicit original scheduler state required")
        jobs[role] = {
            "job": job,
            "state": state,
            "start": start,
            "end": end,
            "time_limit": fields.get("TimeLimit"),
            "observed_at": observed,
        }
    require(set(jobs) == set(JOBS), "Both original scheduler rows required")
    return jobs, observed


def rate(completed, seconds):
    return completed / seconds if seconds > 0 else None


def projected_seconds(remaining, pace):
    if remaining == 0:
        return 0.0
    return remaining / pace if pace is not None and pace > 0 else None


def role_analysis(role, samples, scheduler, as_of, stale_seconds):
    latest = samples[-1]
    remaining = TOTAL - latest["completed"]
    intervals = [
        {
            "studies": b["completed"] - a["completed"],
            "seconds": b["elapsed_seconds"] - a["elapsed_seconds"],
            "studies_per_second": rate(
                b["completed"] - a["completed"], b["elapsed_seconds"] - a["elapsed_seconds"]
            ),
        }
        for a, b in zip(samples, samples[1:])
    ]
    recent = intervals[-3:]
    latest_rate = intervals[-1]["studies_per_second"] if intervals else None
    # Context startup may render retained raw data rapidly. Its full window is
    # throughput under upstream supply, not independent renderer capacity.
    if role == "raw":
        cumulative = rate(latest["completed"] - PREFIX, latest["elapsed_seconds"])
        cumulative_basis = "new_raw_studies_only_excludes522_reused_originals"
    else:
        first = samples[0]
        cumulative = rate(
            latest["completed"] - first["completed"],
            latest["elapsed_seconds"] - first["elapsed_seconds"],
        )
        cumulative_basis = (
            "observed_context_window_includes_upstream_wait_not_intrinsic_renderer_speed"
        )
    observed = latest["observed_at"]
    age = (as_of - observed).total_seconds() if observed else None
    pace_values = [entry["studies_per_second"] for entry in recent]
    if cumulative is not None:
        pace_values.append(cumulative)
    pace_values = [value for value in pace_values if value is not None]
    fast = max(pace_values) if pace_values else None
    slow = min(pace_values) if pace_values else None
    scenarios = {}
    for name, pace in (
        ("latest_interval", latest_rate),
        ("cumulative", cumulative),
        ("fastest_observed_pace", fast),
        ("slowest_observed_pace", slow),
    ):
        seconds = projected_seconds(remaining, pace)
        finish = (
            observed + timedelta(seconds=seconds)
            if observed is not None and seconds is not None
            else None
        )
        scenarios[name] = {
            "studies_per_minute": pace * 60 if pace is not None else None,
            "remaining_seconds_at_observation": seconds,
            "conditional_finish_UTC": timestamp(finish),
            "scheduler_slack_seconds": (scheduler["end"] - finish).total_seconds()
            if scheduler and finish is not None
            else None,
        }
    usable = (
        scheduler
        and scheduler["state"] == "RUNNING"
        and (as_of - scheduler["observed_at"]).total_seconds() <= stale_seconds
        and scheduler["end"] > as_of
    )
    budget = (scheduler["end"] - observed).total_seconds() if scheduler and observed else None
    need = remaining / budget * 60 if budget is not None and budget > 0 else None
    slow_slack = scenarios["slowest_observed_pace"]["scheduler_slack_seconds"]
    fast_slack = scenarios["fastest_observed_pace"]["scheduler_slack_seconds"]
    if not usable:
        risk = "scheduler_unavailable_or_not_running"
    elif observed is None or age > stale_seconds:
        risk = "progress_stale_or_undated_reobserve_before_deciding"
    elif remaining == 0:
        risk = "counts_complete_terminal_verification_still_required"
    elif slow_slack is not None and slow_slack >= 0:
        risk = "observed_pace_scenarios_fit_current_wall_limit_conditionally"
    elif fast_slack is not None and fast_slack < 0:
        risk = "all_observed_pace_scenarios_exceed_current_wall_limit"
    else:
        risk = "wall_limit_at_risk_under_slower_or_stalled_observed_pace"
    return {
        "job": JOBS[role],
        "completed": latest["completed"],
        "total": TOTAL,
        "remaining": remaining,
        "sample_count": len(samples),
        "worker_elapsed_seconds": latest["elapsed_seconds"],
        "progress_observed_at": timestamp(observed),
        "sidecar_age_lower_bound_seconds": age,
        "progress_stale_or_undated": age is None or age > stale_seconds,
        "cumulative_basis": cumulative_basis,
        "recent_intervals": recent,
        "scenarios": scenarios,
        "required_studies_per_minute_at_observation": need,
        "scheduler": None
        if not scheduler
        else {
            "state": scheduler["state"],
            "start_UTC": timestamp(scheduler["start"]),
            "end_UTC": timestamp(scheduler["end"]),
            "time_limit": scheduler["time_limit"],
            "observed_at": timestamp(scheduler["observed_at"]),
            "wall_seconds_remaining_at_analysis": (scheduler["end"] - as_of).total_seconds(),
        },
        "risk": risk,
    }


def analyze(receipts, scheduler_receipt=None, *, as_of, stale_seconds=900):
    require(type(stale_seconds) is int and 1 <= stale_seconds <= 86400, "Bounded staleness horizon")
    as_of = utc(as_of) if isinstance(as_of, str) else as_of.astimezone(timezone.utc)
    samples = progress_samples(receipts, as_of)
    scheduler, scheduler_observed = parse_scheduler(scheduler_receipt, as_of)
    roles = {
        role: role_analysis(role, samples[role], scheduler.get(role), as_of, stale_seconds)
        for role in JOBS
    }
    pipeline = {}
    for name in ("latest_interval", "cumulative", "fastest_observed_pace", "slowest_observed_pace"):
        values = [roles[role]["scenarios"][name]["conditional_finish_UTC"] for role in JOBS]
        pipeline[name] = timestamp(max(utc(value) for value in values)) if all(values) else None
    return {
        "status": "conditional_original_predictive_cache_deadline_analysis",
        "analyzed_at": timestamp(as_of),
        "source_git": SOURCE,
        "roles": roles,
        "raw_minus_context_completed": roles["raw"]["completed"] - roles["context"]["completed"],
        "pipeline_conditional_finish_UTC": pipeline,
        "scheduler_observation_age_seconds": (as_of - scheduler_observed).total_seconds()
        if scheduler_observed
        else None,
        "scheduler_observation_stale": scheduler_observed is None
        or (as_of - scheduler_observed).total_seconds() > stale_seconds,
        "limitations": [
            "Conditional arithmetic from original count/time deltas; not a confidence interval or guarantee.",
            "The slow/fast spread uses at most the last three observed intervals plus cumulative pace; unseen tails may fall outside it.",
            "The context renderer waits on raw supply; its rate is not independent compute throughput.",
            "Sidecars update after completed studies; observation timestamps do not reveal in-flight case age.",
            "Counter completion does not establish original COMPLETED0:0, sealed index or consumer admission.",
            "A forecast never authorizes cancellation, restart, transfer or duplicate work.",
        ],
        "remote_calls": 0,
        "mutations": 0,
        "restart_authorized": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--progress", type=Path, nargs="+", required=True)
    parser.add_argument("--scheduler", type=Path)
    parser.add_argument("--as-of", required=True, help="Explicit UTC analysis timestamp")
    parser.add_argument("--stale-seconds", type=int, default=900)
    args = parser.parse_args()
    require(len(args.progress) <= MAX_RECEIPTS, "At most64 original receipts")
    rows = [read_small(path) for path in args.progress]
    scheduler, proof = read_small(args.scheduler) if args.scheduler else (None, None)
    result = analyze(
        [r[0] for r in rows], scheduler, as_of=args.as_of, stale_seconds=args.stale_seconds
    )
    result["progress_receipts"] = [r[1] for r in rows]
    result["scheduler_receipt"] = proof
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
