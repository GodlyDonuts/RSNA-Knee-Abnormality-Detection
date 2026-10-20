"""Exclusive operational relocation; numerical and input lineage remain frozen."""

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

PREVIOUS = "291283545c930b730bb257abb10c2a0814b055eb"
PREVIOUS_DIR = "initiative/research/predictive-http499-owner-handoff-20261020-v1"
SCIENCE = "44cc559e947560a635c34fee7b1f2a3b7be3a241"
PIDS = {"fitting": 29171, "deployment": 29172, "observer": 29173}
ARMS = ("B", "C", "C0")
PREPHASE = {"owner-intent.json", "launch-intent.json", "launcher.json", "stdout", "stderr",
            "frozen_phase_launcher.py", "frozen-phase-source.json"}
CUTOFFS = {"admission": "2026-10-21T09:15:00+00:00",
           "transfer": "2026-10-21T10:45:00+00:00",
           "dispatch": "2026-10-21T14:00:00+00:00"}
END = "2026-10-22T12:00:00+00:00"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def hex64(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def stamp(value):
    result = datetime.fromisoformat(value)
    require(result.utcoffset() is not None and result.utcoffset().total_seconds() == 0,
            "Explicit UTC required")
    return result.timestamp()


def handoff(value, role="fitting"):
    require(role in PIDS and value["pid"] == PIDS[role] and value["source_git"] == PREVIOUS,
            "Exact preceding owner required")
    status = ("actual_idle_training_observer_stopped_before_dispatch" if role == "observer"
              else "actual_idle_deadline_placement_owner_stopped_before_phase")
    require(value["status"] == status and value["children"] == []
            and value.get("phase_intents", []) == [] and hex64(value["original_intent_sha256"]),
            "Exclusive idle owner proof required")
    if role == "observer":
        require(value["process_absent_after_stop"] is True
                and value["remote_allocations"] == value["remote_mutations"] == 0,
                "Read-only observer stop required")
    else:
        require(value["role"] == role and value["owner_process_absent_after_stop"] is True
                and value["remote_admission_or_dispatch_exists"] is False
                and value["remote_jobs_cancelled"] == []
                and hex64(value["remote_read_only_snapshot_sha256"]),
                "No primary phase or remote cancellation may precede handoff")
    return value


def local_handoff(value, role, repo, process_command, *, now=None):
    handoff(value, role)
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    require(0 <= now - stamp(value["observed_at"]) <= 1800, "Fresh idle handoff required")
    require(not process_command(PIDS[role]).strip(), "Preceding owner remains live")
    base = repo / "artifacts" / Path(PREVIOUS_DIR).name
    journal = (base / "finite-owner" if role == "fitting" else
               base.with_name(base.name + "-deployment") if role == "deployment" else base / "observer")
    if role == "observer":
        # Original observer stores its journal beneath the audit-specific observer suffix.
        candidates = [base, journal, base.with_name(base.name + "-observer")]
        candidates = [p for p in candidates if (p / "observer-intent.json").is_file()]
        require(len(candidates) == 1, "Exact original observer journal required")
        journal = candidates[0]
        allowed, name = {"finite-owner", "observer-intent.json", "launcher.json", "stdout", "stderr"}, "observer-intent.json"
    else:
        allowed, name = PREPHASE, "owner-intent.json"
    require({p.name for p in journal.iterdir()} <= allowed, "Old phase evidence requires reconciliation")
    raw = (journal / name).read_bytes()
    intent = json.loads(raw)
    require(sha(raw) == value["original_intent_sha256"] and intent["pid"] == PIDS[role],
            "Original owner intent changed")
    field = "placement_source_git" if role == "deployment" else "source_git"
    if role == "observer":
        field = "observer_source_git"
    require(intent[field] == PREVIOUS, "Original owner source differs")
    return journal


def decision(value, source_git):
    require(value["status"] == "accepted_all_arms_evc27_deadline_placement"
            and value["placement_source_git"] == source_git and value["science_git"] == SCIENCE
            and value["ownership"] == dict.fromkeys(ARMS, "evc27")
            and value["dispatch_order"] == ["C", "B", "C0"]
            and value["same_six_shared_GPU_logs_reassigned"] is True
            and value["shared_future_bytes"] >= 123 * 2**20
            and value["shared_future_files"] >= 376
            and value["retained_total_scratch_bytes"] == 6 * 2**30
            and value["no_paid_credits"] is True
            and value["training_wave_seconds"] == 57600,
            "Explicit two-wave placement/funding decision required")
    require(value["stage_cutoffs_UTC"] == CUTOFFS and value["preservation_end_UTC"] == END,
            "Do not retain the previous single-wave deadline")
    return value


def return_root(arm):
    require(arm in ARMS, "Exact primary arm required")
    return Path(f"/tmp/rsna-predictive-{arm}-evc27-return-20261020-v1")
