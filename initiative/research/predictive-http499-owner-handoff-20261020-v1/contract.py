"""Exact HTTP499 ancestry over the unchanged source-corrected fitting contract."""

import hashlib
import json
import re
import types
from datetime import datetime, timezone
from pathlib import Path

LEGACY = "e9bdd1249fa7cf75f792b3762c2028bd7beb4337"
LEGACY_DIR = "initiative/research/predictive-source-correction-owner-handoff-20261020-v1"
LEGACY_SHA = "1255181070fc9456c9ed2f41d234ffbc6821a065034e8cfa2b7e4bba2f45a9de"
PRIOR = "2b834507e19c405971228e1bc405462cd370f560"
PRIOR_DIR = "initiative/research/predictive-http-owner-handoff-20261020-v1"
BUILDER = "88dfbf5dcab62e820ad746d1d482c0b8c5471642"
BUILDER_DIR = "initiative/research/predictive-v4-owner-handoff-20261020-v1"
CACHE = "initiative/research/predictive-cache-http499-recovery-20261020-v1"
ROOTS = {r: f"/tmp/rsna-predictive-full-{r}-http499-recovery-20261020-v1" for r in ("raw", "context")}
OLD_ROOTS = {r: f"/tmp/rsna-predictive-full-{r}-data-recovery-20261020-v1" for r in ROOTS}
PREDECESSORS = {"raw": "981958", "context": "981960"}
INTENTS = {
    "raw": "7ad45b9e78c7fc987bd913356b3f91c615f9ab446f92dc626b411da154c9d27d",
    "context": "e32a42c84d2b99a8185d19d18178a77fa972665e6008940e983f15e6aac7425c",
}
PREVIOUS_HTTP = "ea7bf0f44af60ec4b0876e85af5cc1536d15cad671ea45d4ac548adccca0e28c"
HTTP_KEYS = ("http499_recovery_decision_sha256", "previous_source_http_sha256", "source_http_sha256")
OBSERVER_SOURCE = "d4ecc9121d384933ca0146b769d6b5f2cfc27019"
OBSERVER_DIR = "initiative/research/predictive-source-correction-observer-continuation-20261020-v1"
PREPHASE = {"owner-intent.json", "launch-intent.json", "launcher.json", "stdout", "stderr",
            "frozen_phase_launcher.py", "frozen-phase-source.json"}
legacy = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def hex64(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def configure(source):
    global legacy
    require(sha(source.encode()) == LEGACY_SHA, "Exact preceding corrected-source contract required")
    legacy = types.ModuleType("frozen_e9_corrected_source_contract")
    exec(compile(source, "frozen_e9_contract.py", "exec"), legacy.__dict__)
    for name in ("PRIOR_SHA", "BUILDER_SHA", "RESILIENT", "RESILIENT_DIR", "RESILIENT_SHA",
                 "POLICY", "BASELINE", "CORRECTED_OBJECT_REVISION", "PLAN", "RECEIPT",
                 "HELPER", "CORRECTIONS", "CORRECTION_KEYS"):
        globals()[name] = getattr(legacy, name)


def bind(http_source, builder_source):
    require(legacy is not None, "Configure exact preceding contract before binding")
    legacy.bind(http_source, builder_source)
    for name in ("PARENT", "SCIENCE", "OLD", "FIT", "ADMIT_SHA", "REMOTE_SHA"):
        globals()[name] = getattr(legacy, name)


def replace_once(source, old, new):
    require(source.count(old) == 1, "Frozen operational source anchor changed")
    return source.replace(old, new)


def fresh(value, now=None):
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    observed = datetime.fromisoformat(value["observed_at"])
    require(observed.utcoffset() is not None and 0 <= now - observed.timestamp() <= 1800,
            "Fresh timezone-aware exclusive handoff required")


def handoff(value, role):
    require(role in ("fitting", "deployment"), "Exact predecessor role")
    require(value["status"] == "actual_idle_http499_owner_stopped_before_phase"
            and value["role"] == role and value["pid"] == (11013 if role == "fitting" else 11014)
            and value["source_git"] == LEGACY
            and value["phase_intents"] == [] and value["children"] == []
            and value["owner_process_absent_after_stop"] is True
            and value["remote_admission_or_dispatch_exists"] is False
            and value["remote_jobs_cancelled"] == []
            and hex64(value["remote_read_only_snapshot_sha256"])
            and hex64(value["original_intent_sha256"]), "Exact idle11013/11014 handoff required")
    return value


def local_handoff(value, role, repo, process_command):
    handoff(value, role)
    fresh(value)
    require(not process_command(value["pid"]).strip(), "Predecessor still exists")
    old = repo / "artifacts" / Path(LEGACY_DIR).name
    journal = old / "finite-owner" if role == "fitting" else old.with_name(old.name + "-deployment")
    require({p.name for p in journal.iterdir()} <= PREPHASE,
            "Previous owner has phase evidence; reconcile read-only")
    raw = (journal / "owner-intent.json").read_bytes()
    intent = json.loads(raw)
    field = "source_git" if role == "fitting" else "placement_source_git"
    require(sha(raw) == value["original_intent_sha256"] and intent["pid"] == value["pid"]
            and intent[field] == LEGACY, "Original owner intent changed")


def observer_handoff(value, repo, process_command):
    require(value["status"] == "actual_idle_training_observer_stopped_before_dispatch"
            and value["pid"] == 11927 and value["source_git"] == OBSERVER_SOURCE
            and value["children"] == [] and value["process_absent_after_stop"] is True
            and value["remote_allocations"] == 0 and value["remote_mutations"] == 0
            and value["actual_training_updates_observed"] == 0, "Exact idle11927 observer stop required")
    fresh(value)
    require(not process_command(11927).strip(), "Prior observer remains live")
    journal = repo / "artifacts" / Path(OBSERVER_DIR).name
    require({p.name for p in journal.iterdir()} <=
            {"observer-intent.json", "continuation-intent.json", "launcher.json", "stdout", "stderr"},
            "Previous observer has results requiring reconciliation")
    raw = (journal / "observer-intent.json").read_bytes()
    continuation = (journal / "continuation-intent.json").read_bytes()
    intent, continued = json.loads(raw), json.loads(continuation)
    require(sha(raw) == value["original_intent_sha256"]
            and sha(continuation) == value["original_continuation_intent_sha256"]
            and intent["pid"] == continued["pid"] == 11927
            and intent["observer_source_git"] == intent["placement_source_git"] == LEGACY
            and continued["continuation_source_git"] == OBSERVER_SOURCE
            and continued["original_source_git"] == LEGACY, "Original observer/continuation intents changed")
    owner_raw = (repo / "artifacts" / Path(LEGACY_DIR).name / "finite-owner/owner-intent.json").read_bytes()
    require(sha(owner_raw) == continued["fitting_owner_intent_sha256"], "Observed fitting owner changed")


def chain(value):
    require(set(value) == set(PREDECESSORS), "Both immediate HTTP499 predecessors required")
    ancestors = None
    for role, job in PREDECESSORS.items():
        item, terminal = value[role], value[role]["actual_terminal"]
        require(str(item["job"]) == job and item["root"] == OLD_ROOTS[role]
                and item["source_git"] == LEGACY and item["intent_sha256"] == INTENTS[role]
                and str(terminal["job_id"]) == job and terminal["state"] == "FAILED"
                and terminal["exit_code"] == "1:0" and terminal["live_queue_empty"] is True
                and hex64(terminal["sha256"]), "Exact HTTP499 terminal lineage required")
        previous = item["ancestor_chain"]
        legacy.chain(previous)
        require(ancestors is None or previous == ancestors, "Raw/context ancestors differ")
        ancestors = previous
    return sha(canonical(value))


def accepted_http(value):
    result = {key: value[key] for key in HTTP_KEYS}
    require(all(hex64(v) for v in result.values())
            and result["previous_source_http_sha256"] == PREVIOUS_HTTP
            and result["source_http_sha256"] != PREVIOUS_HTTP, "Explicit bounded HTTP499 transport proof required")
    return result


def proofs(*values, expected=None):
    inherited = legacy.proofs(*values, expected=expected)
    http = accepted_http(values[0])
    require(all(accepted_http(value) == http for value in values[1:]), "HTTP499 proof records differ")
    return {**inherited, **http}


def newer_paths(source):
    for role in ROOTS:
        require(OLD_ROOTS[role] in source, "Exact corrected-data root missing")
        source = source.replace(OLD_ROOTS[role], ROOTS[role])
    return source


def derived_admission(source):
    return newer_paths(legacy.derived_admission(source))


def derived_remote(source):
    return newer_paths(legacy.derived_remote(source))


def derived_placement(source):
    return legacy.derived_placement(source)


def derived_portable_remote(source):
    return legacy.derived_portable_remote(source)


def derived_deployment_owner(source):
    return replace_once(legacy.derived_deployment_owner(source), legacy.CACHE, CACHE)


def deployment_funding(value):
    legacy.deployment_funding(value)


def correction_policy():
    return legacy.correction_policy()


def validate_corrected_context_rows(rows, control):
    accepted = control["accepted_http499_proofs"]
    require(accepted_http(control) == accepted, "Corrected input HTTP499 admission changed")
    return legacy.validate_corrected_context_rows(rows, control)


def final_source_lineage(value, directory):
    result = legacy.final_source_lineage(value, directory)
    for proof in result["effective_source_lineage"]["per_arm"].values():
        control = proof["cache_successor_control"]
        require(accepted_http(control) == control["accepted_http499_proofs"],
                "Final model lacks accepted HTTP499 input lineage")
    return result
