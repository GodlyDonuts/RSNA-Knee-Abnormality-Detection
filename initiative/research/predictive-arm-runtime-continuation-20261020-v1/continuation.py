"""Conditional runtime-only continuation; import performs no I/O or mutation.

The caller supplies the exact original portable Owner instance and validators.
This module does not create an owner, connect, upload, push, or submit by itself.
"""

import hashlib
import json
import math
import sqlite3
import time
from pathlib import Path

ARMS = ("B", "C", "C0")
CANONICAL = "8fd43b9818c85cbbff9e608ae5ba7d3ae2c6a26f"
QUALIFIED = "actual_trained_predictive_public_T4_qualification_accepted"
RUNTIME_REJECTION = (
    "Selected measured scenario plus chosen margin exceeds the actual nine-hour limit"
)
JOURNAL_NAME = "predictive-arm-runtime-continuation-20261020-v1"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def evaluate(decision, qualification, candidates, validate_qualification, runtime_plan, policy):
    """Validate the complete combined run first; isolate only genuine time excess."""
    require(qualification["status"] == QUALIFIED, "Fully accepted combined qualification required")
    require(
        len(candidates) == 3 and {x["arm"] for x in candidates} == set(ARMS)
        and len(qualification["models"]) == 3
        and {x["arm"] for x in qualification["models"]} == set(ARMS),
        "All three exact fixed-final candidates must be qualified",
    )
    # An invalid weight, source, precision, device, or memory result stops globally.
    for arm in ARMS:
        validate_qualification(qualification, next(c for c in candidates if c["arm"] == arm))
    results = {}
    for arm in ARMS:
        measured = next(row for row in qualification["models"] if row["arm"] == arm)
        try:
            plan = runtime_plan(decision, qualification, arm, policy)
        except ValueError as error:
            if str(error) != RUNTIME_REJECTION:
                raise
            values = {}
            for row in measured["measurements"]:
                seconds = sum(row[k] for k in ("decode_seconds", "render_seconds", "FP16_seconds"))
                require(math.isfinite(seconds) and seconds > 0, "Invalid public runtime evidence")
                values[row["study_index"]] = max(values.get(row["study_index"], 0), seconds)
            require(set(values) == {0, 1, 2}, "Three unique complete public cases required")
            r = decision["runtime"]
            require(r["selected_scenario"] == "max_case_two_lane", "Original scenario required")
            selected = r["startup_allowance_seconds"] + r["planning_study_count"] * max(values.values()) / 2
            require(
                selected + r["safety_margin_seconds"] > 32400,
                "Only actual nine-hour projection excess may be isolated",
            )
            results[arm] = {
                "eligible": False, "error_type": type(error).__name__, "reason": str(error),
                "selected_estimate_seconds": selected,
                "safety_margin_seconds": r["safety_margin_seconds"],
                "measurement": measured,
            }
        else:
            results[arm] = {"eligible": True, "runtime_plan": plan, "measurement": measured}
    return {
        "status": "all_fixed_arms_independently_evaluated_after_combined_qualification",
        "qualification_sha256": digest(qualification),
        "decision_sha256": digest(decision),
        "roles": results,
        "eligible": [arm for arm in ARMS if results[arm]["eligible"]],
        "rejected": [arm for arm in ARMS if not results[arm]["eligible"]],
        "thresholds_or_final_epoch_changed": False,
        "scores_used": False,
    }


def read_original(path):
    path = Path(path)
    require(not path.is_symlink() and path.is_file(), "Original regular receipt required")
    raw = path.read_bytes()
    require(len(raw) < 8 * 2**20, "Bounded original receipt required")
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def read_boundary(root, ledger, *, now=None):
    """Read only retained originals and the canonical SQLite database, never API."""
    root, ledger = Path(root), Path(ledger)
    require(root.is_dir() and not root.is_symlink()
            and ledger.is_file() and not ledger.is_symlink(), "Original private root/ledger required")
    require(not (ledger.parent / "coordinator-claim").exists()
            and not any(Path(str(ledger) + suffix).exists()
                        for suffix in ("-journal", "-wal", "-shm")),
            "Original ledger must be quiescent without transient sidecars")
    prepared, _ = read_original(root / "prepared.json")
    qualification, _ = read_original(root / "qualification-accepted.json")
    identity = prepared["bundle_identity"]
    require(prepared["status"] == "actual_three_predictive_final_exports_prepared_CPU_only",
            "Original full final bundle required")
    rows, kernels, submissions = [], [], []
    with sqlite3.connect(ledger.resolve().as_uri() + "?mode=ro", uri=True) as db:
        db.execute("PRAGMA query_only = ON")
        for candidate in prepared["models"]:
            arm = candidate["arm"]
            require(arm in ARMS, "Known original final arm required")
            keys = (
                "licensed-parent-full-epoch-serving:predictive-final-v1:" + identity + ":" + arm,
                candidate["model"]["sha256"],
            )
            for record, in db.execute(
                "SELECT record FROM entries WHERE sha IN (?,?)", keys
            ):
                value = json.loads(record)
                rows.append(arm)
                if value.get("kernel_id") is not None or value.get("push") is not None:
                    kernels.append(arm)
                if value.get("submission_id") is not None or value.get("submit") is not None:
                    submissions.append(arm)
    return {
        "status": "actual_predictive_runtime_continuation_boundary_read_only",
        "observed_at": time.time() if now is None else now,
        "deployment_root": str(root),
        "canonical_claim_present": (ledger.parent / "coordinator-claim").exists(),
        "qualification_sha256": digest(qualification),
        "prepared_bundle_identity": identity,
        "compiled_arm_directories": [a for a in ARMS if (root / ("compiled-" + a)).exists()],
        "arm_ledger_rows": rows, "arm_kernel_pushes": kernels, "arm_submissions": submissions,
        "external_API_calls": 0, "remote_writes": 0,
    }


def admit_failure(journal, *, source_git, pid, decision, decision_bytes, process_command, remote,
                  validate_qualification, runtime_plan, policy, now=None):
    """Read-only admission of one stopped owner before any public-arm phase.

    `remote` is a fresh separately collected original root/ledger observation.
    Unknown or previously attempted mutations never become retry authority.
    """
    now = time.time() if now is None else now
    journal = Path(journal)
    intent, intent_sha = read_original(journal / "owner-intent.json")
    stopped, stopped_sha = read_original(journal / "stopped.json")
    require(
        intent["pid"] == pid and intent["placement_source_git"] == source_git
        and intent["canonical_account_control_source_git"] == CANONICAL
        and json.loads(decision_bytes) == decision
        and intent["decision_sha256"] == hashlib.sha256(decision_bytes).hexdigest(),
        "Exact original portable owner/source/decision binding required",
    )
    require(
        stopped["error_type"] == "ValueError" and stopped["automatic_retry"] is False
        and stopped["preserve_original_phases_and_qualified_baseline"] is True
        and not process_command(pid).strip(),
        "Original owner must have actually stopped; never interrupt a healthy owner",
    )
    for arm in ARMS:
        require(
            not any(journal.glob(f"*-{arm}-intent.json"))
            and not any(journal.glob(f"*-{arm}-result.json"))
            and not (journal / f"actual-entry-{arm}.json").exists(),
            "Any original public-arm phase requires separate reconciliation",
        )
    require(not (journal / "actual-trained-runtime-decisions.json").exists(),
            "Original all-arm runtime gate already completed")
    qualification, qualification_sha = read_original(journal / "actual-trained-qualification.json")
    prepared, prepared_sha = read_original(journal / "prepare-qualification-result.json")
    pushed, pushed_sha = read_original(journal / "push-qualification-result.json")
    require(prepared["status"] == "actual_three_predictive_final_exports_prepared_CPU_only",
            "Original prepared exact final exports required")
    matches = []
    for path in journal.glob("result-*.json"):
        value, sha = read_original(path)
        if value.get("status") == "COMPLETE" and value.get("receipt") == qualification:
            matches.append((value, sha))
    require(len(matches) == 1, "One original authenticated COMPLETE qualification result required")
    completed, completed_sha = matches[0]
    require(
        pushed["state"] == "predictive_final_pushed"
        and all(completed[k] == pushed[k] for k in ("slug", "version", "kernel_id"))
        and completed["version"] == 1 and completed["hidden_score_claim"] is False,
        "Original exact qualification push/completion identity required",
    )
    require(
        remote["status"] == "actual_predictive_runtime_continuation_boundary_read_only"
        and 0 <= now - remote["observed_at"] <= 180
        and remote["deployment_root"] == decision["root"]
        and remote["canonical_claim_present"] is False
        and remote["qualification_sha256"] == digest(qualification)
        and remote["prepared_bundle_identity"] == prepared["bundle_identity"]
        and all(remote[k] == [] for k in (
            "compiled_arm_directories", "arm_ledger_rows", "arm_kernel_pushes", "arm_submissions"
        )),
        "Fresh original remote root and ledger must prove no arm action",
    )
    evaluation = evaluate(decision, qualification, prepared["models"],
                          validate_qualification, runtime_plan, policy)
    require(bool(evaluation["rejected"]), "No isolated runtime rejection explains this stop")
    return {
        "status": "actual_runtime_only_original_owner_failure_admitted",
        "admitted_at": now,
        "original_journal": str(journal.resolve()), "original_pid": pid,
        "source_git": source_git, "canonical_source_git": CANONICAL,
        "original_receipts": {
            "owner_intent": intent_sha, "stopped": stopped_sha,
            "qualification": qualification_sha, "prepared": prepared_sha,
            "qualification_push": pushed_sha, "qualification_complete": completed_sha,
            "fresh_remote_boundary": digest(remote),
        },
        "qualification": qualification, "evaluation": evaluation,
        "new_model_dataset_or_qualification": False,
        "automatic_retry": False,
    }


def continue_qualified(owner, admission, *, once, finish_deadline):
    """Executable narrow tail on a separately constructed original portable owner.

    Root must supply a new journal and the original source-bound action/config/
    observe implementations. Their account allowance, quota, locks, nine-hour
    gate, exact public checks and deadlines remain active for every call.
    """
    require(admission["status"] == "actual_runtime_only_original_owner_failure_admitted",
            "Actual isolated runtime-failure admission required")
    require(owner.pin == CANONICAL, "Unchanged canonical phase implementation required")
    require(0 <= time.time() - admission["admitted_at"] <= 180,
            "Fresh exclusive-boundary admission required before continuation")
    require(Path(owner.journal).resolve() != Path(admission["original_journal"]),
            "Continuation must own a distinct journal")
    require(digest(owner.decision) == admission["evaluation"]["decision_sha256"],
            "Original runtime/storage decision must remain unchanged")
    require(
        not any(p.name not in ("owner-intent.json", "launch-intent.json")
                for p in Path(owner.journal).glob("*-intent.json"))
        and not any(Path(owner.journal).glob("*-result.json")),
        "Fresh continuation phase journal required",
    )
    once(Path(owner.journal) / "runtime-continuation-intent.json", admission)
    owner.qualification = admission["qualification"]
    owner.remaining_public = 3  # Rejected roles retain their reserved allowance; no unproven credit.
    roles = admission["evaluation"]["eligible"]
    entries, outcomes = {}, {}
    for arm in roles:
        owner.action("compile", arm)
        owner.action("push", arm)
        while True:
            result = owner.action("status", arm)
            require(result.get("status") not in {"ERROR", "FAILED", "CANCELLED", "CANCEL_ACKNOWLEDGED"},
                    "Public-arm failure: preserve exact original intent; no retry")
            if result.get("status") == "COMPLETE":
                owner.remaining_public -= 1
                break
            owner.sleep()
        entries[arm] = owner.action("submit", arm)
        once(Path(owner.journal) / f"actual-entry-{arm}.json", entries[arm])
    while len(outcomes) < len(roles):
        for arm in roles:
            if arm in outcomes:
                continue
            value = owner.action("outcome", arm)
            if value["outcome"] in ("scored", "failed"):
                outcomes[arm] = value
                once(Path(owner.journal) / f"actual-hidden-outcome-{arm}.json", value)
        if len(outcomes) < len(roles):
            owner.sleep(finish_deadline)
    result = {"entries": entries, "outcomes": outcomes, "evaluation": admission["evaluation"],
              "new_qualification_or_dataset": False, "rejected_roles_not_submitted": True}
    once(Path(owner.journal) / "complete.json", result)
    return result


def derive_deployment_source(source):
    """Three exact operational substitutions for the original portable adapter.

    Its caller still freezes/binds the source and supplies the original HTTP
    path/config adaptations. Inject `runtime_admission` and `continue_runtime`
    into this module's namespace before its explicit run(). No import runs work.
    """
    changes = (
        ('journal = ROOT / "artifacts" / (HERE.name + "-deployment")',
         'journal = ROOT / "artifacts" / ' + repr(JOURNAL_NAME)),
        ('"original_owner": stop["pid"],',
         '"original_owner": runtime_admission["original_pid"],\n'
         '            "runtime_continuation_admission": runtime_admission,\n'
         '            "historical_prephase_owner": stop["pid"],'),
        ("PortableDeployment().run()",
         "continue_runtime(PortableDeployment(), runtime_admission,\n"
         "                         once=p.once, finish_deadline=original.FINISH_END)"),
    )
    for old, new in changes:
        require(source.count(old) == 1, "Frozen portable deployment anchor changed")
        source = source.replace(old, new)
    compile(source, "explicit_runtime_only_deployment_continuation", "exec")
    return source
