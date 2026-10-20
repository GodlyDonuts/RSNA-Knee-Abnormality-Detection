"""Distinct C100 admission; reuse actual C0 inputs without copying MRI or restarting it."""

import copy
import hashlib
import json
import math
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

GIB, MIB = 2**30, 2**20
BASE = "44cc559e947560a635c34fee7b1f2a3b7be3a241"
PRIMARY_SOURCE = "2a224ae05fc7c2fc78d4d508b6407344ac523a1c"
PRIMARY_OWNER_PID = 32377
PRIMARY_AUDIT_REL = "initiative/research/predictive-evc27-deadline-placement-20261020-v1"
CONTINUATION_REL = "artifacts/predictive-evc27-startup-continuation-20261020-v1/continuation-intent.json"
PRIMARY_INTENT_SHA = "b310a8cf6267ea4ce1b40b4efef27ad375b9bc869dfaee435b9c86ed0f40ebd0"
CONTINUATION_SOURCE = "7dcd87e3749998ee7d8f84c08f8f7f118072bf9d"
CONTINUATION_PID = 33107
CONTINUATION_INTENT_SHA = "360bc552517cdc0a886ae36e4efa55100327eebf0a4204b233aa20159c0b7a71"
HTTP499_PROOFS = {
    "http499_recovery_decision_sha256": "6c655a431a17f2e4052913216836816a3d88fbf8bb018b11b2403e120bf35d88",
    "previous_source_http_sha256": "ea7bf0f44af60ec4b0876e85af5cc1536d15cad671ea45d4ac548adccca0e28c",
    "source_http_sha256": "b15e3d56fb536fbb42ea9427ed57fb6c25db052bbcefd8e74139bd39c01c7fdb",
}
CORRECTION_POLICY_SHA = "7264d3840a343a824d2f5bf174c9d96309fdb762a051710cbd772cf64d503129"
CORRECTION_REGRESSION_SHA = "1f641ed02e881cd572bb00afcb40df04b1644285cea827772779095cc964eea1"
PILOT_SOURCE = "068a14a75e4d2b3bd83dda7783528956f7a2ca0f"
PILOT_BINDING = "e11c7c7fe5c1fd7cae87df313fb5d42be7128b7906f8d716a486b0857bbdd37a"
INITIAL = "dbf69ff87261f7a0657c866ef8e41db8b125a62c6245d80017945c634e60158b"
ROSTER = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"
C0 = Path("/tmp/rsna-predictive-evc27-deadline-placement-20261020-v1")
C0_RUN = Path("/tmp/rsna-predictive-full-C0-20261020-v2")
PILOT = Path("/tmp/rsna-predictive-density100-pilot-20261020-v1")
ROOT = Path("/tmp/rsna-predictive-density100-full-20261020-v1")
RUN = Path("/tmp/rsna-predictive-full-C100-20261020-v1")
PYTHON = "/tmp/rsna-predictive-portable-recovery-20261020-v1/runtime/bin/python"
OVERLAY = "/tmp/rsna-predictive-portable-recovery-20261020-v1/overlay"
FIT_CAP, CONTROL_CAP = 12 * GIB, 16 * MIB
# No paid credit for these independent scopes without separate original proof.
G100_RESERVED_BYTES = 6 * GIB + 4 * MIB
DESCRIPTOR_RESERVED_BYTES = MIB
CONTRAST_RESERVED_BYTES = 7 * GIB + 4 * MIB
PRIMARY_OTHER_FITS_RESERVED_BYTES = 24 * GIB
PRIMARY_ADDITIONAL_CONTROLS_RESERVED_BYTES = 64 * MIB
SHARED_BYTES, SHARED_FILES = 123 * MIB, 376
LAST_DISPATCH = datetime(2026, 10, 21, 20, 0, tzinfo=timezone.utc)


def primary_control():
    """Bind the unchanged original capsule and its separately recorded continuation."""
    return {"source_git": PRIMARY_SOURCE, "original_owner_pid": PRIMARY_OWNER_PID,
            "audit_directory": PRIMARY_AUDIT_REL,
            "original_intent_sha256": PRIMARY_INTENT_SHA,
            "continuation_source_git": CONTINUATION_SOURCE,
            "continuation_pid": CONTINUATION_PID,
            "continuation_intent_sha256": CONTINUATION_INTENT_SHA}


def validate_primary_ownership(original_raw, continuation_raw):
    assert hashlib.sha256(original_raw).hexdigest() == PRIMARY_INTENT_SHA, "Original primary intent changed"
    assert hashlib.sha256(continuation_raw).hexdigest() == CONTINUATION_INTENT_SHA, "Actual continuation intent changed"
    owner, resumed = json.loads(original_raw), json.loads(continuation_raw)
    assert owner["source_git"] == PRIMARY_SOURCE and owner["pid"] == PRIMARY_OWNER_PID
    assert resumed["source_git"] == CONTINUATION_SOURCE and resumed["pid"] == CONTINUATION_PID
    assert resumed["original_source_git"] == PRIMARY_SOURCE and resumed["original_owner_pid"] == PRIMARY_OWNER_PID
    assert resumed["original_startup_sha256"]["owner-intent.json"] == PRIMARY_INTENT_SHA
    assert resumed["original_owner_intent_retained"] is True and resumed["only_startup_marker_renamed"] is True
    assert resumed["automatic_retry"] is False
    return primary_control()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def once(path, value):
    path = Path(path)
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as handle:
        handle.write(canonical(value) + b"\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(pending, path)
    finally:
        pending.unlink()


def signature(path):
    path = Path(path)
    s = path.stat()
    require(path.is_absolute() and path.is_file() and not path.is_symlink()
            and s.st_uid == os.getuid() and not s.st_mode & 0o077,
            "Original owner-only regular file required")
    return [s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def read_ref(ref):
    path = Path(ref["path"])
    before = signature(path)
    raw = path.read_bytes()
    require(sha(raw) == ref["sha256"] and signature(path) == before, "Original reference changed")
    return raw


def reference(path):
    path = Path(path)
    before = signature(path)
    raw = path.read_bytes()
    require(signature(path) == before, "Small original file changed")
    return {"path": str(path), "sha256": sha(raw)}


def recipe():
    return dict(arm="C", epochs=6, effective_batch=8, encoder_lr=3e-5, head_lr=3e-5,
                weight_decay=0.02, warmup_fraction=0.05, end_lr_fraction=0.1,
                density_weight=100.0, head_only_updates=3, clip=1.0,
                seed=20261020, schedule_updates=2916)


def validate_pilot(pilot, scheduler):
    require(
        pilot["status"] == "actual_C100_full_stop_resume_engineering_complete"
        and pilot["GPU_job"] == scheduler["GPU_job"] == "981620"
        and pilot["source_git"] == PILOT_SOURCE
        and pilot["binding_sha256"] == PILOT_BINDING
        and pilot["branch"] == "C100"
        and pilot["full_stop_resume_complete_state_and_prediction_bytes_equal"] is True
        and pilot["automatic_full_training_promotion"] is False
        and pilot["clinical_or_held_out_accuracy_claim"] is False
        and scheduler["state"] == "COMPLETED" and scheduler["exit_code"] == "0:0"
        and scheduler["node"] == "evc27",
        "Actual original C100 recovery and scheduler terminal required",
    )
    history = pilot["real_optimizer_history"]
    require(len(history) == 24 and [r["step"] for r in history] == list(range(1, 25))
            and [r["studies"] for r in history] == [8, 4] * 12,
            "Actual complete pilot update history required")
    for row in history:
        require(all(math.isfinite(row[k]) for k in (
            "loss", "exam_loss", "density_loss", "encoder_gradient_norm",
            "head_gradient_norm", "gradient_norm",
        )), "Pilot contains nonfinite arithmetic")
    require(pilot["terminal"]["density_weight"] == 100
            and pilot["terminal"]["steps"] == 24, "Different pilot recipe")


def validate_decision(decision, pilot, pilot_sha, source_git, code_sha):
    validate_pilot(pilot, decision["pilot_scheduler"])
    require(
        decision["status"] == "accepted_separate_full_C100_after_actual_pilot_review"
        and decision["pilot_acceptance_sha256"] == pilot_sha
        and decision["source_git"] == source_git
        and decision["code_sha256"] == code_sha
        and decision["recipe"] == recipe()
        and decision["pilot_warnings_reviewed"] == pilot["engineering_warnings"]
        and isinstance(decision["agent_reasoning"], str) and len(decision["agent_reasoning"]) >= 80
        and decision["observed_C_C0_gradient_risk_reviewed"] is True
        and decision["primary_B_C_C0_unchanged"] is True
        and decision["primary_Kaggle_reserve_seconds"] == 3360
        and decision["engineering_predictions_used_for_selection"] is False
        and decision["automatic_recipe_retuning"] is False,
        "Actual, separately recorded agent engineering decision required",
    )


def validate_reuse(portable, old_binding, old_binding_sha, original_input, index):
    require(
        portable["status"] == "actual_C0_context_transfer_and_CPU_admission_accepted"
        and portable["science_git"] == old_binding["source_git"] == BASE
        and portable["admitted_arms"] == ["C", "B", "C0"]
        and set(portable["binding_sha256"]) == {"B", "C", "C0"}
        and portable["binding_sha256"]["C0"] == old_binding_sha
        and portable["initial_shared_tensor_sha256"]
        == old_binding["expected_initial_shared_tensor_sha256"] == INITIAL
        and old_binding["arm"] == "C0" and old_binding["node"] == "evc27",
        "Exact completed original C0 placement required, not a partial copy",
    )
    require(
        original_input["status"] == "actual_complete3887_predictive_fitting_inputs_accepted"
        and original_input["source_git"] == BASE
        and original_input["code_sha256"] == old_binding["code_sha256"]
        and original_input["context_index_sha256"] == old_binding["context_index"]["sha256"]
        == portable["destination_context_index_sha256"]
        and original_input["all_context_original_hashes_and_stats_verified"] is True
        and original_input["fitting_order_sha256"] == ROSTER
        and original_input["original_context_count"] == len(index) == 3887,
        "Complete original source/roster/input proof required",
    )
    uids = [r["uid"] for r in index]
    require(uids == sorted(set(uids)) and sha(canonical(uids)) == ROSTER,
            "Do not change fitting membership or order")
    require(sum(r["proof"]["bytes"] for r in index) == portable["context_bytes"],
            "Original complete context bytes differ")
    control = old_binding["cache_successor_control"]
    require(portable["placement_control_source_git"]
            == old_binding["placement_control_source_git"] == PRIMARY_SOURCE
            and portable["cache_successor_control"] == original_input["cache_successor_control"]
            == control and control["operational_control_git"] == PRIMARY_SOURCE
            and sha(canonical(control["source_correction_policy"])) == CORRECTION_POLICY_SHA
            and control["source_correction_regression_sha256"] == CORRECTION_REGRESSION_SHA
            and control["accepted_source_correction_proofs"]["source_correction_regression_sha256"]
            == CORRECTION_REGRESSION_SHA
            and control["accepted_http499_proofs"] == HTTP499_PROOFS
            and all(control.get(key) == value for key, value in HTTP499_PROOFS.items()),
            "Exact current source-corrected HTTP499 C0 admission lineage required")
    # main() verifies the retained input's actual bytes through read_ref().
    # Do not replace that proof with a serialization-dependent JSON hash.
    require(set(portable["input_admission_sha256"]) == {"B", "C", "C0"}
            and portable["input_admission_sha256"]["C0"] == old_binding["input_admission"]["sha256"]
            and all(re.fullmatch("[0-9a-f]{64}", h)
                    for h in portable["input_admission_sha256"].values()),
            "Portable C0 must bind the exact retained corrected-input admission")


def validate_primary_jobs(jobs, arms=None):
    """Keep every actual arm/job/node/path binding when carrying paid-log proof."""
    require(set(jobs) <= {"B", "C", "C0"} and (arms is None or set(jobs) == set(arms)),
            "Exact original primary dispatch arms required")
    identities = set()
    for arm, job in jobs.items():
        identity = job["GPU_job"]
        name = "rsna-predictive-full2-" + arm
        require(isinstance(identity, str) and re.fullmatch(r"\d+", identity)
                and identity not in identities and job["arm"] == arm
                and job["node"] == "evc27"
                and job["source_git"] == BASE and job["job_name"] == name
                and job["run_root"] == f"/tmp/rsna-predictive-full-{arm}-20261020-v2"
                and all(re.fullmatch("[0-9a-f]{64}", job[k]) for k in
                        ("binding_sha256", "script_sha256"))
                and job["stdout"] == f"/home/sa305415/rsna-knee/kaggle/{name}-{identity}.out"
                and job["stderr"] == f"/home/sa305415/rsna-knee/kaggle/{name}-{identity}.err",
                "Exact original job/source/script/binding/log identity required")
        identities.add(identity)
    return jobs


def used(path, cap, *, exclude_contexts=False):
    path = Path(path)
    if not path.exists():
        return 0
    require(not path.is_symlink(), "Owned output directory required")
    total = 0
    for item in path.rglob("*"):
        if exclude_contexts and "contexts" in item.relative_to(path).parts:
            continue
        if any(p.startswith("independent-download-") for p in item.relative_to(path).parts):
            continue  # Disposable verification bytes never retire future obligations.
        if item.is_file():
            try:
                total += signature(item)[1]
            except FileNotFoundError:
                require("publication" in item.parts
                        and any(p.startswith("independent-download-") for p in item.parts),
                        "Only temporary verification downloads may disappear")
    require(total <= cap, "Existing scope exceeded its retained bound")
    return total


def node_budget(free, *, contexts_paid, C0_controls_paid, C0_fit_paid,
                pilot_checkpoints_paid, pilot_controls_paid, own_controls_paid=0, own_fit_paid=0):
    scopes = [
        (96 * GIB, contexts_paid), (64 * MIB, C0_controls_paid),
        (12 * GIB, C0_fit_paid), (6 * GIB, pilot_checkpoints_paid),
        (4 * MIB, pilot_controls_paid), (CONTROL_CAP, own_controls_paid),
        (FIT_CAP, own_fit_paid),
    ]
    require(all(type(paid) is int and 0 <= paid <= cap for cap, paid in scopes),
            "Only actual paid bytes within separately funded caps can retire obligations")
    other = (205676085248 + G100_RESERVED_BYTES + DESCRIPTOR_RESERVED_BYTES + CONTRAST_RESERVED_BYTES
             + PRIMARY_OTHER_FITS_RESERVED_BYTES + PRIMARY_ADDITIONAL_CONTROLS_RESERVED_BYTES
             + sum(cap - paid for cap, paid in scopes[:-1]))
    required = 20 * GIB + other + FIT_CAP - own_fit_paid
    require(type(free) is int and free > required, "Retain every original/pilot/C0 liability")
    return {"free_bytes": free, "required_bytes": required, "remaining_bytes": free - required,
            "other_unpaid_node_liabilities_bytes": other,
            "conservative_G100_reserved_bytes_no_paid_credit": G100_RESERVED_BYTES,
            "descriptor_reserved_bytes_no_paid_credit": DESCRIPTOR_RESERVED_BYTES,
            "contrast_reserved_bytes_no_paid_credit": CONTRAST_RESERVED_BYTES,
            "primary_other_fits_reserved_bytes_no_paid_credit": PRIMARY_OTHER_FITS_RESERVED_BYTES,
            "primary_additional_controls_reserved_bytes_no_paid_credit": PRIMARY_ADDITIONAL_CONTROLS_RESERVED_BYTES}


def fresh_node(portable):
    paid = dict(
        contexts_paid=portable["context_bytes"],
        C0_controls_paid=used(C0, 64 * MIB, exclude_contexts=True),
        C0_fit_paid=used(C0_RUN, 12 * GIB),
        pilot_checkpoints_paid=used(PILOT / "engineering", 6 * GIB),
        pilot_controls_paid=used(PILOT / "code", 4 * MIB)
        + sum(signature(p)[1] for p in PILOT.iterdir() if p.is_file()),
        own_controls_paid=used(ROOT, CONTROL_CAP), own_fit_paid=used(RUN, FIT_CAP),
    )
    return node_budget(shutil.disk_usage("/tmp").free, **paid)


def shared_budget(text, paid_logs=0):
    require(type(paid_logs) is int and 0 <= paid_logs <= 6, "Only original six logs may retire files")
    row = next(r.split() for r in text.splitlines() if r.strip().startswith("/home/sa305415"))
    free_bytes, free_files = (int(row[2]) - int(row[1])) * 1024, int(row[6]) - int(row[5])
    require(free_bytes > 20 * GIB + SHARED_BYTES
            and free_files >= 5000 + SHARED_FILES - paid_logs, "Fresh shared reserve required")
    return {"free_bytes": free_bytes, "free_files": free_files,
            "unpaid_bytes": SHARED_BYTES, "unpaid_files": SHARED_FILES - paid_logs,
            "new_shared_bytes": 0, "new_shared_files": 0}


def main(v):
    os.umask(0o077)
    require(v["primary_control"] == primary_control(), "Exact all27 primary continuation required")
    require(os.uname().nodename == "evc27" and not os.environ.get("SLURM_JOB_GPUS")
            and os.environ.get("CUDA_VISIBLE_DEVICES") == "", "Owning-node CPU-only admission")
    require(datetime.now(timezone.utc) < LAST_DISPATCH, "Full C100 launch window expired")
    require(not ROOT.exists() and not RUN.exists(), "Existing C100 state requires reconciliation")
    require(re.fullmatch("[0-9a-f]{40}", v["source_git"]), "Immutable committed source required")
    portable_ref = reference(C0 / "portable-accepted.json")
    portable = json.loads(read_ref(portable_ref))
    binding_ref = reference(C0 / "full-admission/binding-C0.private.json")
    old = json.loads(read_ref(binding_ref))
    original_input = json.loads(read_ref(old["input_admission"]))
    index = json.loads(read_ref(old["context_index"]))
    validate_reuse(portable, old, binding_ref["sha256"], original_input, index)
    pilot_ref = reference(PILOT / "GPU-accepted.json")
    pilot = json.loads(read_ref(pilot_ref))
    files = v["files"]
    hashes = {n: sha(body.encode()) for n, body in files.items()}
    require(hashes == v["code_sha256"], "Frozen source payload differs")
    validate_decision(v["decision"], pilot, pilot_ref["sha256"], v["source_git"], hashes)
    for name, expected in old["code_sha256"].items():
        raw = read_ref({"path": str(C0 / "code" / name), "sha256": expected})
        if name.startswith("rsna/") or name == "alias_provenance.py":
            require(files[name].encode() == raw, "Original numerical/renderer modules changed")
    for row in index:
        require(signature(row["path"]) == row["proof"]["stat"], "Original cached context changed")
    for name in ("runtime_receipt", "alias_regression", "original_preparation",
                 "license", "context_terminal", "targets", "parent_source"):
        read_ref(old[name])
    require(signature(old["parent_weight"]["path"]) == old["parent_weight"]["stat"],
            "Already verified immutable parent changed")
    require(old["runtime_python"] == PYTHON and old["python_paths"] == [OVERLAY],
            "Exact original admitted runtime required")
    require(sum(len(s.encode()) for s in files.values()) <= 2 * MIB, "Small own-code bound")
    node = fresh_node(portable)
    # Retire only physically charged original job-bound log files. No caller
    # integer, guessed inode, or future file earns a reserve credit.
    import operations
    shared = operations.shared(v)
    ROOT.mkdir(mode=0o700)
    for name, body in files.items():
        p = ROOT / "code" / name
        require(p.resolve().is_relative_to((ROOT / "code").resolve()), "Source path escaped")
        p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with p.open("x") as handle:
            handle.write(body)
    decision = {**v["decision"], "status": "accepted_full_predictive_fitting_recipe",
                "full_C100_agent_decision_sha256": sha(canonical(v["decision"]))}
    once(ROOT / "agent-decision.json", v["decision"])
    once(ROOT / "engineering.json", decision)
    funding = {
        "status": "actual_predictive_full_arm_storage_funded", "arm": "C", "branch": "C100",
        "node": "evc27", "run_root": str(RUN), "CPU_job": os.environ["SLURM_JOB_ID"],
        "arm_cap_bytes": FIT_CAP, "standing_free_reserve_bytes": 20 * GIB,
        "actual_node_free_bytes": node["free_bytes"],
        "other_unpaid_node_liabilities_bytes": node["other_unpaid_node_liabilities_bytes"],
        "shared_free_after_all_liabilities_bytes": shared["free_bytes"] - shared["unpaid_bytes"],
        "shared_files_after_all_liabilities": shared["free_files"] - shared["unpaid_files"],
        "all_other_obligations_retained": True, "no_deletion_credit": True,
        "pilot_scope_retained": True, "new_shared_files": 0,
    }
    once(ROOT / "funding.json", funding)
    admitted = {
        "primary_control": primary_control(),
        **copy.deepcopy(original_input), "source_git": v["source_git"], "code_sha256": hashes,
        "original_C0_input_admission_sha256": old["input_admission"]["sha256"],
        "original_portable_admission_sha256": portable_ref["sha256"],
        "original_C0_binding_sha256": binding_ref["sha256"], "branch": "C100",
        "new_MRI_transferred_bytes": 0, "CPU_job": os.environ["SLURM_JOB_ID"],
    }
    once(ROOT / "input.json", admitted)
    binding = {
        **copy.deepcopy(old), "arm": "C", "branch": "C100", "base_science_git": BASE,
        "source_git": v["source_git"], "code_root": str(ROOT / "code"), "code_sha256": hashes,
        "run_root": str(RUN), "density100_pilot": pilot_ref,
        "density100_pilot_source_git": PILOT_SOURCE,
        "original_portable_admission_sha256": portable_ref["sha256"],
        "original_C0_binding_sha256": binding_ref["sha256"],
        "engineering_decision": reference(ROOT / "engineering.json"),
        "funding": reference(ROOT / "funding.json"), "input_admission": reference(ROOT / "input.json"),
    }
    once(ROOT / "binding.private.json", binding)
    result = {
        "status": "actual_separate_C100_full_inputs_CPU_admitted", "branch": "C100",
        "CPU_job": os.environ["SLURM_JOB_ID"], "source_git": v["source_git"],
        "base_science_git": BASE, "density100_pilot_sha256": pilot_ref["sha256"],
        "binding_sha256": reference(ROOT / "binding.private.json")["sha256"],
        "original_portable_admission_sha256": portable_ref["sha256"],
        "context_files_reused_without_transfer": 3887, "new_shared_files": 0,
        "node_capacity": node, "shared_capacity": shared, "GPU_dispatched": False,
    }
    once(ROOT / "admitted.json", result)
    require(used(ROOT, CONTROL_CAP) < CONTROL_CAP, "Bounded own controls/logs required")
    return result


if __name__ == "__main__":
    import sys
    print(json.dumps(main(json.load(sys.stdin))), flush=True)
