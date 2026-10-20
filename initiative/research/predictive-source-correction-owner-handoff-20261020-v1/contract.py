"""Source-data correction handoff; retain exact model science and explicit ancestry."""

import hashlib
import json
import re
import types
from datetime import datetime, timezone
from pathlib import Path

PRIOR = "2b834507e19c405971228e1bc405462cd370f560"
PRIOR_DIR = "initiative/research/predictive-http-owner-handoff-20261020-v1"
PRIOR_SHA = {
    "contract.py": "f0f6028e77cc082b99bbdecdc04e589b2c24f44d4bb14093334390faa1840f69",
    "remote_adapter.py": "6a6ce23fc214690b285f462c875497b51c0ba2a87e9ded545027424f8a40498a",
}
BUILDER = "88dfbf5dcab62e820ad746d1d482c0b8c5471642"
BUILDER_DIR = "initiative/research/predictive-v4-owner-handoff-20261020-v1"
BUILDER_SHA = {
    "contract.py": "b22abf69e8f07c4394f4fd37fddbaf3745b1cac7192c525d59cad4c5bc2b031d",
    "launch.py": "51e3e05cd8299110b1353cbf2c6216b16be40cdc19d3b67433402a1620dd35f4",
}
RESILIENT = "32858835db7dfd0f701368cdf06c836180fc2c29"
RESILIENT_DIR = "initiative/research/predictive-resilient-owner-20261020-v1"
RESILIENT_SHA = "ebc4dac4b1836c7317fd5f03b9492b4adc7e3b26e19de7c32d77c58cf93c8423"
POLICY = "15863fa15803b84c26a4691a723c7049ee4ea8e3"
CACHE = "initiative/research/predictive-cache-data-recovery-20261020-v1"
ROOTS = {r: f"/tmp/rsna-predictive-full-{r}-data-recovery-20261020-v1"
         for r in ("raw", "context")}
OLD_ROOTS = {r: f"/tmp/rsna-predictive-full-{r}-http-recovery-20261020-v1" for r in ROOTS}
OLD_SOURCE = "d7326951eeb986f26eb21183d61c2b52476a7986"
PREDECESSORS = {"raw": "980960", "context": "980969"}
INTENTS = {
    "raw": "530c1068bd680cab3c0a9d7835b3d6644a51706ac2c19f2327ea8bfb23dd9e58",
    "context": "af7aa20d269f9d04463bfb46629268be5390c47040729d31c3cecaf3fbc9ef3d",
}
BASELINE = "52f995f4568e36bfa5fbbaf69a89af41226b4f5d"
CORRECTED_OBJECT_REVISION = "0b5e410288232f570c69ba4f44723356e6ae6ab7"
PLAN = "68b44c1412ced183db223220f39cd4a6e071d40515df4c7b75b40a96ba6d0f3d"
RECEIPT = "2c88c4f53cb3d2332e8079e1cd353eede5a601172febf0ab0b74ce8161f9fbb4"
HELPER = "dc0fabd9e7ea9b7c046cf6edb3af64b1843e8c2748b673e913424c66f853088a"
CORRECTION_KEYS = (
    "effective_source_policy", "source_correction_plan_sha256", "source_correction_receipt_sha256",
    "source_correction_helper_sha256", "source_correction_regression_sha256",
    "effective_source_index_sha256", "original_source_index_sha256", "source_correction_revision",
)
CORRECTIONS = {
    "cfe23f2a97ba3fc07066bfed047c6701d9982a54e4913205a6c30366906a455c":
        (411052, "68efac14c0550118b1eb2ce75601f3b5cdf13a824ca83628ef201e260a4b7a9a", 820652),
    "c057554899ba81249d280e880226bf28b089b1e33235e3cfd101bfc36a399d7a":
        (174560, "ab491d18a6b3fae57a014c27679f32afc526e4d336373e4354f81b8a4b7de2d4", 347616),
}
PREPHASE = {"owner-intent.json", "launch-intent.json", "launcher.json", "stdout", "stderr",
            "frozen_phase_launcher.py", "frozen-phase-source.json"}
prior = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def hex64(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def bind(http_source, builder_source):
    global prior, PARENT, SCIENCE, OLD, FIT, ADMIT_SHA, REMOTE_SHA
    require(sha(http_source.encode()) == PRIOR_SHA["contract.py"]
            and sha(builder_source.encode()) == BUILDER_SHA["contract.py"],
            "Exact preceding operational contracts required")
    prior = types.ModuleType("frozen_source_correction_prior")
    exec(compile(http_source, "frozen_2b_contract.py", "exec"), prior.__dict__)
    prior.bind(builder_source)
    PARENT, SCIENCE, OLD, FIT = prior.PARENT, prior.SCIENCE, prior.OLD, prior.FIT
    ADMIT_SHA, REMOTE_SHA = prior.ADMIT_SHA, prior.REMOTE_SHA


def replace_once(source, old, new):
    require(source.count(old) == 1, "Frozen operational source anchor changed")
    return source.replace(old, new)


def fresh(value, now=None):
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    observed = datetime.fromisoformat(value["observed_at"])
    require(observed.utcoffset() is not None and 0 <= now - observed.timestamp() <= 1800,
            "Fresh timezone-aware exclusive handoff required")


def handoff(value, role):
    pid, source = (96074, RESILIENT) if role == "fitting" else (91866, PRIOR)
    require(value["status"] == "actual_idle_source_correction_owner_stopped_before_phase"
            and value["role"] == role and value["pid"] == pid and value["source_git"] == source
            and value["phase_intents"] == [] and value["children"] == []
            and value["owner_process_absent_after_stop"] is True
            and value["remote_admission_or_dispatch_exists"] is False
            and value["remote_jobs_cancelled"] == []
            and hex64(value["remote_read_only_snapshot_sha256"])
            and hex64(value["original_intent_sha256"]),
            "Exact idle fitting96074/deployment91866 handoff required")
    return value


def local_handoff(value, role, repo, process_command):
    handoff(value, role)
    fresh(value)
    require(not process_command(value["pid"]).strip(), "Predecessor still exists")
    old = repo / "artifacts" / Path(PRIOR_DIR).name
    if role == "fitting":
        controller = repo / "artifacts" / Path(RESILIENT_DIR).name
        require({p.name for p in controller.iterdir()} <=
                {"controller-intent.json", "launcher.json", "stdout", "stderr",
                 "account-read-followup.json"}, "Resilient controller advanced")
        raw = (controller / "controller-intent.json").read_bytes()
        intent = json.loads(raw)
        require(sha(raw) == value["original_intent_sha256"] and intent["pid"] == 96074
                and intent["source_git"] == RESILIENT
                and intent["remote_operational_source_git"] == PRIOR, "Original controller changed")
        journal = old / "finite-owner"
        require(sha((journal / "owner-intent.json").read_bytes()) ==
                value["original_phase_owner_intent_sha256"] ==
                intent["original_owner_intent_sha256"], "Original phase-owner intent changed")
    else:
        journal = old.with_name(old.name + "-deployment")
        raw = (journal / "owner-intent.json").read_bytes()
        intent = json.loads(raw)
        require(sha(raw) == value["original_intent_sha256"] and intent["pid"] == 91866
                and intent["placement_source_git"] == PRIOR, "Original deployment intent changed")
    require({p.name for p in journal.iterdir()} <= PREPHASE,
            "Previous owner has phase evidence; reconcile read-only")


def observer_handoff(value, repo, process_command):
    require(value["status"] == "actual_idle_training_observer_stopped_before_dispatch"
            and value["pid"] == 91955 and value["source_git"] == PRIOR
            and value["children"] == [] and value["process_absent_after_stop"] is True
            and value["remote_allocations"] == 0 and value["remote_mutations"] == 0
            and value["actual_training_updates_observed"] == 0,
            "Exact idle91955 observer stop required")
    fresh(value)
    require(not process_command(91955).strip(), "Prior observer remains live")
    journal = repo / "artifacts" / Path(PRIOR_DIR).name
    require({p.name for p in journal.iterdir()} <=
            {"finite-owner", "observer-intent.json", "launcher.json", "stdout", "stderr"},
            "Previous observer has results requiring reconciliation")
    raw = (journal / "observer-intent.json").read_bytes()
    intent = json.loads(raw)
    require(sha(raw) == value["original_intent_sha256"] and intent["pid"] == 91955
            and intent["observer_source_git"] == PRIOR, "Original observer intent changed")


def chain(value):
    require(set(value) == set(PREDECESSORS), "Both immediate predecessors required")
    ancestors = None
    for role, job in PREDECESSORS.items():
        item, terminal = value[role], value[role]["actual_terminal"]
        require(str(item["job"]) == job and item["root"] == OLD_ROOTS[role]
                and item["source_git"] == OLD_SOURCE and item["intent_sha256"] == INTENTS[role]
                and str(terminal["job_id"]) == job and terminal["state"] == "FAILED"
                and terminal["exit_code"] == "1:0" and terminal["live_queue_empty"] is True
                and hex64(terminal["sha256"]), "Exact source-decode failure lineage required")
        previous = item["ancestor_chain"]
        prior.chain(previous)
        require(ancestors is None or previous == ancestors, "Raw/context ancestors differ")
        ancestors = previous
    return sha(canonical(value))


def validate_correction(value, expected=None):
    require(value["effective_source_policy"] == correction_policy()["policy"]
            and value["source_correction_plan_sha256"] == PLAN
            and value["source_correction_receipt_sha256"] == RECEIPT
            and value["source_correction_helper_sha256"] == HELPER
            and value["source_correction_revision"] == CORRECTED_OBJECT_REVISION
            and all(hex64(value[key]) for key in (
                "source_correction_regression_sha256", "effective_source_index_sha256",
                "original_source_index_sha256")),
            "Exact baseline-plus-two-file correction lineage required")
    if expected is not None:
        require(all(value[key] == val for key, val in expected.items()),
                "Actual correction differs from the separately accepted source proof")
    return {key: value[key] for key in CORRECTION_KEYS}


def proofs(*values, expected=None):
    require(len(values) >= 2, "Independent raw and context proof records required")
    inherited = prior.proofs(*values)
    correction = validate_correction(values[0], expected)
    require(all(validate_correction(value, expected) == correction for value in values[1:]),
            "Raw/context correction proof records differ")
    return {**inherited, **correction}


def newer_paths(source):
    for role in ROOTS:
        require(OLD_ROOTS[role] in source, "Exact prior cache root missing")
        source = source.replace(OLD_ROOTS[role], ROOTS[role])
    return source


def derived_admission(source):
    source = newer_paths(prior.derived_admission(source))
    source = replace_once(source, '    rows = json.loads(index_path.read_text())',
        '    rows = json.loads(index_path.read_text())\n'
        '    validate_corrected_context_rows(rows, _cache_successor)')
    return replace_once(source, '    write_once(out / "actual.json", result)',
        '    result["cache_successor_control"] = _cache_successor\n'
        '    result["input_admission_sha256"] = {\n'
        '        arm: ref(out / f"input-{arm}.json")["sha256"] for arm in arms}\n'
        '    write_once(out / "actual.json", result)')


def derived_remote(source):
    return newer_paths(prior.derived_remote(source))


def derived_placement(source):
    return prior.derived_placement(source)


def derived_deployment_owner(source):
    return replace_once(prior.derived_deployment_owner(source), prior.CACHE, CACHE)


def deployment_funding(value):
    prior.deployment_funding(value)


def derived_portable_remote(source):
    source = replace_once(source, '    p.once(out / "input-C0.json", admission)',
        '    admission["cache_successor_control"] = source["cache_successor_control"]\n'
        '    p.once(out / "input-C0.json", admission)')
    return replace_once(source, '    p.once(root / "portable-accepted.json", result)',
        '    result["cache_successor_control"] = source["cache_successor_control"]\n'
        '    result["input_admission_sha256"] = {"C0": refs["input_admission"]["sha256"]}\n'
        '    p.once(root / "portable-accepted.json", result)')


def correction_policy():
    """Sanitized policy, never a claim that every source file uses revision0b5e."""
    return {
        "policy": "baseline_plus_exact_authoritative_two_file_patch",
        "baseline_source_revision": BASELINE,
        "corrected_object_revision": CORRECTED_OBJECT_REVISION,
        "source_correction_plan_sha256": PLAN,
        "source_correction_receipt_sha256": RECEIPT,
        "changed_source_files": [
            {"before_sha256": before, "before_bytes": values[0],
             "after_sha256": values[1], "after_bytes": values[2]}
            for before, values in sorted(CORRECTIONS.items())
        ],
        "unchanged_source_members_use_baseline": True,
        "decoder_or_targets_changed": False,
        "source_files_skipped": 0,
    }


def validate_corrected_context_rows(rows, control):
    require(len(rows) == 3887, "Complete unchanged fitting roster required")
    validate_correction(control, control["accepted_source_correction_proofs"])
    observed = set()
    for row in rows:
        provenance = row["provenance"]
        require(provenance["original_source_revision"] == BASELINE,
                "Baseline is retained for all source rows")
        correction = provenance.get("source_correction")
        if correction is None:
            continue
        before = correction["before_sha256"]
        require(before in CORRECTIONS and before not in observed,
                "Exactly the two distinct admitted source corrections required")
        old_bytes, after, new_bytes = CORRECTIONS[before]
        expected = dict(
            effective_source_policy=correction_policy()["policy"],
            original_revision=BASELINE, corrected_revision=CORRECTED_OBJECT_REVISION,
            plan_sha256=PLAN, receipt_sha256=RECEIPT, before_sha256=before,
            before_bytes=old_bytes, after_sha256=after, after_bytes=new_bytes,
        )
        require(correction == expected, "Corrected context must bind exact authoritative bytes")
        observed.add(before)
    require(observed == set(CORRECTIONS), "Both corrected studies must be retained in full fitting")
    return True


def final_source_lineage(value, directory):
    """Fail before any account operation unless final inputs match correction proofs."""
    directory = Path(directory)
    proofs_by_arm = {}
    for name, arms, status in (
        ("admission-BC", ("B", "C"), "actual_full3887_predictive_CPU_admission_complete_no_GPU_dispatched"),
        ("contexts-C0", ("C0",), "actual_C0_context_transfer_and_CPU_admission_accepted"),
    ):
        path = directory / ("actual-owner-" + name + ".json")
        raw = path.read_bytes()
        admission = json.loads(raw)
        require(admission["status"] == status
                and set(admission["input_admission_sha256"]) == set(arms),
                "Actual per-arm corrected input admissions required")
        control = admission["cache_successor_control"]
        require(control["source_correction_policy"] == correction_policy(),
                "Effective source policy differs")
        validate_correction(control, control["accepted_source_correction_proofs"])
        for arm in arms:
            final, publication = value["finals"][arm], value["publications"][arm]
            require(final["source_revision"] == BASELINE
                    and final["input_admission_sha256"] == admission["input_admission_sha256"][arm]
                    and hex64(final["input_admission_sha256"])
                    and publication["GPU_job"] == final["GPU_job"],
                    "Final export must bind the actual corrected-input admission")
            proofs_by_arm[arm] = {
                "input_admission_sha256": final["input_admission_sha256"],
                "original_admission_receipt_sha256": sha(raw),
                "cache_successor_control": control,
            }
    value["effective_source_lineage"] = {
        "effective_source_policy": correction_policy(), "per_arm": proofs_by_arm,
        "unchanged_worker_export_binds_this_via_input_admission_sha256": True,
        "no_whole_revision_substitution": True,
    }
    return value
