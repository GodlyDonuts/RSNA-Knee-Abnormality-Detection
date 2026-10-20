"""HTTP cache lineage and exclusive operational handoff; no model changes."""
import hashlib
import json
import re
import types
from pathlib import Path

PRIOR = "88dfbf5dcab62e820ad746d1d482c0b8c5471642"
PRIOR_DIR = "initiative/research/predictive-v4-owner-handoff-20261020-v1"
PRIOR_SHA = {
    "contract.py": "b22abf69e8f07c4394f4fd37fddbaf3745b1cac7192c525d59cad4c5bc2b031d",
    "launch.py": "51e3e05cd8299110b1353cbf2c6216b16be40cdc19d3b67433402a1620dd35f4",
    "remote_adapter.py": "b543392ad9380b5ee06849f12bc80d3baf3a6546d8123efcfef57c193dc7763f",
}
CACHE = "initiative/research/predictive-cache-http-recovery-20261020-v1"
ROOTS = {r: f"/tmp/rsna-predictive-full-{r}-http-recovery-20261020-v1" for r in ("raw", "context")}
OLD_ROOTS = {r: f"/tmp/rsna-predictive-full-{r}-capacity-recovery-20261020-v1" for r in ROOTS}
OLD_SOURCE = "370f2c598f4a25d92fc4764c899a1632ff50b675"
PREDECESSORS = {"raw": "980743", "context": "980744"}
INTENTS = {
    "raw": "2c301386f28e3a5de96886dbbfc031317e73c324c3d58218e3e5360c9debc656",
    "context": "211870c05293866180e5c8e82f08f48a23fe6310e012362f5ef769b647247036",
}
PROOFS = ("capacity_decision_sha256", "mixed_prefix_admission_sha256",
          "http_recovery_decision_sha256", "source_http_sha256")
prior = None


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def bind(source):
    """Install the exact preceding operational contract, locally or in a capsule."""
    global prior, PARENT, SCIENCE, OLD, FIT, ADMIT_SHA, REMOTE_SHA
    require(sha(source.encode()) == PRIOR_SHA["contract.py"], "Prior operational contract changed")
    prior = types.ModuleType("frozen_88_contract")
    exec(compile(source, "frozen_88_contract.py", "exec"), prior.__dict__)
    PARENT, SCIENCE, OLD, FIT = prior.PARENT, prior.SCIENCE, prior.OLD, prior.FIT
    ADMIT_SHA, REMOTE_SHA = prior.ADMIT_SHA, prior.REMOTE_SHA


def replace_once(source, old, new):
    require(source.count(old) == 1, "Frozen operational anchor changed")
    return source.replace(old, new)


def handoff(value, role):
    require(value["status"] == "actual_idle_v4_cache_dependent_owner_stopped_before_phase"
            and value["role"] == role
            and value["pid"] == {"fitting": 87673, "deployment": 87674}[role]
            and value["source_git"] == PRIOR
            and value["phase_intents"] == [] and value["children"] == []
            and value["owner_process_absent_after_stop"] is True
            and value["remote_admission_or_dispatch_exists"] is False
            and value["remote_jobs_cancelled"] == []
            and re.fullmatch("[0-9a-f]{64}", value["remote_read_only_snapshot_sha256"]),
            "Exact idle87673/87674 handoff required")
    return value


def local_handoff(value, role, repo, process_command):
    handoff(value, role)
    require(not process_command(value["pid"]).strip(), "Predecessor still exists")
    journal = repo / "artifacts" / (Path(PRIOR_DIR).name + ("-deployment" if role == "deployment" else ""))
    if role == "fitting":
        journal /= "finite-owner"
    allowed = {"owner-intent.json", "launch-intent.json", "launcher.json", "stdout", "stderr",
               "frozen_phase_launcher.py", "frozen-phase-source.json"}
    require(journal.is_dir() and {p.name for p in journal.iterdir()} <= allowed,
            "Prior owner advanced; reconcile its original phases")


def observer_handoff(value, repo, process_command):
    require(value["status"] == "actual_idle_training_observer_stopped_before_dispatch"
            and value["pid"] == 87804 and value["source_git"] == PRIOR
            and value["children"] == [] and value["process_absent_after_stop"] is True
            and value["remote_allocations"] == 0 and value["remote_mutations"] == 0
            and value["actual_training_updates_observed"] == 0,
            "Exact idle87804 observer stop required")
    require(not process_command(87804).strip(), "Prior observer remains live")
    journal = repo / "artifacts/predictive-training-observer-20261020-v1"
    require(journal.is_dir() and {p.name for p in journal.iterdir()} <=
            {"observer-intent.json", "launcher.json", "stdout", "stderr"},
            "Prior observer has observation/terminal evidence; reconcile it")


def chain(value):
    require(set(value) == set(PREDECESSORS), "Both immediate predecessors required")
    ancestors = None
    for role, job in PREDECESSORS.items():
        item, terminal = value[role], value[role]["actual_terminal"]
        require(str(item["job"]) == job and item["root"] == OLD_ROOTS[role]
                and item["source_git"] == OLD_SOURCE and item["intent_sha256"] == INTENTS[role]
                and str(terminal["job_id"]) == job and terminal["state"] == "FAILED"
                and terminal["exit_code"] == "1:0" and terminal["live_queue_empty"] is True
                and re.fullmatch("[0-9a-f]{64}", terminal["sha256"]),
                "Immediate HTTP-failure lineage differs")
        old = item["ancestor_chain"]
        prior.chain(old)
        require(ancestors is None or old == ancestors, "Raw/context ancestor chains differ")
        ancestors = old
    return sha(canonical(value))


def proofs(*values):
    for key in PROOFS:
        require(all(value[key] == values[0][key] for value in values)
                and re.fullmatch("[0-9a-f]{64}", values[0][key]), "HTTP admission proof differs: " + key)
    return {key: values[0][key] for key in PROOFS}


def newer_paths(source):
    for role in ROOTS:
        require(OLD_ROOTS[role] in source, "Exact capacity root missing")
        source = source.replace(OLD_ROOTS[role], ROOTS[role])
    return source


def derived_admission(source):
    return newer_paths(prior.derived_admission(source))


def derived_remote(source):
    return newer_paths(prior.derived_remote(source))


def derived_placement(source):
    return prior.derived_placement(source)


def derived_deployment_owner(source):
    return replace_once(prior.derived_deployment_owner(source), prior.CACHE, CACHE)


def deployment_funding(value):
    prior.deployment_funding(value)
