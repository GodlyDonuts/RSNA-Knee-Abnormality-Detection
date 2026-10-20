"""One prephase local fitting continuation; original remote operations stay frozen."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ORIGINAL = ROOT / "initiative/research/predictive-http-owner-handoff-20261020-v1"
JOURNAL = ROOT / "artifacts" / ORIGINAL.name / "finite-owner"
OWN_JOURNAL = ROOT / "artifacts" / HERE.name
OPERATIONAL = "2b834507e19c405971228e1bc405462cd370f560"
POLICY_SOURCE = "15863fa15803b84c26a4691a723c7049ee4ea8e3"
OLD_PID = 91865
PREPHASE = {"owner-intent.json", "launcher.json", "stdout", "stderr"}
FILES = ("launch.py", "fitting.py")
PATHS = {
    "evc33": {
        "/tmp/rsna-predictive-full-fitting-20261020-v2/full-admission",
        "/tmp/rsna-predictive-full-fitting-20261020-v2/dispatch",
        "/tmp/rsna-predictive-full-B-20261020-v2",
        "/tmp/rsna-predictive-full-C-20261020-v2",
        "/tmp/rsna-predictive-C0-return-20261020-v1",
        "/tmp/rsna-predictive-final-deployment-20261020-v1-portable-C0",
    },
    "evc27": {
        "/tmp/rsna-predictive-c0-portable-placement-20261020-v1",
        "/tmp/rsna-predictive-full-C0-20261020-v2",
    },
}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def committed(pin, path):
    require(len(pin) == 40 and all(c in "0123456789abcdef" for c in pin), "Exact source pin")
    return subprocess.check_output(
        ["git", "show", pin + ":" + path.relative_to(ROOT).as_posix()], cwd=ROOT,
    )


def module(source, name, path):
    value = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    value.__file__ = str(path)
    sys.modules[name] = value
    exec(compile(source, str(path), "exec"), value.__dict__)
    return value


def process_command(pid):
    return subprocess.run(
        ["ps", "-p", str(pid), "-o", "command="], capture_output=True, text=True,
    ).stdout.strip()


def validate_handoff(handoff, owner, files, live_command, now):
    require(handoff["status"] == "actual_idle_91865_stopped_before_any_phase"
            and handoff["pid"] == OLD_PID and handoff["source_git"] == OPERATIONAL
            and handoff["children_before_stop"] == [] and handoff["children_after_stop"] == []
            and handoff["owner_process_absent_after_stop"] is True
            and handoff["phase_intents"] == [] and handoff["remote_jobs_cancelled"] == [],
            "Exact observed idle-owner handoff required")
    require(owner["pid"] == OLD_PID and owner["source_git"] == OPERATIONAL,
            "Original immutable owner intent required")
    require(not live_command and set(files) == PREPHASE,
            "Original fitting process or phase exists; reconcile read-only")
    age = now - datetime.fromisoformat(handoff["observed_at"]).timestamp()
    require(0 <= age <= 900, "Actual exclusive handoff must be fresh")


def freeze(args):
    own = {name: committed(args.source_git, HERE / name) for name in FILES}
    require(all(raw == (HERE / name).read_bytes() for name, raw in own.items()),
            "Use exact committed continuation source")
    original_intent_bytes = (JOURNAL / "owner-intent.json").read_bytes()
    handoff_bytes = args.handoff.read_bytes()
    handoff = json.loads(handoff_bytes)
    validate_handoff(
        handoff, json.loads(original_intent_bytes), [p.name for p in JOURNAL.iterdir()],
        process_command(OLD_PID), datetime.now(timezone.utc).timestamp(),
    )
    require(handoff["original_owner_intent_sha256"] == sha(original_intent_bytes),
            "Original owner intent changed")
    proof = json.loads(args.exclusive_proof.read_text())
    require(handoff["exclusive_proof_sha256"] == sha(args.exclusive_proof.read_bytes())
            and proof["status"] == "actual_both_nodes_exclusive_prephase_paths_absent"
            and proof["no_mutations"] is True and proof["remote_jobs_cancelled"] == []
            and {n["node"] for n in proof["nodes"]} == {"evc33", "evc27"}
            and all(set(n["paths_exist"]) == PATHS[n["node"]] for n in proof["nodes"])
            and all(not exists for n in proof["nodes"] for exists in n["paths_exist"].values()),
            "Exact original remote paths must remain absent")
    proof_age = datetime.now(timezone.utc).timestamp() - datetime.fromisoformat(
        proof["observed_at"]).timestamp()
    require(0 <= proof_age <= 1800, "Remote exclusive proof must be fresh")
    old_contract = committed(OPERATIONAL, ORIGINAL / "contract.py")
    require(old_contract == (ORIGINAL / "contract.py").read_bytes(), "Frozen original contract")
    prior_contract = sys.modules.get("contract")
    module(old_contract, "contract", ORIGINAL / "contract.py")
    try:
        old_launch = module(
            committed(OPERATIONAL, ORIGINAL / "launch.py"), "resilient_original_launch",
            ORIGINAL / "launch.py",
        )
        old_args = SimpleNamespace(
            mode="fitting", source_git=OPERATIONAL,
            decision=ORIGINAL / "decision.accepted.json",
            handoff=ORIGINAL / "actual-fitting-owner87673-stopped.json",
            deployment_handoff=ORIGINAL / "actual-deployment-owner87674-stopped.json",
            observer_handoff=ORIGINAL / "actual-observer87804-stopped.json",
            socket=args.socket, deadline_utc="2026-10-22T12:00:00+00:00", poll_seconds=120,
        )
        frozen = old_launch.freeze(old_args)
        original_intent = json.loads(original_intent_bytes)
        for field, path in (
            ("decision_sha256", old_args.decision), ("handoff_sha256", old_args.handoff),
            ("deployment_handoff_sha256", old_args.deployment_handoff),
            ("observer_handoff_sha256", old_args.observer_handoff),
        ):
            require(original_intent[field] == sha(path.read_bytes()),
                    "Original decision or historical handoff bytes changed")
    finally:
        if prior_contract is None:
            sys.modules.pop("contract", None)
        else:
            sys.modules["contract"] = prior_contract
    _, _, fit, placement, original, transport, program, payload = frozen
    require(fit.HERE == ORIGINAL and payload["placement_source_git"] == OPERATIONAL,
            "Original audit paths and remote source pins must remain unchanged")
    policy_dir = ROOT / "initiative/research/predictive-deadline-admission-20261020-v1"
    policies = {
        name: module(committed(POLICY_SOURCE, policy_dir / (name + ".py")),
                     "resilient_" + name, policy_dir / (name + ".py"))
        for name in ("policy", "placement_failure")
    }
    runner = module(own["fitting.py"], "resilient_fitting", HERE / "fitting.py")
    instance = fit.Owner(old_args, original, transport, program, payload, JOURNAL)
    return runner, instance, placement, policies, {
        "status": "distinct_prephase_local_orchestration_original_remote_capsule_unchanged",
        "pid": os.getpid(), "source_git": args.source_git,
        "source_sha256": {name: sha(raw) for name, raw in own.items()},
        "remote_operational_source_git": OPERATIONAL, "policy_source_git": POLICY_SOURCE,
        "original_owner_pid": OLD_PID, "original_owner_intent_sha256": sha(original_intent_bytes),
        "actual_handoff_sha256": sha(handoff_bytes),
        "exclusive_proof_sha256": sha(args.exclusive_proof.read_bytes()),
        "original_remote_program_sha256": sha(program.encode()),
        "original_phase_journal": str(JOURNAL.relative_to(ROOT)),
        "original_audit_directory": str(ORIGINAL.relative_to(ROOT)),
        "preservation_deadline_UTC": old_args.deadline_utc,
        "deployment_owner_91866_unchanged": True, "observer_91955_unchanged": True,
        "scientific_recipe_or_remote_scripts_changed": False,
        "automatic_retry": False, "no_new_remote_storage_allocation": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--exclusive-proof", type=Path, required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = parser.parse_args()
    os.umask(0o077)
    runner, owner, placement, policies, intent = freeze(args)
    OWN_JOURNAL.mkdir(mode=0o700, parents=True, exist_ok=True)
    placement.once(OWN_JOURNAL / "controller-intent.json", intent)
    try:
        result = runner.run(
            owner, deadline_policy=policies["policy"], placement_policy=policies["placement_failure"],
            once=placement.once, audit_dir=ORIGINAL,
        )
        placement.once(OWN_JOURNAL / "controller-result.json", result)
    except BaseException as error:
        placement.once(OWN_JOURNAL / "controller-ended.json", {
            "status": "requires_read_only_reconciliation", "error_type": type(error).__name__,
            "automatic_retry": False, "remote_jobs_cancelled": [],
        })
        raise


if __name__ == "__main__":
    main()
