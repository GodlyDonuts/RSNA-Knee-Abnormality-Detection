"""Continue the failed pre-intent observer with its exact original read-only code."""

import argparse
import copy
import hashlib
import json
import os
import subprocess
import sys
import time
import types
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
ORIGINAL = "e9bdd1249fa7cf75f792b3762c2028bd7beb4337"
OLD = "initiative/research/predictive-source-correction-owner-handoff-20261020-v1"
OLD_JOURNAL = ROOT / "artifacts" / Path(OLD).name
FAILURE_SHA = "a70d2656aecc70cf3ffdf5cde172e19dac25fd6d38d48affaef6f0634d915256"
OLD_PID = 11015
DEADLINE = "2026-10-22T12:00:00+00:00"


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def git_file(pin, name):
    require(len(pin) == 40 and set(pin) <= set("0123456789abcdef"), "Exact committed source")
    return subprocess.check_output(["git", "show", pin + ":" + name], cwd=ROOT)


def process_command(pid):
    return subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                          capture_output=True, text=True, check=False).stdout


def predecessor():
    require(not process_command(OLD_PID).strip(), "Original observer still exists")
    require(not (OLD_JOURNAL / "observer-intent.json").exists(),
            "Original observer advanced; reconcile before any continuation")
    local = (OLD_JOURNAL / "observer-failure.json").read_bytes()
    actual = (ROOT / OLD / "actual-observation-failure.json").read_bytes()
    require(local == actual and sha(local) == FAILURE_SHA, "Exact pre-intent failure required")
    failure = json.loads(local)
    require(failure["error_type"] == "FileNotFoundError"
            and failure["remote_mutations"] == 0 and failure["evidence"] == {},
            "Only the original pre-intent startup failure is admitted")
    original = json.loads((OLD_JOURNAL / "launcher.json").read_bytes())
    require(original["PID"] == OLD_PID and original["source_git"] == ORIGINAL,
            "Exact original observer launcher required")
    return {"original_observer_PID": OLD_PID, "original_source_git": ORIGINAL,
            "original_failure_sha256": FAILURE_SHA, "original_observer_intent_absent": True}


def await_owner(*, clock=time.monotonic, sleep=time.sleep, timeout=120):
    require(0 < timeout <= 120, "Bounded startup wait")
    path = OLD_JOURNAL / "finite-owner/owner-intent.json"
    until = clock() + timeout
    while not path.is_file():
        require(clock() < until, "Original fitting owner intent remains absent")
        sleep(min(1, max(0, until - clock())))
    raw = path.read_bytes()
    owner = json.loads(raw)
    require(owner["pid"] == 11013 and owner["source_git"] == ORIGINAL
            and owner["science_git"] == "44cc559e947560a635c34fee7b1f2a3b7be3a241",
            "Exact live fitting owner intent required")
    return sha(raw)


def frozen_observer():
    """Import exact old libraries without their main functions or any remote call."""
    source = {name: git_file(ORIGINAL, OLD + "/" + name)
              for name in ("contract.py", "launch.py", "observe.py", "remote_adapter.py")}
    previous = {name: sys.modules.get(name) for name in ("contract", "launch")}
    try:
        for name in ("contract", "launch"):
            module = types.ModuleType("original_observer_" + name)
            module.__file__ = str(ROOT / OLD / (name + ".py"))
            exec(compile(source[name + ".py"], module.__file__, "exec"), module.__dict__)
            sys.modules[name] = module
        wrapper = types.ModuleType("original_observer_wrapper")
        wrapper.__file__ = str(ROOT / OLD / "observe.py")
        exec(compile(source["observe.py"], wrapper.__file__, "exec"), wrapper.__dict__)
        observer = wrapper.observer()
    finally:
        for name, module in previous.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    require(observer.OWNER_JOURNAL == OLD_JOURNAL / "finite-owner"
            and observer.SUCCESSOR == OLD, "Preserve original fitting/dispatch ownership")
    observer.HERE = HERE  # Only observation outputs move to this distinct directory.
    return observer, {name: sha(raw) for name, raw in source.items()}


def execute(args):
    require(git_file(args.source_git, (HERE / "continue_once.py").relative_to(ROOT).as_posix()) ==
            (HERE / "continue_once.py").read_bytes(), "Exact committed continuation")
    require(30 <= args.poll_seconds <= 300, "Bounded observation polling")
    require(datetime.now(timezone.utc) < datetime.fromisoformat(args.deadline_utc)
            <= datetime.fromisoformat(DEADLINE), "Finite read-only observation deadline")
    old = predecessor()
    owner_sha = await_owner()
    observer, source_sha = frozen_observer()
    # Build and validate once before claiming the new journal. This reads only.
    _, _, payload, provenance = observer.frozen()
    require(payload["mode"] == "progress" and provenance["owner_intent_sha256"] == owner_sha,
            "Exact existing read-only fitting observer")
    journal = ROOT / "artifacts" / HERE.name
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    require(not any(p.name.endswith(("-intent.json", "-result.json", "-failure.json"))
                    for p in journal.iterdir()), "Existing continuation requires reconciliation")
    observer.save(journal / "continuation-intent.json", {
        **old, "pid": os.getpid(), "continuation_source_git": args.source_git,
        "original_sources_sha256": source_sha, "fitting_owner_intent_sha256": owner_sha,
        "original_journal_preserved": True, "old91955_stop_repeated": False,
        "remote_mutations": 0, "GPU_or_CPU_allocations": 0,
    })
    original_args = copy.copy(args)
    original_args.source_git = ORIGINAL
    try:
        observer.run(original_args, journal)
    except Exception as error:
        receipt = {
            "status": "read_only_training_observation_failed", "observed_at": observer.utc(),
            "error_type": type(error).__name__, "remote_mutations": 0,
            "evidence": getattr(error, "evidence", {}), "clinical_accuracy_claim": False,
        }
        observer.save(journal / "observer-failure.json", receipt)
        observer.save(HERE / "actual-observation-failure.json", receipt)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    parser.add_argument("--poll-seconds", type=int, default=120)
    parser.add_argument("--deadline-utc", default=DEADLINE)
    args = parser.parse_args()
    os.umask(0o077)
    execute(args)


if __name__ == "__main__":
    main()
