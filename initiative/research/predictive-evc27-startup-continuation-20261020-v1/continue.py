"""Once-only local startup recovery; preserve the original controller and journal."""

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
SOURCE = "2a224ae05fc7c2fc78d4d508b6407344ac523a1c"
OLD = ROOT / "initiative/research/predictive-evc27-deadline-placement-20261020-v1"
JOURNAL = ROOT / "artifacts" / OLD.name / "finite-owner"
INTENT = "b310a8cf6267ea4ce1b40b4efef27ad375b9bc869dfaee435b9c86ed0f40ebd0"
LAUNCH = "1487bfcf7e32c1f2ace23deb86fe6aa19c3728b0651ce165e127e66514abe8d9"
STDERR = "9455e429a46cc5eee24d18c8d15f2e7a568252237bd34cf3ecc2511ed9a3f131"
STARTUP = {"launch-intent.json", "launcher.json", "owner-ended.json", "owner-intent.json", "stderr", "stdout"}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_source(pin, path):
    require(len(pin) == 40 and all(c in "0123456789abcdef" for c in pin), "Exact source pin required")
    return subprocess.check_output(["git", "show", pin + ":" + path.relative_to(ROOT).as_posix()], cwd=ROOT)


def original():
    def module(name):
        path = OLD / (name + ".py")
        raw = load_source(SOURCE, path)
        require(raw == path.read_bytes(), "Frozen primary controller must remain unchanged")
        result = importlib.util.module_from_spec(importlib.util.spec_from_loader("startup_" + name, None))
        result.__file__ = str(path)
        exec(compile(raw, str(path), "exec"), result.__dict__)
        return result
    previous = sys.modules.get("contract")
    try:
        sys.modules["contract"] = module("contract")
        return module("owner")
    finally:
        if previous is None:
            sys.modules.pop("contract", None)
        else:
            sys.modules["contract"] = previous


def inspect(proof, journal, process_command):
    require(proof["status"] == "actual_local_startup_guard_failure_before_any_primary_phase"
            and proof["pid"] == 32377 and proof["source_git"] == SOURCE
            and proof["process_absent"] is True
            and proof["primary_phase_intents"] == proof["primary_phase_results"] == []
            and proof["remote_operations_executed"] == 0 and proof["remote_jobs_cancelled"] == []
            and proof["deployment_owner32378_untouched"] is True
            and proof["cache_owner28954_untouched"] is True,
            "Exact local-only failure receipt required")
    require(not process_command(32377).strip(), "Original fitting process remains live")
    require({p.name for p in journal.iterdir()} == set(proof["journal_inventory"]) == STARTUP,
            "Only exact original startup files are eligible; phase state cannot continue here")
    hashes = {}
    for path in journal.iterdir():
        require(path.is_file() and not path.is_symlink() and path.stat().st_uid == os.getuid()
                and not path.stat().st_mode & 0o077, "Owner-only original startup files required")
        hashes[path.name] = sha(path.read_bytes())
    require(hashes["owner-intent.json"] == proof["original_intent_sha256"] == INTENT
            and hashes["launch-intent.json"] == proof["original_launch_intent_sha256"] == LAUNCH
            and hashes["stderr"] == proof["stderr_sha256"] == STDERR
            and (journal / "stdout").read_bytes() == b"", "Original failure bytes changed")
    ended = json.loads((journal / "owner-ended.json").read_bytes())
    require(ended == proof["owner_ended"] and ended == {
        "automatic_retry": False, "error_type": "ValueError", "remote_jobs_cancelled": [],
        "status": "requires_read_only_reconciliation"}, "Unexpected original failure")
    owner = json.loads((journal / "owner-intent.json").read_bytes())
    launcher = json.loads((journal / "launcher.json").read_bytes())
    require(owner["pid"] == launcher["pid"] == 32377 and owner["source_git"] == launcher["source_git"] == SOURCE
            and launcher["role"] == "fitting", "Exact original process identity required")
    return hashes


def execute(old, frozen, args, proof, proof_sha, journal, continuation, *, source_git, process_command):
    hashes = inspect(proof, journal, process_command)
    require(not (continuation / "continuation-intent.json").exists(), "Existing continuation cannot repeat")
    continuation.mkdir(mode=0o700, parents=True, exist_ok=True)
    require({p.name for p in continuation.iterdir()} <= {"launch-intent.json", "launcher.json", "stdout", "stderr"},
            "New continuation journal already contains operation evidence")
    p = frozen["p"]
    p.once(continuation / "continuation-intent.json", {
        "pid": os.getpid(), "source_git": source_git, "original_source_git": SOURCE,
        "original_owner_pid": 32377, "original_failure_receipt_sha256": proof_sha,
        "original_startup_sha256": hashes, "automatic_retry": False,
        "only_startup_marker_renamed": True, "original_owner_intent_retained": True,
        "deployment_owner32378_untouched": True})
    source, target = journal / "launch-intent.json", journal / "launcher-started.json"
    # Exclusive link/unlink preserves exact original inode/bytes and cannot replace
    # another receipt. Uncertainty leaves evidence and never authorizes a retry.
    os.link(source, target)
    require(sha(target.read_bytes()) == LAUNCH and sha(source.read_bytes()) == LAUNCH, "Startup marker changed")
    source.unlink()
    require(all(sha((journal / name).read_bytes()) == digest for name, digest in hashes.items()
                if name != "launch-intent.json"), "Original startup evidence must remain unchanged")
    instance = frozen["fit"].Owner(args, frozen["original"], frozen["transport"], frozen["program"],
                                    frozen["payload"], journal)
    instance.placement = p
    instance.transfer = lambda arm=None: old.transfer(instance, arm=arm)
    try:
        result = old.run(instance)
    except BaseException as error:
        p.once(continuation / "continuation-ended.json", {"status": "requires_read_only_reconciliation",
             "error_type": type(error).__name__, "automatic_retry": False, "remote_jobs_cancelled": []})
        raise
    p.once(continuation / "continuation-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--failure", type=Path, default=OLD / "actual-local-startup-failure.json")
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    parsed = parser.parse_args()
    os.umask(0o077)
    require(load_source(parsed.source_git, Path(__file__)) == Path(__file__).read_bytes(),
            "Exact committed continuation source required")
    old = original()
    args = SimpleNamespace(source_git=SOURCE, decision=OLD / "decision.accepted.json",
        handoff=OLD / "actual-fitting-owner-stopped.json", deployment_handoff=OLD / "actual-deployment-owner-stopped.json",
        observer_handoff=OLD / "actual-observer-owner-stopped.json", deadline_utc=old.d.END,
        poll_seconds=120, socket=parsed.socket)
    proof_body = parsed.failure.read_bytes()
    proof = json.loads(proof_body)
    inspect(proof, JOURNAL, old.process_command)
    frozen = old.freeze(args)  # Only original local source/decision/handoff validation.
    intent = json.loads((JOURNAL / "owner-intent.json").read_bytes())
    require(frozen["payload"]["deadline_code_sha256"] == intent["source_sha256"]
            and sha(args.decision.read_bytes()) == intent["decision_sha256"],
            "Unchanged original operational capsule and decision required")
    execute(old, frozen, args, proof, sha(proof_body), JOURNAL,
            ROOT / "artifacts" / HERE.name, source_git=parsed.source_git,
            process_command=old.process_command)


if __name__ == "__main__":
    main()
