"""Distinct finite HTTP successor. No work on import; original journals stay immutable."""

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import generation

ROOT = Path(__file__).resolve().parents[3]
CONTROL = Path(__file__).resolve().parent
REL = CONTROL.relative_to(ROOT).as_posix()
PREVIOUS = "370f2c598f4a25d92fc4764c899a1632ff50b675"
PREVIOUS_REL = "initiative/research/predictive-cache-capacity-recovery-20261020-v1"
HTTP_MODULE = "rsna/competitive/predictive/source_http.py"
ORIGINALS = {"raw": "980743", "context": "980744"}


def sha(body):
    return hashlib.sha256(body).hexdigest()


def git_file(pin, name):
    if len(pin) != 40 or any(c not in "0123456789abcdef" for c in pin):
        raise ValueError("Exact committed source required")
    return subprocess.check_output(["git", "show", pin + ":" + name], cwd=ROOT).decode()


def previous():
    source = git_file(PREVIOUS, PREVIOUS_REL + "/owner.py")
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("capacity_owner", None))
    module.__file__ = str(ROOT / PREVIOUS_REL / "owner.py")
    exec(compile(source, "original_capacity_owner_library", "exec"), module.__dict__)
    return module


def freeze(commit, regression_body, regression_sha256, prior_body, prior_sha256,
           prefix_body, prefix_sha256, *, capacity_body, capacity_sha256,
           http_body, http_sha256, mixed_body=None, mixed_sha256=None):
    old = previous()
    bundle = old.freeze(
        PREVIOUS, regression_body, regression_sha256, prior_body, prior_sha256,
        prefix_body, prefix_sha256, capacity_body=capacity_body,
        capacity_sha256=capacity_sha256,
    )
    own = {name: git_file(commit, REL + "/" + name)
           for name in ("owner.py", "generation.py", "suffix.py")}
    for name, source in own.items():
        assert source.encode() == (CONTROL / name).read_bytes(), "Frozen launch controls required"
    http_source = git_file(commit, HTTP_MODULE)
    assert http_source.encode() == (ROOT / HTTP_MODULE).read_bytes(), "Frozen HTTP helper required"
    assert sha(http_body) == http_sha256
    decision = json.loads(http_body)
    assert decision["status"] == "accepted_original_v4_HTTP_failure_reconciled"
    assert decision["original_jobs"] == ORIGINALS
    assert decision["verified_failure_site"] == "HTTP_response_status_before_body"
    assert decision["all_originals_preserved"] is True
    assert decision["source_http_sha256"] == sha(http_source.encode())
    assert decision["funding"] == old.FUNDING
    assert decision["lane_count"] == 4 and decision["maximum_HTTP_requests"] == 8
    assert decision["scratch_cap_bytes"] == 6 * 2**30
    suffix = git_file(PREVIOUS, PREVIOUS_REL + "/suffix.py") + "\n" + own["suffix.py"]
    if mixed_body is not None:
        assert sha(mixed_body) == mixed_sha256
        receipt = json.loads(mixed_body)
        assert receipt["suffix_reconcile_sha256"] == sha(suffix.encode())
        assert receipt["source_http_sha256"] == decision["source_http_sha256"]
        assert receipt["http_recovery_decision_sha256"] == http_sha256
    bundle["programs"] = generation.programs(bundle["programs"])
    for payload in bundle["payloads"].values():
        payload.update(
            source_git=commit, suffix_reconcile=suffix,
            suffix_reconcile_sha256=sha(suffix.encode()),
            http_recovery_decision_body=http_body.decode(),
            http_recovery_decision_sha256=http_sha256,
            source_http_sha256=sha(http_source.encode()),
            mixed_prefix_admission=None if mixed_body is None else mixed_body.decode(),
            mixed_prefix_admission_sha256=mixed_sha256,
        )
    raw = bundle["payloads"]["raw"]
    raw["files"][HTTP_MODULE] = http_source
    raw["module_sha256"][HTTP_MODULE] = sha(http_source.encode())
    bundle["commit"] = commit
    bundle["sha256"].update({REL + "/" + n: sha(s.encode()) for n, s in own.items()})
    bundle["sha256"][HTTP_MODULE] = sha(http_source.encode())
    bundle["sha256"].update({"generated:" + role: sha(s.encode())
                             for role, s in bundle["programs"].items()})
    bundle["sha256"]["generated:suffix"] = sha(suffix.encode())
    return bundle


def next_action(jobs, phases):
    # The previous state machine is unchanged except for the two exact new failures.
    source = git_file(PREVIOUS, PREVIOUS_REL + "/owner.py")
    start, end = source.index("def next_action("), source.index("\n\ndef command(")
    text = source[start:end].replace('("979664", "979665")', '("980743", "980744")')
    scope = {}
    exec(compile(text, "HTTP_successor_transition", "exec"), scope)
    return scope["next_action"](jobs, phases)


def command(role, transport):
    argv = previous().command(role, transport)
    return [item.replace("-capacity-recovery-v1", "-http-recovery-v1") for item in argv]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for role in ("capacity-decision", "http-decision", "mixed-prefix-admission"):
        parser.add_argument("--" + role, type=Path, required=True)
        parser.add_argument("--" + role + "-sha256", required=True)
    extra, remainder = parser.parse_known_args()
    capacity = previous()
    old = capacity.base_module()
    original = capacity.git_file(capacity.PREVIOUS, capacity.PREVIOUS_REL + "/owner.py")
    body = original[original.index("def main():"):original.index('\n\nif __name__ == "__main__":')]
    body = body.replace("artifacts/predictive-full-cache-alias-recovery-20261020-v3",
                        "artifacts/" + CONTROL.name)
    exec(compile(body, "distinct_HTTP_owner_loop", "exec"), old.__dict__)
    old.CONTROL, old.ORIGINALS, old.REMOTE_ROOTS = CONTROL, ORIGINALS, generation.ROOTS
    old.FUNDING, old.command, old.next_action = capacity.FUNDING, command, next_action
    old.freeze = lambda *args: freeze(
        *args, capacity_body=extra.capacity_decision.read_bytes(),
        capacity_sha256=extra.capacity_decision_sha256,
        http_body=extra.http_decision.read_bytes(), http_sha256=extra.http_decision_sha256,
        mixed_body=extra.mixed_prefix_admission.read_bytes(),
        mixed_sha256=extra.mixed_prefix_admission_sha256,
    )
    sys.argv = [sys.argv[0], *remainder]
    old.main()


if __name__ == "__main__":
    main()
