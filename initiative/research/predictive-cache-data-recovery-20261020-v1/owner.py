"""Distinct finite exact-source successor; no work or remote operations on import."""
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
PREVIOUS = "d7326951eeb986f26eb21183d61c2b52476a7986"
PREVIOUS_REL = "initiative/research/predictive-cache-http-recovery-20261020-v1"
ORIGINALS = {"raw": "980960", "context": "980969"}


def sha(body):
    return hashlib.sha256(body).hexdigest()


def git_file(pin, name):
    if len(pin) != 40 or any(c not in "0123456789abcdef" for c in pin):
        raise ValueError("Exact committed source required")
    return subprocess.check_output(["git", "show", pin + ":" + name], cwd=ROOT).decode()


def previous():
    derived = importlib.util.module_from_spec(importlib.util.spec_from_loader("frozen_http_generation", None))
    exec(compile(git_file(PREVIOUS, PREVIOUS_REL + "/generation.py"), "frozen_http_generation", "exec"), derived.__dict__)
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("frozen_http_owner", None))
    module.__file__ = str(ROOT / PREVIOUS_REL / "owner.py")
    source = git_file(PREVIOUS, PREVIOUS_REL + "/owner.py")
    assert source.count("import generation\n") == 1
    module.generation = derived
    exec(compile(source.replace("import generation\n", ""), "frozen_http_owner", "exec"), module.__dict__)
    return module


def freeze(commit, regression_body, regression_sha256, prior_body, prior_sha256,
           prefix_body, prefix_sha256, *, capacity_body, capacity_sha256,
           http_body, http_sha256, correction_body, correction_sha256,
           plan_body, receipt_body, correction_regression_body, correction_regression_sha256,
           corrected_sources, mixed_body=None, mixed_sha256=None):
    old = previous()
    bundle = old.freeze(PREVIOUS, regression_body, regression_sha256, prior_body, prior_sha256,
                        prefix_body, prefix_sha256, capacity_body=capacity_body,
                        capacity_sha256=capacity_sha256, http_body=http_body, http_sha256=http_sha256)
    own = {name: git_file(commit, REL + "/" + name) for name in
           ("owner.py", "generation.py", "suffix.py", "source_correction.py", "overlay_resume.py")}
    for name, text in own.items():
        assert text.encode() == (CONTROL / name).read_bytes(), "Exact frozen controls required"
    assert sha(correction_body) == correction_sha256
    decision = json.loads(correction_body)
    assert decision["status"] == "accepted_original_source_correction_missing_only_recovery"
    assert decision["original_jobs"] == ORIGINALS
    assert decision["funding"] == bundle["payloads"]["raw"]["funding"]
    assert decision["corrected_sources"] == corrected_sources
    suffix = bundle["payloads"]["raw"]["suffix_reconcile"]
    assert suffix.count("len(roots) <= 3") == 2
    suffix = suffix.replace("len(roots) <= 3", "len(roots) <= 4") + "\n" + own["suffix.py"]
    common = {
        "source_git": commit, "suffix_reconcile": suffix,
        "suffix_reconcile_sha256": sha(suffix.encode()),
        "source_correction_decision_body": correction_body.decode(),
        "source_correction_decision_sha256": correction_sha256,
        "source_correction_plan_body": plan_body.decode(),
        "source_correction_plan_sha256": sha(plan_body),
        "source_correction_receipt_body": receipt_body.decode(),
        "source_correction_receipt_sha256": sha(receipt_body),
        "source_correction_helper_source": own["source_correction.py"],
        "source_correction_helper_sha256": sha(own["source_correction.py"].encode()),
        "source_correction_regression_body": correction_regression_body.decode(),
        "source_correction_regression_sha256": correction_regression_sha256,
        "source_overlay_resume_source": own["overlay_resume.py"],
        "source_overlay_resume_sha256": sha(own["overlay_resume.py"].encode()),
        "corrected_sources": corrected_sources,
        "mixed_prefix_admission": None if mixed_body is None else mixed_body.decode(),
        "mixed_prefix_admission_sha256": mixed_sha256,
    }
    scope = {}
    exec(compile(suffix, "frozen_source_correction_contract", "exec"), scope)
    scope["correction_contract"](common)
    if mixed_body is not None:
        scope["admission_receipt"](common)
    bundle["programs"] = generation.programs(bundle["programs"])
    for payload in bundle["payloads"].values():
        payload.update(common)
    bundle["commit"] = commit
    bundle["sha256"].update({REL + "/" + n: sha(s.encode()) for n, s in own.items()})
    bundle["sha256"].update({"generated:" + role: sha(s.encode()) for role, s in bundle["programs"].items()})
    bundle["sha256"]["generated:suffix"] = sha(suffix.encode())
    return bundle


def next_action(jobs, phases):
    http = previous()
    source = git_file(http.PREVIOUS, http.PREVIOUS_REL + "/owner.py")
    begin, end = source.index("def next_action("), source.index("\n\ndef command(")
    text = source[begin:end].replace('("979664", "979665")', '("980960", "980969")')
    scope = {}
    exec(compile(text, "source_corrected_successor_transition", "exec"), scope)
    return scope["next_action"](jobs, phases)


def command(role, transport):
    return [s.replace("-http-recovery-v1", "-data-recovery-v1")
            for s in previous().command(role, transport)]


def main():
    parser = argparse.ArgumentParser(add_help=False)
    for name in ("capacity-decision", "http-decision", "correction-decision",
                 "correction-regression", "mixed-prefix-admission"):
        parser.add_argument("--" + name, type=Path, required=True)
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--correction-plan", type=Path, required=True)
    parser.add_argument("--correction-receipt", type=Path, required=True)
    extra, remainder = parser.parse_known_args()
    capacity = previous().previous()
    old = capacity.base_module()
    original = capacity.git_file(capacity.PREVIOUS, capacity.PREVIOUS_REL + "/owner.py")
    body = original[original.index("def main():"):original.index('\n\nif __name__ == "__main__":')]
    body = body.replace("artifacts/predictive-full-cache-alias-recovery-20261020-v3", "artifacts/" + CONTROL.name)
    exec(compile(body, "distinct_exact_source_owner_loop", "exec"), old.__dict__)
    old.CONTROL, old.ORIGINALS, old.REMOTE_ROOTS = CONTROL, ORIGINALS, generation.ROOTS
    old.FUNDING, old.command, old.next_action = capacity.FUNDING, command, next_action
    old.freeze = lambda *args: freeze(
        *args, capacity_body=extra.capacity_decision.read_bytes(), capacity_sha256=extra.capacity_decision_sha256,
        http_body=extra.http_decision.read_bytes(), http_sha256=extra.http_decision_sha256,
        correction_body=extra.correction_decision.read_bytes(), correction_sha256=extra.correction_decision_sha256,
        plan_body=extra.correction_plan.read_bytes(), receipt_body=extra.correction_receipt.read_bytes(),
        correction_regression_body=extra.correction_regression.read_bytes(),
        correction_regression_sha256=extra.correction_regression_sha256,
        corrected_sources=json.loads(extra.correction_decision.read_bytes())["corrected_sources"],
        mixed_body=extra.mixed_prefix_admission.read_bytes(), mixed_sha256=extra.mixed_prefix_admission_sha256)
    sys.argv = [sys.argv[0], *remainder]
    old.main()


if __name__ == "__main__":
    main()
