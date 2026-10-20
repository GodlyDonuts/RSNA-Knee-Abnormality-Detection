"""One finite source-correction successor; local preparation has no remote effects."""

import argparse
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import contract as c

ROOT, HERE = Path(__file__).resolve().parents[3], Path(__file__).resolve().parent
_base, _builder, _sources = None, None, None


def git_file(pin, relative):
    c.require(len(pin) == 40 and all(ch in "0123456789abcdef" for ch in pin), "Exact source pin")
    return subprocess.check_output(["git", "show", pin + ":" + relative], cwd=ROOT)


def base():
    global _base, _builder, _sources
    if _base is not None:
        return _base
    _sources = {
        "http": git_file(c.PRIOR, c.PRIOR_DIR + "/contract.py").decode(),
        "builder": git_file(c.BUILDER, c.BUILDER_DIR + "/contract.py").decode(),
    }
    c.bind(_sources["http"], _sources["builder"])
    raw = git_file(c.BUILDER, c.BUILDER_DIR + "/launch.py")
    c.require(c.sha(raw) == c.BUILDER_SHA["launch.py"], "Frozen builder changed")
    source = c.replace_once(raw.decode(), '"accepted_v4_cache_owner_handoff"',
                            '"accepted_source_correction_owner_handoff"')
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader("source_corrected_base", loader=None))
    module.__file__ = str(HERE / "launch.py")
    previous = sys.modules.get("contract")
    sys.modules["contract"] = c
    try:
        exec(compile(source, "frozen88_source_corrected_operations", "exec"), module.__dict__)
    finally:
        if previous is None:
            sys.modules.pop("contract", None)
        else:
            sys.modules["contract"] = previous
    _base, _builder = module, module.build_program
    module.build_program = build_program
    return module


def build_program(own, originals, *, readonly=False):
    base()
    bound = dict(own, **{"contract.py": own["contract.py"] +
                        "\nbind(" + repr(_sources["http"]) + "," + repr(_sources["builder"]) + ")\n"})
    changed = dict(originals, **{"remote.py": c.derived_portable_remote(originals["remote.py"])})
    return _builder(bound, changed, readonly=readonly)


def freeze(args):
    old = base()
    c.observer_handoff(json.loads(args.observer_handoff.read_text()), ROOT, old.process_command)
    # The actual preceding controls remain readable; no original source is patched.
    frozen = old.freeze(args)
    own, originals, _, _, original, _, _, payload = frozen
    decision = json.loads(args.decision.read_text())
    c.require(decision["source_correction_policy"] == c.correction_policy()
              and decision["preserve_original_scientific_sources"] is True
              and c.hex64(decision["source_correction_regression_sha256"]),
              "Explicit baseline-plus-correction decision required")
    expected = {
        "source_correction_helper_sha256": c.HELPER,
        "source_correction_regression_sha256": decision["source_correction_regression_sha256"],
    }
    reader = original.cache_receipts

    def receipts():
        result = reader()
        if result is not None:
            c.proofs(*(result[role]["original_actual"] for role in ("raw", "context")),
                     expected=expected)
        return result

    original.cache_receipts = receipts
    payload["prior_operational_sources_sha256"] = c.PRIOR_SHA
    payload["placement_code_sha256"]["remote.py"] = c.sha(
        c.derived_portable_remote(originals["remote.py"]).encode())
    payload["source_correction_policy"] = c.correction_policy()
    payload["accepted_source_correction_proofs"] = expected
    c.require(set(own) == {"launch.py", "contract.py", "remote_adapter.py"}, "Exact control sources")
    return frozen


def fitting(args, frozen):
    own, _, fit, p, original, transport, program, payload = frozen
    runner_raw = git_file(c.RESILIENT, c.RESILIENT_DIR + "/fitting.py")
    c.require(c.sha(runner_raw) == c.RESILIENT_SHA, "Exact resilient scheduler required")
    runner = base().module(runner_raw, "source_corrected_resilient", HERE / "fitting.py")
    policy_dir = "initiative/research/predictive-deadline-admission-20261020-v1"
    policies = {name: base().module(git_file(c.POLICY, policy_dir + "/" + name + ".py"),
                                    "source_corrected_" + name, ROOT / policy_dir / (name + ".py"))
                for name in ("policy", "placement_failure")}
    journal = ROOT / "artifacts" / HERE.name / "finite-owner"
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    p.once(journal / "owner-intent.json", {
        "pid": os.getpid(), "source_git": args.source_git, "science_git": c.SCIENCE,
        "original_owner_source_git": c.RESILIENT, "original_owner_pid": 96074,
        "prior_operational_source_git": c.PRIOR,
        "source_sha256": {name: c.sha(raw.encode()) for name, raw in own.items()},
        "handoff_sha256": payload["handoff_sha256"],
        "deployment_handoff_sha256": p.file_sha(args.deployment_handoff),
        "observer_handoff_sha256": p.file_sha(args.observer_handoff),
        "decision_sha256": p.file_sha(args.decision),
        "decision_path": str(args.decision.resolve().relative_to(ROOT)),
        "handoff_path": str(args.handoff.resolve().relative_to(ROOT)),
        "source_correction_policy": c.correction_policy(),
        "resilient_run_source_git": c.RESILIENT, "resilient_run_sha256": c.RESILIENT_SHA,
        "policy_source_git": c.POLICY, "automatic_retry": False,
        "prepared_models_reinitialized": False,
    })
    try:
        runner.run(fit.Owner(args, original, transport, program, payload, journal),
                   deadline_policy=policies["policy"], placement_policy=policies["placement_failure"],
                   once=p.once, audit_dir=HERE)
    except BaseException as error:
        p.once(journal / "owner-ended.json", {
            "status": "requires_read_only_reconciliation", "error_type": type(error).__name__,
            "remote_jobs_cancelled": [], "automatic_retry": False,
        })
        raise


def deployment(args, frozen):
    """Bind final models to actual input admissions before any account observation."""
    old = base()
    original_module = old.module

    def module(source, name, file, inject=None):
        result = original_module(source, name, file, inject)
        if name == "successor_portable_deployment":
            read_finals = result.portable_finals

            def finals(original, directory=HERE):
                value = read_finals(original, directory)
                if value is not None:
                    c.final_source_lineage(value, directory)
                return value

            result.portable_finals = finals
        return result

    old.module = module
    try:
        old.deployment(args, frozen)
    finally:
        old.module = original_module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fitting", "deployment"))
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--deployment-decision", type=Path)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--deployment-handoff", type=Path, required=True)
    parser.add_argument("--observer-handoff", type=Path, required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    parser.add_argument("--deadline-utc", default="2026-10-22T12:00:00+00:00")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    os.umask(0o077)
    c.require(args.mode != "deployment" or args.deployment_decision is not None,
              "Explicit deployment decision required")
    frozen = freeze(args)
    (fitting if args.mode == "fitting" else deployment)(args, frozen)


if __name__ == "__main__":
    main()
