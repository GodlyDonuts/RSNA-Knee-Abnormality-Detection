"""Finite local successor launchers; importing/preparing does not run remote work."""
import argparse
import importlib.util
import json
import os
import subprocess
import sys
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import contract as c

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
FILES = ("launch.py", "contract.py", "remote_adapter.py")


def git_file(pin, name):
    c.require(len(pin) == 40 and all(ch in "0123456789abcdef" for ch in pin), "Exact source pin")
    return subprocess.check_output(["git", "show", pin + ":" + name], cwd=ROOT)


def module(source, name, file, inject=None):
    result = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    result.__file__ = str(file)
    result.__dict__.update(inject or {})
    exec(compile(source, str(file), "exec"), result.__dict__)
    return result


@contextmanager
def imports(**items):
    old = {name: sys.modules.get(name) for name in items}
    sys.modules.update(items)
    try:
        yield
    finally:
        for name, value in old.items():
            if value is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = value


def process_command(pid):
    return subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                          capture_output=True, text=True, check=False).stdout


def freeze(args):
    own = {name: git_file(args.source_git, (HERE / name).relative_to(ROOT).as_posix()).decode()
           for name in FILES}
    for name, source in own.items():
        c.require(source.encode() == (HERE / name).read_bytes(), "Invoke committed successor files")
    fitting_text, deployment_text = args.handoff.read_text(), args.deployment_handoff.read_text()
    fit_stop, deploy_stop = json.loads(fitting_text), json.loads(deployment_text)
    c.local_handoff(fit_stop, "fitting", ROOT, process_command)
    c.local_handoff(deploy_stop, "deployment", ROOT, process_command)
    decision = json.loads(args.decision.read_text())
    c.require(decision["status"] == "accepted_v4_cache_owner_handoff"
              and decision["placement_source_git"] == args.source_git
              and decision["science_git"] == c.SCIENCE
              and decision["shared_future_bytes"] >= 123 * 2**20
              and decision["shared_future_files"] >= 376
              and decision["retained_total_scratch_bytes"] == 6 * 2**30
              and decision["no_paid_credits"] is True,
              "Exact successor source/funding decision required")
    originals = {name: git_file(c.PARENT, c.OLD + "/" + name).decode()
                 for name in ("placement.py", "owner.py", "remote.py", "deployment.py")}
    old_p = module(originals["placement.py"], "original_placement", ROOT / c.OLD / "placement.py")
    with imports(placement=old_p):
        fit = module(originals["owner.py"], "original_portable_owner", ROOT / c.OLD / "owner.py")
    # Reuse the original actual979667 preparation and already proven historical handoff.
    # This function performs local reads only; no initialization or remote call.
    old_owner, transport, _, payload = fit.freeze(
        c.PARENT, ROOT / c.OLD / "decision.accepted.json",
        ROOT / c.OLD / "actual-fitting-owner68792-stopped.json",
        ROOT / c.OLD / "actual-deployment-owner77985-stopped.json",
    )
    old_owner.COLLECT = ROOT / c.CACHE
    original_reader = old_owner.cache_receipts

    def receipts():
        result = original_reader()
        if result is None:
            return None
        raw, context = result["raw"]["original_actual"], result["context"]["original_actual"]
        c.require(raw["continuation_chain"] == context["continuation_chain"], "Cache ancestry differs")
        c.chain(context["continuation_chain"])
        for key in ("capacity_decision_sha256", "mixed_prefix_admission_sha256"):
            c.require(raw[key] == context[key] and len(raw[key]) == 64, "Final input admission differs")
        return result

    old_owner.cache_receipts = receipts
    placement_source = c.derived_placement(originals["placement.py"])
    p = module(placement_source, "successor_placement", ROOT / c.OLD / "placement.py",
               {"successor_handoff": c.handoff})
    fit.p, fit.HERE = p, HERE
    payload.update(
        handoff=fit_stop, handoff_text=fitting_text, handoff_sha256=c.sha(fitting_text.encode()),
        placement_source_git=args.source_git, owner_git=args.source_git,
        shared_bytes=decision["shared_future_bytes"], shared_files=decision["shared_future_files"],
        successor_code_sha256={name: c.sha(source.encode()) for name, source in own.items()},
        placement_code_sha256={"placement.py": c.sha(placement_source.encode()),
                              "remote.py": c.sha(originals["remote.py"].encode()),
                              "remote_adapter.py": c.sha(own["remote_adapter.py"].encode())},
    )
    program = build_program(own, originals)
    return own, originals, fit, p, old_owner, transport, program, payload


def build_program(own, originals, *, readonly=False):
    """Pure immutable capsule builder, shared with the allocation-free observer."""
    placement_source = c.derived_placement(originals["placement.py"])
    program = (
        "import sys,json,types\n"
        "c=types.ModuleType('successor_contract');sys.modules['successor_contract']=c\n"
        f"exec(compile({own['contract.py']!r},'successor_contract.py','exec'),c.__dict__)\n"
        "p=types.ModuleType('placement');sys.modules['placement']=p\n"
        "p.successor_handoff=c.handoff\n"
        f"exec(compile({placement_source!r},'placement.py','exec'),p.__dict__)\n"
        "n={'__name__':'explicit_successor_operation'}\n"
        f"exec(compile({originals['remote.py']!r},'frozen_926b_remote.py','exec'),n)\n"
        "a={'__name__':'successor_adapter'}\n"
        f"exec(compile({own['remote_adapter.py']!r},'remote_adapter.py','exec'),a)\n"
        "v=json.load(sys.stdin);a['install'](n,v)\n"
        "try:\n print(json.dumps(n['main'](v),allow_nan=False),flush=True)\n"
        "except BaseException as e:\n"
        " t=e.__traceback__\n"
        " while t.tb_next:t=t.tb_next\n"
        " print(json.dumps({'error_type':type(e).__name__,'site_line':t.tb_lineno}),file=sys.stderr)\n"
        " raise SystemExit(1)\n"
    )
    if readonly:
        prefix = "import sys,json\nv=json.load(sys.stdin)\nassert v['mode']=='progress' and 'token' not in v\n"
        program = prefix + program.replace("v=json.load(sys.stdin);a['install'](n,v)\n", "a['install'](n,v)\n")
    return program


def fitting(args, frozen):
    own, _, fit, p, original, transport, program, payload = frozen
    deadline = datetime.fromisoformat(args.deadline_utc)
    c.require(datetime.now(timezone.utc) < deadline <= datetime(2026, 10, 22, 12, tzinfo=timezone.utc)
              and 30 <= args.poll_seconds <= 300, "Finite bounded fitting owner")
    journal = ROOT / "artifacts" / HERE.name / "finite-owner"
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    p.once(journal / "owner-intent.json", {
        "pid": os.getpid(), "source_git": args.source_git, "science_git": c.SCIENCE,
        "original_owner_source_git": c.PARENT, "original_owner_pid": 80995,
        "handoff_sha256": payload["handoff_sha256"],
        "deployment_handoff_sha256": p.file_sha(args.deployment_handoff),
        "source_sha256": {name: c.sha(body.encode()) for name, body in own.items()},
        "decision_sha256": p.file_sha(args.decision), "deadline": args.deadline_utc,
        "decision_path": str(args.decision.resolve().relative_to(ROOT)),
        "handoff_path": str(args.handoff.resolve().relative_to(ROOT)),
        "automatic_retry": False, "prepared_models_reinitialized": False,
    })
    try:
        fit.Owner(args, original, transport, program, payload, journal).run()
    except BaseException as error:
        p.once(journal / "owner-ended.json", {
            "error_type": type(error).__name__, "status": "requires_read_only_reconciliation",
            "remote_jobs_cancelled": [], "automatic_retry": False,
        })
        raise


def deployment(args, frozen):
    _, originals, fit, p, *_ = frozen
    source = originals["deployment.py"]
    start = source.index('    for name in ("deployment.py", "placement.py", "owner.py"):')
    end = source.index('    stop = stopped_before_phase(', start)
    source = source[:start] + '    # Launcher already verified every frozen successor file.\n' + source[end:]
    source = c.replace_once(source,
        'args.source_git, (HERE / "placement.py").relative_to(ROOT).as_posix()',
        'successor_parent, successor_old + "/placement.py"')
    with imports(placement=p, owner=fit):
        dep = module(source, "successor_portable_deployment", HERE / "deployment.py",
                     {"successor_parent": c.PARENT, "successor_old": c.OLD})
    canonical_source = fit.git_file(
        dep.PARENT, dep.PARENT_DIR.relative_to(ROOT).as_posix() + "/owner.py"
    ).decode()
    canonical_source = c.derived_deployment_owner(canonical_source)
    dep.original_owner = lambda: module(
        canonical_source, "successor_canonical_deployment_owner", dep.PARENT_DIR / "owner.py"
    )

    def stopped(path):
        value = json.loads(path.read_text())
        c.local_handoff(value, "deployment", ROOT, process_command)
        return value

    dep.stopped_before_phase = stopped
    # Existing run() binds its own distinct decision, source and original account phases.
    args.fitting_handoff = args.handoff
    args.decision = args.deployment_decision
    c.deployment_funding(json.loads(args.decision.read_text()))
    dep.run(args)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fitting", "deployment"))
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--deployment-decision", type=Path)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--deployment-handoff", type=Path, required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    parser.add_argument("--deadline-utc", default="2026-10-22T09:00:00+00:00")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    os.umask(0o077)
    c.require(args.mode != "deployment" or args.deployment_decision is not None,
              "Explicit frozen deployment decision required")
    frozen = freeze(args)
    (fitting if args.mode == "fitting" else deployment)(args, frozen)


if __name__ == "__main__":
    main()
