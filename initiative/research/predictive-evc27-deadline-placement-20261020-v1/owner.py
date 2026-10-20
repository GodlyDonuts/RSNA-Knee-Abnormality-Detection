"""One finite all-evc27 owner; no allocation or source mutation on import."""

import argparse
import copy
import importlib.util
import json
import os
import select
import subprocess
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import contract as d

ROOT, HERE = Path(__file__).resolve().parents[3], Path(__file__).resolve().parent
FILES = ("contract.py", "owner.py", "remote.py", "deployment.py")
VERIFIED = "actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified"


def git_file(pin, path):
    d.require(len(pin) == 40 and all(c in "0123456789abcdef" for c in pin), "Exact source pin")
    return subprocess.check_output(["git", "show", pin + ":" + path], cwd=ROOT)


def module(source, name, file, inject=None):
    result = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, None))
    result.__file__ = str(file)
    result.__dict__.update(inject or {})
    exec(compile(source, str(file), "exec"), result.__dict__)
    return result


@contextmanager
def imports(**values):
    before = {k: sys.modules.get(k) for k in values}
    sys.modules.update(values)
    try:
        yield
    finally:
        for k, v in before.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


def process_command(pid):
    return subprocess.run(["ps", "-p", str(pid), "-o", "command="],
                          capture_output=True, text=True, check=False).stdout


def previous():
    contract = module(git_file(d.PREVIOUS, d.PREVIOUS_DIR + "/contract.py"),
                      "frozen_HTTP499_contract", ROOT / d.PREVIOUS_DIR / "contract.py")
    with imports(contract=contract):
        launch = module(git_file(d.PREVIOUS, d.PREVIOUS_DIR + "/launch.py"),
                        "frozen_HTTP499_launcher", ROOT / d.PREVIOUS_DIR / "launch.py")
    launch.base()  # Local immutable-source reads only; no old owner is restarted.
    return launch, contract


def build_program(parent, originals, own):
    inherited = {n: git_file(d.PREVIOUS, d.PREVIOUS_DIR + "/" + n).decode()
                 for n in ("launch.py", "contract.py", "remote_adapter.py")}
    program = parent.build_program(inherited, originals)
    anchor = "v=json.load(sys.stdin);a['install'](n,v)\n"
    d.require(program.count(anchor) == 1, "Frozen HTTP499 capsule changed")
    extension = (
        f"n['__portable_source__']={parent.c.derived_portable_remote(originals['remote.py'])!r}\n"
        "d=types.ModuleType('deadline_contract');sys.modules['deadline_contract']=d\n"
        f"exec(compile({own['contract.py']!r},'deadline_contract.py','exec'),d.__dict__)\n"
        "p.successor_handoff=d.handoff\n"
        "r={'__name__':'deadline_remote_adapter'}\n"
        f"exec(compile({own['remote.py']!r},'deadline_remote.py','exec'),r)\n"
        "r['install'](n,v)\n"
    )
    return program.replace(anchor, anchor + extension)


def freeze(args):
    own = {n: git_file(args.source_git, (HERE / n).relative_to(ROOT).as_posix()).decode()
           for n in FILES}
    for name, source in own.items():
        d.require(source.encode() == (HERE / name).read_bytes(), "Exact committed new controls required")
    stops = {}
    for role, path in (("fitting", args.handoff), ("deployment", args.deployment_handoff),
                       ("observer", args.observer_handoff)):
        stops[role] = json.loads(path.read_bytes())
        d.local_handoff(stops[role], role, ROOT, process_command)
    decision = d.decision(json.loads(args.decision.read_bytes()), args.source_git)
    parent, c = previous()
    d.require(decision["source_correction_policy"] == c.correction_policy()
              and decision["preserve_original_scientific_sources"] is True,
              "Unchanged accepted MRI input lineage required")
    originals = {n: git_file(c.PARENT, c.OLD + "/" + n).decode()
                 for n in ("placement.py", "owner.py", "remote.py", "deployment.py")}
    original_p = module(originals["placement.py"], "original_placement", ROOT / c.OLD / "placement.py")
    with imports(placement=original_p):
        fit = module(originals["owner.py"], "original_portable_owner", ROOT / c.OLD / "owner.py")
    original, transport, _, payload = fit.freeze(
        c.PARENT, ROOT / c.OLD / "decision.accepted.json",
        ROOT / c.OLD / "actual-fitting-owner68792-stopped.json",
        ROOT / c.OLD / "actual-deployment-owner77985-stopped.json")
    original.COLLECT = ROOT / c.CACHE
    original_reader = original.cache_receipts
    expected = {"source_correction_helper_sha256": c.HELPER,
                "source_correction_regression_sha256": decision["source_correction_regression_sha256"]}
    http = c.accepted_http(decision)

    def receipts():
        result = original_reader()
        if result is not None:
            values = [result[r]["original_actual"] for r in ("raw", "context")]
            d.require(values[0]["continuation_chain"] == values[1]["continuation_chain"], "Cache lineage differs")
            c.chain(values[0]["continuation_chain"])
            c.proofs(*values, expected={**expected, **http})
        return result

    original.cache_receipts = receipts
    p = module(c.derived_placement(originals["placement.py"]), "deadline_placement",
               ROOT / c.OLD / "placement.py", {"successor_handoff": d.handoff})
    fit.p, fit.HERE = p, HERE
    hashes = {n: d.sha(s.encode()) for n, s in own.items()}
    payload.update(
        handoff=stops["fitting"], handoff_text=args.handoff.read_text(),
        handoff_sha256=p.file_sha(args.handoff), placement_source_git=args.source_git,
        owner_git=args.source_git, shared_bytes=decision["shared_future_bytes"],
        shared_files=decision["shared_future_files"], deadline_decision=decision,
        deadline_code_sha256=hashes,
        successor_code_sha256={n: d.sha(git_file(d.PREVIOUS, d.PREVIOUS_DIR + "/" + n))
                               for n in ("launch.py", "contract.py", "remote_adapter.py")},
        accepted_source_correction_proofs=expected, accepted_http499_proofs=http,
        source_correction_policy=c.correction_policy(),
        prior_operational_sources_sha256={"contract.py": c.LEGACY_SHA},
        placement_code_sha256={"placement.py": d.sha(c.derived_placement(originals["placement.py"]).encode()),
                              "remote.py": d.sha(c.derived_portable_remote(originals["remote.py"]).encode()),
                              "remote_adapter.py": d.sha(git_file(d.PREVIOUS, d.PREVIOUS_DIR + "/remote_adapter.py"))})
    return dict(own=own, originals=originals, fit=fit, p=p, original=original, transport=transport,
                program=build_program(parent, originals, own), payload=payload,
                parent=parent, lineage=c, decision=decision, stops=stops)


def transfer(owner, *, arm=None):
    """Single original streaming transfer; secrets remain in the live stdin payload."""
    returning = arm is not None
    if returning:
        d.require(arm in d.ARMS, "Exact return arm")
    receive, send = ("return_receive", "return_send") if returning else ("receive", "send")
    destination, source = ("evc33", "evc27") if returning else ("evc27", "evc33")
    phase = "return-" + arm if returning else "contexts-all27"
    p = owner.placement
    p.once(owner.journal / (phase + "-intent.json"), {
        "source": source, "destination": destination, "arm": arm,
        "automatic_retry": False, "placement_source_git": owner.args.source_git})
    extra = {"return_arm": arm} if returning else {}
    value = {**copy.deepcopy(owner.payload), **extra, "mode": receive, "node": destination}
    with (owner.journal / (phase + "-receiver-stderr")).open("xb") as error:
        process = subprocess.Popen(owner.command(destination, receive), stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE, stderr=error, text=True)
        process.stdin.write(owner.transport.envelope(owner.program, value))
        process.stdin.close()
        process.stdin = None
        while time.time() < owner.deadline:
            if select.select([process.stdout], [], [], 30)[0]:
                break
        else:
            raise RuntimeError("Original receiver deadline; reconcile without retransmission")
        endpoint = json.loads(process.stdout.readline())
        d.require(endpoint["host"] == destination and len(endpoint["token"]) == 64, "Original endpoint required")
        sent = owner.invoke(source, send, phase + "-sender", {**extra, "endpoint": endpoint})
        output, _ = process.communicate(timeout=max(1, min(9000, owner.deadline - time.time())))
    d.require(process.returncode == 0, "Original receiver uncertain; do not retry")
    result = json.loads(output)
    d.require(result == sent, "Independent transfer receipts differ")
    p.once(owner.journal / (phase + "-result.json"), result)
    p.once(HERE / ("actual-owner-" + phase + ".json"), result)
    return result


def run(owner, *, clock=time.time, sleep=time.sleep, audit=HERE):
    """Two-wave launch cutoff; preserve every submitted arm independently until noon."""
    p, audit = owner.placement, Path(audit)
    owner.deadline = d.stamp(d.END)
    d.require((owner.journal / "owner-intent.json").exists(), "Original owner intent required")
    d.require(not any(q.name.endswith(("-intent.json", "-result.json", "-transport.json"))
                      and q.name != "owner-intent.json" for q in owner.journal.iterdir()),
              "Never resume a mutation-bearing controller")
    published, complete, failed, returned, unresolved = set(), {}, {}, {}, {}

    def emit(name, result):
        p.once(owner.journal / (name + ".json"), result)
        p.once(audit / ("actual-" + name + ".json"), result)

    def stage(name):
        allowed = clock() <= d.stamp(d.CUTOFFS[name])
        emit("deadline-" + name, {"stage": name, "allowed": allowed,
             "cutoff_UTC": d.CUTOFFS[name], "training_wave_seconds": 57600,
             "queue_delay_is_not_guaranteed": True})
        return allowed

    def finish(reason):
        result = {"status": "actual_all27_predictive_owner_finished", "reason": reason,
                  "placement_source_git": owner.args.source_git, "science_git": d.SCIENCE,
                  "jobs": owner.payload.get("all_dispatched_jobs", {}), "complete": sorted(complete),
                  "failed": sorted(failed), "privately_verified_epochs": sorted(published),
                  "returns": returned, "unresolved": unresolved, "remote_jobs_cancelled": [],
                  "automatic_restart": False, "clinical_accuracy_or_score": False}
        emit("owner-finished", result)
        return result

    while clock() < d.stamp(d.CUTOFFS["admission"]):
        receipts = owner.original.cache_receipts()
        if receipts is not None:
            owner.payload.update(raw_collected=receipts["raw"], context_collected=receipts["context"])
            break
        sleep(min(owner.args.poll_seconds, max(0, d.stamp(d.CUTOFFS["admission"]) - clock())))
    else:
        return finish("complete_inputs_missing_at_two_wave_admission_cutoff")
    if not stage("admission"):
        return finish("admission_cutoff")
    admitted = owner.invoke("evc33", "admit", "admission-source33")
    d.require(admitted["arms"] == ["B", "C"], "Original source admission required; no33 GPU submission")
    if not stage("transfer"):
        return finish("transfer_cutoff")
    portable = owner.transfer()
    d.require(portable["admitted_arms"] == ["C", "B", "C0"], "All three exact destination bindings required")
    owner.payload["portable"] = portable
    if not stage("dispatch"):
        return finish("dispatch_cutoff")
    try:
        dispatched = owner.invoke("evc27", "dispatch", "dispatch-all27")
    except Exception as error:
        unresolved["dispatch"] = {"error_type": type(error).__name__, "automatic_retry": False}
        emit("dispatch-unresolved", unresolved["dispatch"])
        # This reads only durable original per-arm submission proofs. It never
        # submits a missing arm or treats a missing receipt as proof of failure.
        observed = owner.invoke("evc27", "dispatch_reconcile")
        d.require(observed["status"] == "actual_all27_original_dispatch_reconciled_readonly",
                  "Exact read-only dispatch reconciliation required")
        dispatched = {"status": "actual_predictive_placement_jobs_submitted_once",
                      "source_git": d.SCIENCE, "jobs": observed["jobs"]}
        unresolved["dispatch"]["read_only_reconciliation"] = observed
        emit("dispatch-reconciled", observed)
    jobs = dispatched["jobs"]
    d.require(dispatched["status"] == "actual_predictive_placement_jobs_submitted_once"
              and dispatched["source_git"] == d.SCIENCE and set(jobs) <= set(d.ARMS)
              and (set(jobs) == set(d.ARMS) or "dispatch" in unresolved)
              and len({j["GPU_job"] for j in jobs.values()}) == len(jobs)
              and all(j["node"] == "evc27" and j["arm"] == a and j["source_git"] == d.SCIENCE
                      for a, j in jobs.items()), "Three distinct original node27 jobs required")
    owner.payload["all_dispatched_jobs"] = copy.deepcopy(jobs)
    owner.payload["dispatched"] = copy.deepcopy(dispatched)
    emit("known-dispatches-all27", dispatched)
    if not jobs:
        return finish("uncertain_dispatch_without_any_sealed_job_no_retry")
    observation_incidents, early_saved = {}, set()
    while clock() < owner.deadline:
        arms = {}
        for arm, job in jobs.items():
            key = "observation-" + arm
            try:
                progress = owner.invoke("evc27", "progress", extra={"dispatched": {
                    **dispatched, "jobs": {arm: job}}})
                d.require(set(progress["arms"]) == {arm} and progress["arms"][arm]["GPU_job"] == job["GPU_job"],
                          "Exact single-arm read-only observation required")
                arms.update(progress["arms"])
            except Exception as error:
                if key not in unresolved:
                    observation_incidents[arm] = observation_incidents.get(arm, 0) + 1
                    unresolved[key] = {"error_type": type(error).__name__, "training_failure_inferred": False}
                    emit(f"{key}-unresolved-{observation_incidents[arm]:03d}", unresolved[key])
                continue
            if key in unresolved:
                unresolved.pop(key)
                emit(f"{key}-recovered-{observation_incidents[arm]:03d}", {"new_mutations": False})
        for arm, item in arms.items():
            if item.get("first200") and arm not in early_saved:
                emit("first200-" + arm, {"arm": arm, "GPU_job": item["GPU_job"],
                     "science_git": d.SCIENCE, "first200": item["first200"], "clinical_accuracy_claim": False})
                early_saved.add(arm)
            if owner.original.completed(item["slurm"]) and item["terminal"] and arm not in complete:
                complete[arm] = item
                p.once(audit / ("actual-fixed-final-" + arm + ".json"), {
                    **item["terminal"], "actual_Slurm_terminal": item["slurm"],
                    "stderr_bytes": item["stderr_bytes"], "placement_source_git": owner.args.source_git})
            elif owner.original.terminal(item["slurm"]) and not owner.original.completed(item["slurm"]) and arm not in failed:
                failed[arm] = item
                p.once(audit / ("actual-failure-" + arm + ".json"), item)
        available = {a: v for a, v in arms.items() if "publication-" + a not in unresolved}
        candidate = owner.original.ready_epoch({"arms": available}, published)
        if candidate:
            arm, epoch = candidate
            try:
                actual = owner.invoke("evc27", "publish", f"publish-{arm}-{epoch:03d}",
                                      {"arm": arm, "epoch": epoch, "dispatched": dispatched})
                d.require(actual["status"] == VERIFIED and actual["objects"] == 5
                          and (actual["arm"], actual["epoch"], actual["source_git"]) == (arm, epoch, d.SCIENCE),
                          "Exact original private epoch proof required")
                published.add((arm, epoch))
            except Exception as error:
                unresolved["publication-" + arm] = {"arm": arm, "epoch": epoch, "error_type": type(error).__name__,
                                                   "automatic_retry": False}
                emit("publication-unresolved-" + arm, unresolved["publication-" + arm])
            else:
                continue
        # Return each independently completed arm promptly, retaining the others' jobs.
        for arm in d.ARMS:
            if arm in complete and arm not in returned and "return-" + arm not in unresolved \
                    and all((arm, e) in published for e in range(1, 7)):
                try:
                    returned[arm] = owner.transfer(arm=arm)
                except Exception as error:
                    unresolved["return-" + arm] = {"error_type": type(error).__name__, "automatic_retry": False}
                    emit("return-unresolved-" + arm, unresolved["return-" + arm])
        preservation_resolved = all(
            "publication-" + arm in unresolved or (
                all((arm, e) in published for e in range(1, 7))
                and (arm in returned or "return-" + arm in unresolved))
            for arm in complete)
        if len(complete) + len(failed) == len(jobs) and preservation_resolved:
            return finish("all_original_jobs_terminal")
        sleep(min(owner.args.poll_seconds, max(0, owner.deadline - clock())))
    return finish("preservation_deadline_original_jobs_untouched")


def fitting(args, frozen):
    fit, p = frozen["fit"], frozen["p"]
    journal = ROOT / "artifacts" / HERE.name / "finite-owner"
    journal.mkdir(parents=True, exist_ok=True, mode=0o700)
    p.once(journal / "owner-intent.json", {"pid": os.getpid(), "source_git": args.source_git,
        "science_git": d.SCIENCE, "prior_operational_source_git": d.PREVIOUS,
        "original_owner_source_git": d.PREVIOUS, "original_owner_pid": d.PIDS["fitting"],
        "source_sha256": frozen["payload"]["deadline_code_sha256"],
        "decision_sha256": p.file_sha(args.decision), "decision_path": str(args.decision.resolve().relative_to(ROOT)),
        "handoff_sha256": p.file_sha(args.handoff), "automatic_retry": False,
        "separate_observer_replaced_by_sole_controller_progress": True})
    instance = fit.Owner(args, frozen["original"], frozen["transport"], frozen["program"],
                         frozen["payload"], journal)
    instance.placement = p
    instance.transfer = lambda arm=None: transfer(instance, arm=arm)
    try:
        return run(instance)
    except BaseException as error:
        p.once(journal / "owner-ended.json", {"status": "requires_read_only_reconciliation",
             "error_type": type(error).__name__, "remote_jobs_cancelled": [], "automatic_retry": False})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("fitting", "deployment"))
    parser.add_argument("--source-git", required=True)
    for name in ("decision", "handoff", "deployment-handoff", "observer-handoff"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--deployment-decision", type=Path)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    parser.add_argument("--deadline-utc", default=d.END)
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    os.umask(0o077)
    d.require(args.deadline_utc == d.END and 30 <= args.poll_seconds <= 300, "Finite original cutoff required")
    frozen = freeze(args)
    if args.mode == "fitting":
        fitting(args, frozen)
    else:
        d.require(args.deployment_decision is not None, "Explicit deployment decision required")
        with imports(owner=sys.modules[__name__], contract=d):
            dep = module(frozen["own"]["deployment.py"], "all27_deployment", HERE / "deployment.py")
        dep.run(args, frozen)


if __name__ == "__main__":
    main()
