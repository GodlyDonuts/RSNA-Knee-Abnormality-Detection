"""Finite, explicit C100 phases. Status is read-only; every mutation has one intent."""

import argparse
import importlib.util
import json
import os
import re
import shlex
import stat
import subprocess
import time
from pathlib import Path

import admission as a
import derive

REPO = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
JOURNAL = REPO / "artifacts" / HERE.name
MANAGEMENT = "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python"
ORIGINAL_OWNER = REPO / "initiative/research/predictive-http499-owner-handoff-20261020-v1"
ORIGINAL_JOURNAL = REPO / "artifacts" / ORIGINAL_OWNER.name / "finite-owner"
PLACEMENT_PIN = "2b834507e19c405971228e1bc405462cd370f560"
PLACEMENT_PATH = "initiative/research/predictive-c0-portable-placement-20261020-v1/placement.py"


class ObservationUnavailable(RuntimeError):
    """Only a known read-only SSH/CPU-allocation observation failure."""


def git_file(pin, path):
    a.require(re.fullmatch("[0-9a-f]{40}", pin), "Exact immutable source required")
    return subprocess.check_output(["git", "show", pin + ":" + path], cwd=REPO)


def frozen(args):
    controls = {}
    for name in ("admission.py", "operations.py", "coordinator.py", "derive.py",
                 "density100_contract.py", "telemetry.py"):
        p = HERE / name
        # Optional derivation helpers are admitted only if build() actually uses
        # them. Freeze every one present at the committed source.
        if not p.exists():
            continue
        raw = git_file(args.source_git, p.relative_to(REPO).as_posix())
        a.require(raw == p.read_bytes(), "Invoke exact committed C100 controls")
        controls[name] = raw.decode()
    a.require({"admission.py", "operations.py", "coordinator.py", "derive.py"} <= controls.keys(),
              "Complete singleton coordinator required")
    files = derive.build(REPO)
    placement = git_file(PLACEMENT_PIN, PLACEMENT_PATH).decode()
    payload = {
        "source_git": args.source_git, "files": files,
        "code_sha256": {name: a.sha(body.encode()) for name, body in files.items()},
        "placement_source": placement, "placement_sha256": a.sha(placement.encode()),
    }
    program = (
        "import sys,json,types\n"
        "a=types.ModuleType('admission');sys.modules['admission']=a\n"
        f"exec(compile({controls['admission.py']!r},'admission.py','exec'),a.__dict__)\n"
        "o=types.ModuleType('operations');sys.modules['operations']=o\n"
        f"exec(compile({controls['operations.py']!r},'operations.py','exec'),o.__dict__)\n"
        "v=json.load(sys.stdin)\n"
        "try:\n"
        " result=a.main(v) if v['phase']=='admit' else o.main(v)\n"
        " print(json.dumps(result,allow_nan=False),flush=True)\n"
        "except BaseException as error:\n"
        " tb=error.__traceback__\n"
        " while tb.tb_next:tb=tb.tb_next\n"
        " print(json.dumps({'error_type':type(error).__name__,'site_line':tb.tb_lineno}),"
        "file=sys.stderr,flush=True)\n"
        " raise SystemExit(1)\n"
    )
    path = REPO / "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py"
    spec = importlib.util.spec_from_loader("C100_transport", loader=None)
    transport = importlib.util.module_from_spec(spec)
    exec(compile(git_file(a.BASE, path.relative_to(REPO).as_posix()), str(path), "exec"),
         transport.__dict__)
    return payload, program, transport


def original_jobs():
    jobs = {}
    owner = json.loads((ORIGINAL_JOURNAL / "owner-intent.json").read_bytes())
    a.require(owner["source_git"] == a.PRIMARY_SOURCE and owner["science_git"] == a.BASE
              and owner["pid"] == a.PRIMARY_OWNER_PID, "Exact current primary fitting owner required")
    for role in ("BC", "C0"):
        path = ORIGINAL_OWNER / ("actual-owner-dispatch-" + role + ".json")
        if not path.exists():
            continue
        receipt = json.loads(path.read_text())
        phase = "dispatch-" + role
        result = json.loads((ORIGINAL_JOURNAL / (phase + "-result.json")).read_bytes())
        intent = json.loads((ORIGINAL_JOURNAL / (phase + "-intent.json")).read_bytes())
        node = "evc27" if role == "C0" else "evc33"
        a.require(receipt == result and intent["mode"] == "dispatch" and intent["node"] == node
                  and intent["placement_source_git"] == a.PRIMARY_SOURCE
                  and re.fullmatch("[0-9a-f]{64}", intent["program_sha256"]),
                  "Current primary phase intent/result must match actual dispatch receipt")
        a.require(receipt["status"] == "actual_predictive_placement_jobs_submitted_once"
                  and receipt["source_git"] == a.BASE
                  and set(receipt["jobs"]) == ({"B", "C"} if role == "BC" else {"C0"}),
                  "Only exact original primary dispatch receipts can supply shared-log credit")
        a.validate_primary_jobs(receipt["jobs"], {"B", "C"} if role == "BC" else {"C0"})
        jobs.update(receipt["jobs"])
    return a.validate_primary_jobs(jobs)


def token():
    path = REPO / ".env"
    a.require(stat.S_IMODE(path.stat().st_mode) == 0o600
              and subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=REPO).returncode == 0,
              "Ignored owner-only credentials required")
    for line in path.read_text().splitlines():
        key, separator, value = line.strip().partition("=")
        if key == "HF_TOKEN" and separator:
            value = value.strip().strip("\"").strip("'")
            a.require(value.startswith("hf_"), "Private publication credential unavailable")
            return value
    raise RuntimeError("Private publication credential unavailable")


def call(args, frozen_source, phase, *, decision=None, epoch=None):
    base, program, transport = frozen_source
    payload = {**base, "phase": phase, "original_dispatched_jobs": original_jobs()}
    if decision is not None:
        payload["decision"] = decision
    if epoch is not None:
        payload["epoch"] = epoch
    mutation = phase != "status"
    name = phase + (f"-{epoch:03d}" if epoch is not None else "")
    if mutation:
        JOURNAL.mkdir(parents=True, exist_ok=True, mode=0o700)
        a.once(JOURNAL / (name + "-intent.json"), {
            "phase": phase, "source_git": args.source_git, "epoch": epoch,
            "program_sha256": a.sha(program.encode()), "automatic_retry": False,
            "code_sha256": base["code_sha256"],
            "decision_sha256": a.sha(a.canonical(decision)) if decision is not None else None,
        })
    if phase == "publish":
        payload["token"] = token()
    remote = [
        "srun", "--account=skattel", "--partition=normal", "--qos=normal",
        "--nodelist=evc27", "--ntasks=1", "--cpus-per-task=1", "--gres=none",
        "--mem=" + ("8G" if phase == "publish" else "2G"),
        "--time=" + ("00:45:00" if phase == "publish" else "00:10:00"),
        "--immediate=60", "--job-name=rsna-C100-" + phase,
        "env", "CUDA_VISIBLE_DEVICES=", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=1", "MKL_NUM_THREADS=1",
        a.PYTHON if phase == "publish" else MANAGEMENT, "-B", "-c", transport.COMMAND,
    ]
    try:
        result = subprocess.run(
            ["ssh", "-S", args.socket, "-o", "BatchMode=yes", "-o", "ConnectTimeout=15",
             "newton", shlex.join(remote)],
            input=transport.envelope(program, payload), text=True, capture_output=True,
            timeout=3000 if phase == "publish" else 720, check=False,
        )
    finally:
        payload.pop("token", None)
    if mutation:
        a.once(JOURNAL / (name + "-transport.private.json"), {
            "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr,
        })
    allocation_unavailable = (
        result.returncode == 1
        and re.fullmatch(
            r"srun: error: Unable to allocate resources: "
            r"(?:Resources? temporarily unavailable|Requested nodes are busy)\s*",
            result.stderr.strip(),
        ) is not None
    )
    if phase == "status" and not result.stdout.strip() and (
        result.returncode == 255 or allocation_unavailable
    ):
        raise ObservationUnavailable("Original status observation temporarily unavailable")
    a.require(result.returncode == 0, "Operation failed/uncertain; reconcile read-only, never retry")
    value = json.loads(result.stdout)
    if mutation:
        a.once(JOURNAL / (name + "-result.json"), value)
    return value


def observe(args, frozen_source, deadline):
    """Retry only known read-only observation loss, for at most thirty minutes."""
    unavailable_since = None
    while time.time() < deadline:
        now = time.time()
        if unavailable_since is not None and now >= unavailable_since + 1800:
            raise ObservationUnavailable("Thirty-minute read-only observation window expired")
        try:
            return call(args, frozen_source, "status")
        except ObservationUnavailable:
            if unavailable_since is None:
                unavailable_since = time.time()
            remaining = min(deadline, unavailable_since + 1800) - time.time()
            if remaining > 0:
                time.sleep(min(120, remaining))
    return None  # The absolute preservation deadline never expands on reconnect.


def preserve(args, frozen_source):
    """One finite preservation owner, only after actual singleton full dispatch."""
    a.require((JOURNAL / "dispatch-result.json").is_file(), "Actual original dispatch required")
    a.once(JOURNAL / "preservation-owner-intent.json", {
        "source_git": args.source_git, "automatic_retry": False, "pid": os.getpid(),
        "deadline_UTC": "2026-10-22T12:00:00+00:00",
    })
    deadline = a.datetime(2026, 10, 22, 12, tzinfo=a.timezone.utc).timestamp()
    while time.time() < deadline:
        result = observe(args, frozen_source, deadline)
        if result is None:
            break
        a.require(result.get("dispatch", {}).get("source_git") == args.source_git,
                  "Original C100 dispatch differs")
        for epoch in result["ready_epochs"]:
            prior = result["publications"].get(str(epoch))
            if prior is not None:
                a.require(prior["status"].endswith("private_all_objects_download_size_SHA_verified"),
                          "Uncertain original publication requires reconciliation")
                continue
            if (JOURNAL / f"publish-{epoch:03d}-intent.json").exists():
                raise RuntimeError("Prior publication intent without original proof; no retry")
            call(args, frozen_source, "publish", epoch=epoch)
        state = result["full_scheduler"]["state"]
        if state not in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING"}:
            final = observe(args, frozen_source, deadline)
            if final is None:
                break
            a.once(JOURNAL / "preservation-owner-result.json", final)
            return final
        time.sleep(min(120, max(0, deadline - time.time())))
    result = {"status": "finite_C100_preservation_deadline_original_job_untouched",
              "source_git": args.source_git, "no_remote_jobs_cancelled": True}
    a.once(JOURNAL / "preservation-owner-result.json", result)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    parser.add_argument("--phase", choices=("status", "admit", "dispatch", "publish", "preserve"),
                        required=True)
    parser.add_argument("--decision", type=Path)
    parser.add_argument("--epoch", type=int)
    args = parser.parse_args()
    os.umask(0o077)
    frozen_source = frozen(args)
    if args.phase == "preserve":
        result = preserve(args, frozen_source)
    else:
        decision = json.loads(args.decision.read_text()) if args.decision else None
        a.require(args.phase != "admit" or decision is not None,
                  "Actual explicit pilot-reviewed full-run decision required")
        if args.phase == "admit":
            observed = call(args, frozen_source, "status")
            a.require("pilot" in observed and "portable" in observed
                      and "admitted" not in observed, "Complete prerequisites before mutation intent")
            a.validate_decision(
                decision, observed["pilot"], observed["pilot_sha256"], args.source_git,
                frozen_source[0]["code_sha256"],
            )
            a.require(decision["pilot_scheduler"] == observed["pilot_scheduler"],
                      "Decision must bind this actual immutable terminal scheduler result")
        if args.phase == "publish":
            a.require(type(args.epoch) is int and 1 <= args.epoch <= 6, "Exact sealed epoch required")
        result = call(args, frozen_source, args.phase, decision=decision, epoch=args.epoch)
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
