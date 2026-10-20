"""Finite, single-owner read-only observation of the original C100 pilot."""

import argparse
import importlib.util
import json
import os
import re
import shlex
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SPEC = importlib.util.spec_from_file_location("C100_observer_contract", HERE / "contract.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)
MANAGEMENT = "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python"
COMMAND = (
    "import json,sys; e=json.load(sys.stdin); "
    "exec(compile(e['program'],'frozen_C100_readonly_observer','exec'),{'__name__':'__main__'})"
)
POLL_PROGRAM = """
import json, subprocess
accounting = subprocess.check_output(
    ['sacct','-X','-n','-P','-j','981620','--format=JobIDRaw,State,ExitCode,NodeList'], text=True)
queue = subprocess.check_output(['squeue','-h','-u','sa305415','-o','%i|%T'], text=True)
print(json.dumps({'accounting':accounting,'queue':queue}), flush=True)
"""


class ObservationLost(RuntimeError):
    """Only a read-only scheduler poll lost its transport."""


class AllocationUnavailable(RuntimeError):
    """The immediate CPU request explicitly allocated no job."""


def utc():
    return datetime.now(timezone.utc)


def save(path, value, *, replace=False):
    body = value if isinstance(value, bytes) else (
        json.dumps(value, sort_keys=True, allow_nan=False) + "\n").encode()
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as stream:
        stream.write(body)
        stream.flush()
        os.fsync(stream.fileno())
    try:
        if replace:
            os.replace(pending, path)
        else:
            os.link(pending, path)
    finally:
        pending.unlink(missing_ok=True)


def freeze(pin):
    C.require(re.fullmatch("[0-9a-f]{40}", pin), "Exact observer commit required")
    files = {}
    for name in ("contract.py", "collect.py", "observe.py"):
        relative = (HERE / name).relative_to(REPO).as_posix()
        body = subprocess.check_output(["git", "show", pin + ":" + relative], cwd=REPO)
        C.require(body == (HERE / name).read_bytes(), "Observer differs from frozen source")
        files[name] = body.decode()
    return files


def capsule(files):
    return (
        "import sys,types\n"
        "module=types.ModuleType('observer_contract')\n"
        f"exec(compile({files['contract.py']!r},'frozen_observer_contract','exec'),module.__dict__)\n"
        "sys.modules['observer_contract']=module\n"
        f"exec(compile({files['collect.py']!r},'frozen_C100_terminal_reader','exec'),"
        "{'__name__':'__main__'})\n"
    )


def request(socket, program, *, collect=False):
    command = ["env", "TZ=UTC", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1",
               MANAGEMENT, "-B", "-c", COMMAND]
    if collect:
        command = [
            "srun", "--account=skattel", "--partition=normal", "--qos=normal",
            "--nodelist=evc27", "--ntasks=1", "--cpus-per-task=1", "--gres=none",
            "--mem=1G", "--time=00:03:00", "--immediate=60",
            "--job-name=rsna-C100-terminal-read", "env", "CUDA_VISIBLE_DEVICES=",
            *command,
        ]
    try:
        result = subprocess.run(
            ["ssh", "-S", socket, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
             "newton", shlex.join(command)],
            input=json.dumps({"program": program}), capture_output=True, text=True,
            timeout=300 if collect else 40, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        if not collect:
            raise ObservationLost("Read-only scheduler timeout") from exc
        raise RuntimeError("CPU collection outcome uncertain; reconcile without retry") from exc
    if result.returncode == 255 and not collect:
        raise ObservationLost("Read-only scheduler SSH observation lost")
    # No accepted allocation or output: this exact immediate-admission failure is retryable.
    if (collect and result.returncode == 1 and not result.stdout.strip()
            and any(message in result.stderr for message in (
                "Unable to allocate resources: Requested nodes are busy",
                "Unable to allocate resources: Resources temporarily unavailable",
                "Unable to allocate resources: Immediate execution impossible"))):
        raise AllocationUnavailable("Immediate read-only CPU unavailable; no allocation")
    C.require(result.returncode == 0, "Original observation failed; reconcile without retry")
    C.require(len(result.stdout.encode()) <= 2**20, "Bounded sanitized observation required")
    return json.loads(result.stdout)


def collected(value):
    C.require(
        value["status"] == "actual_C100_original_terminal_read_only_collected"
        and value["GPU_job"] == C.JOB and value["node"] == C.NODE
        and value["source_git"] == C.SOURCE and value["binding_sha256"] == C.BINDING
        and str(value["CPU_job"]).isdigit() and value["GPU_allocated"] is False
        and value["original_files_written"] == 0 and value["automatic_promotion"] is False,
        "Exact original terminal collection required",
    )
    state = value["scheduler"]
    C.require(state["GPU_job"] == C.JOB and state["state"] in C.TERMINAL
              and state["live_queue_empty"], "Original terminal not sealed")
    body = value.get("acceptance_body")
    if state["state"] == "COMPLETED" and state["exit_code"] == "0:0":
        C.require(isinstance(body, str), "Original success needs its exact acceptance")
        raw = body.encode()
        C.require(C.sha(raw) == value["acceptance_sha256"]
                  and len(raw) == value["acceptance_bytes"], "Acceptance bytes differ")
        C.acceptance(json.loads(raw))
        return raw
    C.require(body is None, "Terminal failure cannot export a success acceptance")
    return None


def run(*, source, files, submit_sha, journal, audit, socket, poll=120,
        call=request, now=utc, sleep=time.sleep):
    deadline = datetime.fromisoformat(C.DEADLINE)
    lost_since = None
    allocation_since = None
    attempt = 0
    last_state = None
    provenance = {"observer_source_git": source, "original_submit_sha256": submit_sha,
                  "GPU_job": C.JOB, "original_source_git": C.SOURCE,
                  "binding_sha256": C.BINDING, "deadline_UTC": C.DEADLINE}
    while now() < deadline:
        try:
            value = call(socket, POLL_PROGRAM)
            state = C.scheduler(value["accounting"], value["queue"])
        except ObservationLost:
            lost_since = lost_since or now()
            C.require((now() - lost_since).total_seconds() < 1800,
                      "Read-only scheduler observation lost for 30 minutes")
            sleep(min(poll, max(0, (deadline - now()).total_seconds())))
            continue
        lost_since = None
        observed = {**provenance, "observed_UTC": now().isoformat(), "scheduler": state}
        save(journal / "latest-state.json", observed, replace=True)
        identity = (state["state"], state["exit_code"], state["live_queue_empty"])
        if identity != last_state:
            print(json.dumps({"GPU_job": C.JOB, "state": state["state"],
                              "live_queue_empty": state["live_queue_empty"]}), flush=True)
            last_state = identity
        if state["state"] in C.TERMINAL and state["live_queue_empty"]:
            attempt += 1
            save(journal / f"collection-{attempt:02d}-intent.json",
                 {**observed, "maximum_CPU": 1, "maximum_GPU": 0,
                  "no_retry_after_uncertain_allocation": True})
            try:
                result = call(socket, capsule(files), collect=True)
            except AllocationUnavailable:
                allocation_since = allocation_since or now()
                save(journal / f"collection-{attempt:02d}-not-allocated.json",
                     {"status": "immediate_CPU_unavailable_no_allocation",
                      "observed_UTC": now().isoformat()})
                C.require((now() - allocation_since).total_seconds() < 1800,
                          "Terminal CPU collection unavailable for 30 minutes")
            else:
                body = collected(result)
                save(journal / f"collection-{attempt:02d}-result.json", result)
                summary = {k: v for k, v in result.items() if k != "acceptance_body"}
                summary.update(provenance, collected_UTC=now().isoformat())
                if body is not None:
                    save(audit / "actual-accepted.json", body)
                save(audit / ("actual-terminal.json" if body else "actual-terminal-failure.json"),
                     summary)
                return {"status": "original_terminal_collected",
                        "original_terminal_state": result["scheduler"]["state"],
                        "acceptance_collected": body is not None, **provenance}
        sleep(min(poll, max(0, (deadline - now()).total_seconds())))
    return {"status": "finite_observer_deadline_reached_without_collection", **provenance,
            "job_failure_inferred": False, "original_job_changed": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    C.require(30 <= args.poll_seconds <= 300, "Bounded observation cadence")
    C.require(utc() < datetime.fromisoformat(C.DEADLINE), "Original observer deadline passed")
    os.umask(0o077)
    files = freeze(args.source_git)
    submit_body = (HERE.parent / "predictive-density100-pilot-20261020-v1"
                   / "actual-submit.json").read_bytes()
    C.submitted(json.loads(submit_body))
    journal = REPO / "artifacts" / HERE.name
    journal.mkdir(mode=0o700, exist_ok=False)
    save(journal / "intent.json", {
        "source_git": args.source_git, "PID": os.getpid(), "GPU_job": C.JOB,
        "original_source_git": C.SOURCE, "original_submit_sha256": C.sha(submit_body),
        "source_sha256": {name: C.sha(body.encode()) for name, body in files.items()},
        "deadline_UTC": C.DEADLINE, "read_only": True, "maximum_GPU_allocations": 0,
    })
    try:
        result = run(source=args.source_git, files=files, submit_sha=C.sha(submit_body),
                     journal=journal, audit=HERE, socket=args.socket, poll=args.poll_seconds)
    except Exception as exc:
        save(journal / "result.json", {"status": "observer_stopped_requires_reconciliation",
             "error_type": type(exc).__name__, "GPU_job": C.JOB,
             "original_job_changed": False, "automatic_retry": False})
        raise
    save(journal / "result.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
