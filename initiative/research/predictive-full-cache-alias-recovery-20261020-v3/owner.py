"""Finite local owner for one reviewed, missing-only alias-failure recovery.

Preparation is inert. Explicit execution creates one local owner intent; remote
workers are launched only after exact original terminal observations. An
uncertain launch is never retried and healthy dispatched jobs are never killed.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import subprocess
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTROL = Path(__file__).resolve().parent
REL = CONTROL.relative_to(ROOT).as_posix()
RAW_SOURCE = "c78c94d16670f2f5d18bd747db313725c3dfe220"
CONTEXT_SOURCE = "441c4aa1b3dae87c163d6e6b6e9669cf19c7ad39"
ROSTER = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"
ORIGINALS = {"raw": "978318", "context": "978414"}
REMOTE_ROOTS = {
    "raw_original": "/tmp/rsna-predictive-full-raw-cache-20261020-v1",
    "context_original": "/tmp/rsna-predictive-full-context-cache-20261020-v1",
    "raw": "/tmp/rsna-predictive-full-raw-alias-recovery-20261020-v3",
    "context": "/tmp/rsna-predictive-full-context-alias-recovery-20261020-v3",
}
ACTIVE = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"}
CONTEXT_TERMINAL = {"FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL"}
RUNTIME = "/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python"
MANAGEMENT = "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python"
GIB, MIB = 2**30, 2**20
MAX_SECONDS = 30 * 3600
OUTAGE_SECONDS = 30 * 60
FUNDING = {
    "other_node_future_bytes": 62 * GIB + 594 * MIB,
    # Reassign the prior unattempted successor allowances within 121 MiB/376
    # files. Each worker adds its own 2 MiB/3 files; the peer remains funded.
    "other_shared_future_bytes": 119 * MIB,
    "other_shared_future_files": 373,
    "owner_total_shared_future_bytes": 121 * MIB,
    "owner_total_shared_future_files": 376,
    "full_controls_node_only_bytes": 16 * MIB,
    "no_deletion_credit": True,
}


class ObservationLost(RuntimeError):
    """Only a read-only transport failure may be waited through."""


def digest(data):
    return hashlib.sha256(data).hexdigest()


def once(path, value):
    """Publish a complete local receipt exclusively; never replace an original."""
    body = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    temporary = path.with_name(path.name + ".sealed-" + str(os.getpid()))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def git_file(commit, path):
    return subprocess.check_output(["git", "show", commit + ":" + path], cwd=ROOT).decode()


def original_modules(commit):
    prefix = "rsna/competitive/predictive/"
    names = subprocess.check_output(
        ["git", "ls-tree", "--name-only", commit + ":" + prefix.rstrip("/")], cwd=ROOT
    ).decode().splitlines()
    files = {
        prefix + name: git_file(commit, prefix + name) for name in names if name.endswith(".py")
    }
    assert {"inputs.py", "cases.py", "supports.py", "cache.py"} <= set(names)
    # These two empty package controls are exactly how both original launchers
    # materialized their frozen code, rather than today's package bodies.
    files.update({"rsna/__init__.py": "", "rsna/competitive/__init__.py": ""})
    return files


def freeze(commit, regression_body, regression_sha256, prior_body, prior_sha256,
           prefix_body, prefix_sha256):
    assert re.fullmatch(r"[0-9a-f]{40}", commit)
    current_names = [
        REL + "/" + name
        for name in ("owner.py", "program.py.txt", "context.py.txt", "reconcile.py", "collect.py.txt", "contract.py")
    ] + [
        "rsna/competitive/predictive/source_resume.py",
        "rsna/competitive/predictive/cases.py",
        "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py",
    ]
    current = {name: git_file(commit, name) for name in current_names}
    for name, body in current.items():
        assert (ROOT / name).read_bytes() == body.encode(), "Launch source must be committed"
    contract = importlib.util.module_from_spec(importlib.util.spec_from_loader("alias_contract", loader=None))
    exec(current[REL + "/contract.py"], contract.__dict__)
    contract.prior_attempt(prior_body, prior_sha256)
    contract.prefix_admission(prefix_body, prefix_sha256, current[REL + "/reconcile.py"])
    raw = original_modules(RAW_SOURCE)
    original_cases = raw[contract.CASES]
    corrected_cases = current[contract.CASES]
    contract.regression(regression_body, regression_sha256,
                        original_cases=original_cases, corrected_cases=corrected_cases)
    raw[contract.CASES] = corrected_cases
    raw_names = sorted(set(raw) - {"rsna/__init__.py", "rsna/competitive/__init__.py"})
    raw["rsna/competitive/predictive/source_resume.py"] = current[
        "rsna/competitive/predictive/source_resume.py"
    ]
    controls = {
        "original_program": git_file(
            RAW_SOURCE, "initiative/research/predictive-full-raw-cache-20261020-v1/program.py.txt"
        ),
        "raw": git_file(
            RAW_SOURCE, "initiative/research/predictive-raw-input-pilot-20261020-v1/program.py.txt"
        ),
        "conversion": git_file(
            RAW_SOURCE, "initiative/research/predictive-raw16-cache-20261020-v1/group_conversion.py.txt"
        ),
        "helper": git_file(
            RAW_SOURCE, "initiative/research/S6-fitting-source-stream-20261018-v1/source_stream.py.txt"
        ),
        "reconcile": current[REL + "/reconcile.py"],
    }
    shared = {
        "source_git": commit,
        "engineering_and_fresh_funding_accepted": True,
        "funding": FUNDING,
        "alias_regression_receipt": regression_body.decode(),
        "alias_regression_receipt_sha256": regression_sha256,
        "original_cases_sha256": digest(original_cases.encode()),
        "corrected_cases_sha256": digest(corrected_cases.encode()),
        "alias_contract": current[REL + "/contract.py"],
        "alias_contract_sha256": digest(current[REL + "/contract.py"].encode()),
        "prior_attempt_receipt": prior_body.decode(),
        "prior_attempt_receipt_sha256": prior_sha256,
        "prefix_admission_receipt": prefix_body.decode(),
        "prefix_admission_receipt_sha256": prefix_sha256,
    }
    raw_payload = {
        **shared, **controls, "files": raw, "original_numerical_files": raw_names,
        "module_sha256": {name: digest(body.encode()) for name, body in raw.items()},
        "control_sha256": {name: digest(body.encode()) for name, body in controls.items()},
    }
    context = original_modules(CONTEXT_SOURCE)
    assert context[contract.CASES] == original_cases
    context[contract.CASES] = corrected_cases
    context_payload = {
        **shared, "files": context, "reconcile": controls["reconcile"],
        "module_sha256": {name: digest(body.encode()) for name, body in context.items()},
        "reconcile_sha256": digest(controls["reconcile"].encode()),
    }
    spec = importlib.util.spec_from_loader("frozen_continuation_stdin", loader=None)
    transport = importlib.util.module_from_spec(spec)
    exec(current[current_names[-1]], transport.__dict__)
    return {
        "commit": commit, "transport": transport,
        "programs": {role: current[REL + "/" + name] for role, name in (
            ("raw", "program.py.txt"), ("context", "context.py.txt"), ("collect", "collect.py.txt")
        )},
        "payloads": {"raw": raw_payload, "context": context_payload},
        "sha256": {name: digest(body.encode()) for name, body in current.items()},
    }


def parse_jobs(accounting, queue, requested):
    rows = [line.split("|") for line in accounting.splitlines() if line.strip()]
    queued = {line.split("|")[0] for line in queue.splitlines() if line.strip()}
    result = {}
    for job in requested:
        matches = [row for row in rows if row[0] == job]
        if not matches:
            result[job] = {"state": None, "queue_empty": job not in queued}
            continue
        assert len(matches) == 1 and len(matches[0]) >= 3
        _, state, exit_code, *_ = matches[0]
        assert re.fullmatch(r"\d+:\d+", exit_code)
        result[job] = {
            "state": state, "exit_code": exit_code, "queue_empty": job not in queued,
        }
    return result


def terminal(record, allowed):
    return record["state"] in allowed and record["queue_empty"]


def completed(record):
    return terminal(record, {"COMPLETED"}) and record.get("exit_code") == "0:0"


def next_action(jobs, phases):
    """Pure transition policy; absence, stale queue, and uncertainty never launch."""
    for previous_job in ("979573", "979604"):
        prior = jobs[previous_job]
        if prior["state"] is None or prior["state"] in ACTIVE or not prior["queue_empty"]:
            return ("wait",)
        if prior["state"] != "FAILED" or prior.get("exit_code") != "1:0":
            raise RuntimeError("Prior recovery requires exact failed pre-output reconciliation")
    raw, context = (jobs[ORIGINALS[role]] for role in ("raw", "context"))
    for phase in phases.values():
        if phase.get("returncode") not in (None, 0) or phase.get("input_error"):
            raise RuntimeError("Original attempt is uncertain or failed; no automatic retry")
    if raw["state"] is None or raw["state"] in ACTIVE or not raw["queue_empty"]:
        return ("wait",)
    if completed(raw):
        assert "raw" not in phases, "Completed original cannot coexist with a continuation"
        for role, record in (("raw", raw), ("context", context)):
            if completed(record) and role + "_collect" not in phases:
                return ("collect", role, ORIGINALS[role], role + "_original")
        if not context["queue_empty"]:
            return ("wait",)
        if context["state"] is not None and context["state"] not in ACTIVE and not completed(context):
            raise RuntimeError("Original context failed after complete raw; separate recovery required")
    elif terminal(raw, {"FAILED"}) and raw.get("exit_code") == "1:0":
        if "raw" not in phases:
            return ("dispatch", "raw")
        raw_phase = phases["raw"]
        producer = jobs.get(raw_phase.get("job_id"))
        if producer is None or producer["state"] is None:
            return ("wait",)
        if producer["state"] not in ACTIVE and not producer["queue_empty"]:
            return ("wait",)
        if producer["state"] not in ACTIVE and not completed(producer):
            raise RuntimeError("Original raw successor failed; preserve prefix without retry")
        if completed(producer) and raw_phase.get("returncode") == 0 and "raw_collect" not in phases:
            return ("collect", "raw", raw_phase["job_id"], "raw")
        if completed(context) and "context_collect" not in phases:
            return ("collect", "context", ORIGINALS["context"], "context_original")
        ready = raw_phase.get("ready") and (
            producer["state"] in {"RUNNING", "COMPLETING"} or completed(producer)
        )
        if terminal(context, CONTEXT_TERMINAL) and ready and "context" not in phases:
            return ("dispatch", "context")
        if context["state"] is None or context["state"] in ACTIVE or not context["queue_empty"]:
            return ("wait",)
        if not completed(context) and context["state"] not in CONTEXT_TERMINAL:
            raise RuntimeError("Unaccepted original context terminal state")
        if "context" in phases:
            phase = phases["context"]
            state = jobs.get(phase.get("job_id"))
            if state and state["state"] is not None and state["state"] not in ACTIVE:
                if not state["queue_empty"]:
                    return ("wait",)
                if not completed(state):
                    raise RuntimeError("Original context successor failed; no automatic retry")
                if phase.get("returncode") == 0 and "context_collect" not in phases:
                    return ("collect", "context", phase["job_id"], "context")
    else:
        raise RuntimeError("Only exact original raw FAILED 1:0 permits this reviewed alias correction")
    if all(phases.get(role + "_collect", {}).get("accepted") for role in ("raw", "context")):
        return ("complete",)
    return ("wait",)


OBSERVE = """
import json,subprocess,sys
v=json.load(sys.stdin);jobs=','.join(v['jobs'])
a=subprocess.check_output(['sacct','-X','-n','-P','-j',jobs,'--format=JobIDRaw,State,ExitCode'],text=True,timeout=20)
q=subprocess.check_output(['squeue','-h','-u','sa305415','-o','%i|%T'],text=True,timeout=20)
result={'sacct':a,'squeue':q}
if v.get('quota'):
 result['quota']=subprocess.check_output(['lfs','quota','-u','sa305415','/home/sa305415'],text=True,timeout=20)
print(json.dumps(result))
"""


def ssh_args(socket):
    return ["ssh", "-S", socket, "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", "newton"]


def observe(socket, jobs, transport, *, quota=False):
    try:
        call = subprocess.run(
            [*ssh_args(socket), shlex.join([MANAGEMENT, "-B", "-c", transport.COMMAND])],
            input=transport.envelope(OBSERVE, {"jobs": jobs, "quota": quota}),
            capture_output=True, text=True, timeout=65,
        )
    except subprocess.TimeoutExpired as error:
        raise ObservationLost("Read-only SSH observation timed out") from error
    if call.returncode:
        raise ObservationLost("Read-only SSH observation unavailable")
    response = json.loads(call.stdout)
    return parse_jobs(response["sacct"], response["squeue"], jobs), response.get("quota")


def admit_quota(text):
    fields = next(line.split() for line in text.splitlines() if line.strip().startswith("/home/sa305415"))
    # Exact integers only; quota grace/over-limit markers must not be guessed away.
    free_bytes = (int(fields[2]) - int(fields[1])) * 1024
    free_files = int(fields[6]) - int(fields[5])
    assert free_bytes > 20 * GIB + FUNDING["owner_total_shared_future_bytes"]
    assert free_files > 5000 + FUNDING["owner_total_shared_future_files"]
    return {"actual_free_bytes": free_bytes, "actual_free_files": free_files, "funding": FUNDING}


def wrapped(source, role):
    # Only operational progress reporting surrounds the frozen worker. Numerical
    # bodies and owner-only pixels are never rewritten by this launcher.
    root = REMOTE_ROOTS.get(role)
    prelude = """
import hashlib,json,os,sys,time,threading,traceback
from pathlib import Path
job=os.environ['SLURM_JOB_ID']
print(json.dumps({'owner_event':'allocation_started','CPU_job':job}),flush=True)
stop=threading.Event()
"""
    if root:
        prelude += f"""
def intent_ready():
 p=Path({root!r})/'intent.json'
 while not stop.wait(.2):
  try:
   body=p.read_bytes()
   value=json.loads(body)
   if value.get('CPU_job')==job:
    print(json.dumps({{'owner_event':'original_successor_intent_ready','CPU_job':job,
     'source_git':value['source_git'],'intent_sha256':hashlib.sha256(body).hexdigest()}}),flush=True)
    return
  except (FileNotFoundError,json.JSONDecodeError):pass
threading.Thread(target=intent_ready,daemon=True).start()
"""
    return prelude + f"""
try:
 exec(compile({source!r},'exact_frozen_continuation_worker','exec'),{{'__name__':'__main__'}})
except Exception as error:
 frame=traceback.extract_tb(error.__traceback__)[-1]
 print(json.dumps({{'error_type':type(error).__name__,'line':frame.lineno}}),file=sys.stderr,flush=True)
 raise SystemExit(1)
finally:stop.set()
"""


def command(role, transport):
    cpus, memory, limit = (4, "48G", "20:00:00") if role == "raw" else (1, "12G", "20:00:00")
    if role == "collect":
        cpus, memory, limit = 1, "1G", "00:03:00"
    return [
        "srun", "--account=skattel", "--partition=normal", "--qos=normal",
        "--nodelist=evc33", "--ntasks=1", f"--cpus-per-task={cpus}", "--gres=none",
        "--mem=" + memory, "--time=" + limit, "--immediate=60",
        "--job-name=rsna-predictive-" + role + "-alias-recovery-v3",
        "env", "CUDA_VISIBLE_DEVICES=", "PYTHONNOUSERSITE=1", "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=" + ("2" if role == "raw" else "1"), RUNTIME, "-B", "-c",
        transport.COMMAND,
    ]


def credential():
    env = ROOT / ".env"
    assert env.stat().st_mode & 0o077 == 0
    value = next(
        value.strip().strip("\"'")
        for line in env.read_text().splitlines()
        for key, sep, value in [line.partition("=")]
        if sep and key.strip() == "HF_TOKEN"
    )
    assert value.startswith("hf_")
    return value


def launch(name, role, payload, bundle, journal, socket, quota):
    source = bundle["programs"][role]
    safe_payload = {key: value for key, value in payload.items() if key != "token"}
    once(journal / (name + "-intent.json"), {
        "phase": name, "source_git": bundle["commit"], "program_sha256": digest(source.encode()),
        "payload_without_credential_sha256": digest(json.dumps(safe_payload, sort_keys=True).encode()),
        "funding_checked": quota, "automatic_retry": False, "new_GPU": False,
    })
    stdout = (journal / (name + ".stdout")).open("x", encoding="utf8")
    stderr = (journal / (name + ".stderr")).open("x", encoding="utf8")
    try:
        process = subprocess.Popen(
            [*ssh_args(socket), shlex.join(command(role, bundle["transport"]))],
            stdin=subprocess.PIPE, stdout=stdout, stderr=stderr, text=True,
            start_new_session=True,
        )
    finally:
        stdout.close()
        stderr.close()
    phase = {
        "process": process, "role": role, "returncode": None, "offset": 0,
        "source_git": bundle["commit"],
    }
    once(journal / (name + "-local-process.json"), {
        "PID": process.pid, "source_git": bundle["commit"], "no_remote_cancellation_on_owner_exit": True,
    })

    def send():
        try:
            process.stdin.write(bundle["transport"].envelope(wrapped(source, role), payload))
            process.stdin.close()
        except (OSError, ValueError):
            phase["input_error"] = True

    threading.Thread(target=send, daemon=True).start()
    return phase


def refresh(name, phase, journal, control):
    path = journal / (name + ".stdout")
    assert path.stat().st_size < 2 * MIB and (journal / (name + ".stderr")).stat().st_size < 2 * MIB
    with path.open() as stream:
        stream.seek(phase["offset"])
        while True:
            offset = stream.tell()
            line = stream.readline()
            if not line.endswith("\n"):
                phase["offset"] = offset
                break
            phase["offset"] = stream.tell()
            value = json.loads(line)
            if value.get("owner_event") == "allocation_started":
                assert str(value["CPU_job"]).isdecimal() and "job_id" not in phase
                phase["job_id"] = str(value["CPU_job"])
            elif value.get("owner_event") == "original_successor_intent_ready":
                assert value["CPU_job"] == phase["job_id"]
                assert value["source_git"] == phase["source_git"]
                assert re.fullmatch(r"[0-9a-f]{64}", value["intent_sha256"])
                phase["intent_sha256"] = value["intent_sha256"]
                phase["ready"] = True
            elif value.get("status") == "actual_original_full_cache_terminal_collected":
                assert phase["role"] == "collect" and value["studies"] == 3887
                assert value["fitting_roster_sha256"] == ROSTER
                phase["receipt"] = value
            elif value.get("status", "").endswith("_alias_recovery_accepted"):
                phase["worker_terminal"] = value
                phase["ready"] = True  # A fast completed worker may precede monitor's first read.
    phase["returncode"] = phase["process"].poll()
    if phase["returncode"] == 0 and phase["role"] == "collect" and not phase.get("accepted"):
        assert phase.get("receipt"), "Successful collection must have its exact terminal receipt"
        once(control / ("actual-owner-" + name + ".json"), phase["receipt"])
        phase["accepted"] = True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-git", required=True)
    parser.add_argument("--ssh-socket", default="/tmp/rsna-research-20261001-stable.sock")
    parser.add_argument("--maximum-hours", type=float, default=30)
    parser.add_argument("--regression-receipt", type=Path, required=True)
    parser.add_argument("--regression-sha256", required=True)
    parser.add_argument("--prior-attempt-receipt", type=Path, required=True)
    parser.add_argument("--prior-attempt-sha256", required=True)
    parser.add_argument("--prefix-admission-receipt", type=Path, required=True)
    parser.add_argument("--prefix-admission-sha256", required=True)
    args = parser.parse_args()
    assert 0 < args.maximum_hours <= 30
    os.umask(0o077)
    bundle = freeze(args.source_git, args.regression_receipt.read_bytes(), args.regression_sha256,
                    args.prior_attempt_receipt.read_bytes(), args.prior_attempt_sha256,
                    args.prefix_admission_receipt.read_bytes(), args.prefix_admission_sha256)
    journal = ROOT / "artifacts/predictive-full-cache-alias-recovery-20261020-v3"
    journal.mkdir(mode=0o700, exist_ok=True)
    assert not journal.is_symlink() and journal.stat().st_uid == os.getuid()
    assert not journal.stat().st_mode & 0o077
    deadline = time.monotonic() + args.maximum_hours * 3600
    once(journal / "owner-intent.json", {
        "PID": os.getpid(), "source_git": bundle["commit"], "sources": bundle["sha256"],
        "maximum_seconds": args.maximum_hours * 3600, "maximum_observation_loss_seconds": OUTAGE_SECONDS,
        "original_jobs": ORIGINALS, "funding": FUNDING,
        "alias_regression_receipt_sha256": args.regression_sha256,
        "prior_attempt_receipt_sha256": args.prior_attempt_sha256,
        "prefix_admission_receipt_sha256": args.prefix_admission_sha256,
        "no_replacement_allocation_while_corresponding_original_owner_live": True,
        "automatic_retry": False,
    })
    phases, outage_start = {}, None
    try:
        while time.monotonic() < deadline:
            for name, phase in list(phases.items()):
                refresh(name, phase, journal, CONTROL)
            ids = sorted(set(ORIGINALS.values()) | {"979573", "979604"} | {p["job_id"] for p in phases.values() if "job_id" in p})
            try:
                jobs, _ = observe(args.ssh_socket, ids, bundle["transport"])
            except ObservationLost:
                outage_start = outage_start or time.monotonic()
                if time.monotonic() - outage_start >= OUTAGE_SECONDS:
                    raise RuntimeError("Bounded read-only observation outage; preserve all jobs") from None
                time.sleep(30)
                continue
            action = next_action(jobs, phases)
            if action[0] == "complete":
                once(journal / "owner-result.json", {
                    "status": "complete_original_and_missing_only_cache_chain_collected",
                    "source_git": bundle["commit"], "new_GPU_models_or_competition_entries": 0,
                })
                return
            if action[0] == "wait":
                outage_start = None
                time.sleep(30)
                continue
            assert deadline - time.monotonic() > (
                20 * 3600 + 300 if action[0] == "dispatch" else 300
            ), "Do not launch a worker beyond the finite owner's observation budget"
            # Re-read original identities/queues and quota immediately before intent.
            try:
                fresh, quota_text = observe(args.ssh_socket, ids, bundle["transport"], quota=True)
            except ObservationLost:
                outage_start = outage_start or time.monotonic()
                if time.monotonic() - outage_start >= OUTAGE_SECONDS:
                    raise RuntimeError("Bounded preintent observation outage; no mutation") from None
                time.sleep(30)
                continue
            outage_start = None
            if next_action(fresh, phases) != action:
                continue  # Legitimate progress during a read does not authorize the older action.
            funding = admit_quota(quota_text)
            if action[0] == "dispatch":
                role = action[1]
                payload = {**bundle["payloads"][role]}
                if role == "raw":
                    payload["token"] = credential()
                phases[role] = launch(role, role, payload, bundle, journal, args.ssh_socket, funding)
            else:
                _, role, job, root_key = action
                payload = {
                    "role": role, "target_job": job, "root": REMOTE_ROOTS[root_key],
                    "expected_source_git": (
                        RAW_SOURCE if root_key == "raw_original" else
                        CONTEXT_SOURCE if root_key == "context_original" else bundle["commit"]
                    ),
                    "fitting_roster_sha256": ROSTER,
                }
                name = role + "_collect"
                phases[name] = launch(name, "collect", payload, bundle, journal, args.ssh_socket, funding)
            time.sleep(1)
        raise RuntimeError("Finite local observation budget ended; existing jobs remain untouched")
    except Exception as error:
        once(journal / "owner-stopped.json", {
            "status": "stopped_without_retry_or_remote_cancellation",
            "error_type": type(error).__name__, "source_git": bundle["commit"],
            "attempted_phases": sorted(
                p.name.removesuffix("-intent.json")
                for p in journal.glob("*-intent.json") if p.name != "owner-intent.json"
            ), "known_jobs": {
                name: p.get("job_id") for name, p in phases.items()
            },
        })
        raise SystemExit("Inspect original receipts; no automatic mutation retry") from None


if __name__ == "__main__":
    main()
