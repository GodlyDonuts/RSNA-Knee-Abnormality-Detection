"""Finite read-only observer of the original139802342/v1 status phase.

Uses the exact run_once status payload/transport and cloud program from the
original f0cf07e0 commit. Later checkout changes cannot rebind this notebook.
No prepare/push/abort phase, Dataset mutation or competition submission exists.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shlex
import stat
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTROL = Path(__file__).resolve().parent
REL = CONTROL.relative_to(ROOT).as_posix()
SOURCE = "f0cf07e0d763f82a406c48e5880443582bffec83"
SOURCE_SHA = "5d32589edb9efe7f68de536bc01a1d2e50a0da555e9c5b10734cf4351aafd737"
TERMINAL = {"COMPLETE", "ERROR", "FAILED", "CANCEL_ACKNOWLEDGED"}
SLUG = "godlydonuts/rsna-predictive-t4-v2-5d32589edb9e"


def sha(value):
    return hashlib.sha256(value).hexdigest()


def once(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def git_file(commit, name):
    return subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)


def freeze(owner_git):
    assert len(owner_git) == 40 and set(owner_git) <= set("0123456789abcdef")
    assert git_file(owner_git, REL + "/observe_status.py") == Path(__file__).read_bytes()
    names = [REL + "/" + name for name in ("run_once.py.txt", "cloud.py.txt", "benchmark.py.txt")]
    names += [
        "scripts/kaggle/gpu_capacity.py",
        "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py",
    ]
    originals = {name: git_file(SOURCE, name).decode() for name in names}
    modules = (
        subprocess.check_output(
            ["git", "ls-tree", "--name-only", SOURCE + ":rsna/competitive/predictive"], cwd=ROOT
        )
        .decode()
        .splitlines()
    )
    files = {
        "rsna/competitive/predictive/" + name: git_file(
            SOURCE, "rsna/competitive/predictive/" + name
        ).decode()
        for name in modules
        if name.endswith(".py")
    }
    files.update({"rsna/__init__.py": "", "rsna/competitive/__init__.py": ""})
    prepare = json.loads((CONTROL / "actual-prepare.json").read_text())
    push = json.loads((CONTROL / "actual-push.json").read_text())
    assert prepare["source_git"] == SOURCE and prepare["source_sha256"] == SOURCE_SHA
    assert (push["kernel_id"], push["version"], push["slug"]) == (139802342, 1, SLUG)
    assert push["source_sha256"] == SOURCE_SHA
    assert prepare["binding"]["module_sha256"] == {
        n: sha(body.encode()) for n, body in files.items()
    }
    spec = importlib.util.spec_from_loader("original_predictive_T4_status_transport", loader=None)
    transport = importlib.util.module_from_spec(spec)
    exec(compile(originals[names[-1]], names[-1], "exec"), transport.__dict__)
    program = originals[REL + "/cloud.py.txt"]
    # Identical exception-containment wrapper to original run_once.py.txt.
    safe = (
        "import sys\ntry:\n"
        + "".join("    " + line + "\n" for line in program.splitlines())
        + "except Exception as error:\n    import traceback,json\n"
        + "    frame=traceback.extract_tb(error.__traceback__)[-1]\n"
        + '    print(json.dumps({"error_type":type(error).__name__,"line":frame.lineno}),file=sys.stderr)\n'
        + "    raise SystemExit(1)\n"
    )
    return {
        "transport": transport,
        "program": safe,
        "payload": {
            "phase": "status",
            "source_git": SOURCE,
            "capacity": originals["scripts/kaggle/gpu_capacity.py"],
            "files": files,
            "benchmark": originals[REL + "/benchmark.py.txt"],
        },
        "sha256": {name: sha(body.encode()) for name, body in originals.items()},
    }


def token():
    path = ROOT / ".env"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT).returncode == 0
    for line in path.read_text().splitlines():
        key, separator, value = line.partition("=")
        if separator and key.strip() == "KAGGLE_API_TOKEN":
            return value.strip().strip("\"'")
    raise RuntimeError("Original Kaggle credential unavailable")


def validate(actual):
    assert (actual["kernel_id"], actual["version"], actual["slug"]) == (139802342, 1, SLUG)
    assert actual["source_sha256"] == SOURCE_SHA and actual["new_mutations"] == 0
    return actual


def poll(frozen, socket, timeout):
    assert frozen["payload"]["phase"] == "status"
    payload = {**frozen["payload"], "token": token()}
    args = [
        "srun",
        "--account=skattel",
        "--partition=normal",
        "--qos=normal",
        "--nodelist=evc33",
        "--ntasks=1",
        "--cpus-per-task=1",
        "--gres=none",
        "--mem=1G",
        "--time=00:05:00",
        "--job-name=rsna-predictive-T4-v2-status",
        "env",
        "CUDA_VISIBLE_DEVICES=",
        "PYTHONNOUSERSITE=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python",
        "-B",
        "-c",
        frozen["transport"].COMMAND,
    ]
    try:
        result = subprocess.run(
            [
                "ssh",
                "-S",
                socket,
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                "newton",
                shlex.join(args),
            ],
            input=frozen["transport"].envelope(frozen["program"], payload),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    finally:
        payload.pop("token", None)
    if result.returncode:
        return {
            "observer_read_error": True,
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "new_mutations": 0,
        }
    return validate(json.loads(result.stdout))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    args = parser.parse_args()
    os.umask(0o077)
    frozen = freeze(args.owner_git)
    journal = ROOT / "artifacts" / CONTROL.name
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    once(
        journal / "status-observer-intent.json",
        {
            "status": "finite_read_only_T4_observer_started",
            "PID": os.getpid(),
            "owner_git": args.owner_git,
            "original_source_git": SOURCE,
            "kernel_id": 139802342,
            "version": 1,
            "slug": SLUG,
            "maximum_seconds": 7200,
            "poll_interval_seconds": 90,
            "original_control_sha256": frozen["sha256"],
            "maximum_GPU_pushes": 0,
            "competition_submissions": 0,
            "abort_authority": False,
        },
    )
    deadline, last, first_terminal, previous = time.monotonic() + 7200, None, None, None
    while time.monotonic() < deadline:
        started = time.monotonic()
        stamp = str(time.time_ns())
        try:
            actual = poll(frozen, args.socket, max(1, min(300, deadline - started)))
        except (subprocess.TimeoutExpired, OSError, ValueError, AssertionError) as error:
            actual = {
                "observer_read_error": True,
                "error_type": type(error).__name__,
                "new_mutations": 0,
            }
        once(journal / ("status-observer-" + stamp + ".json"), actual)
        if not actual.get("observer_read_error"):
            last = actual
            state = actual["status"]
            if state != previous or state in TERMINAL:
                once(CONTROL / ("actual-status-observer-" + stamp + ".json"), actual)
                print(json.dumps(actual), flush=True)
            previous = state
            if state in TERMINAL:
                first_terminal = first_terminal or started
                # Terminal file availability can lag status. Allow bounded read-only
                # propagation checks; never interpret missing output as acceptance.
                if (
                    "receipt" in actual and "classification" in actual
                ) or started - first_terminal >= 270:
                    break
        time.sleep(max(0, min(90 - (time.monotonic() - started), deadline - time.monotonic())))
    result = {
        "status": "finite_read_only_T4_observer_finished",
        "original_source_git": SOURCE,
        "kernel_id": 139802342,
        "version": 1,
        "last_actual": last,
        "terminal_observed": bool(last and last["status"] in TERMINAL),
        "retained_measurement_available": bool(last and "receipt" in last),
        "new_GPU_pushes_or_aborts_or_submissions": 0,
    }
    once(journal / "status-observer-result.json", result)
    once(CONTROL / "actual-status-observer-final.json", result)
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
