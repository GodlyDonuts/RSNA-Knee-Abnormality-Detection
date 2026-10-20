"""Finite local full-fitting owner; frozen science, no GPUs before complete inputs.

Start exactly once after committing this owner:
  python3 owner.py --owner-git <40-character owner commit>
The prepared v2 source stays fixed when the checkout advances; only the admitted
canonical-alias metadata correction differs from original b361ee40 science.
This owner never restarts training or retries an uncertain external mutation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import stat
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTROL = Path(__file__).resolve().parent
REL = CONTROL.relative_to(ROOT).as_posix()
SCIENCE = None  # Bound to the actually prepared committed v2 source in freeze().
RUNTIME = "/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python"
MANAGEMENT = "/home/sa305415/rsna-knee/kaggle/management-cp312-v1/bin/python"
COLLECT = ROOT / "initiative/research/predictive-full-cache-alias-recovery-20261020-v3"
ACTIVE = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED", "RESIZING"}
VERIFIED = "actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified"
GIB = 2**30


class ObservationLost(RuntimeError):
    pass


def digest(body):
    return hashlib.sha256(body).hexdigest()


def once(path, value):
    body = json.dumps(value, indent=2, allow_nan=False).encode() + b"\n"
    temporary = path.with_name(path.name + ".pending-" + str(os.getpid()))
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def git_file(commit, name):
    assert re.fullmatch("[0-9a-f]{40}", commit)
    return subprocess.check_output(["git", "show", commit + ":" + name], cwd=ROOT)


def freeze(commit):
    global SCIENCE
    names = [
        REL + "/owner.py",
        REL + "/owner_remote.py.txt",
        "initiative/research/paired-logit-head-gradient-20261018-v1/stdin_transport.py",
    ]
    bodies = {name: git_file(commit, name) for name in names}
    # Only at launch must the invoked owner match. Future checkout changes cannot
    # change the already loaded owner/helpers or rebind science to HEAD.
    assert bodies[names[0]] == Path(__file__).read_bytes(), "Invoke the exact committed owner"
    spec = importlib.util.spec_from_loader("frozen_full_owner_transport", loader=None)
    transport = importlib.util.module_from_spec(spec)
    exec(compile(bodies[names[-1]], names[-1], "exec"), transport.__dict__)
    prepared_path = CONTROL / "actual-cpu-prepared.json"
    prepared = json.loads(prepared_path.read_text())
    SCIENCE = prepared["source_git"]
    assert re.fullmatch("[0-9a-f]{40}", SCIENCE) and str(prepared["CPU_job"]).isdecimal()
    assert SCIENCE != "b361ee40939d875d3adf82d5e04dbf2d7d62b0a0"
    assert prepared["original_preparation_source_git"] == "b361ee40939d875d3adf82d5e04dbf2d7d62b0a0"
    proof = (
        ROOT
        / "initiative/research/predictive-alias-failure-20261020-v1/actual-old-fitting-owner-stopped.json"
    )
    namespace = {}
    exec(
        compile(git_file(SCIENCE, REL + "/alias_provenance.py"), "alias_provenance.py", "exec"),
        namespace,
    )
    namespace["validate_owner_stop"](json.loads(proof.read_text()))
    assert digest(proof.read_bytes()) == prepared["old_owner_stopped_sha256"]
    old_journal = ROOT / "artifacts/predictive-full-fitting-20261020-v1/finite-owner"
    assert {p.name for p in old_journal.iterdir()} == {
        "launcher.json",
        "owner-intent.json",
        "stdout",
        "stderr",
    }
    old_process = subprocess.run(
        ["ps", "-p", "59538", "-o", "command="], capture_output=True, text=True
    )
    assert "predictive-full-fitting-20261020-v1/owner.py" not in old_process.stdout, (
        "Old owner remains live"
    )

    assert prepared["status"] == "actual_predictive_full_CPU_source_parent_initialization_accepted"
    assert prepared["node"] == "evc33" and prepared["physical_gpu_count"] == 2
    assert prepared["new_GPU_reservations"] == 0
    assert len({row["shared_tensor_sha256"] for row in prepared["models"].values()}) == 1
    for name in ("worker.py", "admit_cpu.py", "publish_epoch.py", "alias_provenance.py"):
        suffix = "" if name == "alias_provenance.py" else ".txt"
        assert digest(git_file(SCIENCE, REL + "/" + name + suffix)) == prepared["code_sha256"][name]
    return {
        "owner_git": commit,
        "transport": transport,
        "program": bodies[names[1]].decode(),
        "prepared": prepared,
        "prepared_sha256": digest(prepared_path.read_bytes()),
        "owner_sha256": {name: digest(body) for name, body in bodies.items()},
    }


def cache_receipts():
    files = {
        role: COLLECT / ("actual-owner-" + role + "_collect.json") for role in ("raw", "context")
    }
    if not all(path.is_file() for path in files.values()):
        return None
    result = {role: json.loads(path.read_text()) for role, path in files.items()}
    for role, actual in result.items():
        assert actual["status"] == "actual_original_full_cache_terminal_collected"
        assert actual["role"] == role and actual["studies"] == 3887
        assert actual["target_terminal"] == "COMPLETED" and actual["exit_code"] == "0:0"
        assert actual["existing_cache_stat_bindings_checked"] == 3887
        assert actual["target_job"].isdecimal()
        assert actual["original_actual"]["CPU_job"] == actual["target_job"]
        assert actual["original_actual"]["source_git"] == actual["source_git"]
    context = result["context"]
    assert (
        context["original_actual"]["status"]
        == "actual_missing_only3887_exact_FP32_context_alias_recovery_accepted"
    )
    return result


def token():
    path = ROOT / ".env"
    assert stat.S_IMODE(path.stat().st_mode) == 0o600, "Credential file must remain owner-only"
    assert subprocess.run(["git", "check-ignore", "-q", ".env"], cwd=ROOT).returncode == 0
    for line in path.read_text().splitlines():
        key, separator, value = line.strip().partition("=")
        if separator and key == "HF_TOKEN":
            value = value.strip().strip('"').strip("'")
            assert value.startswith("hf_")
            return value
    raise RuntimeError("Private publication token unavailable")


def transport_call(frozen, args, payload, mode, phase=None, journal=None):
    mutation = mode != "progress"
    call_payload = {"mode": mode, "owner_git": frozen["owner_git"], **payload}
    if mutation:
        assert phase and journal
        # The intent deliberately excludes credentials and private source bodies.
        once(
            journal / (phase + "-intent.json"),
            {
                "phase": phase,
                "mode": mode,
                "owner_git": frozen["owner_git"],
                "science_git": SCIENCE,
                "pid": os.getpid(),
                "automatic_retry": False,
                "program_sha256": digest(frozen["program"].encode()),
                "prepared_sha256": frozen["prepared_sha256"],
            },
        )
    if mode == "publish":
        call_payload["token"] = token()  # stdin/RAM only, never command line or receipt.
    if mutation:
        seconds = 3600 if mode == "publish" else 1200
        remote_args = [
            "srun",
            "--account=skattel",
            "--partition=normal",
            "--qos=normal",
            "--nodelist=evc33",
            "--ntasks=1",
            "--cpus-per-task=1",
            "--gres=none",
            "--mem=8G",
            "--time=" + ("00:45:00" if mode == "publish" else "00:15:00"),
            "--job-name=rsna-predictive-full-" + mode,
            "env",
            "CUDA_VISIBLE_DEVICES=",
            "PYTHONNOUSERSITE=1",
            "PYTHONDONTWRITEBYTECODE=1",
            "OMP_NUM_THREADS=1",
            "MKL_NUM_THREADS=1",
            RUNTIME,
            "-B",
            "-c",
            frozen["transport"].COMMAND,
        ]
    else:
        seconds = 50
        remote_args = [MANAGEMENT, "-B", "-c", frozen["transport"].COMMAND]
    try:
        result = subprocess.run(
            [
                "ssh",
                "-S",
                args.socket,
                "-o",
                "BatchMode=yes",
                "-o",
                "ConnectTimeout=15",
                "newton",
                shlex.join(remote_args),
            ],
            input=frozen["transport"].envelope(frozen["program"], call_payload),
            capture_output=True,
            text=True,
            timeout=seconds,
        )
    except (subprocess.TimeoutExpired, OSError) as error:
        if mutation:
            once(
                journal / (phase + "-transport-failed.json"),
                {
                    "error_type": type(error).__name__,
                    "requires_read_only_reconciliation": True,
                    "automatic_retry": False,
                },
            )
            raise RuntimeError(
                "Uncertain original mutation; inspect original intent without retry"
            ) from None
        raise ObservationLost("Read-only Newton observation unavailable") from None
    finally:
        call_payload.pop("token", None)
    if mutation:
        once(
            journal / (phase + "-transport.json"),
            {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr},
        )
    if result.returncode != 0:
        if mutation:
            raise RuntimeError("Original phase failed or uncertain; no automatic retry")
        if result.returncode == 255:
            raise ObservationLost("Read-only SSH observation lost")
        raise RuntimeError("Read-only original progress validation failed")
    try:
        actual = json.loads(result.stdout)
    except ValueError:
        raise RuntimeError("Unparseable original response; inspect without retry") from None
    if mutation:
        once(journal / (phase + "-result.json"), actual)
        once(CONTROL / ("actual-owner-" + phase + ".json"), actual)
    return actual


def completed(state):
    return (
        state.get("state") == "COMPLETED"
        and state.get("exit_code") == "0:0"
        and state["queue_empty"]
    )


def terminal(state):
    return state.get("state") is not None and state["state"] not in ACTIVE and state["queue_empty"]


def ready_epoch(progress, published):
    """Only actually sealed snapshots can request their first publication."""
    candidates = []
    for arm, item in progress["arms"].items():
        assert arm in ("B", "C", "C0")
        assert type(item["sealed_epochs"]) is int and 0 <= item["sealed_epochs"] <= 6
        for epoch in range(1, item["sealed_epochs"] + 1):
            if (arm, epoch) not in published:
                candidates.append((epoch, arm))
    if candidates:
        epoch, arm = min(candidates)
        return arm, epoch
    return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--owner-git", required=True)
    parser.add_argument("--socket", default="/tmp/rsna-research-20261001-stable.sock")
    parser.add_argument("--deadline-utc", default="2026-10-22T09:00:00+00:00")
    parser.add_argument("--poll-seconds", type=int, default=120)
    args = parser.parse_args()
    assert 30 <= args.poll_seconds <= 300
    deadline = datetime.fromisoformat(args.deadline_utc).astimezone(timezone.utc).timestamp()
    assert time.time() < deadline <= datetime(2026, 10, 22, 12, tzinfo=timezone.utc).timestamp()
    os.umask(0o077)
    frozen = freeze(args.owner_git)
    journal = ROOT / "artifacts" / CONTROL.name / "finite-owner"
    journal.mkdir(parents=True, mode=0o700, exist_ok=True)
    once(
        journal / "owner-intent.json",
        {
            "status": "finite_full_fitting_owner_started",
            "owner_git": args.owner_git,
            "science_git": SCIENCE,
            "pid": os.getpid(),
            "prepared_sha256": frozen["prepared_sha256"],
            "owner_sha256": frozen["owner_sha256"],
            "deadline_utc": args.deadline_utc,
            "automatic_retry": False,
            "no_GPU_until_complete_actual3887_context": True,
            "Kaggle_mutations": False,
        },
    )
    print(
        json.dumps({"status": "waiting_actual_full_input_receipts", "pid": os.getpid()}), flush=True
    )
    payload = {"prepared": frozen["prepared"]}
    try:
        while time.time() < deadline:
            receipts = cache_receipts()
            if receipts is not None:
                payload.update(raw_collected=receipts["raw"], context_collected=receipts["context"])
                break
            time.sleep(args.poll_seconds)
        else:
            raise RuntimeError("Finite deadline reached before complete inputs; no GPU launched")
        admitted = transport_call(frozen, args, payload, "admit", "admission", journal)
        assert (
            admitted["status"]
            == "actual_full3887_predictive_CPU_admission_complete_no_GPU_dispatched"
        )
        dispatched = transport_call(
            frozen, args, {**payload, "admitted": admitted}, "dispatch", "dispatch", journal
        )
        assert dispatched["status"] == "actual_three_predictive_full_jobs_submitted_once"
        payload["dispatched"] = dispatched
        print(json.dumps(dispatched), flush=True)
        published, seen_200, seen_terminal, seen_failure = set(), set(), set(), set()
        outage_since = None
        while time.time() < deadline:
            try:
                progress = transport_call(frozen, args, payload, "progress")
                outage_since = None
            except ObservationLost:
                outage_since = outage_since or time.monotonic()
                if time.monotonic() - outage_since > 1800:
                    raise RuntimeError(
                        "Observation lost for30minutes; preserve running jobs, reconcile owner"
                    ) from None
                time.sleep(args.poll_seconds)
                continue
            for arm, item in progress["arms"].items():
                if item["first200"] and arm not in seen_200:
                    once(
                        CONTROL / ("actual-first200-" + arm + ".json"),
                        {
                            "arm": arm,
                            "GPU_job": item["GPU_job"],
                            "source_git": SCIENCE,
                            "status": "actual_first200_production_updates_observed",
                            **item["first200"],
                        },
                    )
                    seen_200.add(arm)
                    print(
                        json.dumps(
                            {
                                "status": "actual_first200_production_updates_observed",
                                "arm": arm,
                                "maxima": item["first200"]["maxima"],
                            }
                        ),
                        flush=True,
                    )
                if completed(item["slurm"]) and item["terminal"] and arm not in seen_terminal:
                    once(
                        CONTROL / ("actual-fixed-final-" + arm + ".json"),
                        {
                            **item["terminal"],
                            "actual_Slurm_terminal": item["slurm"],
                            "stderr_bytes": item["stderr_bytes"],
                        },
                    )
                    seen_terminal.add(arm)
                    print(
                        json.dumps({"status": "actual_full_arm_completed", "arm": arm}), flush=True
                    )
                if (
                    terminal(item["slurm"])
                    and not completed(item["slurm"])
                    and arm not in seen_failure
                ):
                    once(CONTROL / ("actual-failure-" + arm + ".json"), item)
                    seen_failure.add(arm)
                    print(
                        json.dumps(
                            {
                                "status": "actual_original_arm_failed_no_restart",
                                "arm": arm,
                                "Slurm": item["slurm"],
                            }
                        ),
                        flush=True,
                    )
            candidate = ready_epoch(progress, published)
            if candidate:
                arm, epoch = candidate
                phase = "publish-" + arm + "-" + str(epoch).zfill(3)
                actual = transport_call(
                    frozen, args, {**payload, "arm": arm, "epoch": epoch}, "publish", phase, journal
                )
                assert actual["status"] == VERIFIED and actual["objects"] == 5
                assert (actual["arm"], actual["epoch"], actual["source_git"]) == (
                    arm,
                    epoch,
                    SCIENCE,
                )
                published.add((arm, epoch))
                print(
                    json.dumps(
                        {
                            "status": VERIFIED,
                            "arm": arm,
                            "epoch": epoch,
                            "revision": actual["revision"],
                        }
                    ),
                    flush=True,
                )
                continue
            if all(terminal(item["slurm"]) for item in progress["arms"].values()):
                assert seen_terminal | seen_failure == {"B", "C", "C0"}, (
                    "Missing actual terminal receipt"
                )
                result = {
                    "status": "actual_full_fitting_owner_finished",
                    "science_git": SCIENCE,
                    "jobs": dispatched["jobs"],
                    "fixed_final_arms": sorted(seen_terminal),
                    "failed_arms": sorted(seen_failure),
                    "privately_verified_epochs": [
                        {"arm": arm, "epoch": epoch} for arm, epoch in sorted(published)
                    ],
                    "clinical_accuracy_or_Kaggle_score": False,
                    "automatic_restart": False,
                }
                once(journal / "owner-result.json", result)
                once(CONTROL / "actual-owner-finished.json", result)
                print(json.dumps(result), flush=True)
                return
            time.sleep(args.poll_seconds)
        raise RuntimeError(
            "Finite owner deadline reached; originals remain running, reconcile read-only"
        )
    except BaseException as error:
        once(
            journal / "owner-ended.json",
            {
                "status": "owner_ended_requires_read_only_reconciliation",
                "error_type": type(error).__name__,
                "reason": str(error),
                "remote_jobs_cancelled_or_restarted": False,
                "automatic_retry": False,
            },
        )
        raise


if __name__ == "__main__":
    main()
