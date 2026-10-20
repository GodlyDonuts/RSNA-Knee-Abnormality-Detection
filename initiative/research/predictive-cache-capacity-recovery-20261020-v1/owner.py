"""Distinct finite v4 owner; reuses immutable v3 transport, never its owner journal.

No action on import. The parent must freeze this source, obtain actual mixed-prefix
admission and accept capacity funding before invoking the explicit CLI.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONTROL = Path(__file__).resolve().parent
REL = CONTROL.relative_to(ROOT).as_posix()
PREVIOUS = "91224c242c5758fcb25b71c27d0a0b920458ee6b"
PREVIOUS_REL = "initiative/research/predictive-full-cache-alias-recovery-20261020-v3"
ROOTS = {
    "raw": "/tmp/rsna-predictive-full-raw-capacity-recovery-20261020-v1",
    "context": "/tmp/rsna-predictive-full-context-capacity-recovery-20261020-v1",
}
GIB, MIB = 2**30, 2**20
FUNDING = {
    "other_node_future_bytes": 62 * GIB + 594 * MIB,
    "other_shared_future_bytes": 121 * MIB,
    "other_shared_future_files": 373,
    "owner_total_shared_future_bytes": 123 * MIB,
    "owner_total_shared_future_files": 376,
    "full_controls_node_only_bytes": 16 * MIB,
    "no_deletion_credit": True,
}


def sha(body):
    return hashlib.sha256(body).hexdigest()


def git_file(commit, path):
    return subprocess.check_output(["git", "show", commit + ":" + path], cwd=ROOT).decode()


def base_module():
    source = git_file(PREVIOUS, PREVIOUS_REL + "/owner.py")
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader("original_v3_owner", loader=None)
    )
    module.__file__ = str(ROOT / PREVIOUS_REL / "owner.py")
    exec(compile(source, "unchanged_v3_owner_library", "exec"), module.__dict__)
    return module


def freeze(
    commit,
    regression_body,
    regression_sha256,
    prior_body,
    prior_sha256,
    prefix_body,
    prefix_sha256,
    *,
    capacity_body,
    capacity_sha256,
    mixed_body=None,
    mixed_sha256=None,
):
    """Exact v3 numerical/transport bundle plus this distinct operational successor."""
    old = base_module()
    bundle = old.freeze(
        PREVIOUS,
        regression_body,
        regression_sha256,
        prior_body,
        prior_sha256,
        prefix_body,
        prefix_sha256,
    )
    names = [
        "owner.py",
        "suffix.py",
        "program.py.txt",
        "context.py.txt",
        "collect.py.txt",
        "admit.py.txt",
    ]
    current = {name: git_file(commit, REL + "/" + name) for name in names}
    for name, source in current.items():
        assert (CONTROL / name).read_bytes() == source.encode(), (
            "New launch controls must be frozen"
        )
    assert sha(capacity_body) == capacity_sha256
    decision = json.loads(capacity_body)
    assert decision["status"] == "accepted_original_v3_capacity_failure_reconciled"
    assert decision["original_jobs"] == {"raw": "979664", "context": "979665"}
    assert decision["lane_count"] == 4 and decision["scratch_cap_bytes"] == 6 * GIB
    assert decision["funding"] == FUNDING
    if mixed_body is not None:
        assert sha(mixed_body) == mixed_sha256
        receipt = json.loads(mixed_body)
        assert receipt["suffix_reconcile_sha256"] == sha(current["suffix.py"].encode())
    for payload in bundle["payloads"].values():
        payload.update(
            {
                "source_git": commit,
                "funding": FUNDING,
                "capacity_decision_body": capacity_body.decode(),
                "capacity_decision_sha256": capacity_sha256,
                "mixed_prefix_admission": None if mixed_body is None else mixed_body.decode(),
                "mixed_prefix_admission_sha256": mixed_sha256,
                "suffix_reconcile": current["suffix.py"],
                "suffix_reconcile_sha256": sha(current["suffix.py"].encode()),
                "lane_count": 4,
                "scratch_cap_bytes": 6 * GIB,
                "capacity_wait_seconds": 1800,
            }
        )
    bundle["payloads"]["raw"]["context_module_sha256"] = bundle["payloads"]["context"][
        "module_sha256"
    ]
    bundle["commit"] = commit
    bundle["programs"] = {
        key: current[name]
        for key, name in (
            ("raw", "program.py.txt"),
            ("context", "context.py.txt"),
            ("collect", "collect.py.txt"),
            ("admit", "admit.py.txt"),
        )
    }
    bundle["sha256"].update(
        {REL + "/" + name: sha(body.encode()) for name, body in current.items()}
    )
    return bundle


def next_action(jobs, phases):
    """Only both exact original failures permit one new pair; never retry an intent."""
    for job in ("979664", "979665"):
        state = jobs[job]
        if (
            state["state"] is None
            or not state["queue_empty"]
            or state["state"] in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"}
        ):
            return ("wait",)
        assert state["state"] == "FAILED" and state["exit_code"] == "1:0"
    for phase in phases.values():
        assert phase.get("returncode") in (None, 0) and not phase.get("input_error"), (
            "Uncertain phase must not repeat"
        )
    if "raw" not in phases:
        return ("dispatch", "raw")
    for role in ("raw", "context"):
        if role not in phases:
            continue
        phase = phases[role]
        state = jobs.get(phase.get("job_id"))
        if state is None or state["state"] is None:
            continue
        if state["state"] not in {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "SUSPENDED"}:
            if not state["queue_empty"]:
                return ("wait",)
            assert state["state"] == "COMPLETED" and state["exit_code"] == "0:0", (
                "New failure requires separate review"
            )
            if phase.get("returncode") == 0 and role + "_collect" not in phases:
                return ("collect", role, phase["job_id"], role)
    producer = jobs.get(phases["raw"].get("job_id"), {})
    if (
        "context" not in phases
        and phases["raw"].get("ready")
        and producer.get("state") in {"RUNNING", "COMPLETING", "COMPLETED"}
    ):
        return ("dispatch", "context")
    if all(phases.get(role + "_collect", {}).get("accepted") for role in ("raw", "context")):
        return ("complete",)
    return ("wait",)


def command(role, transport):
    cpu, memory, duration = (8, "96G", "20:00:00") if role == "raw" else (1, "12G", "20:00:00")
    if role == "collect":
        cpu, memory, duration = 1, "1G", "00:03:00"
    return [
        "srun",
        "--account=skattel",
        "--partition=normal",
        "--qos=normal",
        "--nodelist=evc33",
        "--ntasks=1",
        f"--cpus-per-task={cpu}",
        "--gres=none",
        "--mem=" + memory,
        "--time=" + duration,
        "--immediate=60",
        "--job-name=rsna-predictive-" + role + "-capacity-recovery-v1",
        "env",
        "CUDA_VISIBLE_DEVICES=",
        "PYTHONNOUSERSITE=1",
        "PYTHONDONTWRITEBYTECODE=1",
        "OMP_NUM_THREADS=" + ("2" if role == "raw" else "1"),
        "/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python",
        "-B",
        "-c",
        transport.COMMAND,
    ]


def main():
    # Additional receipt args are consumed here; the unchanged original finite
    # loop receives its original CLI and a fully frozen, already admitted bundle.
    parser = argparse.ArgumentParser(add_help=False)
    for role in ("capacity-decision", "mixed-prefix-admission"):
        parser.add_argument("--" + role, type=Path, required=True)
        parser.add_argument("--" + role + "-sha256", required=True)
    extra, remainder = parser.parse_known_args()
    old = base_module()
    original_source = git_file(PREVIOUS, PREVIOUS_REL + "/owner.py")
    # Source substitutions affect only original jobs, unique journal and finite
    # observation loop. Definitions below replace dispatch/transition/freezing.
    old_main = original_source[
        original_source.index("def main():") : original_source.index(
            '\n\nif __name__ == "__main__":'
        )
    ]
    old_main = old_main.replace(
        "artifacts/predictive-full-cache-alias-recovery-20261020-v3",
        "artifacts/predictive-cache-capacity-recovery-20261020-v1",
    )
    exec(compile(old_main, "distinct_capacity_owner_loop", "exec"), old.__dict__)
    old.CONTROL = CONTROL
    old.ORIGINALS = {"raw": "979664", "context": "979665"}
    old.REMOTE_ROOTS = ROOTS
    old.FUNDING = FUNDING
    old.command = command
    old.next_action = next_action
    old.freeze = lambda *args: freeze(
        *args,
        capacity_body=extra.capacity_decision.read_bytes(),
        capacity_sha256=extra.capacity_decision_sha256,
        mixed_body=extra.mixed_prefix_admission.read_bytes(),
        mixed_sha256=extra.mixed_prefix_admission_sha256,
    )
    sys.argv = [sys.argv[0], *remainder]
    old.main()


if __name__ == "__main__":
    main()
