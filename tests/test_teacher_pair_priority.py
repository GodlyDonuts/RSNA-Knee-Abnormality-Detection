"""Run the actual bounded metadata follower against simulated Slurm receipts."""

import json
import runpy
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

SOURCE = Path(
    "initiative/research/teacher-pair-full-encoder-20261019-v1/follow_original_normal_priority_once.py.txt"
)


@pytest.mark.parametrize("state", ["PENDING", "RUNNING", "FAILED", "uncertain"])
def test_real_follower_amends_only_original_pending_IDs_and_never_retries(
    state, tmp_path, monkeypatch
):
    src = SOURCE.read_text()
    path = tmp_path / SOURCE
    path.parent.mkdir(parents=True)
    path.write_text(src)
    old = tmp_path / "initiative/research/S6-earlier-H100-execution-20261019-v1"
    old.mkdir(parents=True)
    rows = [
        (
            "actual-GPU-submitted.json",
            "11",
            "actual_complete_teacher_oneH100_recovery_submitted_once",
            "rsna-S6-oneH100-recovery",
        ),
        (
            "actual-full-GPU-submitted.json",
            "22",
            "actual_complete_teacher_full3887_two_role_oneH100_fit_submitted_once",
            "rsna-S6-complete-target-full",
        ),
    ]
    for name, job, status, _ in rows:
        (old / name).write_text(json.dumps({"GPU_job": job, "status": status}))
    changed = set()
    calls = []

    def read(args, **kwargs):
        job = "11" if "11" in args[-1] else "22"
        name = next(n for _, j, _, n in rows if j == job)
        qos = "normal" if job in changed else "preemptable"
        current = "PENDING" if state == "uncertain" else state
        return f"UserId=sa305415(1) Account=skattel ReqNodeList=evc48 JobName={name} JobState={current} RunTime=00:00:00 QOS={qos} Partition={qos}"

    def mutate(args, **kwargs):
        calls.append(args[-1])
        job = "11" if "JobId=11" in args[-1] else "22"
        changed.add(job)
        assert (
            "scontrol update" in args[-1]
            and "Partition=normal" in args[-1]
            and "QOS=normal" in args[-1]
        )
        return SimpleNamespace(returncode=1 if state == "uncertain" else 0, stdout="", stderr="")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(subprocess, "check_output", read)
    monkeypatch.setattr(subprocess, "run", mutate)
    if state == "uncertain":
        with pytest.raises(AssertionError, match="read-only reconciliation"):
            runpy.run_path(str(path))
        assert len(calls) == 1
    else:
        runpy.run_path(str(path))
        assert len(calls) == (2 if state == "PENDING" else 0)
    # The same receipt cannot cause a second owner or second metadata mutation.
    with pytest.raises(AssertionError):
        runpy.run_path(str(path))
    assert len(calls) == (2 if state == "PENDING" else 1 if state == "uncertain" else 0)
