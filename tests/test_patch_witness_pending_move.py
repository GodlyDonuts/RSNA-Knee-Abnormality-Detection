"""A pending-only resource move must never cancel a job that has started."""

import ast
import io
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

P = Path(
    "initiative/audits/patch-witness-gpu-recovery-v1/relocation-v2/dispatch-coordinator.py.txt"
)


def cancellation_program():
    for node in ast.parse(P.read_text()).body:
        if isinstance(node, ast.Assign) and any(
            isinstance(n, ast.Name) and n.id == "cancel" for n in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError("Original cancellation program absent")


@pytest.mark.parametrize(
    "before,race,allowed",
    [("PENDING", False, True), ("PENDING", True, False), ("RUNNING", False, False)],
)
def test_running_job_or_allocation_race_cannot_authorize_replacement(
    monkeypatch, capsys, before, race, allowed
):
    import subprocess
    import sys

    calls = []

    def read(command, **kwargs):
        if command[0] == "scontrol":
            return f"JobName=rsna-patch-witness-recovery-v1 JobState={before} RunTime=00:00:00"
        if "-j" in command and command[command.index("-j") + 1] == "1,2":
            return "1|COMPLETED|0:0|00:02:00|evc31\n2|COMPLETED|0:0|00:03:00|evc48\n"
        return (
            "873279|RUNNING|0:0|1|evc31|cpu=2,gres/gpu=1|\n"
            if race
            else "873279|CANCELLED by owner|0:0|0|None assigned||\n"
        )

    def cancel(command, **kwargs):
        assert "--state=PENDING" in command and "--ctld" in command
        assert "--user=sa305415" in command and command[-1] == "873279"
        calls.append(command)
        return SimpleNamespace(returncode=0, stderr="")

    monkeypatch.setattr(subprocess, "check_output", read)
    monkeypatch.setattr(subprocess, "run", cancel)
    monkeypatch.setattr(
        sys, "stdin", io.StringIO(json.dumps({"source_CPU": "1", "destination_CPU": "2"}))
    )
    if allowed:
        exec(compile(cancellation_program(), str(P), "exec"), {})
        assert (
            json.loads(capsys.readouterr().out)["old_model_optimizer_or_runtime_started"] is False
        )
    else:
        with pytest.raises(AssertionError):
            exec(compile(cancellation_program(), str(P), "exec"), {})
    assert bool(calls) == (before == "PENDING")
