"""Exercise the finite owner's side effects without SSH, credentials or Kaggle."""

import json
import runpy
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

OWNER = Path(__file__).with_name("finish_entry_once.py.txt").read_text()
ACCEPTED = "actual_whole94_fixed_final_public_accepted_no_competition_entry"
ELIGIBLE = {"status": "actual_whole94_fixed_final_entry_eligibility", "eligible": True}
SUBMITTED = {"status": "one_exact_whole94_fixed_final_code_submission_accepted"}
SCORED = {"status": "actual_whole94_fixed_final_hidden_outcome", "outcome": "scored"}
BUSY = {"status": "waiting_canonical_claim_no_mutation"}


def execute(tmp_path, monkeypatch, replies, parent=ACCEPTED, change=False):
    monkeypatch.chdir(tmp_path)
    control = tmp_path / "control"
    control.mkdir()
    for name in ("entry_program.py.txt", "entry_once.py.txt"):
        (control / name).write_text("# stand-in; never executed\n")
    owner = control / "finish_entry_once.py.txt"
    owner.write_text(OWNER)
    local = tmp_path / "artifacts/whole94-final-public-20261018-v2"
    local.mkdir(parents=True)
    (local / "recovery-owner.result.json").write_text(json.dumps({"status": parent}))
    calls = []
    accepted_entries = []

    def fake_run(argv, **kwargs):
        phase = argv[-1]
        calls.append(phase)
        assert replies, "Unexpected additional side effect"
        reply = replies.pop(0)
        if change:
            (control / "entry_program.py.txt").write_text("# changed\n")
        if reply is None:
            return SimpleNamespace(returncode=1, stdout="", stderr="uncertain request")
        if reply == SUBMITTED:
            accepted_entries.append(phase)
        return SimpleNamespace(returncode=0, stdout=json.dumps(reply), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.setattr(time, "sleep", lambda _: None)
    return owner, local, calls, accepted_entries


@pytest.mark.parametrize("variant", ["normal", "no_allowance_then_ready", "claim_busy"])
def test_exactly_one_accepted_entry(tmp_path, monkeypatch, variant):
    replies = [dict(ELIGIBLE), dict(SUBMITTED), dict(SCORED)]
    if variant == "no_allowance_then_ready":
        replies.insert(0, {**ELIGIBLE, "eligible": False})
    if variant == "claim_busy":
        replies[1:1] = [dict(BUSY), dict(ELIGIBLE)]
    owner, local, calls, accepted = execute(tmp_path, monkeypatch, replies)
    runpy.run_path(str(owner))
    assert accepted == ["submit"]
    assert calls[-1] == "outcome"
    assert json.loads((local / "entry/owner.result.json").read_text())["status"] == (
        "actual_hidden_outcome_terminal"
    )


def test_public_failure_never_calls_entry(tmp_path, monkeypatch):
    owner, local, calls, accepted = execute(tmp_path, monkeypatch, [], parent="failed")
    runpy.run_path(str(owner))
    assert calls == accepted == []
    assert "not_accepted" in json.loads((local / "entry/owner.result.json").read_text())["status"]


def test_uncertain_submission_stops_without_retry(tmp_path, monkeypatch):
    owner, local, calls, accepted = execute(tmp_path, monkeypatch, [dict(ELIGIBLE), None])
    with pytest.raises(AssertionError, match="uncertain request"):
        runpy.run_path(str(owner))
    assert calls == ["eligibility", "submit"] and accepted == []
    assert "stopped" in json.loads((local / "entry/owner.result.json").read_text())["status"]


def test_changed_controls_stop_before_submit(tmp_path, monkeypatch):
    owner, _, calls, accepted = execute(tmp_path, monkeypatch, [dict(ELIGIBLE)], change=True)
    with pytest.raises(AssertionError):
        runpy.run_path(str(owner))
    assert calls == ["eligibility"] and accepted == []
