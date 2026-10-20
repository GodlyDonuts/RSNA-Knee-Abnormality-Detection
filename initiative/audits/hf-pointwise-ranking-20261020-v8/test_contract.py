"""Local reserve/owner regressions; no remote calls or publication."""

import ast
import contextlib
import importlib.util
import json
import os
import types
import uuid
from pathlib import Path

import pytest

D = Path(__file__).resolve().parent


def functions():
    source = (D / "program.py.txt").read_text()
    nodes = ast.parse(source).body
    return {n.name: ast.get_source_segment(source, n) for n in nodes if isinstance(n, ast.FunctionDef)}


def accounting():
    scope = {}
    exec(functions()["shared_file_accounting"], scope)
    return scope["shared_file_accounting"]


def test_four_actual_owned_entries_are_charged_once():
    result = accounting()(5375, 4)
    assert result["remaining_future_shared_files"] == 372
    assert result["files_after_all_remaining_liabilities"] == 5003
    assert result["new_permanent_shared_files"] == result["new_shared_byte_reservation"] == 0


@pytest.mark.parametrize("free,owned", [(5371, 4), (5375, 3), (5375, 5), (5375, 8), (5375, True)])
def test_unfunded_or_unproven_file_credit_refused(free, owned):
    with pytest.raises(AssertionError):
        accounting()(free, owned)


def quiescence():
    spec = importlib.util.spec_from_file_location("v8_quiescence_test", D / "quiescence.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture(root):
    journal = root / "artifacts/pointwise-final-deployment-20261020-v1"
    journal.mkdir(parents=True)
    (journal / "continuation-intent.json").write_text(json.dumps({
        "PID": 72266, "original_source_git": "01cbe205080487b3fb39fbf7c0d28e112fcd7d77"}))
    return journal


def test_healthy_owner_without_terminal_prevents_any_mirror(tmp_path):
    fixture(tmp_path)
    with pytest.raises(AssertionError, match="actual pointwise"):
        quiescence().check(tmp_path, processes="", own_pid=1)


@pytest.mark.parametrize("processes", [
    "72266 python3 initiative/research/pointwise-final-deployment-20261020-v1/continue_verified_dataset.py",
    "40 python3 initiative/research/teacher-rank-final-deployment-20261020-v1/follow_once.py",
    "41 ssh newton srun --job-name=rsna-pointwise-serving-status",
])
def test_terminal_file_cannot_override_live_account_process(tmp_path, processes):
    journal = fixture(tmp_path)
    (journal / "complete.json").write_text(json.dumps({"outcome": "scored"}))
    with pytest.raises(AssertionError):
        quiescence().check(tmp_path, processes=processes, own_pid=1)


def test_actual_terminal_and_no_account_processes_admitted(tmp_path):
    journal = fixture(tmp_path)
    (journal / "complete.json").write_text(json.dumps({"outcome": "scored"}))
    result = quiescence().check(tmp_path, processes="123 python3 predictive-full-fitting-20261020-v2/owner.py", own_pid=1)
    assert result["active_known_account_owners_or_transports"] == 0
    assert result["pointwise_terminal_name"] == "complete.json"


def test_both_locks_serialize_four_actual_owned_inodes_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic")
    claim = tmp_path / "claim"
    hf = tmp_path / "publication.lock"
    state = {}

    def new(path, value):
        with path.open("x") as f:
            json.dump(value, f)

    @contextlib.contextmanager
    def publication_lock(path, timeout, poll):
        assert claim.is_dir() and (claim / "owner.json").is_file()
        assert timeout == 30 and poll == 1
        path.mkdir()
        new(path / "owner.json", {"pid": os.getpid(), "job_id": "synthetic"})
        try:
            yield
        finally:
            (path / "owner.json").unlink()
            path.rmdir()

    scope = dict(contextlib=contextlib, Path=Path, os=os, uuid=uuid, json=json)
    scope["subprocess"] = types.SimpleNamespace(check_output=lambda *a, **k: "/home/sa305415 0 1048576000 0 - 994625 1000000 0 -\n")
    exec(functions()["shared_file_accounting"], scope)
    definition = functions()["serialized_milestone_locks"].replace("/home/sa305415/rsna-knee/kaggle/coordinator-claim", str(claim))
    exec("@contextlib.contextmanager\n" + definition, scope)
    pub = types.SimpleNamespace(atomic_new=new, LOCK_RUNNER=tmp_path / "runner.py", LOCK_RUNNER_SHA="pinned", PUBLICATION_LOCK=hf,
                                load_pinned=lambda *a: types.SimpleNamespace(publication_lock=publication_lock))
    with scope["serialized_milestone_locks"](pub) as value:
        state.update(value)
        assert claim.exists() and hf.exists()
    assert state["actually_owned_transient_files_already_charged"] == 4
    assert not claim.exists() and not hf.exists()
