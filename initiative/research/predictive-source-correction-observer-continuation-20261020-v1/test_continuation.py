"""Local startup-race, immutable-observer and duplicate-continuation checks."""

import importlib.util
import io
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture
def c():
    spec = importlib.util.spec_from_file_location("observer_continuation", HERE / "continue_once.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_actual_failed_preintent_observer(c, monkeypatch):
    monkeypatch.setattr(c, "process_command", lambda pid: "")
    assert c.predecessor()["original_failure_sha256"] == c.FAILURE_SHA
    monkeypatch.setattr(c, "process_command", lambda pid: "original observer")
    with pytest.raises(ValueError, match="still exists"):
        c.predecessor()


def test_waits_for_owner_intent_before_any_new_journal(c, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "OLD_JOURNAL", tmp_path)
    now = [0]
    body = {"pid": 11013, "source_git": c.ORIGINAL,
            "science_git": "44cc559e947560a635c34fee7b1f2a3b7be3a241"}
    path = tmp_path / "finite-owner/owner-intent.json"

    def sleep(seconds):
        now[0] += seconds
        if now[0] >= 2:
            path.parent.mkdir(exist_ok=True)
            path.write_text(json.dumps(body))

    digest = c.await_owner(clock=lambda: now[0], sleep=sleep, timeout=3)
    assert digest == c.sha(path.read_bytes()) and now[0] == 2
    assert sorted(p.name for p in tmp_path.iterdir()) == ["finite-owner"]


def test_missing_owner_wait_is_bounded(c, tmp_path, monkeypatch):
    monkeypatch.setattr(c, "OLD_JOURNAL", tmp_path)
    now = [0]
    with pytest.raises(ValueError, match="remains absent"):
        c.await_owner(clock=lambda: now[0], sleep=lambda n: now.__setitem__(0, now[0] + n), timeout=2)
    assert now[0] == 2 and list(tmp_path.iterdir()) == []


def test_exact_original_frozen_capsule_remains_allocation_free(c, monkeypatch):
    observer, hashes = c.frozen_observer()
    _, program, payload, provenance = observer.frozen()
    assert observer.HERE == HERE and observer.SUCCESSOR == c.OLD
    assert payload["mode"] == "progress" and provenance["placement_source_git"] == c.ORIGINAL
    assert set(hashes) == {"observe.py", "launch.py", "contract.py", "remote_adapter.py"}
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"mode":"dispatch"}'))
    with pytest.raises(AssertionError):
        exec(compile(program, "unchanged_readonly_capsule", "exec"), {})


def test_distinct_journal_preserves_failed_observer_and_refuses_repeat(c, tmp_path, monkeypatch):
    observer, _ = c.frozen_observer()
    old_failure = (c.OLD_JOURNAL / "observer-failure.json").read_bytes()
    owner_sha = c.await_owner()
    observer.HERE = tmp_path
    calls = []
    observer.run = lambda args, journal: calls.append((args.source_git, journal))
    monkeypatch.setattr(c, "ROOT", tmp_path)
    monkeypatch.setattr(c, "HERE", tmp_path / HERE.name)
    c.HERE.mkdir()
    (c.HERE / "continue_once.py").write_bytes((HERE / "continue_once.py").read_bytes())
    monkeypatch.setattr(c, "git_file", lambda *a: (HERE / "continue_once.py").read_bytes())
    monkeypatch.setattr(c, "predecessor", lambda: {"original_observer_PID": 11015})
    monkeypatch.setattr(c, "await_owner", lambda: owner_sha)
    monkeypatch.setattr(c, "frozen_observer", lambda: (observer, {"source": "a" * 64}))
    args = SimpleNamespace(source_git="f" * 40, poll_seconds=120, deadline_utc=c.DEADLINE)
    c.execute(args)
    journal = tmp_path / "artifacts" / HERE.name
    assert calls == [(c.ORIGINAL, journal)]
    intent = json.loads((journal / "continuation-intent.json").read_bytes())
    assert intent["continuation_source_git"] == "f" * 40
    assert (c.OLD_JOURNAL / "observer-failure.json").read_bytes() == old_failure
    with pytest.raises(ValueError, match="Existing continuation"):
        c.execute(args)
    assert len(calls) == 1
