"""Exercise the actual failed startup bytes only in isolated temporary journals."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("startup_continue", HERE / "continue.py")
C = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(C)


def fixture(tmp_path):
    journal = tmp_path / "original"
    journal.mkdir(mode=0o700)
    for name in C.STARTUP:
        target = journal / name
        target.write_bytes((C.JOURNAL / name).read_bytes())
        target.chmod(0o600)
    raw = (C.OLD / "actual-local-startup-failure.json").read_bytes()
    return journal, json.loads(raw), C.sha(raw)


def once(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream)


def runtime(journal, *, fail=False):
    calls = []
    original_bytes = {p.name: p.read_bytes() for p in journal.iterdir()}

    def owner(args, original, transport, program, payload, selected):
        assert selected == journal
        assert args.source_git == C.SOURCE
        calls.append("construct")
        return SimpleNamespace(args=args, journal=selected, payload=payload)

    def run(instance):
        calls.append("run")
        # The exact frozen guard, with no remote or mutation phase executed.
        assert not any(p.name.endswith(("-intent.json", "-result.json", "-transport.json"))
                       and p.name != "owner-intent.json" for p in journal.iterdir())
        assert (journal / "launcher-started.json").read_bytes() == original_bytes["launch-intent.json"]
        for name, raw in original_bytes.items():
            if name != "launch-intent.json":
                assert (journal / name).read_bytes() == raw
        assert instance.transfer("C") == (instance, "C")
        if fail:
            raise TimeoutError("simulated observation loss")
        return {"status": "fixture_once"}

    frozen = {"p": SimpleNamespace(once=once), "fit": SimpleNamespace(Owner=owner),
              "original": object(), "transport": object(), "program": "unchanged",
              "payload": {"owner_git": C.SOURCE}}
    old = SimpleNamespace(run=run, transfer=lambda owner, arm: (owner, arm))
    return old, frozen, calls


def test_exact_actual_failed_bytes_and_live_process_rejection(tmp_path):
    journal, proof, _ = fixture(tmp_path)
    hashes = C.inspect(proof, journal, lambda pid: "")
    assert hashes["owner-intent.json"] == C.INTENT
    assert hashes["launch-intent.json"] == C.LAUNCH
    with pytest.raises(ValueError, match="remains live"):
        C.inspect(proof, journal, lambda pid: "python owner.py" if pid == 32377 else "")


@pytest.mark.parametrize("change", ["phase", "stderr", "permission", "proof"])
def test_no_continuation_after_remote_phase_or_changed_evidence(tmp_path, change):
    journal, proof, _ = fixture(tmp_path)
    if change == "phase":
        (journal / "dispatch-intent.json").write_text("{}")
    elif change == "stderr":
        (journal / "stderr").write_text("different error")
    elif change == "permission":
        (journal / "owner-intent.json").chmod(0o644)
    else:
        proof["remote_operations_executed"] = 1
    with pytest.raises(ValueError):
        C.inspect(proof, journal, lambda _: "")


def test_continue_exact_frozen_run_once_retaining_old_intent_and_error(tmp_path):
    journal, proof, digest = fixture(tmp_path)
    old, frozen, calls = runtime(journal)
    new = tmp_path / "continuation"
    args = SimpleNamespace(source_git=C.SOURCE)
    result = C.execute(old, frozen, args, proof, digest, journal, new,
                       source_git="f" * 40, process_command=lambda _: "")
    assert result == {"status": "fixture_once"}
    assert calls == ["construct", "run"]
    receipt = json.loads((new / "continuation-intent.json").read_bytes())
    assert receipt["original_source_git"] == C.SOURCE and receipt["source_git"] == "f" * 40
    assert receipt["original_failure_receipt_sha256"] == digest
    assert receipt["original_startup_sha256"]["owner-intent.json"] == C.INTENT
    assert not (journal / "continuation-intent.json").exists()
    with pytest.raises(ValueError):
        C.execute(old, frozen, args, proof, digest, journal, new,
                  source_git="f" * 40, process_command=lambda _: "")
    assert calls == ["construct", "run"]


def test_failed_run_preserves_original_failure_and_never_retries(tmp_path):
    journal, proof, digest = fixture(tmp_path)
    original_ended = (journal / "owner-ended.json").read_bytes()
    old, frozen, calls = runtime(journal, fail=True)
    new = tmp_path / "continuation"
    with pytest.raises(TimeoutError):
        C.execute(old, frozen, SimpleNamespace(source_git=C.SOURCE), proof, digest, journal, new,
                  source_git="f" * 40, process_command=lambda _: "")
    assert (journal / "owner-ended.json").read_bytes() == original_ended
    assert json.loads((new / "continuation-ended.json").read_bytes())["automatic_retry"] is False
    assert not (new / "continuation-result.json").exists()
    assert calls == ["construct", "run"]


def test_loads_exact_frozen_controller_without_starting_it():
    old = C.original()
    assert old.d.SCIENCE == "44cc559e947560a635c34fee7b1f2a3b7be3a241"
    assert old.run.__name__ == "run" and old.freeze.__name__ == "freeze"
