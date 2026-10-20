"""Read-only observation of original fitting logs, without remote operations."""

import ast
import copy
import importlib.util
import io
import json
import math
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "initiative/research/predictive-training-observer-20261020-v1/observe.py"
spec = importlib.util.spec_from_file_location("training_observer_tests", PATH)
o = importlib.util.module_from_spec(spec)
spec.loader.exec_module(o)


def derived(node):
    placement = types.ModuleType("placement")
    exec(
        compile(o.git_file(o.PIN, o.PORTABLE + "/placement.py"), "pinned_placement", "exec"),
        placement.__dict__,
    )
    original = o.git_file(o.SCIENCE, o.BASE + "/owner_remote.py.txt").decode()
    return placement, placement.derived_remote(original, node, 123 * 2**20, 376)


def dispatch(arms, node):
    jobs = {}
    for index, arm in enumerate(("B", "C", "C0"), 1):
        if arm not in arms:
            continue
        identity = str(1000 + index)
        name = "rsna-predictive-full2-" + arm
        jobs[arm] = {
            "GPU_job": identity,
            "arm": arm,
            "node": node,
            "job_name": name,
            "source_git": o.SCIENCE,
            "binding_sha256": "a" * 64,
            "script_sha256": "b" * 64,
            "run_root": f"/tmp/rsna-predictive-full-{arm}-20261020-v2",
            "stdout": f"{o.LOG_ROOT}/{name}-{identity}.out",
            "stderr": f"{o.LOG_ROOT}/{name}-{identity}.err",
        }
    # Evaluate the result dictionary from the actual generated dispatch function.
    # This catches status/schema replacements made by the portable control.
    _, source = derived(node)
    function = next(
        n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "dispatch"
    )
    assignment = next(
        n
        for n in function.body
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "result" for t in n.targets)
    )
    return eval(
        compile(ast.Expression(assignment.value), "generated_dispatch_result", "eval"),
        {"SCIENCE": o.SCIENCE, "ARMS": tuple(arms), "jobs": jobs},
    )


def record(step, arm="B"):
    result = {key: 1.0 for key in o.TELEMETRY}
    result.update(
        step=step,
        epoch=1,
        studies=8,
        LR=[3e-5, 3e-5],
        encoder_updated=step > 3,
        arm=arm,
        sealed_epoch_sidecars=0,
    )
    return result


def item(step, arm="B"):
    last = record(step, arm)
    return {
        "GPU_job": dispatch([arm], "evc27" if arm == "C0" else "evc33")["jobs"][arm]["GPU_job"],
        "slurm": {"state": "RUNNING", "queue_empty": False, "exit_code": "0:0"},
        "observed_updates": step,
        "latest": last,
        "first200": {
            "actual_updates": 200,
            "history_sha256": "c" * 64,
            "maxima": {key: 1.0 for key in o.MAXIMA},
            "last": record(200, arm),
        }
        if step >= 200
        else None,
        "terminal": None,
        "stderr_bytes": 0,
        "sealed_epochs": 0,
    }


def original_progress(tmp_path, records):
    """Execute the actual pinned reader, replacing only scheduler/filesystem fixtures."""
    source = o.git_file(o.SCIENCE, o.BASE + "/owner_remote.py.txt")
    ns = {"__name__": "frozen_progress_test"}
    exec(compile(source, "actual_frozen_owner_remote", "exec"), ns)
    ns["SCIENCE"], ns["LOG_ROOT"] = o.SCIENCE, tmp_path
    expected = dispatch(["B"], "evc33")
    job = expected["jobs"]["B"]
    for suffix in ("stdout", "stderr"):
        job[suffix] = str(tmp_path / Path(job[suffix]).name)
    Path(job["stdout"]).write_text("".join(json.dumps(v) + "\n" for v in records))
    ns["accounting"] = lambda jobs: {j: {"state": "RUNNING", "queue_empty": False} for j in jobs}
    return ns["log_progress"]({"dispatched": expected}), expected


def test_pinned_reader_retains_peak_regime_and_sanitizer_drops_private_extras(tmp_path):
    rows = [record(step) for step in range(1, 201)]
    rows[-1]["private_case_id"] = "must-not-be-preserved"
    result, jobs = original_progress(tmp_path, rows)
    assert result["arms"]["B"]["first200"]["actual_updates"] == 200
    clean = o.sanitized(result, jobs, {"sha256": "d" * 64}, {})
    assert clean["B"]["latest"]["step"] == 200
    assert clean["B"]["first200"]["last"]["LR"] == [3e-5, 3e-5]
    assert "must-not-be-preserved" not in json.dumps(clean)
    assert clean["B"]["clinical_accuracy_or_logit_saturation_claim"] is False
    first200 = result["arms"]["B"]["first200"]
    assert first200["history_sha256"] == o.sha(o.canonical(rows))


def test_original_reader_rejects_nonfinite_and_noncontiguous_records(tmp_path):
    rows = [record(1), record(2)]
    rows[1]["head_gradient_norm"] = math.inf
    with pytest.raises(AssertionError):
        original_progress(tmp_path, rows)
    rows[1] = record(3)
    with pytest.raises(AssertionError):
        original_progress(tmp_path, rows)


@pytest.mark.parametrize(("node", "arms"), [("evc33", ("B", "C")), ("evc27", ("C0",))])
def test_actual_portable_progress_uses_generated_node_schema(tmp_path, monkeypatch, node, arms):
    placement, source = derived(node)
    monkeypatch.setitem(sys.modules, "placement", placement)
    remote = types.ModuleType("frozen_portable_progress")
    exec(
        compile(o.git_file(o.PIN, o.PORTABLE + "/remote.py"), "pinned_remote", "exec"),
        remote.__dict__,
    )
    ns = {"__name__": "generated_owner_control"}
    exec(compile(source, "generated_owner_control", "exec"), ns)
    ns["SCIENCE"], ns["LOG_ROOT"] = o.SCIENCE, tmp_path
    jobs = dispatch(arms, node)
    for arm, job in jobs["jobs"].items():
        for field in ("stdout", "stderr"):
            job[field] = str(tmp_path / Path(job[field]).name)
        Path(job["stdout"]).write_text(
            "".join(json.dumps(record(step, arm)) + "\n" for step in range(1, 201))
        )
    ns["accounting"] = lambda ids: {
        identity: {"state": "RUNNING", "queue_empty": False, "node": node} for identity in ids
    }
    # Only scheduler/filesystem fixtures are substituted; the portable main and
    # generated original main/log_progress are the exact pinned implementations.
    monkeypatch.setattr(remote, "namespace", lambda value, host: ns)
    handoff_path = ROOT / o.PORTABLE / "actual-fitting-owner68792-stopped.json"
    handoff = handoff_path.read_bytes()
    value = {
        "mode": "progress",
        "node": node,
        "prepared": {"source_git": o.SCIENCE},
        "handoff": json.loads(handoff),
        "handoff_text": handoff.decode(),
        "handoff_sha256": o.sha(handoff),
        "dispatched": jobs,
    }
    actual = remote.main(value)
    clean = o.sanitized(actual, jobs, {"sha256": "d" * 64}, {})
    assert set(clean) == set(arms)
    assert all(row["latest"]["step"] == 200 for row in clean.values())


def fixture_receipts(tmp_path, monkeypatch):
    monkeypatch.setattr(o, "ROOT", tmp_path)
    journal = tmp_path / "journal"
    journal.mkdir()
    monkeypatch.setattr(o, "OWNER_JOURNAL", journal)
    control = tmp_path / o.SUCCESSOR
    control.mkdir(parents=True)
    return journal, control


def test_dispatch_waits_for_actual_matching_receipts_and_binds_nodes(tmp_path, monkeypatch):
    data = dispatch(["B", "C"], "evc33")
    journal, control = fixture_receipts(tmp_path, monkeypatch)
    assert o.dispatches() == ({}, {})
    path = journal / "dispatch-BC-result.json"
    o.save(path, data)
    assert o.dispatches() == ({}, {})
    audit = control / "actual-owner-dispatch-BC.json"
    o.save(audit, data)
    found, refs = o.dispatches()
    assert found == {"evc33": data}
    assert refs["evc33"]["sha256"] == o.sha(path.read_bytes())
    changed = copy.deepcopy(data)
    changed["jobs"]["B"]["node"] = "evc48"
    o.save(audit, changed, replace=True)
    with pytest.raises(ValueError, match="pair"):
        o.dispatches()
    o.save(path, changed, replace=True)
    with pytest.raises(ValueError, match="binding"):
        o.dispatches()


def test_query_has_only_management_progress_and_preserves_rejection(monkeypatch):
    transport = types.SimpleNamespace(
        COMMAND="frozen_command", envelope=lambda program, value: json.dumps(value)
    )
    args = types.SimpleNamespace(socket="/tmp/original.sock")
    jobs = dispatch(["B", "C"], "evc33")
    calls = []

    def fake_run(command, **kw):
        calls.append((command, json.loads(kw["input"])))
        return types.SimpleNamespace(
            returncode=1, stdout="", stderr='{"error_type":"AssertionError","site_line":332}\n'
        )

    monkeypatch.setattr(o.subprocess, "run", fake_run)
    with pytest.raises(o.ObservationRejected) as raised:
        o.query(
            args,
            transport,
            "fixed",
            {"mode": "progress"},
            "evc33",
            jobs,
            jobs["jobs"],
            o.time.time() + 100,
        )
    command, payload = calls[0]
    assert command[:2] == ["ssh", "-S"]
    assert o.MANAGEMENT in command[-1]
    assert not any(word in command[-1] for word in ("srun", "sbatch", "scancel"))
    assert payload["mode"] == "progress" and "token" not in payload
    assert raised.value.evidence["diagnostic"] == {"error_type": "AssertionError", "site_line": 332}
    assert raised.value.evidence["nonfinite_or_other_original_validation_failure_not_retried"]
    assert len(calls) == 1


def test_frozen_capsule_rejects_mutating_modes_before_loading_controls(tmp_path, monkeypatch):
    # Use actual frozen source/preparation, synthetic future owner receipts, no SSH.
    def real_git(pin, name):
        return subprocess.check_output(["git", "show", pin + ":" + name], cwd=ROOT)
    successor_pin = "f" * 40
    own = {name: (ROOT / o.SUCCESSOR / name).read_bytes()
           for name in ("contract.py", "remote_adapter.py", "launch.py")}
    monkeypatch.setattr(o, "git_file", lambda pin, name:
        own[Path(name).name] if pin == successor_pin else real_git(pin, name))
    monkeypatch.setattr(o, "ROOT", tmp_path)
    journal = tmp_path / "artifacts/owner/finite-owner"
    journal.mkdir(parents=True)
    monkeypatch.setattr(o, "OWNER_JOURNAL", journal)
    control = tmp_path / o.SUCCESSOR
    control.mkdir(parents=True)
    handoff = {"status": "actual_idle_v3_cache_dependent_owner_stopped_before_phase",
        "role": "fitting", "pid": 80995, "source_git": o.PIN,
        "phase_intents": [], "children": [], "owner_process_absent_after_stop": True,
        "remote_admission_or_dispatch_exists": False, "remote_jobs_cancelled": [],
        "remote_read_only_snapshot_sha256": "a" * 64}
    decision = {"status": "accepted_v4_cache_owner_handoff", "placement_source_git": successor_pin,
                "shared_future_bytes": 123 * 2**20, "shared_future_files": 376}
    handoff_path, decision_path = control / "handoff.json", control / "decision.json"
    o.save(handoff_path, handoff)
    o.save(decision_path, decision)
    prepared_path = tmp_path / o.BASE / "actual-cpu-prepared.json"
    prepared_path.parent.mkdir(parents=True)
    prepared_path.write_bytes((ROOT / o.BASE / "actual-cpu-prepared.json").read_bytes())
    owner = {"source_git": successor_pin, "science_git": o.SCIENCE,
        "original_owner_source_git": o.PIN, "original_owner_pid": 80995,
        "source_sha256": {name: o.sha(raw) for name, raw in own.items()},
        "decision_sha256": o.sha(decision_path.read_bytes()),
        "handoff_sha256": o.sha(handoff_path.read_bytes()),
        "decision_path": str(decision_path.relative_to(tmp_path)),
        "handoff_path": str(handoff_path.relative_to(tmp_path))}
    o.save(journal / "owner-intent.json", owner)
    _, program, payload, refs = o.frozen()
    assert payload["mode"] == "progress"
    assert payload["placement_source_git"] == successor_pin
    assert refs["original_portable_source_git"] == o.PIN
    assert refs["progress_program_sha256"] == o.sha(program.encode())
    monkeypatch.setattr(sys, "stdin", io.StringIO('{"mode":"dispatch"}'))
    with pytest.raises(AssertionError):
        exec(compile(program, "observer_capsule", "exec"), {"__name__": "capsule_test"})
    owner["source_sha256"]["remote_adapter.py"] = "0" * 64
    o.save(journal / "owner-intent.json", owner, replace=True)
    with pytest.raises(ValueError, match="manifest"):
        o.frozen()


def test_finite_observer_saves_first200_once_then_latest_without_remote_mutations(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(o, "HERE", tmp_path)
    journal = tmp_path / "journal"
    journal.mkdir()
    jobs = {"evc33": dispatch(["B", "C"], "evc33"), "evc27": dispatch(["C0"], "evc27")}
    refs = {n: {"path": n, "sha256": "d" * 64} for n in jobs}
    monkeypatch.setattr(o, "frozen", lambda: (None, "", {}, {}))
    monkeypatch.setattr(o, "dispatches", lambda: (jobs, refs))
    calls = []

    def query(*args):
        node = args[4]
        step = 205 if len(calls) < 2 else 206
        calls.append(node)
        arms = {arm: item(step, arm) for arm in jobs[node]["jobs"]}
        if step == 206:
            for value in arms.values():
                value["slurm"] = {"state": "FAILED", "queue_empty": True, "exit_code": "1:0"}
        return {"status": "actual_predictive_full_job_progress_observed", "arms": arms}

    monkeypatch.setattr(o, "query", query)
    monkeypatch.setattr(o.time, "sleep", lambda seconds: None)
    args = types.SimpleNamespace(
        deadline_utc=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        source_git="a" * 40,
        poll_seconds=120,
    )
    o.run(args, journal)
    assert calls == ["evc33", "evc27", "evc33", "evc27"]
    for arm in ("B", "C", "C0"):
        early = json.loads((tmp_path / f"actual-first200-{arm}.json").read_text())
        assert early["latest"]["step"] == early["observed_updates"] == 205
        assert early["first200"]["last"]["step"] == 200
    latest = json.loads((tmp_path / "actual-latest.json").read_text())
    assert all(v["observed_updates"] == 206 for v in latest["arms"].values())
    terminal = json.loads((journal / "observer-result.json").read_text())
    assert terminal["remote_mutations"] == 0
    assert terminal["first200_preserved"] == ["B", "C", "C0"]


def test_read_only_reconnect_is_bounded_without_mutation_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(o, "HERE", tmp_path)
    journal = tmp_path / "journal"
    journal.mkdir()
    jobs = {"evc33": dispatch(["B", "C"], "evc33")}
    monkeypatch.setattr(o, "frozen", lambda: (None, "", {}, {}))
    monkeypatch.setattr(o, "dispatches", lambda: (jobs, {"evc33": {"sha256": "d" * 64}}))
    calls = []

    def query(*args):
        calls.append(args[4])
        raise subprocess.TimeoutExpired("read-only SSH", 90)

    clock = iter((100.0, 100.0, 1901.0))
    monkeypatch.setattr(o, "query", query)
    monkeypatch.setattr(o.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(o.time, "sleep", lambda seconds: None)
    args = types.SimpleNamespace(
        deadline_utc=(datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
        source_git="a" * 40,
        poll_seconds=120,
    )
    with pytest.raises(ValueError, match="absent30minutes"):
        o.run(args, journal)
    assert calls == ["evc33", "evc33"]
