"""Original receipt compatibility and read-only failure boundaries; no SSH."""

import copy
import importlib.util
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


OBS = load("C100_test_observer", HERE / "observe.py")
C = OBS.C
P = HERE.parent / "predictive-density100-pilot-20261020-v1"
W = load("C100_original_pilot_contract", P / "worker.py")


def accepted():
    original = json.loads((HERE.parent / "predictive-stable-warmup-20261020-v1"
                           / "actual-accepted.json").read_bytes())
    arm = next(x for x in original["arms"] if x["arm"] == "C")
    terminal = copy.deepcopy(arm["terminal"])
    terminal["density_weight"] = 100
    history = copy.deepcopy(arm["real_optimizer_history"])
    for row in history:
        row["loss"] = row["exam_loss"] + 100 * row["density_loss"]
    return {
        "status": "actual_C100_full_stop_resume_engineering_complete",
        "GPU_job": C.JOB, "source_git": C.SOURCE, "original_numerical_source_git": C.SCIENCE,
        "branch": "C100", "original_engineering_receipt_sha256": C.ENGINEERING,
        "objective_measurement_sha256": C.MEASUREMENT,
        "scientific_decision_sha256": C.sha((P / "decision.json").read_bytes()),
        "binding_sha256": C.BINDING,
        "full_stop_resume_complete_state_and_prediction_bytes_equal": True,
        "private_engineering_prediction_sha256": "a" * 64,
        "terminal": terminal, "real_optimizer_history": history,
        "engineering_warnings": W.scientific_warnings(history, terminal, arm),
        "storage": {"engineering_bytes": 123, "controls_and_logs_bytes": 456},
        "new_shared_files": 0, "clinical_or_held_out_accuracy_claim": False,
        "automatic_full_training_promotion": False, "primary_B_C_C0_pipeline_unchanged": True,
    }


def state(name="COMPLETED", code="0:0", queue=""):
    return C.scheduler(f"{C.JOB}|{name}|{code}|evc27|\n", queue)


def result(success=True):
    value = {
        "status": "actual_C100_original_terminal_read_only_collected",
        "CPU_job": "999999", "GPU_job": C.JOB, "node": C.NODE, "source_git": C.SOURCE,
        "binding_sha256": C.BINDING, "scheduler": state() if success else state("FAILED", "1:0"),
        "original_files_written": 0, "GPU_allocated": False, "automatic_promotion": False,
    }
    if success:
        body = json.dumps(accepted(), indent=2) + "\n"
        value.update(acceptance_body=body, acceptance_sha256=C.sha(body.encode()),
                     acceptance_bytes=len(body.encode()), terminal_sha256="b" * 64)
    return value


def test_actual_original_submission_and_real_worker_schema():
    C.submitted(json.loads((P / "actual-submit.json").read_bytes()))
    C.acceptance(accepted())
    assert OBS.collected(result()) == result()["acceptance_body"].encode()


@pytest.mark.parametrize("change", [
    lambda x: x.update(source_git="0" * 40),
    lambda x: x.update(private_prediction_rows=[[0.1] * 12]),
    lambda x: x["terminal"].update(private_case_identifier="not-exportable"),
    lambda x: x["real_optimizer_history"][0].update(loss=float("nan")),
    lambda x: x["real_optimizer_history"][3].update(encoder_updated=False),
    lambda x: x["engineering_warnings"].update(private_notes="not-exportable"),
])
def test_reject_changed_identity_private_additions_and_nonfinite(change):
    value = accepted()
    change(value)
    with pytest.raises(ValueError):
        C.acceptance(value)


def test_purged_job_uses_accountwide_queue_and_exact_identity():
    assert state(queue="1981620|RUNNING\n9816201|RUNNING\n")["live_queue_empty"]
    assert not state(queue="981620|COMPLETING\n")["live_queue_empty"]
    with pytest.raises(ValueError):
        C.scheduler("981620|COMPLETED|0:0|evc42|\n", "")
    with pytest.raises(ValueError):
        C.scheduler("981620.batch|COMPLETED|0:0|evc27|\n", "")


class Clock:
    def __init__(self):
        self.value = datetime.fromisoformat(C.DEADLINE) - timedelta(hours=2)

    def now(self):
        return self.value

    def sleep(self, seconds):
        self.value += timedelta(seconds=seconds)


def setup_run(tmp_path, call, clock=None):
    clock = clock or Clock()
    journal, audit = tmp_path / "journal", tmp_path / "audit"
    journal.mkdir()
    audit.mkdir()
    return dict(source="f" * 40, files={"contract.py": "", "collect.py": ""},
                submit_sha="e" * 64, journal=journal, audit=audit, socket="unused",
                call=call, now=clock.now, sleep=clock.sleep)


def test_poll_loss_then_terminal_collects_exact_bytes_once(tmp_path):
    calls = []
    def call(socket, program, *, collect=False):
        calls.append(collect)
        if len(calls) == 1:
            raise OBS.ObservationLost()
        if collect:
            return result()
        return {"accounting": "981620|COMPLETED|0:0|evc27|\n", "queue": ""}
    args = setup_run(tmp_path, call)
    done = OBS.run(**args)
    assert done["acceptance_collected"]
    assert calls == [False, False, True]
    assert (args["audit"] / "actual-accepted.json").read_bytes() == (
        result()["acceptance_body"].encode())
    assert "acceptance_body" not in json.loads(
        (args["audit"] / "actual-terminal.json").read_bytes())


def test_terminal_failure_is_recorded_without_success_or_resubmit(tmp_path):
    calls = []
    def call(socket, program, *, collect=False):
        calls.append(collect)
        if collect:
            return result(False)
        return {"accounting": "981620|FAILED|1:0|evc27|\n", "queue": ""}
    args = setup_run(tmp_path, call)
    assert not OBS.run(**args)["acceptance_collected"]
    assert calls == [False, True]
    assert (args["audit"] / "actual-terminal-failure.json").exists()
    assert not (args["audit"] / "actual-accepted.json").exists()


def test_uncertain_CPU_attempt_is_never_repeated(tmp_path):
    calls = []
    def call(socket, program, *, collect=False):
        calls.append(collect)
        if collect:
            raise RuntimeError("CPU outcome uncertain")
        return {"accounting": "981620|COMPLETED|0:0|evc27|\n", "queue": ""}
    args = setup_run(tmp_path, call)
    with pytest.raises(RuntimeError, match="uncertain"):
        OBS.run(**args)
    assert calls == [False, True]
    assert (args["journal"] / "collection-01-intent.json").exists()
    assert not (args["journal"] / "collection-02-intent.json").exists()


def test_observation_loss_has_bounded_retry_and_no_job_failure_inference(tmp_path):
    def call(*args, **kwargs):
        raise OBS.ObservationLost()
    args = setup_run(tmp_path, call)
    with pytest.raises(ValueError, match="30 minutes"):
        OBS.run(**args)
    assert not list(args["audit"].iterdir())


def test_deadline_never_collects_or_changes_pending_original(tmp_path):
    def call(*args, **kwargs):
        return {"accounting": "981620|PENDING|0:0|None assigned|\n",
                "queue": "981620|PENDING\n"}
    clock = Clock()
    clock.value = datetime.fromisoformat(C.DEADLINE) - timedelta(seconds=5)
    args = setup_run(tmp_path, call, clock)
    done = OBS.run(**args)
    assert done["job_failure_inferred"] is False
    assert done["original_job_changed"] is False
    assert not list(args["audit"].iterdir())


def test_transport_only_poll_timeout_retry_and_zero_GPU_CPU_command(monkeypatch):
    calls = []
    def execute(command, **kwargs):
        calls.append((command, kwargs))
        raise subprocess.TimeoutExpired(command, 40)
    monkeypatch.setattr(OBS.subprocess, "run", execute)
    with pytest.raises(OBS.ObservationLost):
        OBS.request("socket", "pass")
    with pytest.raises(RuntimeError, match="uncertain"):
        OBS.request("socket", "pass", collect=True)
    command = calls[1][0][-1]
    assert "--gres=none" in command and "--cpus-per-task=1" in command
    assert "--nodelist=evc27" in command and "CUDA_VISIBLE_DEVICES=" in command
    assert all("sbatch" not in c[0][-1] and "scancel" not in c[0][-1] for c in calls)


def test_known_CPU_unavailable_is_retriable_but_unknown_failure_is_not(monkeypatch):
    monkeypatch.setattr(OBS.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="",
        stderr="srun: error: Unable to allocate resources: Requested nodes are busy\n"))
    with pytest.raises(OBS.AllocationUnavailable):
        OBS.request("socket", "pass", collect=True)
    monkeypatch.setattr(OBS.subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=1, stdout="", stderr="srun: task execution failed\n"))
    with pytest.raises(ValueError):
        OBS.request("socket", "pass", collect=True)


def test_uncommitted_source_and_existing_journal_are_rejected(monkeypatch, tmp_path):
    monkeypatch.setattr(OBS.subprocess, "check_output", lambda *a, **k: b"changed")
    with pytest.raises(ValueError, match="frozen source"):
        OBS.freeze("a" * 40)
    journal = tmp_path / "journal"
    journal.mkdir(mode=0o700, exist_ok=False)
    with pytest.raises(FileExistsError):
        journal.mkdir(mode=0o700, exist_ok=False)
    path = journal / "intent.json"
    OBS.save(path, {"original": True})
    with pytest.raises(FileExistsError):
        OBS.save(path, {"replacement": True})
    assert json.loads(path.read_bytes()) == {"original": True}


def test_actual_CPU_reader_roundtrip_no_original_writes_or_private_output(
        monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "observer_contract", C)
    reader = load("C100_test_remote_reader", HERE / "collect.py")
    monkeypatch.setattr(C, "ROOT", str(tmp_path))
    binding = b'{"private_source_paths":["not-for-export"]}\n'
    monkeypatch.setattr(C, "BINDING", C.sha(binding))
    value = accepted()
    prepared = json.loads((P / "actual-prepare.json").read_bytes())
    prepared["binding_sha256"] = C.BINDING
    files = {
        "binding.private.json": binding,
        "prepared.json": json.dumps(prepared).encode(),
        "GPU-run-start.json": json.dumps(
            {"GPU_job": C.JOB, "source_git": C.SOURCE, "branch": "C100"}).encode(),
        "GPU-accepted.json": (json.dumps(value, indent=2) + "\n").encode(),
        "engineering/C/full/terminal.json": json.dumps(value["terminal"]).encode(),
        "scientific-decision.json": (P / "decision.json").read_bytes(),
    }
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        path.chmod(0o600)
    before = {name: ((tmp_path / name).read_bytes(), (tmp_path / name).stat().st_mtime_ns)
              for name in files}
    monkeypatch.setattr(reader.os, "uname", lambda: SimpleNamespace(nodename=C.NODE))
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.setenv("SLURM_JOB_ID", "999999")
    def scheduler(command, **kwargs):
        if command[0] == "sacct":
            return "981620|COMPLETED|0:0|evc27|\n"
        assert command == ["squeue", "-h", "-u", "sa305415", "-o", "%i|%T"]
        return "1981620|RUNNING\n"
    monkeypatch.setattr(reader.subprocess, "check_output", scheduler)
    reader.main()
    output = capsys.readouterr().out
    assert "not-for-export" not in output
    assert OBS.collected(json.loads(output)) == files["GPU-accepted.json"]
    assert before == {
        name: ((tmp_path / name).read_bytes(), (tmp_path / name).stat().st_mtime_ns)
        for name in files}
