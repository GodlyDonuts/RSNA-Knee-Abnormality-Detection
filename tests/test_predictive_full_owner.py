"""Failure-boundary tests for an inert owner; never contact Slurm/HF/Kaggle."""

import importlib.machinery
import importlib.util
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

CONTROL = Path("initiative/research/predictive-full-fitting-20261020-v1")


def load(name, path):
    loader = importlib.machinery.SourceFileLoader(name, str(path))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
    loader.exec_module(module)
    return module


owner = load("predictive_full_owner_test", CONTROL / "owner.py")
remote = load("predictive_full_owner_remote_test", CONTROL / "owner_remote.py.txt")


def frozen():
    return dict(
        owner_git="f" * 40,
        prepared_sha256="a" * 64,
        program="pass",
        transport=SimpleNamespace(
            COMMAND="safe_command",
            envelope=lambda program, payload: json.dumps({"program": program, "payload": payload}),
        ),
    )


def test_duplicate_intent_does_not_contact_ssh(tmp_path, monkeypatch):
    owner.once(tmp_path / "dispatch-intent.json", {"original": True})
    monkeypatch.setattr(owner.subprocess, "run", lambda *a, **k: pytest.fail("Duplicate SSH"))
    with pytest.raises(FileExistsError):
        owner.transport_call(
            frozen(), SimpleNamespace(socket="/sock"), {}, "dispatch", "dispatch", tmp_path
        )


def test_uncertain_mutation_retains_intent_and_cannot_retry(tmp_path, monkeypatch):
    calls = []

    def timeout(*args, **kwargs):
        calls.append(args)
        raise subprocess.TimeoutExpired("ssh", 1200)

    monkeypatch.setattr(owner.subprocess, "run", timeout)
    with pytest.raises(RuntimeError, match="Uncertain original mutation"):
        owner.transport_call(
            frozen(), SimpleNamespace(socket="/sock"), {}, "dispatch", "dispatch", tmp_path
        )
    assert (tmp_path / "dispatch-intent.json").exists()
    assert json.loads((tmp_path / "dispatch-transport-failed.json").read_text())[
        "requires_read_only_reconciliation"
    ]
    with pytest.raises(FileExistsError):
        owner.transport_call(
            frozen(), SimpleNamespace(socket="/sock"), {}, "dispatch", "dispatch", tmp_path
        )
    assert len(calls) == 1


def test_observation_loss_is_not_mutation_authority(monkeypatch):
    monkeypatch.setattr(
        owner.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=255, stdout="", stderr="SSH disconnected"),
    )
    with pytest.raises(owner.ObservationLost):
        owner.transport_call(frozen(), SimpleNamespace(socket="/sock"), {}, "progress")


def test_publication_token_only_stdin_not_argv_or_receipts(tmp_path, monkeypatch):
    secret, calls = "hf_synthetic_secret_not_a_credential", []
    monkeypatch.setattr(owner, "token", lambda: secret)
    monkeypatch.setattr(owner, "CONTROL", tmp_path)

    def run(command, **kwargs):
        calls.append((command, kwargs))
        return SimpleNamespace(returncode=0, stdout='{"status":"synthetic"}', stderr="")

    monkeypatch.setattr(owner.subprocess, "run", run)
    owner.transport_call(
        frozen(),
        SimpleNamespace(socket="/sock"),
        {"arm": "B", "epoch": 1},
        "publish",
        "publish-B-001",
        tmp_path,
    )
    command, kwargs = calls[0]
    assert secret not in repr(command) and secret in kwargs["input"]
    assert all(secret not in p.read_text() for p in tmp_path.glob("*.json"))
    assert "--gres=none" in command[-1]


def test_only_sealed_epochs_publish_once():
    progress = {
        "arms": {"B": {"sealed_epochs": 0}, "C": {"sealed_epochs": 2}, "C0": {"sealed_epochs": 1}}
    }
    assert owner.ready_epoch(progress, set()) == ("C", 1)
    assert owner.ready_epoch(progress, {("C", 1)}) == ("C0", 1)
    assert owner.ready_epoch(progress, {("C", 1), ("C0", 1)}) == ("C", 2)
    assert owner.ready_epoch(progress, {("C", 1), ("C0", 1), ("C", 2)}) is None


@pytest.mark.parametrize(
    "state",
    [
        {"state": None, "queue_empty": True},
        {"state": "COMPLETED", "queue_empty": False, "exit_code": "0:0"},
        {"state": "RUNNING", "queue_empty": False, "exit_code": "0:0"},
    ],
)
def test_missing_or_stale_terminal_is_not_completion(state):
    assert not owner.completed(state) and not owner.terminal(state)


def test_atomic_original_receipt_never_overwritten(tmp_path):
    path = tmp_path / "intent.json"
    owner.once(path, {"first": 1})
    with pytest.raises(FileExistsError):
        owner.once(path, {"replacement": 1})
    assert json.loads(path.read_text()) == {"first": 1}
    assert not list(tmp_path.glob("*.pending-*"))


def test_partial_latest_log_line_waits_but_noncontiguous_history_fails(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "LOG_ROOT", tmp_path)
    monkeypatch.setattr(
        remote,
        "accounting",
        lambda jobs: {
            "123": {
                "state": "RUNNING",
                "queue_empty": False,
                "name": "rsna-predictive-full-B",
                "node": "evc33",
                "exit_code": "0:0",
            }
        },
    )
    row = dict(
        arm="B",
        step=1,
        sealed_epoch_sidecars=0,
        **dict.fromkeys(
            [
                "loss",
                "exam_loss",
                "density_loss",
                "gradient_norm",
                "head_gradient_norm",
                "encoder_gradient_norm",
            ],
            0.3,
        ),
    )
    path = tmp_path / "rsna-predictive-full-B-123.out"
    path.write_text(json.dumps(row) + '\n{"arm":"B","step":2')
    job = dict(
        GPU_job="123",
        source_git=remote.SCIENCE,
        job_name="rsna-predictive-full-B",
        node="evc33",
        stdout=str(path),
        stderr=str(path.with_suffix(".err")),
    )
    payload = {"dispatched": {"jobs": {"B": job}}}
    assert remote.log_progress(payload)["arms"]["B"]["observed_updates"] == 1
    path.write_text(json.dumps(row) + "\n" + json.dumps({**row, "step": 3}) + "\n")
    with pytest.raises(AssertionError):
        remote.log_progress(payload)


def test_physical_four_gpu_node_rejected_before_dispatch(monkeypatch):
    monkeypatch.setattr(
        remote.subprocess,
        "check_output",
        lambda *a, **k: (
            "NodeName=evc33 CfgTRES=cpu=64,gres/gpu=4 Gres=gpu:nvidia_h100_pcie:4(S:0-1)"
        ),
    )
    with pytest.raises(AssertionError):
        remote.check_node()


def test_frozen_owner_reads_recorded_commit_never_head(monkeypatch):
    calls = []
    monkeypatch.setattr(
        owner.subprocess, "check_output", lambda args, **kwargs: calls.append(args) or b"original"
    )
    assert owner.git_file("a" * 40, "some/file.py") == b"original"
    assert calls == [["git", "show", "a" * 40 + ":some/file.py"]]
    with pytest.raises(AssertionError):
        owner.git_file("HEAD", "some/file.py")


def test_telemetry_bound_funds_three_full_runs_without_limiting_checkpoint_files():
    import sys

    row = dict(
        arm="C0",
        step=2916,
        epoch=6,
        studies=8,
        encoder_updated=False,
        study_exposures=23322,
        sealed_epoch_sidecars=6,
        LR=[1.2345678901234567e-5] * 2,
    )
    row.update(
        dict.fromkeys(
            (
                "loss",
                "exam_loss",
                "density_loss",
                "encoder_gradient_norm",
                "head_gradient_norm",
                "gradient_norm",
                "elapsed_seconds",
                "seconds_since_previous_update",
                "studies_per_second_since_previous_update",
                "CPU_context_load_seconds",
            ),
            -sys.float_info.max,
        )
    )
    row.update(
        dict.fromkeys(
            ("peak_GPU_bytes", "output_bytes", "output_files", "peak_host_RSS_bytes"),
            9999999999999999999,
        )
    )
    # Deliberately larger floats/integers than valid actual full-fit telemetry.
    assert len(json.dumps(row)) + 1 < 1536
    assert (len(json.dumps(row)) + 1) * 2916 < 3 * 2**20
    assert remote.SHARED_FUTURE == 96 * 2**20 + 3 * 8 * 2**20
    assert "RLIMIT_FSIZE" not in (CONTROL / "owner_remote.py.txt").read_text()


def test_fresh_dispatch_capacity_failure_creates_no_gpu_intent(tmp_path, monkeypatch):
    monkeypatch.setattr(remote, "ROOT", tmp_path)
    (tmp_path / "full-admission").mkdir()
    admitted = {"arms": list(remote.ARMS)}
    (tmp_path / "full-admission/actual.json").write_text(json.dumps(admitted))
    monkeypatch.setattr(remote, "preparation", lambda value: None)
    monkeypatch.setattr(remote, "check_node", lambda: "physical_two_GPU")
    monkeypatch.setattr(remote, "base_other", lambda value: 1)

    def capacity(*args, **kwargs):
        raise ValueError("Fresh quota no longer sufficient")

    monkeypatch.setattr(remote.runpy, "run_path", lambda *args, **kwargs: {"check_space": capacity})
    monkeypatch.setattr(remote.subprocess, "run", lambda *args, **kwargs: pytest.fail("No sbatch"))
    with pytest.raises(ValueError, match="Fresh quota"):
        remote.dispatch({"admitted": admitted})
    assert not (tmp_path / "dispatch").exists()
