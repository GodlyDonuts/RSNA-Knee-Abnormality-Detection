"""Reject duplicate jobs, stale CPU proofs and changed payloads before sbatch."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import socket
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/newton/dispatch_native384_pilot.py"
SPEC = importlib.util.spec_from_file_location("native_pilot_dispatch", SCRIPT)
assert SPEC and SPEC.loader
dispatcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dispatcher)
BASE = "/home/sa305415/rsna-knee"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stamp(path: Path) -> dict:
    stat = path.stat()
    return {
        "dev": stat.st_dev,
        "ino": stat.st_ino,
        "bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "ctime_ns": stat.st_ctime_ns,
    }


@pytest.fixture
def remote(tmp_path: Path):
    """Only path prefixes and Slurm transport are substituted; program stays exact."""
    base = tmp_path / "remote"
    control = base / "native384-recovery-c9-pilot-v1"
    control.mkdir(parents=True)

    def put(path: Path, data: bytes) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return {"bytes": len(data), "sha256": digest(data)}

    def save(path: Path, value: dict) -> dict:
        return put(path, json.dumps(value, sort_keys=True).encode())

    publication = {"files": {"cache.tar": put(control / "payloads/cache.tar", b"cache")}}
    weights = base / "weights"
    config = base / "config"
    encoder = {
        "weights_path": str(weights),
        "weights_sha256": put(weights, b"weights")["sha256"],
        "config_path": str(config),
        "config_sha256": put(config, b"config")["sha256"],
    }
    binding = {"manifest.json": put(base / "code-stopfix-1f486765/manifest.json", b"accepted")}
    scientific = {
        "rsna/train.py": put(base / "code-delta-recovery-c9fcc7e/rsna/train.py", b"source")[
            "sha256"
        ]
    }
    files = {
        "pilot-config.json": save(control / "pilot-config.json", {"encoder": encoder}),
        "selection.json": save(control / "selection.json", {"input_bindings": binding}),
        "pilot.sbatch": put(control / "pilot.sbatch", b"#!/bin/bash\nexit 0\n"),
    }
    plan = {
        "control": str(control),
        "files": files,
        "pinned_private_cache_publication": publication,
    }
    save(
        control / "cpu-stage-verified.json",
        {
            "status": "private_native_pilot_payloads_cpu_staged_fully_verified",
            "files": files,
            "private_publication": publication,
            "job_id": "865906",
        },
    )
    admission = (
        "import hashlib,json\n"
        "def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()\n"
        "def main(): print(json.dumps({'gpu_dispatch_admitted': True}))\n"
    )
    request = {"plan": plan, "admission_source": admission, "scientific_files": scientific}
    env_path = (
        base / "kaggle-wheelhouse-v1/result-cp312-cu130-finalmain-v1/environment-cp312-cu130.tar"
    )
    put(env_path, b"diagnostic-transport-fixture")
    assets = {
        str(control / "payloads/cache.tar"): publication["files"]["cache.tar"]["sha256"],
        str(weights): encoder["weights_sha256"],
        str(config): encoder["config_sha256"],
        str(env_path): "89597ed4d974e5874c7bb2ffb42545e6c0bb6a9ac796a5839a4b1902917ef338",
    }
    proof = {
        "status": "native_gpu_prerequisites_cpu_hash_verified",
        "job_id": "100",
        "assets": {p: {"sha256": h, "stat": stamp(Path(p))} for p, h in assets.items()},
        "scientific_files": scientific,
        "accepted_input_bindings": binding,
        "plan_sha256": digest(json.dumps(plan, sort_keys=True, separators=(",", ":")).encode()),
        "admission_source_sha256": digest(admission.encode()),
    }
    save(control / "gpu-prerequisites-verified.json", proof)
    return SimpleNamespace(
        base=base,
        control=control,
        request=request,
        proof=proof,
        save=save,
        rows=[],
        cpu_terminal="COMPLETED|0:0",
        submitted=[],
    )


def execute(program: str, remote, monkeypatch, *, sbatch_returncode=0):
    def check_output(command, **kwargs):
        if command[0] == "squeue":
            return "\n".join(remote.rows)
        assert command[0] == "sacct"
        job = command[command.index("-j") + 1]
        terminal = "COMPLETED|0:0" if job == "865906" else remote.cpu_terminal
        return f"{job}|{terminal}\n"

    def run(command, **kwargs):
        if command[0] == "bash":
            assert command[1] == "-n"
            return SimpleNamespace(returncode=0)
        assert command[0] == "sbatch"
        assert (remote.control / "gpu-dispatch-intent.json").exists()
        remote.submitted.append(command)
        return SimpleNamespace(returncode=sbatch_returncode, stdout="101\n", stderr="")

    monkeypatch.setattr(subprocess, "check_output", check_output)
    monkeypatch.setattr(subprocess, "run", run)
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps(remote.request)))
    output = io.StringIO()
    monkeypatch.setattr(sys, "stdout", output)
    exec(compile(program.replace(BASE, str(remote.base)), str(SCRIPT), "exec"), {})
    return json.loads(output.getvalue())


@pytest.mark.parametrize("duplicate", ["gpu-dispatch-intent.json", "gpu-dispatch-result.json"])
def test_remote_intent_or_result_refuses_duplicate(remote, monkeypatch, duplicate):
    remote.save(remote.control / duplicate, {})
    assert (
        execute(dispatcher.CHECK, remote, monkeypatch)["status"]
        == "already_attempted_reconcile_read_only"
    )
    with pytest.raises(AssertionError):
        execute(dispatcher.DISPATCH, remote, monkeypatch)
    assert remote.submitted == []


def test_actual_same_job_refuses_new_dispatch(remote, monkeypatch):
    remote.rows = ["101|PENDING|rsna-native384-recovery-pilot|gres/gpu:1|(Resources)"]
    result = execute(dispatcher.CHECK, remote, monkeypatch)
    assert result["status"] == "already_attempted_reconcile_read_only"
    assert remote.submitted == []


def test_two_live_gpu_roles_never_reserve_gpu(remote, monkeypatch):
    remote.rows = [f"{j}|RUNNING|production|gres/gpu:1|evc{i}" for j, i in [(1, 28), (2, 31)]]
    result = execute(dispatcher.CHECK, remote, monkeypatch)
    assert result["status"] == "waiting_for_gpu_role" and result["gpu_reserved"] is False
    assert remote.submitted == []


@pytest.mark.parametrize(
    "kind",
    [
        "omitted_asset",
        "changed_asset_sha",
        "wrong_plan",
        "wrong_source",
        "wrong_input",
        "wrong_admission",
    ],
)
def test_incomplete_or_changed_cpu_proof_refuses_sbatch(remote, monkeypatch, kind):
    proof = copy.deepcopy(remote.proof)
    if kind == "omitted_asset":
        proof["assets"].pop(next(iter(proof["assets"])))
    elif kind == "changed_asset_sha":
        proof["assets"][next(iter(proof["assets"]))]["sha256"] = "0" * 64
    elif kind == "wrong_plan":
        proof["plan_sha256"] = "0" * 64
    elif kind == "wrong_source":
        proof["scientific_files"] = {"rsna/train.py": "0" * 64}
    elif kind == "wrong_input":
        proof["accepted_input_bindings"] = {"training.json": {"sha256": "0" * 64}}
    else:
        proof["admission_source_sha256"] = "0" * 64
    remote.save(remote.control / "gpu-prerequisites-verified.json", proof)
    with pytest.raises(AssertionError):
        execute(dispatcher.DISPATCH, remote, monkeypatch)
    assert remote.submitted == [] and not (remote.control / "gpu-dispatch-intent.json").exists()


@pytest.mark.parametrize("terminal", ["RUNNING|0:0", "FAILED|1:0", "COMPLETED|1:0"])
def test_cpu_proof_requires_actual_successful_terminal_job(remote, monkeypatch, terminal):
    remote.cpu_terminal = terminal
    with pytest.raises(AssertionError):
        execute(dispatcher.DISPATCH, remote, monkeypatch)
    assert remote.submitted == []


@pytest.mark.parametrize("change", ["different_bytes", "symlink", "same_bytes_new_inode"])
def test_changed_payload_after_cpu_hash_proof_refuses_sbatch(remote, monkeypatch, change):
    path = remote.control / "payloads/cache.tar"
    if change == "different_bytes":
        path.write_bytes(b"other")
    elif change == "symlink":
        other = path.with_suffix(".other")
        path.rename(other)
        path.symlink_to(other)
    else:
        replacement = path.with_suffix(".new")
        replacement.write_bytes(path.read_bytes())
        replacement.replace(path)
    with pytest.raises(AssertionError):
        execute(dispatcher.DISPATCH, remote, monkeypatch)
    assert remote.submitted == []


def test_changed_small_script_refuses_sbatch(remote, monkeypatch):
    (remote.control / "pilot.sbatch").write_text("#!/bin/bash\nexit 1\n")
    with pytest.raises(AssertionError):
        execute(dispatcher.DISPATCH, remote, monkeypatch)
    assert remote.submitted == []


@pytest.mark.parametrize(
    "path", ["code-delta-recovery-c9fcc7e/rsna/train.py", "code-stopfix-1f486765/manifest.json"]
)
def test_changed_scientific_code_or_accepted_manifest_refuses_sbatch(remote, monkeypatch, path):
    (remote.base / path).write_bytes(b"changed")
    with pytest.raises(AssertionError):
        execute(dispatcher.DISPATCH, remote, monkeypatch)
    assert remote.submitted == []


def test_small_script_symlink_is_not_accepted_even_with_identical_bytes(remote, monkeypatch):
    script = remote.control / "pilot.sbatch"
    target = script.with_suffix(".original")
    script.rename(target)
    script.symlink_to(target)
    with pytest.raises(AssertionError):
        execute(dispatcher.DISPATCH, remote, monkeypatch)
    assert remote.submitted == []


def test_fresh_admission_refusal_preserves_no_dispatch(remote, monkeypatch):
    source = remote.request["admission_source"].replace("True", "False")
    remote.request["admission_source"] = source
    remote.proof["admission_source_sha256"] = digest(source.encode())
    remote.save(remote.control / "gpu-prerequisites-verified.json", remote.proof)
    with pytest.raises(AssertionError):
        execute(dispatcher.DISPATCH, remote, monkeypatch)
    assert remote.submitted == [] and not (remote.control / "gpu-dispatch-intent.json").exists()


def test_one_sbatch_attempt_even_after_rejected_transport(remote, monkeypatch):
    with pytest.raises(AssertionError):
        execute(dispatcher.DISPATCH, remote, monkeypatch, sbatch_returncode=1)
    assert len(remote.submitted) == 1
    result = json.loads((remote.control / "gpu-dispatch-result.json").read_text())
    assert result["returncode"] == 1
    with pytest.raises(AssertionError):
        execute(dispatcher.DISPATCH, remote, monkeypatch)
    assert len(remote.submitted) == 1


def test_successful_attempt_has_intent_and_one_result_then_rejects_repeat(remote, monkeypatch):
    result = execute(dispatcher.DISPATCH, remote, monkeypatch)
    assert result["job_id"] == 101 and len(remote.submitted) == 1
    with pytest.raises(AssertionError):
        execute(dispatcher.DISPATCH, remote, monkeypatch)
    assert len(remote.submitted) == 1


def test_default_cli_is_read_only_without_cpu_or_gpu_intent(tmp_path, monkeypatch):
    calls = []

    def ssh(*args, **kwargs):
        calls.append(args[1])
        assert args[1] == [dispatcher.CPU_PYTHON, "-c", dispatcher.CHECK]
        return {"status": "ready_for_cpu_prerequisites"}

    monkeypatch.setattr(dispatcher, "FOLDER", tmp_path)
    monkeypatch.setattr(dispatcher, "request_data", lambda: {})
    monkeypatch.setattr(dispatcher, "ssh", ssh)
    # macOS limits UNIX socket path length; pytest's full path can exceed it.
    with tempfile.TemporaryDirectory(prefix="np-dispatch-", dir="/tmp") as directory:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as channel:
            path = Path(directory) / "socket"
            channel.bind(str(path))
            monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--ssh-control", str(path)])
            dispatcher.main()
    assert len(calls) == 1 and not list(tmp_path.glob("*intent.json"))
