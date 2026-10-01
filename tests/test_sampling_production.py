"""Full sampling jobs must retain accepted inputs, paired roles and signal recovery."""

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from rsna.competitive.backbones import _encoder_provenance
from scripts.newton import dispatch_sampling_production as dispatch
from scripts.newton import sampling_production_runtime as runtime

ROOT = Path(__file__).resolve().parents[1]
AUDIT = ROOT / "initiative/audits/stratified-slices-production-launch-v1"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


publisher = load(ROOT / "scripts/hf/publish_sampling_epoch.py", "sampling_publisher_test")
epoch_dispatch = load(
    ROOT / "scripts/hf/dispatch_sampling_epoch.py", "sampling_epoch_dispatch_test"
)


def actual_recipe(policy):
    config = json.loads((AUDIT / (policy + "-config.json")).read_text())
    architecture = json.loads(
        (ROOT / "configs/competitive/weights/dinov3-large.config.json").read_text()
    )
    return {**config, "encoder": _encoder_provenance(config["encoder"], architecture, "unused_vit")}


@pytest.mark.parametrize("policy", ["uniform", "stratified"])
def test_actual_configs_match_publication_factory_and_fixed_source(policy):
    assert actual_recipe(policy) == publisher.expected_training_config(policy)
    assert epoch_dispatch.PREFIX == publisher.PREFIX
    assert epoch_dispatch.PUBLISHER_SHA == publisher.sha(
        ROOT / "scripts/hf/publish_sampling_epoch.py"
    )
    assert runtime.SOURCE_GIT == publisher.TRAINING_GIT
    assert publisher.sha(AUDIT / "source.json") == publisher.SOURCE_MANIFEST_SHA
    manifest = json.loads((AUDIT / "files.json").read_text())
    for row in manifest["files"].values():
        assert (ROOT / row["local_path"]).stat().st_size == row["bytes"]
        assert publisher.sha(ROOT / row["local_path"]) == row["sha256"]
    script = (AUDIT / (policy + ".sbatch.txt")).read_text()
    assert "--gres=gpu:nvidia_h100_pcie:1" in script and "--cpus-per-task=2" in script
    assert "--signal=B:USR1@300" in script and "--time=24:00:00" in script
    assert "sampling_production_runtime.py " + policy in script
    assert "--job-name=rsna-sampling-" + policy + "-full" in script
    assert "--nodelist=evc31" not in script
    assert script.count("#SBATCH --exclude=") == 1
    assert "--exclude=evc22," in script


@pytest.fixture
def accepted_state():
    return {
        "status": "accepted_inputs_and_funded_pair",
        "staged": True,
        "projected_remaining_gib": 21,
        "projected_remaining_files": 5700,
        "actual_gpu_jobs": [],
        "roles": {
            p: {"output_exists": False, "intent_exists": False, "result": None}
            for p in ("uniform", "stratified")
        },
    }


@pytest.mark.parametrize("policy", ["uniform", "stratified"])
def test_admission_permits_only_unused_accepted_funded_arm(accepted_state, policy):
    dispatch.require_dispatch(accepted_state, policy)
    accepted_state["roles"][policy]["intent_exists"] = True
    with pytest.raises(FileExistsError):
        dispatch.require_dispatch(accepted_state, policy)


@pytest.mark.parametrize(
    "key,value",
    [
        ("status", "pending_CPU"),
        ("staged", False),
        ("projected_remaining_gib", 19.9),
        ("projected_remaining_files", 4999),
        ("actual_gpu_jobs", ["one", "two"]),
    ],
)
def test_missing_prerequisites_never_reserve_GPU(accepted_state, key, value):
    accepted_state[key] = value
    with pytest.raises(ValueError):
        dispatch.require_dispatch(accepted_state, "stratified")


@pytest.fixture
def payload(tmp_path, monkeypatch):
    """Synthetic bytes exercise refusal gates; they are never training evidence."""
    control = tmp_path / "control"
    control.mkdir()
    source = json.loads((AUDIT / "source.json").read_text())
    (control / "source.json").write_text(json.dumps(source))
    monkeypatch.setattr(publisher, "SOURCE_MANIFEST_SHA", publisher.sha(control / "source.json"))
    run = tmp_path / "run"
    folder = run / "epoch-checkpoints"
    folder.mkdir(parents=True)
    raw_config = json.loads((AUDIT / "stratified-config.json").read_text())
    (control / "stratified-config.json").write_text(json.dumps(raw_config))
    files = {
        n: {"sha256": publisher.sha(control / n), "bytes": (control / n).stat().st_size}
        for n in ("source.json", "stratified-config.json")
    }
    (control / "files.json").write_text(
        json.dumps({"runs": {"stratified": str(run)}, "files": files})
    )
    initial = json.loads(
        (
            ROOT
            / "initiative/audits/stratified-slices-pilot-launch-v1/terminal/initialization-verified.json"
        ).read_text()
    )
    provenance = dict(
        status="full_sampling_inputs_and_actual_initialization_accepted",
        source_git=publisher.TRAINING_GIT,
        policy="stratified",
        source_files=source["files"],
        source_manifest_sha256=publisher.SOURCE_MANIFEST_SHA,
        accepted_data_masks_or_folds_changed=False,
        base_weights_reused_not_diagnostic_checkpoint=True,
        strict_determinism=True,
        cache_side=256,
        cache_members=48772,
        training_studies=2247,
        outer_studies=1156,
        checkpoint_cap_bytes=runtime.CHECKPOINT_CAP,
        initialization=initial,
        CPU_proof_sha256=publisher.CPU_PROOF_SHA,
        pilot_proof_sha256=publisher.PILOT_PROOF_SHA,
        config_sha256=publisher.sha(control / "stratified-config.json"),
        payload_manifest_sha256=publisher.sha(control / "files.json"),
    )
    license_file = tmp_path / "LICENSE"
    license_file.write_bytes(b"synthetic license fixture")
    monkeypatch.setattr(publisher, "LICENSE_SHA", publisher.sha(license_file))
    checkpoint = folder / "epoch-001.pt"
    checkpoint.write_bytes(b"Synthetic checkpoint bytes; not a trained model")
    binding = {
        **publisher.EXPECTED_BINDING_FIELDS,
        "config": actual_recipe("stratified"),
        "code_sha256": {
            n.removeprefix("rsna/"): row["sha256"]
            for n, row in source["files"].items()
            if n.startswith("rsna/") and n.endswith(".py")
        },
        "release": {"binding_sha256": publisher.RELEASE_SHA},
    }
    record = dict(
        binding=binding,
        training_studies=2247,
        outer_studies=1156,
        encoder_trainable_parameters=10223616,
    )
    meta = dict(
        status="scheduled_epoch_snapshot",
        epoch=1,
        optimizer_steps=281,
        checkpoint_file=checkpoint.name,
        checkpoint_bytes=checkpoint.stat().st_size,
        checkpoint_sha256=publisher.sha(checkpoint),
        fold=0,
        seed=20260922,
        diagnostic_only=False,
        submission_eligible=True,
        submission_every_epochs=1,
        release_binding_sha256=publisher.RELEASE_SHA,
    )
    return control, run, license_file, record, meta, provenance


def save_payload(payload):
    control, run, license_file, record, meta, provenance = payload
    meta["binding_sha256"] = hashlib.sha256(
        publisher.canonical(record["binding"]).encode()
    ).hexdigest()
    (run / "run.json").write_text(json.dumps(record))
    (run / "sampling-input-provenance.json").write_text(json.dumps(provenance))
    snapshot = run / "epoch-checkpoints/epoch-001.json"
    snapshot.write_text(json.dumps(meta))
    return snapshot, run, license_file, control


def test_source_bound_completed_epoch_accepts_exact_sampling_role(payload):
    manifest, files = publisher.source_files(*save_payload(payload))
    assert manifest["slice_sampling"] == "stratified"
    assert manifest["training_git_commit"] == runtime.SOURCE_GIT
    assert "sampling-input-provenance.json" in files


@pytest.mark.parametrize(
    "field,value",
    [
        ("cache_side", 384),
        ("strict_determinism", False),
        ("source_git", "wrong"),
        ("policy", "uniform"),
        ("CPU_proof_sha256", "bad"),
        ("pilot_proof_sha256", "bad"),
        ("base_weights_reused_not_diagnostic_checkpoint", False),
    ],
)
def test_wrong_input_roles_are_never_preserved_as_sampling_epochs(payload, field, value):
    payload[-1][field] = value
    with pytest.raises((ValueError, FileNotFoundError)):
        publisher.source_files(*save_payload(payload))


@pytest.mark.parametrize(
    "field,value",
    [
        ("optimizer_steps", 280),
        ("diagnostic_only", True),
        ("submission_eligible", False),
        ("checkpoint_file", "../checkpoint.pt"),
    ],
)
def test_incomplete_or_diagnostic_checkpoints_are_refused(payload, field, value):
    payload[-2][field] = value
    with pytest.raises(ValueError):
        publisher.source_files(*save_payload(payload))


@pytest.mark.skipif(os.name != "posix", reason="Scheduler signal forwarding is POSIX")
def test_actual_parent_signal_reaches_supervisor_and_checkpoint_worker(tmp_path):
    helpers = load(ROOT / "tests/test_newton_launch.py", "sampling_signal_fixture")
    ready, checkpoint, worker, launcher = helpers.signal_workers(tmp_path)
    supervisor = ROOT / "scripts/newton/supervise.py"
    command = [
        sys.executable,
        str(supervisor),
        "--grace-seconds",
        "5",
        "--checkpoint",
        str(checkpoint),
        "--",
        sys.executable,
        str(launcher),
        str(worker),
        str(checkpoint),
        str(ready),
    ]
    parent_source = f"""
import importlib.util,json
from pathlib import Path
s=importlib.util.spec_from_file_location('runtime',{str(ROOT / "scripts/newton/sampling_production_runtime.py")!r});m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
stop=m.StopForwarder();stop.install();code=stop.supervise({command!r},Path({str(tmp_path)!r}),Path({str(tmp_path)!r}));print(json.dumps({{'returned':code,'forwarded':stop.sent}}));raise SystemExit(code)
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", parent_source],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and parent.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists()
        parent.send_signal(signal.SIGUSR1)
        stdout, stderr = parent.communicate(timeout=15)
        assert parent.returncode == 75, (stdout, stderr)
        assert checkpoint.read_bytes() == b"atomic-test-checkpoint"
        rows = [json.loads(s) for s in stdout.splitlines()]
        assert rows[-1] == {"returned": 75, "forwarded": True}
        assert rows[-2]["checkpoint_stop_verified"] is True
    finally:
        if parent.poll() is None:
            parent.send_signal(signal.SIGTERM)
            parent.wait(timeout=20)


def test_recovery_stage_intent_stays_in_its_distinct_audit(tmp_path, monkeypatch):
    """A preexisting original intent must not swallow a distinct zero-update recovery."""
    audit = tmp_path / "audit"
    recovery = audit / "uniform-evc31-recovery-v2"
    recovery.mkdir(parents=True)
    (audit / "stage-intent.json").write_text("original unchanged")
    (recovery / "files.json").write_text(json.dumps({"files": {}, "runs": {}}))
    monkeypatch.setattr(dispatch, "AUDIT", audit)
    monkeypatch.setattr(Path, "is_socket", lambda _p: True)
    calls = []

    def remote(_socket, code, payload):
        calls.append(payload)
        if len(calls) == 1:
            return {"status": "accepted_inputs_and_funded_pair", "staged": False}
        return {"status": "staged", "manifest_sha256": payload["manifest_sha256"]}

    monkeypatch.setattr(dispatch, "ssh", remote)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "dispatch",
            "stage",
            "--revision",
            "uniform-evc31-v2",
            "--ssh-control",
            str(tmp_path / "socket"),
        ],
    )
    dispatch.main()
    assert (audit / "stage-intent.json").read_text() == "original unchanged"
    assert (recovery / "stage-intent.json").is_file()
    assert (recovery / "stage-verified.json").is_file()
    assert calls[0]["control"].endswith("uniform-evc31-v2")
