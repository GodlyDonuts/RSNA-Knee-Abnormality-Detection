"""Synthetic guards only: these tests never authenticate or mutate Kaggle."""

import importlib.util
import json
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import torch

SPEC = importlib.util.spec_from_file_location(
    "kaggle_workflow", Path(__file__).resolve().parents[1] / "scripts/kaggle/workflow.py"
)
assert SPEC is not None and SPEC.loader is not None
workflow = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(workflow)


def account(remaining=5, today=0):
    return {
        "checked_at": workflow.utcnow(),
        "entered": True,
        "num_today": today,
        "num_allowed_now": remaining,
    }


def entry(sha="a" * 64):
    return {
        "checkpoint_sha256": sha,
        "state": "pushed",
        "kernel_version": 7,
        "operation": {
            "checkpoint_sha256": sha,
            "kernel_slug": "owner/rsna-a",
            "dataset_slug": "owner/rsna-data-a",
            "manifest_sha256": "b" * 64,
            "kernel_source_sha256": workflow.hashlib.sha256(b"source").hexdigest(),
        },
    }


class FakeRemote:
    username = "owner"

    def __init__(self, remaining=5, today=0, uncertain=False):
        self.remaining = remaining
        self.today = today
        self.uncertain = uncertain
        self.writes = []
        self.observed = []
        self.version = 1
        self.private = True
        self.status = "COMPLETE"
        self.machine_shape = "NvidiaTeslaT4"
        self.code = "source"
        self.reported_sha = "a" * 64

    def dataset(self, slug):
        return {"id": 12, "isPrivate": self.private, "currentVersionNumber": self.version}

    def kernel(self, method, slug, version):
        if method == "GetKernelSessionStatus":
            return {"status": self.status}
        return {
            "metadata": {
                "isPrivate": True,
                "enableInternet": False,
                "machineShape": self.machine_shape,
                "competitionDataSources": [workflow.COMPETITION],
                "datasetDataSources": ["owner/rsna-data-a"],
            },
            "blob": {"source": self.code},
        }

    def output_json(self, slug, version, name):
        return {
            "status": "offline_inference_complete",
            "bundle_manifest_sha256": "b" * 64,
            "checkpoint_sha256": self.reported_sha,
            "studies": 3,
            "public_example_only": True,
        }

    def account(self):
        return account(self.remaining, self.today)

    def competition(self, method, body):
        self.writes.append((method, body))
        self.observed.append(
            {"ref": 42, "description": body["submissionDescription"], "status": "PENDING"}
        )
        if self.uncertain:
            raise workflow.RemoteError("uncertain simulated timeout")
        return {"ref": 42}

    def submissions(self):
        return self.observed


def test_account_ledger_identity_and_reopen(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    with workflow.ledger(path, "owner") as db:
        workflow.save_entry(db, entry(), "test")
    assert path.stat().st_mode & 0o077 == 0
    with workflow.ledger(path, "owner") as db:
        assert workflow.read_entry(db, "a" * 64)["state"] == "pushed"
    with pytest.raises(ValueError, match="another account"):
        with workflow.ledger(path, "someone-else"):
            pass


def test_live_allowance_includes_other_account_submissions(tmp_path):
    with workflow.ledger(tmp_path / "ledger", "owner") as db:
        assert workflow.allowance(db, account(2, 3)) == 2
        assert workflow.allowance(db, account(0, 5)) == 0
        for i in range(5):
            record = entry(str(i) * 64)
            record.update(state="submitted", attempted_at=workflow.utcnow())
            workflow.save_entry(db, record, "other run")
        assert workflow.allowance(db, account()) == 0


@pytest.mark.parametrize(
    "field,value", [("entered", False), ("checked_at", "2001-01-01T00:00:00+00:00")]
)
def test_no_submission_without_entered_account_fresh_quota(tmp_path, field, value):
    with workflow.ledger(tmp_path / "ledger", "owner") as db:
        observed = account()
        observed[field] = value
        with pytest.raises(ValueError):
            workflow.allowance(db, observed)


def test_quota_refresh_required_after_utc_boundary(tmp_path):
    with workflow.ledger(tmp_path / "ledger", "owner") as db:
        before = datetime(2026, 9, 22, 23, 59, tzinfo=timezone.utc)
        observed = account(0, 5)
        observed["checked_at"] = before.isoformat()
        with pytest.raises(ValueError, match="stale"):
            workflow.allowance(db, observed, now=before + timedelta(minutes=6))


def test_idempotent_exact_code_submission_never_uploads_sample_csv(tmp_path):
    remote = FakeRemote()
    with workflow.ledger(tmp_path / "ledger", "owner") as db:
        workflow.save_entry(db, entry(), "test")
        first = workflow.submit(remote, db, "a" * 64)
        second = workflow.submit(remote, db, "a" * 64)
        assert first["submission_id"] == second["submission_id"] == 42
        assert len(remote.writes) == 1
        method, body = remote.writes[0]
        assert method == "CreateCodeSubmission"
        assert body["kernelVersion"] == 7
        assert body["fileName"] == "submission.csv"
        assert "blobFileTokens" not in body


def test_uncertain_response_is_durable_and_blocks_every_run_until_read_only_reconcile(tmp_path):
    path = tmp_path / "ledger"
    remote = FakeRemote(uncertain=True)
    with workflow.ledger(path, "owner") as db:
        workflow.save_entry(db, entry(), "test")
        with pytest.raises(workflow.RemoteError):
            workflow.submit(remote, db, "a" * 64)
    with workflow.ledger(path, "owner") as db:
        assert workflow.read_entry(db, "a" * 64)["state"] == "submit_uncertain"
        with pytest.raises(ValueError, match="Uncertain"):
            workflow.allowance(db, account())
        with pytest.raises(ValueError):
            workflow.submit(remote, db, "a" * 64)
        fixed = workflow.reconcile_submission(remote, db, "a" * 64)
        assert fixed["submission_id"] == 42
        workflow.submit(remote, db, "a" * 64)
    assert len(remote.writes) == 1


def test_missing_uncertain_submission_never_implies_safe_retry(tmp_path):
    remote = FakeRemote(uncertain=True)
    with workflow.ledger(tmp_path / "ledger", "owner") as db:
        workflow.save_entry(db, entry(), "test")
        with pytest.raises(workflow.RemoteError):
            workflow.submit(remote, db, "a" * 64)
        remote.observed.clear()
        with pytest.raises(ValueError, match="keep blocked"):
            workflow.reconcile_submission(remote, db, "a" * 64)
        assert workflow.read_entry(db, "a" * 64)["state"] == "submit_uncertain"


def test_reconciliation_cannot_overwrite_concurrent_success(tmp_path):
    path = tmp_path / "ledger"
    remote = FakeRemote()
    with workflow.ledger(path, "owner") as first:
        workflow.save_entry(first, entry(), "test")
        with workflow.ledger(path, "owner") as second:
            second.execute("PRAGMA busy_timeout=0")
            original = remote.kernel
            attempted = []

            def interleaved(method, slug, version):
                if not attempted:
                    attempted.append(True)
                    with pytest.raises(sqlite3.OperationalError, match="locked"):
                        workflow.submit(remote, second, "a" * 64)
                return original(method, slug, version)

            remote.kernel = interleaved
            workflow.reconcile_push(remote, first, "a" * 64, 7)
            workflow.submit(remote, second, "a" * 64)
            assert workflow.read_entry(first, "a" * 64)["state"] == "submitted"
            with pytest.raises(ValueError):
                workflow.reconcile_push(remote, first, "a" * 64, 7)
    assert len(remote.writes) == 1


@pytest.mark.parametrize(
    "field,value",
    [
        ("version", 2),
        ("machine_shape", "NvidiaTeslaP100"),
        ("private", False),
        ("status", "RUNNING"),
        ("code", "changed"),
        ("reported_sha", "f" * 64),
        ("username", "different"),
    ],
)
def test_remote_bundle_and_account_guards_before_mutation(tmp_path, field, value):
    remote = FakeRemote()
    setattr(remote, field, value)
    with workflow.ledger(tmp_path / "ledger", "owner") as db:
        workflow.save_entry(db, entry(), "test")
        with pytest.raises(ValueError):
            workflow.submit(remote, db, "a" * 64)
    assert not remote.writes


def test_overflow_preserves_queue_and_unknown_reset(tmp_path):
    remote = FakeRemote(remaining=0, today=5)
    with workflow.ledger(tmp_path / "ledger", "owner") as db:
        workflow.save_entry(db, entry(), "test")
        result = workflow.submit(remote, db, "a" * 64)
        assert result["state"] == "pushed"
        assert result["next_quota_check_utc"].endswith("+00:00")
    assert not remote.writes


def test_bundle_rejects_tamper_and_traversal(tmp_path):
    (tmp_path / "weights").write_bytes(b"abc")
    manifest = {"files": {"weights": {"sha256": workflow.digest(tmp_path / "weights"), "bytes": 3}}}
    workflow.verify_payload(tmp_path, manifest)
    (tmp_path / "weights").write_bytes(b"xyz")
    with pytest.raises(ValueError):
        workflow.verify_payload(tmp_path, manifest)
    with pytest.raises(ValueError):
        workflow.relative_file(tmp_path, "../outside")


@pytest.mark.parametrize(
    "change",
    [
        {"epoch": 1},
        {"epoch": 12},
        {"submission_eligible": False},
        {"diagnostic_only": True},
        {"optimizer_steps": 0},
        {"submission_every_epochs": 1},
    ],
)
def test_queue_rejects_unscheduled_untrained_diagnostic_sidecars_before_loading_weights(
    tmp_path, change
):
    record = {
        "status": "scheduled_epoch_snapshot",
        "epoch": 2,
        "submission_every_epochs": 2,
        "submission_eligible": True,
        "diagnostic_only": False,
        "optimizer_steps": 5,
        **change,
    }
    sidecar = tmp_path / "epoch-002.json"
    sidecar.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="Only trained"):
        workflow.validate_snapshot(sidecar)


def test_snapshot_internal_provenance_is_checked_not_only_sidecar(tmp_path, monkeypatch):
    import rsna.competitive.infer as inference

    repo = tmp_path / "repo"
    (repo / "rsna").mkdir(parents=True)
    (repo / "rsna/example.py").write_text("x = 1\n")
    binding = {
        "code_sha256": {"example.py": workflow.digest(repo / "rsna/example.py")},
        "config": {
            "fold": 0,
            "seed": 42,
            "submission_every_epochs": 2,
            "encoder": {"architecture": "dinov3_l16"},
        },
        "release": {"binding_sha256": "b" * 64},
    }
    # Root's real release/checkpoint validation is separately tested. This small
    # fixture isolates our additional sidecar/internal-state consistency guards.
    monkeypatch.setattr(inference, "validate_inference_checkpoint", lambda *a, **kw: binding)
    checkpoint = tmp_path / "epoch-002.pt"
    torch.save({"scheduled_epoch": True, "epoch": 2, "next_update": 0, "step": 3}, checkpoint)
    meta = {
        "status": "scheduled_epoch_snapshot",
        "epoch": 2,
        "submission_every_epochs": 2,
        "submission_eligible": True,
        "diagnostic_only": False,
        "optimizer_steps": 3,
        "checkpoint_file": checkpoint.name,
        "checkpoint_sha256": workflow.digest(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "binding_sha256": workflow.object_hash(binding),
        "release_binding_sha256": "b" * 64,
        "fold": 0,
        "seed": 42,
    }
    sidecar = tmp_path / "epoch-002.json"
    sidecar.write_text(json.dumps(meta))
    workflow.validate_snapshot(sidecar, repo)
    meta["optimizer_steps"] = 4
    sidecar.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="contents"):
        workflow.validate_snapshot(sidecar, repo)


def test_offline_notebook_requires_prepared_binding_before_any_process():
    source = Path(workflow.__file__).with_name("notebook_entry.py").read_text()
    namespace = {"__name__": "test_notebook"}
    exec(compile(source, "notebook_entry.py", "exec"), namespace)
    with pytest.raises(ValueError, match="manifest binding"):
        namespace["run"]()


def test_version_inspection_rejects_concurrent_new_kernel_version():
    remote = workflow.Remote.__new__(workflow.Remote)
    calls = []

    def fake_call(service, method, body):
        calls.append(method)
        assert "versionLabel" not in body
        if method == "GetKernel":
            return {"metadata": {"currentVersionNumber": 7 if len(calls) == 1 else 8}}
        return {"status": "COMPLETE"}

    remote.call = fake_call
    with pytest.raises(ValueError, match="changed during"):
        remote.kernel("GetKernelSessionStatus", "owner/kernel", 7)


def test_complete_local_bundle_binding_and_no_extra_upload_files(tmp_path, monkeypatch):
    repo = tmp_path / "repo"
    (repo / "rsna").mkdir(parents=True)
    (repo / "rsna/__init__.py").write_text("")
    (repo / "pyproject.toml").write_text("synthetic project fixture\n")
    sidecar = tmp_path / "epoch.json"
    sidecar.write_text("{}")
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic fixture, never a real submitted checkpoint")
    config = tmp_path / "config.json"
    config.write_text("{}")
    spec = tmp_path / "encoder.json"
    spec.write_text(
        json.dumps(
            {
                "architecture": "dinov3_l16",
                "config_path": str(config),
                "config_sha256": workflow.digest(config),
                "weights_path": "/not-copied/pretrained.bin",
                "weights_sha256": "c" * 64,
            }
        )
    )
    meta = {
        "checkpoint_sha256": workflow.digest(checkpoint),
        "epoch": 2,
        "release_binding_sha256": "b" * 64,
        "binding_sha256": "c" * 64,
    }
    provenance = {
        key: value
        for key, value in json.loads(spec.read_text()).items()
        if not key.endswith("_path")
    }
    binding = {"config": {"encoder": provenance}}
    monkeypatch.setattr(workflow, "validate_snapshot", lambda *a: (meta, binding, checkpoint))
    monkeypatch.setattr(workflow, "FLOOR", 0)
    wheels = tmp_path / "wheelhouse"
    wheels.mkdir()
    (wheels / "fixture.whl").write_bytes(b"fixture")
    (wheels / "requirements.lock").write_text("fixture==1 --hash=sha256:" + "a" * 64)
    (wheels / "acceptance.json").write_text(
        json.dumps(
            {
                "status": "cpu_offline_install_passed",
                "pyproject_sha256": workflow.digest(repo / "pyproject.toml"),
            }
        )
    )
    files = {
        p.name: {"sha256": workflow.digest(p), "bytes": p.stat().st_size} for p in wheels.iterdir()
    }
    (wheels / "wheelhouse.json").write_text(
        json.dumps(
            {
                "platform": "linux_x86_64",
                "python": "3.12",
                "cuda": "13.0",
                "offline_install_verified": True,
                "files": files,
                "pyproject_sha256": workflow.digest(repo / "pyproject.toml"),
            }
        )
    )
    license_file = tmp_path / "LICENSE.md"
    license_file.write_text("Synthetic licensing fixture for local assembler tests only.")
    output = tmp_path / "prepared"
    slug = "owner/rsna-" + meta["checkpoint_sha256"][:12]
    result = workflow.prepare_bundle(
        sidecar,
        spec,
        wheels,
        output,
        slug,
        slug,
        license_file,
        "https://github.com/facebookresearch/dinov3/blob/" + "d" * 40 + "/LICENSE.md",
        workflow.digest(license_file),
        repo,
    )
    assert workflow.load_operation(output / "operation.json") == result
    assert "weights_path" not in json.loads((output / "dataset/encoder.json").read_text())
    assert (output / "dataset/DINOv3-LICENSE.md").exists()
    (output / "dataset/unexpected.txt").write_text("must not upload")
    with pytest.raises(ValueError, match="extraneous"):
        workflow.load_operation(output / "operation.json")


def test_push_inflight_cannot_be_reconciled_and_submitted_concurrently(tmp_path, monkeypatch):
    path = tmp_path / "ledger"
    operation_path = tmp_path / "operation.json"
    operation_path.write_text("{}")
    operation = entry()["operation"]
    monkeypatch.setattr(workflow, "load_operation", lambda path: operation)
    monkeypatch.setattr(workflow, "kernel_request", lambda *args: {})
    remote = FakeRemote()
    with workflow.ledger(path, "owner") as first:
        queued = entry()
        queued["state"] = "queued"
        workflow.save_entry(first, queued, "test")
        with workflow.ledger(path, "owner") as second:
            second.execute("PRAGMA busy_timeout=0")

            def inflight(service, method, body):
                assert method == "SaveKernel"
                assert workflow.read_entry(second, "a" * 64)["state"] == "push_intent"
                with pytest.raises(sqlite3.OperationalError, match="locked"):
                    workflow.reconcile_push(remote, second, "a" * 64, 7)
                return {"versionNumber": 7}

            remote.call = inflight
            workflow.push(remote, first, operation_path)
            workflow.submit(remote, second, "a" * 64)
            assert workflow.read_entry(first, "a" * 64)["state"] == "submitted"
            with pytest.raises(ValueError):
                workflow.push(remote, first, operation_path)
    assert len(remote.writes) == 1


def test_wheel_metadata_ignores_nested_vendored_distributions(tmp_path):
    import zipfile

    spec = importlib.util.spec_from_file_location(
        "wheel_builder", Path(workflow.__file__).with_name("build_wheelhouse.py")
    )
    assert spec is not None and spec.loader is not None
    builder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(builder)
    wheel = tmp_path / "package.whl"
    with zipfile.ZipFile(wheel, "w") as handle:
        handle.writestr("package-1.dist-info/METADATA", "Name: package\nVersion: 1\n")
        handle.writestr("package/_vendor/other-2.dist-info/METADATA", "Name: other\nVersion: 2\n")
    assert builder.wheel_metadata(wheel)["Name"] == "package"
    with zipfile.ZipFile(wheel, "a") as handle:
        handle.writestr("another-2.dist-info/METADATA", "Name: another\nVersion: 2\n")
    with pytest.raises(ValueError, match="Ambiguous"):
        builder.wheel_metadata(wheel)


def test_kernel_request_pins_verified_t4_machine_shape(tmp_path):
    directory = tmp_path / "kernel"
    directory.mkdir()
    (directory / "run.py").write_text("source")
    (directory / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": "owner/kernel",
                "title": "kernel",
                "is_private": True,
                "enable_internet": False,
                "dataset_sources": ["owner/data"],
                "competition_sources": [workflow.COMPETITION],
            }
        )
    )
    result = workflow.kernel_request(
        tmp_path / "operation.json", {"kernel_slug": "owner/kernel", "dataset_slug": "owner/data"}
    )
    assert result["machineShape"] == "NvidiaTeslaT4" and result["enableGpu"] is True
