import copy
import importlib.util
import json
import sys
import types
from io import StringIO
from pathlib import Path
from unittest.mock import Mock

import pytest

SPEC = importlib.util.spec_from_file_location(
    "compact_publisher", Path(__file__).resolve().parents[1] / "scripts/hf/publish_compact_epoch.py"
)
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


def fixture():
    binding = {
        "diagnostic_only": False,
        "engine": "native_coatnet_bag12_N3_full_v1",
        "fitting_count": 2247,
        "total_steps": 4496,
        "settings": {"epochs": 16, "effective_batch": 8},
        "initialization_sha256": "91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e",
        "initialization_revision": "2c31d82376cc244c907c064332941e6c65be81f1",
        "supervision": {
            "overlay_sha256": "c8fd2e860d8a3913dbeab34a9d1078316a8f893360124eac088453dc2ed6cf51"
        },
    }
    snapshot = {
        "schema_version": 1,
        "epoch": 1,
        "optimizer_steps": 281,
        "checkpoint_bytes": 300_000_000,
        "binding": binding,
    }
    return snapshot, {"binding": copy.deepcopy(binding), "diagnostic_only": False}


def test_original_complete_epoch():
    snapshot, run = fixture()
    publisher.validate_snapshot(snapshot, run, 1)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda s, r: s.update(optimizer_steps=280),
        lambda s, r: s.update(epoch=2),
        lambda s, r: s.update(checkpoint_bytes=125_000_000),
        lambda s, r: r.update(diagnostic_only=True),
        lambda s, r: s["binding"].update(diagnostic_only=True),
        lambda s, r: s["binding"]["settings"].update(epochs=10),
        lambda s, r: s["binding"]["supervision"].update(overlay_sha256="wrong"),
    ],
)
def test_wrong_epoch_diagnostic_recipe_and_scope_refused(mutate):
    snapshot, run = fixture()
    mutate(snapshot, run)
    with pytest.raises(ValueError):
        publisher.validate_snapshot(snapshot, run, 1)


def test_one_upload_then_verified_readonly_reconcile(tmp_path, monkeypatch):
    from contextlib import nullcontext

    root = tmp_path / "run"
    root.mkdir()
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")
    manifest = {"epoch": 1, "checkpoint_sha256": publisher.sha(checkpoint)}
    monkeypatch.setattr(publisher, "ROOT", root)
    monkeypatch.setattr(
        publisher, "source_files", lambda epoch: (manifest, {"checkpoint.pt": checkpoint})
    )
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "1")
    monkeypatch.delenv("SLURM_JOB_GPUS", raising=False)
    common = types.SimpleNamespace(
        shared_publication_lock=nullcontext,
        check_remote=Mock(side_effect=lambda api, paths, revision=None: paths if revision else []),
        verify_downloads=Mock(),
        atomic_new=lambda path, value: path.open("x").write(json.dumps(value)),
    )
    monkeypatch.setattr(publisher, "core", lambda: common)
    api = types.SimpleNamespace(
        repo_info=lambda *a, **kw: types.SimpleNamespace(private=True, sha="parent"),
        create_commit=Mock(return_value=types.SimpleNamespace(oid="newrevision")),
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        types.SimpleNamespace(HfApi=lambda **kw: api, CommitOperationAdd=lambda **kw: kw),
    )

    def execute(mode):
        monkeypatch.setattr(sys, "stdin", StringIO('{"token":"hf_synthetic"}'))
        return publisher.publish(1, mode)

    first = execute("publish")
    assert first["files_verified_by_download"] == 2
    assert execute("reconcile") == first
    assert api.create_commit.call_count == 1
    assert common.verify_downloads.call_count == 1
    with pytest.raises(ValueError, match="Existing attempt"):
        execute("publish")
    assert api.create_commit.call_count == 1
    # Uncertain returned commit must never trigger another upload.
    (root / "publication/epoch-001/commit.json").unlink()
    with pytest.raises(FileNotFoundError):
        execute("reconcile")
    assert api.create_commit.call_count == 1


def test_dispatch_incomplete_epoch_has_no_intent_or_credential_read(tmp_path, monkeypatch):
    import socket
    import tempfile

    from scripts.hf import dispatch_compact_epoch as dispatch

    socket_dir = tempfile.TemporaryDirectory(prefix="compact-test-", dir="/tmp")
    control = Path(socket_dir.name) / "ssh.sock"
    sock = socket.socket(socket.AF_UNIX)
    sock.bind(str(control))
    script = tmp_path / "scripts/hf/publish_compact_epoch.py"
    script.parent.mkdir(parents=True)
    script.write_text("synthetic publisher")
    monkeypatch.setattr(dispatch, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["dispatch", "1", "--ssh-control", str(control)])

    def missing_snapshot(*a, **kw):
        raise ValueError("epoch sidecar absent")

    monkeypatch.setattr(dispatch, "ssh", missing_snapshot)
    token = Mock(side_effect=AssertionError("must not read credential"))
    monkeypatch.setattr(dispatch, "credential", token)
    try:
        with pytest.raises(ValueError, match="sidecar absent"):
            dispatch.main()
    finally:
        sock.close()
        socket_dir.cleanup()
    token.assert_not_called()
    assert not list(tmp_path.rglob("*intent.json"))
