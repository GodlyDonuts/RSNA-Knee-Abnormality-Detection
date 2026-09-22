from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest


def load(name, filename):
    spec = importlib.util.spec_from_file_location(
        name, Path(__file__).resolve().parents[1] / "scripts/kaggle" / filename
    )
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


p = load("publication", "publication.py")
sys.modules["publication"] = p


def hashed(data):
    return {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


@pytest.fixture
def payload(tmp_path, monkeypatch):
    monkeypatch.setattr(p.shutil, "disk_usage", lambda _: SimpleNamespace(free=p.FLOOR + 10**7))
    data = tmp_path / "dataset"
    data.mkdir()
    members = {"code/rsna/example.py": b"x = 1\n", "encoder/config.json": b"{}\n"}
    for name, blob in members.items():
        (data / name).parent.mkdir(parents=True, exist_ok=True)
        (data / name).write_bytes(blob)
    manifest = {"files": {k: hashed(v) for k, v in members.items()}}
    (data / "bundle-manifest.json").write_text(json.dumps(manifest))
    (data / "dataset-metadata.json").write_text("{}")
    return data, manifest, p.expected_files(data, manifest)


def test_single_zip_preserves_logical_paths(payload, tmp_path):
    data, _, files = payload
    archive = tmp_path / "bundle.zip"
    receipt = p.build_zip(data, files, archive)
    with zipfile.ZipFile(archive) as z:
        assert z.namelist() == sorted(files)
        assert z.read("code/rsna/example.py") == b"x = 1\n"
        assert "code.zip" not in z.namelist()
    assert receipt["members"] == 3
    with pytest.raises(ValueError):
        p.build_zip(data, files, archive)


@pytest.mark.parametrize("kind", ["extra", "tamper", "link", "csv"])
def test_payload_rejects_unbound_or_convertible(payload, kind):
    data, manifest, _ = payload
    if kind == "extra":
        (data / "unknown.json").write_text("{}")
    elif kind == "tamper":
        (data / "encoder/config.json").write_text('{"a":1}')
    elif kind == "link":
        (data / "encoder/config.json").unlink()
        (data / "encoder/config.json").symlink_to("../bundle-manifest.json")
    else:
        (data / "sample.csv").write_bytes(b"a\n")
        manifest["files"]["sample.csv"] = hashed(b"a\n")
    with pytest.raises(ValueError):
        p.expected_files(data, manifest)


def test_floor(payload, tmp_path, monkeypatch):
    data, _, files = payload
    monkeypatch.setattr(p.shutil, "disk_usage", lambda _: SimpleNamespace(free=p.FLOOR))
    with pytest.raises(ValueError):
        p.build_zip(data, files, tmp_path / "bundle.zip")


class Remote:
    username = "godlydonuts"

    def __init__(self, pages=None, files=None):
        self.pages = pages or [{}, {}]
        self.files = files or {}
        self.version = 1
        self.calls = []

    def call(self, service, method, body):
        self.calls.append((method, body))
        if method == "ListDatasets":
            return self.pages[min(body["page"] - 1, len(self.pages) - 1)]
        if method == "GetDatasetStatus":
            return {"status": "READY"}
        if method == "ListDatasetFiles":
            return {
                "datasetFiles": [{"name": n, "totalBytes": size} for n, size in self.files.items()]
            }
        raise AssertionError(method)

    def dataset(self, slug):
        return {"id": 123, "isPrivate": True, "currentVersionNumber": self.version}


def test_absence_complete_empty_protobuf_page():
    r = Remote([{"datasets": [{"ref": "godlydonuts/other"}]}, {}])
    result = p.require_absent(r, "godlydonuts/new-target")
    assert result["pages"] == 2 and result["count"] == 1


@pytest.mark.parametrize(
    "pages",
    [
        [{"datasets": [{"ref": "godlydonuts/new-target"}]}],
        [{"datasets": [{"ref": "other/private"}]}],
        [{"datasets": [{"ref": "godlydonuts/old"}, {"ref": "godlydonuts/old"}]}],
        [{"datasets": "unknown"}],
    ],
)
def test_ambiguous_absence_blocks(pages):
    with pytest.raises(ValueError):
        p.require_absent(Remote(pages), "godlydonuts/new-target")


@pytest.mark.parametrize("prefix", ["", "bundle/"])
def test_wrapper_and_exact_full_bytes(payload, prefix):
    data, _, files = payload
    r = Remote(files={prefix + k: v["bytes"] for k, v in files.items()})
    calls = []

    def stream(remote, slug, name, expected):
        calls.append(name)
        assert hashed((data / name.removeprefix(prefix)).read_bytes()) == expected
        return expected

    result = p.reconcile(r, "godlydonuts/test", files, stream)
    assert result["files"] == files and len(calls) == len(files)


@pytest.mark.parametrize("change", ["extra", "missing", "size", "two_manifests", "mixed"])
def test_membership_changes_block(payload, change):
    _, _, files = payload
    actual = {k: v["bytes"] for k, v in files.items()}
    if change == "extra":
        actual["extra.json"] = 2
    elif change == "missing":
        actual.pop("encoder/config.json")
    elif change == "size":
        actual["encoder/config.json"] += 1
    elif change == "two_manifests":
        actual["elsewhere/bundle-manifest.json"] = 0
    else:
        actual["outer/bundle-manifest.json"] = actual.pop("bundle-manifest.json")
    with pytest.raises(ValueError):
        p.map_members(actual, files)


def test_reconcile_version_changes(payload):
    _, _, files = payload
    r = Remote(files={k: v["bytes"] for k, v in files.items()})

    def stream(remote, slug, name, expected):
        remote.version = 2
        return expected

    with pytest.raises(ValueError):
        p.reconcile(r, "godlydonuts/test", files, stream)


def test_create_intent_and_uncertain_never_retries(payload, tmp_path):
    data, _, files = payload
    upload = tmp_path / "upload"
    upload.mkdir()
    archive = p.build_zip(data, files, upload / "bundle.zip")
    journal = tmp_path / "journal"
    calls = []

    def create(*args):
        assert (journal / "intent.json").exists()
        calls.append(1)
        raise TimeoutError("sensitive detail must not enter journal")

    r = Remote()
    context = {"archive_sha256": archive["sha256"]}
    with pytest.raises(RuntimeError):
        p.create_once(
            r, "test-only-not-a-secret", "godlydonuts/test", files, upload, journal, context, create
        )
    assert json.loads((journal / "uncertain.json").read_text())["error_type"] == "TimeoutError"
    assert "sensitive" not in (journal / "uncertain.json").read_text()
    with pytest.raises(ValueError):
        p.create_once(
            r, "test-only-not-a-secret", "godlydonuts/test", files, upload, journal, context, create
        )
    assert len(calls) == 1


def test_stream_tamper_rejected(monkeypatch):
    class Response:
        status_code = 200

        def iter_content(self, size):
            yield b"bad"

        def close(self):
            pass

    remote = SimpleNamespace(
        session=SimpleNamespace(
            cookies=SimpleNamespace(clear=lambda: None), post=lambda *a, **k: Response()
        )
    )
    with pytest.raises(ValueError):
        p.stream_file(remote, "godlydonuts/test", "file.json", hashed(b"yes"))


def test_stream_rejects_untrusted_redirect():
    class Response:
        status_code = 302
        url = p.API
        headers = {"Location": "https://untrusted.example/private"}

        def close(self):
            pass

    remote = SimpleNamespace(
        session=SimpleNamespace(
            cookies=SimpleNamespace(clear=lambda: None), post=lambda *a, **k: Response()
        )
    )
    with pytest.raises(ValueError):
        p.stream_file(remote, "godlydonuts/test", "file.json", hashed(b"yes"))


def test_exact_projection_rejects_unbound_cached_or_native_code(tmp_path):
    c = load("epoch_coordinator", "coordinator.py")
    project = load("epoch_project_code", "project_code.py")
    projected = tmp_path / "projection"
    path = projected / "scripts/kaggle/workflow.py"
    path.parent.mkdir(parents=True)
    path.write_text("VALUE = 1\n")
    manifest = tmp_path / "projection.json"
    manifest.write_text(
        json.dumps({"files": {"scripts/kaggle/workflow.py": hashed(path.read_bytes())}})
    )
    expected = p.digest(manifest)
    c.load_workflow(projected, manifest, expected)
    foreign = projected / "scripts/kaggle/__pycache__/workflow.cpython-312.pyc"
    foreign.parent.mkdir()
    foreign.write_bytes(b"unbound")
    with pytest.raises(ValueError, match="extra/missing files"):
        c.load_workflow(projected, manifest, expected)
    foreign.unlink()
    foreign = path.with_name("unbound.so")
    foreign.write_bytes(b"unbound native code")
    with pytest.raises(ValueError, match="extra/missing files"):
        c.load_workflow(projected, manifest, expected)
    assert project.selected("rsna/competitive/infer.py")
    assert not project.selected("initiative/source/train.csv")


def test_zip_change_rejected_before_any_remote_call(payload, tmp_path):
    data, _, files = payload
    upload = tmp_path / "upload"
    upload.mkdir()
    archive = p.build_zip(data, files, upload / "bundle.zip")
    with zipfile.ZipFile(upload / "bundle.zip", "a") as handle:
        handle.writestr("extra.py", "unbound")
    r = Remote()
    with pytest.raises(ValueError):
        p.create_once(
            r,
            "test-only",
            "godlydonuts/test",
            files,
            upload,
            tmp_path / "journal",
            {"archive_sha256": archive["sha256"]},
            create=lambda *args: None,
        )
    assert not r.calls
