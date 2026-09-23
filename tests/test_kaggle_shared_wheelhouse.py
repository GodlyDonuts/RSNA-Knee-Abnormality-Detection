import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import pytest


def load(name, file):
    spec = importlib.util.spec_from_file_location(
        name,
        (Path(__file__).resolve().parents[1] / "scripts/kaggle" / file)
        if file != "test_legacy_compatibility.py"
        else Path(__file__).with_name("test_kaggle_submission.py"),
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


w = load("shared_workflow", "workflow.py")
n = load("shared_notebook", "notebook_entry.py")


def proof(**changes):
    return dict(
        schema_version=1,
        dataset_slug="owner/wheels",
        dataset_id=42,
        dataset_version=1,
        archive_file="wheelhouse-cp312-cu130.bin",
        archive_bytes=100,
        archive_sha256="a" * 64,
        manifest_sha256="b" * 64,
        pyproject_sha256="c" * 64,
        gpu_probe_slug="owner/probe",
        gpu_probe_version=1,
        gpu_probe_source_sha256=hashlib.sha256(b"probe").hexdigest(),
        gpu_receipt_sha256="d" * 64,
        **changes,
    )


class Remote:
    username = "owner"
    version = 1
    private = True
    source = "probe"

    def __init__(self):
        self.receipt = {
            "status": "passed",
            "wheelhouse_archive_sha256": "a" * 64,
            "wheelhouse_manifest_sha256": "b" * 64,
            "cuda_tensor_backward": True,
            "random_tiny_dinov3_cuda_forward": True,
        }

    def dataset(self, slug):
        return {"isPrivate": self.private, "currentVersionNumber": self.version, "id": 42}

    def kernel(self, method, slug, version):
        if method == "GetKernelSessionStatus":
            return {"status": "COMPLETE"}
        return {
            "metadata": {
                "isPrivate": True,
                "enableInternet": False,
                "machineShape": "NvidiaTeslaT4",
                "datasetDataSources": ["owner/wheels"],
            },
            "blob": {"source": self.source},
        }

    def output_json(self, *args):
        return self.receipt


def test_shared_private_gpu_proof_validated():
    remote = Remote()
    record = proof()
    record["gpu_receipt_sha256"] = w.object_hash(remote.receipt)
    assert w.verify_shared_wheelhouse(remote, record)["id"] == 42


@pytest.mark.parametrize(
    "field,value", [("version", 2), ("private", False), ("source", "tampered")]
)
def test_shared_remote_drift_rejected(field, value):
    remote = Remote()
    record = proof()
    record["gpu_receipt_sha256"] = w.object_hash(remote.receipt)
    setattr(remote, field, value)
    with pytest.raises(ValueError):
        w.verify_shared_wheelhouse(remote, record)


def test_gpu_result_archive_hash_must_match():
    remote = Remote()
    record = proof()
    remote.receipt["wheelhouse_archive_sha256"] = "e" * 64
    record["gpu_receipt_sha256"] = w.object_hash(remote.receipt)
    with pytest.raises(ValueError):
        w.verify_shared_wheelhouse(remote, record)


def fixture_archive(tmp_path, extra=None):
    inputs = tmp_path / "input"
    inputs.mkdir()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    content = {"fixture.whl": b"synthetic wheel", "requirements.lock": b"fixture==1"}
    manifest = {
        "files": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
            for name, data in content.items()
        }
    }
    serialized = json.dumps(manifest).encode()
    (bundle / "wheelhouse.json").write_bytes(serialized)
    archive = inputs / "wheelhouse-cp312-cu130.bin"
    with tarfile.open(archive, "w") as t:
        for name, data in {**content, "wheelhouse.json": serialized}.items():
            item = tarfile.TarInfo("wheelhouse/" + name)
            item.size = len(data)
            t.addfile(item, io.BytesIO(data))
        if extra == "file":
            item = tarfile.TarInfo("wheelhouse/extra")
            item.size = 1
            t.addfile(item, io.BytesIO(b"x"))
        if extra == "link":
            item = tarfile.TarInfo("wheelhouse/link")
            item.type = tarfile.SYMTYPE
            item.linkname = "/etc/passwd"
            t.addfile(item)
    shared = proof()
    shared.update(
        archive_sha256=w.digest(archive),
        archive_bytes=archive.stat().st_size,
        manifest_sha256=hashlib.sha256(serialized).hexdigest(),
    )
    return bundle, inputs, {"shared_wheelhouse": shared}


def test_loader_checks_full_shared_archive_and_payload(tmp_path):
    bundle, inputs, manifest = fixture_archive(tmp_path)
    path, value = n.resolve_wheels(bundle, manifest, inputs, tmp_path / "extracted")
    assert (path / "fixture.whl").read_bytes() == b"synthetic wheel"
    assert value == json.loads((bundle / "wheelhouse.json").read_text())


@pytest.mark.parametrize("extra", ["file", "link"])
def test_loader_rejects_extra_files_and_links_even_with_bound_archive(tmp_path, extra):
    bundle, inputs, manifest = fixture_archive(tmp_path, extra)
    with pytest.raises(ValueError):
        n.resolve_wheels(bundle, manifest, inputs, tmp_path / "extracted")


def test_loader_rejects_changed_archive(tmp_path):
    bundle, inputs, manifest = fixture_archive(tmp_path)
    with (inputs / "wheelhouse-cp312-cu130.bin").open("ab") as handle:
        handle.write(b"x")
    with pytest.raises(ValueError):
        n.resolve_wheels(bundle, manifest, inputs, tmp_path / "extracted")


def test_shared_bundle_has_two_sources_and_omits_wheel_copies(tmp_path, monkeypatch):
    legacy = load("legacy_bundle_fixture", "test_legacy_compatibility.py")
    original = legacy.workflow.prepare_bundle

    def prepare(*args, **kwargs):
        wheelhouse, repo = args[2], args[9]
        record = proof()
        record.update(
            manifest_sha256=w.digest(wheelhouse / "wheelhouse.json"),
            pyproject_sha256=w.digest(repo / "pyproject.toml"),
        )
        path = tmp_path / "shared-proof.json"
        path.write_text(json.dumps(record))
        return original(*args, **kwargs, shared_wheelhouse=path)

    monkeypatch.setattr(legacy.workflow, "prepare_bundle", prepare)
    legacy.test_complete_local_bundle_binding_and_no_extra_upload_files(tmp_path, monkeypatch)
    prepared = tmp_path / "prepared"
    assert not (prepared / "dataset/wheels").exists()
    assert (prepared / "dataset/wheelhouse.json").exists()
    assert (prepared / "dataset/code/chunked_infer.py").read_bytes() == (
        Path(__file__).resolve().parents[1] / "scripts/kaggle/chunked_infer.py"
    ).read_bytes()
    assert "code/chunked_infer.py" in (prepared / "kernel/run.py").read_text()
    operation = json.loads((prepared / "operation.json").read_text())
    metadata = json.loads((prepared / "kernel/kernel-metadata.json").read_text())
    assert metadata["dataset_sources"] == [operation["dataset_slug"], "owner/wheels"]
    request = w.kernel_request(prepared / "operation.json", operation)
    assert request["datasetDataSources"] == metadata["dataset_sources"]
