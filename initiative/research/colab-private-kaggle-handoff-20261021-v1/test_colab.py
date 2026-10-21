import ast
import hashlib
import importlib.util
import io
import tarfile
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


D = load("Colab_training_source_fixture", HERE / "derive.py")
T = load("Colab_training_storage_adapter", HERE / "training.py")
S = load("Colab_hybrid_download_fixture", HERE / "colab_storage.py")


def test_learning_model_data_order_and_fit_call_are_unchanged():
    original = ast.parse(D.original_files()["colab_train.py"])
    adapted = ast.parse(T.build())

    def named(tree, name):
        return next(
            n
            for n in ast.walk(tree)
            if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name == name
        )

    for name in ("Encoder", "Cases", "update", "seal"):
        assert ast.dump(named(original, name), include_attributes=False) == ast.dump(
            named(adapted, name), include_attributes=False
        )

    def call(tree, name):
        return [
            ast.dump(n, include_attributes=False)
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name
        ]

    for name in ("Predictive", "Recipe", "fit"):
        assert call(original, name) == call(adapted, name)
    assert "api.create_commit" not in T.build()
    assert "store.publish(f'epoch-{epoch:03d}'" in T.build()
    assert "store.publish('terminal'" in T.build()


@pytest.mark.parametrize("fault", [None, "traversal", "duplicate", "wrong_hash", "symlink"])
def test_exact_archive_member_verification(tmp_path, monkeypatch, fault):
    root = tmp_path / "out"
    root.mkdir()
    monkeypatch.setattr(S, "ROOT", root)
    data = b"original input fixture"
    name = "contexts/000001.npz"
    if fault == "traversal":
        name = "../escaped.npz"
    archive = tmp_path / "fixture.tar"
    with tarfile.open(archive, "w") as tar:
        member = tarfile.TarInfo(name)
        member.size = len(data)
        if fault == "symlink":
            member.type = tarfile.SYMTYPE
            member.linkname = "/tmp/target"
        tar.addfile(member, io.BytesIO(data))
    proof = {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
    if fault == "wrong_hash":
        proof["sha256"] = "0" * 64
    manifest = {"members": {name: proof}}
    seen = {name} if fault == "duplicate" else set()
    if fault is not None:
        with pytest.raises(AssertionError):
            S.extract(archive, manifest, seen)
    else:
        S.extract(archive, manifest, seen)
        assert seen == {name} and (root / name).read_bytes() == data


@pytest.mark.parametrize("fault", [None, "unsafe_redirect", "oversize", "wrong_hash"])
def test_download_checks_complete_bytes_and_never_forwards_auth(tmp_path, monkeypatch, fault):
    import requests

    body = b"original archive bytes"
    proof = {"bytes": len(body), "sha256": hashlib.sha256(body).hexdigest()}
    if fault == "wrong_hash":
        proof["sha256"] = "0" * 64
    if fault == "oversize":
        proof["bytes"] -= 1
    calls = []

    class Response:
        def __init__(self, status, location=None):
            self.status_code = status
            self.url = "https://api.kaggle.com/test"
            self.headers = {"Location": location}

        def close(self):
            pass

        def iter_content(self, _size):
            yield body

    def post(url, **kwargs):
        assert url.startswith("https://api.kaggle.com/")
        return Response(
            302,
            "https://untrusted.invalid/file"
            if fault == "unsafe_redirect"
            else "https://storage.googleapis.com/private-object",
        )

    def get(url, **kwargs):
        calls.append((url, kwargs))
        assert "headers" not in kwargs and "cookies" not in kwargs
        return Response(200)

    monkeypatch.setattr(requests, "get", get)
    remote = types.SimpleNamespace(
        session=types.SimpleNamespace(cookies=types.SimpleNamespace(clear=lambda: None), post=post)
    )
    target = tmp_path / "archive.bin"
    if fault:
        with pytest.raises(AssertionError):
            S.download_file(remote, "godlydonuts/test", "archive.bin", target, proof)
    else:
        S.download_file(remote, "godlydonuts/test", "archive.bin", target, proof)
        assert target.read_bytes() == body
    if fault == "unsafe_redirect":
        assert not calls


@pytest.mark.parametrize("failure", [None, "create_uncertain", "download_hash"])
def test_epoch_storage_is_private_verified_and_never_recreates(tmp_path, failure):
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"immutable checkpoint fixture")
    destination = tmp_path / "publication"
    destination.mkdir()
    calls = []
    slug = "godlydonuts/rsna-colab-c100-epoch-001-fixture"
    metadata = {"id": 123, "private": True, "version": 1, "slug": slug}

    def create(token, target, title, upload):
        assert token == "test-only-token" and target == slug
        assert {p.name for p in upload.iterdir()} == {"model.pt.bin"}
        calls.append("create")
        if failure == "create_uncertain":
            raise RuntimeError("Uncertain original creation")
        return {"returned": True, "private_requested": True, "final_create_rpc_calls": 1}

    def download(remote, target, name, proof, **kwargs):
        calls.append("independent_download")
        assert proof == {"bytes": checkpoint.stat().st_size, "sha256": S.sha(checkpoint)}
        if failure == "download_hash":
            raise ValueError("Remote bytes differ")

    store = S.Store.__new__(S.Store)
    store.token = "test-only-token"
    store.lease = {"slugs": {"epoch-001": slug}}
    store.create = create
    store.remote = types.SimpleNamespace(call=lambda *args: {"status": "READY"})
    store.pub = types.SimpleNamespace(
        require_absent=lambda *args: {"absent": True},
        metadata=lambda *args: metadata,
        list_files=lambda *args: {"model.pt.bin": checkpoint.stat().st_size},
        stream_file=download,
    )
    if failure:
        with pytest.raises((RuntimeError, ValueError)):
            store.publish("epoch-001", {"model.pt": checkpoint}, destination)
        assert not (destination / "verified.json").exists()
    else:
        result = store.publish("epoch-001", {"model.pt": checkpoint}, destination)
        assert result["private"] and result["revision"] == "Kaggle:123:v1"
        assert calls == ["create", "independent_download"]
    with pytest.raises(AssertionError):
        store.publish("epoch-001", {"model.pt": checkpoint}, destination)
    assert calls.count("create") == 1
