import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


D = load("private_Colab_derivation_test", HERE / "derive.py")
IO = load("private_Colab_storage_test", HERE / "kaggle_io.py")


def nested_upload(tmp_path):
    tree = ast.parse(D.bridge())
    main = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "main")
    upload = next(n for n in main.body if isinstance(n, ast.FunctionDef) and n.name == "upload")
    directory = tmp_path / "upload"
    directory.mkdir()
    namespace = dict(
        ROOT=tmp_path,
        upload_dir=directory,
        manifest={"archives": []},
        digest=lambda p: hashlib.sha256(p.read_bytes()).hexdigest(),
        GIB=2**30,
        json=json,
        os=os,
        KAGGLE_SLUG=D.SLUG,
    )
    exec(
        compile(ast.Module(body=[upload], type_ignores=[]), "actual_derived_upload", "exec"),
        namespace,
    )
    return namespace


def test_completed_HF_archive_is_reused_without_any_Kaggle_file(tmp_path):
    n = nested_upload(tmp_path)
    p = tmp_path / "contexts-000.tar"
    p.write_bytes(b"original archive")
    proof = dict(
        path="inputs/" + p.name, bytes=p.stat().st_size, sha256=n["digest"](p), revision="a" * 40
    )
    (tmp_path / (p.name + ".uploaded.json")).write_text(json.dumps(proof))
    n["upload"](p)
    assert n["manifest"]["archives"] == [{**proof, "provider": "HF"}]
    assert not list((tmp_path / "upload").iterdir())
    p.write_bytes(b"changed archive")
    with pytest.raises(AssertionError):
        n["upload"](p)


def test_remaining_archive_uses_same_inode_and_distinct_opaque_remote_name(tmp_path):
    n = nested_upload(tmp_path)
    p = tmp_path / "contexts-006.tar"
    p.write_bytes(b"original unuploaded archive")
    n["upload"](p)
    target = tmp_path / "upload/contexts-006.tar.bin"
    assert target.stat().st_ino == p.stat().st_ino
    assert target.read_bytes() == p.read_bytes()
    assert n["manifest"]["archives"][0]["provider"] == "Kaggle"
    assert n["manifest"]["archives"][0]["private"] is True
    assert not (tmp_path / "contexts-006.tar.uploaded.json").exists()
    with pytest.raises(FileExistsError):
        n["upload"](p)


def test_derived_bridge_has_no_HF_mutation_and_retains_original_inputs():
    source = D.bridge()
    tree = ast.parse(source)
    calls = {
        n.func.attr
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
    }
    assert not calls & {"create_commit", "create_repo", "delete_repo", "unlink"}
    assert "create_private_inputs" in source
    for name, body in D.original_files().items():
        assert body == D.frozen(D.ORIGINAL + name, D.SCIENCE)
    original = ast.parse(D.frozen(D.ORIGINAL + "resume_bridge.py"))
    for name in ("Archive", "shards", "signature", "original"):

        def get(tree):
            return next(
                n
                for n in tree.body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name == name
            )

        assert ast.dump(get(tree), include_attributes=False) == ast.dump(
            get(original), include_attributes=False
        )


def test_sdk_keeps_one_private_create_with_no_automatic_retry(tmp_path):
    source = D.sources()["publication.py"]
    p = tmp_path / "publication.py"
    p.write_text(source)
    module = load("private_Kaggle_publication_fixture", p)
    call = IO.private_multifile_publisher(module)
    assert call.__name__ == "sdk_create_once"
    upload = tmp_path / "upload"
    upload.mkdir()
    (upload / "not-a-bin.txt").write_text("fixture")
    with pytest.raises(ValueError, match="opaque"):
        call("not-a-token", D.SLUG, "test", upload)
    function = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.FunctionDef) and n.name == "sdk_create_once"
    )
    calls = [
        n
        for n in ast.walk(function)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute)
        and n.func.attr == "create_dataset"
    ]
    assert len(calls) == 1
    assert 'req.title, req.license_name, req.is_private = title, "other", True' in source
    assert "api.with_retry = lambda f: f" in source
