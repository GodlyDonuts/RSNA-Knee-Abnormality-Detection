import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "completion", Path(__file__).with_name("complete_commit.py")
)
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_metadata_only_exact_blob_and_small_files(tmp_path):
    paths = {
        n: tmp_path / n
        for n in (
            "model.pt",
            "provenance.json",
            "own-source.json",
            "source-permission.txt",
            "manifest.json",
        )
    }
    for n, p in paths.items():
        p.write_bytes(n.encode())
    intent = {
        "prefix": M.PREFIX,
        "parent": M.PARENT,
        "files": {n: {"bytes": p.stat().st_size, "sha256": M.sha(p)} for n, p in paths.items()},
    }
    intent["files"]["model.pt"]["sha256"] = M.MODEL
    records = [json.loads(x) for x in M.payload(intent, paths).splitlines()]
    blobs = [r for r in records if r["key"] == "lfsFile"]
    assert len(blobs) == 1 and blobs[0]["value"]["oid"] == M.MODEL
    assert "content" not in blobs[0]["value"]
    assert len([r for r in records if r["key"] == "file"]) == 4
    assert records[0]["value"]["parentCommit"] == M.PARENT
    paths["provenance.json"].write_bytes(b"changed")
    with pytest.raises(AssertionError):
        M.payload(intent, paths)


def test_single_metadata_action_no_upload_or_retry():
    import ast

    tree = ast.parse(Path(M.__file__).read_text())
    names = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert names.count("requests.post") == 1
    assert not any("upload" in n or "create_commit" in n for n in names)
    source = Path(M.__file__).read_text()
    assert "shared_publication_lock" in source and "info.sha == PARENT" in source
    assert "if response.status_code != 200:" in source
