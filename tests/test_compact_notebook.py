"""Scheduled notebook identities and frozen training projection, without cloud calls."""

import ast
import base64
import gzip
import hashlib
import json
import subprocess

import pytest

from scripts.kaggle.compact_notebook import build


@pytest.mark.parametrize("epoch", [4, 8, 12, 16])
def test_scheduled_binding_and_frozen_science(tmp_path, epoch):
    prepared = {
        "epoch": epoch,
        "checkpoint_sha256": "a" * 64,
        "component_manifest_sha256": "b" * 64,
    }
    path = tmp_path / "entry.py"
    result = build(prepared, path)
    text = path.read_text()
    tree = ast.parse(text)
    packed = next(
        n.value
        for n in ast.walk(tree)
        if isinstance(n, ast.Constant) and isinstance(n.value, str) and len(n.value) > 100000
    )
    raw = gzip.decompress(base64.b64decode(packed))
    files = json.loads(raw)
    original = subprocess.check_output(
        [
            "git",
            "show",
            "dcfff319cb7bd7bbae679a20da2b1b9268e8d680:rsna/competitive/compact_train.py",
        ]
    )
    assert files["rsna/competitive/compact_train.py"].encode() == original
    assert hashlib.sha256(raw).hexdigest() == result["projection_sha256"]
    assert result["epoch"] == epoch
    assert result["kernel_slug"] == f"godlydonuts/rsna-compact-offline-e{epoch:03d}-aaaaaaaaaaaa"
    assert result["kernel_slug"] != f"godlydonuts/rsna-compact-n3-e{epoch:03d}-aaaaaaaaaaaa"
    assert f"accept['epoch']=={epoch}" in text
    assert f"epoch={epoch},component_manifest_sha256" in text
    assert "'--aggregation','full'" in text
    assert "'--test-studies',str(competition/'test.csv')" in text
    assert result["source_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("epoch", [True, 5, 0, 17, "8"])
def test_refuse_unscheduled_checkpoint(tmp_path, epoch):
    with pytest.raises(AssertionError):
        build({"epoch": epoch}, tmp_path / "entry.py")
