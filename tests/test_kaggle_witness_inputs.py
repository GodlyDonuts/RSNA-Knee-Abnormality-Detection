"""Bound complete private source extraction before any model training admission."""

import hashlib
import io
import json
import tarfile

import pytest

from scripts.kaggle.witness_inputs import extract_inputs


def payload(tmp_path, name="artifacts/fixture.txt", *, symlink=False, duplicate=False):
    data = tmp_path / "payload"
    data.mkdir()
    raw = b"Original immutable metadata fixture"
    archive = data / "training-inputs.bin"
    with tarfile.open(archive, "w") as t:
        item = tarfile.TarInfo(name)
        item.size = len(raw)
        if symlink:
            item.type = tarfile.SYMTYPE
            item.linkname = "../outside"
            item.size = 0
        t.addfile(item, None if symlink else io.BytesIO(raw))
        if duplicate:
            t.addfile(item, io.BytesIO(raw))
    h = hashlib.sha256(archive.read_bytes()).hexdigest()
    record = {
        "status": "actual_complete_original_release_and_scientific_source_portable_CPU_replay_accepted",
        "archive_sha256": h,
        "archive_bytes": archive.stat().st_size,
        "no_pixels_or_model_weights_in_archive": True,
        "files": {
            name: {"bytes": 0 if symlink else len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        },
    }
    proof = data / "CPU-admission.json"
    proof.write_text(json.dumps(record))
    return data, h, hashlib.sha256(proof.read_bytes()).hexdigest(), raw


def test_complete_bound_archive_extracts_exact_bytes(tmp_path):
    directory, arc, adm, raw = payload(tmp_path)
    output = tmp_path / "exact"
    result = extract_inputs(directory, output, arc, adm)
    assert (output / "artifacts/fixture.txt").read_bytes() == raw
    assert result["files"]["artifacts/fixture.txt"]["bytes"] == len(raw)


@pytest.mark.parametrize("mode", ["symlink", "traversal", "duplicate", "changed_bytes"])
def test_source_archive_rejects_unbound_or_escaping_members(tmp_path, mode):
    directory, arc, adm, _ = payload(
        tmp_path,
        name="../outside" if mode == "traversal" else "artifacts/fixture.txt",
        symlink=mode == "symlink",
        duplicate=mode == "duplicate",
    )
    if mode == "changed_bytes":
        with (directory / "training-inputs.bin").open("ab") as f:
            f.write(b"changed")
    with pytest.raises(ValueError):
        extract_inputs(directory, tmp_path / "exact", arc, adm)
    assert not (tmp_path / "outside").exists()


def test_Kaggle_scratch_budget_does_not_require_cluster_reserve(tmp_path, monkeypatch):
    from collections import namedtuple

    from scripts.kaggle import witness_inputs

    Usage = namedtuple("Usage", "total used free")
    monkeypatch.setattr(
        witness_inputs.shutil, "disk_usage", lambda _: Usage(20 * 2**30, 13 * 2**30, 7 * 2**30)
    )
    record = witness_inputs.kaggle_storage_budget(tmp_path)
    assert record["scratch_reserve_bytes"] == 3 * 2**30
    assert record["cluster_shared_and_node_reserve_changed"] is False
    monkeypatch.setattr(
        witness_inputs.shutil, "disk_usage", lambda _: Usage(20 * 2**30, 16 * 2**30, 4 * 2**30)
    )
    with pytest.raises(OSError, match="cannot fund"):
        witness_inputs.kaggle_storage_budget(tmp_path)
