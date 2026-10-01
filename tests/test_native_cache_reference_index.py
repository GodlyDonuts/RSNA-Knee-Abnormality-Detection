"""Header indexing never replaces full selected-payload verification."""

import copy
import json

import pytest
import test_native_cache_worker as worker_tests

from scripts.newton import native_cache_reference_index as index


@pytest.fixture
def reference(tmp_path):
    inputs = worker_tests.inputs.__wrapped__(tmp_path)
    old = worker_tests.worker.ReferenceArchive(inputs[3], inputs[0]["series"])
    record = index.build_index(inputs[3], old.records, index.PLAN_SHA)
    path = tmp_path / "reference-index.json"
    path.write_text(json.dumps(record))
    return inputs, old, path, index.digest(path)


def test_indexed_reference_returns_exact_scanned_source_members(reference):
    inputs, old, path, sha = reference
    new = worker_tests.worker.ReferenceArchive(inputs[3], inputs[0]["series"], path, sha)
    assert new.offsets == old.offsets
    for name in old.records:
        assert new.read(name) == old.read(name)


def test_indexed_real_dicom_rebuild_is_byte_identical(reference, tmp_path):
    inputs, _, path, sha = reference
    normal = tmp_path / "normal"
    indexed = tmp_path / "indexed"
    worker_tests.build(inputs, normal)
    reader = worker_tests.Source(inputs[2])
    try:
        worker_tests.worker.build_shard(
            inputs[0],
            inputs[1],
            reader,
            worker_tests.worker.ReferenceArchive(inputs[3], inputs[0]["series"], path, sha),
            indexed,
        )
    finally:
        reader.downloads.shutdown(wait=True)
    for side in (256, 384):
        a = normal / f"cache-side-{side}.tar"
        b = indexed / f"cache-side-{side}.tar"
        assert a.read_bytes() == b.read_bytes()


@pytest.mark.parametrize("change", ["offset", "size", "hash", "missing", "scope", "plan"])
def test_bad_index_cannot_redirect_accepted_member(reference, change):
    inputs, old, path, _ = reference
    record = json.loads(path.read_text())
    name = next(iter(record["members"]))
    if change == "offset":
        record["members"][name]["offset"] = inputs[3].stat().st_size
    elif change == "size":
        record["members"][name]["bytes"] += 1
    elif change == "hash":
        record["members"][name]["sha256"] = "0" * 64
    elif change == "missing":
        del record["members"][name]
    elif change == "scope":
        record["selected_member_bytes_verified"] = True
    else:
        record["plan_manifest_sha256"] = "f" * 64
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        index.load_offsets(inputs[3], path, index.digest(path), old.records)


def test_changed_archive_or_index_is_rejected(reference):
    inputs, old, path, sha = reference
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="pinned_reference_index_bytes_changed"):
        index.load_offsets(inputs[3], path, sha, old.records)
    record = json.loads(path.read_text())
    with inputs[3].open("ab") as f:
        f.write(b"modified")
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="reference_index_archive_or_plan_changed"):
        index.load_offsets(inputs[3], path, index.digest(path), old.records)


def test_valid_index_still_rejects_corrupted_selected_payload(reference):
    inputs, old, path, sha = reference
    new = worker_tests.worker.ReferenceArchive(inputs[3], inputs[0]["series"], path, sha)
    name = next(iter(old.offsets))
    offset, _ = old.offsets[name]
    with inputs[3].open("r+b") as f:
        f.seek(offset)
        byte = f.read(1)
        f.seek(offset)
        f.write(bytes([byte[0] ^ 1]))
    with pytest.raises(ValueError, match="source_or_reference_bytes_changed"):
        new.read(name)


def test_header_index_requires_every_expected_member(reference):
    inputs, old, _, _ = reference
    missing = copy.deepcopy(old.records)
    missing["cache/train_series/absent/absent.npz"] = next(iter(missing.values()))
    with pytest.raises(ValueError, match="missing_reference_member_or_changed_archive"):
        index.build_index(inputs[3], missing, index.PLAN_SHA)
