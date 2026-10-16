"""Original-DICOM reconstruction, reference acceptance and safe shard restoration."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from test_pipeline import dicom_bytes

from rsna.preprocess import Source, cache_series

PATH = Path(__file__).resolve().parents[1] / "scripts/newton/build_native_cache_shard.py"
SPEC = importlib.util.spec_from_file_location("native_cache_worker", PATH)
assert SPEC and SPEC.loader
worker = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(worker)


@pytest.fixture
def inputs(tmp_path):
    torch.set_num_threads(1)
    original = tmp_path / "original"
    folder = original / "train_series/1.2/1.2.3"
    folder.mkdir(parents=True)
    objects = []
    for i, z in [(1, 3), (2, 1), (3, 2)]:
        path = folder / f"1.2.3.{i}.dcm"
        data = dicom_bytes(i, z)
        path.write_bytes(data)
        objects.append(
            {
                "split": "train_series",
                "study_uid": "1.2",
                "series_uid": "1.2.3",
                "sop_uid": f"1.2.3.{i}",
                "path": path.relative_to(original).as_posix(),
                "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest(),
                "storage": "lfs",
            }
        )
    accepted = tmp_path / "accepted"
    reader = Source(original)
    try:
        record = cache_series(
            reader, [r["path"] for r in objects], accepted, 256, worker.SOURCE_REVISION
        )
    finally:
        reader.downloads.shutdown(wait=True)
    cache = accepted / record["cache_path"]
    row = {
        "study_uid": "1.2",
        "series_uid": "1.2.3",
        "n_slices": 3,
        "source_bytes": sum(r["bytes"] for r in objects),
        "cache_path": record["cache_path"],
        "cache_bytes": cache.stat().st_size,
        "cache_sha256": worker.digest(cache),
        "cache_metadata_bytes": cache.with_suffix(".json").stat().st_size,
        "cache_metadata_sha256": worker.digest(cache.with_suffix(".json")),
        "plane": "Axial",
        "fluid_sensitive": 1,
        "fat_suppression": 1,
        "selection_tier": 1,
    }
    shard = {
        "shard": 0,
        "series": [row],
        "study_uids": ["1.2"],
        "series_count": 1,
        "slices": 3,
        "source_bytes": row["source_bytes"],
        "selection_sha256": "fixture",
        "regular_members_per_side": 2,
        "archive_upper_bytes_by_side": {"256": 5 * 1024**2, "384": 5 * 1024**2},
        "owned_cache_archive_scratch_bound_bytes": 30 * 1024**2,
    }
    archive = tmp_path / "accepted.tar"
    worker.archive_tree(accepted, archive)
    return shard, pd.DataFrame(objects), original, archive, accepted


def build(inputs, output):
    shard, inventory, original, archive, _ = inputs
    source = Source(original)
    try:
        return worker.build_shard(
            shard, inventory, source, worker.ReferenceArchive(archive, shard["series"]), output
        )
    finally:
        source.downloads.shutdown(wait=True)


def test_real_dicom_paired_reconstruction_and_runtime_restore(inputs, tmp_path):
    before = worker.digest(inputs[3])
    output = tmp_path / "built"
    report = build(inputs, output)
    assert report["status"] == "native_paired_shard_cache_and_loader_verified"
    assert report["max_reference_uint16_difference"] == 0
    assert report["source_bytes_verified"] == inputs[0]["source_bytes"]
    assert not report["full_cohort_built"] and not report["model_training_or_accuracy_claim"]
    assert worker.digest(inputs[3]) == before
    for side, input_side in ((256, 224), (384, 384)):
        archive = output / f"cache-side-{side}.tar"
        record = report["archives"][archive.name]
        destination = tmp_path / f"restored-{side}"
        worker.verify_archive(archive, record, destination)
        files = sorted(p for p in destination.rglob("*") if p.is_file())
        assert len(files) == 2
        with np.load(next(p for p in files if p.suffix == ".npz")) as cache:
            assert cache["volume"].shape == (3, side, side)
        metadata = json.loads(next(p for p in files if p.suffix == ".json").read_text())
        assert metadata["sop_uids"] == ["1.2.3.2", "1.2.3.3", "1.2.3.1"]
        assert metadata["original_shape"] == [3, 16, 32]
        assert report["loaders"][str(side)]["input_side"] == input_side
        assert report["loaders"][str(side)]["valid_sampled_slices"] == 3
        assert not report["loaders"][str(side)]["labels_read"]
    assert len(list(output.glob("series-proofs*.jsonl"))) == 1
    assert not list(output.glob("series-*.json"))
    with pytest.raises(FileExistsError):
        build(inputs, output)


def test_source_corruption_stops_before_verified_shard(inputs, tmp_path):
    path = inputs[2] / inputs[1].iloc[0].path
    path.write_bytes(path.read_bytes() + b"corrupt")
    output = tmp_path / "failed"
    with pytest.raises(ValueError, match="bytes_changed"):
        build(inputs, output)
    assert (output / "build-intent.json").is_file()
    assert not (output / "cache-verified.json").exists()


@pytest.mark.parametrize("change", ["missing", "duplicate", "wrong_study", "wrong_source_count"])
def test_no_source_or_study_can_be_silently_dropped(inputs, change):
    shard, inventory, *_ = inputs
    if change == "missing":
        inventory = inventory.iloc[:-1]
    elif change == "duplicate":
        inventory = pd.concat([inventory, inventory.iloc[:1]], ignore_index=True)
    elif change == "wrong_study":
        inventory.loc[0, "study_uid"] = "other"
    else:
        shard["source_bytes"] += 1
    with pytest.raises(ValueError, match="membership_changed|byte_count_changed"):
        worker.source_members(shard, inventory)


@pytest.mark.parametrize(
    "key",
    [
        "sop_uids",
        "pixel_sha256",
        "positions",
        "pixel_spacing",
        "warnings",
        "intensity_max",
        "resized_shape",
    ],
)
def test_reference_geometry_pixels_and_intensity_must_match(key):
    volume = np.array([[[100, 200]]], dtype=np.uint16)
    with pytest.raises(ValueError, match="binding_changed"):
        worker.compare_reference(volume, {key: ["new"]}, volume, {key: ["old"]})


def test_one_uint16_rounding_level_is_preserved_but_larger_error_is_rejected():
    volume = np.array([[[100, 200]]], dtype=np.uint16)
    record = worker.compare_reference(volume, {}, volume + 1, {})
    assert record["max_absolute_uint16_difference"] == 1 and not record["cause_proven"]
    with pytest.raises(ValueError, match="rounding_band"):
        worker.compare_reference(volume, {}, volume + 2, {})


@pytest.mark.parametrize(
    "path",
    ["../outside.npz", "/absolute.npz", "cache//x.npz", "cache/./x.npz", "cache\\x.npz", ".", ""],
)
def test_unsafe_member_names_are_rejected(path):
    with pytest.raises(ValueError, match="path"):
        worker.safe_path(path)


def fake_archive(tmp_path, names, kinds=None):
    path = tmp_path / "bad.tar"
    members = {}
    with tarfile.open(path, "x") as tar:
        for i, name in enumerate(names):
            item = tarfile.TarInfo(name)
            item.size = 3
            if kinds and kinds[i] == "symlink":
                item.type, item.linkname, item.size = tarfile.SYMTYPE, "outside", 0
            tar.addfile(item, io.BytesIO(b"abc") if item.isfile() else None)
            members[name] = {"bytes": item.size, "sha256": hashlib.sha256(b"abc").hexdigest()}
    return path, {
        "bytes": path.stat().st_size,
        "sha256": worker.digest(path),
        "members": members,
        "regular_members": len(members),
        "unpacked_file_bytes": sum(r["bytes"] for r in members.values()),
    }


@pytest.mark.parametrize(
    "names,kinds",
    [
        (["../outside.npz"], None),
        (["cache/x.npz", "cache/x.npz"], None),
        (["cache/x.npz"], ["symlink"]),
    ],
)
def test_archive_cannot_escape_repeat_or_link(tmp_path, names, kinds):
    archive, record = fake_archive(tmp_path, names, kinds)
    destination = tmp_path / "restored"
    with pytest.raises(ValueError):
        worker.verify_archive(archive, record, destination)
    assert not destination.exists()


def test_restoration_checks_member_bytes_independently_of_whole_tar(tmp_path):
    archive, record = fake_archive(tmp_path, ["cache/x.npz"])
    record["members"]["cache/x.npz"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="bytes_changed"):
        worker.verify_archive(archive, record)


def test_no_verified_receipt_is_written_when_archive_budget_is_exceeded(inputs, tmp_path):
    inputs[0]["archive_upper_bytes_by_side"]["256"] = 1
    output = tmp_path / "failed-bound"
    with pytest.raises(ValueError, match="archive_byte_or_member_bound"):
        build(inputs, output)
    assert not (output / "cache-verified.json").exists()


def test_reference_member_corruption_is_not_reused(inputs, tmp_path):
    shard = copy.deepcopy(inputs[0])
    shard["series"][0]["cache_sha256"] = "0" * 64
    reference = worker.ReferenceArchive(inputs[3], shard["series"])
    with pytest.raises(ValueError, match="bytes_changed"):
        reference.read("cache/" + shard["series"][0]["cache_path"])


def test_changed_full_cohort_manifest_is_rejected_before_any_data_read(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="manifest_binding_changed"):
        worker.load_plan(tmp_path, manifest, 0)


def test_original_metadata_accepts_pinned_git_and_lfs_and_still_checks_full_bytes():
    from types import SimpleNamespace

    raw = b"original git-stored source"
    expected = {
        "a.dcm": {"bytes": 50, "storage": "lfs", "sha256": "a" * 64},
        "b.dcm": {"bytes": len(raw), "storage": "git", "sha256": hashlib.sha256(raw).hexdigest()},
    }
    info = [
        SimpleNamespace(path="a.dcm", size=50, lfs=SimpleNamespace(sha256="a" * 64)),
        SimpleNamespace(path="b.dcm", size=len(raw), lfs=None, blob_id="b" * 40),
    ]
    worker.verify_original_metadata(info, expected)
    assert worker.verify_bytes(raw, expected["b.dcm"]) == raw
    with pytest.raises(ValueError, match="source_or_reference_bytes_changed"):
        worker.verify_bytes(raw[:-1] + b"x", expected["b.dcm"])
    for changes in ({"size": 1}, {"blob_id": "unbound"}, {"lfs": SimpleNamespace(sha256="b" * 64)}):
        changed = copy.deepcopy(info)
        for key, value in changes.items():
            setattr(changed[1], key, value)
        with pytest.raises(ValueError, match="private_original_object_binding_changed"):
            worker.verify_original_metadata(changed, expected)
    with pytest.raises(ValueError, match="private_original_object_binding_changed"):
        worker.verify_original_metadata(info[:1], expected)
    wrong_storage = copy.deepcopy(expected)
    wrong_storage["b.dcm"]["storage"] = "anything"
    with pytest.raises(ValueError, match="private_original_object_binding_changed"):
        worker.verify_original_metadata(info, wrong_storage)


def test_explicit_new_plan_binding_does_not_change_original_default(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text("{}")
    with pytest.raises(ValueError, match="binding_changed"):
        worker.load_plan(tmp_path, manifest, 0)
    with pytest.raises(ValueError, match="binding_changed"):
        worker.load_plan(tmp_path, manifest, 0, "not-a-hash")
    # An exact caller-admitted hash passes the byte guard, then must still
    # satisfy the full original plan/schema/science/inventory admission.
    with pytest.raises(KeyError, match="shards"):
        worker.load_plan(tmp_path, manifest, 0, worker.digest(manifest))
