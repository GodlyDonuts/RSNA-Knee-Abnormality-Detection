import hashlib
import json

import pytest
import torch
from test_competitive_data import fixture

from rsna.competitive.cache_union import MetadataUnion
from rsna.competitive.data import CompetitiveStudyDataset


def test_union_reuses_pixels_and_preserves_all_native_loader_tensors(tmp_path):
    source = tmp_path / "original"
    source.mkdir()
    studies, series, paths = fixture(source)
    union = MetadataUnion.build(tmp_path / "union", [source], series, set(studies.study_uid))
    direct = CompetitiveStudyDataset(studies, series, source, side=16, slices=16)[0]
    actual = CompetitiveStudyDataset(
        studies, series, union.cache, volume_provider=union, side=16, slices=16
    )[0]
    for name, value in direct.items():
        if isinstance(value, torch.Tensor):
            assert torch.equal(value, actual[name]), name
        else:
            assert value == actual[name]
    assert not list(union.cache.rglob("*.npz"))
    assert all(p.exists() for p in paths)
    for p in paths:
        assert (
            union.cache / p.relative_to(source).with_suffix(".json")
        ).read_bytes() == p.with_suffix(".json").read_bytes()


def test_duplicate_source_ownership_is_not_a_silent_fallback(tmp_path):
    roots = [tmp_path / "first", tmp_path / "second"]
    for p in roots:
        p.mkdir()
    studies, series, _ = fixture(roots[0])
    fixture(roots[1])
    with pytest.raises(ValueError, match="exactly one"):
        MetadataUnion.build(tmp_path / "union", roots, series, set(studies.study_uid))


def test_source_and_copied_metadata_tampering_refused(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    studies, series, paths = fixture(source)
    union = MetadataUnion.build(tmp_path / "union", [source], series, set(studies.study_uid))
    index = union.cache / "index.json"
    digest = hashlib.sha256(index.read_bytes()).hexdigest()
    p = union.cache / paths[0].relative_to(source).with_suffix(".json")
    metadata = json.loads(p.read_text())
    metadata["positions"][0][0] += 3
    p.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="metadata changed"):
        MetadataUnion.load(index, digest)
    with pytest.raises(ValueError, match="metadata differs"):
        union(metadata)
    p.write_bytes(paths[0].with_suffix(".json").read_bytes())
    paths[0].write_bytes(b"changed original payload")
    with pytest.raises(ValueError, match="stat changed"):
        union(json.loads(p.read_text()))


def test_escaped_paths_and_changed_index_refused(tmp_path):
    from rsna.competitive.cache_union import relative_path

    with pytest.raises(ValueError):
        relative_path("../escape", "sid")
    with pytest.raises(ValueError):
        relative_path("uid", "../sid")
    source = tmp_path / "source"
    source.mkdir()
    studies, series, _ = fixture(source)
    union = MetadataUnion.build(tmp_path / "union", [source], series, set(studies.study_uid))
    with pytest.raises(ValueError, match="index required"):
        MetadataUnion.load(union.cache / "index.json", "0" * 64)


def test_original_loader_still_hashes_payload_even_if_stat_is_reproduced(tmp_path):
    import os

    source = tmp_path / "source"
    source.mkdir()
    studies, series, paths = fixture(source)
    union = MetadataUnion.build(tmp_path / "union", [source], series, set(studies.study_uid))
    data = CompetitiveStudyDataset(studies, series, union.cache, volume_provider=union, side=16)
    path = paths[0]
    stat = path.stat()
    raw = path.read_bytes()
    path.write_bytes(bytes([raw[0] ^ 1]) + raw[1:])
    os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
    with pytest.raises(ValueError, match="checksum/size"):
        data[0]
