"""Study isolation and bounded original-pixel shard planning."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/newton/native_cache_shards.py"
SPEC = importlib.util.spec_from_file_location("native_cache_shards", PATH)
assert SPEC and SPEC.loader
planner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(planner)


def row(study: str, series: str, slices: int = 200) -> dict:
    return {
        "study_uid": study,
        "series_uid": series,
        "n_slices": slices,
        "source_bytes": slices * 1000,
    }


def test_study_complete_bounds_are_deterministic_with_input_permutation():
    rows = [row(str(i), str(i) + "a") for i in range(15)]
    rows += [row(str(i), str(i) + "b") for i in range(15)]
    shards = planner.pack_studies(rows, planner.GIB)
    assert shards == planner.pack_studies(list(reversed(rows)), planner.GIB)
    seen = set()
    for shard in shards:
        assert not seen.intersection(shard["study_uids"])
        seen.update(shard["study_uids"])
        assert shard["paired_archive_upper_bytes"] <= planner.GIB
        assert (
            sum(shard["archive_upper_bytes_by_side"].values())
            == shard["paired_archive_upper_bytes"]
        )
        assert shard["regular_members_per_side"] == 2 * shard["series_count"]
        for uid in shard["study_uids"]:
            assert {r["series_uid"] for r in shard["series"] if r["study_uid"] == uid} == {
                uid + "a",
                uid + "b",
            }
    assert seen == {str(i) for i in range(15)}


def test_large_study_is_rejected_without_dropping_or_splitting_series():
    with pytest.raises(ValueError, match="single_study_exceeds_shard_bound"):
        planner.pack_studies([row("a", "a1", 2000), row("a", "a2", 2000)], planner.GIB)


@pytest.mark.parametrize("rows", [[], [row("a", "1"), row("a", "1")]])
def test_empty_or_duplicate_series_are_rejected(rows):
    with pytest.raises(ValueError, match="unique_series"):
        planner.pack_studies(rows, planner.GIB)


@pytest.mark.parametrize("count", [0, -1, 1.5])
def test_invalid_depth_is_rejected(count):
    with pytest.raises(ValueError, match="slice_count"):
        planner.pack_studies([row("a", "1", count)], planner.GIB)


def test_original_pixel_bounds_include_both_resolutions_and_noncompressed_overhead():
    result = planner.bound_series(row("a", "1", 123))
    for side in (256, 384):
        assert result["cache_archive_upper_bytes_by_side"][str(side)] > 123 * side * side * 2


def test_corrected_inventory_is_pinned_and_changed_bytes_are_rejected(tmp_path):
    p = tmp_path / planner.INVENTORY
    p.parent.mkdir(parents=True)
    p.write_bytes(b"not the corrected official inventory")
    with pytest.raises(ValueError, match="compressed_binding_changed"):
        planner.effective_inventory(tmp_path)


def test_shard_bound_is_not_silently_rounded_up():
    with pytest.raises(ValueError, match="one_gib"):
        planner.pack_studies([row("a", "1")], planner.GIB - 1)
