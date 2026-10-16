import ast

import pytest

from scripts.kaggle import licensed_pixel_cache_full as full


def test_all_four_shards_cover_full_original_population_once_without_gap_or_overlap():
    rows = []
    for i in range(4):
        lo, hi = full.shard_bounds(i)
        assert hi - lo <= 972
        rows.extend(range(lo, hi))
    assert rows == list(range(3887))
    assert [full.shard_bounds(i) for i in range(4)] == [
        (0, 972),
        (972, 1944),
        (1944, 2916),
        (2916, 3887),
    ]


@pytest.mark.parametrize("bad", [-1, 4, True, 1.2])
def test_no_silent_invalid_shard_coercion(bad):
    with pytest.raises(ValueError):
        full.shard_bounds(bad)


def test_full_entry_keeps_probe_unused_and_enforces_full_index_and_original32_reuse(monkeypatch):
    base = "def _pixel_cache_main():pass\nif __name__=='__main__':\n    _pixel_cache_main()\n"
    monkeypatch.setattr(full, "probe_build", lambda p, c: (base, {"CPU_only": True}))
    source, binding = full.build("synthetic", "synthetic", 3)
    assert binding["lo"] == 2916 and binding["hi"] == 3887
    assert binding["global_fitting_studies"] == 3887
    assert binding["maximum_output_bytes"] == 14 * 2**30
    parsed = ast.parse(source)
    assert isinstance(parsed.body[-1], ast.If)
    assert "_pixel_full_main()" in ast.unparse(parsed.body[-1])
    assert "_pixel_cache_main()" not in ast.unparse(parsed.body[-1])
    assert full.FULL_INDEX_SHA in source and full.PROBE_INDEX_SHA in source
    assert "original32_reused_without_decode_reencode_or_new_archive_copy" in source
    assert "GPU_fitting_admitted" in source
