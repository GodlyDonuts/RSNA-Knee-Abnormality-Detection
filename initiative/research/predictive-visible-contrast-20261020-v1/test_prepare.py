import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("contrast_prepare_test", Path(__file__).with_name("prepare_contexts.py"))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def test_new_cache_preserves_every_old_liability_and_floor():
    quota = "/home/sa305415 1026233008 1048576000 1059061760 - 994621 1000000 1010000 -"
    result = prepare.budget(600 * 2**30, 100 * 2**30, 30 * 2**30, quota)
    expected_old = (20 + 68 + 6 + 320 - 100 + 96 - 30) * 2**30 + 594 * 2**20
    assert result["old_required_bytes"] == expected_old
    assert result["total_required_bytes"] == expected_old + 2**30 + 8 * 2**20
    assert result["shared_files_after_old_liabilities"] == 5003
    assert result["new_shared_bytes_or_files"] == 0
    with pytest.raises(AssertionError):
        prepare.budget(result["total_required_bytes"], 100 * 2**30, 30 * 2**30, quota)
    with pytest.raises(AssertionError):
        prepare.budget(600 * 2**30, 100 * 2**30, 30 * 2**30, quota.replace("994621", "994625"))


def test_physical_cache_bytes_count_original_hardlinks_once(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    a.write_bytes(b"unchanged")
    b.hardlink_to(a)
    assert prepare.physical_bytes([a, b]) == len(b"unchanged")
    c = tmp_path / "c"
    c.symlink_to(a)
    with pytest.raises(AssertionError):
        prepare.physical_bytes([c])
