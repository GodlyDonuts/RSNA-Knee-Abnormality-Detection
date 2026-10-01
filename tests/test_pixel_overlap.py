import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from rsna.pixel_overlap import overlap_candidates, slice_signatures, volume_signature
from scripts.newton.launch_pixel_index import SOURCE_MEMBERS


def image():
    rng = np.random.default_rng(32)
    low = rng.integers(1, 250, (24, 24), dtype=np.uint8)
    return np.repeat(np.repeat(low, 4, axis=0), 4, axis=1)


def test_orientation_intensity_and_zero_margin_changes_keep_a_matching_signature():
    original = image()
    expected = set(slice_signatures(original))
    assert expected
    assert expected == set(slice_signatures(np.rot90(original).astype(np.uint16) * 200 + 20))
    assert expected == set(slice_signatures(np.fliplr(original)))
    assert expected & set(slice_signatures(np.pad(original, 12)))


def test_constant_or_invalid_slices_do_not_masquerade_as_independent_images():
    assert slice_signatures(np.zeros((32, 32), dtype=np.uint16)) == []
    with pytest.raises(ValueError):
        slice_signatures(np.full((32, 32), float("nan")))
    short = volume_signature(np.stack([image()] * 3))
    assert [v["slice"] for v in short] == [0, 1, 2]


def rows(uid, count, number):
    return [
        {
            "study": uid,
            "samples": [
                {"slice": i, "hashes": [f"{number ^ (65535 << (16 * i)):016x}"]}
                for i in range(count)
            ],
        }
    ]


def test_multi_chunk_lookup_finds_all_supported_hamming_distances():
    original = 0x8732479281463712
    changed = original ^ (1 << 3) ^ (1 << 22) ^ (1 << 38)
    result = overlap_candidates(rows("a", 3, original), rows("b", 3, changed))
    assert len(result) == 1
    assert all(row[2] == 3 for row in result[0]["slice_matches"])
    assert result[0]["status"].endswith("original_review_required")
    with pytest.raises(ValueError):
        overlap_candidates([], [], max_hamming=4)


def test_repeated_matches_to_only_one_slice_cannot_accept_a_volume_pair():
    original = 0x8732479281463712
    assert not overlap_candidates(rows("a", 1, original), rows("b", 3, original))
    repeated = [{"samples": [{"slice": i, "hashes": [f"{original:016x}"]} for i in range(8)]}]
    assert not overlap_candidates(repeated, repeated)


def test_private_frozen_source_closure_imports_outside_repository(tmp_path):
    root = Path(__file__).resolve().parents[1]
    for name in SOURCE_MEMBERS:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
    process = subprocess.run(
        [sys.executable, "-B", str(tmp_path / "scripts/newton/index_source_pixels.py"), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path), "PYTHONNOUSERSITE": "1"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert process.returncode == 0, process.stderr
    assert "--receipt" in process.stdout
