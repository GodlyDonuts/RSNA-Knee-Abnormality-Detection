import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.newton.launch_pixel_match import SOURCE_MEMBERS
from scripts.newton.match_pixel_indexes import read_exact_index, save_private_candidates


def test_metadata_stream_requires_exact_original_bytes_no_truncation_or_suffix():
    raw = b'{"samples": []}\n'
    digest = hashlib.sha256(raw).hexdigest()
    assert read_exact_index(io.BytesIO(raw), len(raw), digest) == raw
    for changed in (raw[:-1], raw + b"x", raw.replace(b"samples", b"samplss")):
        with pytest.raises(ValueError):
            read_exact_index(io.BytesIO(changed), len(raw), digest)


def test_private_candidate_cap_is_enforced_before_writing_excess_bytes(tmp_path):
    pairs = [{"left": "a", "right": "b"}]
    output = tmp_path / "result.json"
    save_private_candidates(output, pairs)
    assert json.loads(output.read_bytes()) == pairs
    bounded = tmp_path / "bounded.json"
    with pytest.raises(ValueError, match="cap"):
        save_private_candidates(bounded, pairs, cap_bytes=12)
    assert bounded.stat().st_size <= 12
    with pytest.raises(FileExistsError):
        save_private_candidates(output, pairs)


def test_original_private_match_package_imports_in_isolated_process(tmp_path):
    root = Path(__file__).parents[1]
    for name in SOURCE_MEMBERS:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(root / name, destination)
    result = subprocess.run(
        [sys.executable, "-B", str(tmp_path / "scripts/newton/match_pixel_indexes.py"), "--help"],
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(tmp_path), "PYTHONNOUSERSITE": "1"},
        text=True,
        capture_output=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert "--control" in result.stdout
