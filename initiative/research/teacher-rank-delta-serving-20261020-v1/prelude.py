"""Stdlib-only exact-roster watchdog, inserted before all original heavy imports."""
# ruff: noqa: F821

import csv
import hashlib
import json
import os
import threading
import time
from pathlib import Path

_RANK_DELTA_OUTPUT = Path("/kaggle/working/submission.csv")
_RANK_DELTA_RECEIPT = Path("/kaggle/working/licensed-IID16-epoch-public-accepted.json")
_RANK_DELTA_MOUNTS = (
    Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
    Path("/kaggle/input/rsna-knee-abnormality-detection"),
)


def _rank_delta_discard_outputs():
    for path in (_RANK_DELTA_OUTPUT, _RANK_DELTA_RECEIPT):
        if path.exists():
            path.unlink()


def _rank_delta_expired():
    _rank_delta_discard_outputs()
    print(json.dumps({"teacher_rank_public_watchdog_expired": True}), flush=True)
    os._exit(124)


def _rank_delta_early(mounts):
    started = time.monotonic()
    assert not _RANK_DELTA_OUTPUT.exists() and not _RANK_DELTA_RECEIPT.exists()
    roots = [p for p in mounts if (p / "test.csv").is_file()]
    assert len(roots) == 1, "One unambiguous declared competition mount required"
    root = roots[0]
    assert (root / "test.csv").stat().st_size <= 16 * 2**20
    with (root / "test.csv").open(newline="") as handle:
        records = csv.DictReader(handle)
        assert "StudyInstanceUID" in records.fieldnames
        ids = [row["StudyInstanceUID"] for row in records]
    assert ids and len(ids) == len(set(ids)) and all(isinstance(s, str) and s for s in ids)
    sha = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    timer = None
    if sha == _RANK_DELTA_BINDING["public_roster_sha256"]:
        assert len(ids) == 3
        timer = threading.Timer(_RANK_DELTA_BINDING["public_watchdog_seconds"], _rank_delta_expired)
        timer.daemon = True
        timer.start()
    return {"root": root, "ids": ids, "sha256": sha, "started": started, "timer": timer}


if __name__ == "__main__":
    _RANK_DELTA_EARLY = _rank_delta_early(_RANK_DELTA_MOUNTS)
