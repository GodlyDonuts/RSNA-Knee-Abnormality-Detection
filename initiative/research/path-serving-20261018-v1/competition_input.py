"""Bounded test-DICOM adapter for the NEW dense path serving runner.

Only image metadata and test study identities enter. The accepted decoder and
series-recovery implementation are reused without changing their policy.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import time
from pathlib import Path

import pandas as pd
from online_inputs_fp32 import StreamedFP32PathData
from serving import PARENT

from rsna.competitive.data import CompetitiveStudyDataset
from scripts.kaggle.chunked_infer import prepare_usable_series

COLUMNS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Anatomical_Plane",
    "Fluid_Sensitive",
    "Fat_Suppression",
]
UID = re.compile(r"[0-9]+(?:\.[0-9]+)*\Z")


class CompetitionPathProvider:
    """One disposable native384 study cache per GPU process; no retained MRI copies."""

    def __init__(
        self,
        competition: Path,
        scratch: Path,
        parent,
        *,
        parent_sha256: str,
        capacity_bytes: int = 2 * 2**30,
    ):
        if parent_sha256 != PARENT or type(capacity_bytes) is not int or capacity_bytes < 2**20:
            raise ValueError("Original parent identity and explicit cache budget required")
        if scratch.exists():
            raise FileExistsError("Test scratch must be a fresh owned path")
        # Read only explicit image metadata and the actual test roster, never a
        # public sample CSV, clinical target columns or a training/report table.
        roster = pd.read_csv(competition / "test.csv", usecols=["StudyInstanceUID"], dtype=str)
        ids = roster.StudyInstanceUID.tolist()
        if (
            not ids
            or len(set(ids)) != len(ids)
            or any(not isinstance(u, str) or not UID.fullmatch(u) for u in ids)
        ):
            raise ValueError("Complete distinct original DICOM study UID roster required")
        frame = pd.read_csv(
            competition / "test_series.csv",
            usecols=COLUMNS,
            dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str},
        )
        frame = frame.loc[frame.StudyInstanceUID.isin(ids)].copy()
        if frame.SeriesInstanceUID.duplicated().any() or any(
            not isinstance(u, str) or not UID.fullmatch(u) for u in frame.SeriesInstanceUID
        ):
            raise ValueError("Unique original DICOM series identities required")
        if set(frame.StudyInstanceUID) != set(ids):
            raise ValueError("Test MRI metadata does not cover the actual test roster")
        scratch.mkdir(mode=0o700, parents=True)
        self.root, self.scratch, self.parent = competition.resolve(), scratch.resolve(), parent
        self.ids, self.frame, self.capacity = ids, frame, capacity_bytes
        self.original_binding = {
            "parent_sha256": PARENT,
            "role": "competition_test_image_only",
            "input_policy": "original_native384_sparse16_parent_dense_path",
            "test_roster_sha256": hashlib.sha256(
                json.dumps(ids, separators=(",", ":")).encode()
            ).hexdigest(),
        }
        self.binding = {
            "original": self.original_binding,
            "sampling": "dense",
            "encoder_precision": "float32",
            "encoder_chunk_size": 12,
        }
        self.completed = 0
        self.decode_seconds = 0.0
        self.peak_cache_bytes = 0
        self.failures = {}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        uid = self.ids[index]
        selected = self.frame.loc[self.frame.StudyInstanceUID == uid].copy()
        # Bound even the worst selected/recovery alternatives before decoding;
        # cache capacity applies to all attempted series, including rejected ones.
        count = sum(
            len(list((self.root / "test_series" / uid / sid).glob("*.dcm")))
            for sid in selected.SeriesInstanceUID
        )
        upper = count * (384 * 384 * 2 + 4096) + len(selected) * 65536
        if upper > self.capacity or shutil.disk_usage(self.scratch).free < 3 * 2**30 + upper:
            raise OSError("Current study cache cannot fit its budget and3GiB free reserve")
        owned = Path(tempfile.mkdtemp(prefix="study-", dir=self.scratch))
        started = time.monotonic()
        # On failure the NEW disposable cache is retained for diagnosis. The
        # original competition tree is read-only and never belongs to this path.
        usable, receipt = prepare_usable_series(
            selected,
            [uid],
            dicom_root=self.root,
            cache=owned,
            cache_side=384,
            max_series=6,
            source_revision="kaggle-original-test-image-only",
            allow_unusable_studies=False,
            recover_slices=True,
            validate_geometry=True,
            series_sampling="selected",
            single_decode=True,
        )
        actual = sum(p.stat().st_size for p in owned.rglob("*") if p.is_file())
        if actual > self.capacity:
            raise OSError("Actual current study cache exceeded the explicit bound")
        self.peak_cache_bytes = max(self.peak_cache_bytes, actual)
        self.decode_seconds += time.monotonic() - started
        dataset = CompetitiveStudyDataset(
            pd.DataFrame({"study_uid": [uid]}),
            usable,
            cache=owned,
            slices=16,
            side=384,
            max_series=6,
            augment=False,
            raw_split="test_series",
            slice_sampling="uniform",
            series_sampling="selected",
        )
        data = StreamedFP32PathData(
            dataset,
            self.parent,
            self.original_binding,
            sampling="dense",
            seed=20261017,
            chunk_size=12,
        )
        row = data[0]
        if row["study_uid"] != uid or bool(row["label_mask"].any()):
            raise ValueError("Only current test-study pixels and geometry may enter serving")
        # All maps/geometry are now in the returned in-memory row. The provider
        # retains no dataset or metadata paths referring to this disposable cache.
        del data, dataset
        shutil.rmtree(owned)
        self.completed += 1
        for key, value in receipt["skipped_series"].items():
            self.failures[key] = self.failures.get(key, 0) + value
        return row

    def statistics(self):
        return {
            "completed_image_studies": self.completed,
            "native_decode_seconds": self.decode_seconds,
            "peak_owned_study_cache_bytes": self.peak_cache_bytes,
            "capacity_bytes": self.capacity,
            "skipped_series": dict(self.failures),
            "source_images_modified": False,
            "clinical_reports_or_labels_read": False,
            "actual_T4_runtime_qualified": False,
        }
