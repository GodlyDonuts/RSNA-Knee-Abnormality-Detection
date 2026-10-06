"""Rebuild real fixture DICOMs and retain identical model tensors under eviction."""

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from test_pipeline import dicom_bytes

from rsna.competitive.data import CompetitiveStudyDataset
from rsna.competitive.streaming_cache import StreamingCache
from rsna.preprocess import Source, cache_series
from rsna.schema import TARGETS


def cohort(tmp_path):
    raw = tmp_path / "raw"
    reference = tmp_path / "retained"
    records = []
    metas = []
    for number in range(1, 4):
        study = f"1.2.{number}"
        series = f"1.2.3.{number}"
        paths = []
        for i in range(1, 4):
            import io

            import pydicom

            blob = pydicom.dcmread(io.BytesIO(dicom_bytes(i, float(i))))
            blob.StudyInstanceUID = study
            blob.SeriesInstanceUID = series
            blob.SOPInstanceUID = f"{series}.{i}"
            blob.file_meta.MediaStorageSOPInstanceUID = blob.SOPInstanceUID
            p = raw / "train_series" / study / series / (blob.SOPInstanceUID + ".dcm")
            p.parent.mkdir(parents=True, exist_ok=True)
            blob.save_as(p, enforce_file_format=True)
            paths.append(str(p.relative_to(raw)))
        meta = cache_series(Source(raw), paths, reference, 32, "same-original-revision")
        metas.append(meta)
        records.append(
            dict(
                StudyInstanceUID=study,
                SeriesInstanceUID=series,
                Anatomical_Plane="Sagittal",
                Fluid_Sensitive=1,
                Fat_Suppression=0,
            )
        )
    studies = pd.DataFrame(
        [
            {
                "study_uid": m["study_uid"],
                **{t: float(i % 2) for t in TARGETS},
                **{t + "__mask": 1 for t in TARGETS},
            }
            for i, m in enumerate(metas)
        ]
    )
    return raw, reference, pd.DataFrame(records), studies, metas


def test_owned_LRU_rebuilds_exact_payloads_and_preserves_all_dataset_tensors(tmp_path):
    raw, reference, series, studies, metas = cohort(tmp_path)
    metadata_root = tmp_path / "metadata-only"
    for m in metas:
        p = metadata_root / Path(m["cache_path"]).with_suffix(".json")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes((reference / Path(m["cache_path"]).with_suffix(".json")).read_bytes())
    provider = StreamingCache(
        Source(raw),
        tmp_path / "scratch",
        capacity_bytes=max(
            m["cache_bytes"] + len(json.dumps(m, sort_keys=True).encode()) for m in metas
        ),
    )
    baseline = CompetitiveStudyDataset(studies, series, reference, slices=2, side=32, max_series=3)
    streamed = CompetitiveStudyDataset(
        studies,
        series,
        metadata_root,
        slices=2,
        side=32,
        max_series=3,
        volume_provider=provider.read,
    )
    signatures = {
        p: (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest())
        for p in reference.rglob("*")
        if p.is_file()
    }
    for i in [0, 1, 2, 0]:
        expected = baseline[i]
        actual = streamed[i]
        for k, v in expected.items():
            if isinstance(v, torch.Tensor):
                torch.testing.assert_close(actual[k], v, rtol=0, atol=0)
            else:
                assert actual[k] == v
        assert provider.statistics()["resident_volumes"] == 1
    assert all(
        (p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()) == s
        for p, s in signatures.items()
    )
    assert len(list((tmp_path / "scratch").rglob("*.npz"))) == 1


def test_changed_volume_or_source_binding_cannot_be_silently_reused(tmp_path):
    raw, _, _, _, metas = cohort(tmp_path)
    cache = StreamingCache(Source(raw), tmp_path / "scratch", capacity_bytes=1000000)
    cache.read(metas[0])
    altered = dict(metas[0], cache_sha256="0" * 64)
    with pytest.raises(ValueError, match="binding changed"):
        cache.read(altered)
    p = tmp_path / "scratch" / metas[0]["cache_path"]
    p.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="bytes differ"):
        cache.read(metas[0])


def test_source_identity_and_cap_are_checked_before_decode(tmp_path):
    raw, _, _, _, metas = cohort(tmp_path)
    cache = StreamingCache(Source(raw), tmp_path / "scratch", capacity_bytes=1)
    with pytest.raises(ValueError, match="exceeds"):
        cache.read(metas[0])
    altered = dict(metas[0], cache_path="../retained/source.npz")
    with pytest.raises(ValueError, match="identity"):
        cache.read(altered)
    with pytest.raises(ValueError, match="fresh owned"):
        StreamingCache(Source(raw), tmp_path / "scratch", capacity_bytes=100)
