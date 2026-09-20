from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
from test_pipeline import dicom_bytes

from rsna.audit import audit_cache, inventory_fingerprint
from rsna.dicom import decode_slice, make_volume, order_slices, pixel_fingerprint
from rsna.preprocess import Source, cache_series


def modified_dicom(instance: int = 1, **attributes) -> bytes:
    ds = pydicom.dcmread(io.BytesIO(dicom_bytes(instance, instance)))
    for name, value in attributes.items():
        setattr(ds, name, value)
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


@pytest.mark.parametrize("slope,intercept", [(0, 0), (float("nan"), 0), (1, float("inf"))])
def test_invalid_modality_rescale_is_rejected(slope: float, intercept: float) -> None:
    data = modified_dicom(RescaleSlope=slope, RescaleIntercept=intercept)
    with pytest.raises(ValueError, match="rescale"):
        decode_slice(data, "1.2", "1.2.3", "1.2.3.1")


def test_monochrome1_preserves_between_slice_intensity_order() -> None:
    slices = []
    for index, offset in [(1, 0), (2, 1000)]:
        data = modified_dicom(
            index,
            PhotometricInterpretation="MONOCHROME1",
            PixelData=(np.arange(512, dtype=np.uint16) + offset).tobytes(),
        )
        slices.append(decode_slice(data, "1.2", "1.2.3", f"1.2.3.{index}"))
    volume, _ = make_volume(slices, 32)
    assert volume[0].min() > volume[1].max()


@pytest.mark.parametrize(
    "attribute,value,error",
    [
        ("orientation", (2, 0, 0, 0, 1, 0), "orientation vectors"),
        ("orientation", (1, 0, 0, 0.6, 0.8, 0), "orientation vectors"),
        ("position", (1, 2), "Malformed"),
        ("spacing", (0, 1), "pixel spacing"),
        ("spacing", (-1, 1), "pixel spacing"),
    ],
)
def test_invalid_geometry_cannot_pass_as_a_warning(attribute, value, error) -> None:
    item = decode_slice(dicom_bytes(1, 0), "1.2", "1.2.3", "1.2.3.1")
    setattr(item, attribute, value)
    with pytest.raises(ValueError, match=error):
        order_slices([item])


def test_pixel_fingerprint_includes_shape_but_not_storage_encoding() -> None:
    pixels = np.arange(512, dtype=np.uint16).reshape(16, 32)
    assert pixel_fingerprint(pixels) != pixel_fingerprint(pixels.reshape(32, 16))
    assert pixel_fingerprint(pixels) == pixel_fingerprint(pixels.astype(">i4"))


def make_source_cache(tmp_path: Path, second_split: str = "train_series"):
    source, cache = tmp_path / "source", tmp_path / "cache"
    rows = []
    for split, study, series in [("train_series", "1.2", "1.2.3"), (second_split, "2.3", "2.3.4")]:
        path = f"{split}/{study}/{series}/{series}.1.dcm"
        data = modified_dicom(
            StudyInstanceUID=study,
            SeriesInstanceUID=series,
            SOPInstanceUID=f"{series}.1",
        )
        file = source / path
        file.parent.mkdir(parents=True, exist_ok=True)
        file.write_bytes(data)
        rows.append(
            {
                "split": split,
                "study_uid": study,
                "series_uid": series,
                "sop_uid": f"{series}.1",
                "path": path,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
        cache_series(Source(source), [path], cache, 32, "a" * 40)
    return pd.DataFrame(rows), cache


def test_duplicate_detection_uses_source_pixels_even_if_resize_collapses_them(tmp_path: Path):
    inventory, cache = make_source_cache(tmp_path, "test_series")
    for path in cache.glob("*/*/*.npz"):
        meta_path = path.with_suffix(".json")
        metadata = json.loads(meta_path.read_text())
        # A low-contrast source slice may become constant after resizing/quantization.
        np.savez_compressed(path, volume=np.zeros((1, 32, 32), dtype=np.uint16))
        metadata["cache_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        metadata["cache_bytes"] = path.stat().st_size
        meta_path.write_text(json.dumps(metadata))
    report = audit_cache(inventory, cache, "a" * 40, tmp_path / "audit")
    assert report["cross_study_pixel_duplicate_pairs"] == [("1.2", "2.3")]
    assert report["train_test_duplicate_pairs"] == [("1.2", "2.3")]
    assert report["constant_source_slices_excluded_from_duplicate_detection"] == 0
    assert report["complete"] is True
    assert report["audit_schema_version"] == 2
    assert report["inventory_fingerprint_sha256"] == inventory_fingerprint(inventory.iloc[::-1])
    index = tmp_path / "audit/series_cache_index.parquet"
    assert report["series_cache_index_sha256"] == hashlib.sha256(index.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("n_slices", 2, "metadata coverage"),
        ("positions", [], "Position metadata"),
        ("cache_bytes", 1, "byte size"),
        ("resized_shape", [0, 32], "resized volume"),
        ("intensity_max", float("nan"), "intensity range"),
        ("pixel_nonconstant", [], "constancy"),
        ("path_fingerprint", "invalid", "fingerprint"),
    ],
)
def test_audit_rejects_inconsistent_metadata(tmp_path: Path, field, value, error):
    inventory, cache = make_source_cache(tmp_path)
    path = next(cache.glob("*/*/*.json"))
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match=error):
        audit_cache(inventory, cache, "a" * 40, tmp_path / "audit")


def test_truncated_download_cannot_create_cache(tmp_path: Path, monkeypatch):
    source = Source(tmp_path)
    monkeypatch.setattr(source, "read_series", lambda paths: [])
    with pytest.raises(ValueError, match="incomplete source coverage"):
        cache_series(source, ["train_series/1.2/1.2.3/1.2.3.1.dcm"], tmp_path, 32, "fixture")
    assert not list(tmp_path.glob("*/*/*.npz"))
