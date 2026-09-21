from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
from test_pipeline import dicom_bytes

from rsna.audit import audit_cache
from rsna.preprocess import Source, cache_series
from rsna.series_equivalence import (
    _recheck_plan_inputs,
    build_selection_plan,
    canonical_bytes,
    eligibility_reason,
    equivalence_reason,
    export_series_selection,
)


def fixture(root: Path, other_study: str | None = None, test_overlap: bool = False) -> dict:
    cache, source = root / "cache", root / "source"
    definitions = [
        ("train_series", "1.1", "1.1.1", 0),
        ("train_series", "1.1", "1.1.2", 0),
        ("train_series", "1.1", "1.1.3", 1000),
    ]
    if other_study:
        definitions.append(
            ("test_series" if test_overlap else "train_series", other_study, other_study + ".1", 0)
        )
    inventory_rows, series_rows = [], []
    for split, study, series, offset in definitions:
        paths = []
        for number in [1, 2]:
            sop = f"{series}.{number}"
            ds = pydicom.dcmread(io.BytesIO(dicom_bytes(number, number)))
            ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID = study, series, sop
            ds.file_meta.MediaStorageSOPInstanceUID = sop
            ds.FrameOfReferenceUID = f"{series}.99"
            ds.PixelData = (np.arange(512, dtype=np.uint16) + offset + number * 17).tobytes()
            buffer = io.BytesIO()
            ds.save_as(buffer, enforce_file_format=True)
            data = buffer.getvalue()
            path = f"{split}/{study}/{series}/{sop}.dcm"
            file = source / path
            file.parent.mkdir(parents=True, exist_ok=True)
            file.write_bytes(data)
            paths.append(path)
            inventory_rows.append(
                {
                    "split": split,
                    "study_uid": study,
                    "series_uid": series,
                    "sop_uid": sop,
                    "path": path,
                    "sha256": hashlib.sha256(data).hexdigest(),
                    "bytes": len(data),
                }
            )
        cache_series(Source(source), paths, cache, 32, "a" * 40)
        if split == "train_series":
            series_rows.append(
                {
                    "StudyInstanceUID": study,
                    "SeriesInstanceUID": series,
                    "Fluid_Sensitive": 0,
                    "Fat_Suppression": 0,
                    "Anatomical_Plane": "Sagittal",
                }
            )
    inventory = pd.DataFrame(inventory_rows)
    inventory_path = root / "inventory.parquet"
    inventory.to_parquet(inventory_path, index=False)
    audit_cache(inventory, cache, "a" * 40, root / "audit")
    series_path = root / "series.csv"
    pd.DataFrame(series_rows[::-1]).to_csv(series_path, index=False)
    return {
        "series_path": series_path,
        "inventory_path": inventory_path,
        "audit_path": root / "audit/dicom_audit.json",
        "cache": cache,
        "source_revision": "a" * 40,
    }


def meta_path(inputs: dict, uid: str = "1.1.1") -> Path:
    return inputs["cache"] / "train_series/1.1" / f"{uid}.json"


def test_exact_pair_is_selected_once_with_full_provenance_and_stable_plan(tmp_path: Path):
    inputs = fixture(tmp_path)
    source_before = inputs["series_path"].read_bytes()
    original_meta = meta_path(inputs).read_bytes()
    frame, plan = build_selection_plan(**inputs)
    assert frame.SeriesInstanceUID.tolist() == ["1.1.3", "1.1.1"]
    assert plan["n_excluded_series"] == 1 and plan["n_equivalence_groups"] == 1
    assert plan["series"][1]["representative_series_uid"] == "1.1.1"
    assert plan["series"][1]["retained"] is False
    assert plan["series"][1]["source_row"]["SeriesInstanceUID"] == "1.1.2"
    assert canonical_bytes(plan) == canonical_bytes(build_selection_plan(**inputs)[1])
    output = tmp_path / "output"
    trace = export_series_selection(**inputs, output=output)
    assert (
        trace["plan_sha256"]
        == hashlib.sha256((output / "series_selection_plan.json").read_bytes()).hexdigest()
    )
    assert trace["all_input_cache_bytes_sha256_reverified"] is True
    assert trace["n_retained_series"] == 2
    assert inputs["series_path"].read_bytes() == source_before
    assert meta_path(inputs).read_bytes() == original_meta
    assert meta_path(inputs, "1.1.2").with_suffix(".npz").is_file()
    with pytest.raises(ValueError, match="new directory"):
        export_series_selection(**inputs, output=output)


def test_same_normalized_cache_but_different_native_pixels_is_retained(tmp_path: Path):
    inputs = fixture(tmp_path)
    first = np.load(meta_path(inputs).with_suffix(".npz"))["volume"]
    shifted = np.load(meta_path(inputs, "1.1.3").with_suffix(".npz"))["volume"]
    np.testing.assert_array_equal(first, shifted)
    frame, plan = build_selection_plan(**inputs)
    assert "1.1.3" in set(frame.SeriesInstanceUID)
    group = next(g for g in plan["groups"] if "1.1.3" in g["series_uids"])
    assert group["series_uids"] == ["1.1.3"]


@pytest.mark.parametrize(
    "field,value", [("Anatomical_Plane", "Coronal"), ("Fluid_Sensitive", 1), ("Fat_Suppression", 1)]
)
def test_model_effective_feature_conflict_retains_both(tmp_path: Path, field: str, value):
    inputs = fixture(tmp_path)
    frame = pd.read_csv(inputs["series_path"], dtype={"StudyInstanceUID": str})
    frame.loc[frame.SeriesInstanceUID.eq("1.1.2"), field] = value
    frame.to_csv(inputs["series_path"], index=False)
    selected, plan = build_selection_plan(**inputs)
    assert len(selected) == 3 and plan["n_excluded_series"] == 0
    assert plan["candidate_comparisons"][0]["reason"] == "model_effective_features_differ"


def test_cross_study_exact_series_are_not_deleted_and_leakage_edge_survives(tmp_path: Path):
    inputs = fixture(tmp_path, other_study="2.2")
    selected, plan = build_selection_plan(**inputs)
    assert set(selected.StudyInstanceUID) == {"1.1", "2.2"}
    assert "2.2.1" in set(selected.SeriesInstanceUID)
    assert plan["cross_study_training_edges_preserved"] == [["1.1", "2.2"]]
    assert plan["n_excluded_series"] == 1


def test_train_test_pixel_overlap_blocks_before_any_selection_export(tmp_path: Path):
    inputs = fixture(tmp_path, other_study="2.2", test_overlap=True)
    with pytest.raises(ValueError, match="Train/test image duplicates"):
        export_series_selection(**inputs, output=tmp_path / "output")
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("field", ["pixel_sha256", "pixel_nonconstant", "sop_uids", "positions"])
def test_missing_native_slice_metadata_blocks(tmp_path: Path, field: str):
    inputs = fixture(tmp_path)
    path = meta_path(inputs)
    meta = json.loads(path.read_text())
    meta[field].pop()
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        build_selection_plan(**inputs)


@pytest.mark.parametrize(
    "which",
    ["series", "inventory", "audit", "index", "metadata", "candidate_pixels", "singleton_pixels"],
)
def test_changed_bound_input_after_planning_blocks(tmp_path: Path, which: str):
    inputs = fixture(tmp_path)
    _, plan = build_selection_plan(**inputs)
    if which == "inventory":
        frame = pd.read_parquet(inputs["inventory_path"])
        frame.loc[0, "sha256"] = "f" * 64
        frame.to_parquet(inputs["inventory_path"], index=False)
    else:
        path = {
            "series": inputs["series_path"],
            "audit": inputs["audit_path"],
            "index": inputs["audit_path"].parent / "series_cache_index.parquet",
            "metadata": meta_path(inputs),
            "candidate_pixels": meta_path(inputs).with_suffix(".npz"),
            "singleton_pixels": meta_path(inputs, "1.1.3").with_suffix(".npz"),
        }[which]
        data = bytearray(path.read_bytes())
        data[len(data) // 2] ^= 1  # Same byte length: size-only checking cannot detect this.
        path.write_bytes(data)
    with pytest.raises(ValueError, match="[Cc]hanged"):
        _recheck_plan_inputs(plan)


@pytest.mark.parametrize(
    "field,value",
    [("complete", False), ("decoded_slices", 1), ("inventory_fingerprint_sha256", "f" * 64)],
)
def test_incomplete_or_unbound_full_audit_blocks(tmp_path: Path, field: str, value):
    inputs = fixture(tmp_path)
    path = inputs["audit_path"]
    audit = json.loads(path.read_text())
    audit[field] = value
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        build_selection_plan(**inputs)


@pytest.mark.parametrize("mutation", ["missing", "wrong_study", "duplicate"])
def test_series_inventory_membership_must_be_exact(tmp_path: Path, mutation: str):
    inputs = fixture(tmp_path)
    frame = pd.read_csv(inputs["series_path"], dtype={"StudyInstanceUID": str})
    if mutation == "missing":
        frame = frame.iloc[1:]
    elif mutation == "wrong_study":
        frame.loc[0, "StudyInstanceUID"] = "8.8"
    else:
        frame = pd.concat([frame, frame.iloc[[0]]])
    frame.to_csv(inputs["series_path"], index=False)
    with pytest.raises(ValueError):
        build_selection_plan(**inputs)


def test_geometry_reverse_order_and_constant_slices_are_conservative(tmp_path: Path):
    inputs = fixture(tmp_path)
    left = json.loads(meta_path(inputs).read_text())
    right = json.loads(meta_path(inputs, "1.1.2").read_text())
    row = {"Anatomical_Plane": "Sagittal", "Fluid_Sensitive": "0", "Fat_Suppression": "0"}
    assert equivalence_reason(left, right, row, row) is None
    changed = copy.deepcopy(right)
    changed["positions"][-1][2] += 0.1
    assert equivalence_reason(left, changed, row, row) == "recorded_geometry_differs"
    changed = copy.deepcopy(right)
    changed["orientation"][1] += 5.2e-7  # The real 077 pair has a harmless 5.2e-7 IOP difference.
    assert equivalence_reason(left, changed, row, row) is None
    changed = copy.deepcopy(right)
    changed["pixel_sha256"].reverse()
    assert (
        equivalence_reason(left, changed, row, row)
        == "complete_ordered_native_pixels_or_shape_differ"
    )
    changed = copy.deepcopy(right)
    changed["pixel_nonconstant"] = [False, False]
    assert eligibility_reason(changed) == "no_nonconstant_native_slice"
    changed = copy.deepcopy(right)
    changed["warnings"] = ["duplicate_slice_positions"]
    assert (
        equivalence_reason(left, changed, row, row)
        == "cache_geometry_or_other_warning_requires_review"
    )


def test_pairwise_geometry_prevents_tolerance_chain(tmp_path: Path):
    inputs = fixture(tmp_path)
    # Turn third native sequence into the first in this synthetic fixture, and record
    # successive sub-tolerance position shifts: A-B and B-C pass, A-C fails.
    first = json.loads(meta_path(inputs).read_text())
    for uid, shift in [("1.1.2", 0.0008), ("1.1.3", 0.0016)]:
        path = meta_path(inputs, uid)
        meta = json.loads(path.read_text())
        meta["pixel_sha256"] = first["pixel_sha256"]
        meta["positions"] = [[x, y, z + shift] for x, y, z in first["positions"]]
        path.write_text(json.dumps(meta))
    selected, plan = build_selection_plan(**inputs)
    assert selected.SeriesInstanceUID.tolist() == ["1.1.3", "1.1.1"]
    assert plan["n_excluded_series"] == 1


def test_reviewed_effective_plane_lineage_is_preserved(tmp_path: Path):
    inputs = fixture(tmp_path)
    frame = pd.read_csv(inputs["series_path"], dtype={"StudyInstanceUID": str})
    frame["Anatomical_Plane_Original"] = "Coronal"
    frame["Anatomical_Plane_Review_ID"] = "synthetic-reviewed-plane"
    frame["Anatomical_Plane_Review_SHA256"] = "a" * 64
    frame.to_csv(inputs["series_path"], index=False)
    selected, plan = build_selection_plan(**inputs)
    assert selected.Anatomical_Plane.eq("Sagittal").all()
    assert selected.Anatomical_Plane_Original.eq("Coronal").all()
    assert (
        plan["series"][1]["source_row"]["Anatomical_Plane_Review_ID"] == "synthetic-reviewed-plane"
    )


def test_same_native_metadata_but_different_cached_arrays_are_retained(tmp_path: Path):
    inputs = fixture(tmp_path)
    path = meta_path(inputs, "1.1.2")
    meta = json.loads(path.read_text())
    cache_path = path.with_suffix(".npz")
    with np.load(cache_path) as archive:
        volume = archive["volume"]
    volume[0, 0, 0] ^= 1
    np.savez_compressed(cache_path, volume=volume)
    meta["cache_sha256"] = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    meta["cache_bytes"] = cache_path.stat().st_size
    path.write_text(json.dumps(meta))
    index_path = inputs["audit_path"].parent / "series_cache_index.parquet"
    index = pd.read_parquet(index_path)
    for field in ["cache_sha256", "cache_bytes"]:
        index.loc[index.series_uid.eq("1.1.2"), field] = meta[field]
    index.to_parquet(index_path, index=False)
    audit = json.loads(inputs["audit_path"].read_text())
    audit["series_cache_index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    inputs["audit_path"].write_text(json.dumps(audit))
    selected, plan = build_selection_plan(**inputs)
    assert len(selected) == 3 and plan["n_excluded_series"] == 0
    assert plan["candidate_comparisons"][0]["reason"] == "decoded_cache_arrays_differ"


def test_semantically_wrong_audit_index_cannot_pass_with_a_matching_file_hash(tmp_path: Path):
    inputs = fixture(tmp_path)
    index_path = inputs["audit_path"].parent / "series_cache_index.parquet"
    index = pd.read_parquet(index_path)
    index.loc[0, "study_uid"] = "9.9"
    index.to_parquet(index_path, index=False)
    audit = json.loads(inputs["audit_path"].read_text())
    audit["series_cache_index_sha256"] = hashlib.sha256(index_path.read_bytes()).hexdigest()
    inputs["audit_path"].write_text(json.dumps(audit))
    with pytest.raises(ValueError, match="index membership"):
        build_selection_plan(**inputs)
