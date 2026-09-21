"""Export conservative within-study exact-series selections after a complete audit.

Source images, cache volumes and the supplied effective metadata stay unchanged.
Cross-study image matches remain leakage evidence and never authorize exclusions.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rsna.audit import inventory_fingerprint, validate_inventory
from rsna.labels import validate_dicom_audit
from rsna.schema import CACHE_VERSION
from rsna.series_metadata import LINEAGE_COLUMNS, PLANES, SOURCE_COLUMNS

POLICY: dict = {
    "version": "within-study-exact-series-v1",
    "native_pixels": "complete_ordered_shape_i64le_samples_f64le_before_rescale",
    "constant_slices": "include_every_hash_and_flag_require_one_nonconstant",
    "geometry": "patient_geometry_no_warnings_all_recorded_positions",
    "orientation_atol": 1e-3,
    "position_atol_mm": 1e-3,
    "pixel_spacing_atol_mm": 1e-6,
    "geometry_rtol": 0,
    "cache": "exact_uint16_volume_array_equality",
    "features": ["Anatomical_Plane", "Fluid_Sensitive", "Fat_Suppression"],
    "representative": "lexicographically_smallest_series_uid",
    "groups": "pairwise_compatible_no_tolerance_chaining",
    "cross_study": "never_exclude_keep_complete_leakage_graph",
}


def canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": _file_sha(path), "bytes": path.stat().st_size}


def _verify_binding(reference: dict) -> None:
    path = Path(reference["path"])
    if (
        not path.is_file()
        or path.stat().st_size != reference["bytes"]
        or _file_sha(path) != reference["sha256"]
    ):
        raise ValueError(f"Missing or changed bound input: {path}")


def _series_source(path: Path, inventory: pd.DataFrame, raw_split: str) -> pd.DataFrame:
    source = pd.read_csv(io.BytesIO(path.read_bytes()), dtype=str, keep_default_na=False)
    if list(source.columns) not in (SOURCE_COLUMNS, SOURCE_COLUMNS + LINEAGE_COLUMNS):
        raise ValueError("Expected original or reviewed effective series CSV, before selection")
    if source.empty or source.SeriesInstanceUID.duplicated().any():
        raise ValueError("Series table must be nonempty with unique series UIDs")
    if not source.Anatomical_Plane.isin(PLANES).all():
        raise ValueError("Invalid effective plane")
    for field in ("Fluid_Sensitive", "Fat_Suppression"):
        if not source[field].isin(["0", "1"]).all():
            raise ValueError("Invalid model-effective binary field")
    expected = set(
        inventory.loc[inventory.split.eq(raw_split), ["study_uid", "series_uid"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    actual = set(
        source[["StudyInstanceUID", "SeriesInstanceUID"]].itertuples(index=False, name=None)
    )
    if actual != expected:
        raise ValueError("Effective series table must exactly cover the chosen inventory split")
    return source


def _validate_meta(meta: dict, frame: pd.DataFrame, indexed: dict, cache: Path) -> Path:
    split, study, series = frame.iloc[0][["split", "study_uid", "series_uid"]]
    path = cache / split / study / f"{series}.npz"
    if (meta["split"], meta["study_uid"], meta["series_uid"]) != (split, study, series):
        raise ValueError("Cache metadata study/series membership differs")
    if meta["cache_version"] != CACHE_VERSION:
        raise ValueError("Cache metadata version differs")
    if (
        meta["cache_path"] != str(path.relative_to(cache))
        or meta["n_slices"] != len(frame)
        or meta["path_fingerprint"] != _sha("\n".join(sorted(frame.path)).encode())
    ):
        raise ValueError("Cache metadata path or slice coverage differs")
    if len(set(meta["sop_uids"])) != len(frame) or len(meta["sop_uids"]) != len(frame):
        raise ValueError("Cache metadata has missing or duplicate slices")
    if len(meta["slice_sha256"]) != len(frame) or dict(
        zip(meta["sop_uids"], meta["slice_sha256"])
    ) != dict(zip(frame.sop_uid, frame.sha256)):
        raise ValueError("Cache source hashes differ from pinned inventory")
    if (
        meta["pixel_hash_definition"] != "shape_i64le_and_decoded_samples_f64le_before_rescale"
        or len(meta["pixel_sha256"]) != len(frame)
        or any(
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
            for digest in meta["pixel_sha256"]
        )
        or len(meta["pixel_nonconstant"]) != len(frame)
        or any(type(flag) is not bool for flag in meta["pixel_nonconstant"])
        or len(meta["positions"]) != len(frame)
    ):
        raise ValueError("Cache metadata has incomplete native fingerprints or geometry")
    shape = meta["original_shape"]
    if (
        len(shape) != 3
        or shape[0] != len(frame)
        or any(type(n) is not int or n <= 0 for n in shape)
    ):
        raise ValueError("Invalid native shape binding")
    for field in ("cache_path", "n_slices", "side", "cache_sha256", "cache_bytes"):
        if meta[field] != indexed[field]:
            raise ValueError(f"Cache metadata differs from complete audit index: {field}")
    if not path.is_file() or path.stat().st_size != meta["cache_bytes"]:
        raise ValueError("Missing or resized audited cache")
    return path


def eligibility_reason(meta: dict) -> str | None:
    """Conservative geometry eligibility; absent/ambiguous geometry never excludes."""
    if not any(meta["pixel_nonconstant"]):
        return "no_nonconstant_native_slice"
    if meta["ordering"] != "patient_geometry" or meta["warnings"]:
        return "cache_geometry_or_other_warning_requires_review"
    orientation = np.asarray(meta["orientation"], dtype=float)
    spacing = np.asarray(meta["pixel_spacing"], dtype=float)
    positions = np.asarray(meta["positions"], dtype=float)
    if (
        orientation.shape != (6,)
        or spacing.shape != (2,)
        or positions.shape != (meta["n_slices"], 3)
        or not all(np.isfinite(a).all() for a in [orientation, spacing, positions])
        or (spacing <= 0).any()
    ):
        return "missing_or_invalid_recorded_geometry"
    row, column = orientation.reshape(2, 3)
    if not (
        np.isclose(np.linalg.norm(row), 1, atol=1e-3, rtol=0)
        and np.isclose(np.linalg.norm(column), 1, atol=1e-3, rtol=0)
        and np.isclose(np.dot(row, column), 0, atol=1e-3, rtol=0)
    ):
        return "invalid_orientation_vectors"
    if (np.diff(positions @ np.cross(row, column)) < 1e-4).any():
        return "nonincreasing_recorded_slice_positions"
    return None


def equivalence_reason(left: dict, right: dict, left_row: dict, right_row: dict) -> str | None:
    """Return why a pair cannot be excluded; None still requires exact cache comparison."""
    if left["study_uid"] != right["study_uid"] or left["split"] != right["split"]:
        return "cross_study_or_split_never_exclude"
    for meta in [left, right]:
        reason = eligibility_reason(meta)
        if reason:
            return reason
    for field in ["pixel_sha256", "pixel_nonconstant", "original_shape", "n_slices"]:
        if left[field] != right[field]:
            return "complete_ordered_native_pixels_or_shape_differ"
    for field in POLICY["features"]:
        if str(left_row[field]) != str(right_row[field]):
            return "model_effective_features_differ"
    for field in ["side", "resized_shape", "photometric_interpretation"]:
        if left[field] != right[field]:
            return "cache_shape_or_photometric_interpretation_differ"
    for field, tolerance in [("orientation", 1e-3), ("positions", 1e-3), ("pixel_spacing", 1e-6)]:
        if not np.allclose(left[field], right[field], atol=tolerance, rtol=0):
            return "recorded_geometry_differs"
    return None


def _volume(path: Path, meta: dict) -> np.ndarray:
    data = path.read_bytes()
    if len(data) != meta["cache_bytes"] or _sha(data) != meta["cache_sha256"]:
        raise ValueError("Changed candidate cache pixels/content")
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        volume = archive["volume"]
    if volume.dtype != np.uint16 or volume.shape != (meta["n_slices"], meta["side"], meta["side"]):
        raise ValueError("Candidate cache volume shape/type differs")
    return volume


def build_selection_plan(
    series_path: Path,
    inventory_path: Path,
    audit_path: Path,
    cache: Path,
    source_revision: str,
    raw_split: str = "train_series",
) -> tuple[pd.DataFrame, dict]:
    """Build a deterministic selection and bind inputs; no output or source mutation."""
    if raw_split not in {"train_series", "test_series"}:
        raise ValueError("Unknown raw split")
    index_path = audit_path.parent / "series_cache_index.parquet"
    input_bindings = {
        "source_series": _binding(series_path),
        "dicom_audit": _binding(audit_path),
        "series_cache_index": _binding(index_path),
    }
    inventory = pd.read_parquet(inventory_path)
    validate_inventory(inventory)
    train_uids = set(inventory.loc[inventory.split.eq("train_series"), "study_uid"])
    pairs = validate_dicom_audit(audit_path, inventory, train_uids, source_revision)
    index = pd.read_parquet(index_path)
    keys = ["raw_split", "study_uid", "series_uid"]
    if index.duplicated(keys).any() or set(index[keys].itertuples(index=False, name=None)) != set(
        inventory[["split", "study_uid", "series_uid"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    ):
        raise ValueError("Complete audit index membership differs from inventory")
    indexed = {tuple(row[k] for k in keys): row for row in index.to_dict("records")}
    original = _series_source(series_path, inventory, raw_split)
    rows = original.set_index("SeriesInstanceUID", drop=False).to_dict("index")
    metas: dict[str, dict] = {}
    paths: dict[str, Path] = {}
    references: dict[str, dict] = {}
    buckets: dict[tuple[str, str], list[str]] = {}
    reasons: dict[str, str | None] = {}
    for key, frame in inventory.loc[inventory.split.eq(raw_split)].groupby(
        ["split", "study_uid", "series_uid"]
    ):
        split, study, series = key
        meta_path = cache / split / study / f"{series}.json"
        meta_bytes = meta_path.read_bytes()
        meta = json.loads(meta_bytes)
        if meta["source_revision"] != source_revision:
            raise ValueError("Cache source revision differs")
        paths[series] = _validate_meta(meta, frame, indexed[key], cache)
        metas[series] = meta
        references[series] = {
            "path": str(meta_path.resolve()),
            "sha256": _sha(meta_bytes),
            "bytes": len(meta_bytes),
        }
        reasons[series] = eligibility_reason(meta)
        signature = _sha(
            canonical_bytes(
                [meta["pixel_sha256"], meta["pixel_nonconstant"], meta["original_shape"]]
            )
        )
        buckets.setdefault((study, signature), []).append(series)
    groups, comparisons = [], []
    for (study, signature), members in sorted(buckets.items()):
        partitions: list[list[str]] = []
        for series in sorted(members):
            for group in partitions:
                compatible = True
                for other in group:
                    reason = equivalence_reason(
                        metas[series], metas[other], rows[series], rows[other]
                    )
                    if reason is None and not np.array_equal(
                        _volume(paths[series], metas[series]), _volume(paths[other], metas[other])
                    ):
                        reason = "decoded_cache_arrays_differ"
                    comparisons.append(
                        {
                            "series_uids": sorted([series, other]),
                            "compatible": reason is None,
                            "reason": reason or "native_geometry_features_and_cache_exact",
                        }
                    )
                    if reason is not None:
                        compatible = False
                        break
                if compatible:
                    group.append(series)
                    break
            else:
                partitions.append([series])
        for group in partitions:
            group_id = _sha(canonical_bytes([POLICY, study, group]))
            groups.append(
                {
                    "group_id": group_id,
                    "study_uid": study,
                    "representative_series_uid": group[0],
                    "series_uids": group,
                    "native_sequence_signature_sha256": signature,
                }
            )
    membership = {uid: group for group in groups for uid in group["series_uids"]}
    records = []
    for row in original.to_dict("records"):
        uid = row["SeriesInstanceUID"]
        matched_group = membership[uid]
        representative = matched_group["representative_series_uid"]
        records.append(
            {
                "source_row": row,
                "series_uid": uid,
                "study_uid": row["StudyInstanceUID"],
                "group_id": matched_group["group_id"],
                "representative_series_uid": representative,
                "retained": uid == representative,
                "reason": "exact_equivalent_nonrepresentative"
                if uid != representative
                else reasons[uid]
                or (
                    "exact_equivalent_representative"
                    if len(matched_group["series_uids"]) > 1
                    else "no_compatible_exact_equivalent"
                ),
                "cache_metadata": references[uid],
                "cache_volume": {
                    "path": str(paths[uid].resolve()),
                    "sha256": metas[uid]["cache_sha256"],
                    "bytes": metas[uid]["cache_bytes"],
                },
            }
        )
    retained = {record["series_uid"] for record in records if record["retained"]}
    effective = original.loc[original.SeriesInstanceUID.isin(retained)].copy()
    if set(effective.StudyInstanceUID) != set(original.StudyInstanceUID):
        raise ValueError("Selection removed a study")
    policy_sha = _sha(canonical_bytes(POLICY))
    effective["Series_Equivalence_Group"] = effective.SeriesInstanceUID.map(
        lambda uid: membership[uid]["group_id"]
    )
    effective["Series_Equivalence_Policy_SHA256"] = policy_sha
    plan = {
        "schema_version": 1,
        "status": "validated_mechanical_selection_plan",
        "policy": POLICY,
        "policy_sha256": policy_sha,
        "source_revision": source_revision,
        "cache_version": CACHE_VERSION,
        "raw_split": raw_split,
        **input_bindings,
        "inventory_path": str(inventory_path.resolve()),
        "inventory_fingerprint_sha256": inventory_fingerprint(inventory),
        "n_input_studies": int(original.StudyInstanceUID.nunique()),
        "n_input_series": len(original),
        "n_retained_series": len(effective),
        "n_excluded_series": len(original) - len(effective),
        "n_equivalence_groups": sum(len(group["series_uids"]) > 1 for group in groups),
        "groups": groups,
        "series": records,
        "candidate_comparisons": comparisons,
        "cross_study_training_edges_preserved": [list(pair) for pair in pairs],
        "source_preserved": True,
        "clinical_targets_accessed": False,
        "geometry_scope": "All ordered IPPs; first stored IOP and common pixel spacing. Per-slice IOP is not retained by the cache. SOP/FrameOfReference IDs need not match.",
        "native_pixel_scope": "Prior decoder native hashes and original_shape, bound through source SOP/SHA inventory; no source DICOM redecoding here.",
    }
    for reference in input_bindings.values():
        _verify_binding(reference)
    return effective, plan


def _recheck_plan_inputs(plan: dict) -> None:
    # Recheck small inputs before and after reading all relevant cache content.
    references = [plan[k] for k in ["source_series", "dicom_audit", "series_cache_index"]]
    metadata = [record["cache_metadata"] for record in plan["series"]]
    for reference in references + metadata:
        _verify_binding(reference)
    for record in plan["series"]:
        _verify_binding(record["cache_volume"])
    for reference in references + metadata:
        _verify_binding(reference)
    current = pd.read_parquet(plan["inventory_path"])
    if inventory_fingerprint(current) != plan["inventory_fingerprint_sha256"]:
        raise ValueError("Changed source inventory after selection planning")


def export_series_selection(
    series_path: Path,
    inventory_path: Path,
    audit_path: Path,
    cache: Path,
    source_revision: str,
    output: Path,
    raw_split: str = "train_series",
) -> dict:
    if output.exists():
        raise ValueError("Selection output must be a new directory")
    effective, plan = build_selection_plan(
        series_path, inventory_path, audit_path, cache, source_revision, raw_split
    )
    plan_bytes = canonical_bytes(plan) + b"\n"
    _recheck_plan_inputs(plan)
    csv_bytes = effective.to_csv(index=False).encode()
    summary = {
        "schema_version": 1,
        "status": "validated_selection_export",
        "plan_sha256": _sha(plan_bytes),
        "plan_bytes": len(plan_bytes),
        "policy_sha256": plan["policy_sha256"],
        "source_series_sha256": plan["source_series"]["sha256"],
        "dicom_audit_sha256": plan["dicom_audit"]["sha256"],
        "inventory_fingerprint_sha256": plan["inventory_fingerprint_sha256"],
        "source_revision": source_revision,
        "cache_version": CACHE_VERSION,
        "n_input_studies": plan["n_input_studies"],
        "n_input_series": plan["n_input_series"],
        "n_retained_series": plan["n_retained_series"],
        "n_excluded_series": plan["n_excluded_series"],
        "n_equivalence_groups": plan["n_equivalence_groups"],
        "selected_csv_sha256": _sha(csv_bytes),
        "selected_csv_bytes": len(csv_bytes),
        "all_input_cache_bytes_sha256_reverified": True,
        "full_source_audit_precedes_selection": True,
        "cross_study_sources_or_leakage_edges_removed": False,
        "source_files_deleted_or_modified": False,
        "clinical_target_values_read_or_changed": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    (output / "series_selected.csv").write_bytes(csv_bytes)
    (output / "series_selection_plan.json").write_bytes(plan_bytes)
    (output / "series_selection_audit.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--dicom-audit", type=Path, required=True)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument(
        "--raw-split", choices=["train_series", "test_series"], default="train_series"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            export_series_selection(
                args.series,
                args.inventory,
                args.dicom_audit,
                args.cache,
                args.source_revision,
                args.output,
                args.raw_split,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
