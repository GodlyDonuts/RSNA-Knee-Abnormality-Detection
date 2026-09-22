"""Export explicitly reviewed auxiliary series metadata without changing its source.

This module applies exact UID-bound decisions only. It does not infer planes from
IOP, sequence descriptions, images, or clinical target annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path

import pandas as pd

SOURCE_COLUMNS = [
    "StudyInstanceUID",
    "SeriesInstanceUID",
    "Fluid_Sensitive",
    "Fat_Suppression",
    "Anatomical_Plane",
]
PLANES = {"Sagittal", "Coronal", "Axial"}
LINEAGE_COLUMNS = [
    "Anatomical_Plane_Original",
    "Anatomical_Plane_Review_ID",
    "Anatomical_Plane_Review_SHA256",
]


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate decision JSON field: {key}")
        result[key] = value
    return result


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or value != value.strip():
        raise ValueError(f"Invalid {field}")
    return value


def _digest(value: object) -> str:
    result = _text(value, "SHA256")
    if len(result) != 64 or any(c not in "0123456789abcdef" for c in result):
        raise ValueError("Expected a lowercase SHA256 digest")
    return result


def _evidence(value: object, artifact_root: Path, required: bool) -> list[dict]:
    if not isinstance(value, list) or (required and not value):
        raise ValueError("Reviewed decisions require bound evidence")
    references = []
    seen = set()
    root = artifact_root.resolve()
    for reference in value:
        if not isinstance(reference, dict) or set(reference) not in (
            {"path", "sha256"},
            {"path", "sha256", "bytes"},
        ):
            raise ValueError("Evidence references require path, sha256 and optional bytes")
        name = _text(reference["path"], "evidence path")
        digest = _digest(reference["sha256"])
        path = (root / name).resolve()
        if Path(name).is_absolute() or not path.is_relative_to(root):
            raise ValueError("Evidence path must stay within the artifact root")
        if path in seen:
            raise ValueError("Duplicate evidence path")
        seen.add(path)
        if not path.is_file() or _sha256(path.read_bytes()) != digest:
            raise ValueError(f"Missing or changed evidence: {name}")
        if "bytes" in reference and (
            type(reference["bytes"]) is not int or reference["bytes"] != path.stat().st_size
        ):
            raise ValueError(f"Evidence byte size differs: {name}")
        references.append(dict(reference))
    return references


def apply_series_decision(
    source: Path, decision_path: Path, artifact_root: Path
) -> tuple[pd.DataFrame, dict]:
    """Validate the complete decision before modifying an in-memory source copy."""
    source_bytes = source.read_bytes()
    decision_bytes = decision_path.read_bytes()
    source_hash, decision_hash = _sha256(source_bytes), _sha256(decision_bytes)
    decision = json.loads(decision_bytes, object_pairs_hook=_unique_object)
    if not isinstance(decision, dict):
        raise ValueError("Series metadata decision must be an object")
    if type(decision.get("schema_version")) is not int or decision["schema_version"] != 1:
        raise ValueError("Unsupported series metadata decision schema")
    if decision.get("status") != "accepted":
        raise ValueError("Series metadata decision must be explicitly accepted")
    decision_id = _text(decision.get("decision_id"), "decision ID")
    assessor = _text(decision.get("assessor"), "decision assessor")
    if _digest(decision.get("source_series_sha256")) != source_hash:
        raise ValueError("Official series CSV checksum differs from the reviewed source")
    if "source_series_path" in decision:
        _evidence(
            [{"path": decision["source_series_path"], "sha256": source_hash}],
            artifact_root,
            required=True,
        )
    corrections = decision.get("overrides")
    if not isinstance(corrections, list):
        raise ValueError("Corrections must be an explicit list, including an empty list")
    assessors = decision.get("reviewer_ids", [])
    if not isinstance(assessors, list) or (corrections and len(assessors) != 2):
        raise ValueError("Two independent assessor identities are required")
    identities = [_text(identity, "assessor identity") for identity in assessors]
    if len(set(identities)) != len(identities):
        raise ValueError("Self approval cannot authorize auxiliary metadata corrections")
    if corrections and assessor not in identities:
        raise ValueError("Decision assessor must be one of its independent reviewers")
    if not corrections:
        _text(decision.get("decision"), "explicit no-corrections provenance")
    evidence = _evidence(decision.get("evidence", []), artifact_root, required=bool(corrections))

    original = pd.read_csv(io.BytesIO(source_bytes), dtype=str, keep_default_na=False)
    if list(original.columns) != SOURCE_COLUMNS:
        raise ValueError("Expected the five original auxiliary series CSV columns")
    if original.empty or original.isna().any().any():
        raise ValueError("Series source must be nonempty and complete")
    for field in ("StudyInstanceUID", "SeriesInstanceUID"):
        if not original[field].str.fullmatch(r"[0-9]+(?:\.[0-9]+)+").all():
            raise ValueError(f"Invalid source UID: {field}")
        if original[field].str.len().gt(64).any():
            raise ValueError(f"Source UID exceeds 64 characters: {field}")
    if original.SeriesInstanceUID.duplicated().any():
        raise ValueError("Source series UIDs must be unique")
    if not original.Anatomical_Plane.isin(PLANES).all():
        raise ValueError("Unknown source anatomical plane")
    for field in ("Fluid_Sensitive", "Fat_Suppression"):
        if not original[field].isin(["0", "1"]).all():
            raise ValueError(f"Expected binary source field: {field}")

    validated = []
    seen_series = set()
    for correction in corrections:
        if not isinstance(correction, dict) or set(correction) != {
            "study_uid",
            "series_uid",
            "original_plane",
            "reviewed_plane",
        }:
            raise ValueError(
                "Corrections require exact study_uid, series_uid, original_plane and reviewed_plane"
            )
        study_uid = _text(correction["study_uid"], "correction study UID")
        series_uid = _text(correction["series_uid"], "correction series UID")
        if series_uid in seen_series:
            raise ValueError("Duplicate or conflicting correction series")
        seen_series.add(series_uid)
        before, after = correction["original_plane"], correction["reviewed_plane"]
        if not isinstance(before, str) or not isinstance(after, str):
            raise ValueError("Correction planes must be strings")
        if before not in PLANES or after not in PLANES or before == after:
            raise ValueError("Corrections require distinct known before/after planes")
        selected = original[original.SeriesInstanceUID == series_uid]
        if len(selected) != 1 or selected.iloc[0].StudyInstanceUID != study_uid:
            raise ValueError("Correction study/series membership differs from source")
        if selected.iloc[0].Anatomical_Plane != before:
            raise ValueError("Correction before-plane differs from official source")
        validated.append(dict(correction))

    effective = original.copy(deep=True)
    effective[LINEAGE_COLUMNS[0]] = original.Anatomical_Plane
    effective[LINEAGE_COLUMNS[1]] = ""
    effective[LINEAGE_COLUMNS[2]] = ""
    for correction in validated:
        selected_mask = effective.SeriesInstanceUID == correction["series_uid"]
        effective.loc[selected_mask, "Anatomical_Plane"] = correction["reviewed_plane"]
        effective.loc[selected_mask, LINEAGE_COLUMNS[1]] = decision_id
        effective.loc[selected_mask, LINEAGE_COLUMNS[2]] = decision_hash
    trace = {
        "schema_version": 1,
        "status": "validated_export",
        "decision_id": decision_id,
        "assessor": assessor,
        "source_series_path": decision.get("source_series_path"),
        "source_revision": decision.get("source_revision"),
        "source_series_sha256": source_hash,
        "decision_sha256": decision_hash,
        "reviewer_ids": identities,
        "evidence": evidence,
        "n_studies": int(original.StudyInstanceUID.nunique()),
        "n_series": len(original),
        "n_corrections_registered": len(corrections),
        "n_corrections_applied": len(validated),
        "corrections": validated,
        "decision_provenance": decision.get("decision"),
        "source_preserved": True,
        "only_anatomical_plane_modified": True,
        "heuristic_overrides_applied": False,
        "clinical_target_values_read_or_changed": False,
    }
    return effective, trace


def export_series_metadata(source: Path, decision: Path, artifact_root: Path, output: Path) -> dict:
    """Write a new effective CSV and audit; never overwrite an existing output."""
    effective, trace = apply_series_decision(source, decision, artifact_root)
    csv_bytes = effective.to_csv(index=False).encode()
    trace["effective_csv_sha256"] = _sha256(csv_bytes)
    trace["effective_csv_bytes"] = len(csv_bytes)
    output.mkdir(parents=True, exist_ok=False)
    (output / "series_effective.csv").write_bytes(csv_bytes)
    (output / "series_metadata_audit.json").write_text(json.dumps(trace, indent=2) + "\n")
    return trace


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--decision", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    trace = export_series_metadata(args.source, args.decision, args.artifact_root, args.output)
    print(json.dumps(trace, indent=2))


if __name__ == "__main__":
    main()
