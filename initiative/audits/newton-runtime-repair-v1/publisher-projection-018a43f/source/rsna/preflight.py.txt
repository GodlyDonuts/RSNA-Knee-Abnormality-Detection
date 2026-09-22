"""Recheck original source tables and inventory without deriving training labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from rsna.schema import TARGETS

FILES = ("train.csv", "train_series.csv", "test.csv", "test_series.csv", "sample_submission.csv")


def normalized_report(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", value).casefold().split())


def preflight(source: Path, inventory_path: Path, current: Path | None, output: Path) -> dict:
    tables = {name: pd.read_csv(source / name) for name in FILES}
    train, test = tables["train.csv"], tables["test.csv"]
    inventory = pd.read_parquet(inventory_path)
    issues: list[dict] = []
    checks: dict[str, bool] = {}

    def check(name: str, okay: bool, detail: str = "") -> None:
        checks[name] = bool(okay)
        if not okay:
            issues.append({"check": name, "detail": detail})

    check("training_columns", list(train.columns) == ["StudyInstanceUID", "Report", *TARGETS])
    check("test_has_no_reports", list(test.columns) == ["StudyInstanceUID"])
    check(
        "submission_columns",
        list(tables["sample_submission.csv"].columns) == ["StudyInstanceUID", *TARGETS],
    )
    for split, studies in (("train", train), ("test", test)):
        ids = studies.StudyInstanceUID
        series = tables[f"{split}_series.csv"]
        check(f"{split}_unique_nonempty_studies", ids.notna().all() and not ids.duplicated().any())
        check(
            f"{split}_series_schema",
            list(series.columns)
            == [
                "StudyInstanceUID",
                "SeriesInstanceUID",
                "Fluid_Sensitive",
                "Fat_Suppression",
                "Anatomical_Plane",
            ],
        )
        check(f"{split}_series_identity", not series.SeriesInstanceUID.duplicated().any())
        check(f"{split}_series_coverage", set(ids) == set(series.StudyInstanceUID))
        check(f"{split}_series_no_missing", not series.isna().any().any())
        for field in ("Fluid_Sensitive", "Fat_Suppression"):
            check(f"{split}_{field}_binary", series[field].isin([0, 1]).all())
        check(
            f"{split}_known_planes",
            series.Anatomical_Plane.isin(["Sagittal", "Coronal", "Axial"]).all(),
        )
        observed = inventory[inventory.split == f"{split}_series"]
        expected_pairs = set(
            series[["StudyInstanceUID", "SeriesInstanceUID"]].itertuples(index=False, name=None)
        )
        observed_pairs = set(
            observed[["study_uid", "series_uid"]].itertuples(index=False, name=None)
        )
        check(f"{split}_inventory_series_coverage", observed_pairs == expected_pairs)
    check("train_test_uid_disjoint", not set(train.StudyInstanceUID) & set(test.StudyInstanceUID))
    check(
        "template_test_order",
        tables["sample_submission.csv"].StudyInstanceUID.equals(test.StudyInstanceUID),
    )
    check("nonempty_reports", train.Report.notna().all() and train.Report.str.strip().ne("").all())
    # Inspect label presence/domain, never use gold values to choose an extraction rubric.
    check("official_label_domain", bool(train[list(TARGETS)].stack().isin([0, 1]).all()))
    known = train[list(TARGETS)].notna().sum(axis=1)
    check("official_label_rows_all_or_none", known.isin([0, len(TARGETS)]).all())
    check("inventory_unique_paths", not inventory.path.duplicated().any())
    check("inventory_unique_sops", not inventory.sop_uid.duplicated().any())
    check("inventory_known_splits", inventory.split.isin(["train_series", "test_series"]).all())
    check("inventory_positive_sizes", inventory.bytes.gt(0).all())
    check("inventory_valid_sha256", inventory.sha256.str.fullmatch(r"[a-f0-9]{64}").all())
    check("inventory_storage", inventory.storage.isin(["git", "lfs"]).all())
    expected_paths = (
        inventory.split
        + "/"
        + inventory.study_uid
        + "/"
        + inventory.series_uid
        + "/"
        + inventory.sop_uid
        + ".dcm"
    )
    check("inventory_path_identity", inventory.path.equals(expected_paths.rename("path")))

    file_records = []
    for name in FILES:
        original = (source / name).read_bytes()
        record = {
            "file": name,
            "source_sha256": hashlib.sha256(original).hexdigest(),
            "bytes": len(original),
        }
        if current is not None:
            fresh = (current / name).read_bytes()
            record["current_sha256"] = hashlib.sha256(fresh).hexdigest()
            record["byte_equal"] = original == fresh
            check(f"current_{name}_matches", original == fresh)
        file_records.append(record)

    counts = tables["train_series.csv"].groupby("StudyInstanceUID").size()
    manifest = train[["StudyInstanceUID"]].copy()
    manifest["official_evaluation_only"] = known.eq(len(TARGETS))
    manifest["report_sha256"] = train.Report.map(
        lambda v: hashlib.sha256(str(v).encode()).hexdigest()
    )
    manifest["normalized_report_sha256"] = train.Report.map(
        lambda v: hashlib.sha256(normalized_report(str(v)).encode()).hexdigest()
    )
    manifest["series_count"] = manifest.StudyInstanceUID.map(counts)
    manifest["status"] = "source_only_not_training_labels"
    groups = manifest.groupby("normalized_report_sha256").filter(lambda group: len(group) > 1)
    byte_groups = inventory[inventory.sha256.duplicated(keep=False)][
        ["study_uid", "series_uid", "sha256"]
    ].drop_duplicates()
    cross_study_groups = byte_groups.groupby("sha256").study_uid.nunique()
    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Source metadata and inventory consistency; not decoded pixels or report-label correctness",
        "source_metadata_valid": all(checks.values()),
        "training_ready": False,
        "required_remaining_gates": [
            "complete_independent_reviews_under_current_rubric",
            "full_current_version_image_audit",
            "leakage_safe_label_release",
            "real_data_training_resume_and_inference_smoke",
        ],
        "checks": checks,
        "files": file_records,
        "training_studies": len(train),
        "official_evaluation_studies": int(known.eq(len(TARGETS)).sum()),
        "sample_test_studies": len(test),
        "series": int(inventory.series_uid.nunique()),
        "dicom_records": len(inventory),
        "source_bytes": int(inventory.bytes.sum()),
        "duplicate_report_groups": int(groups.normalized_report_sha256.nunique()),
        "studies_in_duplicate_report_groups": len(groups),
        "cross_study_identical_file_hashes": int(cross_study_groups.gt(1).sum()),
        "patient_disjointness": "Not established by deidentified study UIDs; decoded patient metadata not audited here",
    }
    output.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output / "source_study_manifest.parquet", index=False)
    groups.to_parquet(output / "duplicate_report_groups.parquet", index=False)
    pd.DataFrame(issues, columns=["check", "detail"]).to_csv(
        output / "source_issues.csv", index=False
    )
    (output / "source_preflight.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--current-metadata", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = preflight(args.source, args.inventory, args.current_metadata, args.output)
    print(json.dumps(result, indent=2))
    if not result["source_metadata_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
