from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rsna.archive import import_artifacts, review_index
from rsna.labels import (
    file_sha256,
    read_reviews,
    source_reports_fingerprint,
    validate_dicom_audit,
    validate_review_provenance,
)
from rsna.schema import CACHE_VERSION


def write_annotation(path: Path, uid: str = "study-a") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "study_uid": uid,
        "states": ["negative"] + ["not_mentioned"] * 11,
        "evidence": ["ACL intact."] + [""] * 11,
        "notes": "Explicit source extraction.",
    }
    path.write_text(json.dumps(row) + "\n")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("evidence", "ACL intact."),
        ("states", {str(index): "negative" for index in range(12)}),
        ("notes", ["not a string"]),
        ("unexpected", "gold column or other unapproved field"),
    ],
)
def test_review_records_require_exact_typed_schema(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "batch-000-reviewed.jsonl"
    write_annotation(path)
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))
    with pytest.raises(ValueError):
        read_reviews(tmp_path, {"study-a": "ACL intact."}, "-reviewed")


def review_fixture(root: Path) -> tuple[Path, dict, pd.DataFrame, pd.DataFrame, Path]:
    reports = {"study-a": "ACL intact."}
    rubric = root / "rubric.md"
    rubric.write_text("An explicit versioned competition rubric.")
    provenance = {"batch-000": {}}
    for phase, folder in (("draft", "drafts"), ("reviewed", "reviewed")):
        path = root / folder / f"batch-000-{phase}.jsonl"
        write_annotation(path)
        provenance["batch-000"][phase] = {
            "session_id": f"session-{phase}",
            "agent_id": f"agent-{phase}",
            "workflow_run_id": "run-fixture",
            "n_studies": 1,
            "notes": "Read the report in a separate pass.",
            "sha256": file_sha256(path),
            "rubric_sha256": file_sha256(rubric),
            "source_reports_sha256": source_reports_fingerprint(reports),
        }
    provenance_path = root / "provenance.json"
    provenance["batch-000"]["reviewed"]["parent_artifact_sha256"] = provenance["batch-000"][
        "draft"
    ]["sha256"]
    provenance_path.write_text(json.dumps(provenance))
    drafts = read_reviews(root / "drafts", reports, "-draft")
    reviews = read_reviews(root / "reviewed", reports, "-reviewed")
    return provenance_path, reports, drafts, reviews, rubric


def validate_fixture(root: Path, fixture: tuple) -> dict:
    provenance, reports, drafts, reviews, rubric = fixture
    return validate_review_provenance(
        provenance, reports, drafts, reviews, root / "drafts", root / "reviewed", rubric
    )


def test_complete_source_bound_independent_provenance_is_accepted(tmp_path: Path) -> None:
    fixture = review_fixture(tmp_path)
    assert validate_fixture(tmp_path, fixture)["independent_review_batches"] == 1


@pytest.mark.parametrize("field", ["agent_id", "session_id"])
def test_copied_review_identity_cannot_release(tmp_path: Path, field: str) -> None:
    fixture = review_fixture(tmp_path)
    path = fixture[0]
    provenance = json.loads(path.read_text())
    provenance["batch-000"]["reviewed"][field] = provenance["batch-000"]["draft"][field]
    path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="separate agent and session"):
        validate_fixture(tmp_path, fixture)


@pytest.mark.parametrize("field", ["rubric_sha256", "source_reports_sha256", "sha256"])
def test_unbound_or_changed_review_inputs_cannot_release(tmp_path: Path, field: str) -> None:
    fixture = review_fixture(tmp_path)
    path = fixture[0]
    provenance = json.loads(path.read_text())
    del provenance["batch-000"]["reviewed"][field]
    path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError):
        validate_fixture(tmp_path, fixture)


def test_wrong_batch_membership_cannot_release(tmp_path: Path) -> None:
    fixture = review_fixture(tmp_path)
    fixture[3]["artifact"] = "batch-001-reviewed.jsonl"
    with pytest.raises(ValueError, match="batch membership"):
        validate_fixture(tmp_path, fixture)


@pytest.mark.parametrize("parent_hash", [None, "wrong-draft-hash"])
def test_review_must_bind_the_exact_draft(tmp_path: Path, parent_hash: str | None) -> None:
    fixture = review_fixture(tmp_path)
    path = fixture[0]
    provenance = json.loads(path.read_text())
    provenance["batch-000"]["reviewed"]["parent_artifact_sha256"] = parent_hash
    path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="not bound to its exact draft"):
        validate_fixture(tmp_path, fixture)


def test_report_fingerprint_is_stable_and_content_sensitive() -> None:
    first = {"study-a": "ACL intact.", "study-b": "膝关节。"}
    assert source_reports_fingerprint(first) == source_reports_fingerprint(
        dict(reversed(first.items()))
    )
    assert source_reports_fingerprint(first) != source_reports_fingerprint(
        {**first, "study-a": "ACL torn."}
    )


def dicom_audit_fixture(root: Path) -> tuple[Path, pd.DataFrame]:
    from rsna.audit import inventory_fingerprint

    inventory = pd.DataFrame(
        [
            {
                "split": split,
                "study_uid": uid,
                "series_uid": f"{uid}-series",
                "sop_uid": f"{uid}-sop",
                "path": f"{split}/{uid}/{uid}-series/{uid}-sop.dcm",
                "sha256": "f" * 64,
                "bytes": 1,
            }
            for split, uid in (
                ("train_series", "study-a"),
                ("test_series", "test-a"),
                ("test_series", "test-b"),
            )
        ]
    )
    index = root / "series_cache_index.parquet"
    inventory.to_parquet(index, index=False)
    audit = {
        "audit_schema_version": 2,
        "complete": True,
        "source_revision": "a" * 40,
        "cache_version": CACHE_VERSION,
        "inventory_fingerprint_sha256": inventory_fingerprint(inventory),
        "series_cache_index_sha256": file_sha256(index),
        "studies": 3,
        "series": 3,
        "decoded_slices": 3,
        "source_bytes": 3,
        "study_uids_by_split": {"train_series": ["study-a"], "test_series": ["test-a", "test-b"]},
        "cross_study_pixel_duplicate_pairs": [],
        "train_test_duplicate_pairs": [],
    }
    path = root / "dicom_audit.json"
    path.write_text(json.dumps(audit))
    return path, inventory


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("complete", False),
        ("audit_schema_version", 1),
        ("source_revision", "b" * 40),
        ("cache_version", "obsolete-cache"),
        ("inventory_fingerprint_sha256", "bad"),
        ("decoded_slices", 2),
        ("series_cache_index_sha256", "bad"),
        ("study_uids_by_split", {"train": ["different-study"]}),
        ("cross_study_pixel_duplicate_pairs", [["study-a", "test-a"]]),
        ("cross_study_pixel_duplicate_pairs", [["unknown", "test-a"]]),
        ("train_test_duplicate_pairs", [["study-a", "test-a"]]),
    ],
)
def test_partial_stale_or_leaky_dicom_audit_blocks_release(
    tmp_path: Path, field: str, value: object
) -> None:
    path, inventory = dicom_audit_fixture(tmp_path)
    audit = json.loads(path.read_text())
    audit[field] = value
    path.write_text(json.dumps(audit))
    with pytest.raises(ValueError):
        validate_dicom_audit(path, inventory, {"study-a"}, "a" * 40)


def test_test_only_duplicate_is_not_injected_into_training_groups(tmp_path: Path) -> None:
    path, inventory = dicom_audit_fixture(tmp_path)
    audit = json.loads(path.read_text())
    audit["cross_study_pixel_duplicate_pairs"] = [["test-a", "test-b"]]
    path.write_text(json.dumps(audit))
    assert validate_dicom_audit(path, inventory, {"study-a"}, "a" * 40) == []


def test_archive_rejects_same_agent_second_pass(tmp_path: Path) -> None:
    (tmp_path / "source").mkdir()
    (tmp_path / "reviews").mkdir()
    pd.DataFrame({"StudyInstanceUID": ["study-a"], "Report": ["ACL intact."]}).to_csv(
        tmp_path / "source/train.csv", index=False
    )
    (tmp_path / "reviews/provenance.json").write_text("{}")
    review_index(tmp_path, create_inputs=True)
    for phase in ("draft", "reviewed"):
        path = tmp_path / f"input-{phase}.jsonl"
        write_annotation(path)
        import_artifacts(
            tmp_path,
            [
                {
                    "batch": "batch-000",
                    "phase": phase,
                    "path": str(path),
                    "session_id": "devin-" + "a" * 32,
                    "agent_id": "wfar-" + "b" * 32,
                    "workflow_run_id": "wfr-" + "c" * 32,
                    "n_studies": 1,
                    "notes": "Self-reviewed fixture.",
                }
            ],
        )
    index = review_index(tmp_path)
    assert index["counts"]["reviewed"]["invalid"] == 1
    assert index["complete"] is False
