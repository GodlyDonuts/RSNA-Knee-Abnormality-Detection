import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rsna.data import StudyDataset
from rsna.schema import CACHE_VERSION
from rsna.series_metadata import (
    LINEAGE_COLUMNS,
    SOURCE_COLUMNS,
    apply_series_decision,
    export_series_metadata,
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source = tmp_path / "train_series.csv"
    pd.DataFrame(
        [
            ["1.2.3", "1.2.3.1", 0, 0, "Coronal"],
            ["1.2.3", "1.2.3.2", 1, 1, "Sagittal"],
            ["1.2.4", "1.2.4.1", 1, 0, "Axial"],
        ],
        columns=SOURCE_COLUMNS,
    ).to_csv(source, index=False)
    evidence = tmp_path / "independent-review.json"
    evidence.write_text(json.dumps({"reviewer_ids": ["reader", "reviewer"]}))
    decision = tmp_path / "decision.json"
    decision.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "accepted",
                "decision_id": "explicit-native-plane-review",
                "assessor": "reader",
                "source_series_path": source.name,
                "source_series_sha256": sha(source),
                "reviewer_ids": ["reader", "reviewer"],
                "evidence": [{"path": evidence.name, "sha256": sha(evidence)}],
                "overrides": [
                    {
                        "study_uid": "1.2.3",
                        "series_uid": "1.2.3.1",
                        "original_plane": "Coronal",
                        "reviewed_plane": "Sagittal",
                    },
                    {
                        "study_uid": "1.2.3",
                        "series_uid": "1.2.3.2",
                        "original_plane": "Sagittal",
                        "reviewed_plane": "Coronal",
                    },
                ],
            }
        )
    )
    return source, decision, tmp_path


def change_decision(path: Path, change) -> None:
    data = json.loads(path.read_text())
    change(data)
    path.write_text(json.dumps(data))


def test_exact_export_preserves_source_and_every_nonplane_value(fixture) -> None:
    source, decision, root = fixture
    source_bytes = source.read_bytes()
    result = export_series_metadata(source, decision, root, root / "export")
    effective_path = root / "export/series_effective.csv"
    effective = pd.read_csv(effective_path, keep_default_na=False)
    original = pd.read_csv(source)
    assert source.read_bytes() == source_bytes
    assert list(effective.columns) == SOURCE_COLUMNS + LINEAGE_COLUMNS
    assert effective.Anatomical_Plane.tolist() == ["Sagittal", "Coronal", "Axial"]
    assert effective.Anatomical_Plane_Original.tolist() == original.Anatomical_Plane.tolist()
    for column in SOURCE_COLUMNS[:-1]:
        pd.testing.assert_series_equal(effective[column], original[column])
    assert effective.Anatomical_Plane_Review_ID.tolist() == [
        "explicit-native-plane-review",
        "explicit-native-plane-review",
        "",
    ]
    assert effective.Anatomical_Plane_Review_SHA256.tolist() == [sha(decision), sha(decision), ""]
    assert result["effective_csv_sha256"] == sha(effective_path)
    assert result["n_corrections_registered"] == result["n_corrections_applied"] == 2
    assert json.loads((root / "export/series_metadata_audit.json").read_text()) == result


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("status", "pending", "explicitly accepted"),
        ("source_series_sha256", "0" * 64, "checksum differs"),
        ("reviewer_ids", ["reader", "reader"], "Self approval"),
        ("reviewer_ids", ["reader"], "Two independent"),
        ("overrides", None, "explicit list"),
        ("evidence", [], "bound evidence"),
        ("schema_version", True, "schema"),
    ],
)
def test_invalid_decision_blocks_before_output(fixture, field, value, match) -> None:
    source, decision, root = fixture
    change_decision(decision, lambda data: data.update({field: value}))
    with pytest.raises(ValueError, match=match):
        export_series_metadata(source, decision, root, root / "export")
    assert not (root / "export").exists()


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("study_uid", "1.2.4", "membership"),
        ("series_uid", "1.9.9", "membership"),
        ("original_plane", "Axial", "before-plane"),
        ("reviewed_plane", "Oblique", "distinct known"),
        ("reviewed_plane", "Coronal", "distinct known"),
        ("reviewed_plane", [], "strings"),
    ],
)
def test_wrong_or_stale_correction_rejected(fixture, field, value, match) -> None:
    source, decision, root = fixture
    change_decision(decision, lambda data: data["overrides"][0].update({field: value}))
    with pytest.raises(ValueError, match=match):
        apply_series_decision(source, decision, root)


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_or_conflicting_corrections_rejected(fixture, conflicting) -> None:
    source, decision, root = fixture

    def duplicate(data):
        row = dict(data["overrides"][0])
        if conflicting:
            row["reviewed_plane"] = "Axial"
        data["overrides"].append(row)

    change_decision(decision, duplicate)
    with pytest.raises(ValueError, match="Duplicate or conflicting"):
        apply_series_decision(source, decision, root)


def test_evidence_hash_and_path_escape_rejected(fixture) -> None:
    source, decision, root = fixture
    (root / "independent-review.json").write_text("changed")
    with pytest.raises(ValueError, match="changed evidence"):
        apply_series_decision(source, decision, root)
    change_decision(decision, lambda data: data["evidence"][0].update({"path": "../outside"}))
    with pytest.raises(ValueError, match="within the artifact root"):
        apply_series_decision(source, decision, root)


def test_duplicate_source_series_is_not_deduplicated(fixture) -> None:
    source, decision, root = fixture
    original = pd.read_csv(source)
    pd.concat([original, original.iloc[:1]]).to_csv(source, index=False)
    change_decision(decision, lambda data: data.update({"source_series_sha256": sha(source)}))
    with pytest.raises(ValueError, match="must be unique"):
        apply_series_decision(source, decision, root)


def test_explicit_empty_decision_preserves_all_planes(fixture) -> None:
    source, decision, root = fixture
    change_decision(
        decision,
        lambda data: data.update(
            {
                "overrides": [],
                "reviewer_ids": [],
                "evidence": [],
                "decision": "No corrections are registered; preserve the source without claiming anatomical review.",
            }
        ),
    )
    effective, trace = apply_series_decision(source, decision, root)
    pd.testing.assert_frame_equal(effective[SOURCE_COLUMNS], pd.read_csv(source, dtype=str))
    assert effective.Anatomical_Plane_Review_ID.eq("").all()
    assert trace["n_corrections_applied"] == 0
    assert trace["reviewer_ids"] == []


def test_empty_decision_needs_explicit_provenance(fixture) -> None:
    source, decision, root = fixture
    change_decision(decision, lambda data: data.update({"overrides": []}))
    with pytest.raises(ValueError, match="explicit no-corrections provenance"):
        apply_series_decision(source, decision, root)


def test_bound_evidence_size_is_checked(fixture) -> None:
    source, decision, root = fixture
    evidence = root / "independent-review.json"
    change_decision(
        decision, lambda data: data["evidence"][0].update({"bytes": evidence.stat().st_size})
    )
    apply_series_decision(source, decision, root)
    change_decision(decision, lambda data: data["evidence"][0].update({"bytes": 1}))
    with pytest.raises(ValueError, match="byte size differs"):
        apply_series_decision(source, decision, root)


def test_existing_output_and_reapplying_export_are_rejected(fixture) -> None:
    source, decision, root = fixture
    export_series_metadata(source, decision, root, root / "export")
    with pytest.raises(FileExistsError):
        export_series_metadata(source, decision, root, root / "export")
    exported = root / "export/series_effective.csv"
    change_decision(
        decision,
        lambda data: data.update(
            {
                "source_series_sha256": sha(exported),
                "source_series_path": "export/series_effective.csv",
            }
        ),
    )
    with pytest.raises(ValueError, match="five original"):
        apply_series_decision(exported, decision, root)


def test_duplicate_json_fields_are_rejected(fixture) -> None:
    source, decision, root = fixture
    decision.write_text('{"status":"pending","status":"accepted"}')
    with pytest.raises(ValueError, match="Duplicate decision JSON"):
        apply_series_decision(source, decision, root)


@pytest.mark.parametrize("raw_split", ["train_series", "test_series"])
@pytest.mark.parametrize("augment", [False, True])
def test_effective_export_uses_existing_training_and_inference_loader(
    fixture, raw_split, augment
) -> None:
    source, decision, root = fixture
    export_series_metadata(source, decision, root, root / "export")
    series = pd.read_csv(root / "export/series_effective.csv")
    cache = root / "cache"
    for index, row in series.iterrows():
        path = cache / raw_split / row.StudyInstanceUID / f"{row.SeriesInstanceUID}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, volume=np.full((3, 16, 16), (index + 1) * 1000, dtype=np.uint16))
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "cache_version": CACHE_VERSION,
                    "study_uid": row.StudyInstanceUID,
                    "series_uid": row.SeriesInstanceUID,
                    "side": 16,
                    "source_revision": "synthetic-fixture",
                }
            )
        )
    studies = pd.DataFrame({"study_uid": ["1.2.3", "1.2.4"]})
    dataset = StudyDataset(
        studies,
        series,
        cache,
        slices=2,
        side=16,
        max_series=3,
        augment=augment,
        raw_split=raw_split,
    )
    for index in range(len(studies)):
        sample = dataset[index]
        assert sample["label_mask"].sum() == 0
        for slot in torch.where(sample["series_mask"])[0].tolist():
            # Three distinct constant fixture volumes identify the physical series.
            # Their effective plane must match regardless of priority or augmentation.
            identity = round(sample["images"][slot].mean().item() * 65535 / 1000)
            expected = {1: [1, 0, 0], 2: [0, 1, 0], 3: [0, 0, 1]}[identity]
            assert sample["features"][slot, :3].tolist() == expected
