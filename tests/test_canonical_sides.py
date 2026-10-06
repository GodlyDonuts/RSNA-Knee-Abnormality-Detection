from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna.competitive.canonical_side import (
    load_canonical_side_table,
    validate_canonical_side_policy,
)
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.competitive.train import select_development
from rsna.schema import CACHE_VERSION, TARGETS
from scripts.research.build_canonical_side_table import build


def _manifest() -> pd.DataFrame:
    rows = []
    for index in range(8):
        row = {
            "study_uid": f"uid-{index}",
            "split": "train",
            "leakage_group": f"leak-{index}",
            "report_supervision_allowed": True,
        }
        for target in TARGETS:
            row[target] = 0.0
            row[f"{target}__mask"] = 1 if target == "ACL" else 0
            row[f"{target}__provenance"] = "reviewed"
        rows.append(row)
    return pd.DataFrame(rows)


def _folds() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "study_uid": [f"uid-{index}" for index in range(8)],
            "leakage_group": [f"leak-{index}" for index in range(8)],
            "fold": [1, 1, 2, 2, 3, 3, 0, 0],
        }
    )


def _census(tmp_path: Path) -> Path:
    # uid-0..3 left (x=+60), uid-4..6 right (x=-60), uid-7 ambiguous (x=5).
    rows = []
    offsets = [60, 60, 60, 60, -60, -60, -60, 5]
    for index, offset in enumerate(offsets):
        rows.append(
            {
                "study_uid": f"uid-{index}",
                "series_uid": f"uid-{index}.sag",
                "parse_ok": True,
                "anatomical_plane": "Sagittal",
                "ImagePositionPatient": f"[{offset}, 0, 0]",
                "Laterality": "",
            }
        )
    path = tmp_path / "census.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    return path


def _inputs(tmp_path):
    studies = tmp_path / "studies.parquet"
    folds = tmp_path / "fold.csv"
    census = _census(tmp_path)
    _manifest().to_parquet(studies)
    _folds().to_csv(folds, index=False)
    return studies, folds, census


def _bound(tmp_path):
    studies, folds, census = _inputs(tmp_path)
    artifact = tmp_path / "sides.json"
    build(studies=studies, folds=folds, fold=0, census=census, canonical="R", output=artifact)
    manifest = pd.read_parquet(studies)
    train, outer = select_development(manifest, pd.read_csv(folds), 0, "outer_fold")
    return train, outer, artifact, studies, folds


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_builder_and_loader_produce_mirror_flags(tmp_path):
    train, outer, artifact, studies, folds = _bound(tmp_path)
    frame, receipt = load_canonical_side_table(
        train,
        outer,
        artifact,
        _sha(artifact),
        release_sha256=_sha(studies),
        folds_sha256=_sha(folds),
        fold=0,
    )
    assert len(frame) == 8
    assert receipt["canonical_side"] == "R"
    assert receipt["mirrored_studies"] == 4
    assert receipt["unresolved_side_studies"] == 1
    by_uid = frame.set_index("study_uid")["flip"]
    for index in range(4):
        assert by_uid[f"uid-{index}"]
    for index in (4, 5, 6, 7):
        assert not by_uid[f"uid-{index}"]


def test_loader_rejects_wrong_sha(tmp_path):
    train, outer, artifact, studies, folds = _bound(tmp_path)
    with pytest.raises(ValueError, match="SHA differs"):
        load_canonical_side_table(
            train,
            outer,
            artifact,
            "0" * 64,
            release_sha256=_sha(studies),
            folds_sha256=_sha(folds),
            fold=0,
        )


def test_loader_rejects_roster_mismatch(tmp_path):
    train, outer, artifact, studies, folds = _bound(tmp_path)
    value = json.loads(artifact.read_text())
    value["sides"]["uid-0"] = "R"
    value["sides"]["extra-uid"] = "L"
    artifact.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="train\\+outer roster"):
        load_canonical_side_table(
            train,
            outer,
            artifact,
            _sha(artifact),
            release_sha256=_sha(studies),
            folds_sha256=_sha(folds),
            fold=0,
        )


def test_loader_rejects_invalid_side_value(tmp_path):
    train, outer, artifact, studies, folds = _bound(tmp_path)
    value = json.loads(artifact.read_text())
    value["sides"]["uid-0"] = "B"
    artifact.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="L, R, or null"):
        load_canonical_side_table(
            train,
            outer,
            artifact,
            _sha(artifact),
            release_sha256=_sha(studies),
            folds_sha256=_sha(folds),
            fold=0,
        )


def test_policy_validation():
    validate_canonical_side_policy(
        {"kind": "canonical_side_v1", "sha256": "a" * 64, "enabled": True}
    )
    with pytest.raises(ValueError):
        validate_canonical_side_policy(
            {"kind": "canonical_side_v2", "sha256": "a" * 64, "enabled": True}
        )
    with pytest.raises(ValueError):
        validate_canonical_side_policy({"kind": "canonical_side_v1", "sha256": "a" * 64})


def _dataset_fixture(tmp_path: Path):
    studies = pd.DataFrame(
        [
            {
                "study_uid": "1.2.3",
                TARGETS[0]: 1,
                f"{TARGETS[0]}__mask": 1,
            }
        ]
    )
    series = []
    for i, plane in enumerate(["Sagittal", "Coronal", "Axial"]):
        sid = f"1.2.3.{i + 1}"
        relative = f"train_series/1.2.3/{sid}.npz"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        # Left-to-right gradient makes horizontal mirroring observable.
        volume = np.stack(
            [np.tile(np.arange(16, dtype=np.uint16) * 4096, (16, 1)) for _ in range(4)]
        )
        np.savez_compressed(path, volume=volume)
        sops = [f"{sid}.{j + 1}" for j in range(4)]
        rawpaths = sorted(f"train_series/1.2.3/{sid}/{sop}.dcm" for sop in sops)
        metadata = dict(
            cache_version=CACHE_VERSION,
            study_uid="1.2.3",
            series_uid=sid,
            split="train_series",
            side=16,
            source_revision="a" * 40,
            n_slices=4,
            original_shape=[4, 16, 16],
            sop_uids=sops,
            slice_sha256=[hashlib.sha256(sop.encode()).hexdigest() for sop in sops],
            path_fingerprint=hashlib.sha256("\n".join(rawpaths).encode()).hexdigest(),
            cache_path=relative,
            cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            cache_bytes=path.stat().st_size,
            ordering="patient_geometry",
            positions=[[0, 0, j * 2] for j in range(4)],
            orientation=[1, 0, 0, 0, 1, 0],
        )
        path.with_suffix(".json").write_text(json.dumps(metadata))
        series.append(
            dict(
                StudyInstanceUID="1.2.3",
                SeriesInstanceUID=sid,
                Anatomical_Plane=plane,
                Fluid_Sensitive=1,
                Fat_Suppression=0,
            )
        )
    return studies, pd.DataFrame(series)


def test_dataset_mirrors_when_side_marked(tmp_path):
    studies, series = _dataset_fixture(tmp_path)
    plain = CompetitiveStudyDataset(studies, series, tmp_path, slices=2, side=16, max_series=3)
    sides = pd.DataFrame({"study_uid": ["1.2.3"], "flip": [True]})
    mirrored = CompetitiveStudyDataset(
        studies, series, tmp_path, slices=2, side=16, max_series=3, sides=sides
    )
    base = plain[0]["images"]
    flipped = mirrored[0]["images"]
    for slot in range(3):
        assert not (base[slot] == flipped[slot]).all()
        assert (base[slot] == flipped[slot].flip(-1)).all()
    left = base[0, 0, 0, :, 0]
    right = base[0, 0, 0, :, -1]
    assert (left < right).all()  # increasing gradient sanity check


def test_dataset_requires_roster_coverage(tmp_path):
    studies, series = _dataset_fixture(tmp_path)
    sides = pd.DataFrame({"study_uid": ["other"], "flip": [True]})
    with pytest.raises(ValueError, match="roster"):
        CompetitiveStudyDataset(studies, series, tmp_path, sides=sides)


def test_dataset_rejects_bad_side_table(tmp_path):
    studies, series = _dataset_fixture(tmp_path)
    with pytest.raises(ValueError, match="boolean"):
        CompetitiveStudyDataset(
            studies,
            series,
            tmp_path,
            sides=pd.DataFrame({"study_uid": ["1.2.3"], "flip": [1]}),
        )
    with pytest.raises(ValueError, match="unique"):
        CompetitiveStudyDataset(
            studies,
            series,
            tmp_path,
            sides=pd.DataFrame({"study_uid": ["1.2.3", "1.2.3"], "flip": [True, False]}),
        )
