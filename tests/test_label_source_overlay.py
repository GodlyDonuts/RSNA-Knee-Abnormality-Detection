from __future__ import annotations

import hashlib
import json

import pandas as pd
import pytest

from rsna.competitive.label_source import load_fitting_label_source
from rsna.competitive.supervision import (
    apply_fitting_supervision,
    validate_fitting_supervision,
)
from rsna.competitive.train import select_development
from rsna.schema import TARGETS
from scripts.research.build_label_source_overlay import build


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


def _source() -> pd.DataFrame:
    rows = []
    for index in range(5):
        row = {"StudyInstanceUID": f"uid-{index}"}
        for target in TARGETS:
            row[target] = 0.5 + index * 0.1
        rows.append(row)
    return pd.DataFrame(rows)


def _inputs(tmp_path):
    studies = tmp_path / "studies.parquet"
    folds = tmp_path / "fold.csv"
    source = tmp_path / "source.csv"
    _manifest().to_parquet(studies)
    _folds().to_csv(folds, index=False)
    _source().to_csv(source, index=False)
    return studies, folds, source


def _bound(tmp_path):
    studies, folds, source = _inputs(tmp_path)
    overlay = tmp_path / "overlay.json"
    build(
        studies=studies,
        folds=folds,
        fold=0,
        source=source,
        output=overlay,
        source_name="test-source",
        source_revision="rev-1",
    )
    manifest = pd.read_parquet(studies)
    train, outer = select_development(manifest, pd.read_csv(folds), 0, "outer_fold")
    return train, outer, overlay, studies, folds


def _sha(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_overlay_swaps_complete_fitting_table(tmp_path):
    train, outer, overlay, studies, folds = _bound(tmp_path)
    result, binding = load_fitting_label_source(
        train,
        outer,
        overlay,
        _sha(overlay),
        release_sha256=_sha(studies),
        folds_sha256=_sha(folds),
        fold=0,
    )
    assert binding["fitting_studies"] == len(train)
    assert binding["swapped_cells"] > 0
    covered = result[result.study_uid.isin([f"uid-{index}" for index in range(5)])]
    for target in TARGETS:
        assert (covered[f"{target}__mask"] == 1).all()
    missing = result[result.study_uid == "uid-5"]
    assert (missing[[f"{t}__mask" for t in TARGETS]] == 0).all().all()
    assert outer.ACL.notna().all()


def test_overlay_rejects_wrong_sha(tmp_path):
    train, outer, overlay, studies, folds = _bound(tmp_path)
    with pytest.raises(ValueError, match="SHA differs"):
        load_fitting_label_source(
            train,
            outer,
            overlay,
            "0" * 64,
            release_sha256=_sha(studies),
            folds_sha256=_sha(folds),
            fold=0,
        )


def test_overlay_requires_complete_fitting_coverage(tmp_path):
    train, outer, overlay, studies, folds = _bound(tmp_path)
    value = json.loads(overlay.read_text())
    value["studies"] = value["studies"][:-1]
    value["study_count"] = len(value["studies"])
    overlay.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="complete fitting set"):
        load_fitting_label_source(
            train,
            outer,
            overlay,
            _sha(overlay),
            release_sha256=_sha(studies),
            folds_sha256=_sha(folds),
            fold=0,
        )


def test_overlay_rejects_out_of_range_labels(tmp_path):
    train, outer, overlay, studies, folds = _bound(tmp_path)
    value = json.loads(overlay.read_text())
    value["studies"][0]["labels"]["ACL"] = 1.5
    overlay.write_text(json.dumps(value))
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        load_fitting_label_source(
            train,
            outer,
            overlay,
            _sha(overlay),
            release_sha256=_sha(studies),
            folds_sha256=_sha(folds),
            fold=0,
        )


def test_policy_validation_accepts_label_source_kind():
    validate_fitting_supervision({"kind": "label_source_v1", "sha256": "a" * 64, "enabled": True})
    with pytest.raises(ValueError):
        validate_fitting_supervision(
            {"kind": "label_source_v2", "sha256": "a" * 64, "enabled": True}
        )


def test_apply_dispatches_label_source_and_control_keeps_original(tmp_path):
    train, outer, overlay, studies, folds = _bound(tmp_path)
    original = train.copy()
    candidate, receipt = apply_fitting_supervision(
        train,
        outer,
        overlay,
        {"kind": "label_source_v1", "sha256": _sha(overlay), "enabled": True},
        release_sha256=_sha(studies),
        folds_sha256=_sha(folds),
        fold=0,
    )
    assert receipt["enabled"] and receipt["applied_cells"] > 0
    assert not candidate.equals(original)
    control, control_receipt = apply_fitting_supervision(
        train,
        outer,
        overlay,
        {"kind": "label_source_v1", "sha256": _sha(overlay), "enabled": False},
        release_sha256=_sha(studies),
        folds_sha256=_sha(folds),
        fold=0,
    )
    assert control.equals(original)
    assert control_receipt["applied_cells"] == 0
