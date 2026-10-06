import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from rsna.competitive.label_source import load_fitting_label_source
from rsna.schema import TARGETS
from scripts.research.build_graded_overlay import build


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> Path:
    rows = []
    for index in range(8):
        uid = f"uid-{index}"
        row = {
            "study_uid": uid,
            "leakage_group": f"g{index}",
            "split": "train",
        }
        states = ["positive", "negative", "uncertain", "not_mentioned"]
        for ti, target in enumerate(TARGETS):
            state = states[(index + ti) % 4]
            row[f"{target}__state"] = state
            row[target] = float(state == "positive")
            row[f"{target}__mask"] = float(state in ("positive", "negative"))
            row[f"{target}__provenance"] = "report_reviewed"
        rows.append(row)
    studies = tmp_path / "studies.parquet"
    pd.DataFrame(rows).to_parquet(studies)
    folds = tmp_path / "folds.csv"
    pd.DataFrame(
        {"study_uid": [f"uid-{i}" for i in range(8)], "fold": [i % 2 for i in range(8)]}
    ).to_csv(folds, index=False)
    return studies, folds


def _load_artifact(path: Path) -> dict:
    return json.loads(path.read_text())


def test_graded_builder_maps_states(tmp_path):
    studies, folds = _manifest(tmp_path)
    out = tmp_path / "graded.json"
    build(studies, folds, 0, out, 0.5)
    artifact = _load_artifact(out)
    assert artifact["status"] == "label_source_experiment_only"
    manifest = pd.read_parquet(studies)
    fold_frame = pd.read_csv(folds)
    from rsna.competitive.train import select_development

    train, outer = select_development(manifest, fold_frame, 0, "outer_fold")
    assert artifact["study_count"] == len(train)
    entry = artifact["studies"][0]
    uid = entry["study_uid"]
    row = manifest.set_index("study_uid").loc[uid]
    for target in TARGETS:
        state = row[f"{target}__state"]
        if state == "positive":
            assert entry["labels"][target] == 1.0 and entry["mask"][target] == 1
        elif state == "negative":
            assert entry["labels"][target] == 0.0 and entry["mask"][target] == 1
        elif state == "uncertain":
            assert entry["labels"][target] == 0.5 and entry["mask"][target] == 1
        else:
            assert entry["mask"][target] == 0


def test_uncertain_cells_supervised_count(tmp_path):
    studies, folds = _manifest(tmp_path)
    out = tmp_path / "graded.json"
    receipt = build(studies, folds, 0, out, 0.5)
    manifest = pd.read_parquet(studies)
    fold_frame = pd.read_csv(folds)
    from rsna.competitive.train import select_development

    train, _ = select_development(manifest, fold_frame, 0, "outer_fold")
    manifest = manifest.set_index("study_uid")
    expected = sum(
        (manifest.loc[uid, f"{t}__state"] == "uncertain")
        for uid in train.study_uid
        for t in TARGETS
    )
    assert receipt["uncertain_cells_supervised"] == int(expected)
    assert receipt["uncertain_cells_supervised"] > 0


def test_custom_uncertain_value(tmp_path):
    studies, folds = _manifest(tmp_path)
    out = tmp_path / "graded.json"
    build(studies, folds, 0, out, 0.3)
    artifact = _load_artifact(out)
    assert artifact["source"]["name"] == "v2_graded_states_u0.3"
    uncertain_labels = [
        e["labels"][t]
        for e in artifact["studies"]
        for t in TARGETS
        if e["mask"][t] == 1 and e["labels"][t] not in (0.0, 1.0)
    ]
    assert uncertain_labels and all(v == 0.3 for v in uncertain_labels)


def test_uncertain_out_of_range_rejected(tmp_path):
    studies, folds = _manifest(tmp_path)
    with pytest.raises(ValueError, match="\\[0,1\\]"):
        build(studies, folds, 0, tmp_path / "x.json", 1.5)


def test_loader_accepts_graded_artifact(tmp_path):
    studies, folds = _manifest(tmp_path)
    out = tmp_path / "graded.json"
    receipt = build(studies, folds, 0, out, 0.5)
    manifest = pd.read_parquet(studies)
    fold_frame = pd.read_csv(folds)
    from rsna.competitive.train import select_development

    train, outer = select_development(manifest, fold_frame, 0, "outer_fold")
    train = train.set_index("study_uid", drop=False)
    result, binding = load_fitting_label_source(
        train,
        outer,
        out,
        receipt["overlay_sha256"],
        release_sha256=_sha(studies),
        folds_sha256=_sha(folds),
        fold=0,
    )
    assert binding["fitting_studies"] == len(train)
    assert binding["swapped_cells"] > 0
    manifest_by = manifest.set_index("study_uid")
    for uid in result.study_uid:
        for target in TARGETS:
            state = manifest_by.loc[uid, f"{target}__state"]
            if state == "uncertain":
                assert result.set_index("study_uid").loc[uid, target] == 0.5
                assert result.set_index("study_uid").loc[uid, f"{target}__mask"] == 1
            elif state == "not_mentioned":
                assert result.set_index("study_uid").loc[uid, f"{target}__mask"] == 0


def test_loader_rejects_wrong_sha(tmp_path):
    studies, folds = _manifest(tmp_path)
    out = tmp_path / "graded.json"
    build(studies, folds, 0, out, 0.5)
    manifest = pd.read_parquet(studies)
    fold_frame = pd.read_csv(folds)
    from rsna.competitive.train import select_development

    train, outer = select_development(manifest, fold_frame, 0, "outer_fold")
    train = train.set_index("study_uid", drop=False)
    with pytest.raises(ValueError, match="SHA"):
        load_fitting_label_source(
            train,
            outer,
            out,
            "0" * 64,
            release_sha256=_sha(studies),
            folds_sha256=_sha(folds),
            fold=0,
        )


def test_missing_state_column_rejected(tmp_path):
    studies, folds = _manifest(tmp_path)
    manifest = pd.read_parquet(studies).drop(columns=[f"{TARGETS[0]}__state"])
    manifest.to_parquet(studies)
    with pytest.raises(ValueError, match="graded state"):
        build(studies, folds, 0, tmp_path / "x.json", 0.5)
