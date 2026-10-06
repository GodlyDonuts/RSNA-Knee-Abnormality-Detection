import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from rsna.competitive.label_source import load_fitting_label_source
from rsna.schema import TARGETS
from scripts.research.build_blended_overlay import PUBLIC_TARGETS, V2_TARGETS, build


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _manifest(tmp_path: Path) -> tuple[Path, Path, Path]:
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
    public = tmp_path / "public.csv"
    pd.DataFrame(
        [{"StudyInstanceUID": f"uid-{i}"} | {t: (i + 1) / 10 for t in TARGETS} for i in range(8)]
    ).to_csv(public, index=False)
    return studies, folds, public


def _load_artifact(path: Path) -> dict:
    return json.loads(path.read_text())


def test_blended_builder_per_target_source(tmp_path):
    studies, folds, public = _manifest(tmp_path)
    out = tmp_path / "blend.json"
    receipt = build(studies, public, folds, 0, out)
    artifact = _load_artifact(out)
    assert artifact["status"] == "label_source_experiment_only"
    assert set(artifact["source"]["public_targets"]) == PUBLIC_TARGETS
    assert artifact["source"]["v2_targets"] == V2_TARGETS
    manifest = pd.read_parquet(studies).set_index("study_uid")
    pub = pd.read_csv(public).set_index("StudyInstanceUID")
    for entry in artifact["studies"]:
        uid = entry["study_uid"]
        for target in TARGETS:
            if target in PUBLIC_TARGETS:
                assert entry["labels"][target] == float(pub.loc[uid, target])
                assert entry["mask"][target] == 1
            else:
                state = manifest.loc[uid, f"{target}__state"]
                if state == "positive":
                    assert entry["labels"][target] == 1.0 and entry["mask"][target] == 1
                elif state == "negative":
                    assert entry["labels"][target] == 0.0 and entry["mask"][target] == 1
                else:
                    assert entry["mask"][target] == 0
    assert receipt["fitting_studies"] == len(artifact["studies"])
    assert receipt["skipped_fitting_studies_missing_public"] == 0


def test_missing_public_study_skipped(tmp_path):
    studies, folds, public = _manifest(tmp_path)
    pub = pd.read_csv(public)
    pub = pub[pub.StudyInstanceUID != "uid-0"]
    pub.to_csv(public, index=False)
    out = tmp_path / "blend.json"
    receipt = build(studies, public, folds, 0, out)
    # uid-0 is in fold 0 (outer) so nothing skipped on the fitting side;
    # drop a fitting uid instead
    pub = pd.read_csv(public)
    missing_fitting = [u for u in pub.StudyInstanceUID if int(u.split("-")[1]) % 2 != 0][0]
    pub = pub[pub.StudyInstanceUID != missing_fitting]
    pub.to_csv(public, index=False)
    receipt = build(studies, public, folds, 0, tmp_path / "b2.json")
    assert receipt["skipped_fitting_studies_missing_public"] == 1


def test_loader_accepts_blended_artifact(tmp_path):
    studies, folds, public = _manifest(tmp_path)
    out = tmp_path / "blend.json"
    receipt = build(studies, public, folds, 0, out)
    manifest = pd.read_parquet(studies)
    from rsna.competitive.train import select_development

    train, outer = select_development(manifest, pd.read_csv(folds), 0, "outer_fold")
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
    pub = pd.read_csv(public).set_index("StudyInstanceUID")
    manifest_by = manifest.set_index("study_uid")
    by = result.set_index("study_uid")
    for uid in result.study_uid:
        for target in TARGETS:
            if target in PUBLIC_TARGETS:
                assert by.loc[uid, target] == float(pub.loc[uid, target])
                assert by.loc[uid, f"{target}__mask"] == 1
            elif manifest_by.loc[uid, f"{target}__state"] in ("uncertain", "not_mentioned"):
                assert by.loc[uid, f"{target}__mask"] == 0


def test_loader_rejects_wrong_sha(tmp_path):
    studies, folds, public = _manifest(tmp_path)
    out = tmp_path / "blend.json"
    build(studies, public, folds, 0, out)
    manifest = pd.read_parquet(studies)
    from rsna.competitive.train import select_development

    train, outer = select_development(manifest, pd.read_csv(folds), 0, "outer_fold")
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


def test_public_missing_targets_rejected(tmp_path):
    studies, folds, public = _manifest(tmp_path)
    pub = pd.read_csv(public).drop(columns=[TARGETS[0]])
    pub.to_csv(public, index=False)
    with pytest.raises(ValueError, match="lacks targets"):
        build(studies, public, folds, 0, tmp_path / "x.json")
