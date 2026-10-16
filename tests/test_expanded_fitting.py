import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest
from test_expanded_native_cache_plan import inputs

from rsna.competitive.expanded_fitting import build_overlay, select_population, selected_source
from rsna.competitive.supervision import roster_sha
from rsna.competitive.train import select_development
from rsna.schema import TARGETS
from scripts.newton.expanded_native_cache_plan import select_additional


def frames():
    studies, folds = inputs()
    for t in TARGETS:
        studies[t] = 0.0
        studies[t + "__provenance"] = "report"
    return studies, folds


def test_population_expands_only_safe_fitting_rows_and_outer_is_identical():
    studies, folds = frames()
    original = studies.copy(deep=True)
    baseline, outer = select_development(studies, folds, 0)
    fitting, actual, proof = select_population(studies, folds)
    assert fitting.study_uid.tolist() == ["new", "old"]
    pd.testing.assert_frame_equal(actual, outer)
    pd.testing.assert_frame_equal(studies, original)
    extra, excluded = select_additional(studies, folds)
    assert set(fitting.study_uid) - set(baseline.study_uid) == set(extra.study_uid)
    assert proof["excluded_all_fold_studies"] == len(excluded) == 2
    studies.loc[studies.study_uid.eq("new"), "leakage_group"] = "observed_outer"
    with pytest.raises(ValueError):
        select_population(studies, folds)


def test_numeric_outer_source_values_are_not_interpreted():
    text = "StudyInstanceUID," + ",".join(TARGETS) + "\n"
    text += "new," + ",".join(["0.3333333333333333"] * 12) + "\n"
    text += "outer," + ",".join(["not-a-numeric-fitting-target"] * 12) + "\n"
    values = selected_source(text.encode(), {"new"})
    assert values.index.tolist() == ["new"]
    expected = pd.read_csv(__import__("io").StringIO(text.split("outer,")[0])).iloc[0, 1:]
    assert values.loc["new"].tolist() == expected.tolist()
    with pytest.raises(ValueError, match="coverage"):
        selected_source(text.encode(), {"missing"})


def overlay_files(tmp_path: Path):
    studies, folds = frames()
    baseline, outer = select_development(studies, folds, 0)
    source = tmp_path / "public.csv"
    text = "StudyInstanceUID," + ",".join(TARGETS) + "\n"
    text += "old," + ",".join(["0.3333333333333333"] * 12) + "\n"
    text += "new," + ",".join(["0.75"] * 12) + "\n"
    text += "observed_outer," + ",".join(["not-used"] * 12) + "\n"
    source.write_text(text)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    # Match the original source parser on fitting-only text; non-fitting strings
    # must not force the old decimal conversion through a different path.
    from rsna.competitive.expanded_fitting import selected_source

    source_values = selected_source(source.read_bytes(), {"old"})
    old = {
        "schema_version": 1,
        "status": "label_source_experiment_only",
        "fit_scope": "outer_fold",
        "fold": 0,
        "release_sha256": "a" * 64,
        "folds_sha256": "b" * 64,
        "source": {"csv_sha256": source_sha, "name": "pinned", "revision": "c" * 40},
        "fitting_uid_sha256": roster_sha(baseline),
        "outer_uid_sha256": roster_sha(outer),
        "study_count": 1,
        "studies": [
            {
                "study_uid": "old",
                "labels": {t: float(source_values.loc["old", t]) for t in TARGETS},
                "mask": {t: 1 for t in TARGETS},
            }
        ],
    }
    path = tmp_path / "old.json"
    path.write_text(json.dumps(old))
    return studies, folds, source, source_sha, path


def test_existing_cells_preserved_and_only_new_rows_receive_source_labels(tmp_path):
    studies, folds, source, source_sha, old = overlay_files(tmp_path)
    new = tmp_path / "expanded.json"
    proof = build_overlay(
        studies,
        folds,
        old,
        hashlib.sha256(old.read_bytes()).hexdigest(),
        source,
        source_sha,
        new,
        release_sha256="a" * 64,
        folds_sha256="b" * 64,
    )
    value = json.loads(new.read_text())
    entries = {r["study_uid"]: r for r in value["studies"]}
    assert entries["old"] == json.loads(old.read_text())["studies"][0]
    assert entries["new"]["labels"] == {t: 0.75 for t in TARGETS}
    assert proof["expanded_fitting_count"] == 2
    assert set(entries) == {"new", "old"}
    with pytest.raises(FileExistsError):
        build_overlay(
            studies,
            folds,
            old,
            hashlib.sha256(old.read_bytes()).hexdigest(),
            source,
            source_sha,
            new,
            release_sha256="a" * 64,
            folds_sha256="b" * 64,
        )


def test_old_source_changes_and_mismatched_parent_cannot_silently_enter_expansion(tmp_path):
    studies, folds, source, source_sha, old = overlay_files(tmp_path)
    value = json.loads(old.read_text())
    value["studies"][0]["labels"][TARGETS[0]] = 0.2
    old.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Existing observed"):
        build_overlay(
            studies,
            folds,
            old,
            hashlib.sha256(old.read_bytes()).hexdigest(),
            source,
            source_sha,
            tmp_path / "new.json",
            release_sha256="a" * 64,
            folds_sha256="b" * 64,
        )
