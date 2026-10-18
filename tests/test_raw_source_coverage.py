"""Missing source never becomes complete coverage or training admission."""

import hashlib
import json

import pytest

from rsna.competitive import raw_source_coverage as coverage


@pytest.fixture
def cohort(monkeypatch):
    uids = [f"private-study-{i:04d}" for i in range(3887)]
    h = hashlib.sha256(json.dumps(uids, separators=(",", ":")).encode()).hexdigest()
    monkeypatch.setattr(coverage, "POPULATION_SHA", h)
    rows = [
        dict(study_uid=u, bytes_sum=10, bytes_count=2, series_uid_count_distinct=1) for u in uids
    ]
    return uids, rows


def test_complete_source_volume_is_exact_and_identifiers_stay_private(cohort):
    out = coverage.summarize(*cohort)
    assert out["all_fitting_studies_have_original_source_metadata"] is True
    assert out["covered_original_bytes"] == 38870
    assert out["covered_original_files"] == 7774
    assert out["covered_original_series"] == 3887
    assert out["teacher_generation_or_training_admitted"] is False
    assert out["raw_pixels_or_clinical_labels_read"] is False
    assert "private-study-" not in json.dumps(out)


def test_incomplete_and_empty_inventory_keep_explicit_missing_population(cohort):
    uids, rows = cohort
    out = coverage.summarize(uids, rows[:3000])
    assert out["source_missing_fitting_studies"] == 887
    assert out["covered_original_bytes"] == 30000
    assert out["all_fitting_studies_have_original_source_metadata"] is False
    empty = coverage.summarize(uids, [])
    assert empty["source_missing_fitting_studies"] == 3887
    assert "per_study_quantiles" not in empty


@pytest.mark.parametrize("kind", ["roster", "duplicate", "zero", "impossible_series"])
def test_mixed_or_invalid_metadata_rejected(cohort, kind):
    uids, rows = cohort
    if kind == "roster":
        uids = uids[:-1]
    elif kind == "duplicate":
        rows = rows + rows[:1]
    elif kind == "zero":
        rows[0]["bytes_sum"] = 0
    else:
        rows[0]["series_uid_count_distinct"] = 3
    with pytest.raises(ValueError):
        coverage.summarize(uids, rows)
