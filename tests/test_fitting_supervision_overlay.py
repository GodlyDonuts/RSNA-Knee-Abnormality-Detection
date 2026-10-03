import hashlib
import json

import pandas as pd
import pytest

from rsna.competitive.supervision import (
    apply_fitting_supervision,
    load_fitting_effusion_overlay,
    roster_sha,
    validate_fitting_supervision,
)


@pytest.fixture
def inputs(tmp_path):
    train = pd.DataFrame(
        {
            "study_uid": ["fit-a", "fit-b"],
            "leakage_group": ["a", "b"],
            "split": ["train"] * 2,
            "report_supervision_allowed": [True] * 2,
            "fold": [1, 2],
            "Effusion": [1, 0],
            "Effusion__mask": [0, 1],
            "ACL": [0, 1],
            "ACL__mask": [1, 1],
        }
    )
    outer = train.iloc[[1]].copy()
    outer.loc[:, "study_uid"] = "outer"
    outer.loc[:, "leakage_group"] = "out"
    outer.loc[:, "fold"] = 0
    cell = {
        "study_uid": "fit-a",
        "target": "Effusion",
        "disposition": "conditional_training_only_reuse_recommendation",
        "first_reader_state": "positive",
        "independent_reader_state": "positive",
        "source_report_sha256": "a" * 64,
        "current_annotation_sha256": "b" * 64,
        "assessment_sha256": "c" * 64,
        "original_hold_id": "hold-original",
        "original_obligation_id": "ob-original",
    }
    value = {
        "schema_version": 1,
        "status": "source_review_accepted_experiment_only",
        "target": "Effusion",
        "fit_scope": "outer_fold",
        "fold": 0,
        "release_sha256": "d" * 64,
        "folds_sha256": "e" * 64,
        "fitting_uid_sha256": roster_sha(train),
        "outer_uid_sha256": roster_sha(outer),
        "cell_count": 1,
        "cells": [cell],
        "reviews": dict.fromkeys(
            ["first_reader", "independent_reader", "original_context", "root_assessment"],
            "f" * 64,
        ),
    }
    path = tmp_path / "overlay.json"

    def apply(expected_sha=None):
        raw = json.dumps(value).encode()
        path.write_bytes(raw)
        return load_fitting_effusion_overlay(
            train,
            outer,
            path,
            expected_sha or hashlib.sha256(raw).hexdigest(),
            release_sha256="d" * 64,
            folds_sha256="e" * 64,
            fold=0,
        )

    return train, outer, value, apply


def test_exact_recovery_preserves_source_outer_other_targets_and_roster(inputs):
    train, outer, _, apply = inputs
    original, evaluation = train.copy(deep=True), outer.copy(deep=True)
    result, binding = apply()
    expected = original.copy(deep=True)
    expected.at[0, "Effusion__mask"] = 1
    pd.testing.assert_frame_equal(result, expected)
    pd.testing.assert_frame_equal(train, original)
    pd.testing.assert_frame_equal(outer, evaluation)
    assert binding["recovered_masked_positive_cells"] == 1
    assert binding["evaluation_masks_changed"] is False


@pytest.mark.parametrize(
    "change",
    ["outer_uid", "component", "protected", "disallowed", "already_observed", "not_positive"],
)
def test_rejects_leakage_protected_or_ineligible_original_cells(inputs, change):
    train, outer, value, apply = inputs
    if change == "outer_uid":
        value["cells"][0]["study_uid"] = outer.iloc[0].study_uid
    elif change == "component":
        train.loc[0, "leakage_group"] = outer.iloc[0].leakage_group
    elif change == "protected":
        train.loc[0, "split"] = "gold"
    elif change == "disallowed":
        train.loc[0, "report_supervision_allowed"] = False
    elif change == "already_observed":
        train.loc[0, "Effusion__mask"] = 1
    else:
        train.loc[0, "Effusion"] = 0
    with pytest.raises(ValueError):
        apply()


@pytest.mark.parametrize(
    "change",
    ["duplicate", "target", "uncertain", "reviews", "fold", "release", "roster", "count"],
)
def test_rejects_unbound_or_nonaccepted_admission(inputs, change):
    _, _, value, apply = inputs
    if change == "duplicate":
        value["cells"] *= 2
        value["cell_count"] = 2
    elif change == "target":
        value["cells"][0]["target"] = "ACL"
    elif change == "uncertain":
        value["cells"][0]["independent_reader_state"] = "uncertain"
    elif change == "reviews":
        value["reviews"]["independent_reader"] = "missing"
    elif change == "fold":
        value["fold"] = 1
    elif change == "release":
        value["release_sha256"] = "a" * 64
    elif change == "roster":
        value["fitting_uid_sha256"] = "b" * 64
    else:
        value["cell_count"] = 32
    with pytest.raises(ValueError):
        apply()


def test_rejects_changed_artifact_sha(inputs):
    *_, apply = inputs
    with pytest.raises(ValueError, match="SHA"):
        apply("0" * 64)


@pytest.mark.parametrize("enabled", [False, True])
def test_control_and_candidate_require_same_complete_proposal(inputs, enabled, tmp_path):
    train, outer, value, apply = inputs
    _, admitted = apply()
    artifact = tmp_path / "proposal.json"
    artifact.write_bytes(json.dumps(value).encode())
    result, receipt = apply_fitting_supervision(
        train,
        outer,
        artifact,
        dict(kind="effusion_review_v1", sha256=admitted["overlay_sha256"], enabled=enabled),
        release_sha256="d" * 64,
        folds_sha256="e" * 64,
        fold=0,
    )
    assert result.at[0, "Effusion__mask"] == int(enabled)
    assert train.at[0, "Effusion__mask"] == 0
    assert receipt["applied_cells"] == int(enabled)
    assert receipt["evaluation_masks_changed"] is False


@pytest.mark.parametrize(
    "policy",
    [
        {},
        dict(kind="dense_soft", sha256="a" * 64, enabled=True),
        dict(kind="effusion_review_v1", sha256="a" * 64, enabled=1),
        dict(kind="effusion_review_v1", sha256="bad", enabled=True),
        dict(kind="effusion_review_v1", sha256="a" * 64, enabled=True, missing_is_negative=True),
    ],
)
def test_policy_rejects_arbitrary_dense_or_implicit_targets(policy):
    with pytest.raises(ValueError):
        validate_fitting_supervision(policy)
