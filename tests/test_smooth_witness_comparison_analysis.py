"""Paired inference must retain rare-class undefined draws and reject partial fits."""

import copy

import numpy as np
import pandas as pd
import pytest

from rsna.schema import TARGETS
from scripts.research.compare_smooth_witness import digest, paired_metrics, validate_final


def fixture():
    uids = [f"u{i}" for i in range(8)]
    labels = pd.DataFrame(
        {"study_uid": uids, "leakage_group": ["g0", "g0", "g1", "g1", "g2", "g2", "g3", "g3"]}
    ).set_index("study_uid")
    control = pd.DataFrame(index=uids)
    candidate = pd.DataFrame(index=uids)
    for target in TARGETS:
        labels[target] = [1, 0, 0, 0, 0, 0, 0, 0]
        labels[target + "__mask"] = 1
        control[target] = [0.5] * 8
        candidate[target] = [0.9, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]
    return labels, {"control": control, "candidate": candidate}


def test_paired_ties_and_whole_component_undefined_draws():
    labels, models = fixture()
    result = paired_metrics(labels, models, draws=100, seed=10)
    assert result["macro_auc"] == {"control": 0.5, "candidate": 1.0}
    assert result["primary_meniscus"]["delta"] == 0.5
    assert result["macro_delta_uncertainty"]["undefined_draws"] > 0
    assert (
        result["macro_delta_uncertainty"]["defined_draws"]
        + result["macro_delta_uncertainty"]["undefined_draws"]
        == 100
    )
    assert result["macro_delta_uncertainty"]["percentile_95"] == [0.5, 0.5]
    assert result == paired_metrics(labels, models, draws=100, seed=10)


def test_identical_models_are_paired_zero_not_independent_intervals():
    labels, models = fixture()
    models["candidate"] = models["control"].copy()
    result = paired_metrics(labels, models, draws=100, seed=10)
    assert result["macro_delta"] == 0 and result["macro_delta_uncertainty"]["percentile_95"] == [
        0,
        0,
    ]
    assert result["macro_delta_uncertainty"]["undefined_draws"] > 0


@pytest.mark.parametrize("bad", ["extra", "missing", "duplicate", "nonfinite", "range"])
def test_invalid_prediction_pair_rejected(bad):
    labels, models = fixture()
    if bad == "extra":
        models["candidate"].loc["not-an-outer-study"] = 0.5
    elif bad == "missing":
        models["candidate"] = models["candidate"].iloc[:-1]
    elif bad == "duplicate":
        models["candidate"] = pd.concat([models["candidate"], models["candidate"].iloc[:1]])
    elif bad == "nonfinite":
        models["candidate"].iloc[0, 0] = np.nan
    else:
        models["candidate"].iloc[0, 0] = 2
    with pytest.raises(ValueError):
        paired_metrics(labels, models, draws=100)


def final_fixture():
    config = {"epochs": 3, "diagnostic_study_limit": None}
    binding = {
        "config": config,
        "code_sha256": {"test.py": "code"},
        "training_uids_sha256": "train",
        "outer_uids_sha256": "outer",
        "folds_sha256": "folds",
        "release": {"binding_sha256": "release"},
    }
    run = {
        "status": "completed",
        "diagnostic_run": False,
        "held_out_confirmation_and_gold_used": False,
        "binding": binding,
    }
    result = {
        "status": "completed",
        "optimizer_steps": 843,
        "training_studies": 2247,
        "outer_studies": 1156,
        "binding_sha256": digest(binding),
        "checkpoint_sha256": "weights",
    }
    cpu = {
        "expected_bound_config": {"control": copy.deepcopy(config)},
        "source_sha256": {"rsna/test.py": "code"},
        "training_uids_sha256": "train",
        "outer_uids_sha256": "outer",
        "folds_sha256": "folds",
        "release_binding_sha256": "release",
    }
    pub = {
        "status": "private_original_smooth_witness_epoch_full_download_size_SHA_verified",
        "file_count": 5,
        "epoch": 3,
        "mode": "control",
        "checkpoint_sha256": "weights",
    }
    return run, result, cpu, pub


def test_full_final_proof_accepts_and_partial_or_unpreserved_weights_fail():
    run, result, cpu, pub = final_fixture()
    validate_final("control", run, result, cpu, pub)
    for target, key, value in [
        (result, "optimizer_steps", 842),
        (run, "diagnostic_run", True),
        (run, "held_out_confirmation_and_gold_used", True),
        (pub, "epoch", 2),
        (pub, "checkpoint_sha256", "different"),
    ]:
        before = target[key]
        target[key] = value
        with pytest.raises(ValueError):
            validate_final("control", run, result, cpu, pub)
        target[key] = before
    run["binding"]["outer_uids_sha256"] = "different"
    with pytest.raises(ValueError):
        validate_final("control", run, result, cpu, pub)


def test_memory_requires_its_actual_fixed_final_publication_contract():
    run, result, cpu, pub = final_fixture()
    pub["status"] = "private_original_plane_memory_epoch_full_download_size_SHA_verified"
    with pytest.raises(ValueError):
        validate_final("control", run, result, cpu, pub)
    validate_final("control", run, result, cpu, pub, publication_status=pub["status"])
    with pytest.raises(ValueError):
        validate_final(
            "control", run, result, cpu, pub, publication_status="unverified_private_upload"
        )
