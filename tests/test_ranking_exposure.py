import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna.schema import TARGETS
from scripts.research.audit_ranking_exposure import (
    canonical,
    comparison_pairs,
    digest,
    fitting_inputs,
    summarize_batches,
    verify_original_observed_counts,
)


def arrays(size=3):
    labels = np.full((size, len(TARGETS)), np.nan)
    masks = np.zeros_like(labels)
    labels[:2, 0] = [1, 0]
    masks[:2, 0] = 1
    return labels, masks, np.array([f"group-{i}" for i in range(size)])


def test_effective_pairs_span_single_study_microbatches():
    labels, masks, groups = arrays()
    result = summarize_batches(
        labels, masks, groups, batch_size=1, accumulation_steps=4, seed=4, epochs=2
    )
    target = result["targets"]["ACL"]
    assert target["cross_component_pairs"] == 2
    assert target["within_microbatch_pairs"] == 0
    assert target["updates_with_cross_component_pair"] == 2
    assert result["optimizer_updates_replayed"] == 2
    assert all(row["real_fitting_rows"] == 3 for row in result["per_epoch"])
    # Tail padding never creates duplicate positive exposures or comparisons.
    assert target["updates_with_positive"] == 2
    assert result["targets"]["MCL"]["updates_observed"] == 0


def test_same_component_comparisons_excluded():
    pos, neg = np.array([True, False]), np.array([False, True])
    assert comparison_pairs(pos, neg, np.array(["same", "same"])) == (0, 1)
    assert comparison_pairs(pos, neg, np.array(["first", "second"])) == (1, 0)
    labels, masks, _ = arrays(2)
    result = summarize_batches(
        labels,
        masks,
        np.array(["same", "same"]),
        batch_size=2,
        accumulation_steps=1,
        seed=0,
        epochs=1,
    )
    assert result["targets"]["ACL"]["cross_component_pairs"] == 0
    assert result["targets"]["ACL"]["same_component_pairs_excluded"] == 1


def test_missing_values_have_no_rank_and_replay_is_deterministic():
    labels, masks, groups = arrays(5)
    kwargs = dict(batch_size=2, accumulation_steps=2, seed=19, epochs=3)
    first = summarize_batches(labels, masks, groups, **kwargs)
    labels[masks == 0] = 1
    assert summarize_batches(labels, masks, groups, **kwargs) == first
    assert first["targets"]["ACL"]["fitting_observed_positive"] == 1
    assert first["targets"]["ACL"]["fitting_observed_negative"] == 1


@pytest.mark.parametrize("change", ["soft_label", "nan_observed", "soft_mask", "negative_mask"])
def test_observed_binary_admission(change):
    labels, masks, groups = arrays()
    if change == "soft_label":
        labels[0, 0] = 0.4
    elif change == "nan_observed":
        labels[0, 0] = np.nan
    elif change == "soft_mask":
        masks[0, 0] = 0.4
    else:
        masks[0, 0] = -1
    with pytest.raises(ValueError, match="binary observed"):
        summarize_batches(
            labels, masks, groups, batch_size=1, accumulation_steps=4, seed=0, epochs=1
        )


def input_fixture(tmp_path):
    folds = pd.DataFrame(
        {
            "study_uid": ["a", "b", "c", "d"],
            "leakage_group": ["ga", "gb", "gc", "gd"],
            "fold": [1, 1, 0, 1],
        }
    )
    folds_path = tmp_path / "folds.csv"
    folds.to_csv(folds_path, index=False)
    frame = folds.drop(columns="fold").copy()
    frame["split"] = "train"
    frame["report_supervision_allowed"] = True
    for target in TARGETS:
        frame[f"{target}__mask"] = [1, 1, 1, 0]
        # Sentinel outer values must never be projected into fitting input.
        frame[target] = [1, 0, 99, np.nan]
    studies = tmp_path / "studies.parquet"
    frame.to_parquet(studies, index=False)
    from rsna.competitive import sampling

    binding = {
        "config": {
            "fit_scope": "outer_fold",
            "diagnostic_study_limit": None,
            "fold": 0,
            "batch_size": 1,
            "accumulation_steps": 8,
            "epochs": 3,
            "seed": 20260922,
        },
        "world_size": 1,
        "torch_version": __import__("torch").__version__,
        "targets": list(TARGETS),
        "experimental_fitting_supervision": {"enabled": False, "applied_cells": 0},
        "release": {"manifest_sha256": digest(studies)},
        "code_sha256": {"competitive/sampling.py": digest(Path(sampling.__file__))},
        "folds_sha256": canonical(folds.to_dict("records")),
        "training_uids_sha256": canonical(["a", "b"]),
        "outer_uids_sha256": canonical(["c"]),
    }
    run = tmp_path / "run.json"
    run.write_text(json.dumps({"status": "completed", "binding": binding}))
    return studies, folds_path, run


def test_target_projection_is_fitting_only(tmp_path, monkeypatch):
    studies, folds, run = input_fixture(tmp_path)
    original_read = pd.read_parquet
    calls = []

    def projected(path, *, columns, filters):
        calls.append((columns, filters))
        if "ACL" in columns:
            assert ("study_uid", "in", ["a", "b"]) in filters
        else:
            assert not any(target in columns for target in TARGETS)
        assert ("split", "==", "train") in filters
        return original_read(path, columns=columns, filters=filters)

    monkeypatch.setattr(pd, "read_parquet", projected)
    frame, binding = fitting_inputs(studies, folds, run)
    assert frame.study_uid.tolist() == ["a", "b"]
    assert len(calls) == 2
    assert binding["outer_confirmation_gold_target_values_decoded"] is False
    assert not frame.ACL.eq(99).any()


@pytest.mark.parametrize("change", ["sampler", "roster", "release", "candidate", "folds"])
def test_original_bindings_are_required(tmp_path, change):
    studies, folds, run = input_fixture(tmp_path)
    value = json.loads(run.read_text())
    binding = value["binding"]
    if change == "sampler":
        binding["code_sha256"]["competitive/sampling.py"] = "0" * 64
    elif change == "roster":
        binding["training_uids_sha256"] = "0" * 64
    elif change == "release":
        binding["release"]["manifest_sha256"] = "0" * 64
    elif change == "candidate":
        binding["experimental_fitting_supervision"]["enabled"] = True
    else:
        binding["folds_sha256"] = "0" * 64
    run.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        fitting_inputs(studies, folds, run)


def test_actual_training_observed_mask_sequence(tmp_path):
    original, resumed = tmp_path / "original.out", tmp_path / "resumed.out"
    old = [{"optimizer_steps": n, "observed_cells": float(n)} for n in range(1, 4)]
    new = [{"optimizer_steps": n, "observed_cells": float(n)} for n in range(3, 5)]
    original.write_text("driver header\n" + "\n".join(map(json.dumps, old)))
    resumed.write_text("\n".join(map(json.dumps, new)))
    summary = {
        "optimizer_updates_replayed": 4,
        "observed_cells_sequence_sha256": canonical([1, 2, 3, 4]),
    }
    receipt = verify_original_observed_counts(summary, original, resumed)
    assert receipt["updates_verified"] == 4
    assert receipt["overlap_updates_verified"] == 1
    summary["observed_cells_sequence_sha256"] = canonical([1, 2, 3, 5])
    with pytest.raises(ValueError, match="actual GPU"):
        verify_original_observed_counts(summary, original, resumed)


@pytest.mark.parametrize("change", ["overlap", "gap"])
def test_actual_log_overlap_and_counters_rejected(tmp_path, change):
    original, resumed = tmp_path / "original.out", tmp_path / "resumed.out"
    original.write_text(json.dumps({"optimizer_steps": 1, "observed_cells": 2}))
    resumed.write_text(
        json.dumps({"optimizer_steps": 1 if change == "overlap" else 3, "observed_cells": 3})
    )
    with pytest.raises(ValueError):
        verify_original_observed_counts({"optimizer_updates_replayed": 3}, original, resumed)
