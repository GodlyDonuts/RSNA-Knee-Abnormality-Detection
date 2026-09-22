import numpy as np
import pandas as pd
import pytest

from rsna.competitive.evaluation import (
    evaluate_oof,
    leave_one_positive_component_out,
    paired_group_bootstrap,
)
from rsna.schema import TARGETS


def example():
    # Both folds rank perfectly; shifted fold scales corrupt pooled comparisons.
    actual = [0, 1, 0, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    manifest = pd.DataFrame(
        {
            "study_uid": ["a", "b", "c", "d"],
            "leakage_group": ["ga", "gb", "gc", "gd"],
            "split": "train",
        }
    )
    folds = manifest[["study_uid", "leakage_group"]].copy()
    folds["fold"] = [0, 0, 1, 1]
    pred = pd.DataFrame({"study_uid": manifest.study_uid, "fold": folds.fold, "seed": 123})
    for target in TARGETS:
        manifest[target] = actual
        manifest[target + "__mask"] = 1
        pred[target] = scores
    return manifest, folds, pred


def test_shifted_scale_pooling_trap_and_pair_weighting():
    manifest, folds, pred = example()
    result = evaluate_oof(manifest, folds, pred, seed=123)
    assert result["full_12_target_macro_auc"] == 1
    assert result["pooled_oof_secondary"]["full_12_target_macro_auc"] == 0.75
    assert result["within_fold_pair_totals"]["Fracture"] == 2
    # Duplicate a positive in fold0 as a distinct study: weighting follows P*N.
    extra = manifest.iloc[[1]].copy()
    extra["study_uid"], extra["leakage_group"] = "e", "ge"
    manifest = pd.concat([manifest, extra], ignore_index=True)
    folds = pd.concat(
        [folds, pd.DataFrame([{"study_uid": "e", "leakage_group": "ge", "fold": 0}])],
        ignore_index=True,
    )
    extra_pred = pred.iloc[[1]].copy()
    extra_pred["study_uid"] = "e"
    pred = pd.concat([pred, extra_pred], ignore_index=True)
    pred.loc[pred.fold.eq(1), "Fracture"] = [0.9, 0.8]
    result = evaluate_oof(manifest, folds, pred, seed=123)
    assert result["pair_weighted_target_auc"]["Fracture"] == pytest.approx(2 / 3)


def test_masked_values_and_prediction_order_invariant():
    manifest, folds, pred = example()
    manifest.loc[0, "Fracture__mask"] = 0
    first = evaluate_oof(manifest, folds, pred, seed=123)
    manifest.loc[0, "Fracture"] = np.nan
    second = evaluate_oof(manifest, folds, pred.sample(frac=1, random_state=2), seed=123)
    assert first == second


def test_undefined_not_filled_and_complete_macro_unavailable():
    manifest, folds, pred = example()
    manifest["Fracture"] = 0
    result = evaluate_oof(manifest, folds, pred, seed=123)
    assert result["pair_weighted_target_auc"]["Fracture"] is None
    assert result["full_12_target_macro_auc"] is None
    assert result["pooled_oof_secondary"]["full_12_target_macro_auc"] is None


@pytest.mark.parametrize(
    "tamper", ["missing", "duplicate", "extra", "seed", "fold", "nan", "range", "group"]
)
def test_prediction_contract_rejections(tamper):
    manifest, folds, pred = example()
    if tamper == "missing":
        pred = pred.iloc[1:]
    elif tamper == "duplicate":
        pred = pd.concat([pred, pred.iloc[:1]])
    elif tamper == "extra":
        pred.loc[0, "study_uid"] = "reserved"
    elif tamper == "seed":
        pred.loc[0, "seed"] = 124
    elif tamper == "fold":
        pred.loc[0, "fold"] = 1
    elif tamper == "nan":
        pred.loc[0, "ACL"] = np.nan
    elif tamper == "range":
        pred.loc[0, "ACL"] = 1.01
    else:
        pred["leakage_group"] = "wrong"
    with pytest.raises(ValueError):
        evaluate_oof(manifest, folds, pred, seed=123)


def test_observed_only_requires_explicit_contract():
    manifest, folds, pred = example()
    for target in TARGETS:
        manifest.loc[0, target + "__mask"] = 0
    with pytest.raises(ValueError, match="membership"):
        evaluate_oof(manifest, folds, pred.iloc[1:], seed=123)
    result = evaluate_oof(manifest, folds, pred.iloc[1:], seed=123, membership="observed_train")
    assert result["n_studies"] == 3


def test_paired_bootstrap_identical_models_zero_and_undefined_counted():
    manifest, folds, pred = example()
    result = paired_group_bootstrap(
        manifest,
        folds,
        pred,
        pred.sample(frac=1, random_state=7),
        seed=123,
        n_bootstrap=40,
        bootstrap_seed=99,
    )
    macro = result["full_12_target_macro_difference"]
    assert macro["interval"] == [0, 0]
    assert 0 < macro["undefined_resamples"] < 40
    assert macro["defined_resamples"] + macro["undefined_resamples"] == 40
    assert all(row["interval"] == [0, 0] for row in result["target_differences"].values())
    repeat = paired_group_bootstrap(
        manifest, folds, pred, pred, seed=123, n_bootstrap=40, bootstrap_seed=99
    )
    assert repeat == result


def test_paired_difference_sign_and_grouped_multiplicity():
    manifest, folds, pred = example()
    # Two studies in one component must have the same bootstrap multiplicity.
    manifest.loc[1, "leakage_group"] = "ga"
    folds.loc[1, "leakage_group"] = "ga"
    other = pred.copy()
    for target in TARGETS:
        other[target] = 1 - other[target]
    result = paired_group_bootstrap(manifest, folds, pred, other, seed=123, n_bootstrap=20)
    assert result["full_12_target_macro_difference"]["interval"] == [1, 1]
    assert result["full_12_target_macro_difference"]["undefined_resamples"] == 0


def test_no_seed_averaging_and_rare_component_sensitivity():
    manifest, folds, pred = example()
    other = pred.copy()
    other.seed = 124
    with pytest.raises(ValueError, match="seed"):
        paired_group_bootstrap(manifest, folds, pred, other, seed=123, n_bootstrap=2)
    result = leave_one_positive_component_out(
        manifest, folds, pred, seed=123, targets=("Fracture",)
    )
    assert len(result["results"]) == 2
    assert all(row["auc"] == 1 and row["pairs"] == 1 for row in result["results"])


def test_cli_explicit_inputs_full_validation_and_immutable_output(tmp_path, monkeypatch):
    import json
    import sys

    from rsna.competitive import evaluation

    manifest, folds, pred = example()
    called = []

    def verified(studies, root):
        called.append("full_verified_release")
        return manifest, {"binding_sha256": "synthetic"}

    def overlay(path, frame, binding):
        assert called == ["full_verified_release"]
        called.append("validated_folds")
        return folds

    monkeypatch.setattr(evaluation, "load_verified_release", verified)
    monkeypatch.setattr(evaluation, "load_folds", overlay)
    folder = tmp_path / "folds"
    folder.mkdir()
    folds.to_csv(folder / "fold.csv", index=False)
    (folder / "folds.json").write_text("{}")
    first, second = tmp_path / "first.csv", tmp_path / "second.csv"
    pred.iloc[:2].to_csv(first, index=False)
    pred.iloc[2:].to_csv(second, index=False)
    output = tmp_path / "evaluation"
    argv = [
        "evaluation",
        "--studies",
        str(tmp_path / "studies.parquet"),
        "--artifact-root",
        str(tmp_path),
        "--folds",
        str(folder),
        "--predictions",
        str(first),
        str(second),
        "--seed",
        "123",
        "--output",
        str(output),
    ]
    monkeypatch.setattr(sys, "argv", argv)
    evaluation.main()
    record = json.loads((output / "evaluation.json").read_text())
    assert record["status"] == "complete_training_oof_evaluation"
    assert record["selection_performed"] is False
    assert record["reserved_confirmation_or_gold_evaluated"] is False
    assert len(record["input_prediction_files"]) == 2
    assert str(tmp_path) not in (output / "evaluation.json").read_text()
    called.clear()
    with pytest.raises(ValueError, match="already exists"):
        evaluation.main()
    called.clear()
    monkeypatch.setattr(
        sys, "argv", [*argv[: argv.index(str(second))], *argv[argv.index(str(second)) + 1 :]]
    )
    with pytest.raises(ValueError, match="membership"):
        evaluation.main()


def test_reserved_labels_are_not_evaluated():
    manifest, folds, pred = example()
    baseline = evaluate_oof(manifest, folds, pred, seed=123)
    reserved = {"study_uid": "gold", "leakage_group": "gold-group", "split": "gold_evaluation"}
    for target in TARGETS:
        reserved[target] = "do-not-read-gold"
        reserved[target + "__mask"] = "do-not-read-gold-mask"
    with_reserved = pd.concat([manifest, pd.DataFrame([reserved])], ignore_index=True)
    assert evaluate_oof(with_reserved, folds, pred, seed=123) == baseline
