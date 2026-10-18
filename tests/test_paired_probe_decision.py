"""Prospective decisions use original numerical records, never headline ratios."""

import copy

import pytest

from rsna.competitive.paired_probe_decision import (
    MODEL,
    RECOVERY,
    SOURCE,
    TIMING,
    WORKER,
    decide,
)


def records():
    groups = {}
    for group, count in (("all", 73136096), ("encoder", 72856264), ("head", 279832)):
        groups[group] = {
            "parameters": count,
            "reference_mean_squared_standard_error": 0.1,
            "ordinary": {
                "MSE_against_estimated_dense_gradient": 2.0,
                "empirical_mean_gradient_error": 0.4,
                "MSE_minus_estimated_reference_noise": 1.9,
            },
            "controlled": {
                "MSE_against_estimated_dense_gradient": 1.0,
                "empirical_mean_gradient_error": 0.3,
                "MSE_minus_estimated_reference_noise": 0.9,
            },
            "MSE_ratio": 0.5,
            "within_case_view_empirical_mean_error_ratio": 0.3 / 0.4,
            "reference_noise_fraction_of_ordinary_MSE": 0.05,
        }
    runs = []
    for run, arm in enumerate(("ordinary", "controlled", "controlled", "ordinary")):
        runs.append(
            [
                {
                    "arm": arm,
                    "run": run,
                    "rank": rank,
                    "GPU": "Tesla T4",
                    "actual_optimizer_updates": 6,
                    "initial_identity": {"rank": rank},
                    "step_timings": []
                    if rank
                    else [{"update": i, "seconds": 1.0} for i in range(1, 7)],
                }
                for rank in range(2)
            ]
        )
    r = {
        "status": "actual_encoder_sampling_noise_and_matched_update_cost_measured",
        "worker_sha256": WORKER,
        "benchmark_worker_sha256": TIMING,
        "trained_model_sha256": MODEL,
        "fit_cases": 4,
        "views_per_case": 2,
        "loss_averages_views": False,
        "dense_reference_draws_per_view": 16,
        "paired_sampled_draws_per_view": 32,
        "forward_precision": "actual_FP16_autocast",
        "clinical_accuracy_or_production_fit_accepted": False,
        "groups": groups,
        "forward_probability_MSE": {"ordinary": 1.0, "controlled": 0.5},
        "forward_probability_MSE_ratio": 0.5,
        "actual_update_timing": {
            "order": ["ordinary", "controlled", "controlled", "ordinary"],
            "fresh_two_rank_processes_per_run": True,
            "fixed_parent_initialization_and_all_rank_RNG_equal": True,
            "total_actual_optimizer_updates_all_runs": 24,
            "runs": runs,
            "step_seconds": {"ordinary": [1.0] * 12, "controlled": [1.0] * 12},
            "step_sum_ratio": 1.0,
            "step_sum_ratio_excluding_first_each_run": 1.0,
        },
    }
    recovery = {
        "status": "COMPLETE",
        "kernel_id": 139417155,
        "source_sha256": RECOVERY,
        "receipt": {
            "worker_source_sha256": "f888b4cc327bf1d09adb05376a4ef3c3fa804169a601e7dc7974fb5bb3e24a2e",
            "epochs": 3,
            "updates_per_role": 6,
            "roles": {
                "influence": {
                    "complete_model_AdamW_LR_cursors_allrank_RNG_scaler_history_and_epoch_snapshots_equal": True,
                    "all16_full94_two_view_prediction_bytes_equal": True,
                }
            },
        },
    }
    return {"status": "COMPLETE", "version": 1, "source_sha256": SOURCE, "receipt": r}, recovery


def test_eligible_never_means_accuracy_or_dispatch_and_preserves_inputs():
    actual, recovery = records()
    originals = copy.deepcopy((actual, recovery))
    result = decide(actual, recovery)
    assert result["eligible_for_funded_matched_full_fit"] is True
    assert result["clinical_or_Kaggle_accuracy_gain"] is False
    assert result["production_dispatch_or_resource_admission"] is False
    assert result["proven_gradient_unbiasedness"] is False
    assert (actual, recovery) == originals


def test_gradient_improvement_cannot_hide_worse_mean_gradient_error():
    actual, recovery = records()
    v = actual["receipt"]["groups"]["encoder"]
    v["controlled"]["empirical_mean_gradient_error"] = 0.8
    v["within_case_view_empirical_mean_error_ratio"] = 2.0
    result = decide(actual, recovery)
    assert result["eligible_for_funded_matched_full_fit"] is False
    assert "encoder_empirical_mean_error_increased" in result["reasons"]


def test_noisy_dense_reference_is_inconclusive_not_an_automatic_repeat():
    actual, recovery = records()
    v = actual["receipt"]["groups"]["all"]
    v["reference_mean_squared_standard_error"] = 1.5
    v["reference_noise_fraction_of_ordinary_MSE"] = 0.75
    v["ordinary"]["MSE_minus_estimated_reference_noise"] = 0.5
    v["controlled"]["MSE_minus_estimated_reference_noise"] = -0.5
    result = decide(actual, recovery)
    assert result["status"] == "reference_limited"
    assert not result["eligible_for_funded_matched_full_fit"]
    assert "no automatic repeat" in result["next"]


def test_runtime_cost_blocks_a_numerically_better_estimator():
    actual, recovery = records()
    timing = actual["receipt"]["actual_update_timing"]
    for run in (1, 2):
        for step in timing["runs"][run][0]["step_timings"]:
            step["seconds"] = 2.0
    timing["step_seconds"]["controlled"] = [2.0] * 12
    timing["step_sum_ratio"] = timing["step_sum_ratio_excluding_first_each_run"] = 2.0
    result = decide(actual, recovery)
    assert "optimizer_step_overhead_gate_failed" in result["reasons"]
    assert not result["eligible_for_funded_matched_full_fit"]


@pytest.mark.parametrize(
    "fault", ["source", "recovery", "ratio", "rank", "step", "head_only", "nan"]
)
def test_invalid_or_cherry_picked_records_are_not_scientific_results(fault):
    actual, recovery = records()
    r = actual["receipt"]
    if fault == "source":
        actual["source_sha256"] = "0" * 64
    elif fault == "recovery":
        recovery["status"] = "QUEUED"
    elif fault == "ratio":
        r["groups"]["all"]["MSE_ratio"] = 0.01
    elif fault == "rank":
        r["actual_update_timing"]["runs"][2][1]["initial_identity"] = {"rank": 9}
    elif fault == "step":
        r["actual_update_timing"]["runs"][0][0]["step_timings"][1]["update"] = 1
    elif fault == "head_only":
        del r["groups"]["encoder"]
    else:
        r["forward_probability_MSE"]["controlled"] = float("nan")
    with pytest.raises(ValueError):
        decide(actual, recovery)
