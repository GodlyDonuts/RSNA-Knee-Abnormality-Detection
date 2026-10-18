"""Prospective numerical decision for the original combined encoder experiment.

This never launches training or accepts accuracy. An eligible result only earns
consideration for a separately funded, matched fixed-final production trial.
Finite Monte Carlo mean error is explicitly not a proof of unbiased gradients.
"""

from __future__ import annotations

import hashlib
import json
import math

SOURCE = "b91b621dd82bb9a28472d062ecf407c6fff65caf89060c44adcbc369198708d5"
WORKER = "10866782ded379290d920b19e8f6045293600478c543260978bd0b437e8aced1"
TIMING = "0cb49266262f35641942297cd910a7e2bcfda609c8cb807b5d27828221756bec"
RECOVERY = "4b129e1c5c76bcc226eff30f30e9e59b25df0ba91358771b795f7d65f68ad6b5"
MODEL = "f6a2007e7264fe5021cba4a4b58c0897a1e20f181078b7f566962ea10b2a04f4"


def _number(value, *, positive=False):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Finite numerical measurement required")
    if value < 0 or (positive and value == 0):
        raise ValueError("Nonnegative measurement with a positive denominator required")
    return float(value)


def _equal(actual, expected):
    if not math.isclose(_number(actual), expected, rel_tol=1e-10, abs_tol=1e-15):
        raise ValueError("Reported summary differs from its original measurements")


def decide(actual, recovery):
    """Read original terminal records; return a decision without changing them.

    The predeclared primary gates are at least 25% lower pooled forward and
    complete/encoder gradient MSE, no increased empirical mean-gradient error,
    and at most 10% measured optimizer-step overhead. The new reference-noise
    guard refuses a conclusion when reference MCSE exceeds 25% of ordinary MSE.
    These are engineering screening rules, not significance or clinical tests.
    """
    try:
        return _decide(actual, recovery)
    except (KeyError, TypeError, IndexError, ZeroDivisionError) as exc:
        raise ValueError("Complete original combined and recovery receipts required") from exc


def _decide(actual, recovery):
    if (
        actual["status"] != "COMPLETE"
        or actual["source_sha256"] != SOURCE
        or actual["version"] != 1
        or recovery["status"] != "COMPLETE"
        or recovery["kernel_id"] != 139417155
        or recovery["source_sha256"] != RECOVERY
    ):
        raise ValueError("Exact original terminal source/version and recovery required")
    recovered = recovery["receipt"]
    role = recovered["roles"]["influence"]
    if (
        recovered["worker_source_sha256"]
        != "f888b4cc327bf1d09adb05376a4ef3c3fa804169a601e7dc7974fb5bb3e24a2e"
        or recovered["epochs"] != 3
        or recovered["updates_per_role"] != 6
        or role[
            "complete_model_AdamW_LR_cursors_allrank_RNG_scaler_history_and_epoch_snapshots_equal"
        ]
        is not True
        or role["all16_full94_two_view_prediction_bytes_equal"] is not True
    ):
        raise ValueError("Actual complete numerical recovery is a separate prerequisite")
    r = actual["receipt"]
    if (
        r["status"] != "actual_encoder_sampling_noise_and_matched_update_cost_measured"
        or r["worker_sha256"] != WORKER
        or r["benchmark_worker_sha256"] != TIMING
        or r["trained_model_sha256"] != MODEL
        or r["fit_cases"] != 4
        or r["views_per_case"] != 2
        or r["loss_averages_views"] is not False
        or r["dense_reference_draws_per_view"] != 16
        or r["paired_sampled_draws_per_view"] != 32
        or r["forward_precision"] != "actual_FP16_autocast"
        or r["clinical_accuracy_or_production_fit_accepted"] is not False
    ):
        raise ValueError("Original trained encoder, single-view precision and draw counts required")
    groups = r["groups"]
    if (
        groups["all"]["parameters"] != 73136096
        or groups["encoder"]["parameters"] + groups["head"]["parameters"] != 73136096
        or min(groups[g]["parameters"] for g in ("encoder", "head")) <= 0
    ):
        raise ValueError("Complete encoder and head gradients required")
    reasons, evidence = [], {}
    reference_limited = False
    for group in ("all", "encoder", "head"):
        v = groups[group]
        ref = _number(v["reference_mean_squared_standard_error"])
        ordinary, controlled = v["ordinary"], v["controlled"]
        baseline = _number(ordinary["MSE_against_estimated_dense_gradient"], positive=True)
        candidate = _number(controlled["MSE_against_estimated_dense_gradient"])
        mean_base = _number(ordinary["empirical_mean_gradient_error"], positive=True)
        mean_new = _number(controlled["empirical_mean_gradient_error"])
        ratio, mean_ratio, fraction = candidate / baseline, mean_new / mean_base, ref / baseline
        _equal(v["MSE_ratio"], ratio)
        _equal(v["within_case_view_empirical_mean_error_ratio"], mean_ratio)
        _equal(v["reference_noise_fraction_of_ordinary_MSE"], fraction)
        # Keep signed reference-noise subtraction, including inconclusive negatives.
        for arm in (ordinary, controlled):
            expected = arm["MSE_against_estimated_dense_gradient"] - ref
            reported = arm["MSE_minus_estimated_reference_noise"]
            if not isinstance(reported, (int, float)) or not math.isfinite(reported):
                raise ValueError("Finite unclipped reference-noise subtraction required")
            if not math.isclose(reported, expected, rel_tol=1e-10, abs_tol=1e-15):
                raise ValueError("Reference-noise subtraction must remain unclipped")
        evidence[group] = {
            "MSE_ratio": ratio,
            "empirical_mean_error_ratio": mean_ratio,
            "reference_noise_fraction": fraction,
        }
        if fraction > 0.25:
            reference_limited = True
            reasons.append(group + "_reference_precision_insufficient")
        if ratio > (0.75 if group != "head" else 1.0):
            reasons.append(group + "_gradient_error_gate_failed")
        if mean_ratio > 1.0:
            reasons.append(group + "_empirical_mean_error_increased")
    forward = r["forward_probability_MSE"]
    forward_ratio = _number(forward["controlled"]) / _number(forward["ordinary"], positive=True)
    _equal(r["forward_probability_MSE_ratio"], forward_ratio)
    if forward_ratio > 0.75:
        reasons.append("forward_error_gate_failed")
    timing = r["actual_update_timing"]
    if (
        timing["order"] != ["ordinary", "controlled", "controlled", "ordinary"]
        or timing["fresh_two_rank_processes_per_run"] is not True
        or timing["fixed_parent_initialization_and_all_rank_RNG_equal"] is not True
        or timing["total_actual_optimizer_updates_all_runs"] != 24
        or len(timing["runs"]) != 4
    ):
        raise ValueError("Original matched fresh-process update-cost experiment required")
    times: dict[str, list[float]] = {arm: [] for arm in ("ordinary", "controlled")}
    warm: dict[str, list[float]] = {arm: [] for arm in times}
    for run, pair in enumerate(timing["runs"]):
        if len(pair) != 2:
            raise ValueError("Both actual T4 ranks required")
        arm = timing["order"][run]
        for rank, row in enumerate(pair):
            if (
                row["rank"] != rank
                or row["run"] != run
                or row["arm"] != arm
                or row["GPU"] != "Tesla T4"
                or row["actual_optimizer_updates"] != 6
                or row["initial_identity"] != timing["runs"][0][rank]["initial_identity"]
                or len(row["step_timings"]) != (0 if rank else 6)
            ):
                raise ValueError("Matched original rank/initialization/step records required")
        steps = pair[0]["step_timings"]
        if [s["update"] for s in steps] != list(range(1, 7)):
            raise ValueError("Six contiguous actual optimizer measurements required")
        values = [_number(s["seconds"], positive=True) for s in steps]
        times[arm].extend(values)
        warm[arm].extend(values[1:])
    for arm in times:
        if timing["step_seconds"][arm] != times[arm]:
            raise ValueError("Step summary must equal original rank-zero measurements")
    cost = sum(times["controlled"]) / sum(times["ordinary"])
    warm_cost = sum(warm["controlled"]) / sum(warm["ordinary"])
    _equal(timing["step_sum_ratio"], cost)
    _equal(timing["step_sum_ratio_excluding_first_each_run"], warm_cost)
    if cost > 1.10 or warm_cost > 1.10:
        reasons.append("optimizer_step_overhead_gate_failed")
    digest = hashlib.sha256(
        json.dumps(
            {"actual": actual, "recovery": recovery}, sort_keys=True, allow_nan=False
        ).encode()
    ).hexdigest()
    return {
        "status": "reference_limited"
        if reference_limited
        else (
            "eligible_for_separately_funded_matched_fit"
            if not reasons
            else "do_not_expand_unchanged_mechanism"
        ),
        "eligible_for_funded_matched_full_fit": not reasons,
        "reasons": reasons,
        "source_receipts_canonical_JSON_sha256": digest,
        "gradient_evidence": evidence,
        "forward_MSE_ratio": forward_ratio,
        "step_cost_ratio": cost,
        "warm_step_cost_ratio": warm_cost,
        "proven_gradient_unbiasedness": False,
        "clinical_or_Kaggle_accuracy_gain": False,
        "production_dispatch_or_resource_admission": False,
        "next": "actual fixed-final matched accuracy comparison only after independent source/input/capacity/funding admission"
        if not reasons
        else "preserve original negative or inconclusive result; no automatic repeat",
    }
