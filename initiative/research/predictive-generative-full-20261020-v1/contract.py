"""Actual matched-pilot and full-data provenance required by a future G100 run.

This is not an admission or dispatch coordinator. No synthetic fixture or
partial history can replace GPU986011's actual terminal and separate decision.
"""

import math
import re

import base_C100_derivation as base

PILOT_SOURCE = "473a9a94d90890e8a97e7ed46d157dcf417ad173"
PILOT_BINDING = "41b7f830956bc3c7b47f6b90453cfe7fe00a2399510c5939379872719a49e4c6"
PILOT_PATH = "/tmp/rsna-predictive-generative-routing-20261020-v1/GPU-accepted.json"
ROUTING_SHA = "7a88c915209f2e51a293a046402f58d03a588268281d627ea3ac122d8c3add86"
ROOT = "/tmp/rsna-predictive-generative-full-20261020-v1"
RUN = "/tmp/rsna-predictive-full-G100-20261020-v1"
require = base.require


def validate_pilot(pilot, scheduler):
    require(
        pilot["status"] == "actual_matched_C100_G100_through_peak_engineering_complete"
        and pilot["GPU_job"] == scheduler["GPU_job"] == "986011"
        and pilot["source_git"] == PILOT_SOURCE
        and pilot["binding_sha256"] == PILOT_BINDING
        and pilot["branch"] == "matched_C100_G100_generative_routing"
        and pilot["actual_optimizer_updates"] == 600
        and pilot["fresh_initialization_equal"] is True
        and pilot["G100_full_stop_resume_complete_state_and_prediction_bytes_equal"] is True
        and pilot["primary_B_C_C0_C100_pipeline_unchanged"] is True
        and pilot["automatic_full_training_promotion"] is False
        and pilot["clinical_or_held_out_accuracy_claim"] is False,
        "Actual complete original matched G100 pilot required",
    )
    require(
        (scheduler["state"], scheduler["exit_code"], scheduler["node"])
        == ("COMPLETED", "0:0", "evc27"),
        "Actual original terminal scheduler outcome required",
    )
    require(
        [v["role"] for v in pilot["roles"]] == ["C100", "G100"],
        "Both original matched roles are required",
    )
    for role in pilot["roles"]:
        history = role["real_optimizer_history"]
        require(
            len(history) == 200
            and [v["step"] for v in history] == list(range(1, 201))
            and [v["studies"] for v in history] == [8, 4] * 100
            and [v["encoder_updated"] for v in history] == [False] * 3 + [True] * 197,
            "Full contiguous matched history and real encoder learning required",
        )
        for i, record in enumerate(history):
            fraction = i / 2915
            scale = (
                min(1.0, (i + 1) / (0.05 * 2916))
                if fraction < 0.05
                else (0.1 + 0.9 * (1 + math.cos(math.pi * (fraction - 0.05) / 0.95)) / 2)
            )
            require(
                all(
                    math.isfinite(record[k])
                    for k in (
                        "loss",
                        "exam_loss",
                        "density_loss",
                        "gradient_norm",
                        "encoder_gradient_norm",
                        "head_gradient_norm",
                    )
                )
                and record["encoder_gradient_norm"] > 0
                and record["head_gradient_norm"] > 0
                and len(record["LR"]) == 2
                and all(math.isclose(lr, 3e-5 * scale, rel_tol=1e-12) for lr in record["LR"]),
                "Actual finite original schedule through peak LR required",
            )


def validate(binding, pilot, decision, funding, admitted):
    require(
        binding["arm"] == "C"
        and binding["branch"] == "G100"
        and binding["base_science_git"] == base.BASE
        and binding["node"] == "evc27"
        and binding["run_root"] == RUN
        and binding["code_root"] == ROOT + "/code"
        and binding["expected_initial_shared_tensor_sha256"] == base.INITIAL
        and re.fullmatch("[0-9a-f]{40}", binding["source_git"])
        and binding["source_git"] not in (base.BASE, PILOT_SOURCE),
        "Distinct full-data G100 source/output binding required",
    )
    require(
        binding["generative_pilot"]["path"] == PILOT_PATH
        and all(binding["code_sha256"].get(n) == h for n, h in base.NUMERICAL_SHA.items())
        and binding["code_sha256"]["rsna/competitive/predictive/generative_routing.py"]
        == ROUTING_SHA,
        "Original numerical modules and actual routing primitive required",
    )
    validate_pilot(pilot, decision["pilot_scheduler"])
    require(
        decision["status"] == "accepted_full_predictive_fitting_recipe"
        and decision["branch"] == "G100"
        and decision["pilot_acceptance_sha256"] == binding["generative_pilot"]["sha256"]
        and decision["source_git"] == binding["source_git"]
        and decision["code_sha256"] == binding["code_sha256"]
        and decision["recipe"] == base.recipe()
        and decision["observed_C_C0_gradient_risk_reviewed"] is True
        and decision["matched_peak_LR_diagnostics_reviewed"] is True
        and decision["primary_B_C_C0_unchanged"] is True
        and decision["primary_Kaggle_reserve_seconds"] == 3360
        and decision["engineering_predictions_used_for_selection"] is False
        and decision["automatic_recipe_retuning"] is False
        and isinstance(decision["agent_reasoning"], str)
        and len(decision["agent_reasoning"]) >= 80,
        "Separate review of actual matched results and full recipe required",
    )
    original_decision = {
        k: v for k, v in decision.items() if k != "full_G100_agent_decision_sha256"
    }
    original_decision["status"] = "accepted_separate_full_G100_after_actual_pilot_review"
    require(
        base.sha(base.canonical(original_decision)) == decision["full_G100_agent_decision_sha256"],
        "Original separate full-G100 decision differs",
    )
    require(
        funding["branch"] == admitted["branch"] == "G100"
        and funding["arm"] == "C"
        and funding["run_root"] == RUN
        and funding["arm_cap_bytes"] == 12 * 2**30
        and funding["pilot_scope_retained"] is True
        and funding["all_other_obligations_retained"] is True
        and funding["no_deletion_credit"] is True
        and admitted["source_git"] == binding["source_git"]
        and admitted["code_sha256"] == binding["code_sha256"]
        and admitted["new_MRI_transferred_bytes"] == 0,
        "G100 needs its own full-data funding and input admission",
    )
    for name in ("original_portable_admission_sha256", "original_C0_binding_sha256"):
        require(
            re.fullmatch("[0-9a-f]{64}", binding[name]) and admitted[name] == binding[name],
            "Original complete input-reuse provenance differs",
        )
    return dict(
        branch="G100",
        base_science_git=base.BASE,
        generative_pilot_sha256=binding["generative_pilot"]["sha256"],
        generative_pilot_source_git=PILOT_SOURCE,
        generative_pilot_GPU_job="986011",
        routing_sha256=ROUTING_SHA,
        exam_gradients_through_Bayes_factor=False,
        proper_conditional_density_gradients_attached=True,
        original_portable_admission_sha256=binding["original_portable_admission_sha256"],
        original_C0_binding_sha256=binding["original_C0_binding_sha256"],
    )


def validate_references(binding):
    return validate(
        binding,
        *(
            base.read_json(binding[name])
            for name in ("generative_pilot", "engineering_decision", "funding", "input_admission")
        ),
    )
