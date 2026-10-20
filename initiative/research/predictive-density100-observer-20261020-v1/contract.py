"""Exact original C100 pilot identity and sanitized acceptance schema."""

import hashlib
import math
import re

JOB, NODE = "981620", "evc27"
SOURCE = "068a14a75e4d2b3bd83dda7783528956f7a2ca0f"
SCIENCE = "39e315c65546aa5205518713bc27349ec63e7538"
BINDING = "e11c7c7fe5c1fd7cae87df313fb5d42be7128b7906f8d716a486b0857bbdd37a"
SCRIPT = "2c06ff98ab1a24e548e7b23f6451136f53d2cfd3fab653579258485daa4462e2"
ENGINEERING = "66bf73182bd4aefa6dbf7444a65bd1f841ab4279cabb1bec3ea058ebbbb82b10"
MEASUREMENT = "6f10fc5a21e8149731e1600088e601b97c16be42b0a9e944baa05c67f6920e2a"
ROOT = "/tmp/rsna-predictive-density100-pilot-20261020-v1"
DEADLINE = "2026-10-21T17:15:00+00:00"
ACTIVE = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED", "REQUEUED"}
TERMINAL = {"COMPLETED", "FAILED", "TIMEOUT", "CANCELLED", "OUT_OF_MEMORY", "NODE_FAIL",
            "PREEMPTED", "BOOT_FAIL", "DEADLINE", "REVOKED"}


def require(ok, reason):
    if not ok:
        raise ValueError(reason)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def digest(value):
    return isinstance(value, str) and re.fullmatch("[0-9a-f]{64}", value) is not None


def submitted(value):
    require(
        value["status"] == "actual_single_C100_pilot_GPU_submitted"
        and value["GPU_job"] == JOB and value["node"] == NODE
        and value["source_git"] == SOURCE and value["binding_sha256"] == BINDING
        and value["script_sha256"] == SCRIPT and value["CPU_admission_job"] == "981619"
        and value["branch"] == "C100" and value["density_weight"] == 100
        and value["updates"] == 24 and value["runtime_cap_seconds"] == 1800
        and value["training_studies"] == 12 and value["engineering_studies"] == 4
        and value["scheduler_finish_deadline_UTC"] == DEADLINE
        and value["automatic_full_training_promotion"] is False,
        "Exact original pilot submission required",
    )


def scheduler(accounting, queue):
    rows = [line.split("|") for line in accounting.splitlines()
            if line.split("|")[0] == JOB]
    require(len(rows) == 1 and len(rows[0]) >= 4, "Exact original accounting row missing")
    _, state, code, node, *_ = rows[0]
    state = state.split()[0]
    require(state in ACTIVE | TERMINAL and re.fullmatch(r"\d+:\d+", code),
            "Unrecognized original scheduler state")
    live = [line for line in queue.splitlines() if line.split("|")[0] == JOB]
    require(len(live) <= 1, "Duplicate original live job identity")
    if state == "PENDING":
        require(node in ("", "None assigned", "(null)", "None"), "Pending node binding differs")
    elif state in {"RUNNING", "COMPLETING", "COMPLETED"}:
        require(node == NODE, "Original allocated node differs")
    else:
        require(node in (NODE, "", "None assigned", "(null)", "None"), "Original node differs")
    return {"GPU_job": JOB, "state": state, "exit_code": code, "node": node,
            "live_queue_empty": not live, "accounting_sha256": sha(accounting.encode()),
            "queue_sha256": sha(queue.encode())}


def acceptance(value):
    fields = {
        "status", "GPU_job", "source_git", "original_numerical_source_git", "branch",
        "original_engineering_receipt_sha256", "objective_measurement_sha256",
        "scientific_decision_sha256", "binding_sha256",
        "full_stop_resume_complete_state_and_prediction_bytes_equal",
        "private_engineering_prediction_sha256", "terminal", "real_optimizer_history",
        "engineering_warnings", "storage", "new_shared_files", "clinical_or_held_out_accuracy_claim",
        "automatic_full_training_promotion", "primary_B_C_C0_pipeline_unchanged",
    }
    require(set(value) == fields, "Unexpected acceptance fields; do not export private additions")
    require(
        value["status"] == "actual_C100_full_stop_resume_engineering_complete"
        and value["GPU_job"] == JOB and value["source_git"] == SOURCE
        and value["original_numerical_source_git"] == SCIENCE
        and value["branch"] == "C100" and value["binding_sha256"] == BINDING
        and value["original_engineering_receipt_sha256"] == ENGINEERING
        and value["objective_measurement_sha256"] == MEASUREMENT
        and digest(value["scientific_decision_sha256"])
        and digest(value["private_engineering_prediction_sha256"])
        and value["full_stop_resume_complete_state_and_prediction_bytes_equal"] is True
        and value["clinical_or_held_out_accuracy_claim"] is False
        and value["automatic_full_training_promotion"] is False
        and value["primary_B_C_C0_pipeline_unchanged"] is True and value["new_shared_files"] == 0,
        "Original complete C100 acceptance identity differs",
    )
    history = value["real_optimizer_history"]
    history_fields = {"step", "epoch", "studies", "loss", "exam_loss", "density_loss",
                      "encoder_gradient_norm", "head_gradient_norm", "gradient_norm",
                      "LR", "encoder_updated"}
    require(len(history) == 24, "Complete actual history required")
    for i, row in enumerate(history):
        require(set(row) == history_fields and row["step"] == i + 1
                and row["epoch"] == i // 2 + 1 and row["studies"] == (8 if i % 2 == 0 else 4)
                and row["encoder_updated"] is (i >= 3), "Original update sequence differs")
        require(all(isinstance(row[key], (int, float)) and math.isfinite(row[key]) for key in
                    history_fields - {"LR", "encoder_updated"}), "Nonfinite training evidence")
        require(len(row["LR"]) == 2 and all(math.isfinite(x) and 0 < x <= 3e-5
                                            for x in row["LR"]), "Original finite learning rates")
    terminal = value["terminal"]
    terminal_fields = {
        "arm", "phase", "steps", "optimizer_seconds", "fit_CPU_context_load_seconds",
        "fit_study_exposures", "peak_GPU_bytes", "parameters", "initial_exam_loss",
        "final_total_loss", "source_studies", "disjoint_engineering_prediction_studies",
        "example_source_regions", "context_windows_per_mask", "density_weight",
        "fixed_ImageNet_channel_normalization", "all_encoder_parameters_trainable",
        "clinical_accuracy_or_submission_candidate", "exact_FP32_context_cache",
        "zero_initial_density_means", "production_LR_horizon_updates",
        "training_only_numerical_stability_not_accuracy", "training_only_numerical_ranges",
        "fractional_warmup_clamped",
    }
    require(set(terminal) == terminal_fields and terminal["arm"] == "C"
            and terminal["phase"] == "full" and terminal["steps"] == 24
            and terminal["density_weight"] == 100 and terminal["source_studies"] == 12
            and terminal["disjoint_engineering_prediction_studies"] == 4
            and terminal["fit_study_exposures"] == 144
            and terminal["production_LR_horizon_updates"] == 2916,
            "Original bounded terminal differs")
    ranges = terminal["training_only_numerical_ranges"]
    require(set(ranges) == {
        "maximum_absolute_density_mean", "maximum_absolute_shift",
        "maximum_absolute_log_bayes_factor", "maximum_absolute_local_log_ratio",
        "maximum_absolute_context_logits",
    } and all(isinstance(x, (int, float)) and math.isfinite(x) for x in ranges.values()),
        "Finite training ranges required")
    warning_fields = {
        "last_six_exam_mean", "original_C_last_six_exam_mean", "maximum_exam_loss",
        "original_C_maximum_exam_loss", "maximum_absolute_BF", "original_C_maximum_absolute_BF",
        "last_six_exam_mean_over_twice_original", "maximum_exam_over_twice_original",
        "maximum_BF_over_ten_times_original", "weighted_fixed_Gaussian_constant",
        "density_minus_constant_by_update", "clipped_update_count",
        "warning_is_not_automatic_clinical_rejection_or_promotion",
    }
    warnings = value["engineering_warnings"]
    require(set(warnings) == warning_fields
            and len(warnings["density_minus_constant_by_update"]) == 24,
            "Exact sanitized warning schema required")
    require(set(value["storage"]) == {"engineering_bytes", "controls_and_logs_bytes"}
            and all(type(x) is int and x >= 0 for x in value["storage"].values()),
            "Exact storage counters required")
    # No arbitrary strings, arrays or identifiers can hide in nested diagnostics.
    def finite_tree(item):
        if isinstance(item, dict):
            return all(finite_tree(x) for x in item.values())
        if isinstance(item, list):
            return all(finite_tree(x) for x in item)
        return isinstance(item, (int, float, bool)) and math.isfinite(item)
    require(finite_tree(warnings) and finite_tree(
        {key: x for key, x in terminal.items() if key not in {"arm", "phase"}}),
        "Only finite numeric sanitized diagnostics may be collected")
