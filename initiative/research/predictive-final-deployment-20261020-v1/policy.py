"""Pure final-export, funding and runtime policy; importing performs no work."""

import hashlib
import json
import math
from datetime import datetime, timezone

ARMS = ("B", "C", "C0")
NODE_CAP = 2 * 2**30
SHARED_BYTES = 2**20
SHARED_FILES = 8
PARENT_DATASET = "nartaa/rsna-knee-publication-swa-weights-20261007"
PARENT_ID = 12431568
PUBLIC_ROSTER = "e743f6d8dca6eabedec2f05a98cf86ed820ef1fe8a0eb06338af5b377dc8a59b"
FAMILY = "licensed-parent-full-epoch-serving:predictive-final-v1:"
WORKFLOW_SHA = "2d3074fbf677285687e80c0e6b58721407bc61ddea39a990797da0e1d4e3b900"
CAPACITY_SHA = "cd4dfb8261b6dba74e4e78ee6177a4027a19963560bc6fa3d244e6b878199092"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(value):
    return hashlib.sha256(value.encode() if isinstance(value, str) else value).hexdigest()


def final_outcome(candidate, terminal, scheduler):
    """The independently preserved e6 export also needs actual full-run completion."""
    p = candidate["provenance"]
    require(
        terminal["status"] == "actual_fixed_final_full3887_predictive_arm_complete"
        and terminal["epochs"] == 6
        and terminal["updates"] == 2916
        and terminal["study_exposures"] == 3887 * 6
        and terminal["retained_full_checkpoints"] == 7
        and terminal["immutable_epoch_sidecars"] == 6
        and terminal["all_original_fitting_labels_masks_unchanged"] is True
        and terminal["protected_or_gold_cases_read"] is False,
        "Actual complete fixed-final full-cohort execution required",
    )
    for name in (
        "arm",
        "GPU_job",
        "source_git",
        "binding_sha256",
        "recipe",
        "parent_weight_sha256",
        "parent_source_sha256",
        "source_revision",
        "renderer_sha256",
        "input_admission_sha256",
        "runtime_versions",
        "parameter_count",
        "model_schema_sha256",
        "context_index_sha256",
        "source_context_index_sha256",
        "targets_sha256",
        "fitting_order_sha256",
        "license_sha256",
        "runtime_receipt_sha256",
        "engineering_decision_sha256",
        "funding_sha256",
        "initial_shared_tensor_sha256",
    ):
        require(terminal[name] == p[name], "Final outcome and preserved export differ: " + name)
    require(
        str(scheduler["job_id"]) == str(p["GPU_job"])
        and scheduler["state"] == "COMPLETED"
        and scheduler["exit_code"] == "0:0",
        "Original actual Slurm COMPLETED 0:0 is required",
    )
    # final.pt and epoch-006.pt have distinct serializations: do not equate their SHA.
    require(
        len(terminal["final_checkpoint_sha256"]) == 64 and terminal["final_checkpoint_bytes"] > 0,
        "Actual separate final full checkpoint proof required",
    )


def storage(funding, *, node_free, shared_free, shared_files, used=0, now=None):
    now = datetime.now(timezone.utc) if now is None else now
    checked = datetime.fromisoformat(funding["checked_at"])
    require(0 <= (now - checked).total_seconds() <= 300, "Fresh explicit funding required")
    require(
        funding["status"] == "predictive_final_deployment_incremental_storage_funded"
        and funding["new_node_bytes"] == NODE_CAP
        and funding["new_shared_bytes"] == SHARED_BYTES
        and funding["new_shared_files"] == SHARED_FILES
        and funding["existing_60GiB_or_574MiB_reused"] is False,
        "New deployment liabilities must be explicitly funded",
    )
    names = ("other_node_unpaid_bytes", "other_shared_unpaid_bytes", "other_shared_unpaid_files")
    require(
        all(type(funding[n]) is int and funding[n] >= 0 for n in names),
        "Retained other liabilities required without deletion credit",
    )
    require(0 <= used <= NODE_CAP, "Node-local deployment scope exceeds its bound")
    require(
        node_free >= 20 * 2**30 + funding[names[0]] + NODE_CAP - used
        and shared_free >= 20 * 2**30 + funding[names[1]] + SHARED_BYTES
        and shared_files >= 5000 + funding[names[2]] + SHARED_FILES,
        "Fresh reserve after all other and incremental liabilities is insufficient",
    )


def free_seconds(quota):
    def seconds(value):
        number = float(value.removesuffix("s"))
        require(math.isfinite(number) and number >= 0, "Finite authoritative GPU quota required")
        return number

    # Same authoritative arithmetic as original CPU969788 / actual-budget-decision:
    # minimumTimeAllowed is a floor, not this account's granted 162000s ceiling.
    # This control neither changes billing nor requests paid credits.
    return max(
        0.0,
        seconds(quota["totalTimeAllowed"])
        - seconds(quota.get("timeUsed", "0s"))
        - seconds(quota.get("timeReserved", "0s")),
    )


def runtime_plan(plan, actual_probe, *, now=None):
    """An explicit engineering estimate, never proof of hidden test size/runtime."""
    now = datetime.now(timezone.utc) if now is None else now
    require(
        plan["status"] == "accepted_predictive_final_runtime_budget"
        and 0 <= (now - datetime.fromisoformat(plan["checked_at"])).total_seconds() <= 300
        and plan["probe_receipt_sha256"] == sha(canonical(actual_probe))
        and actual_probe["status"] == "actual_public_MRI_predictive_two_T4_measurements_recorded"
        and actual_probe["completed_device_count"] == 2
        and not actual_probe["device_errors"]
        and actual_probe["host22GiB_device13GiB_memory_checks"] is True
        and actual_probe["submission_csv_absent"] is True,
        "Fresh decision bound to actual measured original two-T4 probe required",
    )
    require(
        type(plan["qualification_cap_seconds"]) is int
        and 1 <= plan["qualification_cap_seconds"] <= 1800
        and type(plan["public_watchdog_seconds"]) is int
        and plan["public_watchdog_seconds"] == 600
        and type(plan["other_reserved_VM_seconds"]) is int
        and plan["other_reserved_VM_seconds"] >= 0
        and type(plan["planning_study_count"]) is int
        and plan["planning_study_count"] > 0
        and len(plan["planning_count_basis"]) >= 20
        and plan["hidden_runtime_proven"] is False
        and plan["public_tail_extrapolation_risk_accepted"] is True,
        "Bounded qualification, public watchdog and explicit hidden-tail limitation required",
    )
    require(
        math.isfinite(plan["startup_allowance_seconds"])
        and plan["startup_allowance_seconds"] >= 0
        and math.isfinite(plan["tail_sensitivity_factor"])
        and plan["tail_sensitivity_factor"] >= 1
        and math.isfinite(plan["safety_margin_seconds"])
        and plan["safety_margin_seconds"] >= 0,
        "Explicit complete-pipeline planning assumptions required",
    )
    return plan


def budget(plan, quota, phase, remaining_public_arms):
    require(phase in ("qualification", "competition"), "Known GPU stage required")
    require(
        type(remaining_public_arms) is int and 1 <= remaining_public_arms <= 3,
        "Explicit remaining independent public executions required",
    )
    required = plan["qualification_cap_seconds"] if phase == "qualification" else 0
    required += remaining_public_arms * plan["public_watchdog_seconds"]
    required += plan["other_reserved_VM_seconds"]
    return {
        "free_VM_seconds": free_seconds(quota),
        "required_VM_seconds": required,
        "available": free_seconds(quota) >= required,
        "paid_credits_used": False,
    }


def hidden_projection(plan, model_measurements, *, qualification_sha256):
    """Report sensitivity; admit the explicit selected scenario, not an invented tail veto."""
    by_study = {}
    for value in model_measurements:
        index = value["study_index"]
        require(type(index) is int and index >= 0, "Actual public study index required")
        seconds = value["decode_seconds"] + value["render_seconds"] + value["FP16_seconds"]
        require(math.isfinite(seconds) and seconds > 0, "Finite positive pipeline timing required")
        by_study[index] = max(by_study.get(index, 0), seconds)
    # The cross-device repeat is a precision check, not a fourth independent MRI.
    require(set(by_study) == {0, 1, 2}, "All three unique public studies required")
    per_study = list(by_study.values())
    require(
        per_study and all(math.isfinite(x) and x > 0 for x in per_study),
        "Actual optimized trained complete-pipeline timing required",
    )
    n, startup = plan["planning_study_count"], plan["startup_allowance_seconds"]
    scenarios = {
        "mean_two_lane": startup + n * sum(per_study) / len(per_study) / 2,
        "max_case_two_lane": startup + n * max(per_study) / 2,
        "tail_stress_two_lane": startup + n * max(per_study) * plan["tail_sensitivity_factor"] / 2,
    }
    decision = plan["entry_decision"]
    require(
        decision["accepted"] is True
        and decision["trained_qualification_sha256"] == qualification_sha256
        and decision["selected_scenario"] in scenarios
        and len(decision["rationale"]) >= 20,
        "Fresh explicit engineering decision must cite this actual trained qualification",
    )
    selected = scenarios[decision["selected_scenario"]]
    require(
        math.isclose(decision["selected_estimate_seconds"], selected, rel_tol=1e-9)
        and selected + plan["safety_margin_seconds"] <= 32400,
        "Selected measured scenario plus chosen margin exceeds the actual nine-hour limit",
    )
    return {
        "planning_studies": n,
        "measured_unique_studies": len(by_study),
        "planning_count_basis": plan["planning_count_basis"],
        "scenarios_seconds": scenarios,
        "entry_decision": decision,
        "safety_margin_seconds": plan["safety_margin_seconds"],
        "competition_limit_seconds": 32400,
        "hidden_runtime_proven": False,
        "limitation": "Three public studies do not establish hidden size or volume tails.",
    }


def no_prior_intent(path, row, phase):
    require(
        not path.exists() and phase not in row,
        "Prior intent/result requires read-only reconciliation; mutation retry prohibited",
    )


def request(source, mode, bundle, dataset_slug, docker, qualification_cap):
    require(mode == "qualification" or mode in ARMS, "Exact qualification or trained arm")
    slug = "godlydonuts/rsna-predictive-" + mode.lower() + "-e6-" + sha(source)[:12]
    return {
        "slug": slug,
        "newTitle": slug.split("/")[1].replace("-", " "),
        "text": source,
        "language": "python",
        "kernelType": "script",
        "isPrivate": True,
        "enableGpu": True,
        "machineShape": "NvidiaTeslaT4",
        "enableInternet": False,
        "datasetDataSources": [PARENT_DATASET, dataset_slug],
        "kernelDataSources": [],
        "competitionDataSources": ["rsna-knee-abnormality-detection"],
        "sessionTimeoutSeconds": qualification_cap if mode == "qualification" else 32400,
        "dockerImage": docker,
        "dockerImagePinningType": "original",
    }
