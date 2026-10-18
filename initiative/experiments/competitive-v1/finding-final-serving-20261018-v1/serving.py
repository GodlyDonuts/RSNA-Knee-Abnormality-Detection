"""Accept only the actual preserved fixed-final sampling-only treatment."""

from __future__ import annotations

from rsna.competitive.licensed_importance_serving import provenance as base_provenance
from rsna.competitive.licensed_importance_serving import validate as base_validate

PLAN = "finding_stratified12plus4_original_density_full3887_fixed3_v1"
BASE_PLAN = "licensed_MRI_parent_uniform_influence_IID16_full3887_fixed3_continuous_v1"
SOURCE = "bf362cdd34dd6cdc375e2867a6a2eab44731ad66"
RAW = "a2753d69e8a23aef5f26fe3037baad64e93a40c6e6f0c0903f0d50c9b0a61441"
COMPONENT = "3581baf78d76d75e1e0514817ca3fba024b0d8febd8ab05e90ba21878478628e"
ROSTER = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"
TARGET = "48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285"
ALLOCATION = {
    "version": "finding_stratified_12plus4_original_density_v1",
    "allocation": "finding_strata",
    "components_sha256": RAW,
    "mixture_density_unchanged": True,
    "images_per_study": 16,
}
RECOVERY = {
    "kernel_id": 139349540,
    "version": 1,
    "source_sha256": "7a4e7db6aa40c70e3a64f5faf2a5f11f3e20423725c76acecfa12a59861e74d2",
    "worker_sha256": "ff8c3991f9b4f4382251dcaec2b52e9726d5a95f04d3393deba0788dfa7e0e55",
    "receipt_canonical_JSON_sha256": "3f65a45e48c02db56fbde5d8252a8e34570b19b9bd70f379ee1b7a61e1bf9277",
}


def exact(actual, expected):
    return (
        isinstance(actual, dict)
        and set(actual) == set(expected)
        and all(type(actual[k]) is type(v) and actual[k] == v for k, v in expected.items())
    )


def binding_ok(binding):
    return (
        binding.get("production_plan") == PLAN
        and binding.get("source_git") == SOURCE
        and binding.get("component_raw_sha256") == RAW
        and binding.get("component_file_sha256") == COMPONENT
        and binding.get("component_roster_sha256") == ROSTER
        and binding.get("original_target_sha256") == TARGET
        and exact(binding.get("actual_finding_recovery"), RECOVERY)
        and "multisource_supervision" not in binding
        and "source_label_role" not in binding
    )


def normalized(training, publication):
    receipt = training.get("receipt", {})
    binding = receipt.get("binding", {})
    proof = publication.get("independent_CPU_state_verification", {})
    files = publication.get("files", {})
    if (
        training.get("epoch") != 3
        or training.get("role") != "influence"
        or receipt.get("fixed_final_complete") is not True
        or receipt.get("status") != "actual_finding_strata_full_parent_epoch_session_complete"
        or not binding_ok(binding)
        or publication.get("status")
        != "actual_finding_strata_epoch_private_full_download_size_SHA_verified"
        or publication.get("objects") != 6
        or publication.get("full_optimizer_privately_verified") is not True
        or files.get("optimizer-checkpoint.pt") != receipt.get("checkpoint")
        or not isinstance(receipt.get("checkpoint"), dict)
        or proof.get("finding_allocation_and_original_targets_exact") is not True
        or proof.get("actual_finding_recovery_kernel") != 139349540
    ):
        raise ValueError(
            "Actual final sampling-only model and six-object optimizer backup required"
        )
    return (
        {
            **training,
            "receipt": {
                **receipt,
                "status": "actual_matched_IID16_full_parent_epoch_session_complete",
                "binding": {**binding, "production_plan": BASE_PLAN},
            },
        },
        {
            **publication,
            "status": "actual_matched_IID16_parent_epoch_private_full_download_size_SHA_verified",
        },
    )


def provenance(training, publication, model_sha):
    return base_provenance(*normalized(training, publication), model_sha)


def validate(snapshot, training, publication, model_sha):
    records = normalized(training, publication)
    if not binding_ok(snapshot.get("binding", {})) or not exact(
        snapshot.get("binding", {}).get("finding_allocation"), ALLOCATION
    ):
        raise ValueError("Exact original-target sampling treatment binding required")
    result = base_validate(
        {**snapshot, "binding": {**snapshot["binding"], "production_plan": BASE_PLAN}},
        *records,
        model_sha,
    )
    return {
        **result,
        "finding_allocation": ALLOCATION,
        "actual_finding_recovery_kernel": 139349540,
        "fixed_final_only": True,
    }
