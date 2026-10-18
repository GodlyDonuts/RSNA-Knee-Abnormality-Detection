"""Retain every original whole-state check and bind the actual sampling treatment."""

from rsna.competitive.licensed_importance_epoch import verify as original_verify

RAW = "a2753d69e8a23aef5f26fe3037baad64e93a40c6e6f0c0903f0d50c9b0a61441"
TENSOR = "3581baf78d76d75e1e0514817ca3fba024b0d8febd8ab05e90ba21878478628e"
ROSTER = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"


def verify(state, snapshot, receipt, *, expected_parameters=73136096):
    binding = state["binding"]
    recovery = binding.get("actual_finding_recovery", {})
    if (
        receipt.get("status") != "actual_finding_strata_full_parent_epoch_session_complete"
        or binding.get("production_plan")
        != "finding_stratified12plus4_original_density_full3887_fixed3_v1"
        or binding.get("component_raw_sha256") != RAW
        or binding.get("component_file_sha256") != TENSOR
        or binding.get("component_roster_sha256") != ROSTER
        or binding.get("original_target_sha256")
        != "48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285"
        or "multisource_supervision" in binding
        or "source_label_role" in binding
        or binding.get("finding_allocation")
        != {
            "version": "finding_stratified_12plus4_original_density_v1",
            "allocation": "finding_strata",
            "components_sha256": RAW,
            "mixture_density_unchanged": True,
            "images_per_study": 16,
        }
        or recovery.get("kernel_id") != 139349540
        or recovery.get("version") != 1
        or recovery.get("source_sha256")
        != "7a4e7db6aa40c70e3a64f5faf2a5f11f3e20423725c76acecfa12a59861e74d2"
        or recovery.get("worker_sha256")
        != "ff8c3991f9b4f4382251dcaec2b52e9726d5a95f04d3393deba0788dfa7e0e55"
        or len(recovery.get("receipt_canonical_JSON_sha256", "")) != 64
    ):
        raise ValueError(
            "Exact sampling-only full tensor, original labels and actual MRI recovery required"
        )
    result = original_verify(
        state,
        snapshot,
        {**receipt, "status": "actual_matched_IID16_full_parent_epoch_session_complete"},
        expected_parameters=expected_parameters,
    )
    return {
        **result,
        "finding_allocation_and_original_targets_exact": True,
        "actual_finding_recovery_kernel": 139349540,
    }
