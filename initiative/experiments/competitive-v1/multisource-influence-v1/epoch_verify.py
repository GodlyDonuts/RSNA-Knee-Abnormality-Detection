"""Keep all original whole-state checks and bind the exact family label policy."""

from rsna.competitive.licensed_importance_epoch import verify as original_verify

TARGET = "65d3730702a098c8c819396dba6122902a4f3720dde9b2cb3f9ffd7ac5eca3b5"
ANNOTATIONS = "54f1c94943bd7049b3e3da79fdcffb1f7083ecb373a64b9e2642aa2fe9230df9"


def verify(state, snapshot, receipt, *, expected_parameters=73136096):
    binding = state["binding"]
    labels = binding.get("multisource_supervision", {})
    recovery = binding.get("actual_family_recovery", {})
    expected_labels = {
        "version": "fitting_safe_common_family_mean_v1",
        "role": "safe_family_mean",
        "diagnostic_only": False,
        "population": 3887,
        "population_sha256": "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42",
        "parent_checkpoint_sha256": "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef",
        "original_target_file_sha256": "48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285",
        "original_labels_sha256": "8ce962aeef9b4ab5e0de65145e875e9caffeddc8fddd66065692f21ca9637444",
        "labels_sha256": "f88ee60e27b1673f3fa7934107c91cc8aef2e810b7cedd34aa725f6470571ddd",
        "mask_sha256": "e80fba0e4d304d013b49895ebe3d3201972a0afd158e95fc296b8c222685f9a1",
        "teacher_sha256": "d9853c7745161807705087bef505f1ceb488f2138264dea3902388662fe37566",
        "target_file_sha256": TARGET,
        "annotation_file_sha256": ANNOTATIONS,
    }
    if (
        receipt.get("status") != "actual_family_influence16_full_parent_epoch_session_complete"
        or receipt.get("role") != "influence"
        or binding.get("role") != "influence"
        or binding.get("production_plan")
        != "licensed_MRI_parent_family_mean_influence16_full3887_fixed3_v1"
        or binding.get("source_label_role") != "safe_family_mean"
        or binding.get("source_label_target_sha256") != TARGET
        or binding.get("annotation_file_sha256") != ANNOTATIONS
        or any(labels.get(k) != value for k, value in expected_labels.items())
        or set(labels) != set(expected_labels) | {"eligible_sha256"}
        or len(labels.get("eligible_sha256", "")) != 64
        or recovery.get("kernel_id") != 139336728
        or recovery.get("version") != 1
        or recovery.get("source_sha256")
        != "3fd77316770abec5b05a6297e6f3a38c73458b86f84cb2d6b6763ecd37c6f825"
        or recovery.get("worker_sha256")
        != "acfa946919bdf1d9fbff3b34c196d4e5c4545117bdadb9ff68da419afbfdd5d5"
        or len(recovery.get("receipt_canonical_JSON_sha256", "")) != 64
    ):
        raise ValueError("Exact complete family target/mask/teacher and actual recovery required")
    # Adapt the receipt's new type tag only. No numerical state is normalized.
    result = original_verify(
        state,
        snapshot,
        {**receipt, "status": "actual_matched_IID16_full_parent_epoch_session_complete"},
        expected_parameters=expected_parameters,
    )
    return {
        **result,
        "actual_family_labels_and_original_teacher_mask_binding_exact": True,
        "actual_family_recovery_kernel": 139336728,
    }
