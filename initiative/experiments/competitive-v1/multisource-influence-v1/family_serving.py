"""Serve only the fully preserved, fixed-final source-label contrast.

The learned graph and the original serving checks remain unchanged. Only the
explicit label-arm metadata tags are adapted for delegation to those checks.
"""

from __future__ import annotations

from rsna.competitive.licensed_importance_serving import provenance as base_provenance
from rsna.competitive.licensed_importance_serving import validate as base_validate

PLAN = "licensed_MRI_parent_family_mean_influence16_full3887_fixed3_v1"
BASE_PLAN = "licensed_MRI_parent_uniform_influence_IID16_full3887_fixed3_continuous_v1"
SOURCE_GIT = "11ce568c63bcc6e984fc9c16ee92e58a8cc27eb3"
TARGET = "65d3730702a098c8c819396dba6122902a4f3720dde9b2cb3f9ffd7ac5eca3b5"
RECOVERY = {
    "kernel_id": 139336728,
    "version": 1,
    "source_sha256": "3fd77316770abec5b05a6297e6f3a38c73458b86f84cb2d6b6763ecd37c6f825",
    "worker_sha256": "acfa946919bdf1d9fbff3b34c196d4e5c4545117bdadb9ff68da419afbfdd5d5",
    "receipt_canonical_JSON_sha256": "2d62a8bcd98101c901f340c418a6b96374c17f671d6275f54c85b177c0a1c82f",
}
LABELS = {
    "version": "fitting_safe_common_family_mean_v1",
    "role": "safe_family_mean",
    "population": 3887,
    "population_sha256": "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42",
    "diagnostic_only": False,
    "parent_checkpoint_sha256": "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef",
    "original_target_file_sha256": "48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285",
    "target_file_sha256": "65d3730702a098c8c819396dba6122902a4f3720dde9b2cb3f9ffd7ac5eca3b5",
    "annotation_file_sha256": "54f1c94943bd7049b3e3da79fdcffb1f7083ecb373a64b9e2642aa2fe9230df9",
    "original_labels_sha256": "8ce962aeef9b4ab5e0de65145e875e9caffeddc8fddd66065692f21ca9637444",
    "labels_sha256": "f88ee60e27b1673f3fa7934107c91cc8aef2e810b7cedd34aa725f6470571ddd",
    "mask_sha256": "e80fba0e4d304d013b49895ebe3d3201972a0afd158e95fc296b8c222685f9a1",
    "teacher_sha256": "d9853c7745161807705087bef505f1ceb488f2138264dea3902388662fe37566",
    "eligible_sha256": "26a7256674684260d97fb06e48f02a92c718650359303e7025dedec76076bd25",
}


def _same_recovery(actual):
    return (
        isinstance(actual, dict)
        and set(actual) == set(RECOVERY)
        and all(type(actual[k]) is type(v) and actual[k] == v for k, v in RECOVERY.items())
    )


def _base_records(training, publication):
    receipt = training.get("receipt", {})
    binding = receipt.get("binding", {})
    proof = publication.get("independent_CPU_state_verification", {})
    if (
        receipt.get("status") != "actual_family_influence16_full_parent_epoch_session_complete"
        or training.get("epoch") != 3
        or training.get("role") != "influence"
        or receipt.get("fixed_final_complete") is not True
        or binding.get("production_plan") != PLAN
        or binding.get("source_git") != SOURCE_GIT
        or binding.get("source_label_role") != "safe_family_mean"
        or binding.get("source_label_target_sha256") != TARGET
        or binding.get("annotation_file_sha256") != LABELS["annotation_file_sha256"]
        or not _same_recovery(binding.get("actual_family_recovery"))
        or publication.get("status")
        != "actual_family_influence_epoch_private_full_download_size_SHA_verified"
        or proof.get("actual_family_recovery_kernel") != RECOVERY["kernel_id"]
        or proof.get("actual_family_labels_and_original_teacher_mask_binding_exact") is not True
    ):
        raise ValueError(
            "Actual fixed-final label-only recovery, training and private proof required"
        )
    # Preserve the actual records. Delegate only the two dispatch type tags and
    # the plan extension; learned tensors and every numerical recipe field stay exact.
    normalized_training = {
        **training,
        "receipt": {
            **receipt,
            "status": "actual_matched_IID16_full_parent_epoch_session_complete",
            "binding": {**binding, "production_plan": BASE_PLAN},
        },
    }
    normalized_publication = {
        **publication,
        "status": "actual_matched_IID16_parent_epoch_private_full_download_size_SHA_verified",
    }
    return normalized_training, normalized_publication


def provenance(training, publication, model_sha):
    normalized_training, normalized_publication = _base_records(training, publication)
    return base_provenance(normalized_training, normalized_publication, model_sha)


def validate(snapshot, training, publication, model_sha):
    normalized_training, normalized_publication = _base_records(training, publication)
    binding = snapshot.get("binding", {})
    label_binding = binding.get("multisource_supervision", {})
    if (
        binding.get("production_plan") != PLAN
        or not _same_recovery(binding.get("actual_family_recovery"))
        or set(label_binding) != set(LABELS)
        or any(
            type(label_binding.get(k)) is not type(v) or label_binding[k] != v
            for k, v in LABELS.items()
        )
        or label_binding.get("original_labels_sha256") == label_binding.get("labels_sha256")
    ):
        raise ValueError(
            "Exact new label arrays, original mask/teacher and actual recovery required"
        )
    normalized_snapshot = {
        **snapshot,
        "binding": {**binding, "production_plan": BASE_PLAN},
    }
    checked = base_validate(
        normalized_snapshot, normalized_training, normalized_publication, model_sha
    )
    return {
        **checked,
        "source_label_role": "safe_family_mean",
        "source_label_target_sha256": TARGET,
        "actual_family_recovery_kernel": RECOVERY["kernel_id"],
        "original_full_learned_graph_checks_retained": True,
    }
