"""Serve only the fully preserved, fixed-final source-label contrast.

The learned graph and the original serving checks remain unchanged. Only the
explicit label-arm metadata tags are adapted for delegation to those checks.
"""

from __future__ import annotations

import hashlib

from rsna.competitive.licensed_importance_serving import provenance as base_provenance
from rsna.competitive.licensed_importance_serving import validate as base_validate

PLAN = "licensed_MRI_parent_S6_label_only_uniform16_full3887_fixed3_v1"
BASE_PLAN = "licensed_MRI_parent_uniform_influence_IID16_full3887_fixed3_continuous_v1"
SOURCE_GIT = "990c8ded67d8158fc57fb1ce6c1a2a8bff9a7612"
TARGET = "5e34bbf40f4f5b21638d2a1dad853912dee98533964a5236a8f5328bd3d388a9"
RECOVERY = {
    "kernel_id": 139230981,
    "version": 1,
    "source_sha256": "0a6f6c5679011625f3fd0098aa8a6b5c37120de7e02a0f5d9fe09d4ee975b9de",
    "worker_sha256": "7b7a1bf3e5a8860a68067d023b2c63b39fc09169b5c626ca769f13b6e238037c",
    "receipt_canonical_JSON_sha256": "e9c5d82a82135adf764681c3214aba25127b614dfde742be54e3840a2a051b87",
}
LABELS = {
    "version": "fixed_S6_source_label_only_overlay_v1",
    "role": "released_S6_source",
    "population": 3887,
    "population_sha256": "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42",
    "diagnostic_only": False,
    "parent_checkpoint_sha256": "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef",
    "original_target_file_sha256": "48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285",
    "target_file_sha256": TARGET,
    "released_S6_raw_source_sha256": "8006cc39d1f564d4031db9692c136a73dfe49a7be1a2e2e8ec8a19fbd8f2fcc6",
    "original_label_array_sha256": "8ce962aeef9b4ab5e0de65145e875e9caffeddc8fddd66065692f21ca9637444",
    "new_label_array_sha256": "5a4bcc95fc1e5c841834781243fc00a390da1f20d3feed055a64065b6b222fac",
    "mask_array_sha256": hashlib.sha256(bytes([1]) * (3887 * 12)).hexdigest(),
    "parent_teacher_array_sha256": "d9853c7745161807705087bef505f1ceb488f2138264dea3902388662fe37566",
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
        receipt.get("status") != "actual_S6_label_only_uniform16_full_parent_epoch_session_complete"
        or training.get("epoch") != 3
        or training.get("role") != "uniform"
        or receipt.get("fixed_final_complete") is not True
        or binding.get("production_plan") != PLAN
        or binding.get("source_git") != SOURCE_GIT
        or binding.get("source_label_role") != "released_S6_source"
        or binding.get("source_label_target_sha256") != TARGET
        or not _same_recovery(binding.get("actual_label_recovery"))
        or publication.get("status")
        != "actual_S6_label_epoch_private_full_download_size_SHA_verified"
        or proof.get("label_role") != "released_S6_source"
        or proof.get("actual_label_recovery_kernel") != RECOVERY["kernel_id"]
        or proof.get("actual_S6_label_bytes_and_original_teacher_mask_binding_exact") is not True
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
    label_binding = binding.get("source_label_contrast", {})
    if (
        binding.get("production_plan") != PLAN
        or not _same_recovery(binding.get("actual_label_recovery"))
        or any(
            type(label_binding.get(k)) is not type(v) or label_binding[k] != v
            for k, v in LABELS.items()
        )
        or label_binding.get("original_label_array_sha256")
        == label_binding.get("new_label_array_sha256")
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
        "source_label_role": "released_S6_source",
        "source_label_target_sha256": TARGET,
        "actual_label_recovery_kernel": RECOVERY["kernel_id"],
        "original_full_learned_graph_checks_retained": True,
    }
