"""Fixed-final deployment provenance for the original paired pointwise/rank fit.

No numerical inference or training change. Actual new-model T4 acceptance remains
separate. A preserved pointwise snapshot may qualify while the original job
continues the later ranking arm; never require/restart the whole job for that.
"""

import hashlib
import json

import torch

SCIENCE = "7f75fd46b5d0e772fd2767400e81d70b593619b4"
GPU_JOB = "975297"
WORKER_SHA = "ffc1fbf914b9e1a97476ac40b3d4cc0c58b7948c45061079a497f1135c2df461"
EXPECTED_BINDING = {
    "BN_running_statistics": "frozen_at_accepted_parent",
    "anatomical_views_per_study": 2,
    "device_name": "NVIDIA H100 PCIe",
    "device_type": "cuda",
    "diagnostic_only": False,
    "fitting_only": True,
    "initial_state_sha256": "a75cb18d1dec13c4fb2f547f017ed92e81bf768f580cee54da0ffa88b712fc78",
    "matched_two_view_three_epoch_comparison": True,
    "parent_checkpoint_sha256": "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef",
    "population": 3887,
    "population_sha256": "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42",
    "precision": "FP16_autocast_FP32_loss_dynamic_scaler",
    "protected_or_gold_used": False,
    "recipe": {
        "clip": 1.0,
        "effective_batch": 8,
        "encoder_lr": 3e-06,
        "epochs": 3,
        "final_lr_fraction": 0.01,
        "head_lr": 3e-05,
        "objective": "pointwise",
        "rank_fraction": 0.5,
        "role": "influence",
        "seed": 20261016,
        "source_weight": 0.25,
        "warmup_fraction": 0.1,
        "weight_decay": 0.02,
        "windows": 16,
    },
    "source_binding": {
        "CPU_full_population_membership_and_six_both_view_input_checks": True,
        "accepted_numerical_source_git": "ecab19d845054848160a7eddad6a1de3426f7c8f",
        "actual_GPU_recovery": "974940",
        "actual_GPU_recovery_receipt_sha256": "1bc918297634a5bc75e3c789cad02fff278a3a8771f54c4d5b40b9f1ea37cb97",
        "faster_reader_used": False,
        "original_pixel_copy_bytes": 38873662650,
        "production_wrapper_source_git": "7f75fd46b5d0e772fd2767400e81d70b593619b4",
        "protected_or_gold_used": False,
        "shared_initial_state_sha256": "a75cb18d1dec13c4fb2f547f017ed92e81bf768f580cee54da0ffa88b712fc78",
        "source_permission_sha256": "34c44b20e4d7ace4eb523b0423b36d36dec826fe6ae54c66346796bba0190e79",
    },
    "study_coupled_batch_loss": True,
    "teacher_proposal_sha256": "2d7515536e378b2775cbacbf4e296713e358ec8f70aba8cc9212a952bed70a77",
    "torch_version": "2.14.0+cu130",
    "total_steps": 1458,
    "version": "full_encoder_two_view_teacher_pair_risk_v1",
    "world_size": 1,
}


def provenance(training, publication, model_sha):
    role = training.get("role")
    expected = {**EXPECTED_BINDING, "recipe": {**EXPECTED_BINDING["recipe"], "objective": role}}
    r = training.get("receipt", {})
    p = publication.get("provenance", {})
    if (
        role not in ("pointwise", "teacher_rank")
        or training.get("status") != "COMPLETE"
        or type(training.get("epoch")) is not int
        or training["epoch"] != 3
        or training.get("kernel_id") is not None
        or training.get("training_Slurm_job") != GPU_JOB
        or training.get("source_sha256") != WORKER_SHA
        or r.get("status") != "actual_teacher_pair_fixed_final_model_for_serving"
        or r.get("completed") != 1458
        or r.get("binding") != expected
        or publication.get("status")
        != "actual_teacher_pair_immutable_epoch_private_all_objects_download_size_SHA_verified"
        or publication.get("GPU_job") != GPU_JOB
        or publication.get("source_git") != SCIENCE
        or publication.get("role") != role
        or publication.get("epoch") != 3
        or publication.get("completed_updates") != 1458
        or publication.get("objects") != 6
        or publication.get("canonical_shared_publication_lock_used") is not True
        or publication.get("prefix")
        != "checkpoints/teacher-pair-full3887-" + SCIENCE[:12] + "/" + str(role) + "/epoch-003"
        or publication.get("model_sha256") != model_sha
        or not isinstance(model_sha, str)
        or len(model_sha) != 64
        or any(c not in "0123456789abcdef" for c in model_sha)
        or r.get("epoch") != {"bytes": publication.get("model_bytes"), "sha256": model_sha}
        or type(publication.get("model_bytes")) is not int
        or not 0 < publication["model_bytes"] < 330 * 2**20
        or p.get("scientific_source_git") != SCIENCE
        or p.get("GPU_job") != GPU_JOB
        or p.get("role") != role
        or p.get("epoch") != 3
        or p.get("completed_updates") != 1458
        or p.get("binding") != expected
        or p.get("snapshot_sha256") != model_sha
        or p.get("trainable_parameters_from_actual_accepted_trainer") != 73136096
        or p.get("source_permission_sha256")
        != expected["source_binding"]["source_permission_sha256"]
        or p.get("original_MRI_labels_UIDs_private_arrays_included") is not False
    ):
        raise ValueError(
            "Actual fixed-final objective, original full source binding and six-object private proof required"
        )
    return 3


def validate(snapshot, training, publication, model_sha):
    provenance(training, publication, model_sha)
    model = snapshot.get("model", {})
    if (
        set(snapshot) != {"binding", "epoch", "completed", "model"}
        or snapshot["binding"] != training["receipt"]["binding"]
        or snapshot["epoch"] != 3
        or snapshot["completed"] != 1458
        or not model
        or not any(n.startswith("backbone.") for n in model)
        or any(
            not isinstance(t, torch.Tensor) or not torch.isfinite(t).all() for t in model.values()
        )
    ):
        raise ValueError("Complete finite final full-encoder snapshot required")
    schema = {n: {"shape": list(t.shape), "dtype": str(t.dtype)} for n, t in model.items()}
    if hashlib.sha256(json.dumps(schema, sort_keys=True).encode()).hexdigest() != publication[
        "provenance"
    ].get("model_schema_sha256"):
        raise ValueError("Exact complete privately verified learned graph schema required")
    return {
        "epoch": 3,
        "completed": 1458,
        "role": training["role"],
        "full_population": 3887,
        "training_windows": 16,
        "serving_windows": 94,
        "training_Slurm_job": GPU_JOB,
        "training_world_size": 1,
        "anatomical_views_per_study": 2,
        "model_sha256": model_sha,
        "actual_new_T4_acceptance": False,
        "clinical_accuracy_claim": False,
    }
