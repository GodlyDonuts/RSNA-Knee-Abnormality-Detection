"""Serve only the fixed-final single encoder-LR treatment through full94 inference."""

from __future__ import annotations

import torch

PARENT = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
PARENT_SOURCE = "493c7b0d44fe5244b02024448a93f5393cdb073781a068951badb3ffd53fcbde"
POPULATION = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"
PROPOSAL = "e812a9f90cd2b3db92d9acafd28688d5ee6fe16fc188d6517c2bab565f79205e"
VERSION = "licensed_full_encoder_uniform_or_teacher_influence_IID16_v1"


SOURCE = "d6af91945483ae4d348504069e4f8ee3b26b6e8d"
WORKER = "b0ce22dbdb38da9a128eda82195590b1fc1e9dcb4fb0e39e150e7327612dbcef"
RECIPE = {
    "epochs": 3,
    "effective_batch": 8,
    "windows": 16,
    "seed": 20261016,
    "encoder_lr": 3e-5,
    "head_lr": 3e-5,
    "weight_decay": 0.02,
    "warmup_fraction": 0.1,
    "final_lr_fraction": 0.01,
    "source_weight": 0.25,
    "clip": 1.0,
    "role": "influence",
}


def source_binding_ok(binding):
    return (
        binding.get("source_git") == SOURCE
        and binding.get("production_plan") == "licensed_influence_encoderLR3e5_fixed3887_3epoch_v1"
        and binding.get("single_factor")
        == {"encoder_lr_control": 3e-6, "encoder_lr_treatment": 3e-5}
        and binding.get("actual_new_LR_recovery_claim") is False
        and binding.get("original_target_sha256")
        == "48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285"
        and "finding_allocation" not in binding
        and "multisource_supervision" not in binding
        and "source_label_role" not in binding
    )


def provenance(training, publication, model_sha):
    r = training.get("receipt", {})
    epoch = training.get("epoch")
    role = training.get("role")
    p = publication.get("independent_CPU_state_verification", {})
    if (
        training.get("status") != "COMPLETE"
        or type(epoch) is not int
        or epoch != 3
        or r.get("status") != "actual_encoderLR3e5_influence_full_parent_epoch_complete"
        or r.get("segment") != epoch
        or r.get("fixed_final_complete") is not True
        or publication.get("objects") != 5
        or r.get("worker_sha256") != WORKER
        or not source_binding_ok(r.get("binding", {}))
        or role != "influence"
        or r.get("role") != role
        or publication.get("role") != role
        or p.get("role") != role
        or p.get("encoder_lr") != 3e-5
        or p.get("head_lr") != 3e-5
        or r.get("completed") != epoch * 486
        or r.get("total") != 1458
        or r.get("epoch", {}).get("sha256") != model_sha
        or publication.get("status")
        != "actual_encoderLR3e5_epoch_private_full_download_size_SHA_verified"
        or publication.get("epoch") != epoch
        or publication.get("completed") != epoch * 486
        or publication.get("original_Kaggle_kernel") != training.get("kernel_id")
        or publication.get("original_source_sha256") != training.get("source_sha256")
        or publication.get("source_git") != r.get("binding", {}).get("source_git")
        or publication.get("files", {}).get("model.pt") != r.get("epoch")
        or p.get("full_learned_graph_exact") is not True
        or p.get("finite_contiguous_updates") != epoch * 486
        or p.get("actual_AdamW_parameters") != 73136096
        or p.get("all_two_rank_RNG_and_scaler_records_present") is not True
        or not isinstance(model_sha, str)
        or len(model_sha) != 64
        or any(c not in "0123456789abcdef" for c in model_sha)
    ):
        raise ValueError("Actual full epoch and complete privately verified learned model required")
    return epoch


def validate(snapshot, training, publication, model_sha):
    epoch = provenance(training, publication, model_sha)
    binding = snapshot.get("binding", {})
    recipe = binding.get("recipe", {})
    if (
        not source_binding_ok(binding)
        or recipe != RECIPE
        or snapshot.get("epoch") != epoch
        or snapshot.get("completed") != epoch * 486
        or not all(binding.get(k) == v for k, v in training["receipt"]["binding"].items())
        or binding.get("production_plan") != "licensed_influence_encoderLR3e5_fixed3887_3epoch_v1"
        or binding.get("parent_checkpoint_sha256") != PARENT
        or binding.get("parent_source_sha256") != PARENT_SOURCE
        or binding.get("population_sha256") != POPULATION
        or binding.get("population") != 3887
        or binding.get("world_size") != 2
        or binding.get("total_steps") != 1458
        or binding.get("diagnostic_only") is not False
        or binding.get("protected_gold_confirmation_used") is not False
        or binding.get("independent_internal_validation") is not False
        or recipe.get("epochs") != 3
        or recipe.get("windows") != 16
        or binding.get("role") != training["role"]
        or recipe.get("role") != training["role"]
        or binding.get("version") != VERSION
        or binding.get("proposal", {}).get("proposal_table")
        != {"bytes": 2923152, "sha256": PROPOSAL}
        or binding.get("actual_recovery", {}).get("kernel_id") != 139130771
        or binding.get("actual_recovery", {}).get("source_sha256")
        != "a01dd9643355dfb1d2b2f7f43e3cda819e0c44e69a0cd7c2acd99440c0e25554"
        or recipe.get("effective_batch") != 8
        or recipe.get("encoder_lr") != 3e-5
        or recipe.get("head_lr") != 3e-5
        or recipe.get("source_weight") != 0.25
    ):
        raise ValueError(
            "Complete fixed matched IID16 full3887 production epoch required for serving"
        )
    model = snapshot.get("model", {})
    if (
        not model
        or not any(n.startswith("backbone.") for n in model)
        or any(
            not isinstance(v, torch.Tensor) or not torch.isfinite(v).all() for v in model.values()
        )
        or any(
            name not in model or model[name].shape != shape
            for name, shape in (
                ("clsW", (12, 1024)),
                ("clsb", (12,)),
                ("norm.weight", (1024,)),
                ("norm.bias", (1024,)),
            )
        )
    ):
        raise ValueError("Complete learned encoder and finding-specific MIL model required")
    return {
        "epoch": epoch,
        "completed": epoch * 486,
        "full_population": 3887,
        "role": training["role"],
        "training_windows": 16,
        "serving_windows": 94,
        "proposal_sha256": PROPOSAL,
        "model_sha256": model_sha,
        "independent_internal_validation": False,
        "clinical_accuracy_claim": False,
        "encoder_lr": 3e-5,
        "head_lr": 3e-5,
        "fixed_final_only": True,
    }
