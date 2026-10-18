"""Serve only the fixed-final paired model; no correction enters inference.

Standalone source embedded in offline serving, with independent provenance from
ordinary/LR/finding models. No actual completed model is implied by this code.
"""

from __future__ import annotations

import torch

SOURCE = "37650602851c929afc1c444b87593a4af09d71ec"
WORKER = "d8536a2140123ab160c7770345a1262256815bc07f7bbf3ae7401e5551b18c94"
PLAN = "paired_parent_logit_IID16_full3887_fixed3_continuous_v1"
VERSION = "paired_parent_logit_IID16_training_v1"
STATUS = "actual_paired_parent_logit_full3887_epoch_session_complete"
PUB_STATUS = "actual_paired_epoch_RAM_private_all6_full_download_size_SHA_verified"
PARENT = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
PARENT_SOURCE = "493c7b0d44fe5244b02024448a93f5393cdb073781a068951badb3ffd53fcbde"
POPULATION = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"
PROPOSAL = "e812a9f90cd2b3db92d9acafd28688d5ee6fe16fc188d6517c2bab565f79205e"
TABLE = {
    "bytes": 1629919,
    "sha256": "bfd9d6d67afc7e7f3c9cd091bfccae8c8fc741b23ea1679deb3da122cd0fb619",
}
RECOVERY = {
    "kernel_id": 139417155,
    "source_sha256": "4b129e1c5c76bcc226eff30f30e9e59b25df0ba91358771b795f7d65f68ad6b5",
    "worker_sha256": "f888b4cc327bf1d09adb05376a4ef3c3fa804169a601e7dc7974fb5bb3e24a2e",
    "receipt_sha256": "8a31b2fc04663805484f0d72d50954dcc07c6723ca3a0ea218d8cdae0fe268bb",
}
RECIPE = {
    "epochs": 3,
    "effective_batch": 8,
    "windows": 16,
    "seed": 20261016,
    "encoder_lr": 3e-6,
    "head_lr": 3e-5,
    "weight_decay": 0.02,
    "warmup_fraction": 0.1,
    "final_lr_fraction": 0.01,
    "source_weight": 0.25,
    "clip": 1.0,
    "role": "influence",
}
FILES = {
    "model.pt",
    "optimizer-checkpoint.pt",
    "run.json",
    "source-permission.txt",
    "own-source.json",
    "provenance.json",
}


def source_binding_ok(binding):
    return (
        binding.get("source_git") == SOURCE
        and binding.get("production_plan") == PLAN
        and binding.get("role") == "influence"
        and binding.get("paired_recovery") == RECOVERY
        and binding.get("correction_table") == TABLE
        and binding.get("original_target_sha256")
        == "48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285"
        and binding.get("parent_checkpoint_sha256") == PARENT
        and binding.get("parent_source_sha256") == PARENT_SOURCE
        and binding.get("proposal", {}).get("proposal_table")
        == {"bytes": 2923152, "sha256": PROPOSAL}
        and binding.get("actual_recovery", {}).get("kernel_id") == 139130771
        and binding.get("actual_recovery", {}).get("source_sha256")
        == "a01dd9643355dfb1d2b2f7f43e3cda819e0c44e69a0cd7c2acd99440c0e25554"
        and all(
            n not in binding
            for n in (
                "single_factor",
                "finding_allocation",
                "multisource_supervision",
                "source_label_role",
            )
        )
    )


def provenance(training, publication, model_sha):
    r = training.get("receipt", {})
    p = publication.get("independent_CPU_state_verification", {})
    if (
        training.get("status") != "COMPLETE"
        or training.get("version") != 1
        or type(training.get("kernel_id")) is not int
        or training["kernel_id"] <= 0
        or training.get("epoch") != 3
        or training.get("role") != "influence"
        or r.get("status") != STATUS
        or r.get("segment") != 3
        or r.get("completed") != 1458
        or r.get("total") != 1458
        or r.get("role") != "influence"
        or r.get("fixed_final_complete") is not True
        or r.get("worker_sha256") != WORKER
        or not source_binding_ok(r.get("binding", {}))
        or r.get("epoch", {}).get("sha256") != model_sha
        or publication.get("status") != PUB_STATUS
        or publication.get("epoch") != 3
        or publication.get("completed") != 1458
        or publication.get("objects") != 6
        or publication.get("original_kernel") != training.get("kernel_id")
        or publication.get("original_source_sha256") != training.get("source_sha256")
        or publication.get("full_optimizer_included_private_HF") is not True
        or publication.get("all_HF_cache_and_Xet_paths_explicitly_RAM_before_import") is not True
        or set(publication.get("files", {})) != FILES
        or any(
            publication["files"].get(name) != r.get(key)
            for name, key in (
                ("model.pt", "epoch"),
                ("optimizer-checkpoint.pt", "checkpoint"),
                ("run.json", "run"),
            )
        )
        or p.get("full_learned_graph_exact") is not True
        or p.get("finite_contiguous_updates") != 1458
        or p.get("actual_AdamW_parameters") != 73136096
        or p.get("all_two_rank_RNG_and_scaler_records_present") is not True
        or p.get("exact_original_recipe_and_real_paired_recovery") is not True
        or p.get("paired_correction_sha256") != TABLE["sha256"]
        or p.get("role") != "influence"
        or not isinstance(model_sha, str)
        or len(model_sha) != 64
        or any(c not in "0123456789abcdef" for c in model_sha)
    ):
        raise ValueError(
            "Fixed-final paired fit and all six independently verified private objects required"
        )
    return 3


def validate(snapshot, training, publication, model_sha):
    provenance(training, publication, model_sha)
    b = snapshot.get("binding", {})
    if (
        not source_binding_ok(b)
        or b.get("version") != VERSION
        or b.get("recipe") != RECIPE
        or snapshot.get("epoch") != 3
        or snapshot.get("completed") != 1458
        or not all(b.get(k) == value for k, value in training["receipt"]["binding"].items())
        or b.get("parent_correction_sha256") != TABLE["sha256"]
        or b.get("population_sha256") != POPULATION
        or b.get("population") != 3887
        or b.get("world_size") != 2
        or b.get("total_steps") != 1458
        or b.get("diagnostic_only") is not False
        or b.get("protected_gold_confirmation_used") is not False
        or b.get("independent_internal_validation") is not False
    ):
        raise ValueError("Complete actual paired recipe/runtime binding required")
    model = snapshot.get("model", {})
    if (
        not model
        or not any(n.startswith("backbone.") for n in model)
        or any(
            not isinstance(t, torch.Tensor) or not torch.isfinite(t).all() for t in model.values()
        )
        or any(
            n not in model or model[n].shape != shape
            for n, shape in (
                ("clsW", (12, 1024)),
                ("clsb", (12,)),
                ("norm.weight", (1024,)),
                ("norm.bias", (1024,)),
            )
        )
    ):
        raise ValueError("Complete original learned encoder and twelve-target MIL graph required")
    return {
        "epoch": 3,
        "completed": 1458,
        "full_population": 3887,
        "role": "influence",
        "training_windows": 16,
        "serving_windows": 94,
        "model_sha256": model_sha,
        "proposal_sha256": PROPOSAL,
        "fixed_final_only": True,
        "parent_logit_correction_applied_in_inference": False,
        "independent_internal_validation": False,
        "clinical_accuracy_claim": False,
    }
