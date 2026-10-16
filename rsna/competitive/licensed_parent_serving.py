"""Validate a fully preserved epoch of the fixed full-parent adaptation."""

from __future__ import annotations

import torch

PARENT = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"
PARENT_SOURCE = "493c7b0d44fe5244b02024448a93f5393cdb073781a068951badb3ffd53fcbde"
POPULATION = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"


def provenance(training, publication, model_sha):
    r = training.get("receipt", {})
    epoch = training.get("segment")
    p = publication.get("independent_CPU_state_verification", {})
    if (
        training.get("status") != "COMPLETE"
        or type(epoch) is not int
        or not 1 <= epoch <= 3
        or r.get("status") != "actual_whole94_full_parent_epoch_session_complete"
        or r.get("segment") != epoch
        or r.get("completed") != epoch * 486
        or r.get("total") != 1458
        or r.get("epoch", {}).get("sha256") != model_sha
        or publication.get("status")
        != "actual_whole94_parent_epoch_private_full_download_size_SHA_verified"
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
        snapshot.get("epoch") != epoch
        or snapshot.get("completed") != epoch * 486
        or not all(binding.get(k) == v for k, v in training["receipt"]["binding"].items())
        or binding.get("production_plan")
        != "licensed_MRI_parent_all94_full3887_fixed3_continuous_v1"
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
        or recipe.get("windows") != 94
        or recipe.get("effective_batch") != 8
        or recipe.get("encoder_lr") != 3e-6
        or recipe.get("head_lr") != 3e-5
        or recipe.get("source_weight") != 0.25
    ):
        raise ValueError("Complete fixed whole94 full3887 production epoch required for serving")
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
        "model_sha256": model_sha,
        "independent_internal_validation": False,
        "clinical_accuracy_claim": False,
    }
