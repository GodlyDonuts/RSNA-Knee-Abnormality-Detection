"""Independent complete-state admission for the matched paired-logit fit."""

from dataclasses import asdict

from rsna.competitive.licensed_importance import Recipe
from rsna.competitive.licensed_parent_epoch import verify as complete
from rsna.competitive.licensed_parent_train import lr_factor
from rsna.competitive.licensed_pixel_bags import POPULATION_SHA

PLAN = "paired_parent_logit_IID16_full3887_fixed3_continuous_v1"
STATUS = "actual_paired_parent_logit_full3887_epoch_session_complete"
VERSION = "paired_parent_logit_IID16_training_v1"
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


def verify(state, snapshot, receipt, *, expected_parameters=73136096):
    binding = state["binding"]
    recipe = Recipe(role="influence")
    if (
        binding.get("production_plan") != PLAN
        or binding.get("version") != VERSION
        or binding.get("recipe") != asdict(recipe)
        or binding.get("role") != "influence"
        or receipt.get("role") != "influence"
        or binding.get("population_sha256") != POPULATION_SHA
        or binding.get("paired_recovery") != RECOVERY
        or binding.get("correction_table") != TABLE
        or binding.get("parent_correction_sha256") != TABLE["sha256"]
        or binding.get("actual_recovery", {}).get("kernel_id") != 139130771
        or binding.get("proposal", {}).get("proposal_table")
        != {
            "bytes": 2923152,
            "sha256": "e812a9f90cd2b3db92d9acafd28688d5ee6fe16fc188d6517c2bab565f79205e",
        }
    ):
        raise ValueError(
            "Exact paired recipe, original population, correction and real recovery required"
        )
    for i, row in enumerate(state["history"]):
        factor = lr_factor(i, 1458, recipe)
        if (
            row["encoder_lr"] != recipe.encoder_lr * factor
            or row["head_lr"] != recipe.head_lr * factor
        ):
            raise ValueError(
                "Complete unchanged original encoder/head learning-rate trajectory required"
            )
    result = complete(
        state,
        snapshot,
        receipt,
        expected_parameters=expected_parameters,
        expected_windows=16,
        expected_status=STATUS,
    )
    return {
        **result,
        "role": "influence",
        "paired_correction_sha256": TABLE["sha256"],
        "exact_original_recipe_and_real_paired_recovery": True,
    }
