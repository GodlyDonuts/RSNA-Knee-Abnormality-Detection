"""Separate strict admission for the single encoder learning-rate treatment."""

from dataclasses import asdict

from rsna.competitive.licensed_importance import VERSION, Recipe
from rsna.competitive.licensed_parent_epoch import verify as complete
from rsna.competitive.licensed_parent_train import lr_factor
from rsna.competitive.licensed_pixel_bags import POPULATION_SHA

PLAN = "licensed_influence_encoderLR3e5_fixed3887_3epoch_v1"
STATUS = "actual_encoderLR3e5_influence_full_parent_epoch_complete"


def verify(state, snapshot, receipt, *, expected_parameters=73136096):
    binding = state["binding"]
    recipe = Recipe(epochs=3, role="influence", encoder_lr=3e-5)
    if (
        binding.get("production_plan") != PLAN
        or binding.get("role") != "influence"
        or receipt.get("role") != "influence"
        or binding.get("version") != VERSION
        or binding.get("recipe") != asdict(recipe)
        or binding.get("population_sha256") != POPULATION_SHA
        or binding.get("single_factor")
        != {"encoder_lr_control": 3e-6, "encoder_lr_treatment": 3e-5}
        or binding.get("proposal", {}).get("proposal_table")
        != {
            "bytes": 2923152,
            "sha256": "e812a9f90cd2b3db92d9acafd28688d5ee6fe16fc188d6517c2bab565f79205e",
        }
        or binding.get("actual_recovery", {}).get("kernel_id") != 139130771
    ):
        raise ValueError("Exact single-factor LR treatment recipe and source binding required")
    for i, row in enumerate(state["history"]):
        expected = 3e-5 * lr_factor(i, 1458, recipe)
        if row["encoder_lr"] != expected or row["head_lr"] != expected:
            raise ValueError("Actual complete treatment learning-rate trajectory required")
    result = complete(
        state,
        snapshot,
        receipt,
        expected_parameters=expected_parameters,
        expected_windows=16,
        expected_status=STATUS,
    )
    return {**result, "role": "influence", "encoder_lr": 3e-5, "head_lr": 3e-5}
