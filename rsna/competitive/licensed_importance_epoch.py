"""Check sampled production state without weakening the original whole94 verifier."""

from __future__ import annotations

from dataclasses import asdict

from rsna.competitive.licensed_importance import VERSION, Recipe
from rsna.competitive.licensed_parent_epoch import verify as verify_complete_state
from rsna.competitive.licensed_pixel_bags import POPULATION_SHA

PROPOSAL_SHA = "e812a9f90cd2b3db92d9acafd28688d5ee6fe16fc188d6517c2bab565f79205e"


def verify(state, snapshot, receipt, *, expected_parameters=73136096):
    binding = state["binding"]
    role = receipt.get("role")
    proposal_tensor = binding.get("teacher_proposal_sha256", "")
    if (
        role not in ("uniform", "influence")
        or binding.get("role") != role
        or binding.get("version") != VERSION
        or binding.get("recipe") != asdict(Recipe(epochs=3, role=role))
        or binding.get("population_sha256") != POPULATION_SHA
        or binding.get("proposal", {}).get("proposal_table")
        != {"bytes": 2923152, "sha256": PROPOSAL_SHA}
        or not isinstance(proposal_tensor, str)
        or len(proposal_tensor) != 64
        or any(c not in "0123456789abcdef" for c in proposal_tensor)
        or binding.get("actual_recovery", {}).get("kernel_id") != 139130771
    ):
        raise ValueError(
            "Exact role, fixed proposal, complete matched recipe and MRI recovery required"
        )
    result = verify_complete_state(
        state,
        snapshot,
        receipt,
        expected_parameters=expected_parameters,
        expected_windows=16,
        expected_status="actual_matched_IID16_full_parent_epoch_session_complete",
    )
    return {**result, "role": role, "sampled_windows": 16, "proposal_sha256": PROPOSAL_SHA}
