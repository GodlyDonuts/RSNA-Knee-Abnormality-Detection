"""Synthetic numerical completeness and treatment-contamination refusals."""

import copy
import runpy
from dataclasses import asdict

import epoch_verify as e
import pytest
import torch

from rsna.competitive.licensed_importance import VERSION, Recipe
from rsna.competitive.licensed_importance_epoch import PROPOSAL_SHA


def fixture():
    state, snapshot, receipt = runpy.run_path("tests/test_licensed_parent_epoch.py")[
        "completed"
    ].__wrapped__()
    state["binding"].update(
        role="influence",
        version=VERSION,
        recipe=asdict(Recipe(role="influence")),
        population_sha256=e.ROSTER,
        proposal={"proposal_table": {"bytes": 2923152, "sha256": PROPOSAL_SHA}},
        teacher_proposal_sha256="a" * 64,
        actual_recovery={"kernel_id": 139130771},
        production_plan="finding_stratified12plus4_original_density_full3887_fixed3_v1",
        component_raw_sha256=e.RAW,
        component_file_sha256=e.TENSOR,
        component_roster_sha256=e.ROSTER,
        original_target_sha256="48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285",
        finding_allocation={
            "version": "finding_stratified_12plus4_original_density_v1",
            "allocation": "finding_strata",
            "components_sha256": e.RAW,
            "mixture_density_unchanged": True,
            "images_per_study": 16,
        },
        actual_finding_recovery={
            "kernel_id": 139349540,
            "version": 1,
            "source_sha256": "7a4e7db6aa40c70e3a64f5faf2a5f11f3e20423725c76acecfa12a59861e74d2",
            "worker_sha256": "ff8c3991f9b4f4382251dcaec2b52e9726d5a95f04d3393deba0788dfa7e0e55",
            "receipt_canonical_JSON_sha256": "a" * 64,
        },
    )
    receipt.update(
        status="actual_finding_strata_full_parent_epoch_session_complete", role="influence"
    )
    return state, snapshot, receipt


def test_complete_numerical_state():
    state, snapshot, receipt = fixture()
    assert e.verify(state, snapshot, receipt, expected_parameters=3)["actual_AdamW_parameters"] == 3


@pytest.mark.parametrize(
    "damage",
    [
        "density",
        "allocation",
        "tensor",
        "label_overlay",
        "labels",
        "recovery",
        "optimizer",
        "snapshot",
        "history",
        "rng",
        "last_batch",
        "status",
    ],
)
def test_no_scientific_or_numerical_shortcuts(damage):
    state, snapshot, receipt = copy.deepcopy(fixture())
    if damage == "density":
        state["binding"]["finding_allocation"]["mixture_density_unchanged"] = False
    elif damage == "allocation":
        state["binding"]["finding_allocation"]["allocation"] = "iid"
    elif damage == "tensor":
        state["binding"]["component_raw_sha256"] = "0" * 64
    elif damage == "label_overlay":
        state["binding"]["multisource_supervision"] = {}
    elif damage == "labels":
        state["binding"]["original_target_sha256"] = "0" * 64
    elif damage == "recovery":
        state["binding"]["actual_finding_recovery"]["kernel_id"] = 139130771
    elif damage == "optimizer":
        state["optimizer"]["state"][0]["step"] = torch.tensor(485.0)
    elif damage == "snapshot":
        snapshot["model"]["backbone.weight"][0] = 2.0
    elif damage == "history":
        state["history"].pop()
    elif damage == "rng":
        state["rng_by_rank"].pop()
    elif damage == "last_batch":
        state["history"][-1]["studies"] = 8
    else:
        receipt["status"] = "actual_matched_IID16_full_parent_epoch_session_complete"
    with pytest.raises(ValueError):
        e.verify(state, snapshot, receipt, expected_parameters=3)
