"""New paired publication contract must reject ordinary or misbound states."""

from dataclasses import asdict

import pytest
from test_licensed_parent_epoch import completed as original

from rsna.competitive import paired_logit_epoch as m


@pytest.fixture
def paired():
    state, snapshot, receipt = original.__wrapped__()
    b = state["binding"]
    r = m.Recipe(role="influence")
    b.update(
        production_plan=m.PLAN,
        version=m.VERSION,
        recipe=asdict(r),
        role="influence",
        population_sha256=m.POPULATION_SHA,
        paired_recovery=m.RECOVERY.copy(),
        correction_table=m.TABLE.copy(),
        parent_correction_sha256=m.TABLE["sha256"],
        actual_recovery={"kernel_id": 139130771},
        proposal={
            "proposal_table": {
                "bytes": 2923152,
                "sha256": "e812a9f90cd2b3db92d9acafd28688d5ee6fe16fc188d6517c2bab565f79205e",
            }
        },
    )
    snapshot["binding"] = b
    receipt.update(status=m.STATUS, role="influence")
    for i, row in enumerate(state["history"]):
        f = m.lr_factor(i, 1458, r)
        row.update(encoder_lr=r.encoder_lr * f, head_lr=r.head_lr * f)
    state["optimizer"]["param_groups"][0]["lr"] = state["history"][-1]["encoder_lr"]
    state["optimizer"]["param_groups"][1]["lr"] = state["history"][-1]["head_lr"]
    return state, snapshot, receipt


def test_complete_paired_epoch(paired):
    assert m.verify(*paired, expected_parameters=3)[
        "exact_original_recipe_and_real_paired_recovery"
    ]


@pytest.mark.parametrize(
    "field", ["ordinary_version", "correction", "recovery", "population", "lr", "AdamW"]
)
def test_rejects_wrong_training_or_resumable_state(paired, field):
    state, _, _ = paired
    if field == "ordinary_version":
        state["binding"]["version"] = "licensed_full_encoder_uniform_or_teacher_influence_IID16_v1"
    elif field == "correction":
        state["binding"]["parent_correction_sha256"] = "0" * 64
    elif field == "recovery":
        state["binding"]["paired_recovery"]["kernel_id"] = 139130771
    elif field == "population":
        state["binding"]["population_sha256"] = "0" * 64
    elif field == "lr":
        state["history"][33]["encoder_lr"] *= 10
    else:
        state["optimizer"]["state"].pop(0)
    with pytest.raises(ValueError):
        m.verify(*paired, expected_parameters=3)
