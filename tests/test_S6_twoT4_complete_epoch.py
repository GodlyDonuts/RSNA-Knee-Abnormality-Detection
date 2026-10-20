"""Synthetic tensor state tests; no actual model or clinical admission here."""

import copy
import importlib.machinery
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import pytest
from test_licensed_parent_epoch import completed

from rsna.competitive.licensed_importance import VERSION, Recipe

D = Path("initiative/research/S6-Kaggle-production-route-20261019-v1")
loader = importlib.machinery.SourceFileLoader("target_twoT4_epoch", str(D / "epoch_check.py.txt"))
spec = importlib.util.spec_from_loader(loader.name, loader)
checker = importlib.util.module_from_spec(spec)
loader.exec_module(checker)


@pytest.fixture
def synthetic_complete_target_epoch():
    state, snapshot, receipt = completed.__wrapped__()
    actual = json.loads(
        Path(
            "initiative/research/S6-Kaggle-existing-output-replay-20261019-v1/actual-accepted.json"
        ).read_text()
    )
    ready = actual["receipt"]["complete_teacher_admission"]
    b = state["binding"]
    b.update(
        version=VERSION,
        recipe=asdict(Recipe(epochs=3, role="influence")),
        role="influence",
        population_sha256=checker.POPULATION,
        production_plan="complete_S6_equal_rank_matched_full3887_v1",
        parent_checkpoint_sha256=checker.PARENT,
        parent_source_sha256=checker.SOURCE,
        teacher_target_role=checker.ROLE,
        teacher_overlay_source_sha256=checker.OVERLAY,
        complete_teacher_admission=copy.deepcopy(ready),
        teacher_transport=copy.deepcopy(ready["roles"][checker.ROLE]),
        source_module_sha256={
            n: h
            for n, h in actual["receipt"]["source_module_sha256"].items()
            if n != "rsna/competitive/complete_teacher_overlay.py"
        },
        proposal={"proposal_table": {"bytes": 2923152, "sha256": checker.PROPOSAL}},
        actual_recovery={"kernel_id": 139130771},
        diagnostic_only=False,
        protected_gold_confirmation_used=False,
        independent_internal_validation=False,
    )
    receipt.update(
        status=checker.STATUS,
        role="influence",
        teacher_target_role=checker.ROLE,
        total=1458,
        fixed_final_complete=False,
        binding=copy.deepcopy(b),
        devices=[
            {
                "GPU": "Tesla T4",
                "completed": 486,
                "trainable_parameters": 73136096,
                "all_BN_buffers_equal_parent": True,
            }
            for _ in range(2)
        ],
    )
    return state, snapshot, receipt, actual


def test_all_original_two_rank_tensor_optimizer_rng_checks_remain(synthetic_complete_target_epoch):
    z = checker.verify(*synthetic_complete_target_epoch, expected_parameters=3)
    assert (
        z["all_two_rank_RNG_and_scaler_records_present"] and z["finite_contiguous_updates"] == 486
    )
    assert z["teacher_target_role"] == checker.ROLE and z["actual_target_engine_CPU"] == 974246


@pytest.mark.parametrize(
    "bad",
    [
        "teacher_bytes",
        "teacher_role",
        "source_module",
        "recipe",
        "engine_not_actual",
        "one_rank",
        "model",
        "optimizer",
        "history",
        "false_final",
    ],
)
def test_mixed_target_or_incomplete_real_state_is_refused(synthetic_complete_target_epoch, bad):
    state, snapshot, receipt, actual = copy.deepcopy(synthetic_complete_target_epoch)
    if bad == "teacher_bytes":
        state["binding"]["teacher_transport"]["new_teacher_array_sha256"] = "0" * 64
    elif bad == "teacher_role":
        receipt["teacher_target_role"] = "parent_only"
    elif bad == "source_module":
        state["binding"]["source_module_sha256"][
            "rsna/competitive/licensed_importance_train.py"
        ] = "0" * 64
    elif bad == "recipe":
        state["binding"]["recipe"]["epochs"] = 10
    elif bad == "engine_not_actual":
        actual["status"] = "prepared"
    elif bad == "one_rank":
        state["rng_by_rank"].pop()
    elif bad == "model":
        snapshot["model"]["backbone.weight"][0] = 2
    elif bad == "optimizer":
        state["optimizer"]["state"].pop(0)
    elif bad == "history":
        state["history"].pop()
    else:
        receipt["fixed_final_complete"] = True
    with pytest.raises(ValueError):
        checker.verify(state, snapshot, receipt, actual, expected_parameters=3)
