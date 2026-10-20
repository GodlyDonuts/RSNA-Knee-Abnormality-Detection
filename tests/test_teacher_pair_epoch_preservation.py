"""Reject publication of mismatched production epochs; no clinical assertions."""

import copy
import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

D = Path("initiative/research/teacher-pair-epoch-preservation-20261020-v1")
loader = importlib.machinery.SourceFileLoader("pair_preserve", str(D / "publish.py.txt"))
m = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
loader.exec_module(m)


def binding():
    from dataclasses import asdict

    from rsna.competitive.teacher_pair import Recipe

    recipe = asdict(Recipe(epochs=3, objective="teacher_rank"))
    meta = {"source_git": m.SOURCE, "source_binding": {"actual_GPU_recovery": "974940"}}
    b = {
        "parent_checkpoint_sha256": m.PARENT,
        "source_binding": meta["source_binding"],
        "initial_state_sha256": m.INITIAL,
        "fitting_only": True,
        "protected_or_gold_used": False,
        "diagnostic_only": False,
        "matched_two_view_three_epoch_comparison": True,
        "version": "full_encoder_two_view_teacher_pair_risk_v1",
        "teacher_proposal_sha256": m.PROPOSAL,
        "recipe": recipe,
        "population": 3887,
        "population_sha256": m.COHORT,
        "world_size": 1,
        "anatomical_views_per_study": 2,
        "study_coupled_batch_loss": True,
        "device_type": "cuda",
        "precision": "FP16_autocast_FP32_loss_dynamic_scaler",
        "BN_running_statistics": "frozen_at_accepted_parent",
        "total_steps": 1458,
        "device_name": "NVIDIA H100 PCIe",
        "torch_version": "2.11.0+cu130",
    }
    return (
        {"binding": b, "epoch": 2, "completed": 972, "model": {}},
        {"binding": copy.deepcopy(b)},
        meta,
        recipe,
    )


def test_exact_immutable_boundary_matches_actual_recipe_contract():
    s, r, meta, recipe = binding()
    assert m.validate_binding(s, r, meta, "teacher_rank", 2, recipe) == s["binding"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("population", 3886),
        ("population_sha256", "bad"),
        ("initial_state_sha256", "different"),
        ("diagnostic_only", True),
        ("protected_or_gold_used", True),
        ("world_size", 2),
        ("anatomical_views_per_study", 1),
        ("total_steps", 486),
        ("teacher_proposal_sha256", "bad"),
        ("BN_running_statistics", "updated"),
        ("device_name", "CPU"),
    ],
)
def test_rejects_wrong_boundaries_and_scientific_bindings(field, value):
    s, r, meta, recipe = binding()
    s["binding"][field] = value
    r["binding"][field] = value
    with pytest.raises(ValueError):
        m.validate_binding(s, r, meta, "teacher_rank", 2, recipe)


def test_rejects_optimizer_payload_and_wrong_objective_or_progress():
    for change in ("optimizer", "wrong_objective", "partial"):
        s, r, meta, recipe = binding()
        if change == "optimizer":
            s["optimizer"] = {}
        elif change == "wrong_objective":
            recipe = {**recipe, "objective": "pointwise"}
        else:
            s["completed"] = 971
        with pytest.raises(ValueError):
            m.validate_binding(s, r, meta, "teacher_rank", 2, recipe)
