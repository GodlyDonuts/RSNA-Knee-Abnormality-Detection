import ast
import copy
import json
from pathlib import Path

import pytest
import test_licensed_importance_serving as fixture_source
from test_licensed_parent_serving_notebook import fixture_parent

from rsna.competitive import encoder_lr_serving as scope
from rsna.competitive import licensed_importance_serving as control
from scripts.kaggle import encoder_lr_serving as compiler
from scripts.kaggle import licensed_importance_serving as original_compiler

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def records():
    snapshot, training, publication, sha = copy.deepcopy(
        fixture_source.accepted_epoch.__wrapped__()
    )
    binding = snapshot["binding"]
    binding.update(
        production_plan="licensed_influence_encoderLR3e5_fixed3887_3epoch_v1",
        source_git=scope.SOURCE,
        recipe=copy.deepcopy(scope.RECIPE),
        single_factor={"encoder_lr_control": 3e-6, "encoder_lr_treatment": 3e-5},
        actual_new_LR_recovery_claim=False,
        original_target_sha256="48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285",
    )
    snapshot.update(epoch=3, completed=1458)
    training.update(epoch=3)
    training["receipt"].update(
        segment=3,
        completed=1458,
        fixed_final_complete=True,
        status="actual_encoderLR3e5_influence_full_parent_epoch_complete",
        worker_sha256=scope.WORKER,
        binding=copy.deepcopy(binding),
    )
    # The actual producer emits only its source binding in the receipt. Runtime
    # recipe/version/population fields belong to the immutable checkpoint.
    for name in (
        "recipe",
        "version",
        "population",
        "population_sha256",
        "world_size",
        "total_steps",
    ):
        training["receipt"]["binding"].pop(name, None)
    publication.update(
        epoch=3,
        completed=1458,
        source_git=scope.SOURCE,
        objects=5,
        status="actual_encoderLR3e5_epoch_private_full_download_size_SHA_verified",
    )
    publication["independent_CPU_state_verification"].update(
        finite_contiguous_updates=1458,
        encoder_lr=3e-5,
        head_lr=3e-5,
    )
    return snapshot, training, publication, sha


def test_new_recipe_acceptance_is_separate_from_control(records):
    before = copy.deepcopy(records)
    assert "recipe" not in records[1]["receipt"]["binding"]
    assert records[0]["binding"]["recipe"] == scope.RECIPE
    result = scope.validate(*records)
    assert result["completed"] == 1458 and result["fixed_final_only"]
    assert result["encoder_lr"] == result["head_lr"] == 3e-5
    assert result["serving_windows"] == 94 and not result["independent_internal_validation"]
    assert records[1:] == before[1:]
    with pytest.raises(ValueError):
        control.validate(*records)


@pytest.mark.parametrize(
    "field",
    [
        "plan",
        "source",
        "LR",
        "head",
        "teacher",
        "targets",
        "mixture",
        "worker",
        "early",
        "pending",
        "publication",
    ],
)
def test_wrong_or_unfinished_model_cannot_reach_serving(records, field):
    snapshot, training, publication, sha = records
    if field == "plan":
        training["receipt"]["binding"]["production_plan"] = "old"
    elif field == "source":
        snapshot["binding"]["source_git"] = "a" * 40
    elif field == "LR":
        snapshot["binding"]["recipe"]["encoder_lr"] = 3e-6
    elif field == "head":
        snapshot["binding"]["recipe"]["head_lr"] = 3e-4
    elif field == "teacher":
        snapshot["binding"]["recipe"]["source_weight"] = 0.5
    elif field == "targets":
        snapshot["binding"]["original_target_sha256"] = "0" * 64
    elif field == "mixture":
        snapshot["binding"]["finding_allocation"] = {}
    elif field == "worker":
        training["receipt"]["worker_sha256"] = "0" * 64
    elif field == "early":
        training["epoch"] = 2
    elif field == "pending":
        training["status"] = "RUNNING"
    else:
        publication["status"] = "unverified"
    with pytest.raises(ValueError):
        scope.validate(snapshot, training, publication, sha)


@pytest.mark.parametrize("mode", ["qualification", "competition"])
def test_compiler_retains_original_full94_two_view_math(monkeypatch, records, mode):
    _, training, publication, _ = records
    parent = fixture_parent()
    monkeypatch.setattr(original_compiler, "PARENT_SOURCE_SHA256", original_compiler.digest(parent))
    helpers = [
        (ROOT / n).read_text()
        for n in (
            "rsna/competitive/bounded_preparation.py",
            "rsna/competitive/encoder_lr_serving.py",
            "rsna/competitive/licensed_pixel_cache.py",
            "rsna/competitive/licensed_full_bag.py",
            "rsna/competitive/licensed_importance.py",
        )
    ]
    source, meta = compiler.build(parent, *helpers, training, publication, mode=mode)
    ns = {"__name__": "synthetic_compile_only_not_GPU_acceptance"}
    exec(compile(source, "new_LR_serving_fixture", "exec"), ns)
    assert ns["ARMS"][0]["file"] == "epoch-003.pt"
    assert ns["infer_pair_probs"]() == "unchanged original two-view math"
    assert ns["gpu_windows"](None, None, 94, 320, "cuda:0") == "unchanged original images"
    assert (ns["infer_probs"](None, None, None) == 1).all()
    assert meta["encoder_peak_lr"] == 3e-5 and meta["fixed_final_only"]
    assert meta["qualification_cannot_submit"] == (mode == "qualification")
    assert not meta["new_weight_Dataset_required"]
    assert not meta["actual_new_final_model_GPU_qualification_claim"]
    assert isinstance(ast.parse(source).body[-1], ast.If)


def test_actual_producer_binding_with_checkpoint_only_runtime_recipe(records):
    snapshot, training, publication, sha = records
    observed = json.loads(
        (
            ROOT
            / "initiative/experiments/competitive-v1/encoder-lr-final-serving-20261018-v1/actual-E1-producer-binding.json"
        ).read_text()
    )
    assert observed["actual_kernel"] == 139394879 and "recipe" not in observed["binding"]
    runtime = {
        k: copy.deepcopy(snapshot["binding"][k])
        for k in (
            "recipe",
            "version",
            "population",
            "population_sha256",
            "world_size",
            "total_steps",
        )
    }
    # Actual producer metadata, synthetic final model/CPU proof. No actual final
    # trained model or GPU acceptance is represented by this format test.
    snapshot["binding"] = {**copy.deepcopy(observed["binding"]), **runtime}
    training["receipt"]["binding"] = copy.deepcopy(observed["binding"])
    assert scope.validate(snapshot, training, publication, sha)["completed"] == 1458
    assert "recipe" not in training["receipt"]["binding"]
    snapshot["binding"]["recipe"]["head_lr"] = 3e-4
    with pytest.raises(ValueError):
        scope.validate(snapshot, training, publication, sha)
