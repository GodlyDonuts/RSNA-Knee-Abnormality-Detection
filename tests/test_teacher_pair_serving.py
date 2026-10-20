"""Synthetic lineage/compiler checks, not actual model or T4 acceptance."""

import copy
import hashlib
import importlib.util
import json
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest
from test_licensed_importance_serving import accepted_epoch as original_epoch
from test_licensed_importance_serving_notebook import sources
from test_licensed_parent_serving_notebook import fixture_parent

from rsna.competitive import teacher_pair_serving as helper

D = Path("initiative/research/teacher-pair-final-serving-20261020-v2")
loader = SourceFileLoader("pair_final_compiler", str(D / "compile.py.txt"))
compiler = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
loader.exec_module(compiler)


@pytest.fixture(params=["pointwise", "teacher_rank"])
def final_pair(request):
    snapshot, training, publication, sha = copy.deepcopy(original_epoch.__wrapped__())
    role = request.param
    binding = copy.deepcopy(helper.EXPECTED_BINDING)
    binding["recipe"]["objective"] = role
    training = {
        "status": "COMPLETE",
        "epoch": 3,
        "kernel_id": None,
        "training_Slurm_job": helper.GPU_JOB,
        "source_sha256": helper.WORKER_SHA,
        "role": role,
        "receipt": {
            "status": "actual_teacher_pair_fixed_final_model_for_serving",
            "completed": 1458,
            "binding": binding,
            "epoch": {"bytes": 292867251, "sha256": sha},
        },
    }
    snapshot.update(binding=copy.deepcopy(binding), epoch=3, completed=1458)
    schema = {
        n: {"shape": list(t.shape), "dtype": str(t.dtype)} for n, t in snapshot["model"].items()
    }
    publication = {
        "status": "actual_teacher_pair_immutable_epoch_private_all_objects_download_size_SHA_verified",
        "GPU_job": helper.GPU_JOB,
        "source_git": helper.SCIENCE,
        "role": role,
        "epoch": 3,
        "completed_updates": 1458,
        "objects": 6,
        "canonical_shared_publication_lock_used": True,
        "prefix": "checkpoints/teacher-pair-full3887-"
        + helper.SCIENCE[:12]
        + "/"
        + role
        + "/epoch-003",
        "model_sha256": sha,
        "model_bytes": 292867251,
        "provenance": {
            "scientific_source_git": helper.SCIENCE,
            "GPU_job": helper.GPU_JOB,
            "role": role,
            "epoch": 3,
            "completed_updates": 1458,
            "binding": copy.deepcopy(binding),
            "snapshot_sha256": sha,
            "trainable_parameters_from_actual_accepted_trainer": 73136096,
            "source_permission_sha256": binding["source_binding"]["source_permission_sha256"],
            "original_MRI_labels_UIDs_private_arrays_included": False,
            "model_schema_sha256": hashlib.sha256(
                json.dumps(schema, sort_keys=True).encode()
            ).hexdigest(),
        },
    }
    return snapshot, training, publication, sha


def test_final_roles_keep_truthful_H100_binding(final_pair):
    s, t, p, sha = final_pair
    result = helper.validate(s, t, p, sha)
    assert result["role"] == t["role"] and result["training_world_size"] == 1
    assert result["training_Slurm_job"] == "975297" and result["serving_windows"] == 94
    assert not result["actual_new_T4_acceptance"]


@pytest.mark.parametrize(
    "change",
    [
        "early",
        "objective",
        "recipe",
        "publication",
        "model",
        "source",
        "schema",
        "head",
        "fake_kernel",
        "protected",
    ],
)
def test_incomplete_or_wrong_objective_model_refused(final_pair, change):
    s, t, p, sha = final_pair
    if change == "early":
        t["epoch"] = 2
    elif change == "objective":
        t["role"] = "teacher_rank" if t["role"] == "pointwise" else "pointwise"
    elif change == "recipe":
        t["receipt"]["binding"]["recipe"]["encoder_lr"] *= 10
    elif change == "publication":
        p["canonical_shared_publication_lock_used"] = False
    elif change == "model":
        p["model_sha256"] = "0" * 64
    elif change == "source":
        t["source_sha256"] = "0" * 64
    elif change == "schema":
        p["provenance"]["model_schema_sha256"] = "0" * 64
    elif change == "head":
        s["model"]["clsW"] = s["model"]["clsW"][:1]
    elif change == "fake_kernel":
        t["kernel_id"] = 975297
    else:
        t["receipt"]["binding"]["protected_or_gold_used"] = True
    with pytest.raises(ValueError):
        helper.validate(s, t, p, sha)


@pytest.mark.parametrize("mode", ["qualification", "competition"])
def test_reuses_original_full94_inference(final_pair, monkeypatch, mode):
    _, training, publication, _ = final_pair
    parent = fixture_parent()
    monkeypatch.setattr(compiler, "PARENT_SOURCE", compiler.digest(parent))
    inputs = sources()
    inputs[1] = Path("rsna/competitive/teacher_pair_serving.py").read_text()
    source, metadata = compiler.build(
        parent,
        *inputs,
        training,
        publication,
        mode=mode,
        legacy_source=Path("scripts/kaggle/licensed_importance_serving.py").read_text(),
    )
    ns = {"__name__": "synthetic_not_actual_T4_acceptance"}
    exec(compile(source, "original_inference_new_final_binding", "exec"), ns)
    assert ns["infer_pair_probs"]() == "unchanged original two-view math"
    assert ns["gpu_windows"](None, None, 94, 320, "cuda:0") == "unchanged original images"
    assert ns["ARMS"][0]["file"] == "epoch-003.pt"
    assert metadata["training_kernel"] is None
    assert metadata["training_Slurm_job"] == "975297"
    assert metadata["objective"] == training["role"]
    assert metadata["training_world_size"] == 1 and metadata["source_loss_weight"] == 0.25
    assert not metadata["actual_new_weight_T4_serving_accepted"]
    assert "actual_teacher_pair_final_epoch_" in source


def test_waiting_final_publication_does_not_create_intent_or_dispatch(tmp_path, monkeypatch):
    loader = SourceFileLoader("pair_final_local_wait_only", str(D / "coordinate_once.py.txt"))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(module)
    monkeypatch.setattr(module, "R", tmp_path)
    monkeypatch.setattr(module, "L", tmp_path / "artifacts")
    result = module.run("compile", "pointwise")
    assert result["status"] == "waiting_actual_fixed_final_private_publication_no_intent"
    assert not (tmp_path / "artifacts").exists()
