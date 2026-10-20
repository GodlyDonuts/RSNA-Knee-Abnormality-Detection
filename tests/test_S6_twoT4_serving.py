"""Synthetic serving provenance/regression tests; no GPU or accuracy acceptance."""

import copy
import importlib.machinery
import importlib.util
from pathlib import Path

import pytest
from test_licensed_importance_serving import accepted_epoch as old_epoch
from test_licensed_importance_serving_notebook import sources
from test_licensed_parent_serving_notebook import fixture_parent
from test_S6_twoT4_complete_epoch import (
    synthetic_complete_target_epoch as synthetic_complete_target_epoch,
)

D = Path("initiative/research/S6-Kaggle-production-route-20261019-v1")


def load(name):
    loader = importlib.machinery.SourceFileLoader(name, str(D / (name + ".py.txt")))
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader))
    loader.exec_module(module)
    return module


helper = load("serving_validate")
compiler = load("serving_compile")


@pytest.fixture
def new_epoch(synthetic_complete_target_epoch):
    _, _, receipt, _ = copy.deepcopy(synthetic_complete_target_epoch)
    snapshot, training, pub, sha = old_epoch.__wrapped__()
    receipt["binding"]["source_git"] = "a" * 40
    snapshot["binding"] = copy.deepcopy(receipt["binding"])
    receipt["epoch"] = copy.deepcopy(training["receipt"]["epoch"])
    training.update(receipt=receipt, role=helper.ROLE, version=1)
    pub.update(
        status="actual_complete_target_twoT4_epoch_private_all_objects_download_size_SHA_verified",
        version=1,
        kernel_id=training["kernel_id"],
        teacher_target_role=helper.ROLE,
        objects=6,
        canonical_shared_publication_lock_used=True,
        model_sha256=sha,
        prefix="checkpoints/complete-S6-target-twoT4-"
        + pub["source_git"][:12]
        + "/"
        + helper.ROLE
        + "/epoch-001",
    )
    pub["independent_CPU_state_verification"].update(
        teacher_target_role=helper.ROLE,
        sampling_role="influence",
        training_windows=16,
        complete_target_sha256=helper.TARGET,
        actual_target_engine_CPU=974246,
        actual_target_engine_kernel=139695847,
    )
    return snapshot, training, pub, sha


def test_complete_target_snapshot_is_distinct_from_sampling_role(new_epoch):
    result = helper.validate(*new_epoch)
    assert result["role"] == helper.ROLE and result["sampling_role"] == "influence"
    assert result["serving_windows"] == 94 and result["training_windows"] == 16
    assert not result["clinical_accuracy_claim"]


@pytest.mark.parametrize(
    "change",
    ["running", "partial", "old_target", "publication", "source", "CPU", "rank", "head", "labels"],
)
def test_mixed_or_incomplete_provenance_refused(new_epoch, change):
    snapshot, training, pub, sha = new_epoch
    if change == "running":
        training["status"] = "RUNNING"
    elif change == "partial":
        snapshot["completed"] -= 1
    elif change == "old_target":
        snapshot["binding"]["teacher_target_role"] = "parent_only"
    elif change == "publication":
        pub["canonical_shared_publication_lock_used"] = False
    elif change == "source":
        pub["original_source_sha256"] = "0" * 64
    elif change == "CPU":
        pub["independent_CPU_state_verification"]["actual_target_engine_CPU"] = 0
    elif change == "rank":
        snapshot["binding"]["world_size"] = 1
    elif change == "head":
        snapshot["model"]["clsW"] = snapshot["model"]["clsW"][:1]
    else:
        snapshot["binding"]["complete_teacher_admission"][
            "labels_masks_and_original_targets_modified"
        ] = True
    with pytest.raises(ValueError):
        helper.validate(snapshot, training, pub, sha)


@pytest.mark.parametrize("mode", ["qualification", "competition"])
def test_same_full94_inference_new_target_weights(monkeypatch, new_epoch, mode):
    _, training, pub, _ = new_epoch
    parent = fixture_parent()
    monkeypatch.setattr(compiler, "PARENT_SOURCE", compiler.sha(parent))
    inputs = sources()
    inputs[1] = (D / "serving_validate.py.txt").read_text()
    source, metadata = compiler.build(
        parent,
        *inputs,
        training,
        pub,
        mode=mode,
        legacy_source=Path("scripts/kaggle/licensed_importance_serving.py").read_text(),
    )
    ns = {"__name__": "synthetic_only"}
    exec(compile(source, "synthetic_complete_target_serving", "exec"), ns)
    assert ns["infer_pair_probs"]() == "unchanged original two-view math"
    assert ns["gpu_windows"](None, None, 94, 320, "cuda:0") == "unchanged original images"
    assert metadata["training_world_size"] == 2 and metadata["teacher_target_role"] == helper.ROLE
    assert metadata["qualification_cannot_submit"] == (mode == "qualification")
    assert (
        metadata["serving_windows"] == 94 and not metadata["actual_new_weight_T4_serving_accepted"]
    )
    assert "actual_complete_target_twoT4_epoch_" in source
    assert ns["ARMS"][0]["file"] == "epoch-001.pt"


def test_serving_preparation_controls_compile_without_touching_live_fitting():
    for name in ("serving_prepare", "serving_prepare_once"):
        source = (D / (name + ".py.txt")).read_text()
        compile(source, name, "exec")
    source = (D / "serving_prepare.py.txt").read_text()
    assert source.index("publication-private-verified.json") < source.index("output.mkdir")
    assert "new_GPU_Dataset_or_competition_mutations" in source
