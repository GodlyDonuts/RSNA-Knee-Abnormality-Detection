import copy

import pytest
import torch

from rsna.competitive import licensed_parent_serving as serving


@pytest.fixture
def accepted_epoch():
    # Synthetic format fixture only; not a real complete model or GPU admission.
    binding = {
        "production_plan": "licensed_MRI_parent_all94_full3887_fixed3_continuous_v1",
        "source_git": "a" * 40,
        "parent_checkpoint_sha256": serving.PARENT,
        "parent_source_sha256": serving.PARENT_SOURCE,
        "population_sha256": serving.POPULATION,
        "population": 3887,
        "world_size": 2,
        "total_steps": 1458,
        "diagnostic_only": False,
        "protected_gold_confirmation_used": False,
        "independent_internal_validation": False,
        "recipe": {
            "epochs": 3,
            "windows": 94,
            "effective_batch": 8,
            "encoder_lr": 3e-6,
            "head_lr": 3e-5,
            "source_weight": 0.25,
        },
    }
    model = {
        "backbone.weight": torch.ones(1),
        "clsW": torch.ones(12, 1024),
        "clsb": torch.zeros(12),
        "norm.weight": torch.ones(1024),
        "norm.bias": torch.zeros(1024),
    }
    snapshot = {"model": model, "binding": binding, "epoch": 1, "completed": 486}
    training = {
        "status": "COMPLETE",
        "segment": 1,
        "kernel_id": 123,
        "source_sha256": "1" * 64,
        "receipt": {
            "status": "actual_whole94_full_parent_epoch_session_complete",
            "segment": 1,
            "completed": 486,
            "total": 1458,
            "epoch": {"bytes": 123, "sha256": "2" * 64},
            "binding": copy.deepcopy(binding),
        },
    }
    publication = {
        "status": "actual_whole94_parent_epoch_private_full_download_size_SHA_verified",
        "epoch": 1,
        "completed": 486,
        "original_Kaggle_kernel": 123,
        "original_source_sha256": "1" * 64,
        "source_git": "a" * 40,
        "files": {"model.pt": training["receipt"]["epoch"]},
        "independent_CPU_state_verification": {
            "full_learned_graph_exact": True,
            "finite_contiguous_updates": 486,
            "actual_AdamW_parameters": 73136096,
            "all_two_rank_RNG_and_scaler_records_present": True,
        },
    }
    return snapshot, training, publication, "2" * 64


def test_deployment_requires_actual_epoch_and_independently_preserved_full_model(accepted_epoch):
    result = serving.validate(*accepted_epoch)
    assert result["epoch"] == 1 and result["completed"] == 486 and result["full_population"] == 3887


@pytest.mark.parametrize(
    "change",
    [
        "pending",
        "diagnostic",
        "partial",
        "parent",
        "population",
        "recipe",
        "windows",
        "unpublished",
        "optimizer",
        "weightsha",
        "nonfinite",
        "head_shape",
    ],
)
def test_wrong_or_diagnostic_weights_cannot_become_a_competition_candidate(accepted_epoch, change):
    snapshot, training, pub, sha = accepted_epoch
    if change == "pending":
        training["status"] = "RUNNING"
    elif change == "diagnostic":
        snapshot["binding"]["diagnostic_only"] = True
    elif change == "partial":
        snapshot["completed"] -= 1
    elif change == "parent":
        snapshot["binding"]["parent_checkpoint_sha256"] = "0" * 64
    elif change == "population":
        snapshot["binding"]["population"] = 16
    elif change == "recipe":
        snapshot["binding"]["recipe"]["epochs"] = 2
    elif change == "windows":
        snapshot["binding"]["recipe"]["windows"] = 12
    elif change == "unpublished":
        pub["status"] = "commit_uncertain"
    elif change == "optimizer":
        pub["independent_CPU_state_verification"]["actual_AdamW_parameters"] = 402956
    elif change == "weightsha":
        sha = "0" * 64
    elif change == "nonfinite":
        snapshot["model"]["backbone.weight"][0] = float("nan")
    else:
        snapshot["model"]["clsW"] = torch.ones(12, 768)
    with pytest.raises(ValueError):
        serving.validate(snapshot, training, pub, sha)
