import copy

import pytest
from test_licensed_parent_serving import accepted_epoch as parent_epoch

from rsna.competitive import licensed_importance_serving as serving
from rsna.competitive import licensed_parent_serving as whole94


@pytest.fixture
def accepted_epoch():
    # Synthetic receipt fixture; no actual model/GPU/accuracy acceptance.
    snapshot, training, publication, sha = copy.deepcopy(parent_epoch.__wrapped__())
    training["epoch"] = training.pop("segment")
    training["role"] = "influence"
    training["receipt"]["status"] = "actual_matched_IID16_full_parent_epoch_session_complete"
    training["receipt"]["role"] = "influence"
    publication["status"] = (
        "actual_matched_IID16_parent_epoch_private_full_download_size_SHA_verified"
    )
    publication["role"] = "influence"
    publication["independent_CPU_state_verification"].update(
        role="influence", sampled_windows=16, proposal_sha256=serving.PROPOSAL
    )
    binding = snapshot["binding"]
    binding.update(
        production_plan="licensed_MRI_parent_uniform_influence_IID16_full3887_fixed3_continuous_v1",
        role="influence",
        version=serving.VERSION,
        proposal={"proposal_table": {"bytes": 2923152, "sha256": serving.PROPOSAL}},
        actual_recovery={
            "kernel_id": 139130771,
            "source_sha256": "a01dd9643355dfb1d2b2f7f43e3cda819e0c44e69a0cd7c2acd99440c0e25554",
        },
    )
    binding["recipe"].update(windows=16, role="influence")
    training["receipt"]["binding"] = copy.deepcopy(binding)
    return snapshot, training, publication, sha


@pytest.mark.parametrize("role", ["uniform", "influence"])
def test_serving_requires_the_same_actual_role_and_uses_original_full94(accepted_epoch, role):
    snapshot, training, publication, sha = accepted_epoch
    for binding in (snapshot["binding"], training["receipt"]["binding"]):
        binding["role"] = binding["recipe"]["role"] = role
    training["role"] = publication["role"] = role
    training["receipt"]["role"] = role
    publication["independent_CPU_state_verification"]["role"] = role
    actual = serving.validate(snapshot, training, publication, sha)
    assert actual["role"] == role and actual["completed"] == 486
    assert actual["training_windows"] == 16 and actual["serving_windows"] == 94
    assert actual["independent_internal_validation"] is False
    with pytest.raises(ValueError):
        whole94.validate(snapshot, training, publication, sha)


@pytest.mark.parametrize(
    "change",
    ["role", "proposal", "recovery", "source", "optimizer", "windows", "partial", "pending"],
)
def test_mixed_roles_unknown_inputs_and_incomplete_epochs_are_rejected(accepted_epoch, change):
    snapshot, training, publication, sha = accepted_epoch
    if change == "role":
        publication["role"] = "uniform"
    elif change == "proposal":
        snapshot["binding"]["proposal"]["proposal_table"]["sha256"] = "0" * 64
    elif change == "recovery":
        snapshot["binding"]["actual_recovery"]["kernel_id"] = 1
    elif change == "source":
        publication["original_source_sha256"] = "0" * 64
    elif change == "optimizer":
        publication["independent_CPU_state_verification"]["actual_AdamW_parameters"] = 1
    elif change == "windows":
        snapshot["binding"]["recipe"]["windows"] = 94
    elif change == "partial":
        snapshot["completed"] -= 1
    else:
        training["status"] = "RUNNING"
    with pytest.raises(ValueError):
        serving.validate(snapshot, training, publication, sha)
