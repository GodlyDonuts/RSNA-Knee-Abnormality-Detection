"""Synthetic serving checks; no actual trained checkpoint admission."""

import copy
import importlib.util
from pathlib import Path

import pytest
from test_licensed_importance_serving import accepted_epoch as base_fixture

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("finding_final_scope", HERE / "serving.py")
scope = importlib.util.module_from_spec(spec)
spec.loader.exec_module(scope)


@pytest.fixture
def records():
    snapshot, training, publication, sha = copy.deepcopy(base_fixture.__wrapped__())
    for binding in (snapshot["binding"], training["receipt"]["binding"]):
        binding.update(
            production_plan=scope.PLAN,
            source_git=scope.SOURCE,
            component_raw_sha256=scope.RAW,
            component_file_sha256=scope.COMPONENT,
            component_roster_sha256=scope.ROSTER,
            original_target_sha256=scope.TARGET,
            finding_allocation=copy.deepcopy(scope.ALLOCATION),
            actual_finding_recovery=copy.deepcopy(scope.RECOVERY),
        )
    training["receipt"]["binding"].pop("finding_allocation")
    snapshot.update(epoch=3, completed=1458)
    training["epoch"] = 3
    training["receipt"].update(
        segment=3,
        completed=1458,
        fixed_final_complete=True,
        checkpoint={"sha256": "3" * 64, "bytes": 10},
        status="actual_finding_strata_full_parent_epoch_session_complete",
    )
    publication.update(
        epoch=3,
        completed=1458,
        source_git=scope.SOURCE,
        objects=6,
        full_optimizer_privately_verified=True,
        status="actual_finding_strata_epoch_private_full_download_size_SHA_verified",
    )
    publication["files"]["optimizer-checkpoint.pt"] = copy.deepcopy(
        training["receipt"]["checkpoint"]
    )
    publication["independent_CPU_state_verification"].update(
        finite_contiguous_updates=1458,
        finding_allocation_and_original_targets_exact=True,
        actual_finding_recovery_kernel=139349540,
    )
    return snapshot, training, publication, sha


def test_final_metadata_extension_preserves_original_full94_and_inputs(records):
    original = copy.deepcopy(records)
    result = scope.validate(*records)
    assert result["completed"] == 1458 and result["serving_windows"] == 94
    assert result["fixed_final_only"] and not result["clinical_accuracy_claim"]
    assert scope.provenance(*records[1:]) == 3
    assert records[0]["binding"] == original[0]["binding"]
    assert records[1:] == original[1:]


@pytest.mark.parametrize(
    "field",
    [
        "production_plan",
        "source_git",
        "component_raw_sha256",
        "component_file_sha256",
        "component_roster_sha256",
        "original_target_sha256",
        "finding_allocation",
        "actual_finding_recovery",
        "multisource_supervision",
        "source_label_role",
    ],
)
@pytest.mark.parametrize("location", ["snapshot", "receipt"])
def test_unknown_or_label_changed_treatment_rejected(records, field, location):
    snapshot, training, publication, sha = records
    binding = snapshot["binding"] if location == "snapshot" else training["receipt"]["binding"]
    binding[field] = "invalid"
    with pytest.raises(ValueError):
        scope.validate(snapshot, training, publication, sha)


@pytest.mark.parametrize(
    "damage",
    [
        "early",
        "pending",
        "backup",
        "optimizer",
        "objects",
        "graph",
        "proof",
        "recovery",
        "model_sha",
        "recipe",
        "nonfinal",
    ],
)
def test_incomplete_or_wrong_model_preservation_refused(records, damage):
    snapshot, training, publication, sha = records
    if damage == "early":
        training["epoch"] = 2
    elif damage == "pending":
        training["status"] = "RUNNING"
    elif damage == "backup":
        publication["full_optimizer_privately_verified"] = False
    elif damage == "optimizer":
        publication["files"]["optimizer-checkpoint.pt"]["sha256"] = "4" * 64
    elif damage == "objects":
        publication["objects"] = 5
    elif damage == "graph":
        publication["independent_CPU_state_verification"]["full_learned_graph_exact"] = False
    elif damage == "proof":
        publication["independent_CPU_state_verification"][
            "finding_allocation_and_original_targets_exact"
        ] = False
    elif damage == "recovery":
        snapshot["binding"]["actual_finding_recovery"]["kernel_id"] = 1
    elif damage == "model_sha":
        sha = "0" * 64
    elif damage == "recipe":
        snapshot["binding"]["recipe"]["windows"] = 94
    else:
        training["receipt"]["fixed_final_complete"] = False
    with pytest.raises(ValueError):
        scope.validate(snapshot, training, publication, sha)


def test_actual_final_receipt_does_not_invent_checkpoint_allocation(records):
    import json

    root = HERE.parents[3]
    actual = json.loads(
        (
            root
            / "initiative/research/finding-stratified-bags-20261018-v1/epoch-3-actual-accepted.json"
        ).read_text()
    )
    publication = json.loads(
        (
            root
            / "initiative/research/finding-stratified-bags-20261018-v1/epoch-3-private-verified.json"
        ).read_text()
    )
    assert "finding_allocation" not in actual["receipt"]["binding"]
    assert scope.provenance(actual, publication, actual["receipt"]["epoch"]["sha256"]) == 3
    snapshot = records[0]
    runtime = {
        k: copy.deepcopy(snapshot["binding"][k])
        for k in (
            "recipe",
            "version",
            "population",
            "population_sha256",
            "world_size",
            "total_steps",
            "finding_allocation",
        )
    }
    snapshot["binding"] = {**copy.deepcopy(actual["receipt"]["binding"]), **runtime}
    assert (
        scope.validate(snapshot, actual, publication, actual["receipt"]["epoch"]["sha256"])[
            "completed"
        ]
        == 1458
    )
    snapshot["binding"]["finding_allocation"]["components_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        scope.validate(snapshot, actual, publication, actual["receipt"]["epoch"]["sha256"])
