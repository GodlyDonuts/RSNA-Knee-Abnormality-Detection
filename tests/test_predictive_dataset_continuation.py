"""An ended uploader may continue only its remaining, identity-bound phases."""
import copy
import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PATH = ROOT / "initiative/research/predictive-final-dataset-continuation-20261021-v1/continue_once.py"
spec = importlib.util.spec_from_file_location("dataset_continuation_test", PATH)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def fixture(tmp_path):
    records = {
        "owner-intent.json": dict(pid=32378, placement_source_git=m.PLACEMENT_PIN),
        "stopped.json": {},
        "stage-qualification-result.json": dict(status="predictive_final_deployment_controls_staged"),
        "prepare-qualification-result.json": dict(status="actual_three_predictive_final_exports_prepared_CPU_only", dataset_slug=m.DATASET, bundle_identity="bound"),
        "dataset-qualification-result.json": dict(status="predictive_dataset_create_returned", new_Dataset_calls=1, new_GPU_or_entries=0),
    }
    for name, value in records.items():
        (tmp_path / name).write_text(json.dumps(value))
    actual = dict(dataset_slug=m.DATASET, bundle_identity="bound", claim_present=False, mutations=0,
                  responses={"GetDataset": dict(HTTP=200, id=m.DATASET_ID, isPrivate=True, currentVersionNumber=1),
                             "GetDatasetStatus": dict(HTTP=200, result={"status": "READY"})},
                  ledger=[dict(value=dict(state="predictive_dataset_create_returned", dataset_slug=m.DATASET))],
                  **{"dataset-response.json": dict(final_create_rpc_calls=1)}, root_files=["prepared.json"])
    return actual


def test_exact_returned_dataset_only(tmp_path):
    assert m.validate_original(tmp_path, fixture(tmp_path), lambda pid: "")["bundle_identity"] == "bound"


@pytest.mark.parametrize("field,value", [("id", 1), ("isPrivate", False), ("currentVersionNumber", 2)])
def test_changed_dataset_rejected(tmp_path, field, value):
    actual = fixture(tmp_path)
    actual["responses"]["GetDataset"][field] = value
    with pytest.raises(AssertionError):
        m.validate_original(tmp_path, actual, lambda pid: "")


def test_active_original_rejected(tmp_path):
    with pytest.raises(AssertionError):
        m.validate_original(tmp_path, fixture(tmp_path), lambda pid: "original owner")


def test_prior_push_or_unresolved_claim_rejected(tmp_path):
    actual = fixture(tmp_path)
    changed = copy.deepcopy(actual)
    changed["claim_present"] = True
    with pytest.raises(AssertionError):
        m.validate_original(tmp_path, changed, lambda pid: "")
    (tmp_path / "push-qualification-intent.json").write_text("{}")
    with pytest.raises(AssertionError):
        m.validate_original(tmp_path, actual, lambda pid: "")


def test_only_completed_prefix_removed():
    body = '''class Owner:
    def run(self):
        self.action("stage")
        self.action("prepare")
        self.action("dataset")
        self.action("dataset-reconcile")
        self.action("compile")
        self.action("push")
'''
    changed = m.remaining_owner(body)
    assert changed[changed.index('        self.action("dataset-reconcile")'):] == body[body.index('        self.action("dataset-reconcile")'):]
    assert 'self.action("dataset")' not in changed
    assert 'self.action("stage")' not in changed
    assert 'self.action("prepare")' not in changed
    assert "self.admit_existing_dataset()" in changed
