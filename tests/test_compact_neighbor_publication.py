import copy

import pytest
import test_compact_publication as original

from scripts.hf import publish_compact_neighbor_epoch as publisher


def fixture():
    snapshot, run = original.fixture()
    for b in (snapshot["binding"], run["binding"]):
        b["engine"] = "native_coatnet_bag12_matched_source_neighbors_v1"
        b["input_policy"] = copy.deepcopy(publisher.POLICY)
    return snapshot, run


def test_different_engines_cannot_be_published_under_each_others_prefix():
    snapshot, run = fixture()
    publisher.validate_snapshot(snapshot, run, 1)
    with pytest.raises(ValueError):
        original.publisher.validate_snapshot(snapshot, run, 1)
    with pytest.raises(ValueError):
        publisher.validate_snapshot(*original.fixture(), 1)
    for key, value in [("input_policy", {}), ("loss_policy", {"unadmitted": True})]:
        snapshot, run = fixture()
        snapshot["binding"][key] = run["binding"][key] = value
        with pytest.raises(ValueError):
            publisher.validate_snapshot(snapshot, run, 1)


def test_exactly_once_commit_and_verified_readonly_reconciliation(tmp_path, monkeypatch):
    monkeypatch.setattr(original, "publisher", publisher)
    original.test_one_upload_then_verified_readonly_reconcile(tmp_path, monkeypatch)
