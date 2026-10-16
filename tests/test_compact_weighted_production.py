import copy
import json
from pathlib import Path

import pytest
import test_compact_publication as original
from test_compact_weighted_recovery import admission

from scripts.hf import publish_compact_weighted_epoch as publisher
from scripts.newton.run_compact_weighted_production import verify_acceptance

ROOT = Path(__file__).resolve().parents[1]


def recovery():
    return json.loads(
        (
            ROOT
            / "initiative/audits/compact-weighted-recovery-dispatch-20261016-v1/actual-GPU-recovery.json"
        ).read_text()
    )["receipt"]


def test_actual_weighted_recovery_is_not_interchangeable_with_old_engine():
    verify_acceptance(admission(), recovery())
    for key, value in [
        ("engine", "native_coatnet_bag12_N3_full_v1"),
        ("status", "old_recovery"),
        ("loss_policy_sha256", "0" * 64),
        ("initial_head_and_official_encoder_match_original_N3", False),
    ]:
        with pytest.raises(ValueError):
            verify_acceptance(admission(), {**recovery(), key: value})
    for name in [
        "complete_model_BN_AdamW_OneCycle_counters_sampler_allRNG_history_equal",
        "exact_final16x12_CSV_equal",
    ]:
        value = copy.deepcopy(recovery())
        value["comparison"][name] = False
        with pytest.raises(ValueError):
            verify_acceptance(admission(), value)


def test_weighted_publication_binds_the_full_actual_fitting_policy():
    snapshot, run = original.fixture()
    with pytest.raises(ValueError):
        publisher.validate_snapshot(snapshot, run, 1)
    for b in (snapshot["binding"], run["binding"]):
        b["engine"] = "native_coatnet_bag12_N3_positive_weighted_v1"
        b["loss_policy"] = copy.deepcopy(admission()["policy"])
    publisher.validate_snapshot(snapshot, run, 1)
    with pytest.raises(ValueError):
        original.publisher.validate_snapshot(snapshot, run, 1)
    for key in ["fitting_uids_sha256", "observed_targets_and_masks_sha256"]:
        bad = copy.deepcopy(snapshot)
        bad["binding"]["loss_policy"][key] = "0" * 64
        other = {"binding": copy.deepcopy(bad["binding"]), "diagnostic_only": False}
        with pytest.raises(ValueError):
            publisher.validate_snapshot(bad, other, 1)


def test_weighted_publication_one_commit_then_verified_readonly_reconciliation(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(original, "publisher", publisher)
    original.test_one_upload_then_verified_readonly_reconcile(tmp_path, monkeypatch)
