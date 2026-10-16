"""The larger-cohort run cannot inherit acceptance from the old population."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import test_compact_publication as original

from scripts.hf import publish_compact_expanded_epoch as publisher
from scripts.newton.run_compact_expanded_production import (
    INPUT_SHA,
    SOURCE_GIT,
    SOURCE_MANIFEST_SHA,
    verify_acceptance,
    verify_source,
)

ROOT = Path(__file__).resolve().parents[1]


def admission():
    return json.loads(
        (
            ROOT
            / "initiative/audits/compact-expanded2592-inputs-20261016-v1/actual-CPU-accepted.json"
        ).read_text()
    )


def synthetic_recovery():
    # Structural admission fixture only, not evidence that pending933382 passed.
    return {
        "status": "actual_expanded2592_H100_recovery_accepted",
        "job_id": "933382",
        "input_admission_sha256": INPUT_SHA,
        "source_git": SOURCE_GIT,
        "source_manifest_sha256": SOURCE_MANIFEST_SHA,
        "node": "evc33",
        "GPU": "NVIDIA H100 PCIe",
        "source_admission_engineering_counts": {"old_fitting": 8, "added_fitting": 8, "outer": 16},
        "initial_head_and_official_encoder_match_original_N3": True,
        "full_fitting_count": 2592,
        "full_outer_count": 1156,
        "full_optimizer_updates": 5184,
        "diagnostic_only": True,
        "clinical_accuracy_claim": False,
        "comparison": {
            "complete_model_BN_AdamW_OneCycle_counters_sampler_allRNG_history_equal": True,
            "exact_final16x12_CSV_equal": True,
            "actual_optimizer_steps": 4,
            "BN_counters_checked": 25,
            "fitting_bag_forwards": 32,
            "clinical_accuracy_claim": False,
        },
        "phases": [
            {"phase": n, "returncode": r} for n, r in [("full", 0), ("stop1", 75), ("resume", 0)]
        ],
    }


def test_complete_expanded_recovery_contract_is_structurally_accepted():
    verify_acceptance(admission(), synthetic_recovery())


@pytest.mark.parametrize(
    "name,value",
    [
        ("job_id", "932859"),
        ("source_git", "dcfff319cb7bd7bbae679a20da2b1b9268e8d680"),
        ("input_admission_sha256", "0" * 64),
        ("full_fitting_count", 2247),
        ("full_optimizer_updates", 4496),
        ("full_outer_count", 16),
        ("GPU", "Tesla T4"),
        ("initial_head_and_official_encoder_match_original_N3", False),
        ("diagnostic_only", False),
        ("clinical_accuracy_claim", True),
    ],
)
def test_old_model_or_population_and_weak_recovery_cannot_admit_full_fit(name, value):
    recovered = synthetic_recovery()
    recovered[name] = value
    with pytest.raises(ValueError):
        verify_acceptance(admission(), recovered)


def test_actual_complete_state_and_csv_cannot_be_replaced_by_only_finite_loss():
    for key in [
        "complete_model_BN_AdamW_OneCycle_counters_sampler_allRNG_history_equal",
        "exact_final16x12_CSV_equal",
    ]:
        recovered = synthetic_recovery()
        recovered["comparison"][key] = False
        with pytest.raises(ValueError):
            verify_acceptance(admission(), recovered)
    recovered = synthetic_recovery()
    recovered["phases"][1]["returncode"] = 0
    with pytest.raises(ValueError):
        verify_acceptance(admission(), recovered)


def test_exact_source_rejects_changed_extra_or_symlinked_members(tmp_path):
    p = tmp_path / "rsna/example.py"
    p.parent.mkdir()
    p.write_text("x=1\n")
    manifest = {
        "source_git": SOURCE_GIT,
        "files": {"rsna/example.py": hashlib.sha256(p.read_bytes()).hexdigest()},
    }
    verify_source(tmp_path, manifest)
    p.write_text("x=2\n")
    with pytest.raises(ValueError):
        verify_source(tmp_path, manifest)
    p.write_text("x=1\n")
    extra = p.with_name("extra.py")
    extra.write_text("pass\n")
    with pytest.raises(ValueError):
        verify_source(tmp_path, manifest)
    extra.unlink()
    p.unlink()
    p.symlink_to(tmp_path / "elsewhere")
    with pytest.raises(ValueError):
        verify_source(tmp_path, manifest)


def expanded_snapshot():
    snapshot, run = original.fixture()
    b = snapshot["binding"]
    snapshot["optimizer_steps"] = 324
    b.update(fitting_count=2592, total_steps=5184)
    b["supervision"]["overlay_sha256"] = admission()["overlay"]["overlay_sha256"]
    b["fitting_population"] = {
        "version": "original_N3_plus345_fitting_only_v1",
        "population": admission()["population"],
        "CPU_input_sha256": INPUT_SHA,
        "metadata_union_sha256": admission()["metadata_union_index_sha256"],
        "sixteen_passes_updates": 5184,
        "compute_matched_to_original": False,
    }
    run["binding"] = copy.deepcopy(b)
    return snapshot, run


def test_expanded_epoch_is_not_an_original_or_weighted_snapshot():
    s, r = expanded_snapshot()
    publisher.validate_snapshot(s, r, 1)
    with pytest.raises(ValueError):
        original.publisher.validate_snapshot(s, r, 1)
    for key, value in [
        ("fitting_count", 2247),
        ("total_steps", 4496),
        ("input_policy", {"neighbor": True}),
        ("loss_policy", {"weighted": True}),
    ]:
        bad = copy.deepcopy(s)
        bad["binding"][key] = value
        other = {"binding": copy.deepcopy(bad["binding"]), "diagnostic_only": False}
        with pytest.raises(ValueError):
            publisher.validate_snapshot(bad, other, 1)
    bad = copy.deepcopy(s)
    bad["binding"]["fitting_population"]["population"]["outer_endpoint_count"] = 16
    with pytest.raises(ValueError):
        publisher.validate_snapshot(
            bad, {"binding": copy.deepcopy(bad["binding"]), "diagnostic_only": False}, 1
        )


def test_expanded_publication_never_retries_an_uncertain_commit(tmp_path, monkeypatch):
    monkeypatch.setattr(original, "publisher", publisher)
    original.test_one_upload_then_verified_readonly_reconcile(tmp_path, monkeypatch)
