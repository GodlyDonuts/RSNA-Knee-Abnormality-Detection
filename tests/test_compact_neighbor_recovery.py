import pytest

from scripts.newton.run_compact_neighbor_recovery import validate_admission


def admitted():
    return {
        "status": "matched_source_neighbor_inputs_admitted",
        "source_git": "3803b717b69db6c5e89e11724c4daee15490cbe2",
        "node": "evc33",
        "fitting_count": 2247,
        "outer_count": 1156,
        "center_channel_bit_equal": True,
        "original_fitting_outer_release_masks_unchanged": True,
        "donor_windows_changed": 1,
        "GPU_used": False,
    }


def test_admission_is_specific_to_real_matched_center_input_evidence():
    validate_admission(admitted())
    for key, value in [
        ("center_channel_bit_equal", False),
        ("fitting_count", 2603),
        ("donor_windows_changed", 0),
        ("source_git", "dcfff319cb7bd7bbae679a20da2b1b9268e8d680"),
        ("status", "native_input_gate_passed"),
        ("GPU_used", True),
    ]:
        with pytest.raises(ValueError, match="admission"):
            validate_admission({**admitted(), key: value})


def test_full_fit_refuses_old_or_incomplete_recovery():
    from scripts.newton.run_compact_neighbor_production import verify_acceptance

    a = admitted()
    from rsna.competitive.compact_source_neighbors import ENGINE, POLICY

    a["input_policy"] = dict(POLICY)
    r = {
        "status": "actual_matched_source_neighbor_H100_recovery_accepted",
        "source_git": a["source_git"],
        "node": "evc33",
        "GPU": "NVIDIA H100 PCIe",
        "input_policy": a["input_policy"],
        "engine": ENGINE,
        "initial_head_and_official_encoder_match_original_N3": True,
        "diagnostic_only": True,
        "comparison": {
            "complete_model_BN_AdamW_OneCycle_counters_sampler_allRNG_history_equal": True,
            "exact_final16x12_CSV_equal": True,
            "actual_optimizer_steps": 4,
            "BN_counters_checked": 25,
            "fitting_bag_forwards": 32,
            "clinical_accuracy_claim": False,
        },
        "phases": [
            {"phase": n, "returncode": code}
            for n, code in [("full", 0), ("stop1", 75), ("resume", 0)]
        ],
    }
    verify_acceptance(a, r)
    import copy

    for key, value in [
        ("status", "old_N3_recovery"),
        ("initial_head_and_official_encoder_match_original_N3", False),
        ("GPU", "NVIDIA Tesla T4"),
    ]:
        with pytest.raises(ValueError):
            verify_acceptance(a, {**r, key: value})
    for field in [
        "exact_final16x12_CSV_equal",
        "complete_model_BN_AdamW_OneCycle_counters_sampler_allRNG_history_equal",
    ]:
        bad = copy.deepcopy(r)
        bad["comparison"][field] = False
        with pytest.raises(ValueError):
            verify_acceptance(a, bad)


def test_relocated_endpoint_requires_explicit_matching_node():
    relocated = {**admitted(), "node": "evc39"}
    with pytest.raises(ValueError):
        validate_admission(relocated)
    validate_admission(relocated, "evc39")
    with pytest.raises(ValueError):
        validate_admission(admitted(), "evc39")
    with pytest.raises(ValueError):
        validate_admission({**admitted(), "node": "evc40"}, "evc40")


def test_relocated_cache_binding_preserves_all_scientific_fields(tmp_path):
    from scripts.newton.probe_compact_neighbor_relocated import validate_relocated_binding

    original = {
        "cache": "/old/cache",
        "source_sha256": {"module": "pinned"},
        "roster": "fixed",
        "policy": "exact",
    }
    relocated = {**original, "cache": str(tmp_path.resolve())}
    validate_relocated_binding(relocated, original, tmp_path)
    with pytest.raises(ValueError):
        validate_relocated_binding({**relocated, "roster": "different"}, original, tmp_path)
    with pytest.raises(ValueError):
        validate_relocated_binding({**relocated, "cache": "/wrong/cache"}, original, tmp_path)
