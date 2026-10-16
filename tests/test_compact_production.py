import copy
import importlib.util
import signal
from pathlib import Path
from types import SimpleNamespace

import pytest

PATH = Path(__file__).resolve().parents[1] / "scripts/newton/run_compact_bag_production.py"
spec = importlib.util.spec_from_file_location("compact_production", PATH)
production = importlib.util.module_from_spec(spec)
spec.loader.exec_module(production)


def proof():
    admission = {
        "status": "original_native_inputs_and_runtime_admitted_on_distinct_evc33",
        "source_git": "dcfff319cb7bd7bbae679a20da2b1b9268e8d680",
        "node": "evc33",
        "fitting_count": 2247,
        "outer_count": 1156,
        "same_original32_input_tensor_sha256": "183097af0095317c7a85069e73c8b2c654e39cfc12cdbc7104ff3139a30bfe9b",
    }
    recovery = {
        "complete_model_BN_AdamW_OneCycle_counters_sampler_allRNG_history_equal": True,
        "exact_final16x12_CSV_equal": True,
        "actual_optimizer_steps": 4,
        "BN_counters_checked": 25,
        "fitting_bag_forwards": 32,
        "diagnostic_only": True,
        "source_git": admission["source_git"],
        "node": "evc33",
        "actual_cuda_peak_bytes": [5, 4, 5],
        "phases": [
            {"phase": "full", "actual_integer_returncode": 0},
            {"phase": "stop1", "actual_integer_returncode": 75},
            {"phase": "resume", "actual_integer_returncode": 0},
        ],
    }
    return admission, recovery


def test_actual_three_phase_acceptance_contract():
    a, b = proof()
    production.verify_acceptance(a, b)


@pytest.mark.parametrize(
    "field,value",
    [
        ("actual_optimizer_steps", 1),
        ("BN_counters_checked", 24),
        ("fitting_bag_forwards", 16),
        ("exact_final16x12_CSV_equal", False),
        ("actual_cuda_peak_bytes", [5]),
        ("actual_cuda_peak_bytes", [5, 0, 5]),
        ("diagnostic_only", False),
        ("node", "other"),
        ("phases", [{"phase": "full", "actual_integer_returncode": 0}]),
    ],
)
def test_incomplete_or_mismatched_evidence_cannot_start_full_fit(field, value):
    a, b = proof()
    b[field] = value
    with pytest.raises(ValueError):
        production.verify_acceptance(a, b)


def test_actual_input_rosters_and_tensor_binding_are_required():
    a, b = proof()
    for field in ("fitting_count", "outer_count", "same_original32_input_tensor_sha256"):
        changed = copy.deepcopy(a)
        changed[field] = None
        with pytest.raises(ValueError):
            production.verify_acceptance(changed, b)


def test_walltime_signal_is_forwarded_only_to_live_owned_child_group(monkeypatch):
    calls = []
    monkeypatch.setattr(production.os, "killpg", lambda pid, sig: calls.append((pid, sig)))
    production.forward_signal(SimpleNamespace(pid=1234, poll=lambda: None), signal.SIGUSR1)
    assert calls == [(1234, signal.SIGUSR1)]
    production.forward_signal(SimpleNamespace(pid=9999, poll=lambda: 0), signal.SIGTERM)
    assert calls == [(1234, signal.SIGUSR1)]
