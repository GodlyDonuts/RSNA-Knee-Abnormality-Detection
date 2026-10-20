"""Failure-specific coordinator admission; all calls remain local."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "initiative/research/predictive-full-cache-alias-recovery-20261020-v1"


def module(name):
    spec = importlib.util.spec_from_file_location("alias_" + name, CONTROL / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


owner, contract = module("owner"), module("contract")


def state(name="FAILED", code="1:0", empty=True):
    return {"state": name, "exit_code": code, "queue_empty": empty}


def jobs(raw=None, context=None):
    return {"978318": raw or state(), "978414": context or state()}


def receipt():
    return {
        "status": contract.STATUS,
        "original_job": "978318",
        "CPU_job": "979999",
        "original_cases_sha256": contract.digest(b"old"),
        "corrected_cases_sha256": contract.digest(b"new"),
        "no_source_downloads": True,
        "prior_successful_contexts_unchanged": True,
        "retained_failed_case_corrected": True,
    }


def admit(value):
    body = json.dumps(value).encode()
    return contract.regression(
        body, contract.digest(body), original_cases="old", corrected_cases="new"
    )


def test_original_failed_1_0_is_required_and_stale_queue_waits():
    assert owner.next_action(jobs(), {}) == ("dispatch", "raw")
    assert owner.next_action(jobs(state(empty=False)), {}) == ("wait",)
    for value in (state("TIMEOUT"), state(code="2:0"), state("NODE_FAIL")):
        with pytest.raises(RuntimeError):
            owner.next_action(jobs(value), {})


def test_existing_attempt_or_unknown_original_never_duplicates():
    assert owner.next_action(jobs(), {"raw": {"returncode": None}}) == ("wait",)
    assert owner.next_action(jobs(state(None)), {}) == ("wait",)
    with pytest.raises(RuntimeError):
        owner.next_action(jobs(), {"raw": {"returncode": 255}})


def test_context_waits_actual_raw_allocation_and_original_context_terminal():
    phases = {"raw": {"job_id": "980000", "ready": True, "returncode": None}}
    actual = jobs()
    actual["980000"] = state("RUNNING", "0:0", False)
    assert owner.next_action(actual, phases) == ("dispatch", "context")
    actual["978414"] = state("RUNNING", "0:0", False)
    assert owner.next_action(actual, phases) == ("wait",)


def test_actual_regression_binds_exact_corrected_and_original_code():
    assert admit(receipt())["status"] == contract.STATUS
    for key in (
        "no_source_downloads",
        "prior_successful_contexts_unchanged",
        "retained_failed_case_corrected",
    ):
        value = receipt()
        value[key] = False
        with pytest.raises(AssertionError):
            admit(value)
    value = receipt()
    value["corrected_cases_sha256"] = contract.digest(b"unreviewed")
    with pytest.raises(AssertionError):
        admit(value)


def test_changed_receipt_bytes_are_rejected():
    body = json.dumps(receipt()).encode()
    with pytest.raises(AssertionError):
        contract.regression(
            body + b" ", contract.digest(body), original_cases="old", corrected_cases="new"
        )


def test_only_cases_correction_can_change_renderer_sources():
    old = {
        contract.CASES: "old",
        "rsna/competitive/predictive/inputs.py": "inputs",
        "rsna/competitive/predictive/supports.py": "supports",
        "rsna/competitive/predictive/cache.py": "cache",
    }
    new = {**old, contract.CASES: "new"}
    original, corrected = contract.renderer_versions(new, old)
    assert original[contract.CASES] != corrected[contract.CASES]
    new["rsna/competitive/predictive/inputs.py"] = "altered"
    with pytest.raises(AssertionError):
        contract.renderer_versions(new, old)


def test_recovery_reuses_successor_log_reservation_and_has_time_to_finish():
    class Transport:
        COMMAND = "stdin-only"

    assert owner.FUNDING["owner_total_shared_future_bytes"] == 121 * 2**20
    assert owner.FUNDING["owner_total_shared_future_files"] == 376
    assert owner.FUNDING["other_node_future_bytes"] == 62 * 2**30 + 594 * 2**20
    for role, cpus in (("raw", 4), ("context", 1)):
        args = owner.command(role, Transport)
        assert "--time=20:00:00" in args and "--gres=none" in args
        assert f"--cpus-per-task={cpus}" in args
        assert "--immediate=60" in args and "--nodelist=evc33" in args


def test_original_controls_remain_independent_and_frozen():
    original = ROOT / "initiative/research/predictive-full-cache-continuation-20261020-v1"
    assert (CONTROL / "reconcile.py").read_bytes() == (original / "reconcile.py").read_bytes()
    assert (CONTROL / "collect.py.txt").read_bytes() == (original / "collect.py.txt").read_bytes()
    assert owner.REMOTE_ROOTS["raw"] != (
        "/tmp/rsna-predictive-full-cache-continuation-20261020-v1"
    )
