"""No remote operations: exercise original terminal/dispatch and storage gates."""

import importlib.util
import json
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "initiative/research/predictive-full-cache-continuation-20261020-v1"
SPEC = importlib.util.spec_from_file_location("continuation_owner", CONTROL / "owner.py")
owner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(owner)


def state(value="RUNNING", *, code="0:0", empty=False):
    return {"state": value, "exit_code": code, "queue_empty": empty}


def originals(raw=None, context=None):
    return {
        "978318": raw or state(),
        "978414": context or state(),
    }


@pytest.mark.parametrize("status", [None, "PENDING", "RUNNING", "COMPLETING", "SUSPENDED"])
def test_live_or_unobserved_raw_never_allocates_even_if_context_failed(status):
    jobs = originals(
        state(status, empty=status is None), state("FAILED", code="1:0", empty=True)
    )
    assert owner.next_action(jobs, {}) == ("wait",)


def test_accounting_timeout_with_live_queue_cannot_launch():
    assert owner.next_action(originals(state("TIMEOUT", code="0:15")), {}) == ("wait",)


def test_exact_timeout_launches_raw_once_and_waits_for_allocated_identity():
    jobs = originals(state("TIMEOUT", code="0:15", empty=True))
    assert owner.next_action(jobs, {}) == ("dispatch", "raw")
    assert owner.next_action(jobs, {"raw": {"returncode": None}}) == ("wait",)


@pytest.mark.parametrize("status", ["FAILED", "CANCELLED", "NODE_FAIL", "OUT_OF_MEMORY", "COMPLETED"])
def test_raw_non_timeout_failure_never_falls_back_to_new_job(status):
    jobs = originals(state(status, code="1:0", empty=True))
    with pytest.raises(RuntimeError):
        owner.next_action(jobs, {})


def test_completed_original_only_collects_and_never_dispatches_successor():
    jobs = originals(state("COMPLETED", empty=True), state("COMPLETED", empty=True))
    assert owner.next_action(jobs, {}) == ("collect", "raw", "978318", "raw_original")
    phases = {"raw_collect": {"returncode": 0, "accepted": True}}
    assert owner.next_action(jobs, phases) == ("collect", "context", "978414", "context_original")
    phases["context_collect"] = {"returncode": 0, "accepted": True}
    assert owner.next_action(jobs, phases) == ("complete",)


@pytest.mark.parametrize("ready,allocated", [(False, "RUNNING"), (True, "PENDING"), (True, None)])
def test_context_requires_actual_started_and_hash_bound_raw_intent(ready, allocated):
    jobs = originals(state("TIMEOUT", empty=True), state("TIMEOUT", empty=True))
    jobs["900000"] = state(allocated, empty=allocated is None)
    phases = {"raw": {"job_id": "900000", "ready": ready, "returncode": None}}
    assert owner.next_action(jobs, phases) == ("wait",)


def test_context_starts_incrementally_after_original_terminal_and_live_raw_successor():
    jobs = originals(state("TIMEOUT", empty=True), state("FAILED", code="1:0", empty=True))
    jobs["900000"] = state()
    phases = {"raw": {"job_id": "900000", "ready": True, "returncode": None}}
    assert owner.next_action(jobs, phases) == ("dispatch", "context")
    phases["context"] = {"returncode": None}
    assert owner.next_action(jobs, phases) == ("wait",)


def test_context_never_overlaps_original_context_owner():
    jobs = originals(state("TIMEOUT", empty=True))
    jobs["900000"] = state()
    phases = {"raw": {"job_id": "900000", "ready": True, "returncode": None}}
    assert owner.next_action(jobs, phases) == ("wait",)


def test_transport_uncertainty_stops_without_a_second_intent():
    jobs = originals(state("TIMEOUT", empty=True))
    for phase in ({"returncode": 255}, {"returncode": None, "input_error": True}):
        with pytest.raises(RuntimeError, match="no automatic retry"):
            owner.next_action(jobs, {"raw": phase})


def test_both_successors_require_actual_terminal_and_collected_receipts():
    jobs = originals(state("TIMEOUT", empty=True), state("TIMEOUT", empty=True))
    jobs.update({"900000": state("COMPLETED", empty=True), "900001": state("COMPLETED", empty=True)})
    phases = {
        "raw": {"job_id": "900000", "ready": True, "returncode": 0},
        "context": {"job_id": "900001", "ready": True, "returncode": 0},
    }
    assert owner.next_action(jobs, phases) == ("collect", "raw", "900000", "raw")
    phases["raw_collect"] = {"returncode": 0, "accepted": True}
    assert owner.next_action(jobs, phases) == ("collect", "context", "900001", "context")
    phases["context_collect"] = {"returncode": 0, "accepted": True}
    assert owner.next_action(jobs, phases) == ("complete",)


def test_terminal_accounting_with_stale_queue_waits_for_consistent_observation():
    jobs = originals(state("COMPLETED", empty=True), state("COMPLETED", empty=False))
    phases = {"raw_collect": {"returncode": 0, "accepted": True}}
    assert owner.next_action(jobs, phases) == ("wait",)
    jobs = originals(state("TIMEOUT", empty=True), state("TIMEOUT", empty=True))
    jobs["900000"] = state("COMPLETED", empty=False)
    phases = {"raw": {"job_id": "900000", "ready": True, "returncode": None}}
    assert owner.next_action(jobs, phases) == ("wait",)
    jobs["900000"] = state("COMPLETED", empty=True)
    jobs["900001"] = state("COMPLETED", empty=False)
    phases.update({
        "raw_collect": {"returncode": 0, "accepted": True},
        "context": {"job_id": "900001", "returncode": None},
    })
    assert owner.next_action(jobs, phases) == ("wait",)


def test_terminal_parser_ignores_steps_but_requires_exact_root():
    result = owner.parse_jobs(
        "978318|TIMEOUT|0:15|\n978318.batch|CANCELLED|0:15|\n978414|RUNNING|0:0|\n",
        "978414|RUNNING\n", ["978318", "978414", "900000"],
    )
    assert owner.terminal(result["978318"], {"TIMEOUT"})
    assert not result["978414"]["queue_empty"]
    assert result["900000"]["state"] is None
    with pytest.raises(AssertionError):
        owner.parse_jobs("978318|TIMEOUT|0:15|\n978318|COMPLETED|0:0|\n", "", ["978318"])


def quota(*, free_kb=30 * 2**20, free_files=5400):
    return f"/home/sa305415 100 {100 + free_kb} 0 - 100 {100 + free_files} 0\n"


def test_quota_preserves_two_successors_and_all_other_future_files():
    result = owner.admit_quota(quota())
    assert result["actual_free_files"] - 368 == 5032
    assert owner.FUNDING["other_node_future_bytes"] == 60 * 2**30 + 570 * 2**20
    with pytest.raises(AssertionError):
        owner.admit_quota(quota(free_files=5368))
    with pytest.raises(AssertionError):
        owner.admit_quota(quota(free_kb=20 * 2**20 + 96 * 1024))


def test_exclusive_atomic_receipt_never_replaces_existing_intent(tmp_path):
    path = tmp_path / "intent.json"
    owner.once(path, {"first": True})
    before = path.read_bytes()
    with pytest.raises(FileExistsError):
        owner.once(path, {"second": True})
    assert path.read_bytes() == before and list(tmp_path.iterdir()) == [path]
    assert path.stat().st_mode & 0o077 == 0


def test_exact_archived_numerical_modules_are_read_from_original_commits():
    raw = owner.original_modules(owner.RAW_SOURCE)
    context = owner.original_modules(owner.CONTEXT_SOURCE)
    assert "rsna/competitive/predictive/context_cache.py" not in raw
    assert "rsna/competitive/predictive/context_cache.py" in context
    for name in ("inputs.py", "cases.py", "supports.py", "cache.py"):
        path = "rsna/competitive/predictive/" + name
        assert raw[path] == context[path]
        assert raw[path] == subprocess.check_output(
            ["git", "show", owner.RAW_SOURCE + ":" + path], cwd=ROOT
        ).decode()
    assert raw["rsna/__init__.py"] == raw["rsna/competitive/__init__.py"] == ""


def test_allocations_are_minimum_cpu_zero_gpu_no_dependency_waits():
    class Transport:
        COMMAND = "stdin-only"

    for role, cpus in (("raw", 4), ("context", 1), ("collect", 1)):
        args = owner.command(role, Transport)
        assert "--ntasks=1" in args and "--gres=none" in args
        assert "--cpus-per-task=" + str(cpus) in args
        assert "--immediate=60" in args and not any("--dependency" in item for item in args)
        assert "CUDA_VISIBLE_DEVICES=" in args and "stdin-only" == args[-1]


def test_wrapper_reports_only_safe_exception_fields_and_never_the_credential():
    program = owner.wrapped("raise ValueError('hf_secret_must_not_escape')", "collect")
    result = subprocess.run(
        ["python3", "-c", program], env={"PATH": "/usr/bin:/bin", "SLURM_JOB_ID": "900000"},
        capture_output=True, text=True,
    )
    assert result.returncode == 1
    assert "hf_secret" not in result.stdout + result.stderr
    assert json.loads(result.stdout)["CPU_job"] == "900000"
    assert json.loads(result.stderr)["error_type"] == "ValueError"


def test_context_ready_notice_must_match_exact_source_and_job(tmp_path):
    class Process:
        @staticmethod
        def poll():
            return None

    events = [
        {"owner_event": "allocation_started", "CPU_job": "900000"},
        {
            "owner_event": "original_successor_intent_ready", "CPU_job": "900000",
            "source_git": "b" * 40, "intent_sha256": "c" * 64,
        },
    ]
    (tmp_path / "raw.stdout").write_text("".join(json.dumps(event) + "\n" for event in events))
    (tmp_path / "raw.stderr").write_text("")
    phase = {"process": Process(), "offset": 0, "role": "raw", "source_git": "a" * 40}
    with pytest.raises(AssertionError):
        owner.refresh("raw", phase, tmp_path, tmp_path)
    assert not phase.get("ready")
