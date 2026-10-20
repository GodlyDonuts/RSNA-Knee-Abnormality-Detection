"""A purged terminal job must not become a queue-query failure or live conflict."""

import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
CONTROL = ROOT / "initiative/research/predictive-full-cache-alias-recovery-20261020-v2"


def load(name):
    spec = importlib.util.spec_from_file_location("alias_v2_" + name, CONTROL / (name + ".py"))
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


owner, recon, contract = load("owner"), load("reconcile"), load("contract")


def failed():
    return {"state": "FAILED", "exit_code": "1:0", "queue_empty": True}


def previous():
    return {
        "status": "actual_alias_recovery_v1_failed_before_output_reconciled",
        "failed_job": "979573",
        "terminal_state": "FAILED",
        "exit_code": "1:0",
        "live_queue_empty": True,
        "raw_output_root_absent": True,
        "context_output_root_absent": True,
        "new_source_downloads": 0,
    }


def payload(value=None):
    body = json.dumps(previous() if value is None else value)
    return {
        "prior_attempt_receipt": body,
        "prior_attempt_receipt_sha256": contract.digest(body.encode()),
    }


def test_purged_root_with_other_users_own_running_jobs_is_terminal():
    receipt = recon.terminal_receipt(
        "978318|FAILED|1:0|\n978318.batch|FAILED|1:0|\n",
        "980001|RUNNING\n980002|PENDING\n",
        job_id="978318",
        allowed_states=("FAILED",),
    )
    assert receipt["live_queue_empty"] and receipt["exit_code"] == "1:0"
    with pytest.raises(ValueError):
        recon.terminal_receipt(
            "978318|FAILED|1:0|\n", "978318|COMPLETING\n",
            job_id="978318", allowed_states=("FAILED",),
        )


def test_guard_uses_account_queue_and_rechecks_actual_failed_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(contract, "PRIOR_ROOTS", (str(tmp_path / "raw"), str(tmp_path / "context")))
    calls = []

    def query(args, **kwargs):
        calls.append(args)
        if args[0] == "sacct":
            return "979573|FAILED|1:0|\n"
        assert args == ["squeue", "-h", "-u", "sa305415", "-o", "%i|%T"]
        return "980010|RUNNING\n"

    contract.prior_attempt_live_guard(payload(), query)
    assert len(calls) == 2
    (tmp_path / "raw").mkdir()
    with pytest.raises(AssertionError):
        contract.prior_attempt_live_guard(payload(), query)
    assert len(calls) == 2  # output ownership is checked before another query


@pytest.mark.parametrize("key,value", [
    ("raw_output_root_absent", False),
    ("context_output_root_absent", False),
    ("new_source_downloads", 1),
    ("terminal_state", "RUNNING"),
    ("failed_job", "978318"),
])
def test_partial_or_wrong_prior_attempt_cannot_admit_a_successor(key, value):
    old = previous()
    old[key] = value
    p = payload(old)
    with pytest.raises(AssertionError):
        contract.prior_attempt(p["prior_attempt_receipt"].encode(), p["prior_attempt_receipt_sha256"])


def test_new_owner_waits_prior_job_and_does_not_treat_purged_id_as_query_failure():
    jobs = owner.parse_jobs(
        "978318|FAILED|1:0|\n978414|FAILED|1:0|\n979573|FAILED|1:0|\n",
        "980000|RUNNING\n",
        ["978318", "978414", "979573"],
    )
    assert owner.next_action(jobs, {}) == ("dispatch", "raw")
    jobs["979573"] = {"state": "RUNNING", "exit_code": "0:0", "queue_empty": False}
    assert owner.next_action(jobs, {}) == ("wait",)


def test_source_contains_no_live_queue_lookup_by_old_job_id():
    for name in ("owner.py", "program.py.txt", "context.py.txt", "collect.py.txt", "contract.py"):
        text = (CONTROL / name).read_text()
        for line in text.splitlines():
            if "squeue" in line and ("check_output" in line or "q=" in line):
                assert '"-j"' not in line and "'-j'" not in line
    assert owner.REMOTE_ROOTS["raw"].endswith("-v2")
    assert owner.REMOTE_ROOTS["context"].endswith("-v2")
