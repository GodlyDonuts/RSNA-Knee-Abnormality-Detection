"""Narrow admission for the actual failed producer and one reviewed alias fix."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

CASES = "rsna/competitive/predictive/cases.py"
STATUS = "actual_duplicate_acquisition_renderer_regression_accepted"
PRIOR_ROOTS = (
    "/tmp/rsna-predictive-full-raw-alias-recovery-20261020-v1",
    "/tmp/rsna-predictive-full-context-alias-recovery-20261020-v1",
)


def digest(body):
    return hashlib.sha256(body).hexdigest()


def regression(body, expected_sha256, *, original_cases, corrected_cases):
    """Require original real-input evidence before using the corrected module."""
    assert re.fullmatch(r"[a-f0-9]{64}", expected_sha256)
    assert digest(body) == expected_sha256
    result = json.loads(body)
    assert result["status"] == STATUS
    assert str(result["original_job"]) == "978318"
    assert str(result["CPU_job"]).isdecimal() and str(result["CPU_job"]) != "978318"
    assert result["original_cases_sha256"] == digest(original_cases.encode())
    assert result["corrected_cases_sha256"] == digest(corrected_cases.encode())
    assert result["original_cases_sha256"] != result["corrected_cases_sha256"]
    for key in (
        "no_source_downloads",
        "prior_successful_contexts_unchanged",
        "retained_failed_case_corrected",
    ):
        assert result[key] is True
    return result


def original_failure(record):
    return (
        record.get("state") == "FAILED"
        and record.get("exit_code") == "1:0"
        and record.get("queue_empty") is True
    )


def prior_attempt(body, expected_sha256):
    """The first successor must have failed before either output root existed."""
    assert re.fullmatch(r"[a-f0-9]{64}", expected_sha256)
    assert digest(body) == expected_sha256
    value = json.loads(body)
    assert value["status"] == "actual_alias_recovery_v1_failed_before_output_reconciled"
    assert str(value["failed_job"]) == "979573"
    assert value["terminal_state"] == "FAILED" and value["exit_code"] == "1:0"
    assert value["live_queue_empty"] is True
    assert value["raw_output_root_absent"] is True
    assert value["context_output_root_absent"] is True
    assert type(value["new_source_downloads"]) is int and value["new_source_downloads"] == 0
    return value


def prior_attempt_live_guard(value, check_output):
    """Use account-wide queue listing; a purged terminal job is still admissible."""
    prior_attempt(
        value["prior_attempt_receipt"].encode(), value["prior_attempt_receipt_sha256"]
    )
    for root in PRIOR_ROOTS:
        assert not Path(root).exists(), "First successor has outputs; reconcile without retry"
    accounting = check_output(
        ["sacct", "-X", "-n", "-P", "-j", "979573", "--format=JobIDRaw,State,ExitCode"],
        text=True,
    )
    queue = check_output(["squeue", "-h", "-u", "sa305415", "-o", "%i|%T"], text=True)
    rows = [line.split("|") for line in accounting.splitlines() if line.split("|")[0] == "979573"]
    assert len(rows) == 1 and rows[0][1:3] == ["FAILED", "1:0"]
    assert not any(line.split("|")[0] == "979573" for line in queue.splitlines())


def renderer_versions(files, original_files):
    """Admit exactly cases.py's reviewed correction, never unrelated source."""
    names = (
        "rsna/competitive/predictive/inputs.py",
        CASES,
        "rsna/competitive/predictive/supports.py",
    )
    assert set(files) == set(original_files)
    for name in files:
        if name != CASES:
            assert files[name] == original_files[name]
    assert files[CASES] != original_files[CASES]
    return (
        {name: digest(original_files[name].encode()) for name in names},
        {name: digest(files[name].encode()) for name in names},
    )
