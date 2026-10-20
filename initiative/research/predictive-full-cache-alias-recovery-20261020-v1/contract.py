"""Narrow admission for the actual failed producer and one reviewed alias fix."""

from __future__ import annotations

import hashlib
import json
import re

CASES = "rsna/competitive/predictive/cases.py"
STATUS = "actual_duplicate_acquisition_renderer_regression_accepted"


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
