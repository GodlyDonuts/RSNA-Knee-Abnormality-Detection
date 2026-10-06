"""The launcher must not disappear from the workload RAM/deadline check."""

from copy import deepcopy

import pytest

from scripts.kaggle.notebook_memory import include_notebook_memory


def fixture():
    components = {
        "parent": 100,
        "resident_workers": 200,
        "preparation_workers": 400,
        "resource_tracker": 10,
    }
    return {
        "process_memory_accounting": {
            "status": "all_live_workload_process_lifetime_RSS_peaks_accounted",
            "process_counts": {
                "parent": 1,
                "resident_workers": 2,
                "preparation_workers": 2,
                "resource_tracker": 1,
            },
            "component_peak_rss_upper_bytes": components,
            "process_tree_peak_rss_upper_bytes": 710,
            "snapshot_before_worker_shutdown": True,
            "total_cgroup_memory_or_page_cache_measured": False,
        },
        "conservative_process_tree_peak_rss_upper_bytes": 710,
        "original_FP32_comparison": {"exact_prediction_values_equal": True},
    }


def test_notebook_peak_is_added_without_altering_model_evidence_or_source_record():
    record = fixture()
    original = deepcopy(record)
    result = include_notebook_memory(record, notebook_peak_bytes=900, elapsed_seconds=600)
    assert result["notebook_and_workload_process_peak_rss_upper_bytes"] == 1610
    assert result["notebook_and_workload_processes_accounted"] == 7
    assert result["original_FP32_comparison"] == original["original_FP32_comparison"]
    assert record == original


@pytest.mark.parametrize("elapsed", [0, -1, 1801, float("nan"), float("inf"), True])
def test_bootstrap_cannot_be_omitted_from_deadline(elapsed):
    with pytest.raises(ValueError):
        include_notebook_memory(fixture(), notebook_peak_bytes=900, elapsed_seconds=elapsed)


def test_complete_workload_can_still_exceed_limit_when_notebook_is_added():
    with pytest.raises(ValueError, match="RAM"):
        include_notebook_memory(fixture(), notebook_peak_bytes=29 * 2**30, elapsed_seconds=600)


@pytest.mark.parametrize(
    "key,value",
    [
        ("process_counts", {"parent": 1}),
        ("snapshot_before_worker_shutdown", False),
        ("process_tree_peak_rss_upper_bytes", 20),
    ],
)
def test_incomplete_or_inconsistent_workload_cannot_be_accepted(key, value):
    record = fixture()
    record["process_memory_accounting"][key] = value
    with pytest.raises(ValueError):
        include_notebook_memory(record, notebook_peak_bytes=900, elapsed_seconds=600)
