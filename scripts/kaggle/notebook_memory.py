"""Include the live notebook launcher in the private workload RSS bound."""

from __future__ import annotations


def include_notebook_memory(record: dict, *, notebook_peak_bytes: int, elapsed_seconds: float):
    """The notebook calls this after its measured inference subprocess finishes.

    The launcher outlives inference, so its own lifetime high-water mark covers
    its memory during preparation/inference as well as bootstrap. Independent
    process peaks are summed conservatively. This is not cgroup/page-cache RAM.
    """
    expected = {"parent": 1, "resident_workers": 2, "preparation_workers": 2, "resource_tracker": 1}
    measured = record["process_memory_accounting"]
    components = measured["component_peak_rss_upper_bytes"]
    if (
        measured["status"] != "all_live_workload_process_lifetime_RSS_peaks_accounted"
        or measured["process_counts"] != expected
        or measured["snapshot_before_worker_shutdown"] is not True
        or measured["total_cgroup_memory_or_page_cache_measured"] is not False
        or set(components) != set(expected)
        or any(type(value) is not int or value <= 0 for value in components.values())
        or type(notebook_peak_bytes) is not int
        or notebook_peak_bytes <= 0
        or type(elapsed_seconds) not in (int, float)
        or not 0 < elapsed_seconds <= 1800
    ):
        raise ValueError("Complete notebook/workload RSS and bounded wall time required")
    workload = sum(components.values())
    if (
        measured["process_tree_peak_rss_upper_bytes"] != workload
        or record["conservative_process_tree_peak_rss_upper_bytes"] != workload
    ):
        raise ValueError("Inconsistent original workload RSS arithmetic")
    upper = workload + notebook_peak_bytes
    if upper > 29 * 2**30:
        raise ValueError("Notebook and all workload process RSS exceeds the RAM envelope")
    return record | {
        "notebook_parent_peak_rss_bytes": notebook_peak_bytes,
        "notebook_and_workload_process_peak_rss_upper_bytes": upper,
        "notebook_and_workload_processes_accounted": 7,
        "notebook_elapsed_seconds_including_bootstrap": elapsed_seconds,
        "total_cgroup_memory_or_page_cache_measured": False,
    }
