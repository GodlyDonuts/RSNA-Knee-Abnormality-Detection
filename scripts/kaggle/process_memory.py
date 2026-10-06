"""Account Linux lifetime RSS peaks while every workload worker is still alive."""

from __future__ import annotations

import os
from pathlib import Path


def linux_peak_rss(pid: int, *, proc_root: Path = Path("/proc")) -> int:
    """Read the kernel high-water mark, not a late sample of current RSS."""
    rows = {}
    for line in (proc_root / str(pid) / "status").read_text().splitlines():
        key, separator, value = line.partition(":")
        if separator:
            rows[key] = value.strip()
    peak = rows.get("VmHWM", "").split()
    if rows.get("Pid") != str(pid) or len(peak) != 2 or peak[1] != "kB":
        raise ValueError("Missing original process identity or Linux lifetime RSS peak")
    value = int(peak[0]) * 1024
    if value <= 0:
        raise ValueError("Invalid process peak RSS")
    return value


class LinuxProcess:
    """Use the mounted kernel process namespace without an extra dependency."""

    def __init__(self, pid, proc_root):
        self.pid, self.proc_root = pid, proc_root

    def cmdline(self):
        return (self.proc_root / str(self.pid) / "cmdline").read_bytes().decode().split("\0")

    def children(self, *, recursive):
        if recursive is not True:
            raise ValueError("Complete descendant accounting required")
        discovered, pending = set(), [self.pid]
        while pending:
            parent = pending.pop()
            tasks = list((self.proc_root / str(parent) / "task").iterdir())
            if not tasks:
                raise ValueError("Process tasks disappeared before accounting")
            for task in tasks:
                for token in (task / "children").read_text().split():
                    pid = int(token)
                    if pid <= 0 or pid == self.pid:
                        raise ValueError("Invalid descendant process identity")
                    if pid not in discovered:
                        discovered.add(pid)
                        pending.append(pid)
                        if len(discovered) > 16:
                            raise ValueError("Unexpectedly large workload process tree")
        return [LinuxProcess(pid, self.proc_root) for pid in sorted(discovered)]


def workload_memory(*, resident_pids, preparation_pids, process=None, proc_root=Path("/proc")):
    """Sum lifetime peaks of parent, both model/preparation workers and tracker.

    Workers must remain alive until this snapshot. Summing independent peaks
    overestimates their simultaneous RSS, but covers preparation/model overlap.
    Reject missing or unclassified descendants rather than accepting a partial
    model-only figure. This measures process RSS, not total cgroup/page-cache use.
    """
    process = LinuxProcess(os.getpid(), proc_root) if process is None else process
    residents, preparations = set(resident_pids), set(preparation_pids)
    if len(residents) != 2 or len(preparations) != 2 or residents & preparations:
        raise ValueError("Two distinct resident and two distinct preparation workers required")
    children = process.children(recursive=True)
    pids = {child.pid for child in children}
    if len(pids) != len(children) or not residents | preparations <= pids:
        raise ValueError("Workload worker absent at memory snapshot")
    extra = pids - residents - preparations
    if len(extra) != 1:
        raise ValueError("Exactly one live multiprocessing resource tracker required")
    tracker = next(child for child in children if child.pid in extra)
    if not any("multiprocessing.resource_tracker" in argument for argument in tracker.cmdline()):
        raise ValueError("Unclassified process would make the RSS accounting incomplete")
    groups = {
        "parent": {process.pid},
        "resident_workers": residents,
        "preparation_workers": preparations,
        "resource_tracker": extra,
    }
    peaks = {
        role: sum(linux_peak_rss(pid, proc_root=proc_root) for pid in ids)
        for role, ids in groups.items()
    }
    return {
        "status": "all_live_workload_process_lifetime_RSS_peaks_accounted",
        "process_counts": {role: len(ids) for role, ids in groups.items()},
        "component_peak_rss_upper_bytes": peaks,
        "process_tree_peak_rss_upper_bytes": sum(peaks.values()),
        "snapshot_before_worker_shutdown": True,
        "total_cgroup_memory_or_page_cache_measured": False,
    }
