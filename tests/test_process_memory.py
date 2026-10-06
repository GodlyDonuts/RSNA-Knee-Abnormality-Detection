"""Full workload RSS includes concurrent preparation and rejects incomplete trees."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.kaggle.process_memory import LinuxProcess, linux_peak_rss, workload_memory


def write_status(root, pid, kib):
    path = root / str(pid)
    path.mkdir()
    (path / "status").write_text(f"Pid:\t{pid}\nVmRSS:\t1 kB\nVmHWM:\t{kib} kB\n")


def tree(root, *, extra=(6,), omitted=()):
    for pid in range(1, 8):
        write_status(root, pid, pid * 100)
    children = [
        SimpleNamespace(pid=pid, cmdline=lambda: ["python", "multiprocessing.resource_tracker"])
        for pid in [2, 3, 4, 5, *extra]
        if pid not in omitted
    ]
    return SimpleNamespace(pid=1, children=lambda recursive: children)


def test_lifetime_peak_counts_all_six_processes_before_shutdown(tmp_path):
    process = tree(tmp_path)
    result = workload_memory(
        resident_pids=[2, 3], preparation_pids=[4, 5], process=process, proc_root=tmp_path
    )
    assert result["process_tree_peak_rss_upper_bytes"] == 2100 * 1024
    assert result["component_peak_rss_upper_bytes"]["preparation_workers"] == 900 * 1024
    assert result["component_peak_rss_upper_bytes"]["resource_tracker"] == 600 * 1024
    assert result["process_counts"] == {
        "parent": 1,
        "resident_workers": 2,
        "preparation_workers": 2,
        "resource_tracker": 1,
    }
    assert not result["total_cgroup_memory_or_page_cache_measured"]


@pytest.mark.parametrize("omitted", [(2,), (4,), (5,)])
def test_missing_finished_worker_cannot_turn_into_a_low_memory_pass(tmp_path, omitted):
    process = tree(tmp_path, omitted=omitted)
    with pytest.raises(ValueError, match="absent"):
        workload_memory(
            resident_pids=[2, 3], preparation_pids=[4, 5], process=process, proc_root=tmp_path
        )


@pytest.mark.parametrize("extra", [(), (6, 7)])
def test_wrong_descendant_count_is_rejected(tmp_path, extra):
    process = tree(tmp_path, extra=extra)
    with pytest.raises(ValueError, match="tracker"):
        workload_memory(
            resident_pids=[2, 3], preparation_pids=[4, 5], process=process, proc_root=tmp_path
        )


def test_unknown_process_is_not_silently_omitted(tmp_path):
    process = tree(tmp_path)
    children = process.children(True)
    children[-1].cmdline = lambda: ["python", "unclassified"]
    with pytest.raises(ValueError, match="Unclassified"):
        workload_memory(
            resident_pids=[2, 3], preparation_pids=[4, 5], process=process, proc_root=tmp_path
        )


@pytest.mark.parametrize("content", ["Pid: 1\n", "Pid: 2\nVmHWM: 10 kB\n", "Pid: 1\nVmHWM: 0 kB\n"])
def test_missing_or_wrong_kernel_peak_does_not_admit_a_memory_envelope(tmp_path, content):
    (tmp_path / "1").mkdir()
    (tmp_path / "1/status").write_text(content)
    with pytest.raises(ValueError):
        linux_peak_rss(1, proc_root=tmp_path)


def test_worker_roles_cannot_double_count_or_lack_a_process(tmp_path):
    with pytest.raises(ValueError, match="distinct"):
        workload_memory(resident_pids=[2, 3], preparation_pids=[3, 4], proc_root=Path(tmp_path))


def test_collect_descendants_from_all_parent_threads_and_nested_workers(tmp_path):
    for pid, children in [(1, "2 3"), (2, "4"), (3, ""), (4, "")]:
        task = tmp_path / str(pid) / "task" / str(pid)
        task.mkdir(parents=True)
        (task / "children").write_text(children)
        (tmp_path / str(pid) / "cmdline").write_bytes(b"python\0worker\0")
    second_thread = tmp_path / "1/task/100"
    second_thread.mkdir()
    (second_thread / "children").write_text("3")
    actual = LinuxProcess(1, tmp_path).children(recursive=True)
    assert [child.pid for child in actual] == [2, 3, 4]
    assert actual[0].cmdline() == ["python", "worker", ""]
