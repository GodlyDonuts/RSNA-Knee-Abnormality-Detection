"""New scheduling/owned-process/deadline glue only; no GPU or clinical inference."""

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("new_path_two_t4_runner", HERE / "run_two_t4.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_estimated_work_balances_long_studies_without_dropping_any():
    costs = [116, 17, 116, 17]
    lanes = m.balanced_lanes(costs)
    assert lanes == [[0, 1], [2, 3]]
    assert [sum(costs[i] for i in lane) for lane in lanes] == [133, 133]
    assert m.balanced_lanes([1, 1, 1]) == [[0, 2], [1]]
    assert m.balanced_lanes([1]) == [[0]]


@pytest.mark.parametrize("costs", [[], [0], [-1], [float("nan")], [True]])
def test_invalid_schedule_cannot_silently_omit_studies(costs):
    with pytest.raises(ValueError):
        m.balanced_lanes(costs)


class FakeProcess:
    def __init__(self, alive, code):
        self.pid = 100
        self.alive = alive
        self.exitcode = code
        self.terminated = 0
        self.killed = 0

    def is_alive(self):
        return self.alive

    def join(self, timeout):
        pass

    def terminate(self):
        self.terminated += 1
        self.alive = False
        self.exitcode = -15

    def kill(self):
        self.killed += 1
        self.alive = False
        self.exitcode = -9


def test_failed_lane_stops_only_other_owned_lane():
    failed, other = FakeProcess(False, 1), FakeProcess(True, None)
    with pytest.raises(RuntimeError):
        m.supervise([failed, other], m.time.monotonic() + 60)
    assert failed.terminated == 0 and other.terminated == 1


def test_shared_deadline_stops_owned_children(monkeypatch):
    children = [FakeProcess(True, None), FakeProcess(True, None)]
    monkeypatch.setattr(m.time, "monotonic", lambda: 100)
    with pytest.raises(TimeoutError):
        m.supervise(children, 99)
    assert all(c.terminated == 1 for c in children)


def test_finished_lanes_are_not_restarted_or_terminated():
    children = [FakeProcess(False, 0), FakeProcess(False, 0)]
    m.supervise(children, m.time.monotonic() + 60)
    assert all(c.terminated == 0 for c in children)


def test_explicit_lane_preserves_requested_original_indices():
    class Provider:
        binding = {"only": "original"}

        def __len__(self):
            return 4

        def __getitem__(self, i):
            return "study-" + str(i)

    p = Provider()
    view = m.IndexedProvider(p, [2, 0])
    assert len(view) == 2 and [view[i] for i in range(2)] == ["study-2", "study-0"]
    with pytest.raises(ValueError):
        m.IndexedProvider(p, [0, 0])
    with pytest.raises(ValueError):
        m.IndexedProvider(p, [4])


def test_planner_counts_paths_but_never_decodes_pixels(tmp_path, monkeypatch):
    import sys

    sys.path[:0] = [str(HERE), str(HERE.parent / "connected-witness-paths-20261017-v1")]
    ids = ["1.1", "1.2"]
    rows = []
    pd.DataFrame({"StudyInstanceUID": ids, "ACL": ["not-read", "not-read"]}).to_csv(
        tmp_path / "test.csv", index=False
    )
    for uid, n in zip(ids, (5, 2), strict=True):
        sid = uid + ".1"
        folder = tmp_path / "test_series" / uid / sid
        folder.mkdir(parents=True)
        for i in range(n):
            (folder / f"{sid}.{i + 1}.dcm").write_text("not-decodable; path count only")
        rows.append(
            {"StudyInstanceUID": uid, "SeriesInstanceUID": sid, "clinical_report": "not-read"}
        )
    pd.DataFrame(rows).to_csv(tmp_path / "test_series.csv", index=False)

    def no_pixel_reads(*a, **k):
        raise AssertionError("pixel bytes must not be read by placement planner")

    monkeypatch.setattr(Path, "read_bytes", no_pixel_reads)
    actual, costs = m.read_test_plan(tmp_path)
    assert actual == ids and costs == [21, 18]


@pytest.mark.parametrize("seconds", ["32401", "nan"])
def test_launcher_cannot_request_more_than_its_shared_budget(tmp_path, seconds):
    argv = []
    for name in ("competition", "work", "output", "parent", "head"):
        argv.extend(["--" + name, str(tmp_path / name)])
    argv.extend(["--parent-sha256", "original", "--head-sha256", "head", "--max-seconds", seconds])
    with pytest.raises(ValueError):
        m.main(argv)


def test_owned_work_cannot_be_created_inside_competition_assets(tmp_path):
    root = tmp_path / "competition"
    root.mkdir()
    argv = [
        "--competition",
        str(root),
        "--work",
        str(root / "cache"),
        "--output",
        str(tmp_path / "submission.csv"),
        "--parent",
        str(tmp_path / "parent.pt"),
        "--head",
        str(tmp_path / "head.pt"),
        "--parent-sha256",
        "original",
        "--head-sha256",
        "head",
    ]
    with pytest.raises(ValueError):
        m.main(argv)
    assert list(root.iterdir()) == []
