from __future__ import annotations

import argparse
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from rsna.rank_fusion import fuse_members, load_member, run
from rsna.schema import TARGETS

COLUMNS = ["StudyInstanceUID", *TARGETS]


def _frame(uids: list[str], values: np.ndarray) -> pd.DataFrame:
    frame = pd.DataFrame(values, columns=list(TARGETS))
    frame.insert(0, "StudyInstanceUID", uids)
    return frame


def test_fuse_members_averages_per_column_percentile_ranks() -> None:
    uids = ["a", "b", "c"]
    template = _frame(uids, np.zeros((3, len(TARGETS))))
    m1 = _frame(uids, np.tile(np.array([0.9, 0.5, 0.1])[:, None], (1, len(TARGETS))))
    m2 = _frame(["b", "c", "a"], np.tile(np.array([0.2, 0.4, 0.6])[:, None], (1, len(TARGETS))))
    fused = fuse_members([m1, m2], template)
    assert list(fused.columns) == COLUMNS
    for target in TARGETS:
        assert fused[target].tolist() == pytest.approx([1.0, 0.5, 0.5])


def test_fuse_members_preserves_single_member_ordering() -> None:
    uids = ["a", "b", "c", "d"]
    template = _frame(uids, np.zeros((4, len(TARGETS))))
    values = np.tile(np.array([0.9, 0.7, 0.4, 0.1])[:, None], (1, len(TARGETS)))
    fused = fuse_members([_frame(uids, values)], template)
    assert fused["ACL"].is_monotonic_decreasing


def test_fuse_members_handles_ties_with_average_ranks() -> None:
    uids = ["a", "b"]
    template = _frame(uids, np.zeros((2, len(TARGETS))))
    member = _frame(uids, np.full((2, len(TARGETS)), 0.5))
    fused = fuse_members([member], template)
    assert (fused[list(TARGETS)] == 0.75).all().all()


def test_fuse_members_requires_at_least_one_member() -> None:
    template = _frame(["a"], np.zeros((1, len(TARGETS))))
    with pytest.raises(ValueError, match="at least one member"):
        fuse_members([], template)


def test_load_member_validates(tmp_path) -> None:
    uids = ["a", "b"]
    template = _frame(uids, np.zeros((2, len(TARGETS))))
    path = tmp_path / "member.csv"
    _frame(uids, np.full((2, len(TARGETS)), 0.5)).to_csv(path, index=False)
    loaded = load_member(path, template)
    assert list(loaded.StudyInstanceUID) == uids

    bad = _frame(["a", "c"], np.full((2, len(TARGETS)), 0.5))
    bad.to_csv(path, index=False)
    with pytest.raises(ValueError, match="differ from the template"):
        load_member(path, template)

    dup = _frame(["a", "a"], np.full((2, len(TARGETS)), 0.5))
    dup.to_csv(path, index=False)
    with pytest.raises(ValueError, match="Duplicate"):
        load_member(path, template)

    out_of_range = _frame(uids, np.full((2, len(TARGETS)), 1.5))
    out_of_range.to_csv(path, index=False)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        load_member(path, template)


def test_run_writes_template_ordered_submission(tmp_path) -> None:
    uids = ["a", "b", "c"]
    template_path = tmp_path / "sample_submission.csv"
    _frame(uids, np.zeros((3, len(TARGETS)))).to_csv(template_path, index=False)
    member_path = tmp_path / "member.csv"
    _frame(uids[::-1], np.tile(np.linspace(0.2, 0.8, 3)[:, None], (1, len(TARGETS)))).to_csv(
        member_path, index=False
    )
    output = tmp_path / "fused.csv"
    run(argparse.Namespace(members=[member_path], template=template_path, output=output))
    fused = pd.read_csv(output)
    assert list(fused.columns) == COLUMNS
    assert list(fused.StudyInstanceUID) == uids
    assert fused["ACL"].tolist() == pytest.approx([1.0, 2 / 3, 1 / 3])


def test_cli(tmp_path) -> None:
    uids = ["a"]
    template_path = tmp_path / "t.csv"
    _frame(uids, np.zeros((1, len(TARGETS)))).to_csv(template_path, index=False)
    member_path = tmp_path / "m.csv"
    _frame(uids, np.full((1, len(TARGETS)), 0.5)).to_csv(member_path, index=False)
    output = tmp_path / "o.csv"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "rsna.rank_fusion",
            "--members",
            str(member_path),
            "--template",
            str(template_path),
            "--output",
            str(output),
        ],
        check=True,
    )
    assert pd.read_csv(output).shape == (1, len(COLUMNS))
