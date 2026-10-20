"""Retained-state boundaries for resuming only missing actual GPU test work."""

import importlib.machinery
import importlib.util
import json
from pathlib import Path

import pytest
import torch

D = Path("initiative/research/teacher-pair-earlier45-20261020-v1")
loader = importlib.machinery.SourceFileLoader(
    "pair_continuation", str(D / "continuation_driver.py.txt")
)
m = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
loader.exec_module(m)


@pytest.fixture
def retained(tmp_path):
    full = tmp_path / "pointwise/full"
    partial = tmp_path / "pointwise/resumed"
    full.mkdir(parents=True)
    partial.mkdir()
    initial = "a75cb18d1dec13c4fb2f547f017ed92e81bf768f580cee54da0ffa88b712fc78"
    (full / "full-process.json").write_text(
        json.dumps(
            {
                "completed_updates": 4,
                "prediction_studies": 16,
                "prediction_sha256": "ba21708a0ddd4d6e723d3a8262e48bf620bf7244bca24e9bf11976376a268859",
                "initial_state_sha256": initial,
                "unchanged_BN": True,
            }
        )
    )
    (partial / "stop1-process.json").write_text(
        json.dumps({"completed_updates": 1, "initial_state_sha256": initial, "unchanged_BN": True})
    )
    torch.save(
        {
            "completed": 1,
            "epoch": 0,
            "cursor": 1,
            "optimizer_boundary": True,
            "history": [{"update": 1}],
            "binding": {"recipe": {"objective": "pointwise"}},
        },
        partial / "checkpoint.pt",
    )
    return tmp_path


def test_retains_completed_full_and_stop1_without_repeating_them(retained):
    result = m.validate_retained(retained, retained)
    assert (
        result["uninterrupted_full_reference_updates"] == 4
        and result["remaining_fresh_processes"] == 4
    )
    assert m.PHASES == (
        ("pointwise", "resume"),
        ("teacher_rank", "full"),
        ("teacher_rank", "stop1"),
        ("teacher_rank", "resume"),
    )


@pytest.mark.parametrize(
    "bad",
    [
        "advanced_checkpoint",
        "different_initial",
        "rank_already_began",
        "resume_complete",
        "wrong_reference",
    ],
)
def test_unknown_or_advanced_retained_state_cannot_be_restarted(retained, bad):
    partial = retained / "pointwise/resumed"
    if bad == "advanced_checkpoint":
        z = torch.load(partial / "checkpoint.pt", weights_only=True)
        z["completed"] = 2
        torch.save(z, partial / "checkpoint.pt")
    elif bad == "different_initial":
        z = json.loads((partial / "stop1-process.json").read_text())
        z["initial_state_sha256"] = "0" * 64
        (partial / "stop1-process.json").write_text(json.dumps(z))
    elif bad == "rank_already_began":
        (retained / "teacher_rank").mkdir()
    elif bad == "resume_complete":
        (partial / "resume-process.json").write_text("{}")
    else:
        path = retained / "pointwise/full/full-process.json"
        z = json.loads(path.read_text())
        z["completed_updates"] = 3
        path.write_text(json.dumps(z))
    with pytest.raises(AssertionError):
        m.validate_retained(retained, retained)


def test_normal_queue_single_dispatch_and_original_decoder_preserved():
    source = (D / "stage_continuation.py.txt").read_text()
    assert "--partition=normal" in source and "--qos=normal" in source and "--no-requeue" in source
    assert "--time=00:20:00" in source and "--time-min=00:10:00" in source
    assert "974708|PREEMPTED|0:0" in source and "Reason=JobHeldUser" in source
    driver = (D / "continuation_driver.py.txt").read_text()
    assert 'ns["compare"](work, name)' in driver and "shutil.rmtree" not in driver
    assert "same" in driver and "original_reference_repeated" in driver
    for name in (
        "stage_continuation.py.txt",
        "continuation_driver.py.txt",
        "continue_normal_once.py.txt",
    ):
        compile((D / name).read_text(), name, "exec")
