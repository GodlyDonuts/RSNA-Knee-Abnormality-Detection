"""New paired deployment budgets and one-model request boundary."""

import ast
import json
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
D = ROOT / "initiative/experiments/competitive-v1/paired-final-scoring-20261018-v1"


@pytest.fixture
def controller():
    return runpy.run_path(str(D / "program.py.txt"), run_name="paired_scoring_test")


@pytest.mark.parametrize("mode,expected", [("qualification", 37800), ("competition", 36000)])
def test_budget_funds_complete_remaining_current_path(controller, mode, expected):
    assert controller["vm_requirement"](mode, 0, 0, 0) == expected
    assert controller["vm_requirement"](mode, 6000, 1800, 0) == expected + 7800


@pytest.mark.parametrize(
    "args",
    [
        ("qualification", 0, 0, 6000),
        ("qualification", -1, 0, 0),
        ("competition", False, 0, 0),
        ("other", 0, 0, 0),
    ],
)
def test_unfinished_fit_and_invalid_budgets_refused(controller, args):
    with pytest.raises(AssertionError):
        controller["vm_requirement"](*args)


def test_unstarted_other_scoring_is_not_fictitious_runtime(controller, tmp_path):
    assert controller["actual_started_other_scoring_future"](tmp_path) == 0


def test_uncertain_other_push_blocks_new_mutation(controller, tmp_path):
    (tmp_path / "competition-push-intent.json").write_text("{}")
    with pytest.raises(ValueError, match="uncertain"):
        controller["actual_started_other_scoring_future"](tmp_path)


def test_started_other_scoring_is_funded_until_actual_terminal(controller, tmp_path):
    for name in ["qualification", "competition"]:
        (tmp_path / (name + "-push-intent.json")).write_text("{}")
        (tmp_path / (name + "-push-response.json")).write_text(
            json.dumps({"kernelId": 123, "versionNumber": 1})
        )
    assert controller["actual_started_other_scoring_future"](tmp_path) == 34200
    (tmp_path / "qualification-actual-accepted.json").write_text(
        json.dumps({"kernel_id": 123, "version": 1, "status": "COMPLETE"})
    )
    assert controller["actual_started_other_scoring_future"](tmp_path) == 32400
    (tmp_path / "competition-actual-failed.json").write_text(
        json.dumps({"kernel_id": 123, "version": 1, "status": "ERROR"})
    )
    assert controller["actual_started_other_scoring_future"](tmp_path) == 0


@pytest.mark.parametrize("mode,timeout", [("qualification", 1800), ("competition", 32400)])
def test_exact_offline_final_model_request(controller, mode, timeout):
    r = controller["request_for"](
        "paired", mode, {"training": {"slug": "godlydonuts/final-paired-only"}}, "model-source"
    )
    assert (
        r["sessionTimeoutSeconds"] == timeout
        and r["enableInternet"] is False
        and r["isPrivate"] is True
    )
    assert r["enableGpu"] is True and r["machineShape"] == "NvidiaTeslaT4"
    assert r["kernelDataSources"][0] == "godlydonuts/final-paired-only"
    assert (r["competitionDataSources"] == []) is (mode == "qualification")
    assert len(r["kernelDataSources"]) == (2 if mode == "qualification" else 1)
    assert r["datasetDataSources"] == [] and "paired-e3-" in r["slug"]


def test_unrelated_candidate_cannot_use_this_coordinator(controller):
    with pytest.raises(AssertionError):
        controller["request_for"]("encoderlr", "competition", {}, "source")


def test_owner_gates_real_six_object_final_and_original_process():
    t = ast.parse((D / "finish_once.py.txt").read_text())
    strings = [
        n.value for n in ast.walk(t) if isinstance(n, ast.Constant) and isinstance(n.value, str)
    ]
    assert "matched_paired_full_fit_complete_all3_private_verified" in strings
    assert "81735" in strings and "paired-full-fitting-20261018-v1/finish_v2_once.py.txt" in strings
    source = (D / "finish_once.py.txt").read_text()
    assert "pub['objects']==6" in source and "actual['receipt']['completed']==1458" in source
    assert (
        "maximum_code_entries':1" in source
        and "pub['original_kernel']==actual['kernel_id']" in source
    )


def test_mismatched_terminal_cannot_retire_runtime_cap(controller, tmp_path):
    (tmp_path / "competition-push-intent.json").write_text("{}")
    (tmp_path / "competition-push-response.json").write_text(
        json.dumps({"kernelId": 123, "versionNumber": 1})
    )
    (tmp_path / "competition-actual-accepted.json").write_text(
        json.dumps({"kernel_id": 999, "version": 1, "status": "COMPLETE"})
    )
    with pytest.raises(AssertionError):
        controller["actual_started_other_scoring_future"](tmp_path)
