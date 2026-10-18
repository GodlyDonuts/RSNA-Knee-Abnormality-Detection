"""New LR candidate's budget/dependency contracts; no remote or model replay."""

import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
loader = importlib.machinery.SourceFileLoader("new_final_LR_scoring", str(HERE / "program.py.txt"))
spec = importlib.util.spec_from_loader(loader.name, loader)
m = importlib.util.module_from_spec(spec)
loader.exec_module(m)


def test_LR_qualification_budget_preserves_other_candidates_and_future_public_run():
    assert m.vm_requirement("qualification", 34200, 32400, 0) == 104400
    assert m.vm_requirement("competition", 32400, 0, 0) == 68400
    assert m.vm_requirement("qualification", 0, 0, 0) == 37800


@pytest.mark.parametrize("unfinished", [6000, 12000, 18000])
def test_unfinished_LR_fit_cannot_admit_inference(unfinished):
    with pytest.raises(AssertionError):
        m.vm_requirement("qualification", 0, 0, unfinished)


@pytest.mark.parametrize("bad", [-1, True, 1800.0])
def test_invalid_liabilities_do_not_create_fictitious_capacity(bad):
    with pytest.raises(AssertionError):
        m.vm_requirement("qualification", bad, 0, 0)


def test_other_candidate_future_retires_only_actual_terminal_receipts(tmp_path):
    assert m.remaining_serving_seconds(tmp_path) == 34200
    (tmp_path / "qualification-actual-accepted.json").write_text("{}")
    assert m.remaining_serving_seconds(tmp_path) == 32400
    (tmp_path / "competition-actual-accepted.json").write_text("{}")
    assert m.remaining_serving_seconds(tmp_path) == 0


def test_failed_other_candidate_is_not_automatically_retried(tmp_path):
    (tmp_path / "qualification-actual-failed.json").write_text("{}")
    assert m.remaining_serving_seconds(tmp_path) == 0


def test_exact_new_LR_source_uses_completed_training_output_offline():
    base = {"training": {"slug": "godlydonuts/actual-new-LR-final-source"}}
    qual = m.request_for("encoderlr", "qualification", base, "actual frozen source")
    public = m.request_for("encoderlr", "competition", base, "actual frozen source")
    assert qual["slug"] != public["slug"]
    assert qual["competitionDataSources"] == []
    assert public["competitionDataSources"] == ["rsna-knee-abnormality-detection"]
    assert public["kernelDataSources"] == [base["training"]["slug"]]
    for request in (qual, public):
        assert request["enableInternet"] is False and request["isPrivate"] is True
        assert request["enableGpu"] is True and request["datasetDataSources"] == []
    assert qual["sessionTimeoutSeconds"] == 1800
    assert public["sessionTimeoutSeconds"] == 32400


def test_finding_candidate_cannot_leak_into_LR_submission_request():
    with pytest.raises(AssertionError):
        m.request_for("finding", "competition", {"training": {"slug": "unused"}}, "unused")
