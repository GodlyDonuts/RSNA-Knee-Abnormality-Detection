"""Simulate the full post-fit chain, including API uncertainty and hidden failure."""

import ast
import contextlib
import io
import json
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

HERE = Path(__file__).parent


@pytest.mark.parametrize(
    "damage",
    [
        None,
        "training_failure",
        "qualification_failure",
        "public_failure",
        "uncertain_entry",
        "zero_slots_once",
        "unknown_status_once",
        "hidden_failure",
    ],
)
def test_one_original_postfit_pipeline(damage):
    calls = []
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        D = root / "scoring"
        D.mkdir()
        for n in ("program.py.txt", "coordinate_once.py.txt"):
            (D / n).write_text("# simulated immutable control\n")
        (D / "actual-staged.json").write_text("{}")
        owner = D / "finish_once.py.txt"
        owner.write_text((HERE / "finish_once.py.txt").read_text())
        F = root / "artifacts/multisource-influence-20261018-v1"
        F.mkdir(parents=True)
        (F / "continuation-owner.result.json").write_text(
            json.dumps(
                {
                    "status": "failure"
                    if damage == "training_failure"
                    else "one_original_family_influence_fixed_final3_fit_and_all_three_private_epochs_verified"
                }
            )
        )
        data = root / "multisource-influence-v1"
        data.mkdir()
        (data / "epoch-3-actual-accepted.json").write_text(
            json.dumps({"kernel_id": 139355023, "receipt": {"completed": 1458}})
        )
        (data / "epoch-3-private-verified.json").write_text(
            json.dumps({"original_Kaggle_kernel": 139355023})
        )
        seen = {}

        def run(args, **kwargs):
            assert Path(args[1]).name == "coordinate_once.py.txt"
            phase, mode = args[2:4]
            calls.append((phase, mode))
            key = (phase, mode)
            seen[key] = seen.get(key, 0) + 1
            if phase == "prepare_family":
                out = {
                    "status": "actual_completed_privately_preserved_family_final_scoring_prepared"
                }
            elif phase == "push":
                out = {"status": "one_exact_matched_final_" + mode + "_pushed"}
            elif phase == "status":
                if damage == mode + "_failure" or (
                    damage == "public_failure" and mode == "competition"
                ):
                    out = {"status": "ERROR"}
                elif damage == "unknown_status_once" and seen[key] == 1:
                    out = {"status": None}
                else:
                    out = {"status": "COMPLETE", "receipt": {"synthetic": True}}
            elif phase == "eligibility":
                out = {
                    "status": "actual_family_final_entry_eligibility",
                    "fresh": {"eligible": not (damage == "zero_slots_once" and seen[key] == 1)},
                }
            elif phase == "submit":
                if damage == "uncertain_entry":
                    return subprocess.CompletedProcess(args, 1, "", "synthetic uncertainty")
                out = {"status": "one_exact_family_final_code_entry_accepted", "submission_id": 123}
            else:
                assert phase == "outcome"
                out = {
                    "status": "actual_family_final_hidden_outcome",
                    "outcome": "failed" if damage == "hidden_failure" else "scored",
                    "public_score": None if damage == "hidden_failure" else 0.949,
                }
            return subprocess.CompletedProcess(args, 0, json.dumps(out), "")

        old = Path.cwd()
        try:
            os.chdir(root)
            with (
                patch("subprocess.run", side_effect=run),
                patch("time.sleep"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                if damage in (
                    "training_failure",
                    "qualification_failure",
                    "public_failure",
                    "uncertain_entry",
                ):
                    with pytest.raises((AssertionError, RuntimeError)):
                        runpy.run_path(str(owner))
                else:
                    runpy.run_path(str(owner))
        finally:
            os.chdir(old)
        result = json.loads(
            (root / "artifacts/family-final-scoring-20261018-v1/owner-result.json").read_text()
        )
        assert (
            calls.count(("push", "qualification")) <= 1
            and calls.count(("push", "competition")) <= 1
            and calls.count(("submit", "competition")) <= 1
        )
        if damage == "training_failure":
            assert calls == []
        if damage == "qualification_failure":
            assert ("push", "competition") not in calls
        if damage == "public_failure":
            assert not any(p == "submit" for p, _ in calls)
        if damage == "uncertain_entry":
            assert calls[-1] == ("submit", "competition")
        if damage == "zero_slots_once":
            assert calls.count(("eligibility", "competition")) == 2
        if damage == "unknown_status_once":
            assert calls.count(("status", "qualification")) == 2
        if damage in (None, "hidden_failure", "zero_slots_once", "unknown_status_once"):
            assert result["status"] == "actual_original_family_final_hidden_terminal"
            assert result["actual"]["outcome"] == (
                "failed" if damage == "hidden_failure" else "scored"
            )
        if damage is None:
            assert calls == [
                ("prepare_family", "qualification"),
                ("push", "qualification"),
                ("status", "qualification"),
                ("push", "competition"),
                ("status", "competition"),
                ("eligibility", "competition"),
                ("submit", "competition"),
                ("outcome", "competition"),
            ]


def test_remote_exception_boundary_does_not_log_credentials_or_signed_URLs(monkeypatch, capsys):
    source = (HERE / "program.py.txt").read_text()
    node = next(
        n
        for n in ast.parse(source).body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "__name__ == '__main__'"
    )

    def fail(v):
        raise RuntimeError("https://example.invalid/?token=do-not-log")

    monkeypatch.setattr(
        sys, "stdin", io.StringIO(json.dumps({"phase": "stage", "token": "do-not-log"}))
    )
    with pytest.raises(SystemExit):
        exec(
            compile(ast.Module(body=[node], type_ignores=[]), "sanitized_remote_boundary", "exec"),
            {"__name__": "__main__", "json": json, "sys": sys, "execute": fail},
        )
    out = capsys.readouterr()
    assert "do-not-log" not in out.err and "example.invalid" not in out.err
    assert json.loads(out.err)["exception_type"] == "RuntimeError"
