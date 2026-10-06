"""Offline integration preserves budgets and frozen numerical-source compatibility."""

import importlib.util
import os
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.kaggle import resident_entry, resident_infer


def notebook_module():
    path = Path(resident_entry.__file__).with_name("notebook_entry.py")
    spec = importlib.util.spec_from_file_location("resident_notebook", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_process_budget_kills_real_descendant(tmp_path):
    marker = tmp_path / "late-write"
    child = (
        'import time; from pathlib import Path; time.sleep(2); Path(%r).write_text("escaped")'
        % str(marker)
    )
    parent = (
        'import subprocess,sys,time; subprocess.Popen([sys.executable,"-c",%r]); time.sleep(10)'
        % child
    )
    with pytest.raises(subprocess.TimeoutExpired):
        notebook_module().bounded_process(
            [sys.executable, "-c", parent], env=os.environ.copy(), timeout=0.15
        )
    time.sleep(2.1)
    assert not marker.exists()


def test_bootstrap_exhausted_does_not_spawn(monkeypatch):
    module = notebook_module()
    monkeypatch.setattr(module.subprocess, "Popen", lambda *a, **k: pytest.fail("spawned"))
    with pytest.raises(ValueError, match="budget"):
        module.bounded_process([], env={}, timeout=0)


def test_entry_routes_original_chunk_runner_and_closes_pool(tmp_path, monkeypatch):
    calls = []

    class Pool:
        def __init__(self, **options):
            calls.append(options)

        def invoke(self, *a, **k):
            pass

        def close(self):
            calls.append("closed")

    def run(args, *, invoke):
        assert args.single_decode is True
        assert invoke.__self__.__class__ is Pool
        return {"status": "passed"}

    monkeypatch.setattr(resident_entry.chunked_infer, "run", run)
    args = Namespace(
        precision="fp32",
        workers=0,
        device="cuda",
        resilient_series=True,
        recover_slices=True,
        isolate_study_errors=True,
        baseline_epoch_one=False,
        allow_diagnostic=False,
        output=tmp_path / "submission.csv",
    )
    result = resident_entry.run(args, remaining_seconds=120, pool_factory=Pool)
    assert calls == [{"workers": 2, "timeout_seconds": 120}, "closed"]
    assert result["hidden_test_runtime_validated"] is False
    args.precision = "bf16"
    with pytest.raises(ValueError, match="FP32"):
        resident_entry.run(args, pool_factory=Pool)


def test_historical_sampling_has_no_silent_policy_change(monkeypatch):
    monkeypatch.delattr(resident_infer.trained_data, "evaluation_slice_sampling")
    assert resident_infer.evaluation_slice_sampling("uniform") == "uniform"
    assert resident_infer.evaluation_slice_sampling("all") == "all"
    with pytest.raises(ValueError, match="Historical"):
        resident_infer.evaluation_slice_sampling("stratified")
