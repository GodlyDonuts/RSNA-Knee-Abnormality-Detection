"""Reject advisory-only or absent determinism in the native recovery experiment."""

import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.newton.enforce_deterministic_runtime import configure


@pytest.mark.parametrize("workspace", [None, "", ":16:8", "incorrect"])
def test_reject_missing_or_unpinned_cuda_workspace(monkeypatch, workspace):
    if workspace is None:
        monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)
    else:
        monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", workspace)
    with pytest.raises(ValueError, match="before starting Python"):
        configure(None)


@pytest.mark.parametrize("enabled,warn_only", [(False, False), (True, True)])
def test_reject_ineffective_or_warning_only_policy(monkeypatch, enabled, warn_only):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    module = SimpleNamespace(
        use_deterministic_algorithms=lambda *a, **k: None,
        backends=SimpleNamespace(cudnn=SimpleNamespace()),
        are_deterministic_algorithms_enabled=lambda: enabled,
        is_deterministic_algorithms_warn_only_enabled=lambda: warn_only,
    )
    with pytest.raises(RuntimeError, match="not enabled"):
        configure(module)


def test_actual_torch_strict_policy_and_restore_process_state(monkeypatch):
    import torch

    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    enabled = torch.are_deterministic_algorithms_enabled()
    advisory = torch.is_deterministic_algorithms_warn_only_enabled()
    benchmark, deterministic = torch.backends.cudnn.benchmark, torch.backends.cudnn.deterministic
    try:
        record = configure(torch)
        assert record["deterministic_algorithms"] and record["warn_only"] is False
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        assert not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=advisory)
        torch.backends.cudnn.benchmark = benchmark
        torch.backends.cudnn.deterministic = deterministic


def test_external_wrapper_imports_checkout_and_forwards_target_help(tmp_path):
    target = tmp_path / "native_probe.py"
    target.write_text("import sys\nassert sys.argv[1:] == ['--help']\nprint('target_imported')\n")
    wrapper = (
        Path(__file__).resolve().parents[1] / "scripts/newton/enforce_deterministic_runtime.py"
    )
    result = subprocess.run(
        [sys.executable, "-I", str(wrapper), "--module", "native_probe", "--help"],
        cwd=tmp_path,
        env={**os.environ, "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "1"},
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "target_imported" in result.stdout
    assert '"deterministic_algorithms": true' in result.stdout
