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


def test_dataset_discovery_never_enters_MRI_tree(tmp_path, monkeypatch):
    module = notebook_module()
    modern = tmp_path / "datasets/owner/model/1"
    modern.mkdir(parents=True)
    (modern / "bundle-manifest.json").write_text("exact")
    legacy = tmp_path / "legacy-wheels"
    legacy.mkdir()
    (legacy / "wheelhouse.bin").write_bytes(b"bound")
    for location in [
        tmp_path / "competitions/competition/train_series/study",
        tmp_path / "rsna-knee-abnormality-detection/test_series/study",
        tmp_path / "unexpected-name/train_series/study",
    ]:
        location.mkdir(parents=True)
        (location / "bundle-manifest.json").write_text("wrong")
    original = module.os.scandir

    def scan(path):
        path = Path(path)
        assert not any(
            p in {"competitions", "rsna-knee-abnormality-detection", "train_series", "test_series"}
            for p in path.relative_to(tmp_path).parts
        )
        return original(path)

    monkeypatch.setattr(module.os, "scandir", scan)
    assert list(module.attached_dataset_files(tmp_path, "bundle-manifest.json")) == [
        modern / "bundle-manifest.json"
    ]
    assert list(module.attached_dataset_files(tmp_path, "wheelhouse.bin")) == [
        legacy / "wheelhouse.bin"
    ]


def test_attached_payload_discovery_rejects_escaping_name(tmp_path):
    with pytest.raises(ValueError, match="basename"):
        list(notebook_module().attached_dataset_files(tmp_path, "../secret"))


def test_payload_discovery_does_not_walk_attached_notebook_source_cache(tmp_path, monkeypatch):
    module = notebook_module()
    output = tmp_path / "notebooks/owner/CPU-input/v1"
    output.mkdir(parents=True)
    receipt = output / "witness-inputs-CPU-accepted.json"
    receipt.write_text("bound")
    (output / "witness-training-source/artifacts/deep/a/b/c/d").mkdir(parents=True)
    (output / "witness-inputs-admission/cache/train_series/private").mkdir(parents=True)
    scan = module.os.scandir

    def guarded(path):
        assert not any(
            p in {"witness-training-source", "witness-inputs-admission"} for p in Path(path).parts
        )
        return scan(path)

    monkeypatch.setattr(module.os, "scandir", guarded)
    assert list(module.attached_dataset_files(tmp_path, receipt.name)) == [receipt]


def test_actual_FP16_overlay_payload_passes_private_training_admission():
    import ast
    import hashlib

    root = Path(__file__).resolve().parents[1]
    worker = (root / "scripts/kaggle/witness_amp_recovery.py").read_text()
    overrides = {
        n: (root / n).read_text()
        for n in ("rsna/competitive/train.py", "rsna/competitive/witness.py")
    }
    declaration = (
        "CODE_OVERRIDES = "
        + repr(overrides)
        + "\nCODE_OVERRIDE_HASHES = "
        + repr({n: hashlib.sha256(s.encode()).hexdigest() for n, s in overrides.items()})
        + "\nSCIENCE_GIT = "
        + repr("a" * 40)
        + "\n"
    )
    worker = worker.replace(
        "from __future__ import annotations\n",
        "from __future__ import annotations\n" + declaration,
        1,
    )
    sources = {
        "witness_recovery.py": worker,
        "witness_launch.py": (root / "scripts/kaggle/witness_amp_launch.py").read_text(),
    }
    hashes = {n: hashlib.sha256(s.encode()).hexdigest() for n, s in sources.items()}
    # Reproduce the actual failed envelope, not an unrelated small fake script.
    assert 64 * 1024 < sum(len(s.encode()) for s in sources.values()) < 128 * 1024
    assert ast.parse(worker)
    module = notebook_module()
    module.validate_private_training_sources(sources, hashes, "0" * 64)
    changed = dict(sources, **{"witness_recovery.py": worker + "\n# tampered"})
    with pytest.raises(ValueError, match="source differs"):
        module.validate_private_training_sources(changed, hashes, "0" * 64)
    oversized = dict(sources, **{"witness_recovery.py": worker + "\n#" + "x" * (128 * 1024)})
    with pytest.raises(ValueError, match="Exact private trainer"):
        module.validate_private_training_sources(oversized, hashes, "0" * 64)
