"""Native production must stay undispatched until actual CPU and quota gates pass."""

import copy
import json
import subprocess
from pathlib import Path

import pytest

from scripts.newton import dispatch_native384_production as coordinator

ROOT = Path(__file__).resolve().parents[1]


def state(mode):
    return {
        "status": "ready_for_cpu_preflight" if mode == "cpu" else "ready_for_live_cpu_recheck",
        "assembly_terminal": "866458|COMPLETED|0:0",
        "projected_remaining_gib": 20.1,
        "projected_remaining_files": 5001,
        "actual_gpu_jobs": [],
        "existing_run": False,
        "existing_cpu_intent": False,
        "existing_gpu_intent": False,
    }


@pytest.mark.parametrize("mode", ["cpu", "gpu"])
@pytest.mark.parametrize(
    "fault",
    [
        None,
        "pending_assembly",
        "failed_assembly",
        "quota_bytes",
        "quota_files",
        "intent",
        "run",
        "gpu_limit",
    ],
)
def test_missing_prerequisites_and_existing_attempts_cannot_dispatch(mode, fault):
    record = copy.deepcopy(state(mode))
    if fault == "pending_assembly":
        record["assembly_terminal"] = "866458|PENDING|0:0"
    elif fault == "failed_assembly":
        record["assembly_terminal"] = "866458|FAILED|2:0"
    elif fault == "quota_bytes":
        record["projected_remaining_gib"] = 19.999
    elif fault == "quota_files":
        record["projected_remaining_files"] = 4999
    elif fault == "intent":
        record["existing_" + mode + "_intent"] = True
    elif fault == "run":
        record["existing_run"] = True
    elif fault == "gpu_limit":
        record["actual_gpu_jobs"] = ["one_gpu_node", "another_gpu_node"]
    if fault is None or (mode == "cpu" and fault == "gpu_limit"):
        coordinator.require_phase(record, mode)
    else:
        with pytest.raises((ValueError, FileExistsError)):
            coordinator.require_phase(record, mode)


def test_launch_scripts_parse_and_use_retained_native_cache_with_strict_training_wrapper():
    folder = ROOT / "initiative/audits/native384-production-preparation-v1"
    for name in ("production.sbatch.txt", "cpu-preflight.sbatch.txt"):
        subprocess.run(["bash", "-n", str(folder / name)], check=True)
    production = (folder / "production.sbatch.txt").read_text()
    assert "cache256" not in production
    assert "runtime_archive.py extract" not in production
    assert "--nodelist=evc31" in production
    assert "--gres=gpu:nvidia_h100_pcie:1" in production
    assert "--cpus-per-task=2" in production
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in production
    assert "enforce_deterministic_runtime.py --module rsna.competitive.train" in production
    assert "native384_admission.py gpu" in production
    assert "native-input-provenance.json" in production
    assert "scripts/newton/supervise.py --grace-seconds 180" in production
    cpu = (folder / "cpu-preflight.sbatch.txt").read_text()
    assert "--gres" not in cpu
    assert "--cpus-per-task=1" in cpu
    assert "native384_admission.py cpu" in cpu


def test_exact_native_production_stage_manifest_binds_all_referenced_sources():
    manifest = json.loads((coordinator.AUDIT / "production-files.json").read_text())
    assert manifest["scientific_source_git"] == "c9fcc7eefb09103942be89bf4fa87410eb9ec5ab"
    for name, row in manifest["files"].items():
        path = ROOT / row["local_source"]
        assert path.stat().st_size == row["bytes"] and coordinator.sha(path) == row["sha256"]
    assert {
        "native384_admission.py",
        "publish_native384_epoch.py",
        "config.json",
        "enforce_deterministic_runtime.py",
        "cpu-preflight.sbatch",
        "production.sbatch",
    } <= set(manifest["files"])
