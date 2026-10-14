import copy
import json
from pathlib import Path

import pytest

from scripts.newton import admit_patch_focus_relocated_inputs as gate

ROOT = Path(__file__).resolve().parents[1]
CPU = json.loads(
    (
        ROOT / "initiative/audits/patch-focus-target-CPU-accepted-20261014-v1/CPU-accepted.json"
    ).read_text()
)
ENV = {"SLURM_JOB_ID": "919999", "SLURM_CPUS_PER_TASK": "1"}


def test_new_node_is_explicit_and_old_source_jobs_cannot_admit():
    assert gate.check_placement("evc26", ENV) == "919999"
    for node, change in [
        ("evc40", {}),
        ("evc26", {"SLURM_JOB_ID": "916922"}),
        ("evc26", {"SLURM_CPUS_PER_TASK": "2"}),
        ("evc26", {"SLURM_JOB_GPUS": "0"}),
        ("evc26", {"CUDA_VISIBLE_DEVICES": "0"}),
    ]:
        with pytest.raises(ValueError):
            gate.check_placement(node, ENV | change)


def test_exact_runtime_versions_and_declared_new_prefix():
    runtime = CPU["runtime"] | {"prefix": gate.RUNTIME}
    gate.check_runtime(runtime, CPU["runtime"])
    for change in [{"torch": "other"}, {"prefix": CPU["runtime"]["prefix"]}, {"python": "3.12.0"}]:
        with pytest.raises(ValueError):
            gate.check_runtime(runtime | change, CPU["runtime"])


def test_old_source_proof_is_not_relabelled_or_modified(monkeypatch):
    manifest = {"source_git": gate.SOURCE_GIT}
    monkeypatch.setattr(gate.admit, "verify_source", lambda code, value: CPU["code_sha256"])
    before = copy.deepcopy(CPU)
    assert gate.check_source(ROOT, manifest, CPU) == CPU["code_sha256"]
    assert CPU == before
    for change in [
        {"node": "evc26"},
        {"CPU_job": "919999"},
        {"source_git": "0" * 40},
        {"status": "other"},
        {"code_sha256": {}},
    ]:
        with pytest.raises(ValueError):
            gate.check_source(ROOT, manifest, CPU | change)
