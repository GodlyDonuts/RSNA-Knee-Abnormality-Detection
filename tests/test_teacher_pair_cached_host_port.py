"""A cached-host move must preserve the complete scientific worker and fit order."""

import ast
import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

D = Path("initiative/research/teacher-pair-cached48-20261020-v1")
loader = importlib.machinery.SourceFileLoader("cached_host_port", str(D / "port.py.txt"))
m = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
loader.exec_module(m)
ORIGINAL = Path("initiative/research/teacher-pair-full-fitting45-20261020-v1")


@pytest.mark.parametrize("name", ["worker.py", "driver.py"])
def test_roundtrip_preserves_entire_numerical_worker_and_objective_sequence(name):
    source = (ORIGINAL / (name + ".txt")).read_text()
    result = m.build(name, source)
    assert m.ROOT48 in result and m.ROOT45 not in result
    inverse = (
        result.replace(m.ROOT48, m.ROOT45)
        .replace(m.PROBE48, m.PROBE45)
        .replace('"evc48"', '"evc45"')
    )
    assert ast.dump(ast.parse(inverse)) == ast.dump(ast.parse(source))


@pytest.mark.parametrize("name", ["worker.py", "driver.py", "stage.py"])
def test_unknown_original_scientific_source_cannot_be_retargeted(name):
    with pytest.raises(ValueError, match="Exact original"):
        m.build(name, (ORIGINAL / (name + ".txt")).read_text() + "\n# changed\n")


def test_cpu_preparation_cannot_dispatch_gpu_or_skip_old_job_guard():
    source = m.build("stage.py", (ORIGINAL / "stage.py.txt").read_text())
    tree = ast.parse(source)
    assert not any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "run"
        for n in ast.walk(tree)
    )
    assert '"975129"' in source and '"Reason=JobHeldUser"' in source
    assert "actual_cached48_full_fitting_host_CPU_prepared_no_GPU_dispatch" in source
    assert 'v["accepted_module_sha256"]' in source
