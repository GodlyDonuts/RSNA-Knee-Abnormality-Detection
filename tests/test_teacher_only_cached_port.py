import ast
import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

D = Path("initiative/research/teacher-only-cached48-20261020-v1")
loader = importlib.machinery.SourceFileLoader("zero_port", str(D / "port.py.txt"))
m = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
loader.exec_module(m)


@pytest.mark.parametrize("name", ["worker.py", "driver.py"])
def test_only_host_paths_change_and_original_numerics_roundtrip(name):
    old = (
        Path("initiative/research/teacher-only-recovery45-20261020-v1") / (name + ".txt")
    ).read_text()
    new = m.build(name, old)
    reverse = new.replace(m.ROOT48, m.ROOT45).replace(m.F48, m.F45).replace('"evc48"', '"evc45"')
    assert ast.dump(ast.parse(reverse)) == ast.dump(ast.parse(old))


def test_preparation_cannot_dispatch_GPU_before_original_cancelled():
    old = Path("initiative/research/teacher-only-recovery45-20261020-v1/stage.py.txt").read_text()
    new = m.build("stage.py", old)
    assert "sbatch" not in new and "scancel" not in new
    assert 'new_probe_meta["gpu_worker_sha256"]' in new
    assert "(20 + 87 + 12 + 24)" in new
    assert "same_numerical_modules_as_original45" in new
