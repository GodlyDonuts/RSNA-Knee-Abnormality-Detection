import ast
import base64
import hashlib
import importlib.util
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

D = Path("initiative/research/selected-window-Kaggle-reader-20261020-v1")


def compiler():
    loader = SourceFileLoader("new_CPU_selected_compiler", str(D / "compile.py.txt"))
    spec = importlib.util.spec_from_loader(loader.name, loader)
    m = importlib.util.module_from_spec(spec)
    loader.exec_module(m)
    return m


def original(c, monkeypatch):
    worker = """def _eval_centers(mask,D,k):return list(range(k))
def eval_windows(vol,mask,k,res,norm=None,mirror=False):
 D=96
 cs = _eval_centers(mask,D,k)
 return [vol+i+int(mirror) for i in cs]
import gc,timm
def load_model():raise RuntimeError('Must never execute model code')
_PROOFS=[]
def main():raise RuntimeError('Must never execute original GPU main')
"""
    files = {
        "rsna/competitive/" + n + ".py": "# synthetic own module"
        for n in ("licensed_parent_population", "licensed_pixel_bags", "licensed_pixel_cache")
    }
    source = "\n".join(
        f"{n}={v!r}"
        for n, v in {
            "FILES": files,
            "WORKER": base64.b64encode(worker.encode()).decode(),
            "PERMISSION": base64.b64encode(b"synthetic permission").decode(),
        }.items()
    )
    monkeypatch.setattr(c, "BASE_SHA", hashlib.sha256(source.encode()).hexdigest())
    return source


def test_exact_selected_centers_order_duplicates_views_and_bad_indices(monkeypatch):
    c = compiler()
    source = original(c, monkeypatch)
    _, prefix, selected, _, _ = c.extract(source)
    assert "timm" not in prefix and "load_model" not in prefix
    ns = {}
    exec(compile(prefix + "\n" + selected, "synthetic-renderer-only", "exec"), ns)
    for mirror in (False, True):
        full = ns["eval_windows"](7, None, 94, 320, mirror=mirror)
        positions = [93, 0, 15, 15, 2, 31, 44, 74, 85, 90, 6, 8, 11, 32, 50, 1]
        assert ns["_selected_original_windows"](7, None, 94, 320, positions, mirror=mirror) == [
            full[i] for i in positions
        ]
    for positions in ([], [94], [-1], [False]):
        with pytest.raises(ValueError):
            ns["_selected_original_windows"](7, None, 94, 320, positions)
    with pytest.raises(ValueError):
        c.extract(source + "\n")


def test_cpu_bootstrap_excludes_model_definitions_and_binds_exact_reader(monkeypatch):
    c = compiler()
    source = original(c, monkeypatch)
    reader = Path("rsna/competitive/licensed_selected_pixel_bags.py").read_text()
    notebook, meta = c.build(source, reader, (D / "worker.py.txt").read_text(), "a" * 40)
    assert meta["maximum_new_GPU_jobs"] == 0 and meta["binding"]["CPU_only"]
    values = {
        n.targets[0].id: ast.literal_eval(n.value)
        for n in ast.parse(notebook).body
        if isinstance(n, ast.Assign)
        and isinstance(n.targets[0], ast.Name)
        and n.targets[0].id in ("FILES", "WORKER")
    }
    worker = base64.b64decode(values["WORKER"]).decode()
    assert "def load_model" not in worker and "import gc, timm" not in worker
    assert "torch.cuda.device_count() == 0" in worker
    assert (
        meta["binding"]["own_module_sha256"]["rsna/competitive/licensed_selected_pixel_bags.py"]
        == c.READER_SHA
    )
    with pytest.raises(ValueError):
        c.build(source, reader + "\n", (D / "worker.py.txt").read_text(), "a" * 40)
