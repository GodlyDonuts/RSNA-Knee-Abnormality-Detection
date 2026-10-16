import ast
import hashlib
from dataclasses import asdict, is_dataclass
from pathlib import Path

import pytest
import torch
from test_released_context_infer import component, inputs

from rsna.competitive.released_context_infer import predict_context_pair
from rsna.competitive.released_context_train import Recipe
from scripts.kaggle.released_context_notebook import _build, bodies, build

ROOT = Path(__file__).resolve().parents[1]


def sources():
    return [
        (ROOT / name).read_text()
        for name in (
            "rsna/competitive/released_context.py",
            "rsna/competitive/released_context_train.py",
            "rsna/competitive/released_context_infer.py",
            "rsna/competitive/bounded_preparation.py",
        )
    ]


def fake_parent():
    return """
import torch,numpy as np,threading,queue,os,json,time
LAB=['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
NORM='imagenet'
def gpu_windows(*args,**kwargs):return 'original pixels'
def main():
    nproc = max(4, min(96, (os.cpu_count() or 8) - 8))
    q = queue.Queue(maxsize=QUEUE_MAX)
    for item in pool.imap_unordered(_pre_one, test_ids, chunksize=2):pass
    print(f"  [gpu {d}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)
if __name__ == "__main__":
    main()"""


@pytest.mark.parametrize("mode", ["qualification", "competition"])
def test_generated_source_preserves_decorators_and_exact_actual_readout(mode):
    kwargs = {"mode": mode, "reference_sha": "b" * 64 if mode == "qualification" else None}
    source, binding = _build(fake_parent(), *sources(), "a" * 64, **kwargs)
    ns = {"__name__": "test_generated_context"}
    exec(compile(source, "test_generated_context", "exec"), ns)
    assert is_dataclass(ns["Recipe"]) and asdict(ns["Recipe"]()) == asdict(Recipe())
    assert (
        binding["mode"] == mode
        and binding["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    )
    value = component()
    m = ns["model_from_component"](value)
    p, c = ns["predict_context_pair"](m, *inputs())
    expected, delta = predict_context_pair(m, *inputs())
    assert torch.equal(p, expected) and torch.equal(c, delta)
    assert not p.requires_grad
    assert "pool.imap_unordered" not in source and "nproc = 2" in source
    assert "sid[:16]" not in source
    parsed = ast.parse(source)
    last = parsed.body[-1]
    assert isinstance(last, ast.If)
    assert "_context_qualify" in ast.unparse(last) and "_context_competition" in ast.unparse(last)
    assert ns["_CONTEXT_MODE"] == mode
    assert binding["qualification_cannot_submit"] == (mode == "qualification")


def test_extractor_keeps_no_grad_and_dataclass_decorators():
    source = """
from __future__ import annotations
from dataclasses import dataclass
@dataclass(frozen=True)
class A:
    n:int=1
@torch.no_grad()
def f(x):
    return x+1
"""
    text = bodies(source, selected={"A", "f"})
    assert "@dataclass(frozen=True)" in text and "@torch.no_grad()" in text
    assert "__future__" not in text


@pytest.mark.parametrize(
    "change", ["parent", "trainer", "helper", "infer", "bounded", "mode", "component", "reference"]
)
def test_unknown_sources_and_invalid_modes_fail_before_source_creation(change):
    parent = fake_parent()
    context, trainer, infer, bounded = sources()
    mode, csha, rsha = "qualification", "a" * 64, "b" * 64
    if change == "parent":
        with pytest.raises(ValueError, match="licensed"):
            build(parent, context, trainer, infer, bounded, csha, mode=mode, reference_sha=rsha)
        return
    if change == "trainer":
        trainer += "\n# numerical source changed\n"
    elif change == "helper":
        context += "\n# helper changed\n"
    elif change == "infer":
        infer += "\n# serving source changed\n"
    elif change == "bounded":
        bounded += "\n# producer source changed\n"
    elif change == "mode":
        mode = "unknown"
    elif change == "component":
        csha = "not a sha"
    else:
        rsha = None
    with pytest.raises(ValueError):
        _build(parent, context, trainer, infer, bounded, csha, mode=mode, reference_sha=rsha)
