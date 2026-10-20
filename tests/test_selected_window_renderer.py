"""Synthetic center-selection semantics; actual MRI acceptance is separate."""

import hashlib
import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

D = Path("initiative/research/selected-window-rendering-20261020-v1")
loader = importlib.machinery.SourceFileLoader("selected_renderer", str(D / "renderer.py.txt"))
m = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
loader.exec_module(m)
SOURCE = """def eval_windows(vol,mask,k,res,norm="original",mirror=False):
    D=len(vol)
    cs = _eval_centers(mask,D,k)
    return [(vol[c],norm,mirror) for c in cs]
"""


@pytest.mark.parametrize("positions", [list(range(94)), [93, 0, 50, 10], [2, 2, 2, 0]])
def test_same_centers_order_duplicates_and_view(monkeypatch, positions):
    monkeypatch.setattr(m, "PARENT_SHA", hashlib.sha256(SOURCE.encode()).hexdigest())
    ns = {"_eval_centers": lambda mask, D, k: list(range(k))}
    exec(SOURCE, ns)
    selected = m.build(SOURCE, ns)
    for mirror in (False, True):
        full = ns["eval_windows"](list(range(94)), None, 94, 320, mirror=mirror)
        assert selected(list(range(94)), None, 94, 320, positions, mirror=mirror) == [
            full[i] for i in positions
        ]


@pytest.mark.parametrize("positions", [[], [-1], [94], [True], [1.0]])
def test_unknown_positions_cannot_change_geometry(monkeypatch, positions):
    monkeypatch.setattr(m, "PARENT_SHA", hashlib.sha256(SOURCE.encode()).hexdigest())
    ns = {"_eval_centers": lambda mask, D, k: list(range(k))}
    selected = m.build(SOURCE, ns)
    with pytest.raises(ValueError):
        selected(list(range(94)), None, 94, 320, positions)


def test_unknown_original_source_refused():
    with pytest.raises(ValueError):
        m.build(SOURCE, {})
