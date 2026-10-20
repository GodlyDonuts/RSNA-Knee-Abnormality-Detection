"""A time-bounded metadata pass must resume without changing image operations."""

import ast
import importlib.machinery
import importlib.util
from pathlib import Path

import pytest

D = Path("initiative/research/selected-window-rendering-20261020-v1")
loader = importlib.machinery.SourceFileLoader(
    "window_prefix_continuation", str(D / "continue_manifest.py.txt")
)
m = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
loader.exec_module(m)
SOURCE = (D / "manifest_stage.py.txt").read_text()


def test_every_decoding_rendering_and_hash_operation_is_unchanged():
    old = next(
        n
        for n in ast.parse(SOURCE).body
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "i"
    )
    new = next(
        n
        for n in ast.parse(m.build(SOURCE)).body
        if isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "i"
    )
    assert ast.dump(old.body[0]) == ast.dump(new.body[0])
    assert ast.dump(new.iter) == ast.dump(ast.parse("range(prefix,3887)", mode="eval").body)


def test_existing_draft_is_not_truncated_or_original_intent_overwritten():
    result = m.build(SOURCE)
    assert 'mode="w+"' not in result and 'mmap_mode="r+"' in result
    assert '(D / "intent.json").write_text(' not in result
    assert '(D / "progress.json").write_text(' not in result
    assert '(K / "intent.json").write_text(' in result
    assert "prefix * 2 + len(data.verified_inputs)" in result
    assert '["975028", "TIMEOUT"]' in result


def test_source_drift_cannot_retarget_an_existing_private_draft():
    with pytest.raises(ValueError, match="Exact original"):
        m.build(SOURCE + "\n# source drift\n")
