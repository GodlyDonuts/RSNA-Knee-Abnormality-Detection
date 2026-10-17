import hashlib
import json
from types import SimpleNamespace

import pytest

from scripts.kaggle import public_reference_script as s


def fixture():
    cells = [f"value_{i}={i}\n" for i in range(8)]
    obj = {
        "nbformat": 4,
        "metadata": {},
        "cells": [
            {"cell_type": "markdown", "source": "Original credits and source links"},
            *[{"cell_type": "code", "source": c} for c in cells],
        ],
    }
    text = json.dumps(obj)
    kwargs = {
        "notebook_sha": s.digest(text),
        "cells_sha": s.digest(json.dumps(cells, ensure_ascii=False, separators=(",", ":"))),
    }
    return text, cells, kwargs


def test_complete_source_and_credits_roundtrip():
    text, cells, kwargs = fixture()
    assert s.code_cells(text, **kwargs) == (cells, "Original credits and source links")
    source, proof = s.build(text, **kwargs)
    compile(source, "synthetic_reference_driver", "exec")
    assert proof["script_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert proof["original_cells_byte_unchanged"]
    assert repr(cells) in source and "Original credits and source links" in source
    assert s.R6_MANIFEST in source and s.PUBLIC_CSV in source
    assert "shell.run_cell" in source and "run_cells(_CELLS" in source
    assert "_own.csv" in source and "exact8cell-script-accepted.json" in source
    # Public example fidelity is checked by the public-run coordinator, never
    # used to reject a different three-study hidden roster during inference.
    assert "if len(ids) == 3" not in source


@pytest.mark.parametrize("field", ["notebook_sha", "cells_sha"])
def test_reject_source_or_cells_drift(field):
    text, _, kwargs = fixture()
    kwargs[field] = "0" * 64
    with pytest.raises(ValueError):
        s.build(text, **kwargs)


@pytest.mark.parametrize("count", [0, 1, 7, 9])
def test_reject_partial_or_extra_cells(count):
    text, cells, _ = fixture()
    obj = json.loads(text)
    code = [f"v={i}" for i in range(count)]
    obj["cells"] = [obj["cells"][0]] + [{"cell_type": "code", "source": c} for c in code]
    text = json.dumps(obj)
    with pytest.raises(ValueError):
        s.code_cells(
            text,
            notebook_sha=s.digest(text),
            cells_sha=s.digest(json.dumps(code, ensure_ascii=False, separators=(",", ":"))),
        )


def test_original_list_sources_are_preserved_not_escaped_twice():
    text, cells, kwargs = fixture()
    obj = json.loads(text)
    for c in obj["cells"]:
        c["source"] = c["source"].splitlines(keepends=True)
    text = json.dumps(obj)
    kwargs["notebook_sha"] = s.digest(text)
    assert s.code_cells(text, **kwargs)[0] == cells


@pytest.mark.parametrize("source", [None, 12, ["ok", 1]])
def test_reject_invalid_source_types(source):
    text, _, kwargs = fixture()
    obj = json.loads(text)
    obj["cells"][1]["source"] = source
    text = json.dumps(obj)
    kwargs["notebook_sha"] = s.digest(text)
    with pytest.raises(ValueError):
        s.code_cells(text, **kwargs)


def test_execution_order_and_no_history_side_effect():
    seen = []

    def runner(cell, **kw):
        seen.append((cell, kw))
        return SimpleNamespace(error_before_exec=None, error_in_exec=None)

    assert s.run_cells(["a", "b", "c"], runner) == [0, 1, 2]
    assert seen == [(c, {"store_history": False}) for c in ("a", "b", "c")]


@pytest.mark.parametrize("field", ["error_before_exec", "error_in_exec"])
def test_actual_error_prevents_later_cells(field):
    seen = []

    def runner(cell, **kw):
        seen.append(cell)
        result = SimpleNamespace(error_before_exec=None, error_in_exec=None)
        setattr(result, field, SyntaxError("bad original stage"))
        return result

    with pytest.raises(RuntimeError, match="cell 0 failed"):
        s.run_cells(["bad", "must not execute"], runner)
    assert seen == ["bad"]


def test_production_defaults_reject_synthetic_source():
    text, _, _ = fixture()
    with pytest.raises(ValueError):
        s.build(text)
