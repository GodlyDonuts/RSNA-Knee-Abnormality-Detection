"""Regression for the actual structured Kaggle log that stopped CPU admission."""

import ast
import json
import math
from pathlib import Path

import pytest


def parser():
    p = Path("initiative/research/S6-Kaggle-existing-output-replay-20261019-v1/continue.py.txt")
    node = next(
        n
        for n in ast.parse(p.read_text()).body
        if isinstance(n, ast.FunctionDef) and n.name == "phase_events"
    )
    scope = {"json": json, "math": math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(p), "exec"), scope)
    return scope["phase_events"]


def test_actual_json_enveloped_stdout_and_stderr_preserve_six_events():
    expected = [
        {"role": a, "phase": b, "elapsed_seconds": float(i + 1)}
        for i, (a, b) in enumerate(
            (a, b)
            for a in ("parent_only", "cross_lineage_equal_rank")
            for b in ("full", "stop", "resume")
        )
    ]
    rows = [
        {"stream_name": "stdout", "time": z["elapsed_seconds"], "data": json.dumps(z) + "\n"}
        for z in expected
    ]
    rows.append({"stream_name": "stderr", "time": 509.3, "data": "KeyError: 'uniform'\n"})
    assert parser()(json.dumps(rows)) == expected
    with pytest.raises(AssertionError):
        parser()(json.dumps(rows[:-2]))
    with pytest.raises(AssertionError):
        parser()(json.dumps(list(reversed(rows))))
