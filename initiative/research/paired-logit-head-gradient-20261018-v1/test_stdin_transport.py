"""New stdin transport carries large source without command interpolation."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "new_bounded_stdin_transport", Path(__file__).with_name("stdin_transport.py")
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def test_large_source_and_private_payload_round_trip_without_shell_interpolation():
    program = "#" + "x" * 20000 + "\nimport json,sys\nv=json.load(sys.stdin);print(json.dumps(v))\n"
    payload = {
        "text": 'literal `backtick` and $(never run)\nquotes " and Unicode 🦕',
        "blob": "a" * 100000,
    }
    r = subprocess.run(
        [sys.executable, "-c", m.COMMAND],
        input=m.envelope(program, payload),
        text=True,
        capture_output=True,
        check=True,
    )
    assert json.loads(r.stdout) == payload and len(m.COMMAND) < 300


@pytest.mark.parametrize(
    "program,payload", [("", {}), (None, {}), ("pass", []), ("pass", {"bad": float("nan")})]
)
def test_invalid_or_nonfinite_payload_refused_before_remote_execution(program, payload):
    with pytest.raises(ValueError):
        m.envelope(program, payload)
