# ruff: noqa: F821
"""Reuse the exact launched pointwise transport; derive only the distinct ranking route."""

import importlib.util
import json
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parents[2]
_spec = importlib.util.spec_from_file_location("rank_deployment_delta", _THIS / "derive.py")
_delta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_delta)
_parent = _delta.parent_files(_REPO, json.loads((_THIS / "frozen-parent.json").read_text()))
exec(
    compile(_delta.runner_source(_parent), "frozen_pointwise_transport_rank_binding", "exec"),
    globals(),
)


def controls(pin):
    files = (
        "derive.py",
        "run_once.py",
        "follow_once.py",
        "frozen-parent.json",
        "config.json",
        "engineering-decision.json",
    )
    frozen = {}
    for name in files:
        p = D / name
        data = git_bytes(pin, p)
        assert data == p.read_bytes(), "Exact committed ranking control source required"
        frozen[name] = data
    config = json.loads(frozen["config.json"])
    _delta.accepted(config, frozen["engineering-decision.json"].decode())
    return _delta.sources(_parent, config, frozen["engineering-decision.json"].decode())


if __name__ == "__main__":
    main()
