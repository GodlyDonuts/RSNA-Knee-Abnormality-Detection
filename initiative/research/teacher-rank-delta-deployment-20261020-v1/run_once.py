# ruff: noqa: F821
"""Exact frozen account transport; one candidate version, no duplicate qualification."""

import importlib.util
import json
from pathlib import Path

_THIS = Path(__file__).resolve().parent
_REPO = _THIS.parents[2]
_spec = importlib.util.spec_from_file_location("rank_single_run_derivation", _THIS / "derive.py")
_delta = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_delta)
exec(compile(_delta.runner_source(_REPO), "frozen_rank_delta_transport", "exec"), globals())


def controls(pin):
    own = {}
    for name in (
        "derive.py", "policy_functions.py", "run_once.py", "follow_once.py",
        "engineering-decision.json", "config.json",
    ):
        path = D / name
        own[name] = git_bytes(pin, path)
        assert own[name] == path.read_bytes(), "Exact committed delta controls required"
    serving = {}
    for name in ("adapter.py", "publiccheck.py", "prelude.py"):
        path = R / _delta.SERVING / name
        data = git_bytes(pin, path)
        assert data == path.read_bytes(), "Exact committed serving delta required"
        serving["rank_delta_adapter.py" if name == "adapter.py" else name] = data.decode()
    inherited_path = R / _delta.INHERITED_PATH
    inherited = git_bytes(pin, inherited_path)
    assert inherited == inherited_path.read_bytes()
    return _delta.sources(
        R, json.loads(own["config.json"]), own["engineering-decision.json"].decode(),
        own["policy_functions.py"].decode(), serving, inherited.decode(),
    )


if __name__ == "__main__":
    main()
