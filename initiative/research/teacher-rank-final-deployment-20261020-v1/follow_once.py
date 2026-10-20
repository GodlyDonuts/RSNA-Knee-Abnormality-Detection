# ruff: noqa: F821
"""Finite separate ranking owner. Disabled until a concrete runtime decision is frozen."""

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("rank_frozen_transport", _HERE / "run_once.py")
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)
exec(
    compile(
        _runner._delta.owner_source(_runner._parent), "frozen_pointwise_owner_rank_route", "exec"
    ),
    globals(),
)

if __name__ == "__main__":
    main()
