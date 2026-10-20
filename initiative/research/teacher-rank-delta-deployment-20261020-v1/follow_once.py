"""Finite exclusive owner of one public ranking version and at most one submission."""

import importlib.util
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("rank_delta_single_phase", _HERE / "run_once.py")
_runner = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_runner)
exec(
    compile(_runner._delta.owner_source(_HERE.parents[2]), "frozen_rank_delta_owner", "exec"),
    globals(),
)

if __name__ == "__main__":
    main()  # noqa: F821
