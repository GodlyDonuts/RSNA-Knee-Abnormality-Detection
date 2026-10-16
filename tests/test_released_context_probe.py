import ast
import hashlib
from pathlib import Path

import pytest

from scripts.kaggle.released_context_probe import _build, build
from scripts.kaggle.series_boundary_notebook import GUARD


def test_generated_probe_uses_parent_definitions_but_never_calls_parent_main():
    parent = 'def main():\n    raise AssertionError("parent submission executed")\n' + GUARD
    helper = Path("rsna/competitive/released_context.py").read_text()
    source, binding = _build(parent, helper, ["engineering-" + str(i) for i in range(32)])
    parsed = ast.parse(source)
    calls = [
        n.func.id
        for n in ast.walk(parsed)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "main" not in calls and "_context_probe_main" in calls
    assert binding["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert binding["studies"] == 32 and not binding["clinical_labels_reports"]
    assert "train_series.csv" in source and "train_images" in source
    assert "source_weight" in source  # helper is embedded; no fitting call exists.
    assert "optimizer.step" not in source
    assert "'source_weight':" not in source
    assert "'fitting_or_optimizer':False" in source
    assert "labels.csv" not in source and "train.csv" not in source
    assert "np.savez(handle,**arrays)" in source


@pytest.mark.parametrize("change", ["size", "duplicate", "suffix", "guard"])
def test_probe_rejects_incomplete_membership_or_changed_source_shape(change):
    parent = "def main():\n    pass\n" + GUARD
    roster = [str(i) for i in range(32)]
    if change == "size":
        roster.pop()
    elif change == "duplicate":
        roster[-1] = roster[0]
    elif change == "suffix":
        parent += "\nprint('unbound')"
    else:
        parent = "def main():\n    pass\n"
    with pytest.raises(ValueError):
        _build(parent, "", roster)


def test_public_builder_requires_original_licensed_source_bytes():
    with pytest.raises(ValueError, match="previously scored"):
        build("def main(): pass", "", [str(i) for i in range(32)])
