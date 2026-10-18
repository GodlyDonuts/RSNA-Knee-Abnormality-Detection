"""The new final-analysis loader must bind __file__ without changing AUC math."""

import runpy
import types
from pathlib import Path

import numpy as np
import pytest


def test_original_rank_source_requires_file_and_recovery_preserves_weighted_auc(tmp_path):
    path = Path("scripts/rank_learning.py").resolve()
    source = path.read_text()
    broken = types.ModuleType("original_component_uncertainty")
    with pytest.raises(NameError, match="__file__"):
        exec(compile(source, "original_rank_functions", "exec"), broken.__dict__)
    copied = tmp_path / "rank-analysis-source-v2.py"
    copied.write_text(source)
    fixed = types.ModuleType("original_component_uncertainty")
    fixed.__file__ = str(copied)
    exec(compile(source, str(copied), "exec"), fixed.__dict__)
    normal = runpy.run_path(str(path), run_name="reference_rank_functions")
    for weights in (np.ones(6), np.array([0, 1, 2, 3, 0, 1])):
        truth = np.array([0, 1, 0, 1, 1, 0])
        scores = np.array([0.1, 0.8, 0.2, 0.7, 0.7, 0.7])
        actual = fixed.weighted_auc(fixed.auc_plan(truth, scores), weights)
        expected = normal["weighted_auc"](normal["auc_plan"](truth, scores), weights)
        assert actual == expected
