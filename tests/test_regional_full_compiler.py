import ast
from pathlib import Path

import pytest

from scripts.kaggle.regional_feature_full import NEW_ARCHIVE_CAP, _build, build
from scripts.kaggle.series_boundary_notebook import GUARD

ROOT = Path(__file__).resolve().parents[1]


def sources():
    return [
        (ROOT / name).read_text()
        for name in (
            "rsna/competitive/released_context.py",
            "rsna/competitive/regional_features.py",
            "rsna/competitive/bounded_preparation.py",
        )
    ]


def test_full_cache_reuses_accepted32_and_encodes_only_new3855():
    parent = "def original_renderer():\n    return 19\n" + GUARD + "\n"
    ids = [f"safe-{i:04d}" for i in range(3887)]
    text, binding = _build(parent, *sources(), ids)
    tree = ast.parse(text)
    assert binding["reused_studies"] == 32 and binding["new_CNN_studies"] == 3855
    assert binding["studies"] == 3887 and binding["new_archive_cap_bytes"] == NEW_ARCHIVE_CAP
    assert "bounded_results(pool,_regional_pre_one,_REGIONAL_ROSTER[32:],limit=4)" in text
    assert "records=list(reused)" in text and "dict(r,source_key='regional32')" in text
    assert "'source_key':'self'" in text
    main = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_regional_probe_main"
    )
    calls = [
        n.func.id
        for n in ast.walk(main)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "_regional_reuse32" in calls and "_regional_old_cache" not in calls
    assert "old[sid]" not in ast.unparse(main)
    assert "released-regional-features-full" in text
    assert "regional-full-features-accepted.json" in text
    assert "same_forward_global_regional_and_parent_logits" in text
    old = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "original_renderer"
    )
    assert ast.dump(old, include_attributes=False) == ast.dump(
        ast.parse(parent).body[0], include_attributes=False
    )


def test_full_cache_cannot_use_an_arbitrary_roster_or_public_parent():
    parent = "def old():\n    pass\n" + GUARD + "\n"
    ids = [f"safe-{i:04d}" for i in range(3887)]
    with pytest.raises(ValueError):
        build(parent, *sources(), ids)
    for roster in (ids[:32], ids[:-1], ids[::-1], ids[:-1] + [ids[0]]):
        with pytest.raises(ValueError):
            _build(parent, *sources(), roster)
