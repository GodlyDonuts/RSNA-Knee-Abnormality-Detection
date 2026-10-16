import ast
from pathlib import Path

import pytest

from scripts.kaggle.regional_feature_probe import _build, build
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


def test_new_probe_keeps_old_functions_and_calls_only_private_entrypoint():
    parent = "def original_renderer():\n    return 19\n" + GUARD + "\n"
    text, binding = _build(parent, *sources(), [f"safe-{i:02d}" for i in range(32)])
    tree = ast.parse(text)
    assert binding["studies"] == 32 and binding["competition_submission"] is False
    assert binding["representation"] == "same_forward_global_regional_v1"
    assert binding["same_forward_global_regional_and_parent_logits"] is True
    old = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "original_renderer"
    )
    assert ast.dump(old, include_attributes=False) == ast.dump(
        ast.parse(parent).body[0], include_attributes=False
    )
    assert "bounded_results(pool,_regional_pre_one,_REGIONAL_ROSTER,limit=4)" in text
    assert "cacheN" not in text and "tarfile.open(archive_path,'x:')" in text
    assert "'storage_dtype':'float16'" in text and "'channels_retained':1024" in text
    guard = tree.body[-1]
    assert isinstance(guard, ast.If)
    assert [
        n.func.id
        for n in ast.walk(guard)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ] == ["_regional_probe_main"]
    summary = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "regional_summary"
    )
    assert len(summary.decorator_list) == 1
    assert binding["new_feature_cap_bytes"] == 128 * 2**20
    assert "global_features.append(actual);parent_logits.append(p[0].cpu().numpy())" in text
    assert "'features':np.stack(global_features),'parent_logits':np.stack(parent_logits)" in text
    assert "assert error<=.001" not in text
    assert "assert pe<=.003" in text
    assert "'old_global_embeddings_used_as_new_inputs':False" in text
    assert "regional-probe-observation.json" in text


def test_unbound_public_parent_wrong_roster_and_helpers_are_rejected():
    parent = "def old():\n    pass\n" + GUARD + "\n"
    roster = [f"safe-{i:02d}" for i in range(32)]
    context, regional, bounded = sources()
    with pytest.raises(ValueError):
        build(parent, context, regional, bounded, roster)
    for ids in [roster[:31], roster[::-1], roster[:-1] + [roster[0]]]:
        with pytest.raises(ValueError):
            _build(parent, context, regional, bounded, ids)
    with pytest.raises(ValueError):
        _build(parent, context + "\n", regional, bounded, roster)
    with pytest.raises(ValueError):
        _build(parent, context, regional, bounded + "\n", roster)
