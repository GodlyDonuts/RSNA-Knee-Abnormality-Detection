import ast
from pathlib import Path

import pytest
from test_licensed_importance_serving import accepted_epoch as accepted_epoch
from test_licensed_parent_serving_notebook import fixture_parent

from scripts.kaggle import licensed_importance_serving as compiler

ROOT = Path(__file__).resolve().parents[1]


def sources():
    return [
        (ROOT / n).read_text()
        for n in (
            "rsna/competitive/bounded_preparation.py",
            "rsna/competitive/licensed_importance_serving.py",
            "rsna/competitive/licensed_pixel_cache.py",
            "rsna/competitive/licensed_full_bag.py",
            "rsna/competitive/licensed_importance.py",
        )
    ]


@pytest.mark.parametrize("mode", ["qualification", "competition"])
def test_synthetic_notebook_keeps_parent_full94_math_and_requires_actual_weights(
    monkeypatch, accepted_epoch, mode
):
    _, training, publication, _ = accepted_epoch
    parent = fixture_parent()
    monkeypatch.setattr(compiler, "PARENT_SOURCE_SHA256", compiler.digest(parent))
    source, binding = compiler.build(parent, *sources(), training, publication, mode=mode)
    ns = {"__name__": "synthetic_compile_only_not_GPU_acceptance"}
    exec(compile(source, "synthetic_IID16_serving_fixture", "exec"), ns)
    assert ns["ARMS"][0]["file"] == "epoch-001.pt"
    assert ns["infer_pair_probs"]() == "unchanged original two-view math"
    assert ns["gpu_windows"](None, None, 94, 320, "cuda:0") == "unchanged original images"
    assert (ns["infer_probs"](None, None, None) == 1).all()
    assert "weights_only=True" in source and "pool.imap_unordered" not in source
    assert "class ImportanceReadout" in ns["_ADAPTED_IMPORTANCE"]
    assert "class TrainingGraph(importance['ImportanceReadout']" in source
    assert binding["role"] == "influence" and binding["training_windows"] == 16
    assert binding["serving_windows"] == 94 and binding["new_weight_Dataset_required"] is False
    assert binding["qualification_cannot_submit"] == (mode == "qualification")
    assert isinstance(ast.parse(source).body[-1], ast.If)


@pytest.mark.parametrize(
    "change", ["parent", "bounded", "helper", "codec", "mixin", "sampler", "mode"]
)
def test_unaccepted_source_or_mode_cannot_generate_a_notebook(monkeypatch, accepted_epoch, change):
    _, training, publication, _ = accepted_epoch
    parent = fixture_parent()
    monkeypatch.setattr(compiler, "PARENT_SOURCE_SHA256", compiler.digest(parent))
    helpers = sources()
    mode = "qualification"
    if change == "parent":
        parent += "changed"
    elif change == "mode":
        mode = "diagnostic"
    else:
        helpers[["bounded", "helper", "codec", "mixin", "sampler"].index(change)] += "# changed\n"
    with pytest.raises(ValueError):
        compiler.build(parent, *helpers, training, publication, mode=mode)
