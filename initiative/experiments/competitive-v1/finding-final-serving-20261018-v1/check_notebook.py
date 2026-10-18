"""Synthetic compiler checks retain the exact parent graph and full94 inference."""

import importlib.util
from pathlib import Path

import pytest
from check_serving import records as records
from test_licensed_importance_serving_notebook import sources
from test_licensed_parent_serving_notebook import fixture_parent

HERE = Path(__file__).parent
spec = importlib.util.spec_from_file_location("finding_notebook_compiler", HERE / "notebook.py")
compiler = importlib.util.module_from_spec(spec)
spec.loader.exec_module(compiler)


@pytest.mark.parametrize("mode", ["qualification", "competition"])
def test_original_full94_math_and_final_weight_only(monkeypatch, records, mode):
    _, training, publication, _ = records
    parent = fixture_parent()
    monkeypatch.setattr(compiler.original, "PARENT_SOURCE_SHA256", compiler.original.digest(parent))
    bounded, base, codec, mixin, importance = sources()
    source, binding = compiler.build(
        parent,
        bounded,
        base,
        (HERE / "serving.py").read_text(),
        codec,
        mixin,
        importance,
        training,
        publication,
        mode=mode,
    )
    ns = {"__name__": "synthetic_compile_only"}
    exec(compile(source, "synthetic_finding_notebook", "exec"), ns)
    assert ns["ARMS"][0]["file"] == "epoch-003.pt"
    assert ns["infer_pair_probs"]() == "unchanged original two-view math"
    assert ns["gpu_windows"](None, None, 94, 320, "cuda:0") == "unchanged original images"
    assert binding["fixed_final_only"] and binding["serving_windows"] == 94
    assert binding["original_target_sha256"] == compiler.label_scope.TARGET
    assert binding["qualification_cannot_submit"] == (mode == "qualification")
    assert "actual_finding_recovery_kernel=139349540" in source
    assert "safe_family_mean" not in source


@pytest.mark.parametrize("damage", ["helper", "base", "compiler", "early"])
def test_modified_sources_or_earlier_epoch_refused(monkeypatch, records, damage):
    _, training, publication, _ = records
    parent = fixture_parent()
    monkeypatch.setattr(compiler.original, "PARENT_SOURCE_SHA256", compiler.original.digest(parent))
    bounded, base, codec, mixin, importance = sources()
    helper = (HERE / "serving.py").read_text()
    if damage == "helper":
        helper += "# changed\n"
    elif damage == "base":
        base += "# changed\n"
    elif damage == "compiler":
        monkeypatch.setattr(compiler, "BASE_COMPILER_SHA", "0" * 64)
    else:
        training["epoch"] = 2
    with pytest.raises(ValueError):
        compiler.build(
            parent,
            bounded,
            base,
            helper,
            codec,
            mixin,
            importance,
            training,
            publication,
            mode="qualification",
        )
