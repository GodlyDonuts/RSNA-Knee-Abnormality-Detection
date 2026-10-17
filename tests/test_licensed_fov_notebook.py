"""Prevent altered sources or accidental competition execution in the GPU experiment."""

from pathlib import Path

import pytest

from scripts.kaggle import licensed_fov_notebook as c

ROOT = Path(__file__).resolve().parents[1]


def fixture_parent():
    return """# Original CUDA boot deliberately stripped before inactive definitions.
raise RuntimeError("The discarded original boot must not execute")
# %% [code]
INPUT_RES = 320
def main():
    raise RuntimeError("The competition main must not execute")
if __name__ == "__main__":
    main()"""


def helpers():
    return [
        (ROOT / name).read_text()
        for name in (
            "rsna/competitive/licensed_pixel_cache.py",
            "rsna/competitive/licensed_fov_qualification.py",
        )
    ]


def test_original_definitions_are_inert_until_explicit_qualification(monkeypatch):
    parent = fixture_parent()
    monkeypatch.setattr(c, "PARENT_SOURCE_SHA256", c.digest(parent))
    source, binding = c.build(parent, *helpers())
    ns = {"__name__": "synthetic_non_GPU_compiler_test"}
    exec(compile(source, "inert_compiler_test", "exec"), ns)
    assert "The discarded original boot" not in ns["_FOV_PARENT_BODY"]
    original = {"__name__": "inactive_original_definitions"}
    exec(ns["_FOV_PARENT_BODY"], original)
    assert original["INPUT_RES"] == 320
    assert binding["roles"] == [320, 384] and binding["maximum_runtime_seconds"] == 1800
    assert binding["competition_entry"] is False
    assert binding["new_weight_or_image_Dataset_required"] is False
    assert binding["source_sha256"] == c.digest(source)


@pytest.mark.parametrize("change", ["parent", "codec", "runner"])
def test_unknown_input_sources_refuse_before_GPU_source_generation(monkeypatch, change):
    parent = fixture_parent()
    monkeypatch.setattr(c, "PARENT_SOURCE_SHA256", c.digest(parent))
    values = [parent, *helpers()]
    values[["parent", "codec", "runner"].index(change)] += "\n# unaccepted revision\n"
    with pytest.raises(ValueError):
        c.build(*values)


@pytest.mark.parametrize(
    "change", ["duplicate_guard", "after_main", "wrong_side", "duplicate_side"]
)
def test_ambiguous_control_or_original_execution_boundary_refuses(monkeypatch, change):
    parent = fixture_parent()
    if change == "duplicate_guard":
        parent += c.GUARD
    elif change == "after_main":
        parent += "\nraise RuntimeError('Unaccepted live code')"
    elif change == "wrong_side":
        parent = parent.replace("INPUT_RES = 320", "INPUT_RES = 384")
    else:
        parent = parent.replace("INPUT_RES = 320", "INPUT_RES = 320\nINPUT_RES = 320")
    monkeypatch.setattr(c, "PARENT_SOURCE_SHA256", c.digest(parent))
    with pytest.raises(ValueError):
        c.build(parent, *helpers())
