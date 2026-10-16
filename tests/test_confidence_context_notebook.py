import hashlib
from pathlib import Path

import pytest
import torch
from test_confidence_context_infer import synthetic
from test_released_context_infer import inputs
from test_released_context_notebook import fake_parent, sources

from rsna.competitive import confidence_context_infer as c
from scripts.kaggle.confidence_context_notebook import _build, build


@pytest.mark.parametrize("mode", ["qualification", "competition"])
def test_compiled_typed_component_matches_standalone_real_math(tmp_path, mode):
    cp, terminal, recovery, parity, _, _, _, _ = synthetic(tmp_path)
    out = tmp_path / "component.pt"
    receipt = c.export_component(cp, terminal, recovery, parity, out)
    value = torch.load(out, weights_only=True)
    infer = (
        Path(__file__).resolve().parents[1] / "rsna/competitive/confidence_context_infer.py"
    ).read_text()
    text, binding = _build(
        fake_parent(),
        *sources(),
        infer,
        receipt["component_sha256"],
        mode=mode,
        reference_sha="b" * 64 if mode == "qualification" else None,
    )
    ns = {"__name__": "synthetic_confidence_notebook"}
    exec(compile(text, "synthetic_confidence_notebook", "exec"), ns)
    model = ns["model_from_component"](value)
    got, delta = ns["predict_context_pair"](model, *inputs())
    expected, correction = c.predict_context_pair(c.model_from_component(value), *inputs())
    assert torch.equal(got, expected) and torch.equal(delta, correction)
    assert ns["FORMAT"] == c.FORMAT and ns["FIXED_RECIPE"] == c.FIXED_RECIPE
    assert binding["source_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert binding["original_bounded_MRI_path_unchanged"]
    assert "nproc = 2" in text and "limit=4" in text and "maxsize=2" in text
    with pytest.raises(ValueError):
        build(fake_parent(), *sources(), infer, "a" * 64, mode="competition")


def test_changed_confidence_source_rejected():
    infer = "# unknown source"
    with pytest.raises(ValueError, match="confidence serving"):
        _build(fake_parent(), *sources(), infer, "a" * 64, mode="competition")
