import ast
import contextlib
import hashlib
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_regional_context_infer import component, inputs
from test_released_context_notebook import fake_parent

from rsna.competitive.regional_context_infer import predict_pair
from rsna.competitive.released_context_train import Recipe
from scripts.kaggle.regional_context_notebook import SOURCE_HASHES, _build, build

ROOT = Path(__file__).resolve().parents[1]


def sources():
    return {
        k: (ROOT / "rsna/competitive" / (name + ".py")).read_text()
        for k, name in {
            "context": "released_context",
            "trainer": "released_context_train",
            "summaries": "regional_features",
            "regional": "regional_context_train",
            "infer": "regional_context_infer",
            "common": "released_context_infer",
            "bounded": "bounded_preparation",
        }.items()
    }


def generated(role="regional", mode="qualification"):
    return _build(
        fake_parent(),
        sources(),
        "a" * 64,
        role=role,
        mode=mode,
        reference_sha="b" * 64 if mode == "qualification" else None,
    )


@pytest.mark.parametrize("role", ["global_repeat", "regional"])
@pytest.mark.parametrize("mode", ["qualification", "competition"])
def test_generated_helpers_exact_math_role_quantization_and_runtime_modes(role, mode):
    text, receipt = generated(role, mode)
    ns = {"__name__": "generated_spatial_test"}
    exec(compile(text, "generated_spatial_test", "exec"), ns)
    assert asdict(ns["Recipe"](embedding_dim=8192)) == asdict(Recipe(embedding_dim=8192))
    model = ns["model_from_component"](component(role), role)
    actual, correction = ns["predict_pair"](model, role, *inputs())
    expected, delta = predict_pair(model, role, *inputs())
    assert torch.equal(actual, expected) and torch.equal(correction, delta)
    assert not actual.requires_grad
    assert receipt["source_sha256"] == hashlib.sha256(text.encode()).hexdigest()
    assert receipt["source_module_sha256"] == SOURCE_HASHES and receipt["role"] == role
    assert ns["_SPATIAL_ROLE"] == role and ns["_SPATIAL_MODE"] == mode
    assert receipt["qualification_cannot_submit"] == (mode == "qualification")
    assert "pool.imap_unordered" not in text and "sid[:16]" not in text
    assert "nproc = 2" in text and "queue.Queue(maxsize=2)" in text
    assert "register_forward_pre_hook" in text
    tree = ast.parse(text)
    qualifier = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_spatial_qualify"
    )
    assert not any(
        isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "main"
        for n in ast.walk(qualifier)
    )
    assert "tar.extractfile" in ast.unparse(qualifier)
    assert isinstance(tree.body[-1], ast.If)
    assert "_spatial_qualify" in ast.unparse(tree.body[-1])
    assert "_spatial_competition" in ast.unparse(tree.body[-1])


@pytest.mark.parametrize("changed", list(SOURCE_HASHES))
def test_each_changed_source_is_refused_before_compilation(changed):
    value = sources()
    value[changed] += "\n# changed\n"
    with pytest.raises(ValueError):
        _build(fake_parent(), value, "a" * 64, role="regional", mode="competition")


@pytest.mark.parametrize("changed", ["parent", "role", "mode", "component", "reference", "extra"])
def test_unknown_contracts_and_wrong_role_are_refused(changed):
    value = sources()
    kwargs = {"role": "regional", "mode": "qualification", "reference_sha": "b" * 64}
    sha = "a" * 64
    if changed == "parent":
        with pytest.raises(ValueError, match="licensed"):
            build(fake_parent(), value, sha, **kwargs)
        return
    if changed == "role":
        kwargs["role"] = "other"
    elif changed == "mode":
        kwargs["mode"] = "other"
    elif changed == "component":
        sha = "invalid"
    elif changed == "reference":
        kwargs["reference_sha"] = None
    else:
        value["unknown"] = ""
    with pytest.raises(ValueError):
        _build(fake_parent(), value, sha, **kwargs)


def test_encoder_error_removes_active_observer_before_next_study():
    text, _ = generated()
    ns = {"__name__": "generated_failure_test"}
    exec(compile(text, "generated_failure_test", "exec"), ns)
    captures = []

    class Capture:
        def __init__(self, backbone):
            self.removed = False
            self.handle = SimpleNamespace(remove=self.remove)
            captures.append(self)

        def begin(self):
            pass

        def remove(self):
            self.removed = True

    class Parent:
        backbone = "fake graph, no GPU or clinical data"

        def modules(self):
            return [SimpleNamespace(training=False)]

        def encode(self, _):
            raise RuntimeError("synthetic invalid study")

    class TorchProxy:
        cuda = SimpleNamespace(device=lambda _: contextlib.nullcontext())

        def __getattr__(self, name):
            return getattr(torch, name)

        def autocast(self, *args, **kwargs):
            return contextlib.nullcontext()

    ns["RegionalCapture"], ns["torch"] = Capture, TorchProxy()
    plain = {
        "mirror": False,
        "centers": torch.arange(1, 95),
        "present": torch.ones(94, dtype=torch.bool),
        "windows": torch.zeros(1),
    }
    mirrored = dict(plain, mirror=True)
    model = Parent()
    for _ in range(2):
        with pytest.raises(RuntimeError, match="synthetic"):
            ns["infer_pair_probs"](model, plain, mirrored, "fake-cpu-device")
        assert not ns["_spatial_captures"]
    assert len(captures) == 2 and all(c.removed for c in captures)
    assert ns["_spatial_counts"]["pairs_started"] == 2
    assert ns["_spatial_counts"]["pairs_completed"] == 0
