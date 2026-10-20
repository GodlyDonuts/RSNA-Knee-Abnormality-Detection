"""New real-trainer shape/recovery tests with a tiny encoder, never MRI acceptance."""

import copy
import importlib.util
import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from rsna.competitive.cached_attention_training import CachedChangeReadout, Recipe, ReferenceCases
from rsna.competitive.licensed_parent_train import PARENT_SHA


class Parent(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(12, 4)
        self.norm = nn.LayerNorm(4)
        self.att = nn.Sequential(nn.Linear(4, 6), nn.Tanh(), nn.Dropout(0.3), nn.Linear(6, 12))
        self.clsW = nn.Parameter(torch.randn(12, 4) * 0.2)
        self.clsb = nn.Parameter(torch.zeros(12))

    def encode(self, images):
        return self.backbone(images.flatten(2))

    def forward(self, images):
        h = self.norm(self.encode(images))
        w = self.att(h).softmax(1)
        return (torch.einsum("bkn,bkf->bnf", w, h) * self.clsW).sum(-1) + self.clsb


class Model(CachedChangeReadout, Parent):
    pass


@pytest.fixture
def fixture():
    torch.set_num_threads(1)
    torch.manual_seed(23)
    model = Model().eval()
    images = torch.rand(17, 2, 94, 3, 2, 2)
    with torch.no_grad():
        h = model.norm(model.encode(images.reshape(34, 94, 3, 2, 2)))
        scores = model.att(h).reshape(17, 2, 94, 12)
        votes = torch.einsum("bkd,td->bkt", h, model.clsW).reshape(17, 2, 94, 12)
        parent = (scores.softmax(2) * votes).sum(2) + model.clsb

    class Original:
        uids = [f"synthetic-fitting-{i:04d}" for i in range(17)]

        def __len__(self):
            return 17

        def case(self, index, positions, mirror):
            return {
                "images": images[index, int(mirror), positions],
                "parent_logits": parent[index, int(mirror)],
                "labels": torch.full((12,), 0.2),
                "mask": torch.ones(12, dtype=torch.bool),
            }

    proposals = torch.full((17, 94), 1 / 94, dtype=torch.float64)
    original = Original()
    data = ReferenceCases(original, proposals, original.uids, scores, votes, "a" * 64)
    return model, data, scores, votes


def engine():
    path = Path("initiative/research/cached-attention-encoder-20261020-v1/engine.py.txt")
    loader = SourceFileLoader("new_cached_attention_engine_builder", str(path))
    builder = importlib.util.module_from_spec(importlib.util.spec_from_loader(loader.name, loader))
    loader.exec_module(builder)
    original = Path("rsna/competitive/licensed_importance_train.py").read_text()
    source = builder.build(original)
    ns = {"__name__": "new_cached_attention_numerical_engine", "__file__": "synthetic_only.py"}
    exec(compile(source, "new_cached_attention_numerical_engine", "exec"), ns)
    return ns["fit"], source


def eq(left, right):
    if isinstance(left, torch.Tensor):
        return torch.equal(left, right)
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(eq(left[k], right[k]) for k in left)
    if isinstance(left, (tuple, list)):
        return (
            type(left) is type(right)
            and len(left) == len(right)
            and all(eq(a, b) for a, b in zip(left, right))
        )
    return left == right


@pytest.mark.parametrize("variant", ["reference", "linear"])
def test_complete_optimizer_resume_and_original_inference(fixture, tmp_path, variant):
    model, data, _, _ = fixture
    fit, source = engine()
    recipe = Recipe(epochs=2, variant=variant)
    initial = copy.deepcopy(model.state_dict())

    def fresh(output, *, resume=False, stop_after=None):
        m = copy.deepcopy(model)
        m.load_state_dict(initial)
        m.cached_attention_variant = variant
        random.seed(5)
        np.random.seed(5)
        torch.manual_seed(5)
        result = fit(
            m,
            data,
            output,
            {"parent_checkpoint_sha256": PARENT_SHA, "diagnostic_only": True},
            recipe,
            device=torch.device("cpu"),
            resume=resume,
            stop_after=stop_after,
        )
        return m, result

    full, result = fresh(tmp_path / "full")
    assert result["completed"] == 6  # 17 studies exercises actual final-one-study normalization.
    _, stopped = fresh(tmp_path / "resumed", stop_after=1)
    assert stopped["completed"] == 1
    resumed, result = fresh(tmp_path / "resumed", resume=True)
    a = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed/checkpoint.pt", weights_only=True)
    assert eq(a, b)
    assert all(h["anatomical_views"] == 2 * h["studies"] for h in a["history"])
    assert [h["studies"] for h in a["history"]] == [8, 8, 1, 8, 8, 1]
    assert a["binding"]["attention_reference_sha256"] == "a" * 64
    assert a["binding"]["model_head_mode"] == "eval_with_grad_all_parameters_FP32_head"
    assert any(
        not torch.equal(a["model"][n], initial[n]) for n in initial if n.startswith("backbone.")
    )
    assert any(
        not torch.equal(a["model"][n], initial[n]) for n in initial if not n.startswith("backbone.")
    )
    case = data.case(0, list(range(16)), False)["images"].unsqueeze(0)
    with torch.no_grad():
        assert torch.equal(full(case), resumed(case))
        assert torch.equal(full(case), Parent.forward(full, case))
    assert set(a["model"]) == set(initial)
    for ep in (1, 2):
        assert eq(
            torch.load(tmp_path / f"full/epoch-checkpoints/epoch-{ep:03d}.pt", weights_only=True),
            torch.load(
                tmp_path / f"resumed/epoch-checkpoints/epoch-{ep:03d}.pt", weights_only=True
            ),
        )
    assert "if world != 1" in source


def test_reference_is_aligned_and_duplicates_and_views_are_preserved(fixture):
    model, data, scores, votes = fixture
    positions = [1] * 8 + [3] * 8
    case = data.case(1, positions, True)
    assert torch.equal(case["attention_reference"]["indices"], torch.tensor(positions))
    assert torch.equal(case["attention_reference"]["scores"], scores[1, 1])
    assert torch.equal(case["images"][0], case["images"][7])
    model.cached_attention_variant = "linear"
    ref = {n: t.unsqueeze(0) for n, t in case["attention_reference"].items()}
    with torch.no_grad():
        logits = model(case["images"].unsqueeze(0), torch.full((1, 16), 1 / 94).log(), ref)
    full = (scores[1, 1].softmax(0) * votes[1, 1]).sum(0) + model.clsb
    assert torch.allclose(logits[0], full, atol=2e-7, rtol=0)
    model.train()
    with pytest.raises(ValueError, match="deterministic"):
        model(case["images"].unsqueeze(0), torch.full((1, 16), 1 / 94).log(), ref)
    with pytest.raises(ValueError):
        ReferenceCases(
            data.original, data._proposals, list(reversed(data.uids)), scores, votes, "a" * 64
        )
