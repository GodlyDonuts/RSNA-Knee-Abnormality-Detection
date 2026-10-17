from dataclasses import replace

import pytest
import torch
from torch import nn

from rsna.competitive.checkpoint import capture_rng
from rsna.competitive.licensed_full_bag import CheckpointedEncode
from rsna.competitive.licensed_importance import (
    UNIFORM_FLOOR,
    ImportanceReadout,
    ProposalCases,
    Recipe,
    sample,
    validate_proposal,
)
from rsna.competitive.licensed_importance_train import fit
from rsna.competitive.licensed_parent_train import PARENT_SHA, train_with_frozen_bn


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for name in a:
            equal(a[name], b[name])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    else:
        assert a == b


def proposal():
    influence = torch.arange(1, 95, dtype=torch.float64)
    return UNIFORM_FLOOR + 0.75 * influence / influence.sum()


class OriginalCases:
    def __init__(self, n=15):
        self.uids = [f"synthetic-MRI-{i}" for i in range(n)]
        g = torch.Generator().manual_seed(17)
        self.images = torch.randn(n, 94, 3, 4, 4, generator=g)
        self.labels = torch.rand(n, 12, generator=g)
        self.teacher = torch.randn(n, 2, 12, generator=g)

    def __len__(self):
        return len(self.uids)

    def case(self, index, selected, mirror):
        assert len(selected) == len(set(selected))  # Original reader contract stays strict.
        images = self.images[index, selected]
        return {
            "images": images.flip(-1) if mirror else images,
            "labels": self.labels[index],
            "mask": torch.ones(12, dtype=torch.bool),
            "parent_logits": self.teacher[index, int(mirror)],
        }


def cases(n=15):
    return ProposalCases(OriginalCases(n), proposal().repeat(n, 1))


class OriginalModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 4, 1), nn.BatchNorm2d(4), nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        self.norm = nn.LayerNorm(4)
        self.att = nn.Sequential(nn.Linear(4, 3), nn.Tanh(), nn.Dropout(0.2), nn.Linear(3, 12))
        self.clsW = nn.Parameter(torch.randn(12, 4))
        self.clsb = nn.Parameter(torch.zeros(12))

    def encode(self, x):
        # Match the actual recomputation chunk shape when testing gradient/RNG
        # identity. Different convolution batch shapes need not be bit-identical.
        values = [self.backbone(chunk) for chunk in x.flatten(0, 1).split(8)]
        return torch.cat(values).view(*x.shape[:2], 4)

    def forward(self, x):
        h = self.norm(self.encode(x))
        pooled = torch.einsum("bkn,bkf->bnf", self.att(h).softmax(1), h)
        return (pooled * self.clsW).sum(-1) + self.clsb


class Model(ImportanceReadout, CheckpointedEncode, OriginalModel):
    pass


@pytest.mark.parametrize("kind", ["size", "negative", "mass", "dtype", "grad"])
def test_bad_proposals_refused_before_sampling(kind):
    q = proposal()
    if kind == "size":
        q = q[:-1]
    elif kind == "negative":
        q[0] = -1
    elif kind == "mass":
        q *= 0.99
    elif kind == "dtype":
        q = q.float()
    else:
        q.requires_grad_(True)
    with pytest.raises(ValueError):
        validate_proposal(q)


def test_common_draws_and_view_are_stateless_but_change_between_epochs():
    before = capture_rng(torch.device("cpu"))
    uniform = Recipe()
    adaptive = replace(uniform, role="influence")
    a = sample("synthetic", uniform, 0, proposal())
    b = sample("synthetic", adaptive, 0, proposal())
    equal(before, capture_rng(torch.device("cpu")))
    equal(a, sample("synthetic", uniform, 0, proposal()))
    assert a[1] == b[1] and a[0] != b[0]
    assert a[0] != sample("synthetic", uniform, 1, proposal())[0]
    assert len(a[0]) == len(b[0]) == 16
    assert torch.allclose(b[2].double().exp(), proposal()[b[0]], atol=1e-8, rtol=1e-6)
    # A constant teacher falls back to uniform, producing the very same IID bag.
    equal(a, sample("synthetic", adaptive, 0, torch.full((94,), 1 / 94, dtype=torch.float64)))


def test_original_unique_reader_reused_with_duplicate_draw_order_and_exact_targets():
    data = cases()
    selected = [2, 5, 2, 5] * 4
    row = data.case(0, selected, True)
    assert torch.equal(row["images"], data.original.images[0, selected].flip(-1))
    assert torch.equal(row["labels"], data.original.labels[0])
    assert torch.equal(row["parent_logits"], data.original.teacher[0, 1])
    q = data.proposal(0)
    q[0] = 0
    assert data.proposal(0)[0] > 0
    with pytest.raises(ValueError):
        data.case(0, selected[:-1], False)


def test_inverse_proposal_matches_independent_scalar_weighted_readout_with_duplicates():
    torch.manual_seed(25)
    model = Model().eval()
    images = torch.randn(1, 16, 3, 4, 4)
    logq = torch.linspace(-5.5, -1.0, 16).reshape(1, 16)
    h = model.norm(model.encode(images))
    scores = model.att(h).double()
    weights = (scores - logq.double().unsqueeze(-1)).exp()
    values = torch.einsum("bkd,td->bkt", h.double(), model.clsW.double())
    expected = (weights * values).sum(1) / weights.sum(1) + model.clsb.double()
    actual = model(images, logq)
    assert torch.allclose(actual.double(), expected, atol=3e-7, rtol=1e-6)


def test_full_inference_keeps_original_graph_and_uniform_training_gradients():
    torch.manual_seed(71)
    original = OriginalModel()
    model = Model()
    model.load_state_dict(original.state_dict(), strict=True)
    assert original.state_dict().keys() == model.state_dict().keys()
    x = torch.randn(1, 16, 3, 4, 4)
    original.eval()
    model.eval()
    assert torch.equal(original(x), model(x))
    train_with_frozen_bn(original)
    train_with_frozen_bn(model)
    torch.manual_seed(99)
    expected = original(x)
    expected.square().mean().backward()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(99)
    actual = model(x, torch.full((1, 16), -4.0))
    actual.square().mean().backward()
    assert torch.equal(actual, expected)
    assert torch.equal(torch.get_rng_state(), rng)
    for p, q in zip(original.parameters(), model.parameters(), strict=True):
        assert torch.allclose(p.grad, q.grad, atol=1e-6, rtol=1e-5)


@pytest.mark.parametrize("role", ["uniform", "influence"])
def test_complete_CPU_optimizer_recovery_and_final_odd_batch(role, tmp_path):
    torch.set_num_threads(1)
    recipe = Recipe(role=role, epochs=2)
    data = cases()
    binding = {"parent_checkpoint_sha256": PARENT_SHA, "diagnostic_only": True}
    torch.manual_seed(337)
    model = Model()
    initial = {n: v.clone() for n, v in model.state_dict().items()}
    fit(model, data, tmp_path / "full", binding, recipe, device=torch.device("cpu"))
    torch.manual_seed(337)
    stopped = Model()
    fit(
        stopped,
        data,
        tmp_path / "resume",
        binding,
        recipe,
        device=torch.device("cpu"),
        stop_after=1,
    )
    torch.manual_seed(772)
    fit(
        Model(), data, tmp_path / "resume", binding, recipe, device=torch.device("cpu"), resume=True
    )
    left = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    right = torch.load(tmp_path / "resume/checkpoint.pt", weights_only=True)
    equal(left, right)
    assert [r["studies"] for r in left["history"]] == [8, 7, 8, 7]
    assert left["binding"]["teacher_proposal_sha256"] == data.proposal_sha256
    assert not torch.equal(initial["backbone.0.weight"], left["model"]["backbone.0.weight"])
    for name in ("running_mean", "running_var", "num_batches_tracked"):
        assert torch.equal(initial["backbone.1." + name], left["model"]["backbone.1." + name])
    with pytest.raises(ValueError, match="Exact source"):
        fit(
            Model(),
            data,
            tmp_path / "resume",
            binding,
            replace(recipe, role="influence" if role == "uniform" else "uniform"),
            device=torch.device("cpu"),
            resume=True,
        )


def test_new_graph_refuses_learnable_or_nonfinite_sampling_weights():
    model = Model()
    x = torch.randn(1, 16, 3, 4, 4)
    for q in (torch.full((1, 16), -4.0, requires_grad=True), torch.full((1, 16), float("nan"))):
        with pytest.raises(ValueError):
            model(x, q)
