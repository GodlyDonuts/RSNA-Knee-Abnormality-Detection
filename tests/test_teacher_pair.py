import copy
import hashlib
import importlib.util
import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

from rsna.competitive.checkpoint import capture_rng, restore_rng
from rsna.competitive.licensed_importance import ProposalCases, sample
from rsna.competitive.licensed_parent_train import PARENT_SHA
from rsna.competitive.released_context import anchored_loss
from rsna.competitive.teacher_pair import Recipe, coupled_batch, ranking_loss


def inputs():
    g = torch.Generator().manual_seed(7)
    z = torch.randn(6, 12, generator=g).requires_grad_()
    t = torch.randn(6, 12, generator=g).requires_grad_()
    y = torch.rand(3, 12, generator=g).repeat_interleave(2, 0).requires_grad_()
    m = (torch.rand(3, 12, generator=g) > 0.3).repeat_interleave(2, 0)
    return z, t, y, m


def test_independent_scalar_pair_loop_and_source_BCE_is_unchanged():
    z, t, y, m = inputs()
    actual = ranking_loss(z, t, y, m)
    base = anchored_loss(z, t, y, m)
    q = t.detach().sigmoid().reshape(3, 2, 12).mean(1)
    s = torch.logit(z.sigmoid().reshape(3, 2, 12).mean(1))
    a = 1 - 0.25 * m.float()
    replacement = []
    for k in range(12):
        weight, total = 0.0, torch.tensor(0.0)
        for i in range(3):
            for j in range(3):
                w = max(0.0, float(q[i, k] - q[j, k]))
                weight += w
                total = total + w * F.softplus(-(s[i, k] - s[j, k]))
        point = (
            a[:, k]
            * F.binary_cross_entropy_with_logits(
                z[:, k], t[:, k].detach().sigmoid(), reduction="none"
            )
        ).mean()
        replacement.append(a[:, k].mean() * total / weight - point if weight else point * 0)
    assert torch.allclose(actual, base + 0.5 * torch.stack(replacement).mean(), atol=1e-7)
    actual.backward()
    assert t.grad is None and y.grad is None
    assert z.grad is not None and torch.isfinite(z.grad).all()
    # Source-label dependence is exactly the original masked source BCE dependence.
    changed = y.detach().clone()
    changed[m] = 1 - changed[m]
    difference = ranking_loss(z.detach(), t.detach(), changed, m) - actual.detach()
    expected = anchored_loss(z.detach(), t.detach(), changed, m) - base.detach()
    assert torch.allclose(difference, expected, atol=1e-7)


@pytest.mark.parametrize("kind", ["fraction_zero", "singleton", "teacher_ties", "masked_nan"])
def test_exact_old_loss_when_no_teacher_rank_replacement(kind):
    z, t, y, m = inputs()
    fraction = 0.5
    if kind == "fraction_zero":
        fraction = 0
    elif kind == "singleton":
        z, t, y, m = (x[:2] for x in (z, t, y, m))
    elif kind == "teacher_ties":
        t = torch.zeros_like(t)
    else:
        t = torch.zeros_like(t)
        y = y.detach().clone()
        y[~m] = float("nan")
    a, b = z.detach().clone().requires_grad_(), z.detach().clone().requires_grad_()
    first = ranking_loss(a, t, y, m, fraction=fraction)
    second = anchored_loss(b, t, y, m)
    assert torch.equal(first, second)
    first.backward()
    second.backward()
    assert torch.equal(a.grad, b.grad)


def test_all_source_masked_still_has_dense_teacher_pair_gradients():
    z, t, y, m = inputs()
    m[:] = False
    y = torch.full_like(y, float("nan"))
    loss = ranking_loss(z, t, y, m)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all() and z.grad.abs().sum() > 0


def test_pair_order_uses_probability_TTA_not_mean_logits_and_excludes_same_study():
    z = torch.tensor([[8.0], [-2.0], [0.0], [0.0]], requires_grad=True)
    t = torch.tensor([[2.0], [2.0], [-2.0], [-2.0]])
    y = torch.zeros_like(z)
    m = torch.zeros_like(z, dtype=torch.bool)
    pair = F.softplus(-(torch.logit(z[:2].sigmoid().mean()) - torch.logit(z[2:].sigmoid().mean())))
    expected = 0.5 * anchored_loss(z, t, y, m) + 0.5 * pair
    actual = ranking_loss(z, t, y, m)
    assert torch.allclose(actual, expected, atol=1e-7)
    assert not torch.allclose(pair, F.softplus(-(z[:2].mean() - z[2:].mean())))


def test_study_and_within_study_view_permutation_invariance_and_extreme_finiteness():
    z, t, y, m = inputs()
    a = ranking_loss(z, t, y, m)
    order = [5, 4, 1, 0, 3, 2]
    assert torch.allclose(a, ranking_loss(z[order], t[order], y[order], m[order]), atol=1e-7)
    z = (z.detach() * 80).requires_grad_()
    t = t.detach() * 40
    loss = ranking_loss(z, t, y, m)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(z.grad).all()


@pytest.mark.parametrize("kind", ["views", "labels", "mask", "fraction", "dtype"])
def test_refuse_invalid_pair_semantics(kind):
    z, t, y, m = inputs()
    fraction = 0.5
    if kind == "views":
        z, t, y, m = (v[:3] for v in (z, t, y, m))
    elif kind == "labels":
        y = y.detach().clone()
        m[:] = True
        y[0, 0] = 0.7
        y[1, 0] = 0.2
    elif kind == "mask":
        m[0, 0] = True
        m[1, 0] = False
    elif kind == "fraction":
        fraction = float("nan")
    else:
        z = z.double()
    with pytest.raises(ValueError):
        ranking_loss(z, t, y, m, fraction=fraction)


class OriginalCases:
    def __init__(self, n=17):
        self.uids = [f"synthetic-study-{i}" for i in range(n)]
        self.calls = []

    def __len__(self):
        return len(self.uids)

    def case(self, index, positions, mirror):
        self.calls.append((index, list(positions), mirror))
        pixels = (
            (
                torch.tensor(positions).float()[:, None, None, None] / 94
                + index / 100
                + int(mirror) / 10
            )
            .expand(-1, 3, 2, 2)
            .contiguous()
        )
        y = torch.arange(12).float().remainder(2)
        mask = torch.arange(12).remainder(3) != 0
        y[~mask] = float("nan")
        return {
            "images": pixels,
            "labels": y,
            "mask": mask,
            "parent_logits": torch.linspace(-1, 1, 12) + index / 20 + int(mirror) / 30,
        }


def data(n=17):
    original = OriginalCases(n)
    return ProposalCases(original, torch.full((n, 94), 1 / 94, dtype=torch.float64))


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(nn.Linear(3, 9), nn.Dropout(0.2), nn.Tanh())
        self.head = nn.Linear(9, 12)

    def forward(self, images, logq):
        x = images.mean((3, 4)).mean(1)
        return self.head(self.backbone(x)) + logq.mean() * 0


def seed():
    random.seed(17)
    np.random.seed(17)
    torch.manual_seed(17)


def same(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            same(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b):
            same(x, y)
    else:
        assert a == b


def engine(tmp_path):
    path = Path("initiative/research/teacher-pair-full-encoder-20261019-v1/compiler.py.txt")
    spec = importlib.util.spec_from_loader(
        "own_teacher_pair_compiler", SourceFileLoader("own_teacher_pair_compiler", str(path))
    )
    compiler = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(compiler)
    raw = compiler.build(Path("rsna/competitive/licensed_importance_train.py").read_text())
    out = tmp_path / "engine.py"
    out.write_text(raw)
    spec = importlib.util.spec_from_file_location("distinct_teacher_pair_engine", out)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_coupled_pointwise_matches_independent_per_view_loss_gradient_and_draws():
    seed()
    first = Model()
    second = copy.deepcopy(first)
    d = data(8)
    r = Recipe(epochs=2)
    rng = capture_rng(torch.device("cpu"))
    joint, record = coupled_batch(first, d, list(range(8)), r, 0, torch.device("cpu"))
    joint.backward()
    assert record["anatomical_views"] == 16 and record["studies"] == 8
    state = capture_rng(torch.device("cpu"))
    restore_rng(rng, torch.device("cpu"))
    values = []
    for i in range(8):
        positions, mirror, q = sample(d.uids[i], r, 0, d.proposal(i))
        for view in (mirror, not mirror):
            case = d.case(i, positions, view)
            logits = second(case["images"][None], q[None]).float()
            loss = (
                anchored_loss(
                    logits, case["parent_logits"][None], case["labels"][None], case["mask"][None]
                )
                / 16
            )
            values.append(loss.detach())
            loss.backward()
    assert torch.allclose(joint.detach(), torch.stack(values).sum(), atol=1e-7)
    same(capture_rng(torch.device("cpu")), state)
    for a, b in zip(first.parameters(), second.parameters(), strict=True):
        assert torch.allclose(a.grad, b.grad, atol=2e-7, rtol=2e-6)
    for i in range(0, 16, 2):
        a, b = d.original.calls[i : i + 2]
        assert a[0] == b[0] and a[1] == b[1] and a[2] != b[2]


@pytest.mark.parametrize("objective", ["pointwise", "teacher_rank"])
def test_actual_new_engine_exact_full_stop_resume_complete_states_and_epoch_snapshots(
    tmp_path, objective
):
    runner = engine(tmp_path)
    r = Recipe(epochs=2, objective=objective)
    d = data(17)
    binding = {"parent_checkpoint_sha256": PARENT_SHA, "diagnostic_only": True, "synthetic": True}
    seed()
    full = Model()
    runner.fit(full, d, tmp_path / "full", binding, r, device=torch.device("cpu"))
    seed()
    stopped = Model()
    runner.fit(
        stopped, d, tmp_path / "resume", binding, r, device=torch.device("cpu"), stop_after=1
    )
    seed()
    resumed = Model()
    runner.fit(resumed, d, tmp_path / "resume", binding, r, device=torch.device("cpu"), resume=True)
    a = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    b = torch.load(tmp_path / "resume/checkpoint.pt", weights_only=True)
    same(a, b)
    assert a["completed"] == 6 and a["binding"]["anatomical_views_per_study"] == 2
    assert all(h["anatomical_views"] == 2 * h["studies"] for h in a["history"])
    assert [h["studies"] for h in a["history"]] == [8, 8, 1, 8, 8, 1]
    for e in (1, 2):
        same(
            torch.load(tmp_path / f"full/epoch-checkpoints/epoch-{e:03d}.pt", weights_only=True),
            torch.load(tmp_path / f"resume/epoch-checkpoints/epoch-{e:03d}.pt", weights_only=True),
        )
    x = torch.ones(1, 16, 3, 2, 2)
    q = torch.zeros(1, 16)
    full.eval()
    resumed.eval()
    assert full(x, q).detach().numpy().tobytes() == resumed(x, q).detach().numpy().tobytes()
    with pytest.raises(ValueError):
        runner.fit(
            Model(),
            d,
            tmp_path / "resume",
            binding,
            Recipe(epochs=2, objective="teacher_rank" if objective == "pointwise" else "pointwise"),
            device=torch.device("cpu"),
            resume=True,
        )


def test_no_role_sampling_or_initialization_difference_but_ranking_updates_differ(tmp_path):
    runner = engine(tmp_path)
    d = data(8)
    binding = {"parent_checkpoint_sha256": PARENT_SHA, "diagnostic_only": True}
    results = []
    for objective in ("pointwise", "teacher_rank"):
        seed()
        model = Model()
        initial = hashlib.sha256(
            b"".join(v.detach().numpy().tobytes() for v in model.state_dict().values())
        ).hexdigest()
        runner.fit(
            model,
            d,
            tmp_path / objective,
            binding,
            Recipe(epochs=2, objective=objective),
            device=torch.device("cpu"),
        )
        results.append((initial, model))
    assert results[0][0] == results[1][0]
    assert any(
        not torch.equal(a, b)
        for a, b in zip(results[0][1].parameters(), results[1][1].parameters(), strict=True)
    )
