import numpy as np
import pytest
import torch

from rsna.competitive import licensed_cached_mil as m
from rsna.competitive.mil_sampling import readout_moments


def original():
    generator = torch.Generator().manual_seed(781)
    shapes = {
        "norm.weight": (1024,),
        "norm.bias": (1024,),
        "att.0.weight": (256, 1024),
        "att.0.bias": (256,),
        "att.3.weight": (12, 256),
        "att.3.bias": (12,),
        "clsW": (12, 1024),
        "clsb": (12,),
    }
    state = {k: torch.randn(shape, generator=generator) * 0.02 for k, shape in shapes.items()}
    state["norm.weight"].add_(1)
    return {
        "format": "licensed_parent_MIL_readout_v1",
        "parent_checkpoint_sha256": m.PARENT_SHA,
        "source_notebook_sha256": "493c7b0d44fe5244b02024448a93f5393cdb073781a068951badb3ffd53fcbde",
        "model": state,
    }


class Data:
    def __init__(self, label_role="original", count=9):
        self.uids = [f"synthetic-{i}" for i in range(count)]
        g = torch.Generator().manual_seed(123)
        self.x = torch.randn(count, 2, 94, 1024, generator=g)
        self.teacher = torch.randn(count, 2, 12, generator=g)
        y = (torch.rand(count, 12, generator=g) > 0.5).float()
        self.y = y if label_role == "original" else 1 - y

    def __len__(self):
        return len(self.uids)

    def case(self, i, selected, mirror):
        return {
            "features": self.x[i, int(mirror), selected],
            "parent_logits": self.teacher[i, int(mirror)],
            "labels": self.y[i],
            "mask": torch.ones(12, dtype=torch.bool),
        }


def binding():
    return {
        "parent_checkpoint_sha256": m.PARENT_SHA,
        "original_readout_sha256": m.READOUT_SHA,
        "protected_gold_confirmation_used": False,
        "synthetic_engineering_only": True,
    }


def equal(a, b):
    if isinstance(a, torch.Tensor):
        return isinstance(b, torch.Tensor) and torch.equal(a, b)
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    return type(a) is type(b) and a == b


def test_initial_readout_is_exact_original_FP32_math():
    torch.set_num_threads(1)
    parent = original()
    data = Data()
    model = m.Readout(parent).eval()
    for x in data.x[:2]:
        expected = readout_moments(parent, x)[3]
        assert np.array_equal(model(x).sigmoid().mean(0).detach().numpy(), expected)
    assert sum(p.numel() for p in model.parameters()) == 279832
    assert not any(n.startswith("backbone.") for n in model.state_dict())


def test_factorial_draws_share_views_and_do_not_change_global_RNG():
    before = torch.get_rng_state().clone()
    for e in range(3):
        results = [
            m.observations("synthetic-common", e, m.Recipe(role, k))
            for role in ("original", "released_S6")
            for k in (16, 94)
        ]
        assert len({view for _, view in results}) == 1
        assert results[0] == results[2] and results[1] == results[3]
        assert len(results[0][0]) == 16 and results[1][0] == list(range(94))
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize("role", ["original", "released_S6"])
@pytest.mark.parametrize("windows", [16, 94])
def test_full_stop1_resume_complete_state_and_prediction_bytes_equal(tmp_path, role, windows):
    parent = original()
    data = Data(role)
    recipe = m.Recipe(role, windows, epochs=2)
    full = tmp_path / "full"
    split = tmp_path / "split"
    out = m.fit(parent, data, full, binding(), recipe)
    assert (
        out["completed"] == 4
        and out["trainable_parameters"] == 279832
        and not out["encoder_trained"]
    )
    stopped = m.fit(parent, data, split, binding(), recipe, stop_after=1)
    assert stopped["completed"] == 1
    m.fit(parent, data, split, binding(), recipe, resume=True)
    a = torch.load(full / "checkpoint.pt", weights_only=True)
    b = torch.load(split / "checkpoint.pt", weights_only=True)
    assert equal(a, b)
    assert [r["studies"] for r in a["history"]] == [8, 1, 8, 1]
    p = m.predict(parent, a, list(data.x[:3]))
    q = m.predict(parent, b, list(data.x[:3]))
    assert p.numpy().tobytes() == q.numpy().tobytes()
    for epoch in (1, 2):
        assert equal(
            torch.load(full / f"epoch-checkpoints/epoch-{epoch:03d}.pt", weights_only=True),
            torch.load(split / f"epoch-checkpoints/epoch-{epoch:03d}.pt", weights_only=True),
        )
    assert any(not torch.equal(v, parent["model"][n]) for n, v in a["model"].items())


def test_source_values_actually_change_learning(tmp_path):
    parent = original()
    states = []
    for role in ("original", "released_S6"):
        m.fit(parent, Data(role), tmp_path / role, binding(), m.Recipe(role, 94, epochs=2))
        states.append(torch.load(tmp_path / role / "checkpoint.pt", weights_only=True))
    assert any(not torch.equal(v, states[1]["model"][n]) for n, v in states[0]["model"].items())
    assert equal(states[0]["rng"], states[1]["rng"])


@pytest.mark.parametrize(
    "change", ["recipe", "source", "readout", "protected", "duplicate", "immutable"]
)
def test_mixed_provenance_or_duplicate_dispatch_is_refused(tmp_path, change):
    parent = original()
    data = Data()
    recipe = m.Recipe("original", 16, epochs=2)
    out = tmp_path / "run"
    m.fit(parent, data, out, binding(), recipe, stop_after=1)
    b = binding()
    if change == "recipe":
        recipe = m.Recipe("released_S6", 16, epochs=2)
    if change == "source":
        b["source_file_sha"] = "different"
    if change == "readout":
        b["original_readout_sha256"] = "0" * 64
    if change == "protected":
        b["protected_gold_confirmation_used"] = True
    if change == "duplicate":
        data.uids[-1] = data.uids[0]
    with pytest.raises(ValueError):
        m.fit(parent, data, out, b, recipe, resume=change != "immutable")


@pytest.mark.parametrize(
    "kwargs",
    [{"windows": 12}, {"label_role": "unknown"}, {"source_weight": float("nan")}, {"epochs": True}],
)
def test_unaccepted_recipe_is_refused(kwargs):
    args = {"label_role": "original", "windows": 94, **kwargs}
    with pytest.raises(ValueError):
        m.Recipe(**args)


def test_unbound_source_readout_cannot_initialize():
    value = original()
    value["source_notebook_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        m.Readout(value)


def test_prediction_refuses_nonfinite_cached_features():
    value = original()
    head = m.Readout(value)
    features = Data().x[0].clone()
    features[0, 0, 0] = float("nan")
    with pytest.raises(ValueError):
        m.predict(value, {"model": head.state_dict()}, [features])
