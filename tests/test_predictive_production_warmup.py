"""Production-horizon scheduling and neutral-density initialization, on CPU only."""

import copy
import random
from dataclasses import replace

import numpy as np
import pytest
import torch
from torch import nn

from rsna.competitive.predictive.density import Supports
from rsna.competitive.predictive.model import matched_modules
from rsna.competitive.predictive.train import Recipe, fit, restore_rng, rng_state


@pytest.fixture(autouse=True)
def one_cpu_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, np.ndarray):
        assert np.array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert len(a) == len(b)
        for left, right in zip(a, b):
            equal(left, right)
    else:
        assert a == b


def modules():
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(12, 4))
    return matched_modules(encoder, 4, hidden=4, classes=2, descriptor_width=3)


class Data:
    def __init__(self):
        generator = torch.Generator().manual_seed(59)
        images = torch.rand(2, 3, 2, 2, generator=generator)
        metadata = torch.rand(4, 8, generator=generator)
        targets = torch.rand(4, 3, generator=generator) / 20
        support = Supports(
            torch.tensor([0, 1, 2, 3, 4, 8]),
            torch.tensor([0, 1, 2, 3, 0, 1, 2, 3]),
            torch.zeros(2, 5),
            4,
        )
        self.contexts = [
            dict(
                images=images, token_metadata=metadata[:2], region_metadata=metadata,
                targets=targets, observed=torch.tensor([True, False, True, False]),
                supports=support,
            ),
            dict(
                images=1 - images, token_metadata=metadata[:2], region_metadata=metadata,
                targets=targets, observed=torch.tensor([False, True, False, True]),
                supports=support,
            ),
        ]

    def __len__(self):
        return 3

    def case(self, index, epoch):
        random.random()
        np.random.rand()
        torch.rand(1)
        return {
            "contexts": self.contexts,
            "labels": torch.tensor([[float(index % 2), 0.25]]),
            "mask": torch.ones(1, 2, dtype=torch.bool),
        }


def test_zero_density_heads_keep_shared_state_and_global_initialization_rng():
    torch.manual_seed(31)
    encoder = nn.Sequential(nn.Flatten(), nn.Linear(12, 4))
    before = copy.deepcopy(rng_state())
    B, C, receipt = matched_modules(
        encoder, 4, hidden=4, classes=2, descriptor_width=3, seed=23
    )
    equal(rng_state(), before)
    B2, C0, _ = matched_modules(
        encoder, 4, hidden=4, classes=2, descriptor_width=3, seed=23
    )
    equal(B.state_dict(), B2.state_dict())
    equal(C.state_dict(), C0.state_dict())
    assert receipt["same_shared_tensor_values"]
    for layer in (B.reconstruct, C.negative, C.shift):
        assert torch.count_nonzero(layer.weight) == torch.count_nonzero(layer.bias) == 0
    context = Data().contexts[0]
    C.eval()
    with torch.inference_mode():
        output = C(**context)
    torch.testing.assert_close(output["evidence"].log_bayes_factor, torch.zeros(1, 2))
    assert torch.count_nonzero(C.prior.weight) > 0  # The shared context head is unchanged.


@pytest.mark.parametrize("arm", ["B", "C", "C0"])
def test_first24_production_warmup_and_complete_recovery(tmp_path, arm):
    B, C, _ = modules()
    base = B if arm == "B" else C
    recipe = Recipe(
        arm, epochs=12, effective_batch=2, encoder_lr=3e-5, head_lr=3e-5,
        head_only_updates=1, schedule_updates=2916, density_weight=0 if arm == "C0" else 0.1,
    )
    initial = copy.deepcopy(rng_state())
    binding = {"synthetic_engineering_only": True}
    full = fit(copy.deepcopy(base), Data(), recipe, binding=binding, output=tmp_path / "full")
    restore_rng(initial)
    fit(
        copy.deepcopy(base), Data(), recipe, binding=binding,
        output=tmp_path / "stopped", stop_after=7,
    )
    resumed = fit(
        copy.deepcopy(base), Data(), recipe, binding=binding, output=tmp_path / "resumed",
        resume=tmp_path / "stopped/stopped.pt",
    )
    equal(full, resumed)
    assert full["step"] == 24
    assert full["recipe"]["schedule_updates"] == 2916
    for index, record in enumerate(full["history"], 1):
        expected = 3e-5 * index / (0.05 * 2916)
        assert record["LR"] == pytest.approx([expected, expected])
        assert 0 < record["LR"][0] < 3e-5
    assert full["history"][0]["LR"][0] == pytest.approx(2.05761316872428e-7)
    with pytest.raises(ValueError, match="binding differs"):
        fit(
            copy.deepcopy(base), Data(), replace(recipe, schedule_updates=2917),
            binding=binding, output=tmp_path / "changed",
            resume=tmp_path / "stopped/stopped.pt",
        )


@pytest.mark.parametrize("value", [0, -1, True, 2.5, "2916"])
def test_schedule_requires_positive_integer(value):
    with pytest.raises(ValueError):
        Recipe("C", schedule_updates=value)


def test_shorter_schedule_refused_before_output_and_default_unchanged(tmp_path):
    B, _, _ = modules()
    binding = {"synthetic_engineering_only": True}
    recipe = Recipe("B", epochs=2, effective_batch=2)
    with pytest.raises(ValueError, match="cover every"):
        fit(
            copy.deepcopy(B), Data(), replace(recipe, schedule_updates=3),
            binding=binding, output=tmp_path / "invalid",
        )
    assert not (tmp_path / "invalid").exists()
    initial = copy.deepcopy(rng_state())
    implicit = fit(copy.deepcopy(B), Data(), recipe, binding=binding, output=tmp_path / "implicit")
    restore_rng(initial)
    explicit = fit(
        copy.deepcopy(B), Data(), replace(recipe, schedule_updates=4),
        binding=binding, output=tmp_path / "explicit",
    )
    explicit["recipe"]["schedule_updates"] = None
    equal(implicit, explicit)
