"""Exact serving optimization on synthetic variable-matrix MRI source groups."""

import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from rsna.competitive.predictive import inputs
from rsna.competitive.predictive.cases import RawGroup, assemble
from rsna.competitive.predictive.infer import predict
from rsna.competitive.predictive.model import matched_modules

ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "initiative/research/predictive-final-serving-20261020-v1/descriptor_cache.py"
spec = importlib.util.spec_from_file_location("predictive_descriptor_cache", HELPER)
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)


def synthetic_groups(*, frames_per_group=4, matrices=((80, 96), (96, 112), (128, 128))):
    """Unequal matrices, acquisitions, nonconsecutive aliases and repeated frames."""
    result = []
    for group_index, (height, width) in enumerate(matrices):
        y = torch.linspace(-1, 1, height)[:, None]
        x = torch.linspace(-1, 1, width)[None, :]
        frames = [
            (
                0.5
                + 0.18 * torch.cos((x * x + y * y) * (7 + i))
                + 0.13 * torch.sin(x * 11 + i / 3)
                + 0.09 * torch.cos(y * 13 - group_index)
            )
            for i in range(frames_per_group)
        ]
        # Insert a real duplicate before the last frame; first occurrence order
        # differs from numeric alias order and must not be silently sorted.
        frames.insert(-1, frames[0].clone())
        aliases = [1000 * (group_index + 1) + 17 - i for i in range(frames_per_group)]
        aliases.insert(-1, aliases[0])
        ordinal = list(range(frames_per_group))
        ordinal.insert(-1, ordinal[0])
        result.append(
            RawGroup(
                torch.stack(frames),
                torch.tensor(aliases),
                torch.full((len(frames),), group_index, dtype=torch.int64),
                torch.tensor(ordinal),
            )
        )
    return result


def assert_contexts_equal(left, right):
    assert len(left) == len(right) == 2
    for actual, expected in zip(left, right):
        assert set(actual) == set(expected)
        for name in ("images", "token_metadata", "region_metadata", "targets", "observed"):
            assert torch.equal(actual[name], expected[name]), name
        a, b = actual["supports"], expected["supports"]
        assert a.region_count == b.region_count
        for name in ("indptr", "regions", "log_prior"):
            assert torch.equal(getattr(a, name), getattr(b, name)), name


@pytest.fixture(autouse=True)
def one_cpu_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


@pytest.mark.parametrize("context_cap", [1, 7, 24])
def test_descriptor_once_bitwise_complete_masks_supports_and_probabilities(
    monkeypatch, context_cap
):
    groups = synthetic_groups()
    raw_before = [group.frames.clone() for group in groups]
    rng_before = torch.get_rng_state().clone()
    original_descriptor = inputs.descriptor64
    calls = []

    def counted(patch):
        calls.append(tuple(patch.shape))
        return original_descriptor(patch)

    monkeypatch.setattr(inputs, "descriptor64", counted)
    original = assemble(groups, side=32, halo=8, context_cap=context_cap)
    assert len(calls) == 2 * 3 * 4 * 16
    calls.clear()
    cached = cache.cache_descriptors_once(groups)
    optimized = assemble(cached, side=32, halo=8, context_cap=context_cap)
    assert len(calls) == 3 * 4 * 16
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert_contexts_equal(optimized, original)
    assert torch.equal(original[0]["observed"], ~original[1]["observed"])
    assert all(context["supports"].region_count == 3 * 4 * 16 for context in optimized)
    for raw, prepared, before in zip(groups, cached, raw_before):
        assert raw.descriptors is None
        assert torch.equal(raw.frames, before)
        for name in ("frames", "aliases", "acquisition", "ordinal"):
            assert getattr(raw, name) is getattr(prepared, name)
        assert prepared.descriptors.numel() * prepared.descriptors.element_size() == 4 * 4096

    with torch.random.fork_rng():
        torch.manual_seed(613)
        encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        baseline, conditional, _ = matched_modules(encoder, 8, hidden=16)
        # A nonzero conditional decoder prevents a false parity pass in which
        # only shared context logits contribute to the C/C0 prediction.
        with torch.no_grad():
            conditional.negative.weight.normal_(0, 0.007)
            conditional.negative.bias.normal_(0, 0.003)
            conditional.shift.weight.normal_(0, 0.002)
            conditional.shift.bias.normal_(0, 0.001)
        for model in (baseline, conditional):
            assert torch.equal(predict(model, original), predict(model, optimized))


def test_cap_is_checked_before_any_descriptor_allocation_or_computation(monkeypatch):
    def forbidden(_):
        raise AssertionError("Descriptor work must not begin before complete byte admission")

    monkeypatch.setattr(inputs, "descriptor64", forbidden)
    with pytest.raises(ValueError, match="exceeds cap"):
        cache.cache_descriptors_once(synthetic_groups(), max_descriptor_bytes=3 * 4 * 4096 - 1)


@pytest.mark.parametrize("kind", ["alias", "dtype", "bounds", "metadata", "cross_group", "cached"])
def test_invalid_or_unbound_source_groups_are_rejected(kind):
    groups = synthetic_groups()
    first = groups[0]
    if kind == "alias":
        changed = first.frames.clone()
        changed[-2, 0, 0] += 0.1
        groups[0] = replace(first, frames=changed)
    elif kind == "dtype":
        groups[0] = replace(first, frames=first.frames.double())
    elif kind == "bounds":
        changed = first.frames.clone()
        changed[0, 0, 0] = 1.01
        groups[0] = replace(first, frames=changed)
    elif kind == "metadata":
        groups[0] = replace(first, ordinal=first.ordinal[:-1])
    elif kind == "cross_group":
        changed = groups[1].aliases.clone()
        changed[0] = first.aliases[0]
        groups[1] = replace(groups[1], aliases=changed)
    else:
        groups[0] = replace(first, descriptors=torch.zeros(64, 64))
    with pytest.raises(ValueError):
        cache.cache_descriptors_once(groups)
