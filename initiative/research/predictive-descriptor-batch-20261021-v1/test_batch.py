import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


fast = load("CPU_batched_descriptors", HERE / "descriptor_batch.py")
f = load("original_descriptor_fixtures", ROOT / "tests/test_predictive_descriptor_cache.py")


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize(
    "shape", [(8, 8), (20, 24), (128, 128), (160, 320), (239, 241), (240, 240)]
)
def test_descriptor_exact_including_zero_one_and_discontinuous_patches(shape):
    with torch.random.fork_rng():
        torch.manual_seed(213)
        p = torch.rand(19, shape[0] * 2, shape[1] * 2)[:, ::2, ::2]
        p[0].zero_()
        p[1].fill_(1)
        p[2].fill_(0.5)
        original = torch.stack([f.inputs.descriptor64(x) for x in p])
        assert torch.equal(fast.descriptor_batch(p), original)
        assert torch.equal(fast.descriptor_batch(p.contiguous()), original)


@pytest.mark.parametrize("cap", [1, 7, 24])
def test_complete_variable_matrix_contexts_supports_and_source_unchanged(cap):
    groups = f.synthetic_groups(frames_per_group=18, matrices=((81, 99), (96, 112), (129, 132)))
    before = [g.frames.clone() for g in groups]
    rng = torch.get_rng_state().clone()
    original = f.cache.cache_descriptors_once(groups)
    optimized = fast.cache_descriptors_once(groups)
    for a, b in zip(original, optimized, strict=True):
        assert torch.equal(a.descriptors, b.descriptors)
    f.assert_contexts_equal(
        f.assemble(original, side=32, context_cap=cap),
        f.assemble(optimized, side=32, context_cap=cap),
    )
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(a.frames, b) for a, b in zip(groups, before, strict=True))
    assert all(g.descriptors is None for g in groups)


@pytest.mark.parametrize(
    "kind", ["alias", "dtype", "bounds", "metadata", "cross_group", "cached", "cap"]
)
def test_invalid_groups_still_rejected(kind):
    groups = f.synthetic_groups()
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
    elif kind == "cached":
        groups[0] = replace(first, descriptors=torch.zeros(64, 64))
    with pytest.raises(ValueError):
        fast.cache_descriptors_once(groups, max_descriptor_bytes=1 if kind == "cap" else 32 * 2**20)
