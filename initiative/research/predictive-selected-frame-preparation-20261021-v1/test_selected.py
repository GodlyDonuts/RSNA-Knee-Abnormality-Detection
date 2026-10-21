import importlib.util
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


fast = load("selected_frame_candidate", HERE / "selected_frames.py")
f = load("original_descriptor_fixtures", ROOT / "tests/test_predictive_descriptor_cache.py")


@pytest.fixture(autouse=True)
def one_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.mark.parametrize("context_cap", [1, 7, 24, 94])
@pytest.mark.parametrize("halo", [0, 3, 8])
def test_all_original_context_bytes_with_variable_shapes_and_duplicate_metadata(context_cap, halo):
    groups = f.synthetic_groups(frames_per_group=18, matrices=((81, 99), (96, 112), (129, 132)))
    # Alias occurrence crosses acquisition with a different parity; masking
    # must still use the representative. Other canonical frames use both acqs.
    for group_id, g in enumerate(groups):
        acq = torch.arange(len(g.frames)) % 2 + group_id * 2
        order = g.ordinal.clone()
        acq[-2] = group_id * 2 + 1
        order[-2] = 100
        groups[group_id] = replace(g, acquisition=acq, ordinal=order)
    groups = f.cache.cache_descriptors_once(groups)
    source = [g.frames.clone() for g in groups]
    desc = [g.descriptors.clone() for g in groups]
    rng = torch.get_rng_state().clone()
    original = f.assemble(groups, side=37, halo=halo, context_cap=context_cap)
    candidate = fast.assemble(groups, side=37, halo=halo, context_cap=context_cap)
    f.assert_contexts_equal(original, candidate)
    assert all(x["supports"].region_count == 18 * 3 * 16 for x in candidate)
    assert torch.equal(rng, torch.get_rng_state())
    assert all(torch.equal(g.frames, x) for g, x in zip(groups, source, strict=True))
    assert all(torch.equal(g.descriptors, x) for g, x in zip(groups, desc, strict=True))


def test_original_nonzero_B_C_probabilities_equal_and_no_shared_context_mutation():
    groups = f.cache.cache_descriptors_once(f.synthetic_groups())
    expected = f.assemble(groups, side=32, context_cap=7)
    actual = fast.assemble(groups, side=32, context_cap=7)
    with torch.random.fork_rng():
        torch.manual_seed(614)
        encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
        baseline, conditional, _ = f.matched_modules(encoder, 8, hidden=16)
        with torch.no_grad():
            conditional.negative.weight.normal_(0, 0.007)
            conditional.negative.bias.normal_(0, 0.003)
            conditional.shift.weight.normal_(0, 0.002)
            conditional.shift.bias.normal_(0, 0.001)
        for model in (baseline, conditional):
            assert torch.equal(f.predict(model, expected), f.predict(model, actual))
    for key in ("images", "token_metadata", "region_metadata", "targets", "observed"):
        assert actual[0][key].data_ptr() != actual[1][key].data_ptr()
    for key in ("indptr", "regions", "log_prior"):
        assert (
            getattr(actual[0]["supports"], key).data_ptr()
            != getattr(actual[1]["supports"], key).data_ptr()
        )


@pytest.mark.parametrize(
    "kind",
    [
        "uncached",
        "nan_unused",
        "invalid_unused",
        "alias",
        "metadata",
        "cross_group",
        "cached_shape",
        "cached_nan",
        "ambiguous_ordinal",
        "halo",
    ],
)
def test_complete_original_admission_including_unused_source_still_required(kind):
    groups = f.cache.cache_descriptors_once(f.synthetic_groups(frames_per_group=9))
    first = groups[0]
    if kind == "uncached":
        groups[0] = replace(first, descriptors=None)
    elif kind in ("nan_unused", "invalid_unused", "alias"):
        changed = first.frames.clone()
        if kind == "alias":
            changed[-2, 0, 0] += 0.1
        else:
            changed[4, 0, 0] = float("nan") if kind == "nan_unused" else 1.01
        groups[0] = replace(first, frames=changed)
    elif kind == "metadata":
        groups[0] = replace(first, ordinal=first.ordinal[:-1])
    elif kind == "cross_group":
        changed = groups[1].aliases.clone()
        changed[0] = first.aliases[0]
        groups[1] = replace(groups[1], aliases=changed)
    elif kind == "cached_shape":
        groups[0] = replace(first, descriptors=first.descriptors[:-1])
    elif kind == "cached_nan":
        changed = first.descriptors.clone()
        changed[0, 0] = float("nan")
        groups[0] = replace(first, descriptors=changed)
    elif kind == "ambiguous_ordinal":
        changed = first.ordinal.clone()
        changed[1] = changed[0]
        groups[0] = replace(first, ordinal=changed)
    with pytest.raises(ValueError):
        fast.assemble(groups, side=32, context_cap=1, halo=99 if kind == "halo" else 8)
