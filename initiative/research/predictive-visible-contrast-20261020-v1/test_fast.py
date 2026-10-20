from dataclasses import replace

import pytest
import torch

from rsna.competitive.predictive import cases, visible_contrast, visible_contrast_fast
from rsna.competitive.predictive.inputs import masked_study


@pytest.mark.parametrize("shape,halo", [((96, 112), 8), ((103, 127), 8), ((32, 40), 0)])
def test_geometry_selects_exact_original_visible_positions(shape, halo):
    values = torch.arange(shape[0] * shape[1], dtype=torch.float32).reshape(shape)
    frames = (values / values.max())[None].repeat(6, 1, 1)
    group = cases.RawGroup(frames, torch.tensor([0, 1, 1, 2, 3, 4]),
                           torch.tensor([0, 1, 2, 3, 4, 5]), torch.tensor([0, 4, 8, 9, 16, 19]))
    for pattern in (0, 1):
        view = masked_study(frames, group.aliases, group.acquisition, group.ordinal,
                            pattern=pattern, halo=halo)
        for i in (0, 1, 3, 4, 5):
            parity = (pattern - int(group.ordinal[i]) // 4 - int(group.acquisition[i])) % 2
            positions = visible_contrast_fast.visible_positions(*shape, parity=parity, halo=halo)
            visible = values[~view.raw_hidden[i]]
            selected = torch.linspace(0, len(visible)-1, min(4096, len(visible))).round().long()
            assert torch.equal(values.flatten()[positions], visible[selected])
        assert visible_contrast.acquisition_scales([group], pattern=pattern, halo=halo) == \
            visible_contrast_fast.acquisition_scales([group], pattern=pattern, halo=halo)


def equal_contexts(left, right):
    for a, b in zip(left, right, strict=True):
        assert a.keys() == b.keys()
        for key in a:
            if key == "supports":
                assert a[key].region_count == b[key].region_count
                for field in ("indptr", "regions", "log_prior"):
                    assert torch.equal(getattr(a[key], field), getattr(b[key], field))
            else:
                assert a[key].dtype == b[key].dtype and torch.equal(a[key], b[key])


def test_full_candidate_is_bitwise_equivalent_and_hidden_intervention_invariant():
    generator = torch.Generator().manual_seed(918)
    frames = torch.rand((3, 99, 117), generator=generator) * .04 + .01
    frames[1] = frames[0]
    group = cases.RawGroup(frames, torch.tensor([0, 0, 1]),
                           torch.tensor([0, 1, 2]), torch.tensor([0, 0, 5]))
    other = cases.RawGroup(torch.rand((2, 128, 96), generator=generator), torch.tensor([2, 3]),
                           torch.tensor([3, 3]), torch.tensor([4, 9]))
    groups = [group, other]
    rng = torch.get_rng_state().clone()
    functions = (cases.assemble, visible_contrast.assemble, visible_contrast.acquisition_scales)
    for normalize in (True, False):
        old, old_scales = visible_contrast.assemble(groups, side=64, classes=2, normalize_images=normalize)
        new, new_scales = visible_contrast_fast.assemble(groups, side=64, classes=2, normalize_images=normalize)
        equal_contexts(old, new)
        assert old_scales == new_scales
    original, scales = visible_contrast_fast.assemble(groups, side=64, classes=2)
    changed = []
    for g in groups:
        view = masked_study(g.frames, g.aliases, g.acquisition, g.ordinal, pattern=0)
        frames = g.frames.clone()
        frames[view.raw_hidden] = 1 - frames[view.raw_hidden]
        changed.append(replace(g, frames=frames, descriptors=None))
    altered, altered_scales = visible_contrast_fast.assemble(changed, side=64, classes=2)
    assert scales[0] == altered_scales[0]
    for key in ("images", "token_metadata", "region_metadata", "observed"):
        assert torch.equal(original[0][key], altered[0][key])
    assert not torch.equal(original[0]["targets"], altered[0]["targets"])
    assert torch.equal(rng, torch.get_rng_state())
    assert functions == (cases.assemble, visible_contrast.assemble, visible_contrast.acquisition_scales)


def test_equivalent_path_keeps_original_alias_validation():
    group = cases.RawGroup(torch.rand(2, 96, 96), torch.tensor([0, 0]),
                           torch.tensor([0, 0]), torch.tensor([0, 1]))
    with pytest.raises(ValueError, match="Alias frames disagree"):
        visible_contrast_fast.assemble([group])
