from dataclasses import replace

import torch

from rsna.competitive.predictive import cases, visible_contrast
from rsna.competitive.predictive.inputs import masked_study
from rsna.competitive.predictive.model import Predictive


def groups():
    # Different encoding offsets/ranges across two original acquisitions.
    ramp = torch.linspace(0, 1, 96 * 96).reshape(96, 96)
    frames = torch.stack([.002 + .02 * ramp, .002 + .02 * ramp, .5 + .03 * ramp])
    return [cases.RawGroup(frames, torch.tensor([0, 0, 1]), torch.tensor([0, 0, 1]),
                           torch.tensor([0, 0, 0]))]


def test_original_targets_supports_and_matched_control_are_preserved():
    original = cases.assemble(groups(), side=32, context_cap=2, classes=2)
    candidate, scales = visible_contrast.assemble(groups(), side=32, context_cap=2, classes=2)
    control, control_scales = visible_contrast.assemble(
        groups(), side=32, context_cap=2, classes=2, normalize_images=False,
    )
    assert scales == control_scales and set(scales) == {0, 1}
    for raw, new, matched in zip(original, candidate, control):
        assert torch.equal(raw["images"], matched["images"])
        assert not torch.equal(raw["images"], new["images"])
        for name in ("targets", "observed"):
            assert torch.equal(raw[name], new[name]) and torch.equal(raw[name], matched[name])
        for name in ("token_metadata", "region_metadata"):
            assert new[name].shape[1] == 10
            assert torch.equal(raw[name], new[name][:, :8])
            assert torch.equal(new[name], matched[name])
        for name in ("indptr", "regions", "log_prior"):
            assert torch.equal(getattr(raw["supports"], name), getattr(new["supports"], name))


def test_withheld_pixels_cannot_change_context_scaling_or_metadata():
    original = groups()
    g = original[0]
    mask = masked_study(g.frames, g.aliases, g.acquisition, g.ordinal, pattern=0).raw_hidden
    changed = [replace(g, frames=g.frames.clone().masked_fill(mask, .91), descriptors=None)]
    before, scales = visible_contrast.assemble(original, side=32, context_cap=2, classes=2)
    after, changed_scales = visible_contrast.assemble(changed, side=32, context_cap=2, classes=2)
    assert scales[0] == changed_scales[0]
    for name in ("images", "token_metadata", "region_metadata", "observed"):
        assert torch.equal(before[0][name], after[0][name])
    assert not torch.equal(before[0]["targets"], after[0]["targets"])
    assert not torch.equal(before[1]["images"], after[1]["images"])


def test_alias_repetition_does_not_change_scale_and_does_not_modify_source():
    g = groups()[0]
    old = g.frames.clone()
    a = visible_contrast.acquisition_scales([g], pattern=0)
    without_alias = [replace(g, **{name: getattr(g, name)[[0, 2]]
                                  for name in ("frames", "aliases", "acquisition", "ordinal")})]
    assert a == visible_contrast.acquisition_scales(without_alias, pattern=0)
    visible_contrast.assemble([g], side=32, context_cap=2, classes=2)
    assert torch.equal(g.frames, old)


def test_alias_only_acquisition_reuses_canonical_scale_coordinates():
    g = groups()[0]
    g = replace(g, acquisition=torch.tensor([0, 4, 1]))
    contexts, scales = visible_contrast.assemble([g], side=32, context_cap=2, classes=2)
    assert all(set(each) == {0, 1} for each in scales.values())
    assert all(torch.isfinite(c["token_metadata"]).all() for c in contexts)


def test_flat_visible_source_is_bounded_and_scale_is_supplied_to_predictor():
    g = groups()[0]
    g = replace(g, frames=torch.full_like(g.frames, .503))
    contexts, scales = visible_contrast.assemble([g], side=32, context_cap=2, classes=2)
    assert all(s.span == .01 for each in scales.values() for s in each.values())
    encoder = torch.nn.Sequential(torch.nn.Flatten(), torch.nn.Linear(3 * 32 * 32, 8))
    model = Predictive(encoder, 8, hidden=8, metadata_width=10, classes=2)
    context = contexts[0]
    output = model(*(context[k] for k in (
        "images", "token_metadata", "region_metadata", "targets", "observed", "supports",
    )))
    loss = output["logits"].square().sum() - .1 * output["evidence"].log_p1.mean()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.position[0].in_features == 10
