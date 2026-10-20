"""Source-pixel causal interventions and independent dense-oracle comparisons."""

import copy
import importlib.util
import sys
from pathlib import Path

import pytest
import torch

from rsna.competitive.predictive.density import Supports, evidence, two_mask_loss
from rsna.competitive.predictive.inputs import fixed_metadata_scale, masked_study, triplets


@pytest.mark.parametrize("arm", ["B", "C", "C0"])
def test_actual_new_loop_recovery_and_complete_target_roster(tmp_path, arm):
    """CPU engineering: real new loop, never a clinical/model acceptance claim."""
    import numpy as np
    from torch import nn

    from rsna.competitive.predictive.cases import RawGroup, assemble
    from rsna.competitive.predictive.model import matched_modules
    from rsna.competitive.predictive.train import Recipe, fit, restore_rng, rng_state

    groups = [
        RawGroup(
            torch.rand(2, 64, 64), torch.tensor([0, 1]), torch.tensor([0, 0]), torch.tensor([0, 1])
        ),
        RawGroup(torch.rand(1, 80, 80), torch.tensor([2]), torch.tensor([1]), torch.tensor([0])),
    ]
    contexts = assemble(groups, side=32, halo=2, context_cap=2, classes=2)
    assert len(contexts[0]["images"]) == 2
    assert len(contexts[0]["targets"]) == 48  # Unselected frame is still present.
    assert contexts[0]["supports"].region_count == 48
    before = copy.deepcopy(contexts)
    altered = []
    for g in groups:
        mask = masked_study(
            g.frames, g.aliases, g.acquisition, g.ordinal, pattern=0, halo=2
        ).raw_hidden
        value = g.frames.clone()
        value[mask] = 1 - value[mask]
        altered.append(RawGroup(value, g.aliases, g.acquisition, g.ordinal))
    changed = assemble(altered, side=32, halo=2, context_cap=2, classes=2)
    assert torch.equal(contexts[0]["images"], changed[0]["images"])
    assert not torch.equal(contexts[0]["targets"], changed[0]["targets"])
    encoder = nn.Sequential(nn.Conv2d(3, 4, 3), nn.GELU(), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    B, C, _ = matched_modules(encoder, 4, hidden=8, classes=2)
    base = B if arm == "B" else C
    from rsna.competitive.predictive.infer import predict
    from rsna.competitive.predictive.train import objective

    base.eval()
    inference_recipe = Recipe(arm, density_weight=0 if arm == "C0" else 0.1)
    probabilities = predict(base, contexts)
    with torch.inference_mode():
        _, measures = objective(
            base,
            {
                "contexts": contexts,
                "labels": torch.zeros(1, 2),
                "mask": torch.ones(1, 2, dtype=torch.bool),
            },
            inference_recipe,
            torch.device("cpu"),
        )
    torch.testing.assert_close(probabilities, measures["probabilities"][0], atol=1e-7, rtol=1e-6)
    with pytest.raises(ValueError):
        predict(base, [{**contexts[0], "labels": torch.ones(1, 2)}, contexts[1]])
    with pytest.raises(ValueError):
        predict(base, contexts, precision="fp16")

    class Data:
        def __len__(self):
            return 3

        def case(self, index, epoch):
            # Exercise restored RNG through a common stochastic loader feature.
            torch.rand(1)
            np.random.rand()
            return {
                "contexts": contexts,
                "labels": torch.tensor([[float(index % 2), 0.25]]),
                "mask": torch.tensor([[True, True]]),
            }

    recipe = Recipe(
        arm,
        epochs=2,
        effective_batch=2,
        head_only_updates=1,
        warmup_fraction=0.4,  # Fractional warmup length must never exceed declared rates.
        density_weight=0 if arm == "C0" else 0.1,
    )
    binding = {
        "synthetic_engineering_only": True,
        "immutable_fixture": "three-cases-two-matrixgroups",
    }
    initial_rng = rng_state()
    complete = fit(copy.deepcopy(base), Data(), recipe, binding=binding, output=tmp_path / "full")
    restore_rng(initial_rng)
    stopped = fit(
        copy.deepcopy(base), Data(), recipe, binding=binding, output=tmp_path / "stop", stop_after=1
    )
    assert stopped["step"] == 1
    resumed = fit(
        copy.deepcopy(base),
        Data(),
        recipe,
        binding=binding,
        output=tmp_path / "resumed",
        resume=tmp_path / "stop/stopped.pt",
    )
    assert complete["step"] == resumed["step"] == 4
    assert complete["history"] == resumed["history"]
    assert all(
        0 < row["LR"][0] <= recipe.encoder_lr and 0 < row["LR"][1] <= recipe.head_lr
        for row in complete["history"]
    )
    assert [r["studies"] for r in complete["history"]] == [2, 1, 2, 1]
    assert all(r["encoder_gradient_norm"] > 0 for r in complete["history"])
    assert not complete["history"][0]["encoder_updated"]
    assert complete["history"][1]["encoder_updated"]

    def equal(a, b):
        if isinstance(a, torch.Tensor):
            assert torch.equal(a, b)
        elif isinstance(a, np.ndarray):
            assert np.array_equal(a, b)
        elif isinstance(a, dict):
            assert a.keys() == b.keys()
            for k in a:
                equal(a[k], b[k])
        elif isinstance(a, (list, tuple)):
            assert len(a) == len(b)
            for x, y in zip(a, b):
                equal(x, y)
        else:
            assert a == b

    equal(complete, resumed)
    for x, y in zip(before, contexts):
        assert torch.equal(x["images"], y["images"])


P = Path(
    "initiative/research/predictive-evidence-proposal-20261020-v1/reference/predictive_evidence.py"
)
spec = importlib.util.spec_from_file_location("provided_predictive_dense_oracle", P)
oracle = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = oracle
spec.loader.exec_module(oracle)


def test_source_target_intervention_cannot_change_any_overlapping_context_channel():
    g = torch.Generator().manual_seed(20)
    first = torch.randint(0, 4096, (64, 64), generator=g, dtype=torch.int32)
    other = torch.randint(0, 4096, (64, 64), generator=g, dtype=torch.int32)
    raw = torch.stack((first, other, first))
    frames = torch.stack([fixed_metadata_scale(x, bits_stored=12) for x in raw])
    alias = torch.tensor([0, 1, 0])
    acq = torch.zeros(3, dtype=torch.int64)
    order = torch.tensor([0, 1, 0])
    a = masked_study(frames, alias, acq, order, pattern=0, halo=2)
    perturbed = raw.clone()
    perturbed[a.raw_hidden] = 4095 - perturbed[a.raw_hidden]
    changed = torch.stack([fixed_metadata_scale(x, bits_stored=12) for x in perturbed])
    b = masked_study(changed, alias, acq, order, pattern=0, halo=2)
    assert torch.equal(a.context_frames, b.context_frames)
    assert not torch.equal(a.descriptors[a.observed], b.descriptors[b.observed])
    centers = torch.tensor([0, 1, 2])
    neighbors = torch.tensor([[0, 0, 1], [0, 1, 2], [1, 2, 2]])
    assert torch.equal(
        triplets(a, centers, neighbors, side=32), triplets(b, centers, neighbors, side=32)
    )
    assert len(a.descriptors) == 32  # Duplicate source frame contributes no extra region.
    complement = masked_study(frames, alias, acq, order, pattern=1, halo=2)
    assert torch.equal(a.observed, ~complement.observed)
    assert (~a.raw_hidden).any() and (~complement.raw_hidden).any()


def test_raw_pixels_cannot_replace_masked_input_and_cross_series_neighbors_refused():
    x = torch.rand(2, 64, 64)
    a = masked_study(
        x, torch.arange(2), torch.arange(2), torch.zeros(2, dtype=torch.int64), pattern=0, halo=2
    )
    with pytest.raises(ValueError):
        triplets(x, torch.tensor([0]), torch.tensor([[0, 0, 0]]))
    with pytest.raises(ValueError):
        triplets(a, torch.tensor([0]), torch.tensor([[0, 0, 1]]))


def fixture():
    g = torch.Generator().manual_seed(33)
    t = torch.randn(2, 5, 4, generator=g, dtype=torch.float64)
    mu = torch.randn(2, 2, 5, 4, generator=g, dtype=torch.float64, requires_grad=True)
    delta = torch.randn(2, 2, 5, 4, generator=g, dtype=torch.float64, requires_grad=True) * 0.1
    var = torch.zeros(2, 4, dtype=torch.float64)
    observed = torch.tensor([[1, 0, 1, 0, 1], [0, 0, 0, 0, 0]], dtype=torch.bool)
    supports = Supports(torch.arange(6), torch.arange(5), torch.zeros(2, 5, dtype=torch.float64), 5)
    return t, mu, delta, var, supports, observed


def test_lossless_original_integer_cache_preserves_all_masked_inputs_and_targets(tmp_path):
    from rsna.competitive.predictive.cache import SourceGroup, load, normalized, write
    from rsna.competitive.predictive.cases import RawGroup, assemble

    g = torch.Generator().manual_seed(741)
    sources = [
        SourceGroup(
            torch.randint(0, 4096, (2, 96, 96), generator=g, dtype=torch.int32),
            torch.tensor([12, 12]),
            torch.tensor([False, False]),
            torch.tensor([False, False]),
            torch.tensor([0, 1]),
            torch.tensor([0, 0]),
            torch.tensor([0, 1]),
        ),
        SourceGroup(
            torch.randint(-32768, 32768, (1, 128, 128), generator=g, dtype=torch.int32),
            torch.tensor([16]),
            torch.tensor([True]),
            torch.tensor([True]),
            torch.tensor([2]),
            torch.tensor([1]),
            torch.tensor([0]),
        ),
    ]
    original = [RawGroup(normalized(x), x.aliases, x.acquisition, x.ordinal) for x in sources]
    path = tmp_path / "private-source.npz"
    proof = write(path, sources, {"synthetic_encoding_fixture_only": True}, cap_bytes=2**20)
    assert path.stat().st_mode & 0o077 == 0
    restored, provenance = load(path, proof)
    assert provenance == {"synthetic_encoding_fixture_only": True}
    for a, b in zip(original, restored):
        assert torch.equal(a.frames, b.frames)
    a, b = (assemble(x, side=32, context_cap=2, classes=2) for x in (original, restored))
    for first, second in zip(a, b):
        for name in ("images", "targets", "observed", "token_metadata", "region_metadata"):
            assert torch.equal(first[name], second[name])
        for name in ("indptr", "regions", "log_prior"):
            assert torch.equal(getattr(first["supports"], name), getattr(second["supports"], name))
    with pytest.raises(FileExistsError):
        write(path, sources, {"synthetic_encoding_fixture_only": True}, cap_bytes=2**20)
    with path.open("ab") as f:
        f.write(b"changed")
    with pytest.raises(ValueError):
        load(path, proof)


def test_sparse_complete_roster_density_and_gradient_equal_independent_provided_oracle():
    t, mu, delta, var, s, o = fixture()
    a = evidence(t, mu, delta, var, s, o, edge_chunk=2)
    dense = torch.eye(5, dtype=torch.bool)[None].expand(2, -1, -1)
    b = oracle.predictive_evidence(t, mu, delta, var, dense, s.log_prior, o)
    for key in (
        "log_p0",
        "log_p1",
        "log_bayes_factor",
        "local_log_ratio",
        "no_observation_prior_mass",
    ):
        torch.testing.assert_close(getattr(a, key), getattr(b, key), atol=1e-12, rtol=1e-12)
    left = torch.autograd.grad(a.log_p1.sum(), (mu, delta), retain_graph=True)
    right = torch.autograd.grad(b.log_p1.sum(), (mu, delta))
    for x, y in zip(left, right):
        torch.testing.assert_close(x, y, atol=1e-12, rtol=1e-12)
    assert a.log_bayes_factor[1].abs().max() < 1e-12
    assert (a.log_bayes_factor[0] >= a.no_observation_prior_mass[0].log() - 1e-12).all()


def test_repeated_evidence_inside_support_refused_and_alternative_split_invariant():
    t, mu, d, var, s, o = fixture()
    bad = Supports(torch.tensor([0, 2]), torch.tensor([0, 0]), torch.zeros(2, 1, dtype=t.dtype), 5)
    with pytest.raises(ValueError):
        evidence(t, mu, d, var, bad, o)
    split = Supports(
        torch.arange(11), torch.arange(5).repeat_interleave(2), torch.zeros(2, 10, dtype=t.dtype), 5
    )
    torch.testing.assert_close(
        evidence(t, mu, d, var, s, o).log_bayes_factor,
        evidence(t, mu, d, var, split, o).log_bayes_factor,
    )


def test_training_uses_BCE_of_mean_posterior_not_mean_BCE():
    t, mu, d, var, s, o = fixture()
    a = evidence(t, mu, d, var, s, o)
    z = torch.tensor(
        [[[2.0, -1.0], [-2.0, 3.0]], [[0.0, 1.0], [2.0, -3.0]]], dtype=t.dtype, requires_grad=True
    )
    q = torch.tensor([[0.2, 0.8], [0.4, 0.6]], dtype=t.dtype)
    m = torch.ones_like(q, dtype=torch.bool)
    loss, out = two_mask_loss([a, a], z, q, m, [o, o], descriptor_width=4, density_weight=0)
    p = (z + torch.stack([a.log_bayes_factor, a.log_bayes_factor], 1)).sigmoid().mean(1)
    torch.testing.assert_close(loss, torch.nn.functional.binary_cross_entropy(p, q))
    loss.backward()
    assert torch.isfinite(z.grad).all()
    torch.testing.assert_close(out["probabilities"], p)


def test_new_B_C_modules_train_encoder_without_teacher_or_target_leakage():
    import copy

    from torch import nn

    from rsna.competitive.predictive.model import (
        Predictive,
        ReconstructionControl,
        control_two_mask_loss,
    )

    torch.manual_seed(6)
    encoder = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.BatchNorm2d(8),
        nn.GELU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, 16),
    )
    common = copy.deepcopy(encoder.state_dict())
    images = [torch.rand(6, 3, 32, 32) for _ in range(2)]
    token = torch.rand(6, 8)
    region = torch.rand(16, 8)
    targets = torch.rand(16, 64)
    observations = [torch.arange(16) % 2 == i for i in (0, 1)]
    supports = Supports(torch.arange(17), torch.arange(16), torch.zeros(2, 16), 16)
    q = torch.tensor([[0.2, 0.8]])
    mask = torch.ones_like(q, dtype=torch.bool)
    for cls in (Predictive, ReconstructionControl):
        backbone = copy.deepcopy(encoder)
        backbone.load_state_dict(common)
        model = cls(backbone, 16, hidden=16, classes=2).train()
        bn = model.encoder[1].running_mean.clone()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        before = model.encoder[0].weight.detach().clone()
        if cls is Predictive:
            first = model.predict_hypotheses(images[0], token, region)
            changed = targets.flip(0)
            a = model(images[0], token, region, targets, observations[0], supports)
            b = model(images[0], token, region, changed, observations[0], supports)
            # Targets enter only the likelihood scorer, never the context means.
            second = model.predict_hypotheses(images[0], token, region)
            for x, y in zip(first, second):
                torch.testing.assert_close(x, y, atol=0, rtol=0)
            assert not hasattr(model, "parent_logits")
            torch.testing.assert_close(
                a["evidence"].log_bayes_factor, torch.zeros(1, 2), atol=1e-6, rtol=0
            )
            outputs = [
                model(x, token, region, targets, o, supports) for x, o in zip(images, observations)
            ]
            loss, _ = two_mask_loss(
                [x["evidence"] for x in outputs],
                torch.stack([x["context_logits"] for x in outputs])[None],
                q,
                mask,
                [o[None] for o in observations],
            )
            assert b["logits"].shape == (2,)
        else:
            outputs = [model(x, token, region, targets, o) for x, o in zip(images, observations)]
            loss, _ = control_two_mask_loss(outputs, q, mask)
            # The strong comparator actually receives the observed evidence.
            altered = model(images[0], token, region, targets + 1, observations[0])
            assert not torch.equal(altered["logits"], outputs[0]["logits"])
        assert torch.isfinite(loss)
        loss.backward()
        assert model.encoder[0].weight.grad.abs().sum() > 0
        optimizer.step()
        assert not torch.equal(before, model.encoder[0].weight)
        assert torch.equal(bn, model.encoder[1].running_mean)


def test_complete_multiscale_priors_ignore_observation_pattern_and_shared_init_exact():
    from torch import nn

    from rsna.competitive.predictive.model import matched_modules
    from rsna.competitive.predictive.supports import build_supports, coordinates

    raw = torch.rand(4, 64, 64)
    alias = torch.arange(4)
    acq = torch.zeros(4, dtype=torch.int64)
    ordinal = torch.arange(4)
    a = masked_study(raw, alias, acq, ordinal, pattern=0, halo=2)
    b = masked_study(raw, alias, acq, ordinal, pattern=1, halo=2)
    left = build_supports(a.region_frame, a.region_tile, acq, ordinal, classes=2)
    right = build_supports(b.region_frame, b.region_tile, acq, ordinal, classes=2)
    assert torch.equal(left.indptr, right.indptr) and torch.equal(left.regions, right.regions)
    torch.testing.assert_close(left.log_prior.log_softmax(-1).exp().sum(-1), torch.ones(2))
    ta, ra = coordinates(a, ordinal)
    tb, rb = coordinates(b, ordinal)
    assert torch.equal(ta, tb) and torch.equal(ra, rb)
    before = torch.get_rng_state().clone()
    backbone = nn.Sequential(nn.Conv2d(3, 4, 1), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    before = torch.get_rng_state().clone()
    B, C, proof = matched_modules(backbone, 4, hidden=16, classes=2, seed=11)
    assert torch.equal(before, torch.get_rng_state())
    assert proof["same_shared_tensor_values"] and proof["C_parameters"] != proof["B_parameters"]
    for n in proof["shared_tensor_names"]:
        assert torch.equal(B.state_dict()[n], C.state_dict()[n])


def test_exact_context_cache_preserves_probabilities_and_source_binding(tmp_path):
    """Cached FP32 inputs preserve actual readout and RNG; changed source is refused."""
    from torch import nn

    from rsna.competitive.predictive import context_cache
    from rsna.competitive.predictive.cases import RawGroup, assemble
    from rsna.competitive.predictive.infer import predict
    from rsna.competitive.predictive.model import matched_modules

    groups = [
        RawGroup(
            torch.rand(3, 64, 64),
            torch.arange(3),
            torch.zeros(3, dtype=torch.int64),
            torch.arange(3),
        )
    ]
    contexts = assemble(groups, side=32, context_cap=2, classes=2)
    encoder = nn.Sequential(nn.Conv2d(3, 4, 3), nn.AdaptiveAvgPool2d(1), nn.Flatten())
    B, C, _ = matched_modules(encoder, 4, classes=2, hidden=8)
    provenance = {"raw_cache_sha256": "a" * 64, "side": 32, "context_cap": 2, "classes": 2}
    before = torch.get_rng_state().clone()
    path = tmp_path / "contexts.npz"
    proof = context_cache.write(path, contexts, provenance, cap_bytes=2**20)
    restored = context_cache.load(path, proof, provenance=provenance)
    assert torch.equal(before, torch.get_rng_state())
    for left, right in zip(contexts, restored):
        for key in context_cache.KEYS:
            assert torch.equal(left[key], right[key])
        for key in ("indptr", "regions", "log_prior"):
            assert torch.equal(getattr(left["supports"], key), getattr(right["supports"], key))
    for model in (B, C):
        model.eval()
        torch.testing.assert_close(
            predict(model, contexts), predict(model, restored), rtol=0, atol=0
        )
    with pytest.raises(ValueError):
        context_cache.load(path, proof, provenance={**provenance, "context_cap": 3})
    with pytest.raises(FileExistsError):
        context_cache.write(path, contexts, provenance, cap_bytes=2**20)
