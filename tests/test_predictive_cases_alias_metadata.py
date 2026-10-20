"""Regression for canonical metadata when an acquisition contains only aliases."""

import hashlib
from dataclasses import replace

import pytest
import torch

from rsna.competitive.predictive.cases import RawGroup, assemble
from rsna.competitive.predictive.inputs import masked_study


@pytest.fixture(autouse=True)
def one_cpu_thread():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def groups_with_aliases(kind):
    groups = []
    for group_id, (height, width, count) in enumerate(((40, 48, 3), (48, 40, 2))):
        pixels = torch.arange(height * width, dtype=torch.float32).reshape(height, width)
        frames = torch.stack([((pixels + 31 * i) % 251) / 250 for i in range(count)])
        aliases = [19 - i + 100 * group_id for i in range(count)]
        acquisition = [10 + 20 * group_id] * count
        ordinal = list(range(count))
        if kind == "partial":
            acquisition[-1] = 70 + group_id
        if kind != "unique":
            frames = torch.cat((frames, frames[[1, 0]].clone()))
            aliases.extend((aliases[1], aliases[0]))
            duplicate_acq = {
                "same": acquisition[0],
                "existing": 30,
                "partial": 70 + group_id,
                "absent": 90,
            }[kind]
            acquisition.extend((duplicate_acq, duplicate_acq))
            ordinal.extend((11, 12))
        groups.append(
            RawGroup(
                frames, torch.tensor(aliases), torch.tensor(acquisition), torch.tensor(ordinal)
            )
        )
    return groups


def fingerprint(contexts):
    """All bytes of both masks, rendered centers, metadata, targets and CSR priors."""
    digest = hashlib.sha256()
    for context in contexts:
        tensors = {name: value for name, value in context.items() if name != "supports"}
        supports = context["supports"]
        tensors.update(
            {name: getattr(supports, name) for name in ("indptr", "regions", "log_prior")}
        )
        digest.update(str(supports.region_count).encode())
        for name, tensor in sorted(tensors.items()):
            digest.update(f"{name}/{tensor.dtype}/{tuple(tensor.shape)}".encode())
            digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


# Captured from the unchanged pre-fix renderer before modifying cases.py.
LEGACY = {
    ("unique", 1): "de6bca5e8c74de09a77afab4e80eb49396543bb91b33c32b2c08a6299082262e",
    ("unique", 4): "2770d13bea71b8c14249f867fcf3edb5f6dc6a261799f16d03bfe65d6be78623",
    ("unique", 94): "184386a4d2d17e68c802c2256472c888c955882ef1f867c9591a7ea3c2296d15",
    ("same", 1): "de6bca5e8c74de09a77afab4e80eb49396543bb91b33c32b2c08a6299082262e",
    ("same", 4): "2770d13bea71b8c14249f867fcf3edb5f6dc6a261799f16d03bfe65d6be78623",
    ("same", 94): "3f20bf400da8a1eebd781db8505030729d6c5dd071530730ed71014e994cc71b",
    ("existing", 1): "de6bca5e8c74de09a77afab4e80eb49396543bb91b33c32b2c08a6299082262e",
    ("existing", 4): "2770d13bea71b8c14249f867fcf3edb5f6dc6a261799f16d03bfe65d6be78623",
    ("existing", 94): "184386a4d2d17e68c802c2256472c888c955882ef1f867c9591a7ea3c2296d15",
    ("partial", 1): "418aefdbb481b5b2fb46c6e175e9c19516cbd98d3c6df2e25410aa9f6bf44478",
    ("partial", 4): "63efdbd53c0e9ac0351712befc317460d50e6b0d6c5261aff17a4ffe052d6409",
    ("partial", 94): "780a7604e1ab2d701508ee540892bf9ce1f6c4f41eae6ec64be3f7ec73da88a5",
}


@pytest.mark.parametrize("kind", ["unique", "same", "existing", "partial"])
@pytest.mark.parametrize("context_cap", [1, 4, 94])
def test_previously_successful_contexts_remain_bitwise_equal(kind, context_cap):
    contexts = assemble(groups_with_aliases(kind), side=24, halo=2, context_cap=context_cap)
    assert fingerprint(contexts) == LEGACY[(kind, context_cap)]


@pytest.mark.parametrize("context_cap", [1, 4, 94])
@pytest.mark.parametrize("cached", [False, True])
def test_entirely_duplicate_acquisition_keeps_complete_canonical_evidence(context_cap, cached):
    canonical = groups_with_aliases("unique")
    duplicated = groups_with_aliases("absent")
    assert all(90 not in group.acquisition[:-2] for group in duplicated)
    if cached:
        duplicated = [
            replace(
                group,
                descriptors=masked_study(
                    group.frames,
                    group.aliases,
                    group.acquisition,
                    group.ordinal,
                    pattern=0,
                    halo=2,
                ).descriptors,
            )
            for group in duplicated
        ]
    before = [(g.frames.clone(), g.acquisition.clone(), g.ordinal.clone()) for g in duplicated]
    actual = assemble(duplicated, side=24, halo=2, context_cap=context_cap)
    expected = assemble(canonical, side=24, halo=2, context_cap=context_cap)
    assert fingerprint(actual) == fingerprint(expected)
    assert torch.equal(actual[0]["observed"], ~actual[1]["observed"])
    for context in actual:
        assert context["supports"].region_count == 5 * 16
        assert torch.allclose(context["supports"].log_prior.exp().sum(1), torch.ones(12))
    for group, values in zip(duplicated, before):
        assert all(
            torch.equal(getattr(group, key), value)
            for key, value in zip(("frames", "acquisition", "ordinal"), values)
        )


@pytest.mark.parametrize("cached", [False, True])
def test_duplicate_alias_with_unequal_pixels_is_rejected(cached):
    groups = groups_with_aliases("absent")
    group = groups[0]
    descriptors = (
        masked_study(
            group.frames, group.aliases, group.acquisition, group.ordinal, pattern=0, halo=2
        ).descriptors
        if cached
        else None
    )
    changed = group.frames.clone()
    changed[-1, 0, 0] = 0.5
    groups[0] = replace(group, frames=changed, descriptors=descriptors)
    with pytest.raises(ValueError, match="Alias frames disagree"):
        assemble(groups, side=24, halo=2, context_cap=4)
