import copy
import json
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel

from rsna.competitive.adapters import ResidualLinear
from rsna.competitive.checkpoint import capture_rng, load_checkpoint, restore_rng, save_checkpoint
from rsna.competitive.delta_checkpoint import FrozenBaseDelta
from rsna.competitive.loss import effective_batch_loss, observed_counts
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.relations import (
    effective_relation_loss,
    relation_model_kwargs,
    relation_study_count,
    relations_for_batch,
)
from rsna.competitive.train import TrainConfig

POLICY = dict(weight=0.05, max_pairs=4, target_mode="physical", hidden_dim=12, head_seed=19)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = ResidualLinear(nn.Linear(1, 8), rank=2, alpha=2, dropout=0)
        self.calls = 0

    def forward(self, images):
        self.calls += 1
        return self.linear(images.mean((2, 3)))


def batch():
    torch.manual_seed(61)
    positions = torch.zeros(4, 3, 6, 2)
    positions[..., 0] = torch.tensor([-0.13, -0.07, 0, 0.02, 0.1, 0.17])
    positions[..., 1] = 1
    positions[2] = 0  # This genuine classification study has no known geometry.
    return dict(
        images=torch.rand(4, 3, 6, 1, 16, 16),
        features=torch.eye(3, 5).repeat(4, 1, 1),
        series_mask=torch.ones(4, 3, dtype=torch.bool),
        slice_mask=torch.ones(4, 3, 6, dtype=torch.bool),
        positions=positions,
        sample_valid=torch.tensor([True, False, True, False]),
        study_uid=["a", "b", "c", "d"],
        labels=torch.randint(2, (4, 12)).float(),
        label_mask=torch.tensor([True, False, True, False])[:, None].expand(4, 12).float(),
    )


def select(data, part):
    return {name: value[part] for name, value in data.items()}


def inputs(data):
    return tuple(
        data[name] for name in ("images", "features", "series_mask", "slice_mask", "positions")
    )


def model(aux=True):
    torch.manual_seed(17)
    return CompetitiveStudyNet(
        Encoder(), 8, hidden_dim=12, **(relation_model_kwargs(POLICY) if aux else {})
    )


def test_initial_classification_rng_and_relation_metadata_separation():
    reference = model(False).eval()
    rng = torch.get_rng_state().clone()
    candidate = model().eval()
    assert torch.equal(torch.get_rng_state(), rng)
    for name, tensor in reference.state_dict().items():
        torch.testing.assert_close(candidate.state_dict()[name], tensor, rtol=0, atol=0)
    data = batch()
    relations = relations_for_batch(data, POLICY, seed=12, epoch=0)
    before = candidate.encoder.calls
    logits, loss = candidate(*inputs(data), relations=relations)
    assert candidate.encoder.calls - before == 2  # Original72 images/chunk64, no second encoding.
    torch.testing.assert_close(logits, reference(*inputs(data)), rtol=0, atol=0)
    changed = copy.deepcopy(data)
    changed["positions"][0, ..., 0] += 10
    changed["features"] *= 2
    _, changed_loss = candidate(*inputs(changed), relations=relations)
    torch.testing.assert_close(loss, changed_loss, rtol=0, atol=0)
    # Remove only the dormant training head after restoring the complete model.
    candidate.relation_head = None
    torch.testing.assert_close(candidate(*inputs(data)), logits, rtol=0, atol=0)


def gradient_step(network, data, count, policy=POLICY):
    relations = relations_for_batch(data, policy, seed=12, epoch=0)
    logits, relation_sum = network(*inputs(data), relations=relations)
    loss = effective_batch_loss(logits, data["labels"], data["label_mask"], count[0])
    loss += policy["weight"] * effective_relation_loss(relation_sum, count[1])
    loss.backward()
    return loss.detach()


def test_accumulation_preserves_pair_roster_and_global_mean_with_unknown_geometry():
    data = batch()
    reference = model()
    candidate = model()
    whole = relations_for_batch(data, POLICY, seed=12, epoch=0)
    pieces = [select(data, slice(0, 2)), select(data, slice(2, 4))]
    relation_pieces = [relations_for_batch(p, POLICY, seed=12, epoch=0) for p in pieces]
    assert whole.study_count == 1
    torch.testing.assert_close(whole.indices, relation_pieces[0].indices, rtol=0, atol=0)
    torch.testing.assert_close(whole.targets, relation_pieces[0].targets, rtol=0, atol=0)
    counts = (
        observed_counts([data["label_mask"]]),
        relation_study_count(relation_pieces, torch.device("cpu")),
    )
    expected = gradient_step(reference, data, counts)
    actual = sum(gradient_step(candidate, p, counts) for p in pieces)
    torch.testing.assert_close(actual, expected)
    for name, p in candidate.named_parameters():
        if p.requires_grad:
            assert p.grad is not None
            torch.testing.assert_close(
                p.grad, dict(reference.named_parameters())[name].grad, atol=1e-6, rtol=1e-5
            )


def test_uid_epoch_sampling_survives_reordering_and_shuffle_preserves_each_series():
    data = batch()
    data["sample_valid"].fill_(True)
    original = relations_for_batch(data, POLICY, seed=12, epoch=2)
    order = [3, 0, 2, 1]
    reordered = {
        name: [value[i] for i in order] if isinstance(value, list) else value[order]
        for name, value in data.items()
    }
    replay = relations_for_batch(reordered, POLICY, seed=12, epoch=2)
    shuffled = relations_for_batch(data, {**POLICY, "target_mode": "shuffled"}, seed=12, epoch=2)
    torch.testing.assert_close(shuffled.indices, original.indices, rtol=0, atol=0)
    for study in range(4):
        for series in range(3):
            selected = (original.indices[:, 0] == study) & (original.indices[:, 1] == series)
            repeated = (replay.indices[:, 0] == order.index(study)) & (
                replay.indices[:, 1] == series
            )
            torch.testing.assert_close(
                replay.indices[repeated, 1:], original.indices[selected, 1:], rtol=0, atol=0
            )
            torch.testing.assert_close(
                replay.targets[repeated], original.targets[selected], rtol=0, atol=0
            )
            torch.testing.assert_close(
                shuffled.targets[selected].sort().values,
                original.targets[selected].sort().values,
                rtol=0,
                atol=0,
            )


def test_zero_weight_preserves_shared_classification_gradients():
    data = batch()
    reference = model(False)
    candidate = model()
    counts = observed_counts([data["label_mask"]])
    effective_batch_loss(
        reference(*inputs(data)), data["labels"], data["label_mask"], counts
    ).backward()
    relation = relations_for_batch(data, POLICY, seed=12, epoch=0)
    gradient_step(
        candidate, data, (counts, torch.tensor(relation.study_count)), {**POLICY, "weight": 0}
    )
    for name, parameter in reference.named_parameters():
        if parameter.requires_grad:
            torch.testing.assert_close(
                dict(candidate.named_parameters())[name].grad, parameter.grad, rtol=0, atol=0
            )
    for parameter in candidate.relation_head.parameters():
        assert parameter.grad is not None and not parameter.grad.any()


def ddp_rank(rank, rendezvous):
    torch.set_num_threads(1)
    data = batch()
    reference = model()
    relation = relations_for_batch(data, POLICY, seed=12, epoch=0)
    gradient_step(
        reference,
        data,
        (
            observed_counts([data["label_mask"]]),
            relation_study_count([relation], torch.device("cpu")),
        ),
    )
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        candidate = model()
        wrapped = DistributedDataParallel(candidate)
        pieces = [select(data, slice(rank, rank + 1)), select(data, slice(rank + 2, rank + 3))]
        relations = [relations_for_batch(p, POLICY, seed=12, epoch=0) for p in pieces]
        counts = (
            observed_counts([p["label_mask"] for p in pieces]),
            relation_study_count(relations, torch.device("cpu")),
        )
        with wrapped.no_sync():
            gradient_step(wrapped, pieces[0], counts)
        gradient_step(wrapped, pieces[1], counts)
        for name, p in candidate.named_parameters():
            if p.requires_grad:
                assert p.grad is not None
                torch.testing.assert_close(
                    p.grad, dict(reference.named_parameters())[name].grad, atol=1e-6, rtol=1e-5
                )
    finally:
        dist.destroy_process_group()


def test_two_rank_tail_padding_and_zero_geometry_rank_match_global_gradient(tmp_path):
    mp.spawn(ddp_rank, args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=True)


def test_delta_adapter_head_optimizer_and_rng_restore_exactly(tmp_path):
    data = batch()
    network = model()
    original = network.encoder.linear.weight.detach().clone()
    provenance = dict(pretrained=True, weights_sha256="a" * 64, config_sha256="b" * 64)
    guard = FrozenBaseDelta(network, provenance)
    optimizer = torch.optim.AdamW([p for p in network.parameters() if p.requires_grad], lr=0.01)
    counts = (observed_counts([data["label_mask"]]), torch.tensor(1.0))

    def update(n, o):
        o.zero_grad(set_to_none=True)
        gradient_step(n, data, counts)
        o.step()

    update(network, optimizer)
    binding = {
        "world_size": 1,
        "config": {"relation": POLICY, "checkpoint_format": "frozen_base_delta"},
    }
    path = tmp_path / "delta.pt"
    save_checkpoint(
        path,
        {
            **guard.checkpoint_model(network),
            "binding": binding,
            "optimizer_boundary": True,
            "optimizer": optimizer.state_dict(),
            "rng_by_rank": [capture_rng(torch.device("cpu"))],
        },
        min_free_gib=0,
    )
    update(network, optimizer)
    expected = copy.deepcopy(network.state_dict())
    expected_optimizer = copy.deepcopy(optimizer.state_dict())
    restored = model()
    restored_guard = FrozenBaseDelta(restored, provenance)
    restored_optimizer = torch.optim.AdamW(
        [p for p in restored.parameters() if p.requires_grad], lr=0.01
    )
    state = load_checkpoint(path, binding)
    restored_guard.restore(restored, state)
    restored_optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng_by_rank"][0], torch.device("cpu"))
    update(restored, restored_optimizer)
    for name, tensor in expected.items():
        torch.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)
    for key, row in expected_optimizer["state"].items():
        for field, value in row.items():
            torch.testing.assert_close(
                restored_optimizer.state_dict()["state"][key][field], value, rtol=0, atol=0
            )
    assert torch.equal(restored.encoder.linear.weight, original)
    assert any(name.startswith("relation_head.") for name in state["model"])
    changed = copy.deepcopy(binding)
    changed["config"]["relation"]["target_mode"] = "shuffled"
    with pytest.raises(ValueError):
        load_checkpoint(path, changed)


@pytest.mark.parametrize(
    "field,value",
    [
        ("weight", True),
        ("weight", float("nan")),
        ("weight", -1),
        ("max_pairs", 0),
        ("max_pairs", 513),
        ("target_mode", {}),
        ("hidden_dim", 0),
        ("head_seed", True),
    ],
)
def test_invalid_relation_policies_never_enter_training(field, value):
    with pytest.raises(ValueError):
        TrainConfig(diagnostic_study_limit=4, relation={**POLICY, field: value}).validate()


def test_relation_seed_bound_is_checked_before_loading_data():
    with pytest.raises(ValueError, match="seed below"):
        TrainConfig(seed=2**63, relation=POLICY).validate()


@pytest.mark.parametrize("suffix", ["config", "pilot-config"])
def test_prepared_actual_recipes_isolate_only_weight_and_geometry_targets(suffix):
    root = Path(__file__).resolve().parents[1]
    folder = root / "initiative/experiments/competitive-v1/slice-relations-v1"
    configs = {
        name: json.loads((folder / f"{name}-{suffix}.json").read_text())
        for name in ("zero", "physical", "shuffled")
    }
    for config in configs.values():
        TrainConfig(**config).validate()
    reference = copy.deepcopy(configs["physical"])
    reference.pop("relation")
    for config in configs.values():
        assert {k: v for k, v in config.items() if k != "relation"} == reference
        assert {
            k: v for k, v in config["relation"].items() if k not in {"weight", "target_mode"}
        } == {
            "max_pairs": 32,
            "hidden_dim": 128,
            "head_seed": 20261001,
        }
    assert configs["zero"]["relation"]["weight"] == 0
    assert (
        configs["physical"]["relation"]["weight"]
        == configs["shuffled"]["relation"]["weight"]
        == 0.05
    )
    assert configs["physical"]["relation"]["target_mode"] == "physical"
    assert configs["shuffled"]["relation"]["target_mode"] == "shuffled"
    assert reference["slice_sampling"] == "uniform"
    assert reference["encoder"]["adapter"]["rank"] == 32
    assert reference["encoder"]["source_revision"] == "ea8dc2863c51be0a264bab82070e3e8836b02d51"
