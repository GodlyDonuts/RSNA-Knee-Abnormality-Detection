import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from rsna.competitive.checkpoint import capture_rng, load_checkpoint, restore_rng, save_checkpoint
from rsna.competitive.delta_checkpoint import FrozenBaseDelta, restore_checkpoint_model
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.train import TrainConfig

PROVENANCE = {"pretrained": True, "weights_sha256": "a" * 64, "config_sha256": "b" * 64}


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(1, 8)
        self.base.requires_grad_(False)
        self.register_buffer("base_scale", torch.tensor(1.0))
        self.adapter = nn.Linear(1, 8, bias=False)
        self.dropout = nn.Dropout(0.2)

    def forward(self, value):
        value = value.mean((-1, -2))
        return self.base(value) * self.base_scale + self.adapter(self.dropout(value))


def model_and_optimizer(context):
    torch.manual_seed(37)
    random.seed(37)
    np.random.seed(37)
    model = CompetitiveStudyNet(Encoder(), 8, hidden_dim=8, slice_context=context)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
    return model, optimizer


def update(model, optimizer):
    optimizer.zero_grad()
    # Exercise all three CPU RNG streams, dropout, unequal coverage and context.
    images = torch.rand(1, 3, 4, 1, 16, 16) * (random.random() + np.random.rand()) / 2
    features = torch.eye(3, 5).unsqueeze(0)
    series_mask = torch.ones(1, 3, dtype=torch.bool)
    slice_mask = torch.ones(1, 3, 4, dtype=torch.bool)
    slice_mask[0, 1, -1] = False
    positions = torch.zeros(1, 3, 4, 2)
    loss = model(images, features, series_mask, slice_mask, positions).square().mean()
    loss.backward()
    optimizer.step()


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b)
        for left, right in zip(a, b, strict=True):
            equal(left, right)
    else:
        assert a == b


@pytest.mark.parametrize("context", ["none", "neighbor3"])
def test_delta_disk_recovery_preserves_model_optimizer_all_rng_and_next_updates(tmp_path, context):
    model, optimizer = model_and_optimizer(context)
    guard = FrozenBaseDelta(model, PROVENANCE)
    for _ in range(3):
        update(model, optimizer)
    binding = {"config": {"checkpoint_format": "frozen_base_delta"}, "world_size": 1}
    state = {
        **guard.checkpoint_model(model),
        "binding": binding,
        "optimizer_boundary": True,
        "optimizer": optimizer.state_dict(),
        "rng_by_rank": [capture_rng(torch.device("cpu"))],
        "epoch": 0,
        "next_update": 3,
        "step": 3,
    }
    save_checkpoint(tmp_path / "delta.pt", state, min_free_gib=0)
    assert not any(name in state["model"] for name in guard.frozen)
    for _ in range(4):
        update(model, optimizer)
    expected_rng = capture_rng(torch.device("cpu"))
    checkpoint = load_checkpoint(tmp_path / "delta.pt", binding)
    restored, restored_optimizer = model_and_optimizer(context)
    restored_guard = FrozenBaseDelta(restored, PROVENANCE)
    restore_checkpoint_model(restored, checkpoint, restored_guard)
    restored_optimizer.load_state_dict(checkpoint["optimizer"])
    restore_rng(checkpoint["rng_by_rank"][0], torch.device("cpu"))
    for _ in range(4):
        update(restored, restored_optimizer)
    equal(model.state_dict(), restored.state_dict())
    equal(optimizer.state_dict(), restored_optimizer.state_dict())
    equal(expected_rng, capture_rng(torch.device("cpu")))
    model.eval()
    restored.eval()
    torch.manual_seed(1)
    inputs = (
        torch.rand(1, 3, 4, 1, 16, 16),
        torch.eye(3, 5).unsqueeze(0),
        torch.ones(1, 3, dtype=torch.bool),
        torch.ones(1, 3, 4, dtype=torch.bool),
        torch.zeros(1, 3, 4, 2),
    )
    assert torch.equal(model(*inputs), restored(*inputs))


@pytest.mark.parametrize("change", ["base", "buffer", "partition"])
def test_refuse_omitting_a_mutated_frozen_base_or_changed_parameter_partition(change):
    model, _ = model_and_optimizer("none")
    guard = FrozenBaseDelta(model, PROVENANCE)
    with torch.no_grad():
        if change == "base":
            model.encoder.base.weight[0, 0] += 0.01
        elif change == "buffer":
            model.encoder.base_scale.fill_(2)
        else:
            model.encoder.base.requires_grad_(True)
    with pytest.raises(ValueError, match="changed"):
        guard.checkpoint_model(model)


@pytest.mark.parametrize(
    "change", ["missing", "extra", "shape", "dtype", "nan", "base_sha", "schema", "provenance"]
)
def test_delta_corruption_rejected_before_touching_the_destination_model(change):
    model, _ = model_and_optimizer("none")
    guard = FrozenBaseDelta(model, PROVENANCE)
    checkpoint = copy.deepcopy(guard.checkpoint_model(model))
    name = guard.trainable[0]
    if change == "missing":
        del checkpoint["model"][name]
    elif change == "extra":
        checkpoint["model"]["encoder.base.weight"] = model.encoder.base.weight.detach().clone()
    elif change == "shape":
        checkpoint["model"][name] = torch.zeros(2)
    elif change == "dtype":
        checkpoint["model"][name] = checkpoint["model"][name].double()
    elif change == "nan":
        checkpoint["model"][name].fill_(float("nan"))
    elif change == "base_sha":
        checkpoint["model_storage"]["frozen_tensors"][guard.frozen[0]]["sha256"] = "0" * 64
    elif change == "schema":
        checkpoint["schema_version"] = 1
    else:
        checkpoint["model_storage"]["encoder_provenance"]["weights_sha256"] = "0" * 64
    before = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError):
        guard.restore(model, checkpoint)
    equal(before, model.state_dict())


def test_wrong_reconstructed_base_and_missing_base_guard_rejected():
    model, _ = model_and_optimizer("none")
    checkpoint = FrozenBaseDelta(model, PROVENANCE).checkpoint_model(model)
    with pytest.raises(ValueError, match="verified frozen base"):
        restore_checkpoint_model(model, checkpoint, None)
    with torch.no_grad():
        model.encoder.base.weight.add_(0.1)
    with pytest.raises(ValueError, match="base"):
        FrozenBaseDelta(model, PROVENANCE).restore(model, checkpoint)


def test_full_checkpoint_remains_strict_and_delta_config_is_opt_in():
    model, _ = model_and_optimizer("none")
    state = copy.deepcopy(model.state_dict())
    restore_checkpoint_model(model, {"schema_version": 1, "model": state}, None)
    assert TrainConfig().checkpoint_format == "full"
    with pytest.raises(ValueError, match="frozen pretrained base"):
        TrainConfig(checkpoint_format="frozen_base_delta", diagnostic_study_limit=2).validate()


def test_recovery_envelope_cannot_silently_change_checkpoint_format(tmp_path):
    binding = {"config": {"checkpoint_format": "frozen_base_delta"}, "world_size": 1}
    save_checkpoint(
        tmp_path / "bad.pt",
        {"schema_version": 1, "binding": binding, "optimizer_boundary": True, "rng_by_rank": [{}]},
        min_free_gib=0,
    )
    with pytest.raises(ValueError, match="differs"):
        load_checkpoint(tmp_path / "bad.pt", binding)
