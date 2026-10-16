import copy

import pytest
import torch
from torch import nn

from rsna.competitive.licensed_full_bag import CheckpointedEncode
from rsna.competitive.licensed_parent_train import train_with_frozen_bn


class Base(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Flatten(), nn.Linear(48, 16), nn.BatchNorm1d(16), nn.Dropout(0.3), nn.Linear(16, 8)
        )
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(8, 12))
        self.reference_calls = 0

    def encode(self, x):
        self.reference_calls += 1
        return torch.cat([self.backbone(v) for v in x.flatten(0, 1).split(8)]).view(*x.shape[:2], 8)

    def forward(self, x):
        # Every window participates in the same study objective.
        return self.head(self.encode(x).mean(1))


class WholeStudy(CheckpointedEncode, Base):
    pass


def test_full94_forward_gradient_buffers_and_RNG_exact_vs_same_graph_without_checkpoint():
    torch.set_num_threads(1)
    torch.manual_seed(13)
    original = Base()
    whole = WholeStudy()
    whole.load_state_dict(original.state_dict(), strict=True)
    initial = copy.deepcopy(original.state_dict())
    x = torch.randn(2, 94, 3, 4, 4)
    train_with_frozen_bn(original)
    train_with_frozen_bn(whole)
    torch.manual_seed(90)
    a = original(x)
    a.square().mean().backward()
    rng = torch.get_rng_state().clone()
    torch.manual_seed(90)
    b = whole(x)
    b.square().mean().backward()
    assert torch.equal(a, b)
    assert torch.equal(rng, torch.get_rng_state())
    for (name, p), (other, q) in zip(
        original.named_parameters(), whole.named_parameters(), strict=True
    ):
        assert name == other and p.grad is not None and torch.equal(p.grad, q.grad)
    for name, value in whole.named_buffers():
        assert torch.equal(value, initial[name])
    assert whole.reference_calls == 0
    assert whole.encode(x).shape == (2, 94, 8)


def test_no_grad_and_inference_delegate_to_original_parent_encoder():
    m = WholeStudy()
    x = torch.randn(1, 94, 3, 4, 4)
    m.eval()
    m(x)
    assert m.reference_calls == 1
    m.train()
    with torch.no_grad():
        m(x)
    assert m.reference_calls == 2


def test_checkpointed_training_rejects_mutable_BN_buffers_instead_of_double_updating():
    model = WholeStudy().train()
    initial = copy.deepcopy(model.state_dict())
    with pytest.raises(ValueError, match="Freeze parent BN"):
        model(torch.randn(1, 94, 3, 4, 4))
    assert all(torch.equal(value, initial[name]) for name, value in model.state_dict().items())
