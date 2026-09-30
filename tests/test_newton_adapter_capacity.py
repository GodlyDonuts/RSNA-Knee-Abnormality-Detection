import copy

import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from rsna.competitive.adapters import install_adapters
from scripts.newton.verify_adapter_pilot import adapter_parameter_count


def graph(rank):
    config = DINOv3ViTConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_register_tokens=4,
    )
    policy = dict(
        rank=rank, alpha=8.0, dropout=0.05, ffn_last_layers=12, gradient_checkpointing=True
    )
    with torch.device("meta"):
        encoder = DINOv3ViTModel(config)
        provenance = {"adapter": install_adapters(encoder, policy)}
    return encoder, provenance


@pytest.mark.parametrize("rank,expected", [(32, 10_223_616), (64, 20_447_232)])
def test_pilot_verifier_accepts_actual_production_capacity(rank, expected):
    encoder, provenance = graph(rank)
    assert adapter_parameter_count(encoder, provenance) == expected


def test_pilot_verifier_rejects_false_capacity_or_projection_membership():
    encoder, provenance = graph(64)
    wrong = copy.deepcopy(provenance)
    wrong["adapter"]["trainable_parameters"] = 10_223_616
    with pytest.raises(ValueError, match="capacity differs"):
        adapter_parameter_count(encoder, wrong)
    wrong = copy.deepcopy(provenance)
    wrong["adapter"]["target_modules"].pop()
    with pytest.raises(ValueError, match="capacity differs"):
        adapter_parameter_count(encoder, wrong)
    next(
        value for name, value in encoder.named_parameters() if name.endswith(".lora_A")
    ).requires_grad_(False)
    with pytest.raises(ValueError, match="capacity differs"):
        adapter_parameter_count(encoder, provenance)


def test_pilot_verifier_rejects_trainable_pretrained_tensor():
    encoder, provenance = graph(64)
    next(
        value
        for name, value in encoder.named_parameters()
        if not name.endswith((".lora_A", ".lora_B"))
    ).requires_grad_(True)
    with pytest.raises(ValueError, match="Only declared residual"):
        adapter_parameter_count(encoder, provenance)
