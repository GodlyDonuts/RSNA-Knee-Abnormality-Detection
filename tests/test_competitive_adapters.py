import copy
import hashlib
import json

import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from rsna.competitive import backbones
from rsna.competitive.adapters import (
    ResidualLinear,
    adapter_provenance,
    configure_encoder_training,
    install_adapters,
    validate_adapter,
)
from rsna.competitive.checkpoint import capture_rng, restore_rng
from rsna.competitive.train import TrainConfig, learning_rate_multiplier

POLICY = dict(rank=4, alpha=8.0, dropout=0.05, ffn_last_layers=1, gradient_checkpointing=True)


def small_model():
    return DINOv3ViTModel(
        DINOv3ViTConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            patch_size=16,
            num_register_tokens=4,
            pos_embed_rescale=None,
        )
    )


def test_zero_residual_exact_and_gradients_change_only_adapters():
    torch.manual_seed(12)
    base = small_model().eval()
    state = {name: tensor.clone() for name, tensor in base.state_dict().items()}
    images = torch.rand(2, 3, 32, 32)
    with torch.no_grad():
        expected = base(pixel_values=images).pooler_output
    record = install_adapters(base, POLICY)
    encoder = backbones._GrayEncoder(base, 16)
    configure_encoder_training(encoder, True)
    base.eval()
    with torch.no_grad():
        torch.testing.assert_close(
            base(pixel_values=images).pooler_output, expected, rtol=0, atol=0
        )
    assert (
        sum(p.numel() for p in base.parameters() if p.requires_grad)
        == record["trainable_parameters"]
    )
    optimizer = torch.optim.AdamW([p for p in base.parameters() if p.requires_grad], lr=0.01)
    target = torch.randn(2, 32)
    for _ in range(3):
        base.train()
        optimizer.zero_grad()
        (base(pixel_values=images).pooler_output - target).square().mean().backward()
        assert all(p.grad is None for name, p in base.named_parameters() if name in state)
        assert all(
            p.grad is not None and torch.isfinite(p.grad).all()
            for p in base.parameters()
            if p.requires_grad
        )
        optimizer.step()
    for name, tensor in state.items():
        assert torch.equal(base.state_dict()[name], tensor), name
    assert all(p.abs().sum() > 0 for name, p in base.named_parameters() if name.endswith("lora_B"))
    assert any(
        p.grad.abs().sum() > 0 for name, p in base.named_parameters() if name.endswith("lora_A")
    )
    with pytest.raises(ValueError, match="original encoder"):
        configure_encoder_training(encoder, False)


def test_a32_full_production_graph_and_count():
    config = DINOv3ViTConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_register_tokens=4,
    )
    policy = dict(POLICY, rank=32, ffn_last_layers=12)
    with torch.device("meta"):
        model = DINOv3ViTModel(config)
        record = install_adapters(model, policy)
    assert record == adapter_provenance(policy, config.to_dict())
    assert record["trainable_parameters"] == 10_223_616
    assert len(record["target_modules"]) == 120
    assert all(
        name.endswith((".lora_A", ".lora_B"))
        for name, p in model.named_parameters()
        if p.requires_grad
    )
    with pytest.raises(ValueError, match="already adapted"):
        install_adapters(model, policy)


def test_checkpointed_backprop_matches_direct_backprop():
    torch.manual_seed(31)
    direct = small_model()
    install_adapters(direct, dict(POLICY, dropout=0.0, gradient_checkpointing=False))
    checkpointed = copy.deepcopy(direct)
    checkpointed.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs={"use_reentrant": False}
    )
    images = torch.rand(1, 3, 32, 32)
    for model in [direct, checkpointed]:
        model(pixel_values=images).pooler_output[:, 0].sum().backward()
    for (n1, p1), (n2, p2) in zip(
        direct.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        assert n1 == n2
        if p1.requires_grad:
            torch.testing.assert_close(p1.grad, p2.grad, rtol=0, atol=0)


def test_adapter_optimizer_and_dropout_resume_exactly():
    torch.manual_seed(45)
    model = small_model()
    install_adapters(model, POLICY)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.01)
    images = torch.rand(1, 3, 32, 32)

    def update(net, opt):
        net.train()
        opt.zero_grad()
        net(pixel_values=images).pooler_output[:, 0].square().sum().backward()
        opt.step()

    update(model, optimizer)
    saved_model = copy.deepcopy(model.state_dict())
    saved_optimizer = copy.deepcopy(optimizer.state_dict())
    rng = capture_rng(torch.device("cpu"))
    update(model, optimizer)
    expected = copy.deepcopy(model.state_dict())
    restored = small_model()
    install_adapters(restored, POLICY)
    restored.load_state_dict(saved_model, strict=True)
    restored_optimizer = torch.optim.AdamW(
        [p for p in restored.parameters() if p.requires_grad], lr=0.01
    )
    restored_optimizer.load_state_dict(saved_optimizer)
    restore_rng(rng, torch.device("cpu"))
    update(restored, restored_optimizer)
    assert all(torch.equal(expected[name], value) for name, value in restored.state_dict().items())


def test_adapter_factory_reload_without_pretrained_file(tmp_path, monkeypatch):
    # Real HF graph reduced only in this test; production still enforces full L dimensions.
    config = small_model().config.to_dict()
    monkeypatch.setattr(backbones, "_validate_architecture", lambda *args: None)
    monkeypatch.setattr(backbones, "_encoder_dimensions", lambda *args: (16, 32))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    weights_path = tmp_path / "weights.pt"
    torch.save(small_model().state_dict(), weights_path)
    spec = dict(
        architecture="dinov3_l16",
        config_path=str(config_path),
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        weights_path=str(weights_path),
        weights_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        source_id="synthetic/test",
        source_revision="b" * 40,
        license="test-only",
        state_dict_format="pytorch",
        transformers_version="5.6.1",
        adapter=POLICY,
    )
    encoder, dim, provenance = backbones.create_encoder(spec)
    assert dim == 32
    configure_encoder_training(encoder, True)
    with torch.no_grad():
        for module in encoder.modules():
            if isinstance(module, ResidualLinear):
                module.lora_B.normal_(std=0.01)
    encoder.eval()
    state = {name: tensor.clone() for name, tensor in encoder.state_dict().items()}
    weights_path.unlink()
    restored, dim2, returned = backbones.create_encoder_from_training_checkpoint(
        provenance,
        config_path,
        encoder_state=state,
    )
    images = torch.rand(1, 1, 32, 32)
    with torch.no_grad():
        torch.testing.assert_close(restored(images), encoder(images), rtol=0, atol=0)
    assert returned == provenance and dim2 == dim
    assert not any(p.requires_grad for p in restored.parameters())
    missing = dict(state)
    del missing[next(name for name in missing if name.endswith("lora_B"))]
    with pytest.raises(RuntimeError, match="Missing key"):
        backbones.create_encoder_from_training_checkpoint(
            provenance, config_path, encoder_state=missing
        )


@pytest.mark.parametrize(
    "change",
    [
        dict(rank=True),
        dict(alpha=float("nan")),
        dict(dropout=1),
        dict(ffn_last_layers=25),
        dict(unknown=True),
    ],
)
def test_invalid_adapter_policy_is_rejected(change):
    with pytest.raises(ValueError):
        validate_adapter(dict(POLICY, **change), 24)


def test_cosine_schedule_and_resume_use_absolute_update():
    values = [learning_rate_multiplier(i, 20, 2, "cosine") for i in range(20)]
    assert values[:3] == [0.5, 1.0, 1.0]
    assert values[-1] == 0
    assert values[2:] == sorted(values[2:], reverse=True)
    assert [learning_rate_multiplier(i, 20, 2, "cosine") for i in range(8, 20)] == values[8:]
    assert learning_rate_multiplier(19, 20, 2, "constant") == 1
    config = TrainConfig(
        encoder={"architecture": "dinov3_l16", "adapter": POLICY}, lr_schedule="cosine"
    )
    config.validate()
    with pytest.raises(ValueError, match="frozen"):
        TrainConfig(encoder=config.encoder, freeze_encoder=False).validate()
