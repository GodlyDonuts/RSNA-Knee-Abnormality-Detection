from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rsna.competitive import backbones
from rsna.competitive.backbones import create_encoder
from rsna.competitive.model import CompetitiveStudyNet


class RecordingEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 8)
        self.dropout = nn.Dropout(0.5)
        self.seen = []

    def forward(self, x):
        assert torch.isfinite(x).all()
        self.seen.append(x.detach().clone())
        return self.dropout(self.linear(x.mean(dim=(2, 3))))


def inputs():
    torch.manual_seed(12)
    images = torch.rand(2, 2, 3, 1, 16, 16)
    sm = torch.tensor([[True, True], [True, False]])
    km = torch.tensor(
        [[[True, True, False], [True, False, False]], [[True, True, True], [False, False, False]]]
    )
    return images, torch.rand(2, 2, 5), sm, km, torch.zeros(2, 2, 3, 2)


@pytest.mark.parametrize("pooling", ["attention", "max"])
def test_padding_never_encoded_and_does_not_change_predictions(pooling):
    encoder = RecordingEncoder()
    model = CompetitiveStudyNet(
        encoder, 8, pooling=pooling, hidden_dim=12, encode_chunk_size=2
    ).eval()
    batch = inputs()
    baseline = model(*batch)
    assert baseline.shape == (2, 12)
    assert sum(len(chunk) for chunk in encoder.seen) == 6
    assert max(map(len, encoder.seen)) == 2
    images = torch.full((2, 4, 5, 1, 16, 16), torch.nan)
    features = torch.full((2, 4, 5), torch.nan)
    positions = torch.full((2, 4, 5, 2), torch.nan)
    sm = torch.zeros(2, 4, dtype=torch.bool)
    km = torch.zeros(2, 4, 5, dtype=torch.bool)
    images[:, :2, :3], features[:, :2], positions[:, :2, :3] = batch[0], batch[1], batch[4]
    sm[:, :2], km[:, :2, :3] = batch[2], batch[3]
    images[~km], features[~sm], positions[~km] = torch.nan, torch.nan, torch.nan
    assert torch.allclose(model(images, features, sm, km, positions), baseline, atol=1e-6)


@pytest.mark.parametrize("pooling", ["attention", "max"])
def test_trainable_gradients_and_masked_image_gradients(pooling):
    model = CompetitiveStudyNet(RecordingEncoder(), 8, pooling=pooling, hidden_dim=12)
    batch = list(inputs())
    batch[0].requires_grad_()
    model(*batch).square().sum().backward()
    assert torch.isfinite(batch[0].grad).all()
    assert torch.count_nonzero(batch[0].grad[~batch[3]]) == 0
    assert torch.count_nonzero(batch[0].grad[batch[3]]) > 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_one_valid_slice_frozen_encoder_eval_and_resume():
    encoder, dim, provenance = create_encoder(
        {"architecture": "diagnostic_tiny", "embedding_dim": 16}
    )
    assert provenance["diagnostic_only"] and not provenance["pretrained"]
    encoder.requires_grad_(False)
    model = CompetitiveStudyNet(encoder, dim).train()
    assert not model.encoder.training and model.projection.training
    batch = (
        torch.rand(1, 1, 1, 1, 16, 16),
        torch.zeros(1, 1, 5),
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1, 1, 1, dtype=torch.bool),
        torch.zeros(1, 1, 1, 2),
    )
    result = model(*batch)
    assert torch.isfinite(result).all()
    restored = CompetitiveStudyNet(create_encoder({"architecture": "diagnostic_tiny"})[0], 16)
    restored.load_state_dict(model.state_dict(), strict=True)
    restored.encoder.requires_grad_(False)
    restored.train()
    assert torch.equal(result, restored(*batch))
    bad = dict(model.state_dict())
    bad.pop("output_bias")
    with pytest.raises(RuntimeError):
        restored.load_state_dict(bad, strict=True)


@pytest.mark.parametrize(
    "change",
    [
        "empty_study",
        "empty_series",
        "orphan_slice",
        "nan_valid",
        "image_range",
        "unknown_position",
        "wrong_mask_type",
        "nonfinite_feature",
    ],
)
def test_invalid_batch_fails_before_encoder(change):
    encoder = RecordingEncoder()
    model = CompetitiveStudyNet(encoder, 8)
    batch = list(inputs())
    if change == "empty_study":
        batch[2][0] = False
        batch[3][0] = False
    elif change == "empty_series":
        batch[3][0, 0] = False
    elif change == "orphan_slice":
        batch[3][1, 1, 0] = True
    elif change == "nan_valid":
        batch[0][0, 0, 0] = torch.nan
    elif change == "image_range":
        batch[0][0, 0, 0] = 1.2
    elif change == "unknown_position":
        batch[4][0, 0, 0, 0] = 2
    elif change == "wrong_mask_type":
        batch[3] = batch[3].float()
    else:
        batch[1][0, 0, 0] = torch.nan
    with pytest.raises(ValueError):
        model(*batch)
    assert not encoder.seen


class FakeHF(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 768)
        self.last_input = None

    def forward(self, pixel_values):
        self.last_input = pixel_values.detach().clone()
        return SimpleNamespace(pooler_output=self.projection(pixel_values.mean((2, 3))))


def local_spec(tmp_path: Path, monkeypatch):
    config = {
        "model_type": "dinov2_with_registers",
        "hidden_size": 768,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "patch_size": 14,
        "num_register_tokens": 4,
        "num_channels": 3,
    }
    config_path, weights_path = tmp_path / "config.json", tmp_path / "weights.pt"
    config_path.write_text(json.dumps(config))
    model = FakeHF()
    torch.save(model.state_dict(), weights_path)
    monkeypatch.setattr(backbones.importlib.metadata, "version", lambda _: "5.6.1")
    monkeypatch.setattr(backbones, "_construct", lambda *args: model)
    return dict(
        architecture="dinov2_b14_reg",
        config_path=str(config_path),
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        weights_path=str(weights_path),
        weights_sha256=hashlib.sha256(weights_path.read_bytes()).hexdigest(),
        transformers_version="5.6.1",
        source_id="facebook/test",
        source_revision="a" * 40,
        license="apache-2.0",
        state_dict_format="pytorch",
    ), model


def test_offline_factory_normalization_provenance_and_strict_state(tmp_path, monkeypatch):
    spec, base = local_spec(tmp_path, monkeypatch)
    encoder, dim, provenance = create_encoder(spec)
    assert (
        dim == 768
        and "path" not in json.dumps(provenance)
        and str(tmp_path) not in json.dumps(provenance)
    )
    encoder(torch.full((1, 1, 28, 28), 0.5))
    expected = (torch.tensor([0.5, 0.5, 0.5]) - torch.tensor([0.485, 0.456, 0.406])) / torch.tensor(
        [0.229, 0.224, 0.225]
    )
    assert torch.allclose(base.last_input[0, :, 0, 0], expected)
    with pytest.raises(ValueError, match="divisible"):
        encoder(torch.zeros(1, 1, 25, 25))
    state = base.state_dict()
    state.pop("projection.bias")
    torch.save(state, spec["weights_path"])
    spec["weights_sha256"] = backbones._sha256(Path(spec["weights_path"]))
    with pytest.raises(RuntimeError, match="Missing key"):
        create_encoder(spec)


@pytest.mark.parametrize(
    "change",
    [
        "weight_bytes",
        "config_bytes",
        "wrong_architecture",
        "wrong_version",
        "mutable_revision",
        "wrapped_state",
        "nonfinite_state",
        "remote_identity",
    ],
)
def test_factory_rejects_mismatch(tmp_path, monkeypatch, change):
    spec, base = local_spec(tmp_path, monkeypatch)
    if change in {"weight_bytes", "config_bytes"}:
        Path(spec["weights_path" if change == "weight_bytes" else "config_path"]).write_bytes(
            b"bad"
        )
    elif change == "wrong_architecture":
        spec["architecture"] = "dinov3_b16"
    elif change == "wrong_version":
        spec["transformers_version"] = "5.5.0"
    elif change == "mutable_revision":
        spec["source_revision"] = "main"
    elif change == "remote_identity":
        spec["source_id"] = "https://host/secret"
    else:
        state = base.state_dict()
        if change == "wrapped_state":
            state = {"state_dict": state}
        else:
            state["projection.bias"][0] = torch.nan
        torch.save(state, spec["weights_path"])
        spec["weights_sha256"] = backbones._sha256(Path(spec["weights_path"]))
    with pytest.raises(ValueError):
        create_encoder(spec)


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None, reason="optional transformers"
)
@pytest.mark.parametrize("architecture", ["dinov2_b14_reg", "dinov3_b16", "convnext_tiny"])
def test_real_transformers_small_random_architecture_forward(architecture):
    # Small randomly initialized topology probes, not pretrained-model acceptance.
    if architecture == "dinov2_b14_reg":
        config = dict(
            hidden_size=16,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=28,
            patch_size=14,
            num_register_tokens=4,
        )
        side, multiple, dim = 28, 14, 16
    elif architecture == "dinov3_b16":
        config = dict(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            image_size=32,
            patch_size=16,
            num_register_tokens=4,
        )
        side, multiple, dim = 32, 16, 16
    else:
        config = dict(hidden_sizes=[4, 8, 16, 32], depths=[1, 1, 1, 1], num_labels=2)
        side, multiple, dim = 32, 32, 32
    base = backbones._construct(architecture, config, "ConvNextForImageClassification")
    if architecture == "convnext_tiny":
        base = base.convnext
    wrapper = backbones._GrayEncoder(base, multiple).eval()
    with torch.no_grad():
        output = wrapper(torch.rand(1, 1, side, side))
    assert output.shape == (1, dim) and torch.isfinite(output).all()
    wrapper.requires_grad_(False)
    wrapper.requires_grad_(True)
    wrapper(torch.rand(1, 1, side, side)).square().sum().backward()
    for name, parameter in wrapper.named_parameters():
        if name == "backbone.embeddings.mask_token":
            assert not parameter.requires_grad and parameter.grad is None
        else:
            assert parameter.requires_grad and parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def test_safetensors_local_loading(tmp_path, monkeypatch):
    safetensors = pytest.importorskip("safetensors.torch")
    spec, base = local_spec(tmp_path, monkeypatch)
    path = tmp_path / "model.safetensors"
    safetensors.save_file(base.state_dict(), str(path))
    spec.update(
        weights_path=str(path),
        weights_sha256=backbones._sha256(path),
        state_dict_format="safetensors",
    )
    encoder, dim, _ = create_encoder(spec)
    assert encoder(torch.zeros(1, 1, 28, 28)).shape == (1, dim)


def _large_config():
    # Source-derived topology fixture, not a copy of the gated HF config file.
    return dict(
        model_type="dinov3_vit",
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        patch_size=16,
        num_register_tokens=4,
        num_channels=3,
        image_size=224,
        hidden_act="gelu",
        use_gated_mlp=False,
        rope_theta=100.0,
        query_bias=True,
        key_bias=False,
        value_bias=True,
        proj_bias=True,
        mlp_bias=True,
        layer_norm_eps=1e-5,
        layerscale_value=1e-5,
        pos_embed_shift=None,
        pos_embed_jitter=None,
        pos_embed_rescale=2.0,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("hidden_size", 768),
        ("intermediate_size", 6144),
        ("num_hidden_layers", 12),
        ("num_attention_heads", 12),
        ("num_register_tokens", 0),
        ("patch_size", 14),
        ("use_gated_mlp", True),
        ("hidden_act", "silu"),
        ("rope_theta", 10000.0),
        ("key_bias", True),
        ("query_bias", False),
        ("value_bias", False),
        ("proj_bias", False),
        ("mlp_bias", False),
        ("layer_norm_eps", 1e-6),
        ("pos_embed_rescale", None),
        ("pos_embed_shift", 0.1),
        ("pos_embed_jitter", 2),
        ("untie_global_and_local_cls_norm", True),
        ("layerscale_value", float("nan")),
    ],
)
def test_large_rejects_other_variant_or_semantic_mismatch(field, value):
    config = _large_config()
    backbones._validate_architecture("dinov3_l16", config)
    config[field] = value
    with pytest.raises(ValueError):
        backbones._validate_architecture("dinov3_l16", config)


def test_large_factory_1024_dimensions_strict_state_and_source_binding(tmp_path, monkeypatch):
    spec, _ = local_spec(tmp_path, monkeypatch)
    config = _large_config()
    config_path = Path(spec["config_path"])
    config_path.write_text(json.dumps(config))
    base = FakeHF()
    base.projection = nn.Linear(3, 1024)
    torch.save(base.state_dict(), spec["weights_path"])
    constructed = []

    def construct(architecture, actual_config, model_class):
        constructed.append((architecture, actual_config))
        return base

    monkeypatch.setattr(backbones, "_construct", construct)
    spec.update(
        architecture="dinov3_l16",
        config_sha256=backbones._sha256(config_path),
        weights_sha256=backbones._sha256(Path(spec["weights_path"])),
        source_id="facebook/dinov3-vitl16-pretrain-lvd1689m",
        license="dinov3",
    )
    encoder, dim, provenance = create_encoder(spec)
    assert dim == 1024 and provenance["embedding_dim"] == 1024
    assert constructed == [("dinov3_l16", config)]
    assert encoder(torch.rand(1, 1, 32, 32)).shape == (1, 1024)
    assert provenance["source_revision"] == spec["source_revision"]
    assert provenance["weights_sha256"] == spec["weights_sha256"]
    bad = base.state_dict()
    bad["projection.weight"] = bad["projection.weight"][:768]
    torch.save(bad, spec["weights_path"])
    spec["weights_sha256"] = backbones._sha256(Path(spec["weights_path"]))
    with pytest.raises(RuntimeError, match="size mismatch"):
        create_encoder(spec)


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None, reason="optional transformers"
)
def test_large_complete_meta_topology_without_weight_allocation():
    config = _large_config()
    backbones._validate_architecture("dinov3_l16", config)
    with torch.device("meta"):
        model = backbones._construct("dinov3_l16", config, "ConvNextForImageClassification")
    state = model.state_dict()
    assert all(value.device.type == "meta" for value in state.values())
    assert state["model.layer.23.mlp.up_proj.weight"].shape == (4096, 1024)
    assert state["model.layer.23.attention.q_proj.weight"].shape == (1024, 1024)
    assert state["embeddings.register_tokens"].shape == (1, 4, 1024)
    assert state["norm.weight"].shape == (1024,)
    assert not any("k_proj.bias" in key for key in state)
    assert not any("model.layer.24." in key for key in state)


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None, reason="optional transformers"
)
def test_large_width_reduced_depth_real_forward_head_and_unused_mask_token():
    # Actual Large width/head count/MLP, one random block for bounded CPU memory.
    # This is neither a full 24-block execution nor pretrained-weight acceptance.
    config = _large_config()
    config.update(num_hidden_layers=1, image_size=32)
    base = backbones._construct("dinov3_l16", config, "ConvNextForImageClassification")
    encoder = backbones._GrayEncoder(base, 16)
    encoder.requires_grad_(False)
    encoder.requires_grad_(True)
    model = CompetitiveStudyNet(encoder, 1024, hidden_dim=16, encode_chunk_size=1).eval()
    batch = (
        torch.rand(1, 1, 1, 1, 32, 32),
        torch.zeros(1, 1, 5),
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1, 1, 1, dtype=torch.bool),
        torch.zeros(1, 1, 1, 2),
    )
    output = model(*batch)
    assert output.shape == (1, 12) and torch.isfinite(output).all()
    output.square().mean().backward()
    for name, parameter in model.named_parameters():
        if name == "encoder.backbone.embeddings.mask_token":
            assert not parameter.requires_grad and parameter.grad is None
        else:
            assert parameter.requires_grad and parameter.grad is not None, name
            assert torch.isfinite(parameter.grad).all(), name


def _trained_encoder_fixture(tmp_path, monkeypatch):
    spec, _ = local_spec(tmp_path, monkeypatch)
    monkeypatch.setattr(backbones, "_construct", lambda *args: FakeHF())
    encoder, dim, provenance = create_encoder(spec)
    with torch.no_grad():
        encoder.backbone.projection.bias.add_(0.4)
    encoder.requires_grad_(False)
    model = CompetitiveStudyNet(encoder, dim, hidden_dim=12).eval()
    checkpoint = {key: value.detach().clone() for key, value in model.state_dict().items()}
    trained = {
        key.removeprefix("encoder."): value
        for key, value in checkpoint.items()
        if key.startswith("encoder.")
    }
    Path(spec["weights_path"]).unlink()
    return Path(spec["config_path"]), provenance, model, checkpoint, trained


def test_checkpoint_encoder_restore_without_original_weights(tmp_path, monkeypatch):
    config_path, provenance, model, checkpoint, state = _trained_encoder_fixture(
        tmp_path, monkeypatch
    )

    def forbid_original_weight_loading(*args, **kwargs):
        raise AssertionError("No pretrained weight loading is allowed for checkpoint restore")

    monkeypatch.setattr(torch, "load", forbid_original_weight_loading)
    encoder, dimension, restored_provenance = backbones.create_encoder_from_training_checkpoint(
        provenance, config_path, encoder_state=state
    )
    assert dimension == 768 and restored_provenance == provenance
    assert not encoder.training and not any(p.requires_grad for p in encoder.parameters())
    restored = CompetitiveStudyNet(encoder, dimension, hidden_dim=12).eval()
    restored.load_state_dict(checkpoint, strict=True)
    batch = list(inputs())
    batch[0] = torch.rand(2, 2, 3, 1, 28, 28)
    with torch.no_grad():
        torch.testing.assert_close(restored(*batch), model(*batch), rtol=0, atol=0)
    missing_head = dict(checkpoint)
    missing_head.pop("output_bias")
    with pytest.raises(RuntimeError, match="Missing key"):
        restored.load_state_dict(missing_head, strict=True)


@pytest.mark.parametrize(
    "mutation",
    [
        "config_bytes",
        "config_missing",
        "architecture",
        "dimension",
        "float_dimension",
        "source_revision",
        "config_hash",
        "weight_hash",
        "version",
        "source_id",
        "license",
        "pooling",
        "input_contract",
        "pretrained",
        "numeric_pretrained",
        "model_class",
        "extra_path",
        "missing_field",
        "missing_state",
        "nonfinite_state",
        "extra_state",
        "changed_mean",
        "changed_std",
        "prefixed_state",
    ],
)
def test_checkpoint_encoder_restore_rejects_tampered_binding_or_state(
    tmp_path, monkeypatch, mutation
):
    path, provenance, _, _, state = _trained_encoder_fixture(tmp_path, monkeypatch)
    provenance = dict(provenance)
    state = dict(state)
    if mutation == "config_bytes":
        path.write_bytes(path.read_bytes() + b" ")
    elif mutation == "config_missing":
        path.unlink()
    elif mutation == "architecture":
        provenance["architecture"] = "unknown"
    elif mutation == "dimension":
        provenance["embedding_dim"] = 1024
    elif mutation == "float_dimension":
        provenance["embedding_dim"] = 768.0
    elif mutation == "source_revision":
        provenance["source_revision"] = "main"
    elif mutation in {"config_hash", "weight_hash"}:
        provenance["config_sha256" if mutation == "config_hash" else "weights_sha256"] = "invalid"
    elif mutation == "version":
        provenance["transformers_version"] = "5.5.0"
    elif mutation == "source_id":
        provenance["source_id"] = "https://untrusted/model"
    elif mutation == "license":
        provenance["license"] = ""
    elif mutation == "pooling":
        provenance["pooling"] = "mean_patch"
    elif mutation == "input_contract":
        provenance["input_contract"] = "other"
    elif mutation == "pretrained":
        provenance["pretrained"] = False
    elif mutation == "numeric_pretrained":
        provenance["pretrained"] = 1
    elif mutation == "model_class":
        provenance["model_class"] = "other"
    elif mutation == "extra_path":
        provenance["weights_path"] = "missing-unneeded-file.pt"
    elif mutation == "missing_field":
        provenance.pop("weights_sha256")
    elif mutation == "missing_state":
        state.pop("backbone.projection.weight")
    elif mutation == "nonfinite_state":
        state["backbone.projection.weight"][0, 0] = torch.nan
    elif mutation == "extra_state":
        state["unexpected"] = torch.ones(1)
    elif mutation == "changed_mean":
        state["mean"] = torch.zeros_like(state["mean"])
    elif mutation == "changed_std":
        state["std"] = state["std"].double()
    else:
        state = {"encoder." + key: value for key, value in state.items()}
    with pytest.raises((ValueError, RuntimeError)):
        backbones.create_encoder_from_training_checkpoint(provenance, path, encoder_state=state)


@pytest.mark.skipif(
    importlib.util.find_spec("transformers") is None, reason="optional transformers"
)
def test_real_random_dino_checkpoint_roundtrip_no_pretrained_file(tmp_path, monkeypatch):
    # Test-only one-block depth reduction keeps the real HF DINO graph bounded;
    # its production embedding width, normalization, registers and state remain real.
    config = _large_config()
    config.update(
        hidden_size=768, intermediate_size=3072, num_hidden_layers=12, num_attention_heads=12
    )
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    constructor = backbones._construct

    def reduced_constructor(architecture, actual_config, model_class):
        return constructor(architecture, dict(actual_config, num_hidden_layers=1), model_class)

    monkeypatch.setattr(backbones, "_construct", reduced_constructor)
    base = reduced_constructor("dinov3_b16", config, "ConvNextForImageClassification")
    encoder = backbones._GrayEncoder(base, 16).requires_grad_(False).eval()
    with torch.no_grad():
        base.embeddings.cls_token.add_(0.05)
    source = dict(
        architecture="dinov3_b16",
        config_sha256=backbones._sha256(config_path),
        weights_sha256="a" * 64,
        transformers_version="5.6.1",
        source_id="synthetic/test",
        source_revision="b" * 40,
        license="test-only",
        state_dict_format="safetensors",
    )
    provenance = backbones._encoder_provenance(source, config, "ConvNextForImageClassification")
    model = CompetitiveStudyNet(encoder, 768, hidden_dim=8).eval()
    whole = {key: value.detach().clone() for key, value in model.state_dict().items()}
    state = {
        key.removeprefix("encoder."): value
        for key, value in whole.items()
        if key.startswith("encoder.")
    }
    restored_encoder, dim, returned = backbones.create_encoder_from_training_checkpoint(
        provenance, config_path, encoder_state=state
    )
    restored = CompetitiveStudyNet(restored_encoder, dim, hidden_dim=8).eval()
    restored.load_state_dict(whole, strict=True)
    batch = (
        torch.rand(1, 1, 1, 1, 32, 32),
        torch.zeros(1, 1, 5),
        torch.ones(1, 1, dtype=torch.bool),
        torch.ones(1, 1, 1, dtype=torch.bool),
        torch.zeros(1, 1, 1, 2),
    )
    with torch.no_grad():
        torch.testing.assert_close(model(*batch), restored(*batch), rtol=0, atol=0)
    assert returned == provenance and dim == 768
