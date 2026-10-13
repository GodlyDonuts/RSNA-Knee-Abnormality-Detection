"""Offline encoders: explicit architectures, local checksummed HF-format states."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import torch
from torch import Tensor, nn

from rsna.competitive.adapters import (
    adapter_provenance,
    install_adapters,
    learned_residual_parameters,
)
from rsna.competitive.suffix_adapters import (
    check_trainable_partition,
    install_suffix_adapter,
    installed_policy,
    keep_frozen_prefix_eval,
    suffix_adapter_provenance,
    suffix_forward,
)

TRANSFORMERS_VERSION = "5.6.1"
ARCHITECTURES = {"dinov2_b14_reg", "dinov3_b16", "dinov3_l16", "convnext_tiny"}
_DINOV3_L16_LEGACY_SOURCE = {
    "architecture": "dinov3_l16",
    "source_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "source_revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
    "config_sha256": "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5",
    "weights_sha256": "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
    "state_dict_format": "safetensors",
    "transformers_version": TRANSFORMERS_VERSION,
}
_DINOV3_LAYER_MAPPING = "dinov3_official_layer_to_model_layer_transformers_5_6_1_v1"


def _pretrained_key_mapping(spec: Mapping[str, Any]) -> str | None:
    if all(spec.get(key) == value for key, value in _DINOV3_L16_LEGACY_SOURCE.items()):
        return _DINOV3_LAYER_MAPPING
    return None


def _adapt_pretrained_state(spec: Mapping[str, Any], state: dict[str, Tensor]) -> dict[str, Tensor]:
    """Apply only the reviewed official source's documented HF key migration.

    Transformers 5.6.1 conversion_mapping.py maps DINOv3 ``layer.`` keys to
    ``model.layer.`` during from_pretrained. Our offline constructor bypasses
    that loader, so reproduce the leading-prefix rename for the exact admitted
    artifact. Tensor objects/values and original file hashes remain untouched.
    The caller must still strict-load every resulting key and tensor shape.
    """
    if _pretrained_key_mapping(spec) is None:
        return state
    if any(key.startswith("model.layer.") for key in state):
        raise ValueError("Official DINOv3 source has mixed or already migrated layer keys")
    adapted: dict[str, Tensor] = {}
    renamed = 0
    for key, tensor in state.items():
        target = "model." + key if key.startswith("layer.") else key
        if target in adapted:
            raise ValueError("DINOv3 checkpoint key migration collision")
        adapted[target] = tensor
        renamed += target != key
    if not renamed:
        raise ValueError("Official DINOv3 source is missing its expected layer namespace")
    return adapted


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checked_file(spec: Mapping[str, Any], kind: str) -> Path:
    value = spec.get(f"{kind}_path")
    expected = spec.get(f"{kind}_sha256")
    if not isinstance(value, (str, Path)) or not isinstance(expected, str):
        raise ValueError(f"Local {kind} path and SHA256 are required")
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"Invalid {kind} SHA256")
    path = Path(value)
    if not path.is_file() or _sha256(path) != expected:
        raise ValueError(f"Local {kind} missing or checksum mismatch")
    return path


def _validate_architecture(architecture: str, config: Mapping[str, Any]) -> None:
    expected: dict[str, Any]
    if architecture == "dinov2_b14_reg":
        expected = dict(
            model_type="dinov2_with_registers",
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            patch_size=14,
            num_register_tokens=4,
            num_channels=3,
        )
    elif architecture in {"dinov3_b16", "dinov3_l16"}:
        large = architecture == "dinov3_l16"
        expected = dict(
            model_type="dinov3_vit",
            hidden_size=1024 if large else 768,
            intermediate_size=4096 if large else 3072,
            num_hidden_layers=24 if large else 12,
            num_attention_heads=16 if large else 12,
            patch_size=16,
            num_register_tokens=4,
            num_channels=3,
        )
    else:
        expected = dict(
            model_type="convnext",
            depths=[3, 3, 9, 3],
            hidden_sizes=[96, 192, 384, 768],
            num_stages=4,
            patch_size=4,
            num_channels=3,
        )
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Pinned architecture mismatch: {key}")
    if architecture == "dinov3_l16":
        # The public Meta LVD ViT-L constructor uses the ordinary GELU MLP,
        # shared CLS/local norm and four registers, unlike L+ or satellite L.
        # Optional values below match the pinned Transformers 5.6.1 defaults.
        semantic_defaults = dict(
            hidden_act="gelu",
            use_gated_mlp=False,
            rope_theta=100.0,
            query_bias=True,
            key_bias=False,
            value_bias=True,
            proj_bias=True,
            mlp_bias=True,
            layer_norm_eps=1e-5,
            pos_embed_shift=None,
            pos_embed_jitter=None,
            pos_embed_rescale=2.0,
            untie_global_and_local_cls_norm=False,
        )
        for key, value in semantic_defaults.items():
            if config.get(key, value) != value:
                raise ValueError(f"Pinned DINOv3-L semantics mismatch: {key}")
        # This is an initialization scalar, subsequently replaced by every strict
        # checkpoint LayerScale tensor. Preserve the exact source config value.
        layer_scale = config.get("layerscale_value", 1.0)
        if (
            isinstance(layer_scale, bool)
            or not isinstance(layer_scale, (int, float))
            or not math.isfinite(layer_scale)
            or layer_scale <= 0
        ):
            raise ValueError("Invalid DINOv3-L LayerScale initialization")


def _construct(architecture: str, config: dict, model_class: str) -> nn.Module:
    # Explicit constructors never call the Hub, remote code, or from_pretrained.
    from transformers import (
        ConvNextConfig,
        ConvNextForImageClassification,
        ConvNextModel,
        Dinov2WithRegistersConfig,
        Dinov2WithRegistersModel,
        DINOv3ViTConfig,
        DINOv3ViTModel,
    )

    if architecture == "dinov2_b14_reg":
        return Dinov2WithRegistersModel(Dinov2WithRegistersConfig(**config))
    if architecture in {"dinov3_b16", "dinov3_l16"}:
        return DINOv3ViTModel(DINOv3ViTConfig(**config))
    cls = (
        ConvNextForImageClassification
        if model_class == "ConvNextForImageClassification"
        else ConvNextModel
    )
    return cls(ConvNextConfig(**config))


class _GrayEncoder(nn.Module):
    """Input gray [N,1,H,W] in [0,1]; output normalized-model global embeddings."""

    mean: Tensor
    std: Tensor

    def __init__(self, backbone: nn.Module, patch_multiple: int, input_channels: int = 1):
        super().__init__()
        if type(input_channels) is not int or input_channels not in {1, 3}:
            raise ValueError("One or three bound encoder channels required")
        self.input_channels = input_channels
        self.backbone = backbone
        self.patch_multiple = patch_multiple
        self.requires_grad_(True)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None])

    def requires_grad_(self, requires_grad: bool = True) -> _GrayEncoder:
        super().requires_grad_(requires_grad)
        # Residual adapters (rs-LoRA A/B or dense suffix deltas) never unfreeze
        # the official tensors they alias, even for requires_grad_(True).
        learned = learned_residual_parameters(self.backbone)
        if requires_grad and learned:
            self.backbone.requires_grad_(False)
            for parameter in learned:
                parameter.requires_grad_(True)
        # The unmasked global-embedding task never consumes the pretrained mask
        # token. Keep it outside DDP/optimizer even when the backbone is unfrozen.
        for name, parameter in self.backbone.named_parameters():
            if name == "embeddings.mask_token":
                parameter.requires_grad_(False)
        if installed_policy(self.backbone) is not None:
            check_trainable_partition(self.backbone)
        return self

    def train(self, mode: bool = True) -> _GrayEncoder:
        super().train(mode)
        # The suffix recipe's frozen prefix and RoPE are always deterministic.
        if installed_policy(self.backbone) is not None:
            keep_frozen_prefix_eval(self.backbone)
        return self

    def _backbone_output(self, images: Tensor) -> tuple[Tensor, Tensor]:
        pixels = self._normalized_rgb(images)
        if installed_policy(self.backbone) is not None:
            return suffix_forward(self.backbone, pixels)
        output = self.backbone(pixel_values=pixels)
        return output.last_hidden_state, output.pooler_output

    def _normalized_rgb(self, images: Tensor) -> Tensor:
        if (
            images.ndim != 4
            or images.shape[1] != self.input_channels
            or not images.is_floating_point()
        ):
            raise ValueError("Encoder requires its bound floating [N,C,H,W] input")
        if not torch.isfinite(images).all() or (images < 0).any() or (images > 1).any():
            raise ValueError("Encoder grayscale values must be finite in [0,1]")
        if any(n < self.patch_multiple or n % self.patch_multiple for n in images.shape[-2:]):
            raise ValueError("Image dimensions must be divisible by encoder patch/stride size")
        rgb = images.expand(-1, 3, -1, -1) if self.input_channels == 1 else images
        return (rgb - self.mean) / self.std

    def forward(self, images: Tensor) -> Tensor:
        if installed_policy(self.backbone) is not None:
            return self._backbone_output(images)[1]
        return self.backbone(pixel_values=self._normalized_rgb(images)).pooler_output

    def forward_features(self, images: Tensor) -> tuple[Tensor, Tensor]:
        """Return normalized DINOv3 CLS and spatial patch tokens, excluding registers."""
        config = getattr(self.backbone, "config", None)
        if config is None or getattr(config, "model_type", None) != "dinov3_vit":
            raise ValueError("Spatial patch features require the pinned DINOv3 ViT")
        tokens, pooled = self._backbone_output(images)
        height, width = (n // self.patch_multiple for n in images.shape[-2:])
        prefix = 1 + config.num_register_tokens
        if tokens.shape != (
            len(images),
            prefix + height * width,
            pooled.shape[-1],
        ) or not torch.equal(tokens[:, 0], pooled):
            raise ValueError("DINOv3 patch/register layout differs from the pinned contract")
        return pooled, tokens[:, prefix:].reshape(len(images), height, width, pooled.shape[-1])


class _DiagnosticEncoder(nn.Module):
    def __init__(self, embedding_dim: int, input_channels: int = 1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 8, 3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8, embedding_dim),
        )

    def forward(self, images: Tensor) -> Tensor:
        return self.layers(images)

    def forward_features(self, images: Tensor) -> tuple[Tensor, Tensor]:
        maps = self.layers[1](self.layers[0](images))
        local = nn.functional.adaptive_avg_pool2d(maps, (2, 2)).permute(0, 2, 3, 1)
        return self.forward(images), self.layers[-1](local)


def _validate_pretrained_metadata(spec: Mapping[str, Any]) -> None:
    architecture = spec.get("architecture")
    if not isinstance(architecture, str) or architecture not in ARCHITECTURES:
        raise ValueError("Unsupported encoder architecture")
    for key, pattern in [
        ("source_id", r"[\w.-]+/[\w.-]+"),
        ("source_revision", r"[0-9a-f]{40}"),
        ("license", r"[\w.+-]+"),
        ("config_sha256", r"[0-9a-f]{64}"),
        ("weights_sha256", r"[0-9a-f]{64}"),
    ]:
        if not isinstance(spec.get(key), str) or not re.fullmatch(pattern, spec[key]):
            raise ValueError(f"Invalid encoder provenance {key}")
    if spec.get("transformers_version") != TRANSFORMERS_VERSION:
        raise ValueError("Unsupported pinned transformers version")
    if importlib.metadata.version("transformers") != TRANSFORMERS_VERSION:
        raise ValueError("Installed transformers version does not match manifest")
    if not isinstance(spec.get("state_dict_format"), str) or spec["state_dict_format"] not in {
        "safetensors",
        "pytorch",
    }:
        raise ValueError("Unsupported state dictionary format")


def _encoder_dimensions(architecture: str) -> tuple[int, int]:
    return {
        "dinov2_b14_reg": (14, 768),
        "dinov3_b16": (16, 768),
        "dinov3_l16": (16, 1024),
        "convnext_tiny": (32, 768),
    }[architecture]


def _encoder_provenance(spec: Mapping[str, Any], config: dict, model_class: str) -> dict[str, Any]:
    architecture = spec["architecture"]
    _, dimension = _encoder_dimensions(architecture)
    source_keys = {
        "architecture",
        "config_sha256",
        "weights_sha256",
        "transformers_version",
        "source_id",
        "source_revision",
        "license",
        "state_dict_format",
    }
    result = {key: spec[key] for key in sorted(source_keys)}
    result.update(
        embedding_dim=dimension,
        pretrained=True,
        input_contract="gray_0_1_imagenet_rgb_v1",
        model_class=model_class if architecture == "convnext_tiny" else config["model_type"],
        pooling="layer_normalized_global_mean"
        if architecture == "convnext_tiny"
        else "normalized_cls",
    )
    mapping = _pretrained_key_mapping(spec)
    if mapping is not None:
        result["pretrained_state_key_mapping"] = mapping
    if "adapter" in spec:
        if architecture != "dinov3_l16":
            raise ValueError("Production adapters require the pinned DINOv3-L")
        policy = spec["adapter"]
        if not isinstance(policy, Mapping):
            raise ValueError("Adapter policy must be an object")
        # Input manifests contain the policy; saved provenance also binds derived facts.
        policy = {
            key: policy[key]
            for key in ("rank", "alpha", "dropout", "ffn_last_layers", "gradient_checkpointing")
        }
        result["adapter"] = adapter_provenance(policy, config)
    if "suffix_adapter" in spec:
        if architecture != "dinov3_l16" or "adapter" in spec:
            raise ValueError("Suffix adapters require the pinned DINOv3-L without all-block LoRA")
        if not isinstance(spec["suffix_adapter"], Mapping):
            raise ValueError("Suffix adapter policy must be an object")
        result["suffix_adapter"] = suffix_adapter_provenance(spec["suffix_adapter"], config)
    return result


_THREE_CHANNEL_CONTRACTS = {
    # Default 3-channel provenance remains the neighbor-delta contract.
    "neighbor_delta": ("neighbor_delta_0_1_v1", "neighbor_delta_0_1_imagenet_rgb_v1"),
    # Three ordered source-rendered public slices, each uint8/255, one ImageNet pass.
    "public_ordered_triplet": (
        "public_ordered_triplet_0_1_v1",
        "public_ordered_triplet_0_1_imagenet_rgb_v1",
    ),
}


def create_encoder(
    spec: Mapping[str, Any], *, input_channels: int = 1, input_contract: str = "neighbor_delta"
) -> tuple[nn.Module, int, dict[str, Any]]:
    """Build a local encoder. Diagnostic usage must also be gated by the trainer.

    Production specs bind architecture, config/weights SHA256, source_id,
    immutable source_revision, license, installed transformers_version and
    state_dict_format (safetensors or pytorch). Paths are excluded from provenance.
    PyTorch files must contain a plain state dictionary, not a pickled model.
    """
    if type(input_channels) is not int or input_channels not in {1, 3}:
        raise ValueError("One or three encoder input channels required")
    if input_contract not in _THREE_CHANNEL_CONTRACTS or (
        input_contract != "neighbor_delta" and input_channels != 3
    ):
        raise ValueError("Unknown or incompatible encoder input contract")
    diagnostic_contract, pretrained_contract = _THREE_CHANNEL_CONTRACTS[input_contract]
    architecture = spec.get("architecture")
    if input_channels == 3 and architecture not in {"dinov3_l16", "diagnostic_tiny"}:
        raise ValueError("Neighbor input requires the pinned DINOv3-L or engineering encoder")
    if architecture == "diagnostic_tiny":
        if set(spec) - {"architecture", "embedding_dim"}:
            raise ValueError("Unexpected diagnostic encoder fields")
        dim = spec.get("embedding_dim", 16)
        if isinstance(dim, bool) or not isinstance(dim, int) or not 1 <= dim <= 256:
            raise ValueError("Invalid diagnostic embedding dimension")
        return (
            _DiagnosticEncoder(dim, input_channels),
            dim,
            {
                "architecture": architecture,
                "embedding_dim": dim,
                "pretrained": False,
                "diagnostic_only": True,
                "input_contract": "gray_0_1_v1" if input_channels == 1 else diagnostic_contract,
            },
        )
    if not isinstance(architecture, str) or architecture not in ARCHITECTURES:
        raise ValueError("Unsupported encoder architecture")
    required = {
        "architecture",
        "config_path",
        "config_sha256",
        "weights_path",
        "weights_sha256",
        "transformers_version",
        "source_id",
        "source_revision",
        "license",
        "state_dict_format",
    }
    if not required <= set(spec) or set(spec) - required - {
        "model_class",
        "adapter",
        "suffix_adapter",
    }:
        raise ValueError("Missing or unknown encoder manifest fields")
    _validate_pretrained_metadata(spec)
    config_path = _checked_file(spec, "config")
    weights_path = _checked_file(spec, "weights")
    config = json.loads(config_path.read_text())
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    _validate_architecture(architecture, config)
    model_class = spec.get("model_class", "ConvNextForImageClassification")
    if architecture != "convnext_tiny" and "model_class" in spec:
        raise ValueError("model_class is only configurable for ConvNext")
    if model_class not in {"ConvNextForImageClassification", "ConvNextModel"}:
        raise ValueError("Unsupported ConvNext state class")
    model = _construct(architecture, config, model_class)
    if spec["state_dict_format"] == "safetensors":
        from safetensors.torch import load_file

        state = load_file(str(weights_path), device="cpu")
    else:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    if (
        not isinstance(state, dict)
        or not state
        or not all(
            isinstance(k, str) and isinstance(v, Tensor) and torch.isfinite(v).all()
            for k, v in state.items()
        )
    ):
        raise ValueError("Checkpoint must be a finite plain tensor state dictionary")
    model.load_state_dict(_adapt_pretrained_state(spec, state), strict=True)
    if architecture == "convnext_tiny" and model_class == "ConvNextForImageClassification":
        model = cast(nn.Module, model.convnext)
    patch, embedding_dim = _encoder_dimensions(architecture)
    provenance = _encoder_provenance(spec, config, model_class)
    if "adapter" in spec:
        installed = install_adapters(model, spec["adapter"])
        if installed != provenance["adapter"]:
            raise ValueError("Installed adapters differ from encoder provenance")
    if "suffix_adapter" in spec:
        # Installed only after the strict official load; base tensors stay aliased.
        if install_suffix_adapter(model, spec["suffix_adapter"]) != provenance["suffix_adapter"]:
            raise ValueError("Installed suffix adapters differ from encoder provenance")
    if input_channels == 3:
        provenance["input_contract"] = pretrained_contract
    return _GrayEncoder(model, patch, input_channels), embedding_dim, provenance


def create_encoder_from_training_checkpoint(
    provenance: Mapping[str, Any],
    config_path: Path,
    *,
    encoder_state: Mapping[str, Tensor],
) -> tuple[nn.Module, int, dict[str, Any]]:
    """Restore a production encoder without the original pretrained weight file.

    The caller must first authenticate the complete trained checkpoint's hash and
    release lineage. Pass its saved encoder provenance and all ``encoder.*`` model
    state entries with that prefix removed. Every encoder tensor is required and
    loaded strictly before return; the caller must still strict-load the complete
    study-model state to validate its pooling/head parameters. This is not a
    training initializer or an uninitialized-architecture factory.

    The original pretrained-weight SHA remains historical ancestry in the returned
    unchanged provenance; it is NOT reverified here. Only the local config bytes,
    architecture, saved provenance and supplied trained encoder state are checked.
    No original pretrained file is read, fetched or required. Diagnostic models
    continue to use their explicit diagnostic factory path.
    """
    _validate_pretrained_metadata(provenance)
    architecture = provenance["architecture"]
    checked_config = _checked_file(
        {"config_path": config_path, "config_sha256": provenance["config_sha256"]}, "config"
    )
    config = json.loads(checked_config.read_text())
    if not isinstance(config, dict):
        raise ValueError("Config must be a JSON object")
    _validate_architecture(architecture, config)
    model_class = provenance.get("model_class")
    if architecture == "convnext_tiny":
        if model_class not in {"ConvNextForImageClassification", "ConvNextModel"}:
            raise ValueError("Unsupported ConvNext checkpoint source class")
    else:
        model_class = "ConvNextForImageClassification"  # ignored by explicit DINO constructors
    expected = _encoder_provenance(provenance, config, model_class)
    three_channel = {pretrained for _, pretrained in _THREE_CHANNEL_CONTRACTS.values()}
    input_channels = 3 if provenance.get("input_contract") in three_channel else 1
    if input_channels == 3:
        if architecture != "dinov3_l16":
            raise ValueError("Neighbor checkpoint requires pinned DINOv3-L")
        expected["input_contract"] = provenance["input_contract"]
    if (
        set(provenance) != set(expected)
        or dict(provenance) != expected
        or type(provenance.get("embedding_dim")) is not int
        or provenance.get("pretrained") is not True
    ):
        raise ValueError("Saved encoder provenance/architecture contract mismatch")
    if (
        not isinstance(encoder_state, Mapping)
        or not encoder_state
        or not all(
            isinstance(key, str) and isinstance(value, Tensor) and torch.isfinite(value).all()
            for key, value in encoder_state.items()
        )
    ):
        raise ValueError("Trained encoder state must be a finite complete tensor dictionary")
    model = _construct(architecture, config, model_class)
    if architecture == "convnext_tiny" and model_class == "ConvNextForImageClassification":
        model = cast(nn.Module, model.convnext)
    if "adapter" in expected:
        policy = {
            key: expected["adapter"][key]
            for key in ("rank", "alpha", "dropout", "ffn_last_layers", "gradient_checkpointing")
        }
        if install_adapters(model, policy) != expected["adapter"]:
            raise ValueError("Checkpoint adapter architecture differs from its provenance")
    if "suffix_adapter" in expected:
        if install_suffix_adapter(model, expected["suffix_adapter"]) != expected["suffix_adapter"]:
            raise ValueError("Checkpoint suffix adapter differs from its provenance")
    patch, dimension = _encoder_dimensions(architecture)
    encoder = _GrayEncoder(model, patch, input_channels)
    # Buffers describe the fixed transform, not learned weights. Strict key/shape
    # loading alone would otherwise accept a silently changed preprocessing rule.
    for key in ("mean", "std"):
        saved = encoder_state.get(key)
        expected_buffer = getattr(encoder, key)
        if (
            not isinstance(saved, Tensor)
            or saved.dtype != expected_buffer.dtype
            or not torch.equal(saved.cpu(), expected_buffer)
        ):
            raise ValueError("Trained encoder normalization buffers differ from its provenance")
    encoder.load_state_dict(dict(encoder_state), strict=True)
    encoder.requires_grad_(False).eval()
    return encoder, dimension, expected
