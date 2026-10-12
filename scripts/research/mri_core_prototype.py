"""Isolated MRI-CORE (DINOv2-layout ViT-B/16) loader and rs-LoRA adapter prototype.

Research-only and opt-in. The module deliberately lives outside ``rsna/`` so the
``code_sha256`` binding of every existing training/inference checkpoint is
unchanged, and ``create_encoder``/``TrainConfig`` keep rejecting this
architecture. It is never relabelled as DINOv3. Sources, verified facts and
explicit unknowns: docs/mri-core-backbone-prototype-20261012.md.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rsna.competitive.adapters import ResidualLinear, validate_adapter

ARCHITECTURE = "mri_core_vitb16"
MODEL_FAMILY = "dinov2_vit"
SOURCE_REPOSITORY = "https://github.com/mazurowski-lab/mri_foundation"
SOURCE_PAPER = "arXiv:2506.12186"
LICENSE = "Apache-2.0"
CHECKPOINT_FILENAME = "MRI_CORE_vitb.pth"
# Prefix recorded by the 2026-10-06 decode (PR #4); the full digest must be
# supplied explicitly and is always rechecked against the local file bytes.
WEIGHTS_SHA256_PREFIX = "0ca90aeba09568f2"
CHECKPOINT_LAYOUT = "dinov2_teacher_backbone_unchunked_v1"
KEY_MAPPING = "strip_teacher_backbone_split_fused_qkv_v1"
POSITION_INTERPOLATION = "cls_kept_grid_bicubic_align_corners_false_no_antialias_fp32_v1"
INPUT_CONTRACT = "gray_n1hw_unit_interval_replicated_rgb_v1"
POOLING = "final_layernorm_cls"
DISCARDED_TEACHER_PREFIXES = ("dino_head.", "ibot_head.")
DISCARDED_BACKBONE_KEYS = ("mask_token",)
INPUT_NORMALIZATIONS = {
    # The upstream preprocessing constants are not yet primary-source verified;
    # provenance always records that status.
    "imagenet_mean_std": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}
CONFIG_FIELDS = (
    "embed_dim",
    "depth",
    "num_heads",
    "mlp_hidden_dim",
    "patch_size",
    "native_grid",
    "in_chans",
    "layerscale",
    "layer_norm_eps",
)
PRODUCTION_CONFIG: dict[str, Any] = {
    "embed_dim": 768,
    "depth": 12,
    "num_heads": 12,
    "mlp_hidden_dim": 3072,
    "patch_size": 16,
    "native_grid": 14,
    "in_chans": 3,
    "layerscale": True,
    "layer_norm_eps": 1e-6,
}
SPEC_FIELDS = {
    "architecture",
    "weights_path",
    "weights_sha256",
    "source_revision",
    "architecture_config",
    "image_side",
    "input_normalization",
}


def _is_hex(value: Any, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(char in "0123456789abcdef" for char in value)
    )


def canonical_sha256(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode()).hexdigest()


def validate_architecture_config(config: Any) -> dict[str, Any]:
    if not isinstance(config, Mapping) or set(config) != set(CONFIG_FIELDS):
        raise ValueError("MRI-CORE requires the exact bound architecture config fields")
    config = dict(config)
    for name in CONFIG_FIELDS[:7]:
        if type(config[name]) is not int or config[name] < 1:
            raise ValueError(f"Invalid MRI-CORE config field {name}")
    if type(config["layerscale"]) is not bool or type(config["layer_norm_eps"]) is not float:
        raise ValueError("Invalid MRI-CORE layerscale/epsilon")
    if config["embed_dim"] % config["num_heads"] or config["in_chans"] != 3:
        raise ValueError("Invalid MRI-CORE head width or channel count")
    if config != PRODUCTION_CONFIG:
        raise ValueError("Architecture config differs from the bound MRI-CORE ViT-B/16")
    return config


def expected_checkpoint_shapes(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Original DINOv2 backbone key layout, after stripping ``teacher.backbone.``."""
    d, h, p, g = (config[k] for k in ("embed_dim", "mlp_hidden_dim", "patch_size", "native_grid"))
    shapes: dict[str, tuple[int, ...]] = {
        "cls_token": (1, 1, d),
        "pos_embed": (1, 1 + g * g, d),
        "mask_token": (1, d),
        "patch_embed.proj.weight": (d, config["in_chans"], p, p),
        "patch_embed.proj.bias": (d,),
        "norm.weight": (d,),
        "norm.bias": (d,),
    }
    for i in range(config["depth"]):
        block = f"blocks.{i}."
        shapes.update(
            {
                block + "norm1.weight": (d,),
                block + "norm1.bias": (d,),
                block + "attn.qkv.weight": (3 * d, d),
                block + "attn.qkv.bias": (3 * d,),
                block + "attn.proj.weight": (d, d),
                block + "attn.proj.bias": (d,),
                block + "norm2.weight": (d,),
                block + "norm2.bias": (d,),
                block + "mlp.fc1.weight": (h, d),
                block + "mlp.fc1.bias": (h,),
                block + "mlp.fc2.weight": (d, h),
                block + "mlp.fc2.bias": (d,),
            }
        )
        if config["layerscale"]:
            shapes[block + "ls1.gamma"] = (d,)
            shapes[block + "ls2.gamma"] = (d,)
    return shapes


def interpolate_position_embedding(pos_embed: Tensor, native_grid: int, target_grid: int) -> Tensor:
    if pos_embed.ndim != 3 or pos_embed.shape[:2] != (1, 1 + native_grid * native_grid):
        raise ValueError("Position embedding does not match the native grid")
    if target_grid == native_grid:
        return pos_embed.clone()
    width = pos_embed.shape[-1]
    grid = pos_embed[:, 1:].float().reshape(1, native_grid, native_grid, width).permute(0, 3, 1, 2)
    grid = F.interpolate(
        grid, size=(target_grid, target_grid), mode="bicubic", align_corners=False, antialias=False
    )
    grid = grid.permute(0, 2, 3, 1).reshape(1, target_grid * target_grid, width)
    return torch.cat([pos_embed[:, :1], grid.to(pos_embed.dtype)], dim=1)


def map_checkpoint(
    raw: Any, config: Mapping[str, Any], target_grid: int
) -> tuple[dict[str, Tensor], list[str]]:
    """Fail-closed MRI-CORE teacher checkpoint -> prototype module state."""
    if not isinstance(raw, Mapping) or set(raw) != {"teacher"}:
        raise ValueError("MRI-CORE checkpoint must contain exactly the teacher state")
    teacher = raw["teacher"]
    if not isinstance(teacher, Mapping) or not all(
        isinstance(k, str) and isinstance(v, Tensor) for k, v in teacher.items()
    ):
        raise ValueError("MRI-CORE teacher state must map names to tensors")
    backbone = {k[9:]: v for k, v in teacher.items() if k.startswith("backbone.")}
    other = sorted(k for k in teacher if not k.startswith("backbone."))
    unexpected = [k for k in other if not k.startswith(DISCARDED_TEACHER_PREFIXES)]
    expected = expected_checkpoint_shapes(config)
    unexpected += sorted("backbone." + k for k in set(backbone) - set(expected))
    missing = sorted(set(expected) - set(backbone))
    if unexpected or missing:
        raise ValueError(
            f"MRI-CORE checkpoint keys differ: unexpected={unexpected[:5]} missing={missing[:5]}"
        )
    for name, shape in expected.items():
        value = backbone[name]
        if tuple(value.shape) != shape or not value.is_floating_point():
            raise ValueError(f"MRI-CORE tensor {name} has an unexpected shape/dtype")
        if not torch.isfinite(value).all():
            raise ValueError(f"MRI-CORE tensor {name} is not finite")
    mapped: dict[str, Tensor] = {}
    for name, value in backbone.items():
        if name in DISCARDED_BACKBONE_KEYS:
            continue
        if name.endswith(("attn.qkv.weight", "attn.qkv.bias")):
            stem, suffix = name.rsplit(".qkv.", 1)
            # DINOv2 reshapes the fused output to (..., 3, heads, head_dim): q, k, v order.
            for part, chunk in zip("qkv", value.chunk(3, dim=0)):
                mapped[f"{stem}.{part}.{suffix}"] = chunk.clone()
        elif name == "pos_embed":
            mapped[name] = interpolate_position_embedding(value, config["native_grid"], target_grid)
        else:
            mapped[name] = value.clone()
    discarded = other + ["backbone." + k for k in DISCARDED_BACKBONE_KEYS]
    return mapped, discarded


class _LayerScale(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(width))

    def forward(self, x: Tensor) -> Tensor:
        return x * self.gamma


class _Attention(nn.Module):
    def __init__(self, width: int, heads: int):
        super().__init__()
        self.heads = heads
        self.q, self.k, self.v = (nn.Linear(width, width) for _ in range(3))
        self.proj = nn.Linear(width, width)

    def forward(self, x: Tensor) -> Tensor:
        batch, tokens, width = x.shape
        q, k, v = (
            layer(x).reshape(batch, tokens, self.heads, -1).transpose(1, 2)
            for layer in (self.q, self.k, self.v)
        )
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(batch, tokens, width))


class _Mlp(nn.Module):
    def __init__(self, width: int, hidden: int):
        super().__init__()
        self.fc1, self.fc2 = nn.Linear(width, hidden), nn.Linear(hidden, width)

    def forward(self, x: Tensor) -> Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _Block(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        width, eps = config["embed_dim"], config["layer_norm_eps"]
        self.norm1 = nn.LayerNorm(width, eps=eps)
        self.attn = _Attention(width, config["num_heads"])
        self.ls1 = _LayerScale(width) if config["layerscale"] else nn.Identity()
        self.norm2 = nn.LayerNorm(width, eps=eps)
        self.mlp = _Mlp(width, config["mlp_hidden_dim"])
        self.ls2 = _LayerScale(width) if config["layerscale"] else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.ls1(self.attn(self.norm1(x)))
        return x + self.ls2(self.mlp(self.norm2(x)))


class _PatchEmbed(nn.Module):
    def __init__(self, config: Mapping[str, Any]):
        super().__init__()
        p = config["patch_size"]
        self.proj = nn.Conv2d(config["in_chans"], config["embed_dim"], p, stride=p)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x).flatten(2).transpose(1, 2)


class MRICoreViT(nn.Module):
    """DINOv2 ViT forward with a load-time-interpolated, fixed token grid."""

    def __init__(self, config: Mapping[str, Any], target_grid: int):
        super().__init__()
        self.config, self.grid = dict(config), target_grid
        width = config["embed_dim"]
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + target_grid * target_grid, width))
        self.patch_embed = _PatchEmbed(config)
        self.blocks = nn.ModuleList(_Block(config) for _ in range(config["depth"]))
        self.norm = nn.LayerNorm(width, eps=config["layer_norm_eps"])
        self.gradient_checkpointing = False

    def forward(self, pixels: Tensor) -> Tensor:
        if tuple(pixels.shape[-2:]) != (self.grid * self.config["patch_size"],) * 2:
            raise ValueError("MRI-CORE input side differs from its bound interpolated grid")
        x = self.patch_embed(pixels)
        x = torch.cat([self.cls_token.expand(len(x), -1, -1), x], dim=1) + self.pos_embed
        for block in self.blocks:
            if self.gradient_checkpointing and torch.is_grad_enabled():
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.norm(x)


def adapter_target_modules(config: Mapping[str, Any], policy: Mapping[str, Any]) -> list[str]:
    depth = config["depth"]
    names: list[str] = []
    for layer in range(depth):
        names.extend(f"blocks.{layer}.attn.{part}" for part in ("q", "k", "v", "proj"))
        if layer >= depth - policy["ffn_last_layers"]:
            names.extend(f"blocks.{layer}.mlp.{part}" for part in ("fc1", "fc2"))
    return names


def adapter_trainable_parameters(config: Mapping[str, Any], policy: Mapping[str, Any]) -> int:
    width, hidden = config["embed_dim"], config["mlp_hidden_dim"]
    return policy["rank"] * (
        config["depth"] * 4 * 2 * width + policy["ffn_last_layers"] * 2 * (width + hidden)
    )


def install_mri_core_adapters(backbone: MRICoreViT, spec: Mapping[str, Any]) -> dict[str, Any]:
    policy = validate_adapter(spec, backbone.config["depth"])
    if any(isinstance(m, ResidualLinear) for m in backbone.modules()):
        raise ValueError("MRI-CORE adapters are already installed")
    names = adapter_target_modules(backbone.config, policy)
    for name in names:
        parent_name, child = name.rsplit(".", 1)
        parent, base = backbone.get_submodule(parent_name), backbone.get_submodule(name)
        if type(base) is not nn.Linear:
            raise ValueError(f"MRI-CORE adapter target {name} is not an original Linear")
        setattr(
            parent,
            child,
            ResidualLinear(base, policy["rank"], policy["alpha"], policy["dropout"]),
        )
    installed = [n for n, m in backbone.named_modules() if isinstance(m, ResidualLinear)]
    count = sum(
        m.lora_A.numel() + m.lora_B.numel()
        for m in backbone.modules()
        if isinstance(m, ResidualLinear)
    )
    expected_count = adapter_trainable_parameters(backbone.config, policy)
    if installed != names or count != expected_count:
        raise ValueError("MRI-CORE adapter graph differs from its bound target modules")
    backbone.gradient_checkpointing = policy["gradient_checkpointing"]
    return {
        "method": "rs_lora_v1",
        **policy,
        "scaling": "alpha/sqrt(rank)",
        "base_frozen": True,
        "zero_initial_residual": True,
        "target_modules": names,
        "trainable_parameters": count,
    }


class MRICoreEncoder(nn.Module):
    """Gray [N,1,H,W] in [0,1] -> final-LayerNorm CLS embedding."""

    mean: Tensor
    std: Tensor

    def __init__(self, backbone: MRICoreViT, normalization: str):
        super().__init__()
        mean, std = INPUT_NORMALIZATIONS[normalization]
        self.backbone = backbone
        self.patch_multiple = backbone.config["patch_size"]
        self.register_buffer("mean", torch.tensor(mean)[None, :, None, None])
        self.register_buffer("std", torch.tensor(std)[None, :, None, None])

    def requires_grad_(self, requires_grad: bool = True) -> MRICoreEncoder:
        super().requires_grad_(requires_grad)
        adapted = [m for m in self.backbone.modules() if isinstance(m, ResidualLinear)]
        if requires_grad and adapted:
            self.backbone.requires_grad_(False)
            for module in adapted:
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)
        return self

    def normalized_rgb(self, images: Tensor) -> Tensor:
        if images.ndim != 4 or images.shape[1] != 1 or not images.is_floating_point():
            raise ValueError("MRI-CORE prototype requires floating gray [N,1,H,W] input")
        if not torch.isfinite(images).all() or (images < 0).any() or (images > 1).any():
            raise ValueError("Encoder grayscale values must be finite in [0,1]")
        return (images.expand(-1, 3, -1, -1) - self.mean) / self.std

    def forward(self, images: Tensor) -> Tensor:
        return self.backbone(self.normalized_rgb(images))[:, 0]


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def create_mri_core_encoder(
    spec: Mapping[str, Any], **kwargs: Any
) -> tuple[MRICoreEncoder, int, dict[str, Any]]:
    """Explicit opt-in factory with the ``create_encoder`` return contract."""
    if kwargs:
        raise ValueError("MRI-CORE prototype supports only single-channel gray input")
    if not isinstance(spec, Mapping) or not SPEC_FIELDS <= set(spec) <= SPEC_FIELDS | {"adapter"}:
        raise ValueError("MRI-CORE spec requires the exact explicit source/config fields")
    if spec["architecture"] != ARCHITECTURE:
        raise ValueError("MRI-CORE loader refuses other (including DINOv2/DINOv3) labels")
    if not _is_hex(spec["weights_sha256"], 64) or not spec["weights_sha256"].startswith(
        WEIGHTS_SHA256_PREFIX
    ):
        raise ValueError("MRI-CORE weights digest is not the recorded checkpoint identity")
    if not _is_hex(spec["source_revision"], 40):
        raise ValueError("MRI-CORE source repository revision must be an exact commit")
    if spec["input_normalization"] not in INPUT_NORMALIZATIONS:
        raise ValueError("Unknown MRI-CORE input normalization")
    config = validate_architecture_config(spec["architecture_config"])
    side, patch = spec["image_side"], config["patch_size"]
    if type(side) is not int or side < patch or side % patch:
        raise ValueError("MRI-CORE image side must be a positive patch multiple")
    path = Path(spec["weights_path"])
    if not path.is_file() or _file_sha256(path) != spec["weights_sha256"]:
        raise ValueError("MRI-CORE weights file is absent or differs from its bound digest")
    raw = torch.load(path, map_location="cpu", weights_only=True)
    mapped, discarded = map_checkpoint(raw, config, side // patch)
    backbone = MRICoreViT(config, side // patch)
    target = backbone.state_dict()
    if set(mapped) != set(target) or any(mapped[k].shape != target[k].shape for k in target):
        raise ValueError("Mapped MRI-CORE state differs from the prototype module")
    backbone.load_state_dict(mapped, strict=True)
    adapter = None
    if "adapter" in spec:
        adapter = install_mri_core_adapters(backbone, spec["adapter"])
    encoder = MRICoreEncoder(backbone, spec["input_normalization"])
    provenance = {
        "architecture": ARCHITECTURE,
        "model_family": MODEL_FAMILY,
        "source_repository": SOURCE_REPOSITORY,
        "source_revision": spec["source_revision"],
        "source_paper": SOURCE_PAPER,
        "license": LICENSE,
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "weights_sha256": spec["weights_sha256"],
        "architecture_config": config,
        "config_sha256": canonical_sha256(config),
        "checkpoint_layout": CHECKPOINT_LAYOUT,
        "key_mapping": KEY_MAPPING,
        "discarded_checkpoint_keys": discarded,
        "position_interpolation": {
            "method": POSITION_INTERPOLATION,
            "native_grid": config["native_grid"],
            "target_grid": side // patch,
        },
        "image_side": side,
        "embedding_dim": config["embed_dim"],
        "input_contract": INPUT_CONTRACT,
        "input_normalization": spec["input_normalization"],
        "input_normalization_source_verified": False,
        "pooling": POOLING,
        "pretrained": True,
        "knee_coverage": "unknown",
        "competition_data_overlap": "unknown",
        **({"adapter": adapter} if adapter is not None else {}),
    }
    return encoder, config["embed_dim"], provenance


def vit_forward_flops(width: int, depth: int, hidden: int, patches: int, prefix: int) -> int:
    """Approximate dense FLOPs (2 x multiply-accumulates) for one patch-16 slice."""
    tokens = patches + prefix
    per_layer = 2 * tokens * (4 * width * width + 2 * width * hidden) + 4 * tokens * tokens * width
    return depth * per_layer + 2 * patches * 3 * 16 * 16 * width
