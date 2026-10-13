"""Frozen-prefix12 DINOv3-L suffix adaptation: dense deltas versus suffix-only R32.

Research prototype for issue #40. Nothing under ``rsna/`` imports this module, so
the live trainer, ``rs_lora_v1`` and every inference ``code_sha256`` binding over
``rsna/**/*.py`` are unchanged. It never reads weights, data or the network.

Both matched arms share one frozen prefix and one suffix target set; only the
residual parameterization differs:

* ``dense_suffix_delta_v1``: ``y = W0 x + b0 + D dropout(x)``, ``D`` zero-initialized,
  full rank, anchored to the immutable official ``W0``/``b0``.
* ``suffix_rs_lora_v1``: the existing ``ResidualLinear`` (rank/alpha/dropout), installed
  only on the same suffix targets.

The exact frozen-prefix cache contract is in ``README.md``; ``run_prefix`` and
``run_suffix`` are the split of the pinned Transformers 5.6.1 ``DINOv3ViTModel``
forward that such a cache must reproduce.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rsna.competitive.adapters import ResidualLinear

ATTENTION_PARTS = ("q", "k", "v", "o")
MLP_PARTS = ("up", "down")
DENSE_KEYS = {"frozen_prefix_layers", "dropout"}
LORA_KEYS = {"frozen_prefix_layers", "rank", "alpha", "dropout"}
DELTA_SUFFIX = ".dense_delta"


def suffix_targets(config: Mapping[str, Any], frozen_prefix_layers: int) -> list[str]:
    depth = config["num_hidden_layers"]
    if type(frozen_prefix_layers) is not int or not 1 <= frozen_prefix_layers < depth:
        raise ValueError("Frozen prefix must leave a nonempty trainable suffix")
    names: list[str] = []
    for layer in range(frozen_prefix_layers, depth):
        names.extend(f"model.layer.{layer}.attention.{p}_proj" for p in ATTENTION_PARTS)
        names.extend(f"model.layer.{layer}.mlp.{p}_proj" for p in MLP_PARTS)
    return names


def _check_config(config: Mapping[str, Any]) -> None:
    if config.get("model_type") != "dinov3_vit":
        raise ValueError("Suffix adaptation requires DINOv3 ViT")
    if config.get("use_gated_mlp", False):
        raise ValueError("Gated MLP target names differ from the pinned DINOv3-L")
    # A cached prefix is exact only if the frozen prefix is deterministic.
    for key in ("drop_path_rate", "attention_dropout"):
        if config.get(key, 0.0) != 0.0:
            raise ValueError(f"Frozen-prefix caching requires {key}=0")


def _validate_dropout(value: object) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or not 0 <= value < 1
    ):
        raise ValueError("Invalid residual dropout")
    return float(value)


def dense_provenance(policy: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, Mapping) or set(policy) != DENSE_KEYS:
        raise ValueError("Dense suffix delta requires exactly frozen_prefix_layers/dropout")
    _check_config(config)
    dropout = _validate_dropout(policy["dropout"])
    names = suffix_targets(config, policy["frozen_prefix_layers"])
    width, inner = config["hidden_size"], config["intermediate_size"]
    per_layer = 4 * width * width + 2 * width * inner
    suffix = config["num_hidden_layers"] - policy["frozen_prefix_layers"]
    return {
        "method": "dense_suffix_delta_v1",
        "frozen_prefix_layers": policy["frozen_prefix_layers"],
        "dropout": dropout,
        "scaling": 1.0,
        "rank": "full",
        "anchor": "official_pretrained_weight_and_bias_immutable",
        "base_frozen": True,
        "biases_norms_layerscale_frozen": True,
        "zero_initial_residual": True,
        "weight_decay_anchor": "decoupled_decay_of_delta_equals_decay_toward_official_weight",
        "target_modules": names,
        "trainable_parameters": suffix * per_layer,
    }


def lora_provenance(policy: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(policy, Mapping) or set(policy) != LORA_KEYS:
        raise ValueError("Suffix LoRA requires exactly frozen_prefix_layers/rank/alpha/dropout")
    _check_config(config)
    rank, alpha = policy["rank"], policy["alpha"]
    if type(rank) is not int or not 1 <= rank <= 256:
        raise ValueError("Invalid adapter rank")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not alpha > 0:
        raise ValueError("Invalid adapter alpha")
    names = suffix_targets(config, policy["frozen_prefix_layers"])
    width, inner = config["hidden_size"], config["intermediate_size"]
    suffix = config["num_hidden_layers"] - policy["frozen_prefix_layers"]
    return {
        "method": "suffix_rs_lora_v1",
        "frozen_prefix_layers": policy["frozen_prefix_layers"],
        "rank": rank,
        "alpha": float(alpha),
        "dropout": _validate_dropout(policy["dropout"]),
        "scaling": "alpha/sqrt(rank)",
        "base_frozen": True,
        "zero_initial_residual": True,
        "target_modules": names,
        "trainable_parameters": suffix * rank * (4 * 2 * width + 2 * (width + inner)),
    }


class DenseDeltaLinear(nn.Linear):
    """Alias the official weight/bias (state keys unchanged) and add ``dense_delta``."""

    def __init__(self, base: nn.Linear, dropout: float):
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.dense_delta = nn.Parameter(
            torch.zeros_like(base.weight, memory_format=torch.contiguous_format)
        )
        self.adapter_dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = F.linear(self.adapter_dropout(inputs), self.dense_delta)
        return F.linear(inputs, self.weight, self.bias) + residual


def _replace(backbone: nn.Module, names: list[str], build) -> None:
    config = getattr(backbone, "config", None)
    if config is None or config.model_type != "dinov3_vit":
        raise ValueError("Suffix adaptation requires DINOv3 ViT")
    modules = dict(backbone.named_modules())
    for name in names:
        if type(modules.get(name)) is not nn.Linear:
            raise ValueError(f"Missing, changed or already adapted DINOv3 projection: {name}")
    if any(type(m) is not nn.Linear and isinstance(m, nn.Linear) for m in modules.values()):
        raise ValueError("Backbone already contains adapted projections")
    backbone.requires_grad_(False)
    for name in names:
        parent, attribute = name.rsplit(".", 1)
        setattr(modules[parent], attribute, build(modules[name]))


def install_dense_suffix_deltas(backbone: nn.Module, policy: Mapping[str, Any]) -> dict[str, Any]:
    provenance = dense_provenance(policy, cast(Any, backbone).config.to_dict())
    _replace(
        backbone,
        provenance["target_modules"],
        lambda base: DenseDeltaLinear(base, provenance["dropout"]),
    )
    _check_graph(backbone, provenance, (DELTA_SUFFIX,))
    return provenance


def install_suffix_lora(backbone: nn.Module, policy: Mapping[str, Any]) -> dict[str, Any]:
    provenance = lora_provenance(policy, cast(Any, backbone).config.to_dict())
    _replace(
        backbone,
        provenance["target_modules"],
        lambda base: ResidualLinear(base, policy["rank"], policy["alpha"], policy["dropout"]),
    )
    _check_graph(backbone, provenance, (".lora_A", ".lora_B"))
    return provenance


def _check_graph(backbone: nn.Module, provenance: Mapping[str, Any], suffixes: tuple) -> None:
    trainable = {n: p for n, p in backbone.named_parameters() if p.requires_grad}
    expected = {
        f"{module}{suffix}" for module in provenance["target_modules"] for suffix in suffixes
    }
    if set(trainable) != expected:
        raise ValueError("Trainable parameter graph differs from the declared suffix targets")
    if sum(p.numel() for p in trainable.values()) != provenance["trainable_parameters"]:
        raise ValueError("Adapter parameter count differs from its declared architecture")
    frozen = [p for p in backbone.parameters() if not p.requires_grad]
    frozen_storage = {p.untyped_storage().data_ptr() for p in frozen}
    if any(p.untyped_storage().data_ptr() in frozen_storage for p in trainable.values()):
        raise ValueError("A learned delta aliases an immutable official tensor")


def official_state_fingerprint(backbone: nn.Module) -> str:
    """SHA256 over every non-learned backbone state tensor (name, dtype, shape, bytes)."""
    learned = {n for n, p in backbone.named_parameters() if p.requires_grad}
    digest = hashlib.sha256()
    for name, tensor in sorted(backbone.state_dict(keep_vars=True).items()):
        if name in learned:
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}\n".encode())
        digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def learned_state(backbone: nn.Module) -> dict[str, Tensor]:
    """Only the learned residual tensors; the official anchor is never serialized."""
    return {n: p.detach().clone() for n, p in backbone.named_parameters() if p.requires_grad}


def restore_learned_state(backbone: nn.Module, state: Mapping[str, Tensor]) -> None:
    params = {n: p for n, p in backbone.named_parameters() if p.requires_grad}
    if set(state) != set(params):
        raise ValueError("Learned residual state keys differ from the installed graph")
    for name, value in state.items():
        target = params[name]
        if (
            not isinstance(value, Tensor)
            or value.shape != target.shape
            or value.dtype != target.dtype
            or not torch.isfinite(value).all()
        ):
            raise ValueError(f"Invalid learned residual tensor: {name}")
    with torch.no_grad():
        for name, value in state.items():
            # Copy into existing storage so optimizer references/aliases stay intact.
            params[name].copy_(value)


def _layers(backbone: nn.Module) -> nn.ModuleList:
    layers = backbone.get_submodule("model.layer")
    if not isinstance(layers, nn.ModuleList):
        raise ValueError("DINOv3 layer stack differs from the pinned contract")
    return layers


def rope_context(backbone: nn.Module, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
    """Exact RoPE (cos, sin) the full forward uses; rejects stochastic train-mode RoPE.

    The pinned config sets ``pos_embed_rescale=2.0``; Transformers applies that
    coordinate augmentation only while the RoPE module is training. A cache can
    only reproduce the deterministic evaluation-mode context.
    """
    rope = backbone.get_submodule("rope_embeddings")
    if rope.training:
        raise ValueError("Train-mode DINOv3 RoPE is stochastic; caching requires eval RoPE")
    cos, sin = rope(pixel_values)
    return cos, sin


def run_prefix(
    backbone: nn.Module, pixel_values: Tensor, frozen_prefix_layers: int
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Normalized RGB [N,3,H,W] -> hidden states after the first ``k`` layers + RoPE."""
    pixel_values = pixel_values.to(
        cast(Any, backbone.get_submodule("embeddings.patch_embeddings")).weight.dtype
    )
    hidden = backbone.get_submodule("embeddings")(pixel_values, bool_masked_pos=None)
    position_embeddings = rope_context(backbone, pixel_values)
    for layer in _layers(backbone)[:frozen_prefix_layers]:
        hidden = layer(hidden, position_embeddings=position_embeddings)
    return hidden, position_embeddings


def run_suffix(
    backbone: nn.Module,
    hidden: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    frozen_prefix_layers: int,
) -> tuple[Tensor, Tensor]:
    """Cached prefix hidden states -> (last_hidden_state, pooler_output)."""
    for layer in _layers(backbone)[frozen_prefix_layers:]:
        hidden = layer(hidden, position_embeddings=position_embeddings)
    sequence = backbone.get_submodule("norm")(hidden)
    return sequence, sequence[:, 0, :]
