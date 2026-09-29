"""Rank-stabilized residual updates inside a frozen, local DINOv3 encoder."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, cast

from torch import Tensor, nn
from torch.nn import functional as F


def validate_adapter(spec: Mapping[str, Any], depth: int) -> dict[str, Any]:
    expected = {"rank", "alpha", "dropout", "ffn_last_layers", "gradient_checkpointing"}
    if not isinstance(spec, Mapping) or set(spec) != expected:
        raise ValueError("Adapter requires the exact rank/alpha/dropout/FFN/checkpointing policy")
    rank, last = spec["rank"], spec["ffn_last_layers"]
    if type(rank) is not int or not 1 <= rank <= 256:
        raise ValueError("Invalid adapter rank")
    if type(last) is not int or not 0 <= last <= depth:
        raise ValueError("Invalid adapter FFN depth")
    for name in ("alpha", "dropout"):
        value = spec[name]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError("Invalid adapter scaling/dropout")
    if spec["alpha"] <= 0 or not 0 <= spec["dropout"] < 1:
        raise ValueError("Invalid adapter scaling/dropout range")
    if type(spec["gradient_checkpointing"]) is not bool:
        raise ValueError("Adapter checkpointing policy must be boolean")
    return dict(spec)


def adapter_provenance(spec: Mapping[str, Any], config: Mapping[str, Any]) -> dict[str, Any]:
    depth, width, intermediate = (
        config["num_hidden_layers"],
        config["hidden_size"],
        config["intermediate_size"],
    )
    policy = validate_adapter(spec, depth)
    names: list[str] = []
    for layer in range(depth):
        names.extend(f"model.layer.{layer}.attention.{part}_proj" for part in ("q", "k", "v", "o"))
        if layer >= depth - policy["ffn_last_layers"]:
            names.extend(f"model.layer.{layer}.mlp.{part}_proj" for part in ("up", "down"))
    return {
        "method": "rs_lora_v1",
        **policy,
        "scaling": "alpha/sqrt(rank)",
        "base_frozen": True,
        "zero_initial_residual": True,
        "target_modules": names,
        "trainable_parameters": policy["rank"]
        * (depth * 4 * 2 * width + policy["ffn_last_layers"] * 2 * (width + intermediate)),
    }


class ResidualLinear(nn.Linear):
    """Preserve original weight/bias state keys and add separately named A/B tensors."""

    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float):
        # Module initialization avoids allocating a redundant full-sized base matrix.
        nn.Module.__init__(self)
        self.in_features, self.out_features = base.in_features, base.out_features
        self.weight = base.weight
        self.bias = base.bias
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)
        self.lora_A = nn.Parameter(base.weight.new_empty(rank, base.in_features))
        self.lora_B = nn.Parameter(base.weight.new_zeros(base.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.adapter_dropout = nn.Dropout(dropout)
        self.scaling = alpha / math.sqrt(rank)

    def forward(self, inputs: Tensor) -> Tensor:
        residual = F.linear(F.linear(self.adapter_dropout(inputs), self.lora_A), self.lora_B)
        return F.linear(inputs, self.weight, self.bias) + residual * self.scaling


def install_adapters(backbone: nn.Module, policy: Mapping[str, Any]) -> dict[str, Any]:
    config = getattr(backbone, "config", None)
    if config is None or config.model_type != "dinov3_vit":
        raise ValueError("Internal adapters require DINOv3 ViT")
    provenance = adapter_provenance(policy, config.to_dict())
    modules = dict(backbone.named_modules())
    # Validate the complete graph before replacing anything.
    for name in provenance["target_modules"]:
        module = modules.get(name)
        if type(module) is not nn.Linear:
            raise ValueError(f"Missing, changed or already adapted DINOv3 projection: {name}")
    backbone.requires_grad_(False)
    for name in provenance["target_modules"]:
        parent_name, attribute = name.rsplit(".", 1)
        setattr(
            modules[parent_name],
            attribute,
            ResidualLinear(modules[name], policy["rank"], policy["alpha"], policy["dropout"]),
        )
    actual = sum(p.numel() for p in backbone.parameters() if p.requires_grad)
    if actual != provenance["trainable_parameters"]:
        raise ValueError("Adapter parameter count differs from its declared architecture")
    if policy["gradient_checkpointing"]:
        cast(Any, backbone).gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    return provenance


def configure_encoder_training(encoder: nn.Module, freeze_base: bool) -> None:
    adapted = [module for module in encoder.modules() if isinstance(module, ResidualLinear)]
    if adapted and not freeze_base:
        raise ValueError("Adapter training requires the original encoder parameters frozen")
    encoder.requires_grad_(not freeze_base)
    for module in adapted:
        module.lora_A.requires_grad_(True)
        module.lora_B.requires_grad_(True)
    if adapted and any(
        p.requires_grad
        for name, p in encoder.named_parameters()
        if not name.endswith((".lora_A", ".lora_B"))
    ):
        raise ValueError("A base encoder parameter unexpectedly entered adapter training")
