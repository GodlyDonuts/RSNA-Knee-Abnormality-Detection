"""Frozen-prefix DINOv3 suffix adaptation: full-rank deltas or matched suffix-only R32.

Canonical integration of the reviewed issue #40 prototype
(``tools/research/dense_suffix_delta``). It is selected only by an explicit
``encoder.suffix_adapter`` policy; the default encoder and the all-block
``encoder.adapter`` (``rs_lora_v1``) path are unchanged.

Both matched methods share one recipe, which differs from the live all-block
R32 recipe and must be reported as such:

* embeddings, RoPE and layers ``0..k-1`` are a frozen prefix that always runs in
  evaluation mode, without gradients and without autocast (FP32, the official
  tensor dtype). Live training RoPE applies the stochastic ``pos_embed_rescale``
  coordinate augmentation; here RoPE is always the deterministic evaluation value;
* layers ``k..`` and the final norm run under the caller's precision policy; only
  separately named residual tensors on the same 72 suffix projections learn.

``dense_suffix_delta_v1``: ``y = W0 x + b0 + D dropout(x)``, ``D`` zero-initialized.
``suffix_rs_lora_v1``: the existing ``ResidualLinear`` (rank/alpha/dropout).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Callable, Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn

from rsna.competitive.adapters import (
    LEARNED_RESIDUAL_SUFFIXES,
    DenseDeltaLinear,
    ResidualLinear,
)

DENSE = "dense_suffix_delta_v1"
LORA = "suffix_rs_lora_v1"
POLICY_KEYS = {
    DENSE: {"method", "frozen_prefix_layers", "dropout", "gradient_checkpointing"},
    LORA: {"method", "frozen_prefix_layers", "rank", "alpha", "dropout", "gradient_checkpointing"},
}
DERIVED_KEYS = {
    "scaling",
    "base_frozen",
    "biases_norms_layerscale_frozen",
    "zero_initial_residual",
    "frozen_prefix_execution",
    "rope",
    "target_modules",
    "trainable_parameters",
    "anchor",
    "rank_kind",
}
PREFIX_EXECUTION = "eval_no_grad_fp32_no_autocast"
ROPE = "deterministic_eval_mode_for_prefix_and_suffix"
_ATTRIBUTE = "_rsna_suffix_adapter"


def suffix_targets(config: Mapping[str, Any], frozen_prefix_layers: int) -> list[str]:
    depth = config["num_hidden_layers"]
    if type(frozen_prefix_layers) is not int or not 1 <= frozen_prefix_layers < depth:
        raise ValueError("Frozen prefix must leave a nonempty trainable suffix")
    names: list[str] = []
    for layer in range(frozen_prefix_layers, depth):
        names.extend(f"model.layer.{layer}.attention.{p}_proj" for p in ("q", "k", "v", "o"))
        names.extend(f"model.layer.{layer}.mlp.{p}_proj" for p in ("up", "down"))
    return names


def _finite(value: object) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value)


def validate_suffix_adapter(spec: object, depth: int) -> dict[str, Any]:
    """Validate the exact input policy (no derived fields) for a ``depth``-layer backbone."""
    if not isinstance(spec, Mapping) or spec.get("method") not in POLICY_KEYS:
        raise ValueError("Suffix adapter requires an explicit dense or suffix R32 method")
    if set(spec) != POLICY_KEYS[spec["method"]]:
        raise ValueError("Suffix adapter requires its exact method policy fields")
    prefix = spec["frozen_prefix_layers"]
    if type(prefix) is not int or not 1 <= prefix < depth:
        raise ValueError("Frozen prefix must leave a nonempty trainable suffix")
    if not _finite(spec["dropout"]) or not 0 <= spec["dropout"] < 1:
        raise ValueError("Invalid residual dropout")
    if type(spec["gradient_checkpointing"]) is not bool:
        raise ValueError("Suffix checkpointing policy must be boolean")
    if spec["method"] == LORA:
        if type(spec["rank"]) is not int or not 1 <= spec["rank"] <= 256:
            raise ValueError("Invalid adapter rank")
        if not _finite(spec["alpha"]) or not spec["alpha"] > 0:
            raise ValueError("Invalid adapter alpha")
    return dict(spec)


def _check_config(config: Mapping[str, Any]) -> None:
    if config.get("model_type") != "dinov3_vit":
        raise ValueError("Suffix adaptation requires DINOv3 ViT")
    if config.get("use_gated_mlp", False):
        raise ValueError("Gated MLP target names differ from the pinned DINOv3-L")
    # The frozen prefix and the cached prefix contract are exact only if deterministic.
    for key in ("drop_path_rate", "attention_dropout"):
        if config.get(key, 0.0) != 0.0:
            raise ValueError(f"Frozen-prefix suffix adaptation requires {key}=0")


def suffix_adapter_provenance(spec: Mapping[str, Any], config: Mapping[str, Any]) -> dict:
    policy = validate_suffix_adapter(
        {key: spec[key] for key in spec if key not in DERIVED_KEYS}, config["num_hidden_layers"]
    )
    _check_config(config)
    names = suffix_targets(config, policy["frozen_prefix_layers"])
    width, inner = config["hidden_size"], config["intermediate_size"]
    suffix = config["num_hidden_layers"] - policy["frozen_prefix_layers"]
    common = {
        **policy,
        "dropout": float(policy["dropout"]),
        "base_frozen": True,
        "biases_norms_layerscale_frozen": True,
        "zero_initial_residual": True,
        "frozen_prefix_execution": PREFIX_EXECUTION,
        "rope": ROPE,
        "target_modules": names,
    }
    if policy["method"] == DENSE:
        return {
            **common,
            "scaling": 1.0,
            "rank_kind": "full",
            "anchor": "official_pretrained_weight_and_bias_immutable",
            "trainable_parameters": suffix * (4 * width * width + 2 * width * inner),
        }
    return {
        **common,
        "alpha": float(policy["alpha"]),
        "scaling": "alpha/sqrt(rank)",
        "rank_kind": "low_rank",
        "anchor": "official_pretrained_weight_and_bias_immutable",
        "trainable_parameters": suffix * policy["rank"] * (4 * 2 * width + 2 * (width + inner)),
    }


def _replace(backbone: nn.Module, names: list[str], build: Callable[[nn.Linear], nn.Module]):
    modules = dict(backbone.named_modules())
    for name in names:
        if type(modules.get(name)) is not nn.Linear:
            raise ValueError(f"Missing, changed or already adapted DINOv3 projection: {name}")
    if any(type(m) is not nn.Linear and isinstance(m, nn.Linear) for m in modules.values()):
        raise ValueError("Backbone already contains adapted projections")
    backbone.requires_grad_(False)
    for name in names:
        parent, attribute = name.rsplit(".", 1)
        replacement = build(cast(nn.Linear, modules[name]))
        replacement.train(modules[name].training)
        setattr(modules[parent], attribute, replacement)


def install_suffix_adapter(backbone: nn.Module, spec: Mapping[str, Any]) -> dict[str, Any]:
    """Install on a strictly loaded official backbone; returns the bound provenance.

    LoRA ``A`` initialization draws from the global CPU generator inside a forked
    RNG scope, so the caller's subsequent head initialization and RNG stream are
    identical in the dense and R32 arms.
    """
    config = getattr(backbone, "config", None)
    if config is None or getattr(config, "model_type", None) != "dinov3_vit":
        raise ValueError("Suffix adaptation requires DINOv3 ViT")
    if hasattr(backbone, _ATTRIBUTE):
        raise ValueError("Backbone already contains adapted projections")
    provenance = suffix_adapter_provenance(spec, config.to_dict())
    dropout = provenance["dropout"]
    with torch.random.fork_rng(devices=[]):
        if provenance["method"] == DENSE:
            _replace(
                backbone,
                provenance["target_modules"],
                lambda base: DenseDeltaLinear(base, dropout),
            )
        else:
            _replace(
                backbone,
                provenance["target_modules"],
                lambda base: ResidualLinear(base, provenance["rank"], provenance["alpha"], dropout),
            )
    _check_graph(backbone, provenance)
    if provenance["gradient_checkpointing"]:
        cast(Any, backbone).gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
    setattr(backbone, _ATTRIBUTE, dict(provenance))
    return provenance


def installed_policy(backbone: nn.Module) -> dict[str, Any] | None:
    value = getattr(backbone, _ATTRIBUTE, None)
    return None if value is None else dict(value)


def _expected_learned(provenance: Mapping[str, Any]) -> set[str]:
    suffixes = (".dense_delta",) if provenance["method"] == DENSE else (".lora_A", ".lora_B")
    return {f"{m}{s}" for m in provenance["target_modules"] for s in suffixes}


def _check_graph(backbone: nn.Module, provenance: Mapping[str, Any]) -> None:
    trainable = {n: p for n, p in backbone.named_parameters() if p.requires_grad}
    if set(trainable) != _expected_learned(provenance):
        raise ValueError("Trainable parameter graph differs from the declared suffix targets")
    if sum(p.numel() for p in trainable.values()) != provenance["trainable_parameters"]:
        raise ValueError("Adapter parameter count differs from its declared architecture")
    frozen = {
        p.untyped_storage().data_ptr()
        for p in backbone.parameters()
        if not p.requires_grad and p.device.type != "meta"
    }
    if any(
        p.device.type != "meta" and p.untyped_storage().data_ptr() in frozen
        for p in trainable.values()
    ):
        raise ValueError("A learned delta aliases an immutable official tensor")


def check_trainable_partition(backbone: nn.Module) -> None:
    """Refuse any official tensor requiring gradients in a suffix-adapted backbone."""
    provenance = installed_policy(backbone)
    if provenance is None:
        raise ValueError("Backbone has no installed suffix adapter")
    trainable = {n for n, p in backbone.named_parameters() if p.requires_grad}
    if trainable and trainable != _expected_learned(provenance):
        raise ValueError("An official DINOv3 tensor entered suffix adapter training")


def official_state_fingerprint(backbone: nn.Module) -> str:
    """SHA256 over every non-residual backbone state tensor (name, dtype, shape, bytes)."""
    digest = hashlib.sha256()
    for name, tensor in sorted(backbone.state_dict(keep_vars=True).items()):
        if name.endswith(LEARNED_RESIDUAL_SUFFIXES):
            continue
        value = tensor.detach().cpu().contiguous()
        digest.update(f"{name}|{value.dtype}|{tuple(value.shape)}\n".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def frozen_prefix_modules(backbone: nn.Module, frozen_prefix_layers: int) -> list[nn.Module]:
    layers = backbone.get_submodule("model.layer")
    if not isinstance(layers, nn.ModuleList):
        raise ValueError("DINOv3 layer stack differs from the pinned contract")
    return [
        backbone.get_submodule("embeddings"),
        backbone.get_submodule("rope_embeddings"),
        *layers[:frozen_prefix_layers],
    ]


def keep_frozen_prefix_eval(backbone: nn.Module) -> None:
    provenance = installed_policy(backbone)
    if provenance is None:
        raise ValueError("Backbone has no installed suffix adapter")
    for module in frozen_prefix_modules(backbone, provenance["frozen_prefix_layers"]):
        module.eval()


def rope_context(backbone: nn.Module, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
    """Exact deterministic RoPE (cos, sin); train-mode RoPE is stochastic and refused."""
    rope = backbone.get_submodule("rope_embeddings")
    if rope.training:
        raise ValueError("Train-mode DINOv3 RoPE is stochastic; suffix recipe requires eval RoPE")
    cos, sin = rope(pixel_values)
    return cos, sin


def run_prefix(
    backbone: nn.Module, pixel_values: Tensor, frozen_prefix_layers: int
) -> tuple[Tensor, tuple[Tensor, Tensor]]:
    """Normalized RGB [N,3,H,W] -> FP32 hidden states after layer ``k-1`` plus RoPE.

    Runs without gradients and with autocast disabled; never casts official tensors.
    """
    modules = frozen_prefix_modules(backbone, frozen_prefix_layers)
    if any(module.training for module in modules):
        raise ValueError("Frozen prefix must run in evaluation mode")
    weight = cast(Any, backbone.get_submodule("embeddings.patch_embeddings")).weight
    if pixel_values.dtype != weight.dtype:
        raise ValueError("Prefix input dtype differs from the official tensor dtype")
    with torch.no_grad(), torch.autocast(pixel_values.device.type, enabled=False):
        hidden = modules[0](pixel_values, bool_masked_pos=None)
        position_embeddings = rope_context(backbone, pixel_values)
        for layer in modules[2:]:
            hidden = layer(hidden, position_embeddings=position_embeddings)
    return hidden, position_embeddings


def run_suffix(
    backbone: nn.Module,
    hidden: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    frozen_prefix_layers: int,
) -> tuple[Tensor, Tensor]:
    """Prefix hidden states -> (last_hidden_state, pooler_output) under caller precision."""
    layers = backbone.get_submodule("model.layer")
    assert isinstance(layers, nn.ModuleList)
    for layer in layers[frozen_prefix_layers:]:
        hidden = layer(hidden, position_embeddings=position_embeddings)
    sequence = backbone.get_submodule("norm")(hidden)
    return sequence, sequence[:, 0, :]


def suffix_forward(backbone: nn.Module, pixel_values: Tensor) -> tuple[Tensor, Tensor]:
    provenance = installed_policy(backbone)
    if provenance is None:
        raise ValueError("Backbone has no installed suffix adapter")
    prefix = provenance["frozen_prefix_layers"]
    hidden, rope = run_prefix(backbone, pixel_values, prefix)
    return run_suffix(backbone, hidden, rope, prefix)
