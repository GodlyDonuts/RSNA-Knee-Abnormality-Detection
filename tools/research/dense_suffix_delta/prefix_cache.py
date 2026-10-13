"""Exact frozen-prefix feature cache records for suffix-only DINOv3 fitting (issue #40).

A record is valid only for one fixed, non-augmented source-input contract, one exact
config/official-tensor identity and one frozen prefix depth. Hidden states are stored
in the dtype the prefix produced; any other dtype is rejected, never cast.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

import torch
from torch import Tensor, nn

from tools.research.dense_suffix_delta import dense_delta as dd

RECORD_KEYS = {"identity", "slice_mask", "hidden", "rope_cos", "rope_sin", "digests"}


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{value.dtype}|{tuple(value.shape)}\n".encode())
    digest.update(value.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def prefix_identity(
    backbone: nn.Module, frozen_prefix_layers: int, source_contract: Mapping[str, Any]
) -> dict[str, Any]:
    if source_contract.get("augment") is not False:
        raise ValueError("Frozen-prefix cache requires a fixed non-augmented source contract")
    config = cast(Any, backbone).config.to_dict()
    dd.suffix_targets(config, frozen_prefix_layers)
    return {
        "method": "dinov3_frozen_prefix_cache_v1",
        "frozen_prefix_layers": frozen_prefix_layers,
        "config_sha256": _json_sha256(config),
        "official_state_sha256": dd.official_state_fingerprint(backbone),
        "source_contract_sha256": _json_sha256(dict(source_contract)),
        "tokens_prefix": 1 + config["num_register_tokens"],
        "hidden_dtype": str(
            cast(Any, backbone.get_submodule("embeddings.patch_embeddings")).weight.dtype
        ),
    }


@torch.no_grad()
def build_record(
    backbone: nn.Module,
    normalized_windows: Tensor,
    slice_mask: Tensor,
    frozen_prefix_layers: int,
    source_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """[S,W,3,H,W] normalized windows + [S,W] mask -> acquired-only prefix record.

    Mirrors ``CompetitiveStudyModel``: only ``images[slice_mask]`` enter the encoder.
    """
    if backbone.training:
        raise ValueError("Prefix cache must be built in eval mode")
    if slice_mask.dtype != torch.bool or slice_mask.shape != normalized_windows.shape[:2]:
        raise ValueError("Cache slice mask must be boolean [series, windows]")
    hidden, (cos, sin) = dd.run_prefix(
        backbone, normalized_windows[slice_mask], frozen_prefix_layers
    )
    record: dict[str, Any] = {
        "identity": prefix_identity(backbone, frozen_prefix_layers, source_contract),
        "slice_mask": slice_mask.clone(),
        "hidden": hidden.clone(),
        "rope_cos": cos.clone(),
        "rope_sin": sin.clone(),
    }
    record["digests"] = {k: tensor_sha256(record[k]) for k in RECORD_KEYS - {"identity", "digests"}}
    return record


def validate_record(
    record: Mapping[str, Any],
    backbone: nn.Module,
    frozen_prefix_layers: int,
    source_contract: Mapping[str, Any],
) -> None:
    if set(record) != RECORD_KEYS:
        raise ValueError("Cache record fields differ from the frozen-prefix contract")
    if record["identity"] != prefix_identity(backbone, frozen_prefix_layers, source_contract):
        raise ValueError("Cache config/official-tensor/source/prefix identity mismatch")
    for key, digest in record["digests"].items():
        if tensor_sha256(record[key]) != digest:
            raise ValueError(f"Cached {key} identity mismatch")
    mask, hidden = record["slice_mask"], record["hidden"]
    if mask.dtype != torch.bool or hidden.shape[0] != int(mask.sum()):
        raise ValueError("Cached mask identity mismatch")
    if str(hidden.dtype) != record["identity"]["hidden_dtype"]:
        raise ValueError("Cached hidden dtype differs from the prefix output dtype")
    patch = int(cast(Any, backbone).config.patch_size)
    grid = hidden.shape[1] - record["identity"]["tokens_prefix"]
    side = int(round(grid**0.5))
    if side * side != grid:
        raise ValueError("Cached token layout differs from a square patch grid")
    probe = torch.zeros(1, 3, side * patch, side * patch, dtype=hidden.dtype)
    cos, sin = dd.rope_context(backbone, probe)
    if not (torch.equal(cos, record["rope_cos"]) and torch.equal(sin, record["rope_sin"])):
        raise ValueError("Cached RoPE identity mismatch")


def run_cached_suffix(
    record: Mapping[str, Any],
    backbone: nn.Module,
    frozen_prefix_layers: int,
    source_contract: Mapping[str, Any],
) -> Tensor:
    """Validated cached record -> acquired-image pooler_output [n_acquired, D]."""
    validate_record(record, backbone, frozen_prefix_layers, source_contract)
    rope = (record["rope_cos"], record["rope_sin"])
    return dd.run_suffix(backbone, record["hidden"], rope, frozen_prefix_layers)[1]


def storage_bytes(
    config: Mapping[str, Any],
    height: int,
    width: int,
    images: int,
    element_bytes: int = 4,
) -> int:
    """Exact hidden-state bytes for ``images`` cached windows (RoPE/masks are negligible)."""
    tokens = (
        1
        + config["num_register_tokens"]
        + (height // config["patch_size"]) * (width // config["patch_size"])
    )
    return images * tokens * config["hidden_size"] * element_bytes


@torch.no_grad()
def verify_by_recompute(
    record: Mapping[str, Any],
    backbone: nn.Module,
    normalized_windows: Tensor,
    frozen_prefix_layers: int,
    source_contract: Mapping[str, Any],
) -> None:
    """Digests catch corruption only; a self-consistent wrong record needs recomputation."""
    validate_record(record, backbone, frozen_prefix_layers, source_contract)
    fresh = build_record(
        backbone, normalized_windows, record["slice_mask"], frozen_prefix_layers, source_contract
    )
    if fresh["digests"] != record["digests"]:
        raise ValueError("Cached hidden states differ from recomputed official prefix")
