"""Exact frozen-prefix feature records for the suffix-adapter recipe; never a lossy cast.

The canonical trainer streams acquired windows through the frozen prefix online and
does not read these records. They exist so a funded cache can be produced and
checked against one exact identity: official config/tensors, prefix depth/execution,
public source policy/index/revision, cache member bytes, slot/window layout and FP32
hidden dtype. ``measure`` reads an actual public index read-only to report the real
acquired-window storage before any funding claim.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
from torch import Tensor, nn

from rsna.competitive import public_input as pi
from rsna.competitive import suffix_adapters as sa

METHOD = "dinov3_frozen_prefix_cache_v2"
RECORD_KEYS = {"identity", "slice_mask", "window_slots", "hidden", "rope_cos", "rope_sin", "digests"}
TENSOR_KEYS = RECORD_KEYS - {"identity", "digests"}
HIDDEN_DTYPE = torch.float32


def tensor_sha256(tensor: Tensor) -> str:
    value = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256(f"{value.dtype}|{tuple(value.shape)}\n".encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def source_identity(policy: dict, index_sha256: str, source_revision: str) -> dict[str, Any]:
    pi.validate_public_input(policy)
    return {
        "public_input_sha256": pi.canonical_sha256(policy),
        "index_sha256": index_sha256,
        "source_revision": source_revision,
        "augment": False,
    }


def member_identity(index: Mapping[str, Any], study_uid: str) -> dict[str, Any]:
    member = index["members"][study_uid]
    return {"study_uid": study_uid, **{key: member[key] for key in ("path", "bytes", "sha256")}}


def _backbone(encoder: nn.Module) -> nn.Module:
    backbone = getattr(encoder, "backbone", None)
    if not isinstance(backbone, nn.Module) or sa.installed_policy(backbone) is None:
        raise ValueError("Prefix records require a suffix-adapted official encoder")
    return backbone


def prefix_identity(encoder: nn.Module, member: Mapping, source: Mapping) -> dict[str, Any]:
    backbone = _backbone(encoder)
    policy = cast(dict, sa.installed_policy(backbone))
    config = cast(Any, backbone).config.to_dict()
    if source.get("augment") is not False:
        raise ValueError("Frozen-prefix records require the fixed non-augmented source")
    return {
        "method": METHOD,
        "frozen_prefix_layers": policy["frozen_prefix_layers"],
        "prefix_execution": sa.PREFIX_EXECUTION,
        "rope": sa.ROPE,
        "config_sha256": pi.canonical_sha256(config),
        "official_state_sha256": sa.official_state_fingerprint(backbone),
        "input_channels": getattr(encoder, "input_channels", None),
        "source": dict(source),
        "member": dict(member),
        "tokens_prefix": 1 + config["num_register_tokens"],
        "hidden_dtype": str(HIDDEN_DTYPE),
    }


def _slot_layout(inputs: Mapping[str, Tensor]) -> tuple[Tensor, Tensor]:
    slot, mask = inputs["series_mask"], inputs["slice_mask"]
    if (
        slot.dtype != torch.bool
        or mask.dtype != torch.bool
        or slot.ndim != 1
        or mask.shape != (len(slot), mask.shape[1])
        or not torch.equal(mask, slot[:, None].expand_as(mask))
        or not slot.any()
    ):
        raise ValueError("Every acquired slot must contribute exactly all its windows")
    return mask, torch.nonzero(mask).to(torch.int64)


@torch.no_grad()
def build_record(
    encoder: nn.Module, inputs: Mapping[str, Tensor], member: Mapping, source: Mapping
) -> dict[str, Any]:
    """Dataset item -> acquired-only FP32 prefix record in ``images[slice_mask]`` order."""
    backbone = _backbone(encoder)
    policy = cast(dict, sa.installed_policy(backbone))
    mask, slots = _slot_layout(inputs)
    images = inputs["images"]
    if images.dtype != HIDDEN_DTYPE or images.shape[:2] != mask.shape:
        raise ValueError("Prefix records require the dataset's float32 windows")
    pixels = cast(Any, encoder)._normalized_rgb(images[mask])
    hidden, (cos, sin) = sa.run_prefix(backbone, pixels, policy["frozen_prefix_layers"])
    if hidden.dtype != HIDDEN_DTYPE:
        raise ValueError("Prefix output dtype differs; refusing any implicit cast")
    record: dict[str, Any] = {
        "identity": prefix_identity(encoder, member, source),
        "slice_mask": mask.clone(),
        "window_slots": slots,
        "hidden": hidden.clone(),
        "rope_cos": cos.clone(),
        "rope_sin": sin.clone(),
    }
    record["digests"] = {key: tensor_sha256(record[key]) for key in TENSOR_KEYS}
    return record


def validate_record(
    record: Mapping[str, Any], encoder: nn.Module, member: Mapping, source: Mapping
) -> None:
    if set(record) != RECORD_KEYS:
        raise ValueError("Prefix record fields differ from the contract")
    if record["identity"] != prefix_identity(encoder, member, source):
        raise ValueError("Prefix record config/official/source/member/prefix identity mismatch")
    if not isinstance(record["digests"], dict) or set(record["digests"]) != TENSOR_KEYS:
        raise ValueError("Prefix record digest set differs")
    for key in TENSOR_KEYS:
        if not isinstance(record[key], Tensor) or tensor_sha256(record[key]) != record["digests"][key]:
            raise ValueError(f"Prefix record {key} identity mismatch")
    mask, slots, hidden = record["slice_mask"], record["window_slots"], record["hidden"]
    if (
        mask.dtype != torch.bool
        or slots.dtype != torch.int64
        or not torch.equal(slots, torch.nonzero(mask))
        or not torch.equal(mask, mask.any(dim=1, keepdim=True).expand_as(mask))
    ):
        raise ValueError("Prefix record slot/window identity mismatch")
    if hidden.dtype != HIDDEN_DTYPE or hidden.ndim != 3 or hidden.shape[0] != len(slots):
        raise ValueError("Prefix record hidden dtype/rows differ")
    backbone = _backbone(encoder)
    config = cast(Any, backbone).config
    grid = hidden.shape[1] - record["identity"]["tokens_prefix"]
    side = int(round(grid**0.5))
    if side * side != grid or hidden.shape[2] != config.hidden_size:
        raise ValueError("Prefix record token layout differs from the pinned grid")
    patch = int(config.patch_size)
    probe = torch.zeros(1, 3, side * patch, side * patch, dtype=hidden.dtype, device=hidden.device)
    cos, sin = sa.rope_context(backbone, probe)
    if not (torch.equal(cos, record["rope_cos"]) and torch.equal(sin, record["rope_sin"])):
        raise ValueError("Prefix record RoPE identity mismatch")


def run_cached_suffix(
    record: Mapping[str, Any], encoder: nn.Module, member: Mapping, source: Mapping
) -> Tensor:
    """Validated record -> acquired-window pooler_output, identical to the online path."""
    validate_record(record, encoder, member, source)
    backbone = _backbone(encoder)
    policy = cast(dict, sa.installed_policy(backbone))
    return sa.run_suffix(
        backbone,
        record["hidden"],
        (record["rope_cos"], record["rope_sin"]),
        policy["frozen_prefix_layers"],
    )[1]


@torch.no_grad()
def verify_by_recompute(
    record: Mapping[str, Any],
    encoder: nn.Module,
    inputs: Mapping[str, Tensor],
    member: Mapping,
    source: Mapping,
) -> None:
    """Digests detect corruption only; a self-consistent wrong record needs recomputation."""
    validate_record(record, encoder, member, source)
    if build_record(encoder, inputs, member, source)["digests"] != record["digests"]:
        raise ValueError("Prefix record differs from the recomputed official prefix")


def window_bytes(config: Mapping[str, Any], side: int = pi.SIDE) -> int:
    tokens = 1 + config["num_register_tokens"] + (side // config["patch_size"]) ** 2
    return tokens * config["hidden_size"] * HIDDEN_DTYPE.itemsize


def measure_storage(
    cache: Path,
    policy: dict,
    source_revision: str,
    config: Mapping[str, Any],
    rosters: Mapping[str, list[str]] | None = None,
) -> dict[str, Any]:
    """Read-only acquired-window FP32 hidden bytes from every verified actual member."""
    index, index_sha = pi.load_public_index(cache, policy, source_revision)
    root = Path(cache) / pi.CACHE_DIRECTORY
    per_window = window_bytes(config)
    acquired: dict[str, int] = {}
    for uid in sorted(index["members"]):
        member = index["members"][uid]
        payload = (root / member["path"]).read_bytes()
        if len(payload) != member["bytes"] or hashlib.sha256(payload).hexdigest() != member["sha256"]:
            raise ValueError("Public triplet member bytes differ from the cache index")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            slot = archive["slot_mask"]
        if slot.dtype != np.bool_ or slot.shape != (pi.SLOTS,) or not slot.any():
            raise ValueError("Public triplet member slot mask differs from the contract")
        acquired[uid] = int(slot.sum())

    def summary(uids) -> dict[str, int]:
        slots = sum(acquired[uid] for uid in uids)
        return {
            "members": len(uids),
            "acquired_slots": slots,
            "acquired_windows": slots * pi.VIEWS,
            "hidden_bytes_fp32": slots * pi.VIEWS * per_window,
            "dense_6x10_bound_bytes_fp32": len(uids) * pi.SLOTS * pi.VIEWS * per_window,
        }

    result: dict[str, Any] = {
        "status": "actual_public_index_read_only_storage_measurement",
        "index_sha256": index_sha,
        "source_revision": source_revision,
        "bytes_per_window_fp32": per_window,
        "all": summary(sorted(acquired)),
        "rope_and_masks_negligible": True,
        "cache_written": False,
    }
    for name, uids in (rosters or {}).items():
        expected = index[f"{name}_uids_sha256"]
        if name not in {"fitting", "evaluation"} or pi.roster_sha256(uids) != expected:
            raise ValueError("Supplied roster differs from the public index roster")
        result[name] = summary(sorted(uids))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Read-only prefix-cache storage measurement")
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--train-config", type=Path, required=True)
    parser.add_argument("--dinov3-config", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--fitting-uids", type=Path)
    parser.add_argument("--evaluation-uids", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rosters = {
        name: json.loads(path.read_text())
        for name, path in (("fitting", args.fitting_uids), ("evaluation", args.evaluation_uids))
        if path is not None
    }
    result = measure_storage(
        args.cache,
        json.loads(args.train_config.read_text())["public_input"],
        args.source_revision,
        json.loads(args.dinov3_config.read_text()),
        rosters,
    )
    with args.output.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
