"""Lossless raw-sample research cache, not the old percentile-normalized cache.

Original decoded integer samples, explicit encoding metadata and exact fixed
descriptors are retained privately. No diagnosis labels, parent features/logits
or pixel-derived normalization. A wrapper must admit original sources/storage.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from rsna.competitive.predictive.cases import RawGroup
from rsna.competitive.predictive.inputs import fixed_metadata_scale, masked_study

VERSION = "original_integer_samples_fixed_encoding_complete_descriptor_cache_v1"


@dataclass(frozen=True)
class SourceGroup:
    raw: torch.Tensor
    bits: torch.Tensor
    signed: torch.Tensor
    monochrome1: torch.Tensor
    aliases: torch.Tensor
    acquisition: torch.Tensor
    ordinal: torch.Tensor


def normalized(g):
    if (
        not isinstance(g, SourceGroup)
        or g.raw.ndim != 3
        or g.raw.dtype != torch.int32
        or g.raw.device.type != "cpu"
        or g.raw.requires_grad
        or not len(g.raw)
        or any(
            x.shape != (len(g.raw),) or x.device.type != "cpu"
            for x in (g.bits, g.signed, g.monochrome1, g.aliases, g.acquisition, g.ordinal)
        )
        or any(x.dtype != torch.int64 for x in (g.bits, g.aliases, g.acquisition, g.ordinal))
        or g.signed.dtype != torch.bool
        or g.monochrome1.dtype != torch.bool
    ):
        raise ValueError(
            "Original integer samples and original per-frame encoding/source metadata required"
        )
    return torch.stack(
        [
            fixed_metadata_scale(
                raw,
                bits_stored=int(g.bits[i]),
                signed=bool(g.signed[i]),
                photometric="MONOCHROME1" if g.monochrome1[i] else "MONOCHROME2",
            )
            for i, raw in enumerate(g.raw)
        ]
    )


def signature(path):
    s = Path(path).stat()
    return [s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def write(path, groups, provenance, *, cap_bytes):
    """Exclusive original-source cache; no quantization or source deletion.

    The source wrapper must check original DICOM identity/size/SHA, positive
    rescale slope, geometry and fitting membership before calling this method.
    Expected source proofs remain private in provenance. No cache authorizes
    labels, source mixing or a competition candidate.
    """
    if not groups or not provenance or type(cap_bytes) is not int or cap_bytes < 1:
        raise ValueError("Explicit original source provenance and separately funded cap required")
    path = Path(path)
    arrays = {}
    total_regions = 0
    for i, g in enumerate(groups):
        frames = normalized(g)
        view = masked_study(frames, g.aliases, g.acquisition, g.ordinal, pattern=0)
        prefix = f"g{i:04d}_"
        # Signed samples retain their original two's-complement bits. Bounds were
        # checked by fixed_metadata_scale; decoding below restores signed values.
        arrays[prefix + "raw"] = (g.raw.numpy() & 65535).astype("<u2")
        for name in ("bits", "signed", "monochrome1", "aliases", "acquisition", "ordinal"):
            arrays[prefix + name] = getattr(g, name).numpy().copy()
        arrays[prefix + "descriptors"] = view.descriptors.numpy().copy()
        total_regions += len(view.descriptors)
    manifest = {
        "version": VERSION,
        "groups": len(groups),
        "regions": total_regions,
        "source": provenance,
    }
    arrays["manifest"] = np.frombuffer(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(), dtype=np.uint8
    ).copy()
    # Conservative DEFLATE/ZIP overhead bound, admitted before a byte is written.
    # This rejects undersized budgets rather than truncating original samples.
    uncompressed = sum(a.nbytes for a in arrays.values())
    if uncompressed + uncompressed // 100 + len(arrays) * 1024 + 65536 > cap_bytes:
        raise ValueError("Funded cache cap smaller than conservative lossless payload bound")
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as f:
        np.savez_compressed(f, **arrays)
    if path.stat().st_size > cap_bytes:
        raise ValueError("Unexpected cache growth; retain original, no overwrite or deletion")
    with path.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    return {
        "version": VERSION,
        "bytes": path.stat().st_size,
        "sha256": sha,
        "stat": signature(path),
        "groups": len(groups),
        "regions": total_regions,
        "uncompressed_array_bytes": uncompressed,
        "clinical_or_training_acceptance": False,
    }


def load(path, proof):
    """Reuse original owning-node whole-byte proof; verify current immutable stats.

    A new node must independently verify bytes and admit its own stat signature.
    Do not pass a compiler proof from another inode or bypass the source wrapper.
    """
    path = Path(path)
    if (
        path.is_symlink()
        or path.stat().st_uid != os.getuid()
        or proof.get("version") != VERSION
        or signature(path) != proof.get("stat")
    ):
        raise ValueError("Owning-node original whole-byte admission/stat binding required")
    result = []
    with np.load(path, allow_pickle=False) as z:
        manifest = json.loads(z["manifest"].tobytes())
        if manifest["version"] != VERSION or manifest["groups"] != proof["groups"]:
            raise ValueError("Exact original cache schema/bindings required")
        expected = {"manifest"}
        for i in range(manifest["groups"]):
            prefix = f"g{i:04d}_"
            members = {}
            for name in (
                "raw",
                "bits",
                "signed",
                "monochrome1",
                "aliases",
                "acquisition",
                "ordinal",
                "descriptors",
            ):
                expected.add(prefix + name)
                members[name] = torch.from_numpy(z[prefix + name].copy())
            raw = members.pop("raw").to(torch.int32)
            if ((raw < 0) | (raw > 65535)).any():
                raise ValueError("Stored integer samples outside lossless encoding")
            raw = torch.where(members["signed"][:, None, None] & (raw >= 32768), raw - 65536, raw)
            descriptors = members.pop("descriptors")
            group = SourceGroup(raw, **members)
            frames = normalized(group)
            result.append(
                RawGroup(frames, group.aliases, group.acquisition, group.ordinal, descriptors)
            )
        if set(z.files) != expected or sum(len(g.descriptors) for g in result) != proof["regions"]:
            raise ValueError("Only complete original groups and descriptor rosters accepted")
    if signature(path) != proof["stat"]:
        raise ValueError("Original cache changed during input loading")
    return result, manifest["source"]
