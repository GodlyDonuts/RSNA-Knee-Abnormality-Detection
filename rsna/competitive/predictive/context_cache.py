"""Exact FP32 source-masked input cache; no encoder features or diagnosis labels.

A source wrapper must bind original raw-cache SHA, recipe/renderer and membership.
This is an execution optimization for deterministic masked views, not acceptance
of a model/cohort, a changed renderer or an accuracy result.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from rsna.competitive.predictive.cache import signature
from rsna.competitive.predictive.density import Supports

VERSION = "exact_FP32_two_complementary_raw_contexts_complete_targets_priors_v1"
KEYS = ("images", "token_metadata", "region_metadata", "targets", "observed")


def write(path, contexts, provenance, *, cap_bytes):
    if len(contexts) != 2 or not provenance or type(cap_bytes) is not int or cap_bytes < 1:
        raise ValueError("Two admitted contexts and bounded original provenance required")
    arrays = {}
    for i, context in enumerate(contexts):
        if set(context) != {*KEYS, "supports"}:
            raise ValueError("Only actual model inputs, without labels/logits/features")
        for key in KEYS:
            tensor = context[key]
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.device.type != "cpu"
                or tensor.requires_grad
                or tensor.dtype != (torch.bool if key == "observed" else torch.float32)
                or (key != "observed" and not torch.isfinite(tensor).all())
            ):
                raise ValueError("Exact detached finite CPU FP32 inputs required")
            arrays[f"c{i}_{key}"] = tensor.numpy().copy()
        supports = context["supports"]
        if not isinstance(supports, Supports):
            raise ValueError("Complete original support prior required")
        for key in ("indptr", "regions", "log_prior"):
            tensor = getattr(supports, key)
            if tensor.device.type != "cpu" or tensor.requires_grad:
                raise ValueError("Original detached CPU support arrays required")
            arrays[f"c{i}_{key}"] = tensor.numpy().copy()
        arrays[f"c{i}_region_count"] = np.asarray(supports.region_count, dtype="<i8")
    manifest = {"version": VERSION, "source": provenance}
    arrays["manifest"] = np.frombuffer(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(), dtype=np.uint8
    ).copy()
    upper = sum(a.nbytes for a in arrays.values()) + 65536
    if upper > cap_bytes:
        raise ValueError("Uncompressed input exceeds separately funded context cap")
    path = Path(path)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as f:
        np.savez_compressed(f, **arrays)
    if path.stat().st_size > cap_bytes:
        raise ValueError("Preserve unexpected cache growth, do not overwrite")
    with path.open("rb") as f:
        digest = hashlib.file_digest(f, "sha256").hexdigest()
    return {
        "version": VERSION,
        "bytes": path.stat().st_size,
        "sha256": digest,
        "stat": signature(path),
    }


def load(path, proof, *, provenance):
    path = Path(path)
    if (
        path.is_symlink()
        or path.stat().st_uid != os.getuid()
        or proof.get("version") != VERSION
        or signature(path) != proof.get("stat")
    ):
        raise ValueError("Original owning-node immutable context admission required")
    expected = {"manifest"}
    result = []
    with np.load(path, allow_pickle=False) as arrays:
        manifest = json.loads(arrays["manifest"].tobytes())
        if manifest != {"version": VERSION, "source": provenance}:
            raise ValueError("Raw source or exact rendering recipe differs")
        for i in (0, 1):
            context = {}
            for key in KEYS:
                name = f"c{i}_{key}"
                expected.add(name)
                context[key] = torch.from_numpy(arrays[name].copy())
                if context[key].dtype != (torch.bool if key == "observed" else torch.float32):
                    raise ValueError("Original FP32 context dtype changed")
            names = [f"c{i}_{key}" for key in ("indptr", "regions", "log_prior", "region_count")]
            expected.update(names)
            context["supports"] = Supports(
                *(torch.from_numpy(arrays[n].copy()) for n in names[:3]), int(arrays[names[3]])
            )
            result.append(context)
        if set(arrays.files) != expected:
            raise ValueError("Unexpected cached label, feature or missing input")
    if signature(path) != proof["stat"]:
        raise ValueError("Context changed during loading")
    return result
