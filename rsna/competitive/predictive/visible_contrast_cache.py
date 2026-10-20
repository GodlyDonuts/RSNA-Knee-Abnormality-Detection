"""Prepare one prospective contrast input and reuse the existing control images.

This module does not dispatch work or admit a cohort. A separately admitted
sender can prepare one normalized file in bounded scratch, stream it, and reuse
that scratch for the next study. The destination retains one new context file
per study; a matched ten-coordinate control reuses the original image cache.
No current trainer, serving entrypoint, or running cache calls this module.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path

import numpy as np
import torch

from rsna.competitive.predictive import cache, context_cache, visible_contrast_fast

VERSION = "visible_contrast_one_new_cache_virtual_matched_control_v1"
SOURCE_KEYS = {"candidate_sha256", "fast_helper_sha256", "source_git"}


def matched_control(original, normalized):
    """Reuse original images with precisely the normalized arm's scale metadata.

    Matching the unchanged descriptor targets, supports, masks and first eight
    metadata coordinates is necessary before images can be paired. Callers must
    additionally bind both files to the same original study and raw source.
    Returned dictionaries alias immutable input tensors; neither input changes.
    """
    if len(original) != 2 or len(normalized) != 2:
        raise ValueError("Exactly two complete complementary contexts required")
    result = []
    for old, new in zip(original, normalized, strict=True):
        if set(old) != {*context_cache.KEYS, "supports"} or old.keys() != new.keys():
            raise ValueError("Complete original context fields required")
        for key in ("images", "targets", "observed"):
            if old[key].shape != new[key].shape or old[key].dtype != new[key].dtype:
                raise ValueError("Original and contrast tensor geometry differs")
        for key in ("targets", "observed"):
            if not torch.equal(old[key], new[key]):
                raise ValueError("Contrast targets or masks differ from original")
        for key in ("token_metadata", "region_metadata"):
            a, b = old[key], new[key]
            if (a.ndim != 2 or b.ndim != 2 or a.shape != (len(b), 8)
                    or b.shape[1] != 10 or a.dtype != b.dtype
                    or not torch.equal(a, b[:, :8])):
                raise ValueError("Original eight-coordinate metadata differs")
        a, b = old["supports"], new["supports"]
        if (a.region_count != b.region_count
                or any(not torch.equal(getattr(a, k), getattr(b, k))
                       for k in ("indptr", "regions", "log_prior"))):
            raise ValueError("Complete support priors differ")
        result.append({**old, "token_metadata": new["token_metadata"],
                       "region_metadata": new["region_metadata"]})
    return result


def prepare_one(raw_row, original_row, destination, *, source, cap_bytes,
                side=320, halo=8, context_cap=24, classes=12):
    """Read admitted originals and write only the new normalized variant.

    Owner-node byte proofs, complete membership/terminal checks, source-module
    hashes and total storage funding belong to the caller. This function checks
    the per-study identity, immutable stats, actual target/prior equivalence,
    exclusive output, unchanged RNG, and a full destination tensor roundtrip.
    It returns a private row, including the original control reference; never
    export that row or its identifiers into a public receipt.
    """
    if (set(source) != SOURCE_KEYS
            or not re.fullmatch(r"[0-9a-f]{40}", source["source_git"])
            or any(not re.fullmatch(r"[0-9a-f]{64}", source[k])
                   for k in SOURCE_KEYS - {"source_git"})):
        raise ValueError("Exact candidate, equivalent helper and source pins required")
    if source != source_pins(source["source_git"]):
        raise ValueError("Loaded contrast implementation differs from its pins")
    if (not isinstance(raw_row.get("uid"), str) or not raw_row["uid"]
            or raw_row["uid"] != original_row.get("uid")):
        raise ValueError("The exact same original study must be paired")
    if original_row["provenance"]["original_raw_cache_sha256"] != raw_row["proof"]["sha256"]:
        raise ValueError("The original control belongs to a different raw source")
    if any(original_row["provenance"][key] != value for key, value in
           dict(side=side, halo=halo, context_cap=context_cap, classes=classes).items()):
        raise ValueError("The original context rendering recipe must be unchanged")
    destination = Path(destination)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError("Preserve and reconcile the original destination")
    if type(cap_bytes) is not int or cap_bytes < 1:
        raise ValueError("A separately funded per-study byte bound is required")
    rng = torch.get_rng_state().clone()
    groups, _ = cache.load(raw_row["path"], raw_row["proof"])
    original = context_cache.load(original_row["path"], original_row["proof"],
                                  provenance=original_row["provenance"])
    normalized, _ = visible_contrast_fast.assemble(
        groups, side=side, halo=halo, context_cap=context_cap, classes=classes,
        normalize_images=True)
    matched_control(original, normalized)
    provenance = {"preparation_version": VERSION, **source,
                  "original_raw_cache_sha256": raw_row["proof"]["sha256"],
                  "original_context_sha256": original_row["proof"]["sha256"],
                  "role": "normalized", "metadata_width": 10,
                  "side": side, "halo": halo, "context_cap": context_cap, "classes": classes}
    proof = context_cache.write(destination, normalized, provenance, cap_bytes=cap_bytes)
    restored = context_cache.load(destination, proof, provenance=provenance)
    for a, b in zip(normalized, restored, strict=True):
        if any(not torch.equal(a[k], b[k]) for k in context_cache.KEYS):
            raise ValueError("Serialized contrast input differs; preserve partial output")
    matched_control(original, restored)
    if (cache.signature(raw_row["path"]) != raw_row["proof"]["stat"]
            or cache.signature(original_row["path"]) != original_row["proof"]["stat"]
            or not torch.equal(rng, torch.get_rng_state())):
        raise ValueError("Original source or global RNG changed during preparation")
    return {"uid": raw_row["uid"], "path": str(destination), "proof": proof,
            "provenance": provenance, "original_control": original_row}


def _original_reference(row):
    original_row = row["original_control"]
    provenance = row["provenance"]
    if (row["uid"] != original_row["uid"]
            or provenance["preparation_version"] != VERSION
            or provenance["original_context_sha256"] != original_row["proof"]["sha256"]
            or provenance["original_raw_cache_sha256"] !=
                original_row["provenance"]["original_raw_cache_sha256"]
            or provenance["metadata_width"] != 10 or provenance["role"] != "normalized"):
        raise ValueError("Exact original-control provenance binding differs")
    return original_row


def load_pair(row):
    """Owning-node load of both arms, mainly for admission/equivalence checks.

    Relocation must independently hash both destination files and replace their
    stat proofs. A source-node proof is never sufficient on another node.
    """
    original_row = _original_reference(row)
    provenance = row["provenance"]
    normalized = context_cache.load(row["path"], row["proof"], provenance=provenance)
    original = context_cache.load(original_row["path"], original_row["proof"],
                                  provenance=original_row["provenance"])
    return {"control": matched_control(original, normalized), "normalized": normalized}


def load_role(row, role):
    """Read one arm without decompressing the other arm's image tensors.

    The preparation proof already establishes identical targets and complete
    supports. The control reads the original inputs plus only the new small
    metadata arrays. This relies on the original owning-node byte/stat proofs;
    it is not a substitute for independent verification after transfer.
    """
    if role not in ("control", "normalized"):
        raise ValueError("An explicit matched contrast role is required")
    old_row = _original_reference(row)
    if role == "normalized":
        return context_cache.load(row["path"], row["proof"], provenance=row["provenance"])
    original = context_cache.load(old_row["path"], old_row["proof"], provenance=old_row["provenance"])
    path, proof = Path(row["path"]), row["proof"]
    if (path.is_symlink() or path.stat().st_uid != os.getuid()
            or proof["version"] != context_cache.VERSION or cache.signature(path) != proof["stat"]):
        raise ValueError("Owning-node immutable contrast proof required")
    result = []
    with np.load(path, allow_pickle=False) as arrays:
        if json.loads(arrays["manifest"].tobytes()) != {
            "version": context_cache.VERSION, "source": row["provenance"]
        }:
            raise ValueError("Original contrast provenance differs")
        expected = {"manifest"} | {
            f"c{i}_{key}" for i in (0, 1)
            for key in (*context_cache.KEYS, "indptr", "regions", "log_prior", "region_count")
        }
        if set(arrays.files) != expected:
            raise ValueError("Incomplete original contrast artifact")
        for i, old in enumerate(original):
            fields = {}
            for key in ("token_metadata", "region_metadata"):
                value = torch.from_numpy(arrays[f"c{i}_{key}"].copy())
                if (value.dtype != torch.float32 or value.shape != (len(old[key]), 10)
                        or old[key].shape != (len(old[key]), 8)
                        or not torch.isfinite(value).all()
                        or not torch.equal(value[:, :8], old[key])):
                    raise ValueError("Matched original metadata differs")
                fields[key] = value
            result.append({**old, **fields})
    if (cache.signature(path) != proof["stat"]
            or cache.signature(old_row["path"]) != old_row["proof"]["stat"]):
        raise ValueError("Original inputs changed while loading matched metadata")
    return result


def source_pins(source_git):
    """Build prospective pins from the loaded helpers; callers still admit them."""
    from rsna.competitive.predictive import visible_contrast

    return {"source_git": source_git,
            "candidate_sha256": hashlib.sha256(Path(visible_contrast.__file__).read_bytes()).hexdigest(),
            "fast_helper_sha256": hashlib.sha256(Path(visible_contrast_fast.__file__).read_bytes()).hexdigest()}


def iter_units(raw_rows, original_rows, *, expected_count, roster_sha256,
               scratch_parent, source, per_study_cap_bytes, **recipe):
    """Yield one bounded temporary file at a time for a synchronous sender.

    The consumer must finish reading the yielded file before advancing. Only
    this iterator's new scratch is removed, including on an explicit close or
    exception; original inputs and receiver outputs are never deleted. Callers
    must close the iterator on transport failure (e.g. contextlib.closing).

    This is not a dispatch or receiver: source terminal/roster admission, exact
    destination hashing/EOF and the aggregate output bound remain mandatory
    in a separately accepted transport. No receiver index may keep a temporary
    source path or reuse its inode proof on another node.
    """
    if type(expected_count) is not int or expected_count < 1:
        raise ValueError("An explicitly admitted complete cohort is required")
    if len(raw_rows) != expected_count or len(original_rows) != expected_count:
        raise ValueError("Complete paired cohort length differs")
    uids = [row["uid"] for row in raw_rows]
    if (any(not isinstance(uid, str) or not uid for uid in uids)
            or uids != sorted(set(uids))
            or uids != [row["uid"] for row in original_rows]
            or hashlib.sha256(json.dumps(uids, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
                != roster_sha256):
        raise ValueError("Exact complete original membership and order required")
    for rows in (raw_rows, original_rows):
        paths = [Path(row["path"]).resolve() for row in rows]
        if len(set(paths)) != expected_count:
            raise ValueError("One original file cannot represent multiple studies")
        for row in rows:
            if cache.signature(row["path"]) != row["proof"]["stat"]:
                raise ValueError("An original input changed before streaming")
    for raw, old in zip(raw_rows, original_rows, strict=True):
        if old["provenance"]["original_raw_cache_sha256"] != raw["proof"]["sha256"]:
            raise ValueError("Complete paired cohort raw-source binding differs")
    for ordinal, (raw, old) in enumerate(zip(raw_rows, original_rows, strict=True)):
        with tempfile.TemporaryDirectory(prefix="contrast-stream-unit-", dir=scratch_parent) as root:
            path = Path(root) / "normalized.private.npz"
            row = prepare_one(raw, old, path, source=source, cap_bytes=per_study_cap_bytes, **recipe)
            yield ordinal, row
