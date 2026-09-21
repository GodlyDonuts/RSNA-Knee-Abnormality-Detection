"""Recover an explicitly reviewed two-file source correction into a new cache tree.

The pinned baseline, original source bytes and strict decoder remain unchanged.
A successful migration is not a full DICOM audit or a training release.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import shutil
import zlib
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import pydicom
import torch

from rsna.audit import inventory_fingerprint, validate_inventory
from rsna.preprocess import Source, cache_series
from rsna.schema import CACHE_VERSION

KEYS = ["split", "study_uid", "series_uid"]


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _binding(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": _sha(path), "bytes": path.stat().st_size}


def _safe_path(root: Path, name: object) -> Path:
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Expected a relative source/evidence path")
    path = (root / name).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Source/evidence path escapes its root")
    return path


def _verify(path: Path, digest: object, size: object) -> None:
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(c not in "0123456789abcdef" for c in digest)
        or type(size) is not int
        or size <= 0
        or not path.is_file()
        or path.stat().st_size != size
        or _sha(path) != digest
    ):
        raise ValueError(f"Missing or changed source/cache/evidence: {path}")


def _revision(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError("Source revisions must be pinned lowercase commit SHAs")
    return value


def _read_json(path: Path) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    value = json.loads(path.read_text(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def validate_source_correction(
    baseline_inventory: Path,
    patch_path: Path,
    receipt_path: Path,
    artifact_root: Path,
    official_source_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict, Path, set[tuple[str, str, str]]]:
    """Validate the immutable plan/receipt and exact full inventory delta offline."""
    patch, receipt = _read_json(patch_path), _read_json(receipt_path)
    if (
        patch.get("schema_version") != 1
        or patch.get("status") != "accepted_exact_authoritative_source_replacement"
        or patch.get("original_pinned_revision_preserved") is not True
        or patch.get("decoder_changed") is not False
        or patch.get("repo_private_verified") is not True
    ):
        raise ValueError("An accepted, preserved, private source correction is required")
    identities = [patch.get("agent_id"), patch.get("independent_agent_id")]
    if (
        any(not isinstance(value, str) or not value.strip() for value in identities)
        or len(set(identities)) != 2
    ):
        raise ValueError("Source correction requires two distinct assessor identities")
    baseline_revision = _revision(patch.get("baseline_source_revision"))
    parent_revision = _revision(patch.get("publication_parent_revision"))
    corrected_revision = _revision(receipt.get("corrected_source_revision"))
    if corrected_revision in {baseline_revision, parent_revision}:
        raise ValueError("Corrected revision must be a distinct immutable commit")
    if (
        receipt.get("schema_version") != 1
        or receipt.get("status") != "verified_private_hf_source_correction"
        or receipt.get("baseline_source_revision") != baseline_revision
        or receipt.get("publication_parent_revision") != parent_revision
        or receipt.get("repo_id") != patch.get("repo_id")
        or receipt.get("private") is not True
        or receipt.get("remote_objects_verified") is not True
        or receipt.get("source_correction_plan_sha256") != _sha(patch_path)
    ):
        raise ValueError("Verified private publication receipt does not bind this correction")
    evidence = patch.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Source correction requires preserved bound evidence")
    for item in evidence:
        path = _safe_path(artifact_root, item["path"])
        _verify(path, item["sha256"], item["bytes"])
    baseline = pd.read_parquet(baseline_inventory)
    validate_inventory(baseline)
    if (baseline.groupby("series_uid").study_uid.nunique() != 1).any():
        raise ValueError("Series UID appears in multiple studies")
    if not baseline.sha256.str.fullmatch(r"[0-9a-f]{64}").all():
        raise ValueError("Inventory source hashes must be exact SHA256 digests")
    if inventory_fingerprint(baseline) != patch.get("baseline_inventory_fingerprint_sha256"):
        raise ValueError("Baseline inventory differs from the accepted correction")
    corrected_reference = patch["corrected_inventory"]
    corrected_path = _safe_path(patch_path.parent, corrected_reference["path"])
    _verify(corrected_path, corrected_reference["sha256"], corrected_reference["bytes"])
    if receipt.get("corrected_inventory_sha256") != corrected_reference["sha256"]:
        raise ValueError("Receipt corrected inventory hash differs")
    corrected = pd.read_parquet(corrected_path)
    validate_inventory(corrected)
    if list(corrected.columns) != list(baseline.columns):
        raise ValueError("Corrected inventory schema differs from baseline")
    changes = patch.get("changes")
    if not isinstance(changes, list) or len(changes) != 2:
        raise ValueError("This recovery requires exactly two explicit source replacements")
    expected = baseline.set_index("path", drop=False).copy(deep=True)
    seen: set[str] = set()
    affected: set[tuple[str, str, str]] = set()
    for change in changes:
        name = change["path"]
        if name in seen or name not in expected.index:
            raise ValueError("Duplicate or unknown source correction path")
        seen.add(name)
        before, after = change["before"], change["after"]
        if set(before) != set(baseline.columns) or set(after) != set(baseline.columns):
            raise ValueError("Correction must bind the entire original and replacement row")
        if expected.loc[name].to_dict() != before:
            raise ValueError("Correction before row differs from baseline")
        if {key for key in before if before[key] != after[key]} != {"sha256", "bytes"}:
            raise ValueError("Only exact SHA256/size changes are permitted")
        for field in ["path", "study_uid", "series_uid", "sop_uid"]:
            if change[field] != before[field] or after[field] != before[field]:
                raise ValueError("Correction source path/SOP/series/study identity differs")
        official_path = _safe_path(artifact_root, change["official_local_path"])
        _verify(official_path, after["sha256"], after["bytes"])
        data = official_path.read_bytes()
        if change.get("official_crc32") != f"{zlib.crc32(data) & 0xFFFFFFFF:08x}":
            raise ValueError("Official source CRC differs from the accepted evidence")
        ds = pydicom.dcmread(io.BytesIO(data))
        if (str(ds.StudyInstanceUID), str(ds.SeriesInstanceUID), str(ds.SOPInstanceUID)) != tuple(
            after[k] for k in ["study_uid", "series_uid", "sop_uid"]
        ):
            raise ValueError("Authoritative original has incorrect DICOM UIDs")
        pixels = ds.pixel_array
        if (
            list(pixels.shape) != change["official_shape"]
            or str(pixels.dtype) != change["official_dtype"]
        ):
            raise ValueError("Authoritative original decode shape/dtype differs")
        expected.loc[name, ["sha256", "bytes"]] = [after["sha256"], after["bytes"]]
        affected.add(tuple(after[k] for k in KEYS))
    if len(affected) != 2:
        raise ValueError("Expected two distinct affected series")
    try:
        pd.testing.assert_frame_equal(
            expected.reset_index(drop=True).sort_values("path").reset_index(drop=True),
            corrected.sort_values("path").reset_index(drop=True),
        )
    except AssertionError as exc:
        raise ValueError(
            "Corrected inventory has changes beyond the exact two replacements"
        ) from exc
    if inventory_fingerprint(corrected) != patch.get("corrected_inventory_fingerprint_sha256"):
        raise ValueError("Corrected inventory fingerprint differs")
    affected_mask = np.zeros(len(corrected), dtype=bool)
    for split, study, series in affected:
        affected_mask |= (
            (corrected.split == split)
            & (corrected.study_uid == study)
            & (corrected.series_uid == series)
        ).to_numpy()
    affected_rows = corrected.loc[affected_mask]
    counts = {
        "n_changed_source_files": 2,
        "n_source_files": len(baseline),
        "n_unchanged_source_files": len(baseline) - 2,
        "n_affected_series": 2,
        "n_affected_series_slices": len(affected_rows),
        "source_bytes_before": int(baseline.bytes.sum()),
        "source_bytes_after": int(corrected.bytes.sum()),
    }
    if any(patch.get(key) != value for key, value in counts.items()):
        raise ValueError("Correction coverage/byte counts differ")
    changed_paths = receipt.get("changed_source_paths")
    if not isinstance(changed_paths, list) or len(changed_paths) != 2 or set(changed_paths) != seen:
        raise ValueError("Receipt changed-source membership differs")
    uploads = receipt.get("uploaded_files")
    if not isinstance(uploads, list) or any(not isinstance(row, dict) for row in uploads):
        raise ValueError("Receipt needs verified remote object references")
    uploaded = {item["path"]: item for item in uploads}
    if len(uploaded) != len(uploads):
        raise ValueError("Duplicate uploaded object references")
    if {name for name in uploaded if name.endswith(".dcm")} != seen:
        raise ValueError("Remote DICOM replacements differ from the exact accepted changes")
    for change in changes:
        item = uploaded[change["path"]]
        if (item.get("sha256"), item.get("bytes")) != (
            change["after"]["sha256"],
            change["after"]["bytes"],
        ):
            raise ValueError("Remote authoritative object hash/size differs")
    expected_local_paths = set(affected_rows.path)
    found_local_paths = {
        str(path.relative_to(official_source_root)) for path in official_source_root.rglob("*.dcm")
    }
    if expected_local_paths != found_local_paths:
        raise ValueError("Effective local source tree must exactly cover both affected series")
    for row in affected_rows.itertuples(index=False):
        path = _safe_path(official_source_root, row.path)
        _verify(path, row.sha256, int(cast(int, row.bytes)))
        ds = pydicom.dcmread(path, stop_before_pixels=True)
        if (str(ds.StudyInstanceUID), str(ds.SeriesInstanceUID), str(ds.SOPInstanceUID)) != (
            row.study_uid,
            row.series_uid,
            row.sop_uid,
        ):
            raise ValueError("Effective local source DICOM UID mismatch")
    return baseline, corrected, patch, receipt, corrected_path, affected


def _cache_identity(meta: dict, frame: pd.DataFrame, revision: str, path: Path) -> None:
    key = tuple(frame.iloc[0][KEYS])
    if (meta["split"], meta["study_uid"], meta["series_uid"]) != key:
        raise ValueError("Baseline cache study/series identity mismatch")
    if meta["source_revision"] != revision or meta["cache_version"] != CACHE_VERSION:
        raise ValueError("Baseline cache revision/transform differs")
    if meta["cache_path"] != "/".join([*key[:2], key[2] + ".npz"]):
        raise ValueError("Baseline cache relative path differs")
    if (
        meta["n_slices"] != len(frame)
        or len(meta["sop_uids"]) != len(frame)
        or len(set(meta["sop_uids"])) != len(frame)
        or len(meta["slice_sha256"]) != len(frame)
        or dict(zip(meta["sop_uids"], meta["slice_sha256"]))
        != dict(zip(frame.sop_uid, frame.sha256))
        or meta["path_fingerprint"]
        != hashlib.sha256("\n".join(sorted(frame.path)).encode()).hexdigest()
    ):
        raise ValueError("Baseline cache source hash/path/SOP membership differs")
    if (
        len(meta["original_shape"]) != 3
        or meta["original_shape"][0] != len(frame)
        or any(type(n) is not int or n <= 0 for n in meta["original_shape"])
        or type(meta["side"]) is not int
        or meta["side"] < 16
        or len(meta["positions"]) != len(frame)
        or len(meta["pixel_sha256"]) != len(frame)
        or len(meta["pixel_nonconstant"]) != len(frame)
        or meta["pixel_hash_definition"] != "shape_i64le_and_decoded_samples_f64le_before_rescale"
    ):
        raise ValueError("Baseline cache slice/shape/native evidence coverage differs")
    if not path.is_file() or path.stat().st_size != meta["cache_bytes"]:
        raise ValueError("Missing or resized baseline cache")


def _reserve(path: Path, minimum: float) -> None:
    if not np.isfinite(minimum) or minimum < 3:
        raise ValueError("Disk reserve must be at least 3 GiB; real recovery defaults to 20 GiB")
    existing = path
    while not existing.exists():
        existing = existing.parent
    if shutil.disk_usage(existing).free < minimum * 1024**3:
        raise OSError("Insufficient free space for the recovery reserve")


def recover_source_cache(
    baseline_inventory: Path,
    baseline_cache: Path,
    patch_path: Path,
    receipt_path: Path,
    artifact_root: Path,
    official_source_root: Path,
    output: Path,
    min_free_gib: float = 20,
) -> dict:
    """Create a new effective inventory/cache; never rewrite baseline files."""
    if output.exists():
        raise ValueError("Recovery output must be a new directory")
    for protected in [baseline_cache, baseline_inventory, official_source_root]:
        if output.resolve().is_relative_to(protected.resolve()):
            raise ValueError("Recovery output must not be inside a protected source/cache")
    _reserve(output, min_free_gib)
    inputs = [_binding(p) for p in [patch_path, receipt_path]]
    baseline, corrected, patch, receipt, corrected_path, affected = validate_source_correction(
        baseline_inventory, patch_path, receipt_path, artifact_root, official_source_root
    )
    inputs.append(_binding(corrected_path))
    baseline_revision, new_revision = (
        patch["baseline_source_revision"],
        receipt["corrected_source_revision"],
    )
    old_groups = {tuple(key): frame for key, frame in baseline.groupby(KEYS)}
    new_groups = {tuple(key): frame for key, frame in corrected.groupby(KEYS)}
    expected_keys = set(old_groups)
    npz_keys = {
        tuple(path.relative_to(baseline_cache).parts[:2]) + (path.stem,)
        for path in baseline_cache.glob("*/*/*.npz")
    }
    json_keys = {
        tuple(path.relative_to(baseline_cache).parts[:2]) + (path.stem,)
        for path in baseline_cache.glob("*/*/*.json")
    }
    if npz_keys != json_keys or npz_keys - expected_keys or expected_keys - affected - npz_keys:
        raise ValueError("Baseline cache has orphan, extra or missing unchanged series")
    reuse = []
    sides = set()
    for key in sorted(expected_keys - affected):
        old_frame, new_frame = old_groups[key], new_groups[key]
        if (
            not old_frame.sort_values("path")
            .reset_index(drop=True)
            .equals(new_frame.sort_values("path").reset_index(drop=True))
        ):
            raise ValueError("Changed-source series cannot be reused")
        path = baseline_cache / key[0] / key[1] / (key[2] + ".npz")
        meta_path = path.with_suffix(".json")
        meta = _read_json(meta_path)
        _cache_identity(meta, old_frame, baseline_revision, path)
        sides.add(meta["side"])
        reuse.append((key, path, _binding(meta_path)))
    if len(sides) != 1:
        raise ValueError("Recovery requires one established unchanged-cache resolution")
    side = sides.pop()
    parent = output.parent
    while not parent.exists():
        parent = parent.parent
    if baseline_cache.stat().st_dev != parent.stat().st_dev:
        raise ValueError("Hardlink cache reuse requires output on the baseline filesystem")
    for reference in inputs:
        _verify(Path(reference["path"]), reference["sha256"], reference["bytes"])
    output.mkdir(parents=True, exist_ok=False)
    cache = output / "cache"
    cache.mkdir()
    status = {
        "schema_version": 1,
        "status": "in_progress",
        "baseline_source_revision": baseline_revision,
        "corrected_source_revision": new_revision,
        "cache_version": CACHE_VERSION,
        "cache_side": side,
        "n_reused_series": 0,
        "n_fresh_series": 0,
        "n_cache_slices": 0,
        "expected_series": len(new_groups),
        "expected_slices": len(corrected),
        "source_files_modified": False,
        "baseline_cache_modified": False,
        "full_audit_complete": False,
        "training_ready": False,
        "input_bindings": inputs,
    }
    status_path = output / "source_recovery_status.json"

    def save_status():
        temporary = status_path.with_suffix(".partial.json")
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(status_path)

    save_status()
    try:
        _reserve(output, min_free_gib)
        shutil.copyfile(corrected_path, output / "inventory_effective.parquet")
        _verify(
            output / "inventory_effective.parquet",
            patch["corrected_inventory"]["sha256"],
            patch["corrected_inventory"]["bytes"],
        )
        with (output / "cache_recovery_lineage.jsonl").open("x") as lineage:
            for key, old_path, old_meta_reference in reuse:
                _verify(
                    Path(old_meta_reference["path"]),
                    old_meta_reference["sha256"],
                    old_meta_reference["bytes"],
                )
                meta = _read_json(old_path.with_suffix(".json"))
                _verify(old_path, meta["cache_sha256"], meta["cache_bytes"])
                destination = cache / meta["cache_path"]
                destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(old_path, destination)
                rebased = {**meta, "source_revision": new_revision}
                destination.with_suffix(".json").write_text(json.dumps(rebased, sort_keys=True))
                record = {
                    "operation": "hardlink_immutable_npz_copy_rebased_metadata",
                    "split": key[0],
                    "study_uid": key[1],
                    "series_uid": key[2],
                    "n_slices": meta["n_slices"],
                    "baseline_metadata": old_meta_reference,
                    "new_metadata": _binding(destination.with_suffix(".json")),
                    "cache_sha256": meta["cache_sha256"],
                    "cache_bytes": meta["cache_bytes"],
                    "path_fingerprint": meta["path_fingerprint"],
                    "baseline_source_revision": baseline_revision,
                    "corrected_source_revision": new_revision,
                    "source_content_membership_unchanged": True,
                    "metadata_changed_fields": ["source_revision"],
                }
                lineage.write(json.dumps(record, sort_keys=True) + "\n")
                status["n_reused_series"] += 1
                status["n_cache_slices"] += meta["n_slices"]
                if status["n_reused_series"] % 1000 == 0:
                    _reserve(output, min_free_gib)
                    save_status()
                    print(
                        f"Verified/reused {status['n_reused_series']}/{len(reuse)} series",
                        flush=True,
                    )
            source = Source(official_source_root)
            for key in sorted(affected):
                frame = new_groups[key]
                # Recheck every local file immediately before the unchanged decoder reads it.
                for row in frame.itertuples(index=False):
                    _verify(
                        _safe_path(official_source_root, row.path),
                        row.sha256,
                        int(cast(int, row.bytes)),
                    )
                meta = cache_series(
                    source, frame.path.tolist(), cache, side, new_revision, min_free_gib
                )
                destination = cache / meta["cache_path"]
                _cache_identity(meta, frame, new_revision, destination)
                _verify(destination, meta["cache_sha256"], meta["cache_bytes"])
                lineage.write(
                    json.dumps(
                        {
                            "operation": "fresh_full_series_decode_from_effective_authoritative_sources",
                            "split": key[0],
                            "study_uid": key[1],
                            "series_uid": key[2],
                            "n_slices": meta["n_slices"],
                            "new_metadata": _binding(destination.with_suffix(".json")),
                            "cache_sha256": meta["cache_sha256"],
                            "cache_bytes": meta["cache_bytes"],
                            "source_revision": new_revision,
                            "changed_source_paths": sorted(
                                change["path"]
                                for change in patch["changes"]
                                if change["series_uid"] == key[2]
                            ),
                            "original_source_samples_reconstructed_or_padded": False,
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                status["n_fresh_series"] += 1
                status["n_cache_slices"] += len(frame)
                save_status()
        for reference in inputs:
            _verify(Path(reference["path"]), reference["sha256"], reference["bytes"])
        for _, _, reference in reuse:
            _verify(Path(reference["path"]), reference["sha256"], reference["bytes"])
        if (
            inventory_fingerprint(pd.read_parquet(baseline_inventory))
            != patch["baseline_inventory_fingerprint_sha256"]
        ):
            raise ValueError("Baseline inventory changed during recovery")
        found_npz = {
            tuple(p.relative_to(cache).parts[:2]) + (p.stem,) for p in cache.glob("*/*/*.npz")
        }
        found_json = {
            tuple(p.relative_to(cache).parts[:2]) + (p.stem,) for p in cache.glob("*/*/*.json")
        }
        if (
            found_npz != set(new_groups)
            or found_json != found_npz
            or status["n_cache_slices"] != len(corrected)
        ):
            raise ValueError("Recovered cache coverage/counts are incomplete")
        status.update(
            status="cache_recovery_complete_full_audit_required",
            n_effective_source_files=len(corrected),
            effective_source_bytes=int(corrected.bytes.sum()),
            effective_inventory=_binding(output / "inventory_effective.parquet"),
            lineage=_binding(output / "cache_recovery_lineage.jsonl"),
            mandatory_next_step="Run rsna.audit against this full effective inventory/cache and corrected source revision; enforce all duplicate/leakage and label-release gates afterward.",
        )
        save_status()
        return status
    except Exception as exc:
        status.update(
            status="failed_partial_output_preserved",
            failure_type=type(exc).__name__,
            failure=str(exc),
        )
        save_status()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-inventory", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--patch", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--official-source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-free-gib", type=float, default=20)
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(
        json.dumps(
            recover_source_cache(
                args.baseline_inventory,
                args.baseline_cache,
                args.patch,
                args.receipt,
                args.artifact_root,
                args.official_source_root,
                args.output,
                args.min_free_gib,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
