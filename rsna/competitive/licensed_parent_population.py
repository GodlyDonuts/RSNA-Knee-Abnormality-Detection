"""Join exact immutable pixel shards to the original full fitting target bundle.

The wrapper supplies actual original source/collection receipts. All identities,
pixels, labels and teacher arrays remain private inputs, never constants here.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from rsna.competitive.licensed_pixel_bags import POPULATION_SHA, PixelBags

PROBE_SHA = "3a1567989bf65779aeb19785dafdedef7386fb6bedba5ec45da6d3a67a9a79e3"
TARGET_SHA = "48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285"
PARENT_SOURCE_SHA = "493c7b0d44fe5244b02024448a93f5393cdb073781a068951badb3ffd53fcbde"
CODEC_SHA = "7bdaf41a864b499401ff4a62e6288d30eead3849c0fda5c49cc67b3b2d0e2ab5"
FULL_INDEX_SHA = "5a1a4c77be60a46973c9513b7115ba68edfc7bf76c6205a89de61217a75bf113"
SOURCE_SHAS = (
    "a1e397b02232597f24419d6f8a7641146880d9af23ed51a6d9fd90702e969fc9",
    "de046eb0aa17f63f998d4eb33f23979fd539d911c5593165baf6b18aca469ea6",
    "230f83898475dff858c89f3cdeb38cdc6ad9f8f577366594d2c8bf3022e25934",
    "4e7107a786066b14e4c2bf6a7cec3bb942ed8876610d39d13a0d02dc768c9065",
)
KERNEL_IDS = (139103622, 139103665, 139103730, 139103781)


def producer_proof(actual):
    """Admit original immutable Kaggle bytes independently of a Newton backup.

    Source identity/privacy/input metadata is checked by the authenticated outer
    coordinator. This does not claim a downloaded Newton replica was verified.
    """
    i = actual.get("shard")
    receipt = actual.get("receipt", {})
    if (
        type(i) is not int
        or not 0 <= i < 4
        or actual.get("status") != "COMPLETE"
        or actual.get("version") != 1
        or actual.get("kernel_id") != KERNEL_IDS[i]
        or actual.get("source_sha256") != SOURCE_SHAS[i]
        or receipt.get(
            "all_new_original_volume_mask_and_both_view_tensors_bit_equal_after_roundtrip"
        )
        is not True
        or receipt.get("original32_reused_without_decode_reencode_or_new_archive_copy") is not True
        or receipt.get("reports_clinical_labels_protected_values_read") is not False
        or receipt.get("new_weight_download_or_encoder_optimization") is not False
    ):
        raise ValueError(
            "Actual full original Kaggle producer and every-case parity receipt required"
        )
    return {
        "status": "actual_original_Kaggle_pixel_shard_verified",
        "kernel_id": KERNEL_IDS[i],
        "version": 1,
        "source_sha256": SOURCE_SHAS[i],
        "shard": i,
        "studies": receipt.get("studies"),
        "new_cases": receipt.get("new_original_MRI_decodes"),
        "reused_original32": receipt.get("original32_reused"),
        "global_uid_sha256": receipt.get("global_uid_sha256"),
        "empty_studies": receipt.get("empty_studies"),
        "decode_errors": receipt.get("decode_errors"),
        "all_new_member_pixel_mask_and_both94_input_view_hashes_exact": True,
        "all_fitting_membership_exact": True,
        "original32_retained_no_archive_transfer_or_reencoding": True,
        "original_Kaggle_receipt": receipt,
        **{
            name: receipt.get(name)
            for name in ("index_bytes", "index_sha256", "archive_bytes", "archive_sha256")
        },
        "Newton_replica_collection_or_independent_replay_claim": False,
    }


def validate_proofs(proofs):
    """Require the four completed originals and their actual private byte checks."""
    if not isinstance(proofs, list) or len(proofs) != 4:
        raise ValueError("All four actual original collection proofs required")
    for i, value in enumerate(proofs):
        receipt = value.get("original_Kaggle_receipt", {})
        lo, hi = i * 972, min(3887, (i + 1) * 972)
        if (
            value.get("status")
            not in (
                "actual_full_pixel_shard_collected_verified",
                "actual_original_Kaggle_pixel_shard_verified",
            )
            or value.get("source_sha256") != SOURCE_SHAS[i]
            or value.get("shard") != i
            or value.get("studies") != hi - lo
            or value.get("global_uid_sha256") != POPULATION_SHA
            or value.get("empty_studies") != 0
            or value.get("decode_errors") != 0
            or value.get("all_new_member_pixel_mask_and_both94_input_view_hashes_exact") is not True
            or value.get("all_fitting_membership_exact") is not True
            or value.get("original32_retained_no_archive_transfer_or_reencoding") is not True
            or receipt.get("status") != "actual_licensed_parent_full_fitting_pixel_shard_complete"
            or (receipt.get("shard"), receipt.get("lo"), receipt.get("hi")) != (i, lo, hi)
            or receipt.get("parent_source_sha256") != PARENT_SOURCE_SHA
            or receipt.get("codec_sha256") != CODEC_SHA
            or receipt.get("global_uid_sha256") != POPULATION_SHA
            or receipt.get("original_full_index_sha256") != FULL_INDEX_SHA
            or receipt.get("original_probe_pixel_index_sha256") != PROBE_SHA
            or receipt.get("empty_studies") != 0
            or receipt.get("decode_errors") != 0
        ):
            raise ValueError("Actual exact clean original pixel source/byte receipts required")
        for name in ("index", "archive"):
            size, digest = value.get(name + "_bytes"), value.get(name + "_sha256")
            if (
                type(size) is not int
                or not 0 < size <= (2**20 if name == "index" else 14 * 2**30)
                or not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
                or receipt.get(name + "_bytes") != size
                or receipt.get(name + "_sha256") != digest
            ):
                raise ValueError("Matching actual whole-byte size/SHA receipts required")
    if (
        sum(p.get("new_cases", -3887) for p in proofs) != 3855
        or sum(p.get("reused_original32", -3887) for p in proofs) != 32
    ):
        raise ValueError("Exactly original32 reused and3855 newly decoded studies required")


def locate(inputs, name, size, digest):
    """Search mounted outputs only, pruning raw competition trees before descent."""
    found = []
    for directory in Path(inputs).iterdir():
        if not directory.is_dir():
            continue
        for root, dirs, files in os.walk(directory, followlinks=False):
            if any(
                n in files or n in dirs
                for n in ("train.csv", "test.csv", "train_series", "train_images")
            ):
                dirs[:] = []
                continue
            if len(Path(root).relative_to(directory).parts) >= 5:
                dirs[:] = []
            if name in files:
                p = Path(root) / name
                if not p.is_symlink() and p.stat().st_size == size:
                    if hashlib.sha256(p.read_bytes()).hexdigest() == digest:
                        found.append(p)
    if len(found) != 1:
        raise ValueError("One exact original immutable mounted input required")
    return found[0]


def join_records(probe, shards, proofs):
    """Check original32 reuse, ranges and complete fitting roster before pixels."""
    validate_proofs(proofs)
    if (
        len(probe.get("records", [])) != 32
        or probe.get("parent_source_sha256") != PARENT_SOURCE_SHA
        or probe.get("codec_sha256") != CODEC_SHA
        or len(shards) != 4
    ):
        raise ValueError("Exact original32 and four admitted shard indices required")
    old = {r["study_uid"]: r for r in probe["records"]}
    if len(old) != 32:
        raise ValueError("Unique original32 identities required")
    records = []
    for i, index in enumerate(shards):
        lo, hi = i * 972, min(3887, (i + 1) * 972)
        rows = index.get("records", [])
        if (
            (index.get("shard"), index.get("lo"), index.get("hi")) != (i, lo, hi)
            or len(rows) != hi - lo
            or index.get("global_uid_sha256") != POPULATION_SHA
            or index.get("parent_source_sha256") != PARENT_SOURCE_SHA
            or index.get("codec_sha256") != CODEC_SHA
            or index.get("original_full_feature_index_sha256") != FULL_INDEX_SHA
            or index.get("original_probe_pixel_index_sha256") != PROBE_SHA
        ):
            raise ValueError("Original complete contiguous fitting shard indices required")
        for row in rows:
            if (
                row.get("has_original_MRI_pixels") is not True
                or row.get("preprocessing_error_type") is not None
            ):
                raise ValueError("Every original admitted study must contain clean MRI pixels")
            reused = row["study_uid"] in old
            if row.get("source_key") != ("probe32" if reused else "self"):
                raise ValueError("Original pixel holder identity differs")
            if reused:
                expected = {
                    **old[row["study_uid"]],
                    "source_key": "probe32",
                    "has_original_MRI_pixels": True,
                }
                if row != expected:
                    raise ValueError("Original32 immutable pixel record changed")
            records.append({**row, "holder": "probe32" if reused else "s" + str(i)})
    uids = [r["study_uid"] for r in records]
    if (
        len(uids) != 3887
        or uids != sorted(set(uids))
        or hashlib.sha256(json.dumps(uids, separators=(",", ":")).encode()).hexdigest()
        != POPULATION_SHA
        or sum(r["holder"] == "probe32" for r in records) != 32
    ):
        raise ValueError("Exact original full3887 ordered fitting population required")
    return records


def load(inputs, proofs, render):
    validate_proofs(proofs)
    probe_path = locate(inputs, "index.private.json", 21324, PROBE_SHA)
    paths = [
        locate(inputs, "index.private.json", p["index_bytes"], p["index_sha256"]) for p in proofs
    ]
    rows = join_records(
        json.loads(probe_path.read_text()), [json.loads(p.read_text()) for p in paths], proofs
    )
    target_path = locate(inputs, "targets.private.npz", 483656, TARGET_SHA)
    with np.load(target_path, allow_pickle=False) as arrays:
        if set(arrays.files) != {"uids", "labels", "masks", "parent_logits"}:
            raise ValueError("Exact original private target member schema required")
        targets = {name: arrays[name].copy() for name in arrays.files}
    roots = {
        "probe32": probe_path.parent / "pixels.tar",
        **{"s" + str(i): p.parent / "pixels.tar" for i, p in enumerate(paths)},
    }
    if roots["probe32"].stat().st_size != 326799360 or any(
        roots["s" + str(i)].stat().st_size != proofs[i]["archive_bytes"] for i in range(4)
    ):
        raise ValueError("Original admitted read-only archive sizes required")
    return PixelBags(rows, roots, targets, render)
