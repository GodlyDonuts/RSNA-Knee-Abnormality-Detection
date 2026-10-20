"""Pure application of the already accepted two-file authoritative correction.

No decoding, downloads, file writes, exclusions, label changes or padding occur
here. Returned member-level lineage contains private source identifiers and must
remain private. Unchanged source records are shared, not copied or modified.
"""

import hashlib
import json

PLAN_SHA = "68b44c1412ced183db223220f39cd4a6e071d40515df4c7b75b40a96ba6d0f3d"
RECEIPT_SHA = "2c88c4f53cb3d2332e8079e1cd353eede5a601172febf0ab0b74ce8161f9fbb4"
ORIGINAL_REVISION = "52f995f4568e36bfa5fbbaf69a89af41226b4f5d"
CORRECTED_REVISION = "0b5e410288232f570c69ba4f44723356e6ae6ab7"
IDENTITY = ("split", "study_uid", "series_uid", "sop_uid", "path")
EXPECTED = {
    "cfe23f2a97ba3fc07066bfed047c6701d9982a54e4913205a6c30366906a455c": (
        411052, "68efac14c0550118b1eb2ce75601f3b5cdf13a824ca83628ef201e260a4b7a9a",
        820652, [640, 640],
    ),
    "c057554899ba81249d280e880226bf28b089b1e33235e3cfd101bfc36a399d7a": (
        174560, "ab491d18a6b3fae57a014c27679f32afc526e4d336373e4354f81b8a4b7de2d4",
        347616, [416, 416],
    ),
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(body):
    return hashlib.sha256(body).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def validate(plan_body, receipt_body):
    """Bind exact historical accepted evidence, including verified uploaded bytes."""
    require(isinstance(plan_body, bytes) and isinstance(receipt_body, bytes),
            "Original plan and receipt bytes required")
    require(sha(plan_body) == PLAN_SHA and sha(receipt_body) == RECEIPT_SHA,
            "Exact accepted source-correction evidence changed")
    plan, receipt = json.loads(plan_body), json.loads(receipt_body)
    require(
        plan["status"] == "accepted_exact_authoritative_source_replacement"
        and receipt["status"] == "verified_private_hf_source_correction"
        and plan["schema_version"] == receipt["schema_version"] == 1
        and plan["baseline_source_revision"] == receipt["baseline_source_revision"]
        == ORIGINAL_REVISION
        and receipt["corrected_source_revision"] == CORRECTED_REVISION
        and plan["repo_id"] == receipt["repo_id"] == "GodlyDonuts/RSNA-Data"
        and plan["repo_private_verified"] is receipt["private"] is True
        and receipt["remote_objects_verified"] is True
        and receipt["source_correction_plan_sha256"] == PLAN_SHA
        and receipt["corrected_inventory_sha256"] == plan["corrected_inventory"]["sha256"]
        and plan["original_pinned_revision_preserved"] is True
        and plan["decoder_changed"] is False and plan["clinical_targets_changed"] is False
        and plan["n_changed_source_files"] == len(plan["changes"]) == 2,
        "Accepted authoritative correction contract differs",
    )
    changes = {}
    for change in plan["changes"]:
        before, after = change["before"], change["after"]
        require(
            set(before) == set(after)
            and {key for key in before if before[key] != after[key]} == {"sha256", "bytes"}
            and all(before[key] == after[key] == change[key] for key in IDENTITY[1:])
            and before["sha256"] in EXPECTED,
            "Only the two unchanged-identity source payloads may differ",
        )
        old_size, new_sha, new_size, shape = EXPECTED[before["sha256"]]
        require(
            before["bytes"] == old_size and after["sha256"] == new_sha
            and after["bytes"] == new_size and change["official_shape"] == shape
            and change["official_dtype"] == "uint16",
            "Exact authoritative pixel payload differs",
        )
        uploaded = [row for row in receipt["uploaded_files"] if row["path"] == change["path"]]
        require(
            len(uploaded) == 1 and uploaded[0]["sha256"] == new_sha
            and uploaded[0]["bytes"] == new_size and change["path"] not in changes,
            "Original independently verified corrected object required",
        )
        changes[change["path"]] = change
    require(
        set(changes) == set(receipt["changed_source_paths"])
        and len(receipt["changed_source_paths"]) == 2
        and {c["before"]["sha256"] for c in changes.values()} == set(EXPECTED),
        "Exact two-file correction membership required",
    )
    return changes


def apply(records, plan_body, receipt_body):
    """Return both source tables and private lineage, altering only two size/SHA pairs.

    The caller must separately admit its original full fitting roster and retain
    original bytes. Both known changes must occur exactly once; this cannot be
    used to silently create a partial-source or already-corrected source table.
    """
    changes = validate(plan_body, receipt_body)
    require(isinstance(records, list) and records, "Original ordered source records required")
    corrected = list(records)
    seen_uids, seen_paths, applied = set(), set(), []
    original_bytes, member_count = 0, 0
    known_bad_hashes = set(EXPECTED)
    for source_ordinal, record in enumerate(records):
        uid, rows = record["uid"], record["source_files"]
        require(uid not in seen_uids and isinstance(rows, list) and rows,
                "Original unique study records with source members required")
        seen_uids.add(uid)
        new_rows = None
        for member_ordinal, row in enumerate(rows):
            path = row["path"]
            require(path not in seen_paths and row["study_uid"] == uid,
                    "Original source membership or study identity differs")
            require(type(row["bytes"]) is int and row["bytes"] > 0,
                    "Exact positive original source bytes required")
            seen_paths.add(path)
            original_bytes += row["bytes"]
            member_count += 1
            if path not in changes:
                require(row["sha256"] not in known_bad_hashes,
                        "Known defective source hash has a different original identity")
                continue
            before, after = changes[path]["before"], changes[path]["after"]
            require(
                all(row[key] == before[key] for key in (*IDENTITY, "bytes", "sha256"))
                and all(row[key] == before[key] for key in before if key in row),
                "Original correction member differs from its exact before state",
            )
            effective = {**row, "bytes": after["bytes"], "sha256": after["sha256"]}
            if new_rows is None:
                new_rows = list(rows)
            new_rows[member_ordinal] = effective
            applied.append({
                "source_ordinal": source_ordinal, "member_ordinal": member_ordinal,
                "original_revision": ORIGINAL_REVISION, "effective_revision": CORRECTED_REVISION,
                "before": dict(row), "after": dict(effective),
            })
        if new_rows is not None:
            corrected[source_ordinal] = {**record, "source_files": new_rows}
    require(len(applied) == 2, "Both exact authoritative corrections must occur once")
    delta = sum(row["after"]["bytes"] - row["before"]["bytes"] for row in applied)
    require(delta == 582656, "Exact two-file size delta differs")
    lineage = {
        "status": "exact_two_authoritative_source_rows_overlaid_without_decoder_change",
        "source_correction_plan_sha256": PLAN_SHA, "source_correction_receipt_sha256": RECEIPT_SHA,
        "original_revision": ORIGINAL_REVISION, "corrected_revision": CORRECTED_REVISION,
        "studies": len(records), "source_members": member_count, "changed_source_members": 2,
        "original_source_bytes": original_bytes, "effective_source_bytes": original_bytes + delta,
        "byte_delta": delta, "changes": applied,
        "source_identifiers_order_labels_and_all_other_fields_preserved": True,
        "originals_modified": False, "decoder_changed": False, "files_skipped": 0,
    }
    return {"original_records": records, "records": corrected, "lineage": lineage}
