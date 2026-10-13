"""Accepted corrected original-source object table for the public-input cohort.

The full-cohort cache producer must fetch original DICOM objects at the corrected
source revision, restricted to the accepted release series, with the two official
correction-plan replacement hashes/bytes overlaid. Historical baseline hashes for
the corrected files are invalid for this run. This module only binds metadata; it
reads no pixels and acquires nothing. Outputs contain study/series/SOP identifiers
and therefore stay owning-node; only ``summary`` is sanitized.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Mapping
from pathlib import PurePosixPath
from typing import Any

import pandas as pd

from rsna.audit import inventory_fingerprint, validate_inventory
from rsna.competitive.public_input import canonical_sha256, roster_sha256

CORRECTED_SOURCE_REVISION = "0b5e410288232f570c69ba4f44723356e6ae6ab7"
HISTORICAL_SOURCE_REVISION = "52f995f4568e36bfa5fbbaf69a89af41226b4f5d"
CORRECTION_PLAN_FILE_SHA256 = "68b44c1412ced183db223220f39cd4a6e071d40515df4c7b75b40a96ba6d0f3d"
CORRECTION_PLAN_SHA256 = "360a9aa30cf9f1bd024475557ce928e6a7ab1381f7757e6e186d166d569d130b"
COLUMNS = ["split", "study_uid", "series_uid", "sop_uid", "path", "sha256", "bytes", "storage"]
IDENTITY = ["split", "study_uid", "series_uid", "sop_uid", "path", "storage"]
# Actual CPU906794 receipt in initiative/audits/public-triplet-full-source-planning-20261013-v1.
ACCEPTED_SOURCE_TABLE = {
    "fitting_studies": 2247,
    "development_studies": 1156,
    "fitting_roster_sha256": "e9d38ee21a114973177b89f4602129c7dc047927b8499991d3be30fd89737f3a",
    "development_roster_sha256": "2f365bb0ce8e052f22dcab78bb2bbb7e03c0919ab28ba94ab9d3d8d6e08f0fdb",
    "source_series": 18906,
    "excluded_source_series": 25,
    "source_files": 643501,
    "source_bytes": 449644551848,
    "corrected_source_files": 2,
    "corrected_source_byte_delta": 582656,
}
UID = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]{0,127}")
SHA = re.compile(r"[0-9a-f]{64}")


def _strict_json(data: bytes) -> dict:
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate JSON field")
            result[key] = value
        return result

    value = json.loads(data.decode(), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("Expected a JSON object")
    return value


def _uid(value: object) -> str:
    if not isinstance(value, str) or not UID.fullmatch(value) or ".." in value:
        raise ValueError("Unsafe study/series UID")
    return value


def load_correction(
    plan_bytes: bytes,
    receipt: Mapping[str, Any],
    *,
    plan_file_sha256: str = CORRECTION_PLAN_FILE_SHA256,
    plan_sha256: str = CORRECTION_PLAN_SHA256,
    corrected_revision: str = CORRECTED_SOURCE_REVISION,
) -> dict[str, dict]:
    """Return ``{path: change}`` for the accepted, receipt-bound official correction."""
    plan = _strict_json(plan_bytes)
    if (
        hashlib.sha256(plan_bytes).hexdigest() != plan_file_sha256
        or canonical_sha256(plan) != plan_sha256
    ):
        raise ValueError("Correction plan differs from the accepted plan")
    if (
        plan.get("schema_version") != 1
        or plan.get("status") != "accepted_exact_authoritative_source_replacement"
        or plan.get("original_pinned_revision_preserved") is not True
        or plan.get("decoder_changed") is not False
    ):
        raise ValueError("An accepted preserved source correction is required")
    baseline = plan.get("baseline_source_revision")
    if (
        receipt.get("status") != "verified_private_hf_source_correction"
        or receipt.get("remote_objects_verified") is not True
        or receipt.get("source_correction_plan_sha256") != plan_file_sha256
        or receipt.get("baseline_source_revision") != baseline
        or receipt.get("corrected_source_revision") != corrected_revision
        or corrected_revision in {baseline, plan.get("publication_parent_revision")}
    ):
        raise ValueError("Correction receipt does not bind the corrected source revision")
    changes = plan.get("changes")
    if not isinstance(changes, list) or len(changes) != plan.get("n_changed_source_files"):
        raise ValueError("Correction change count differs")
    result: dict[str, dict] = {}
    for change in changes:
        before, after = change.get("before"), change.get("after")
        if (
            not isinstance(before, dict)
            or not isinstance(after, dict)
            or set(before) != set(COLUMNS)
            or set(after) != set(COLUMNS)
            or {key for key in COLUMNS if before[key] != after[key]} != {"sha256", "bytes"}
            or any(change.get(key) != before[key] for key in ("path", "study_uid", "series_uid"))
            or any(not SHA.fullmatch(str(row["sha256"])) for row in (before, after))
            or any(type(row["bytes"]) is not int or row["bytes"] <= 0 for row in (before, after))
            or before["path"] in result
        ):
            raise ValueError("Correction must replace only exact SHA256/bytes of one object")
        result[before["path"]] = {"before": before, "after": after}
    if set(result) != set(receipt.get("changed_source_paths") or []):
        raise ValueError("Receipt changed-source membership differs")
    return result


def accepted_source_objects(
    inventory: pd.DataFrame,
    accepted_series: pd.DataFrame,
    fitting_uids: Iterable[str],
    evaluation_uids: Iterable[str],
    plan_bytes: bytes,
    receipt: Mapping[str, Any],
    *,
    source_revision: str,
    expected: Mapping[str, Any] = ACCEPTED_SOURCE_TABLE,
    plan_file_sha256: str = CORRECTION_PLAN_FILE_SHA256,
    plan_sha256: str = CORRECTION_PLAN_SHA256,
) -> tuple[dict[str, list[dict]], dict]:
    """Bind the per-study original-object table before any producer intent/fetch.

    ``inventory`` is the complete historical baseline inventory (fingerprint bound
    by the plan); ``accepted_series`` is the release series table. Returns
    ``({study_uid: [object, ...]}, sanitized_summary)``.
    """
    if source_revision == HISTORICAL_SOURCE_REVISION:
        raise ValueError("Historical source revision is invalid for corrected objects")
    if source_revision != CORRECTED_SOURCE_REVISION:
        raise ValueError("Objects must be acquired at the corrected source revision")
    fitting, evaluation = list(fitting_uids), list(evaluation_uids)
    for uid in [*fitting, *evaluation]:
        _uid(uid)
    if (
        len(set(fitting)) != len(fitting)
        or len(set(evaluation)) != len(evaluation)
        or set(fitting) & set(evaluation)
        or (len(fitting), len(evaluation))
        != (expected["fitting_studies"], expected["development_studies"])
        or roster_sha256(fitting) != expected["fitting_roster_sha256"]
        or roster_sha256(evaluation) != expected["development_roster_sha256"]
    ):
        raise ValueError("Rosters must be the accepted disjoint unique fitting/development sets")
    cohort = set(fitting) | set(evaluation)
    changes = load_correction(
        plan_bytes,
        receipt,
        plan_file_sha256=plan_file_sha256,
        plan_sha256=plan_sha256,
        corrected_revision=source_revision,
    )
    plan = _strict_json(plan_bytes)
    if list(inventory.columns) != COLUMNS:
        raise ValueError("Source inventory schema differs")
    validate_inventory(inventory)
    if not inventory.sha256.astype(str).str.fullmatch(SHA.pattern).all():
        raise ValueError("Source inventory hashes must be exact SHA256 digests")
    if inventory_fingerprint(inventory) != plan.get("baseline_inventory_fingerprint_sha256"):
        raise ValueError("Source inventory is not the correction-plan baseline")
    if {"StudyInstanceUID", "SeriesInstanceUID"} - set(accepted_series.columns):
        raise ValueError("Accepted series table needs study/series UIDs")
    series_table = accepted_series[["StudyInstanceUID", "SeriesInstanceUID"]].astype(str)
    series_table = series_table[series_table.StudyInstanceUID.isin(cohort)]
    if series_table.SeriesInstanceUID.duplicated().any():
        raise ValueError("Accepted series must be unique and belong to one study")
    allowed = dict(zip(series_table.SeriesInstanceUID, series_table.StudyInstanceUID, strict=True))
    for uid in allowed:
        _uid(uid)
    if len(allowed) != expected["source_series"] or set(allowed.values()) != cohort:
        raise ValueError("Accepted series do not cover the cohort exactly")

    rows = inventory[inventory.study_uid.isin(cohort)]
    if set(rows.split) != {"train_series"}:
        raise ValueError("Cohort objects must come only from the training source split")
    excluded = set(rows.series_uid) - set(allowed)
    if len(excluded) != expected["excluded_source_series"]:
        raise ValueError("Unaccepted source series count differs")
    rows = rows[rows.series_uid.isin(allowed)]
    objects: dict[str, list[dict]] = {uid: [] for uid in sorted(cohort)}
    corrected = delta = 0
    for row in rows.sort_values("path").to_dict("records"):
        record = {key: row[key] for key in COLUMNS}
        record["bytes"] = int(record["bytes"])
        path = PurePosixPath(record["path"])
        if allowed[record["series_uid"]] != record["study_uid"]:
            raise ValueError("Accepted series belongs to a different study")
        if path.is_absolute() or ".." in path.parts or "\\" in record["path"]:
            raise ValueError("Unsafe source object path")
        change = changes.get(record["path"])
        if change is not None:
            if record != change["before"]:
                raise ValueError("Correction before row differs from the inventory")
            record = dict(change["after"])
            corrected += 1
            delta += change["after"]["bytes"] - change["before"]["bytes"]
        objects[record["study_uid"]].append(record)
    records = [record for study in objects.values() for record in study]
    files, total = len(records), sum(record["bytes"] for record in records)
    if (
        any(not study for study in objects.values())
        or {record["series_uid"] for record in records} != set(allowed)
        or files != expected["source_files"]
        or total != expected["source_bytes"]
        or corrected != expected["corrected_source_files"]
        or delta != expected["corrected_source_byte_delta"]
    ):
        raise ValueError("Accepted corrected source table differs from the bound totals")
    summary = {
        "source_revision": source_revision,
        "correction_plan_sha256": plan_sha256,
        "fitting_studies": len(fitting),
        "development_studies": len(evaluation),
        "fitting_roster_sha256": roster_sha256(fitting),
        "development_roster_sha256": roster_sha256(evaluation),
        "source_series": len(allowed),
        "excluded_source_series": len(excluded),
        "source_files": files,
        "source_bytes": total,
        "corrected_source_files": corrected,
        "corrected_source_byte_delta": delta,
        "object_table_sha256": canonical_sha256(objects),
    }
    return objects, summary
