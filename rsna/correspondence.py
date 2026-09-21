"""Enforce reviewed report-to-image exclusions without changing source annotations.

The release registry has a fixed location and is required even when empty. Its
policies are explicit reviewed decisions; this module never infers correspondence.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from rsna.adjudications import _bound_path, _json, _object
from rsna.labels import file_sha256, source_reports_fingerprint
from rsna.schema import TARGETS

REGISTRY_PATH = Path("initiative/audits/correspondence-release-registry.json")


def _validate_proof(record: dict, artifact_root: Path) -> None:
    proof_path = _bound_path(record.get("verified_technical_proof"), artifact_root)
    proof = _object(proof_path)
    uid = record["study_uid"]
    if (
        proof.get("case") != record["case"]
        or proof.get("study_uid") != uid
        or proof.get("status") not in {"verified", "verified_body_region_correspondence_failure"}
    ):
        raise ValueError("Correspondence proof is unverified or identifies another case/study")
    if proof.get("stage") == "root_verified_correspondence_source_identity_binding":
        review_path = _bound_path(proof.get("verified_review"), artifact_root)
        handoff_path = _bound_path(proof.get("source_handoff"), artifact_root)
        review, handoff = _object(review_path), _object(handoff_path)
        source = {"path": record["source_path"], "sha256": record["source_batch_sha256"]}
        if (
            proof.get("source_batch") != source
            or proof.get("source_report_sha256") != record["source_report_sha256"]
            or review.get("case") != record["case"]
            or review.get("study_uid", uid) != uid
            or review.get("status") != "verified"
            or handoff.get("case") != record["case"]
            or handoff.get("study_uid") != uid
            or not isinstance(review.get("bindings"), list)
            or not any(
                isinstance(item, dict) and item.get("sha256") == proof["source_handoff"]["sha256"]
                for item in review["bindings"]
            )
        ):
            raise ValueError("Correspondence identity wrapper is not bound to its verified source")


def inspect_correspondence(reports: dict[str, str], artifact_root: Path) -> dict:
    registry_path = artifact_root / REGISTRY_PATH
    if not registry_path.is_file():
        raise ValueError("Required correspondence release registry is missing")
    registry = _object(registry_path)
    if (
        type(registry.get("schema_version")) is not int
        or registry["schema_version"] != 1
        or registry.get("target_order") != list(TARGETS)
        or not isinstance(registry.get("policies"), list)
    ):
        raise ValueError("Invalid correspondence registry schema or target order")
    decisions = []
    seen: set[str] = set()
    policy_paths: set[Path] = set()
    expected_paths = {
        path.resolve()
        for path in registry_path.parent.glob("correspondence-release-dispositions-v*.json")
        if path.is_file()
    }
    provided_paths = {_bound_path(reference, artifact_root) for reference in registry["policies"]}
    if provided_paths != expected_paths:
        raise ValueError("Correspondence registry does not cover exactly the preserved policies")
    for reference in registry["policies"]:
        path = _bound_path(reference, artifact_root)
        if path in policy_paths:
            raise ValueError("Duplicate correspondence policy")
        policy_paths.add(path)
        policy = _object(path)
        if (
            type(policy.get("schema_version")) is not int
            or policy["schema_version"] != 1
            or policy.get("stage") != "root_verified_correspondence_release_requirements"
            or policy.get("target_order") != list(TARGETS)
            or not isinstance(policy.get("records"), list)
        ):
            raise ValueError("Invalid correspondence policy schema or target order")
        for record in policy["records"]:
            if not isinstance(record, dict):
                raise ValueError("Invalid correspondence decision")
            uid = record.get("study_uid")
            if not isinstance(uid, str) or uid not in reports or uid in seen:
                raise ValueError("Unknown or duplicate correspondence study")
            seen.add(uid)
            if (
                record.get("report_derived_training_supervision_allowed") is not False
                or not isinstance(record.get("required_report_loss_mask"), list)
                or len(record["required_report_loss_mask"]) != len(TARGETS)
                or any(value is not False for value in record["required_report_loss_mask"])
            ):
                raise ValueError("Correspondence exclusions require twelve explicit false masks")
            if any(
                not isinstance(record.get(key), str) or not record[key].strip()
                for key in ("case", "status", "reason", "scope")
            ):
                raise ValueError("Missing correspondence decision context")
            source_path = _bound_path(
                {"path": record.get("source_path"), "sha256": record.get("source_batch_sha256")},
                artifact_root,
            )
            batch = _json(source_path)
            if not isinstance(batch, list) or any(
                not isinstance(row, dict)
                or set(row) != {"study_uid", "report"}
                or not isinstance(row["study_uid"], str)
                or not isinstance(row["report"], str)
                for row in batch
            ):
                raise ValueError("Correspondence source must be a blinded report batch")
            source = {row["study_uid"]: row["report"] for row in batch}
            if (
                len(source) != len(batch)
                or uid not in source
                or any(key not in reports or reports[key] != value for key, value in source.items())
                or record.get("source_report_sha256")
                != hashlib.sha256(reports[uid].encode()).hexdigest()
            ):
                raise ValueError("Correspondence source differs from release reports")
            _validate_proof(record, artifact_root)
            decisions.append({**record, "policy": reference})
    return {
        "schema_version": 1,
        "registry_path": str(REGISTRY_PATH),
        "registry_sha256": file_sha256(registry_path),
        "source_reports_sha256": source_reports_fingerprint(reports),
        "target_order": list(TARGETS),
        "policy_count": len(policy_paths),
        "study_count": len(decisions),
        "decisions": decisions,
        "policy": "Mask report-derived supervision for reviewed correspondence exclusions; preserve source annotations, gold evaluation, and leakage groups.",
    }


def apply_correspondence(
    studies: pd.DataFrame, reports: dict[str, str], artifact_root: Path
) -> tuple[pd.DataFrame, dict]:
    if (
        studies.study_uid.duplicated().any()
        or set(studies.study_uid) != set(reports)
        or dict(zip(studies.study_uid, studies.Report)) != reports
    ):
        raise ValueError("Correspondence studies differ from release reports")
    audit = inspect_correspondence(reports, artifact_root)
    effective = studies.copy(deep=True)
    by_uid = {row["study_uid"]: row for row in audit["decisions"]}
    excluded = effective.study_uid.isin(by_uid)
    gold = effective.split.eq("gold_evaluation")
    masked = excluded & ~gold
    effective["report_supervision_allowed"] = ~excluded
    effective["report_correspondence_status"] = effective.study_uid.map(
        lambda uid: by_uid[uid]["status"] if uid in by_uid else ""
    )
    effective["report_correspondence_reason"] = effective.study_uid.map(
        lambda uid: by_uid[uid]["reason"] if uid in by_uid else ""
    )
    changed = {}
    for target in TARGETS:
        column = f"{target}__mask"
        changed[target] = int(effective.loc[masked, column].ne(0).sum())
        effective.loc[masked, column] = 0.0
        effective.loc[masked, f"{target}__provenance"] = "report_correspondence_masked"
    audit["report_masked_studies"] = sorted(effective.loc[masked, "study_uid"].tolist())
    audit["official_gold_preserved_studies"] = sorted(
        effective.loc[excluded & gold, "study_uid"].tolist()
    )
    audit["newly_masked_targets"] = changed
    return effective, audit
