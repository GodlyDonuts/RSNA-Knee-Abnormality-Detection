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
MULTI_ASSESSOR_STAGE = "verified_multi_assessor_correspondence_source_binding"

# Exact historical dispositions, not keyword-based inference from arbitrary notes.
# These originals predate the explicit boolean used by the later cache reviews.
_NATIVE_MASK_DECISIONS = {
    "Keep whole-study report supervision fully masked in the current single-knee training pipeline. Preserve original source/images/official gold. Any later side-specific derivative requires validated series-to-side/report binding and parent patient grouping for leakage control; no arbitrary side selection or label cloning.",
    "Retain whole-study report supervision fully masked for current single-knee training. Preserve raw source, images and official gold unchanged. Any later derived side-specific unit requires a separately validated series-to-side/report binding and parent-patient leakage grouping; no arbitrary selection, label cloning or automatic virtual split.",
    "Keep all10 informative report targets uncertain/masked and2unmentioned. Preserve whole source, raw side tags, caches and gold. Any later per-side derivative needs verified series-to-side/report binding and shared parent grouping. No clinical labels or arbitrary side selection from technical images.",
}
_ASSESSMENT_CLAIMS = {
    "independent_technical_correspondence_assessment": "/conclusion/single_knee_training_recommendation",
    "independent_native_technical_correspondence_visual_review": "/single_knee_training_recommendation",
    "independent_technical_visual_assessment": "/training_recommendation/disposition",
    "independent_cached_representative_visual_assessment": "/assessment",
    "independent_cached_pixel_visual_assessment": "/technical_interpretation",
}


def _assessor_id(value: object) -> str:
    """Require the actual canonical task identity, not whitespace/path aliases."""
    if (
        not isinstance(value, str)
        or value != value.strip()
        or not (value == "/root" or value.startswith("/root/"))
        or any(
            not part or any(char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in part)
            for part in value.split("/")[1:]
        )
    ):
        raise ValueError("Correspondence assessor identity is missing or noncanonical")
    return value


def _selected(document: object, pointer: object) -> object:
    """Resolve explicit JSON pointers; never search for a similar case or UID."""
    if not isinstance(pointer, str) or (pointer and not pointer.startswith("/")):
        raise ValueError("Invalid correspondence proof selector")
    value = document
    for token in pointer.split("/")[1:]:
        key = token.replace("~1", "/").replace("~0", "~")
        if isinstance(value, list) and key.isdigit() and str(int(key)) == key:
            index = int(key)
            if index >= len(value):
                raise ValueError("Correspondence proof selector is outside its document")
            value = value[index]
        elif isinstance(value, dict) and key in value:
            value = value[key]
        else:
            raise ValueError("Correspondence proof selector does not identify evidence")
    return value


def _claim(document: dict, claim: object) -> object:
    if not isinstance(claim, dict) or set(claim) != {"pointer", "value"}:
        raise ValueError("Invalid correspondence evidence claim")
    actual = _selected(document, claim["pointer"])
    if type(actual) is not type(claim["value"]) or actual != claim["value"]:
        raise ValueError("Correspondence claim differs from its preserved proof")
    return actual


def _validate_multi_assessor_proof(record: dict, proof: dict, artifact_root: Path) -> None:
    source_ref = {"path": record["source_path"], "sha256": record["source_batch_sha256"]}
    batch = _json(_bound_path(source_ref, artifact_root))
    index = proof.get("source_index")
    if (
        type(proof.get("schema_version")) is not int
        or proof["schema_version"] != 1
        or proof.get("source_batch") != source_ref
        or proof.get("source_report_sha256") != record["source_report_sha256"]
        or type(index) is not int
        or not isinstance(batch, list)
        or not 0 <= index < len(batch)
        or batch[index].get("study_uid") != record["study_uid"]
    ):
        raise ValueError("Multi-assessor proof differs from the exact source record")
    batch_number = Path(record["source_path"]).stem.removeprefix("batch-")
    if record["case"] != f"{batch_number}:{index:02d}":
        raise ValueError("Correspondence case differs from the bound source index")
    quotes = proof.get("source_evidence")
    if (
        not isinstance(quotes, list)
        or not quotes
        or any(not isinstance(q, str) or not q or q not in batch[index]["report"] for q in quotes)
    ):
        raise ValueError("Correspondence source quotation is absent or changed")
    root_ref = proof.get("root_decision")
    independent_ref = proof.get("independent_assessment")
    if not isinstance(root_ref, dict) or not isinstance(independent_ref, dict):
        raise ValueError("Correspondence proof requires preserved assessor references")
    root = _object(_bound_path(root_ref, artifact_root))
    independent = _object(_bound_path(independent_ref, artifact_root))
    root_agent = _assessor_id(root.get("agent_id", root.get("assessor")))
    independent_agent = _assessor_id(independent.get("agent_id"))
    wrapper_agent = _assessor_id(proof.get("agent_id"))
    if (
        root_agent != "/root"
        or independent_agent in {root_agent, wrapper_agent}
        or independent.get("study_uid") != record["study_uid"]
        or independent.get("case", independent.get("source_case", record["case"])) != record["case"]
        or independent.get("source_batch_index", index) != index
        or independent.get("source_batch_sha256", record["source_batch_sha256"])
        != record["source_batch_sha256"]
        or any(
            independent.get(key, record["source_report_sha256"]) != record["source_report_sha256"]
            for key in ("source_report_sha256", "source_report_utf8_sha256")
        )
    ):
        raise ValueError("Correspondence proof lacks distinct bound root/independent assessors")
    independent_claim = _claim(independent, proof.get("independent_claim"))
    independent_stage = independent.get("stage")
    if (
        not isinstance(independent_claim, str)
        or not independent_claim.strip()
        or not isinstance(independent_stage, str)
        or proof["independent_claim"]["pointer"] != _ASSESSMENT_CLAIMS.get(independent_stage)
    ):
        raise ValueError("Correspondence proof lacks independent assessment content")
    selection = proof.get("root_selector")
    selected = _selected(root, selection)
    if not isinstance(selected, dict) or selected.get("study_uid") != record["study_uid"]:
        raise ValueError("Root correspondence decision identifies another study")
    disposition = _claim(selected, proof.get("root_disposition"))
    kind = proof.get("proof_kind")
    if kind == "reviewed_cache_aggregate":
        if (
            not isinstance(selection, str)
            or not selection.startswith("/cases/")
            or selection.count("/") != 2
            or root.get("stage") != "independent_root_source_header_cache_display_replay"
            or root.get("status") != "verified"
            or selected.get("case") != record["case"]
            or proof["root_disposition"]["pointer"] != "/required_report_supervision_mask_all"
            or disposition is not True
            or selected.get("agent_visual", {}).get("sha256") != independent_ref["sha256"]
        ):
            raise ValueError("Cache proof lacks an explicit bound whole-study exclusion")
        handoff_ref = proof.get("source_handoff")
        if not isinstance(handoff_ref, dict):
            raise ValueError("Cache correspondence proof requires its original source handoff")
        handoff = _object(_bound_path(handoff_ref, artifact_root))
        if (
            selected.get("joint_handoff", {}).get("sha256") != handoff_ref["sha256"]
            or handoff.get("agent_id") != independent_agent
            or handoff.get("case") != record["case"]
            or handoff.get("study_uid") != record["study_uid"]
            or handoff.get("source_report_sha256") != record["source_report_sha256"]
        ):
            raise ValueError("Cache proof handoff is not bound to the original source")
    elif kind == "reviewed_native_aggregate":
        if selection != "" or root.get("case", record["case"]) != record["case"]:
            raise ValueError("Native proof selector/case differs from its original decision")
        if "source_batch" in root:
            source_matches = (
                root["source_batch"] == source_ref and root.get("source_index") == index
            )
            root_binding = root.get("independent_visual_assessment", {}).get("sha256")
            exclusion = (
                root.get("stage") == "independently_supported_source_correspondence_quarantine"
                and root.get("single_knee_inclusion_status") == "quarantined_undivided_study"
                and proof["root_disposition"]["pointer"] == "/report_supervision_action"
                and disposition == "retain_all12_target_masks; zero confident report targets"
                and independent.get("source_report_utf8_sha256") == record["source_report_sha256"]
            )
        else:
            source_matches = (
                root.get("source_batch_path") == record["source_path"]
                and root.get("source_batch_sha256") == record["source_batch_sha256"]
                and root.get("source_report_sha256") == record["source_report_sha256"]
            )
            root_binding = root.get("evidence_files", {}).get(Path(independent_ref["path"]).name)
            for binding in root.get("archived_bindings", []):
                if binding.get("archived_path") == independent_ref["path"]:
                    root_binding = binding.get("sha256")
            exclusion = (
                proof["root_disposition"]["pointer"] == "/decision"
                and isinstance(disposition, str)
                and disposition in _NATIVE_MASK_DECISIONS
            )
        if not source_matches or root_binding != independent_ref["sha256"] or not exclusion:
            raise ValueError("Native proof lacks the original source-bound exclusion decision")
    else:
        raise ValueError("Unknown multi-assessor correspondence proof kind")


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
    if proof.get("stage") == MULTI_ASSESSOR_STAGE:
        _validate_multi_assessor_proof(record, proof, artifact_root)
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
        binding_version = policy.get("proof_binding_version", 1)
        if type(binding_version) is not int or binding_version not in {1, 2}:
            raise ValueError("Unsupported correspondence proof binding version")
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
            if binding_version == 2:
                proof = _object(_bound_path(record.get("verified_technical_proof"), artifact_root))
                if proof.get("stage") != MULTI_ASSESSOR_STAGE:
                    raise ValueError(
                        "Correspondence policy requires a multi-assessor source binding"
                    )
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
