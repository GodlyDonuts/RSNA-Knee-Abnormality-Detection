"""Bind reviewed report-similarity edges before assigning leakage-safe splits.

This is an explicit evidence registry, not a fuzzy matcher or label extractor.
The sole supported relation removes one redundant ``. `` beside an existing
sentence stop. Raw reports, exact-report groups, and annotations remain intact.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from rsna.adjudications import _bound_path, _json, _object
from rsna.labels import file_sha256, source_reports_fingerprint

REGISTRY_PATH = Path("initiative/audits/report-similarity-registry.json")
RELATION = "single_redundant_full_stop"
ASSESSMENT_STAGES = {
    "root": "root_independent_complete_source_structure_preread",
    "independent": "independent_source_only_correspondence_preread",
}
SOURCE_FIRST_ORDER = "full_sources_before_other_assessor_findings"


def _source(endpoint: object, reports: dict[str, str], artifact_root: Path) -> dict:
    if not isinstance(endpoint, dict) or set(endpoint) != {
        "study_uid",
        "source_batch",
        "source_report_sha256",
    }:
        raise ValueError("Invalid report-similarity endpoint")
    uid = endpoint["study_uid"]
    if not isinstance(uid, str) or uid not in reports:
        raise ValueError("Unknown report-similarity study")
    path = _bound_path(endpoint["source_batch"], artifact_root)
    if (
        path.parent != (artifact_root / "initiative/reviews/inputs").resolve()
        or re.fullmatch(r"batch-\d{3}\.json", path.name) is None
    ):
        raise ValueError("Report-similarity source must be a canonical blinded batch")
    batch = _json(path)
    if not isinstance(batch, list) or any(
        not isinstance(row, dict)
        or set(row) != {"study_uid", "report"}
        or not isinstance(row["study_uid"], str)
        or not isinstance(row["report"], str)
        for row in batch
    ):
        raise ValueError("Invalid report-similarity source batch")
    source = {row["study_uid"]: row["report"] for row in batch}
    if (
        len(source) != len(batch)
        or uid not in source
        or any(key not in reports or reports[key] != value for key, value in source.items())
        or endpoint["source_report_sha256"] != hashlib.sha256(reports[uid].encode()).hexdigest()
    ):
        raise ValueError("Report-similarity source differs from release reports")
    return endpoint


def _single_redundant_stop(left: str, right: str) -> bool:
    # Check one local deletion, never normalize all punctuation (e.g. decimals).
    for longer, shorter in ((left, right), (right, left)):
        for match in re.finditer(r"(?=\. \. )", longer):
            at = match.start()
            if longer[:at] + ". " + longer[at + 4 :] == shorter:
                return True
    return False


def _assessments(record: dict, reports: dict[str, str], artifact_root: Path) -> dict:
    references = record.get("source_assessments")
    if not isinstance(references, dict) or set(references) != set(ASSESSMENT_STAGES):
        raise ValueError("Two separately identified source assessments are required")
    identities: dict[str, str] = {}
    paths: set[Path] = set()
    for role, stage in ASSESSMENT_STAGES.items():
        path = _bound_path(references[role], artifact_root)
        assessment = _object(path)
        identity = assessment.get("agent_id")
        if (
            assessment.get("stage") != stage
            or not isinstance(identity, str)
            or not identity.strip()
            or identity != identity.strip()
            or identity in identities.values()
            or path in paths
        ):
            raise ValueError("Report-similarity source assessments are not independent")
        identities[role] = identity
        paths.add(path)
        rows = assessment.get("records")
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Invalid report-similarity source assessment records")
        for endpoint in (record["left"], record["right"]):
            uid = endpoint["study_uid"]
            matches = [row for row in rows if row.get("study_uid") == uid]
            if len(matches) != 1:
                raise ValueError("Each source assessment must contain both studies exactly once")
            row = matches[0]
            source = row.get("source_batch" if role == "root" else "input_batch")
            expected = endpoint["source_batch"]
            if (
                row.get("report") != reports[uid]
                or row.get("source_report_sha256") != endpoint["source_report_sha256"]
                or not isinstance(source, dict)
                or source.get("sha256") != expected["sha256"]
                or source.get("path") not in {expected["path"], "recovered/" + expected["path"]}
            ):
                raise ValueError("Source assessment is not bound to the complete original report")
    proof = _object(_bound_path(record.get("difference_proof"), artifact_root))
    joint = _object(_bound_path(record.get("joint_source_comparison"), artifact_root))
    if (
        proof.get("stage") != "source_bound_report_similarity_edge_for_final_leakage_QA"
        or proof.get("assessor") != identities["root"]
        or proof.get("study_uids") != [record[side]["study_uid"] for side in ("left", "right")]
        or proof.get("source_report_sha256")
        != [record[side]["source_report_sha256"] for side in ("left", "right")]
        or proof.get("root_source_freeze_sha256") != references["root"]["sha256"]
        or proof.get("independent_source_freeze_sha256") != references["independent"]["sha256"]
        or proof.get("joint_source_comparison_sha256")
        != record["joint_source_comparison"]["sha256"]
        or joint.get("stage") != "separate_independent_source_structure_comparison"
        or joint.get("agent_id") != identities["independent"]
        or not isinstance(joint.get("root_source_freeze"), dict)
        or joint["root_source_freeze"].get("sha256") != references["root"]["sha256"]
        or not isinstance(joint.get("own_source_freeze"), dict)
        or joint["own_source_freeze"].get("sha256") != references["independent"]["sha256"]
    ):
        raise ValueError("Report-similarity proof does not bind this pair and both assessments")
    rows = joint.get("records")
    if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
        raise ValueError("Invalid report-similarity joint comparison records")
    for endpoint in (record["left"], record["right"]):
        matches = [row for row in rows if row.get("study_uid") == endpoint["study_uid"]]
        if (
            len(matches) != 1
            or matches[0].get("source_report_sha256") != endpoint["source_report_sha256"]
            or matches[0].get("material_structural_disagreement") is not False
        ):
            raise ValueError("Joint comparison must agree on both exact source reports")
    return identities


def inspect_report_similarity(reports: dict[str, str], artifact_root: Path) -> dict:
    """Validate all registered edges using only blinded original report text."""
    registry_path = artifact_root / REGISTRY_PATH
    if not registry_path.is_file():
        raise ValueError("Required report-similarity registry is missing")
    registry = _object(registry_path)
    if (
        type(registry.get("schema_version")) is not int
        or registry["schema_version"] != 1
        or not isinstance(registry.get("policies"), list)
    ):
        raise ValueError("Invalid report-similarity registry schema")
    expected_paths = {
        path.resolve()
        for path in registry_path.parent.glob("report-similarity-dispositions-v*.json")
        if path.is_file()
    }
    paths = [_bound_path(reference, artifact_root) for reference in registry["policies"]]
    if len(set(paths)) != len(paths) or set(paths) != expected_paths:
        raise ValueError("Report-similarity registry must cover every policy exactly once")
    decisions = []
    seen: set[tuple[str, ...]] = set()
    for reference, path in zip(registry["policies"], paths):
        policy = _object(path)
        if (
            type(policy.get("schema_version")) is not int
            or policy["schema_version"] != 1
            or policy.get("stage") != "reviewed_report_similarity_leakage_edges"
            or not isinstance(policy.get("records"), list)
            or not policy["records"]
        ):
            raise ValueError("Invalid report-similarity policy schema")
        for record in policy["records"]:
            if not isinstance(record, dict) or record.get("relation") != RELATION:
                raise ValueError("Unsupported report-similarity relation")
            if record.get("status") != "accepted" or record.get(
                "source_assessment_order"
            ) != dict.fromkeys(ASSESSMENT_STAGES, SOURCE_FIRST_ORDER):
                raise ValueError("Report-similarity edge lacks accepted source-first admission")
            left = _source(record.get("left"), reports, artifact_root)
            right = _source(record.get("right"), reports, artifact_root)
            pair = tuple(sorted((left["study_uid"], right["study_uid"])))
            if pair[0] == pair[1] or pair in seen:
                raise ValueError("Duplicate or self-linked report-similarity pair")
            seen.add(pair)
            if not _single_redundant_stop(reports[pair[0]], reports[pair[1]]):
                raise ValueError("Reports differ beyond one redundant full stop and space")
            identities = _assessments(record, reports, artifact_root)
            decisions.append({**record, "policy": reference, "assessor_ids": identities})
    return {
        "schema_version": 1,
        "registry_path": str(REGISTRY_PATH),
        "registry_sha256": file_sha256(registry_path),
        "source_reports_sha256": source_reports_fingerprint(reports),
        "policy_count": len(paths),
        "edge_count": len(decisions),
        "decisions": decisions,
        "scope": "Additional report-similarity leakage edges; no clinical, patient, site or pixel identity inference.",
    }


def similarity_pairs(audit: dict) -> list[tuple[str, str]]:
    return [(row["left"]["study_uid"], row["right"]["study_uid"]) for row in audit["decisions"]]


def bind_similarity_splits(audit: dict, studies: pd.DataFrame) -> dict:
    """Record resulting full components after all edges and exclusions are applied."""
    by_uid = studies.set_index("study_uid", verify_integrity=True)
    decisions = []
    for row, (left, right) in zip(audit["decisions"], similarity_pairs(audit)):
        group = by_uid.loc[left, "leakage_group"]
        if group != by_uid.loc[right, "leakage_group"]:
            raise ValueError("Required report-similarity edge was not enforced")
        component = studies.loc[studies.leakage_group.eq(group), ["study_uid", "split"]]
        decisions.append(
            {
                **row,
                "leakage_group": group,
                "component_splits": component.sort_values("study_uid").to_dict(orient="records"),
            }
        )
    return {**audit, "decisions": decisions, "edges_enforced": True}
