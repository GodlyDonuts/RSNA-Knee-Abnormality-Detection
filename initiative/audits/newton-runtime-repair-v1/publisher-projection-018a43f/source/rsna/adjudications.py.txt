"""Validate preserved report corrections before producing effective release annotations.

This module validates explicit human/agent decisions; it never infers report labels.
The registry location is fixed beside the reviewed folder, with no omission switch.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from rsna.labels import file_sha256, source_reports_fingerprint
from rsna.schema import STATES, TARGETS


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON field: {key}")
        result[key] = value
    return result


def _json(path: Path) -> object:
    return json.loads(path.read_text(), object_pairs_hook=_unique_object)


def _object(path: Path) -> dict:
    value = _json(path)
    if not isinstance(value, dict):
        raise ValueError(f"Expected an object: {path.name}")
    return value


def _bound_path(reference: object, artifact_root: Path) -> Path:
    if not isinstance(reference, dict) or set(reference) != {"path", "sha256"}:
        raise ValueError("Evidence references require path and sha256")
    name, digest = reference["path"], reference["sha256"]
    if not isinstance(name, str) or not name or Path(name).is_absolute():
        raise ValueError("Evidence paths must be relative to the artifact root")
    path = (artifact_root / name).resolve()
    if not path.is_relative_to(artifact_root.resolve()):
        raise ValueError("Evidence path escapes the artifact root")
    if not path.is_file() or digest != file_sha256(path):
        raise ValueError(f"Missing or changed evidence checksum: {name}")
    return path


def _identity(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Missing independent adjudication identity")
    return value


def _key(row: dict) -> tuple[str, str]:
    uid, target = row.get("study_uid"), row.get("target")
    if not isinstance(uid, str) or not uid or target not in TARGETS:
        raise ValueError("Unknown correction study/target")
    return uid, target


def _records(value: object) -> dict[tuple[str, str], dict]:
    if not isinstance(value, list):
        raise ValueError("Adjudication records must be a list")
    result = {}
    for row in value:
        if not isinstance(row, dict):
            raise ValueError("Invalid adjudication record")
        key = _key(row)
        if key in result:
            raise ValueError(f"Duplicate/conflicting adjudication target: {key}")
        result[key] = row
    return result


def _annotation(value: object, report: str) -> dict:
    if not isinstance(value, dict) or set(value) != {"state", "evidence"}:
        raise ValueError("Corrections require exact state and evidence")
    state, evidence = value["state"], value["evidence"]
    if state not in STATES or not isinstance(evidence, str):
        raise ValueError("Invalid correction annotation")
    if state == "not_mentioned":
        if evidence:
            raise ValueError("Missing-state corrections require empty evidence")
    elif not evidence.strip() or evidence not in report:
        raise ValueError("Correction quote is not verbatim source evidence")
    return value


def _context(document: dict, source_hash: str, rubric_hash: str) -> None:
    if (
        document.get("source_reports_sha256") != source_hash
        or document.get("rubric_sha256") != rubric_hash
    ):
        raise ValueError("Adjudication source reports/rubric binding mismatch")


def _bound_batch(
    base_reference: object,
    source_reference: object,
    reports: dict[str, str],
    reviews_folder: Path,
    artifact_root: Path,
) -> tuple[dict, dict]:
    base_path = _bound_path(base_reference, artifact_root)
    source_path = _bound_path(source_reference, artifact_root)
    if base_path.parent != reviews_folder.resolve() or not base_path.name.endswith(
        "-reviewed.jsonl"
    ):
        raise ValueError("Correction base must be an original reviewed artifact in this release")
    base_rows = {}
    for line in base_path.read_text().splitlines():
        row = json.loads(line, object_pairs_hook=_unique_object)
        uid = row["study_uid"]
        if uid in base_rows:
            raise ValueError("Duplicate study in correction base")
        base_rows[uid] = row
    source_rows = _json(source_path)
    if not isinstance(source_rows, list) or any(
        not isinstance(row, dict) or set(row) != {"study_uid", "report"} for row in source_rows
    ):
        raise ValueError("Correction source must be the blinded report batch")
    source = {row["study_uid"]: row["report"] for row in source_rows}
    if len(source) != len(source_rows) or set(source) != set(base_rows):
        raise ValueError("Correction source/base study membership mismatch")
    if any(uid not in reports or reports[uid] != report for uid, report in source.items()):
        raise ValueError("Correction source differs from release reports")
    return base_rows, source


def _case_changes(
    case: dict,
    case_reference: dict,
    reports: dict[str, str],
    reviews_folder: Path,
    artifact_root: Path,
    source_hash: str,
    rubric_hash: str,
) -> list[dict]:
    if type(case.get("schema_version")) is not int or case["schema_version"] != 1:
        raise ValueError("Unsupported correction case schema")
    case_id = _identity(case.get("id"))
    _context(case, source_hash, rubric_hash)
    paths = {
        name: _bound_path(case.get(name), artifact_root)
        for name in ("proposal", "independent_assessment", "decision")
    }
    proposal = _object(paths["proposal"])
    assessment = _object(paths["independent_assessment"])
    decision = _object(paths["decision"])
    _context(decision, source_hash, rubric_hash)
    if decision.get("status") not in {
        "accepted",
        "independently_adjudicated_pending_release_integration",
    }:
        raise ValueError("Correction requires an explicitly accepted independent decision")
    # Support the two preserved evidence formats without rewriting either decision.
    if ("accepted_masks" in decision) == ("accepted_changes" in decision):
        raise ValueError("Decision requires exactly one supported accepted-record format")
    masks_format = "accepted_masks" in decision
    preservation_field = (
        "original_review_preserved" if masks_format else "original_artifacts_preserved"
    )
    if decision.get(preservation_field) is not True:
        raise ValueError("Decision must preserve the original reviewed artifact")
    proposer = _identity(decision.get("proposer_agent_id"))
    adjudicator = _identity(decision.get("adjudicator_agent_id"))
    if proposer == adjudicator:
        raise ValueError("Self approval cannot authorize report corrections")
    if proposal.get("agent_id") != proposer or assessment.get("agent_id") != adjudicator:
        raise ValueError("Decision identities do not match the independent evidence")
    session = _identity(assessment.get("session_id"))
    if session == proposal.get("session_id", proposer):
        raise ValueError("Self approval cannot reuse the proposer session")
    if assessment.get("stage") != "independent_source_assessment":
        raise ValueError("An independent source assessment is required")
    if not masks_format:
        _context(proposal, source_hash, rubric_hash)
        _context(assessment, source_hash, rubric_hash)
    decision_bindings = {
        "source_reports_sha256": source_hash,
        "rubric_sha256": rubric_hash,
        "proposal_sha256": case["proposal"]["sha256"],
        "independent_assessment_sha256": case["independent_assessment"]["sha256"],
    }
    if masks_format:
        decision_bindings["original_review_sha256"] = case["base_review"]["sha256"]
    supporting = case.get("supporting_evidence")
    if not isinstance(supporting, dict):
        raise ValueError("Supporting decision evidence must be explicitly listed")
    for field, reference in supporting.items():
        if not field.endswith("_sha256") or field in decision_bindings:
            raise ValueError("Duplicate/invalid decision evidence binding")
        _bound_path(reference, artifact_root)
        decision_bindings[field] = reference["sha256"]
    if {key for key in decision if key.endswith("_sha256")} != set(decision_bindings) or any(
        decision.get(field) != digest for field, digest in decision_bindings.items()
    ):
        raise ValueError("Accepted decision evidence bindings are missing or changed")
    proposed = _records(proposal.get("proposed_masks" if masks_format else "proposed_errata"))
    assessed = _records(assessment.get("records"))
    accepted = _records(decision.get("accepted_masks" if masks_format else "accepted_changes"))
    changes = _records(case.get("changes"))
    if not changes or set(changes) != set(accepted) or decision.get("n_changes") != len(changes):
        raise ValueError("Correction changes must exactly cover the accepted decision")
    if isinstance(decision["n_changes"], bool):
        raise ValueError("Invalid accepted correction count")
    result = []
    for (uid, target), change in changes.items():
        base_reference = case.get("base_review") if masks_format else change.get("base_review")
        source_reference = case.get("source_batch") if masks_format else change.get("source_batch")
        base_rows, source = _bound_batch(
            base_reference, source_reference, reports, reviews_folder, artifact_root
        )
        if uid not in source or (uid, target) not in proposed or (uid, target) not in assessed:
            raise ValueError("Correction lacks source/proposal/independent assessment")
        before = _annotation(change.get("before"), source[uid])
        after = _annotation(change.get("after"), source[uid])
        index = TARGETS.index(target)
        base = base_rows[uid]
        if before != {"state": base["states"][index], "evidence": base["evidence"][index]}:
            raise ValueError("Correction before state/evidence differs from the preserved review")
        if before == after:
            raise ValueError("No-op decisions belong in semantic QA, not corrections")
        proposal_row = proposed[uid, target]
        assessed_row = assessed[uid, target]
        accepted_row = accepted[uid, target]
        assert isinstance(base_reference, dict) and isinstance(source_reference, dict)
        if masks_format:
            if (
                proposal.get("source_sha256") != source_reference["sha256"]
                or assessment.get("source_sha256") != source_reference["sha256"]
                or assessment.get("rubric_sha256") != rubric_hash
                or proposal.get("artifact_sha256") != base_reference["sha256"]
            ):
                raise ValueError("Proposal/assessment source, rubric or base binding mismatch")
        elif (
            proposal_row.get("source_sha256") != source_reference["sha256"]
            or proposal_row.get("source_path") != source_reference["path"]
            or proposal_row.get("artifact_sha256") != base_reference["sha256"]
            or proposal_row.get("artifact_path") != base_reference["path"]
            or assessed_row.get("source_sha256") != source_reference["sha256"]
            or accepted_row.get("source_sha256") != source_reference["sha256"]
            or accepted_row.get("original_review_sha256") != base_reference["sha256"]
            or accepted_row.get("original_review_path") != base_reference["path"]
        ):
            raise ValueError(
                "Per-target proposal/assessment/decision source or base binding mismatch"
            )
        if (
            proposal_row.get("published_state" if masks_format else "before_state")
            != before["state"]
            or proposal_row.get("proposed_state") != after["state"]
        ):
            raise ValueError("Correction state differs from the independently proposed change")
        for record in (assessed_row, accepted_row):
            if (
                record.get("independent_source_state") != after["state"]
                or record.get("source_evidence") != after["evidence"]
            ):
                raise ValueError("Correction state/quote differs from the accepted assessment")
        if not masks_format and (
            proposal_row.get("source_quote") != after["evidence"]
            or any(
                accepted_row.get(f"{phase}_{field}") != annotation[field]
                for phase, annotation in (("before", before), ("after", after))
                for field in ("state", "evidence")
            )
        ):
            raise ValueError("Correction before/after differs from the accepted exact annotation")
        reason = _identity(accepted_row.get("reason"))
        result.append(
            {
                "case_id": case_id,
                "study_uid": uid,
                "target": target,
                "before": before,
                "after": after,
                "reason": reason,
                "proposer_agent_id": proposer,
                "adjudicator_agent_id": adjudicator,
                "adjudicator_session_id": session,
                "case": case_reference,
                "base_review": base_reference,
                "source_batch": source_reference,
                "source_reports_sha256": source_hash,
                "rubric_sha256": rubric_hash,
                "proposal": case["proposal"],
                "independent_assessment": case["independent_assessment"],
                "decision": case["decision"],
                "supporting_evidence": supporting,
            }
        )
    return result


def inspect_adjudications(
    reports: dict[str, str], reviews_folder: Path, rubric_path: Path, artifact_root: Path
) -> dict:
    """Validate all registered evidence, including pending QA, without releasing labels."""
    directory = reviews_folder.parent / "adjudications"
    registry_path = directory / "registry.json"
    if not registry_path.is_file():
        raise ValueError(
            "Required adjacent adjudication registry is missing (explicit empty required)"
        )
    registry = _object(registry_path)
    if (
        set(registry)
        != {"schema_version", "source_reports_sha256", "rubric_sha256", "cases", "semantic_qa"}
        or type(registry["schema_version"]) is not int
        or registry["schema_version"] != 1
    ):
        raise ValueError("Invalid adjudication registry schema")
    source_hash, rubric_hash = source_reports_fingerprint(reports), file_sha256(rubric_path)
    _context(registry, source_hash, rubric_hash)
    registered = {}
    for name, subdirectory in (("cases", "cases"), ("semantic_qa", "qa")):
        references = registry[name]
        if not isinstance(references, list):
            raise ValueError("Registry cases and semantic QA must be explicit lists")
        paths = [_bound_path(reference, artifact_root) for reference in references]
        discovered = {path.resolve() for path in (directory / subdirectory).rglob("*.json")}
        if len(paths) != len(set(paths)) or set(paths) != discovered:
            raise ValueError(f"Registry omitted or duplicated {name} artifacts")
        registered[name] = list(zip(paths, references))
    changes = []
    seen_ids: set[str] = set()
    seen_targets: set[tuple[str, str]] = set()
    for path, reference in registered["cases"]:
        case = _object(path)
        case_changes = _case_changes(
            case, reference, reports, reviews_folder, artifact_root, source_hash, rubric_hash
        )
        if case["id"] in seen_ids:
            raise ValueError("Duplicate correction case identity")
        seen_ids.add(case["id"])
        for change in case_changes:
            key = _key(change)
            if key in seen_targets:
                raise ValueError(f"Duplicate/conflicting correction target: {key}")
            seen_targets.add(key)
        changes.extend(case_changes)
    review_artifacts = {
        str(path.resolve().relative_to(artifact_root.resolve())): file_sha256(path)
        for path in sorted(reviews_folder.glob("*-reviewed.jsonl"))
    }
    qa = []
    for path, reference in registered["semantic_qa"]:
        issue = _object(path)
        if type(issue.get("schema_version")) is not int or issue["schema_version"] != 1:
            raise ValueError("Invalid semantic QA schema")
        issue_id = _identity(issue.get("id"))
        if issue_id in seen_ids:
            raise ValueError("Duplicate correction/semantic QA identity")
        seen_ids.add(issue_id)
        _context(issue, source_hash, rubric_hash)
        if (
            issue.get("status") not in {"pending", "resolved"}
            or issue.get("scope") != "all_release_reviews"
        ):
            raise ValueError("Semantic QA requires explicit status and full release scope")
        _identity(issue.get("reason"))
        if not isinstance(issue.get("evidence"), list) or not issue["evidence"]:
            raise ValueError("Semantic QA requires bound evidence")
        for evidence in issue["evidence"]:
            _bound_path(evidence, artifact_root)
        if issue["status"] == "resolved":
            resolution = _object(_bound_path(issue.get("resolution"), artifact_root))
            _context(resolution, source_hash, rubric_hash)
            if resolution.get("status") != "resolved" or resolution.get("issue_id") != issue_id:
                raise ValueError("Semantic QA lacks an explicit matching resolution")
            for field in ("agent_id", "session_id", "notes"):
                _identity(resolution.get(field))
            if resolution.get("review_artifacts") != review_artifacts:
                raise ValueError("Semantic QA resolution does not cover the exact release reviews")
            expected_cases = {entry["path"]: entry["sha256"] for _, entry in registered["cases"]}
            if resolution.get("correction_cases") != expected_cases:
                raise ValueError("Semantic QA resolution does not cover the exact correction cases")
        qa.append({**issue, "artifact": reference})
    return {
        "schema_version": 1,
        "registry_sha256": file_sha256(registry_path),
        "source_reports_sha256": source_hash,
        "rubric_sha256": rubric_hash,
        "n_changes": len(changes),
        "changes": changes,
        "semantic_qa": qa,
        "pending_semantic_qa": [issue["id"] for issue in qa if issue["status"] == "pending"],
    }


def apply_adjudications(
    reviews: pd.DataFrame,
    reports: dict[str, str],
    reviews_folder: Path,
    rubric_path: Path,
    artifact_root: Path,
) -> tuple[pd.DataFrame, dict]:
    """Produce effective annotations only after all registered semantic QA resolves."""
    plan = inspect_adjudications(reports, reviews_folder, rubric_path, artifact_root)
    if plan["pending_semantic_qa"]:
        raise ValueError(f"Pending semantic QA blocks label release: {plan['pending_semantic_qa']}")
    if reviews.study_uid.duplicated().any() or set(reviews.study_uid) != set(reports):
        raise ValueError("Correction application requires complete unique reviewed study coverage")
    effective = reviews.copy(deep=True).set_index("study_uid")
    for target in TARGETS:
        for field in ("state", "evidence"):
            effective[f"{target}__original_{field}"] = effective[f"{target}__{field}"]
        effective[f"{target}__correction"] = ""
    for change in plan["changes"]:
        uid, target = change["study_uid"], change["target"]
        if effective.loc[uid, "artifact"] != Path(change["base_review"]["path"]).name or any(
            effective.loc[uid, f"{target}__{field}"] != change["before"][field]
            for field in ("state", "evidence")
        ):
            raise ValueError("Effective annotations differ from the bound correction base")
        for field in ("state", "evidence"):
            effective.loc[uid, f"{target}__{field}"] = change["after"][field]
        effective.loc[uid, f"{target}__correction"] = json.dumps(
            change, ensure_ascii=False, sort_keys=True
        )
    return effective.reset_index(), plan
