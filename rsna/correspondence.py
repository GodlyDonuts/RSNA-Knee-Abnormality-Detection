"""Enforce reviewed report-to-image exclusions without changing source annotations.

The release registry has a fixed location and is required even when empty. Its
policies are explicit reviewed decisions; this module never infers correspondence.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path

import pandas as pd

from rsna.adjudications import _bound_path, _json, _object
from rsna.labels import file_sha256, source_reports_fingerprint
from rsna.schema import TARGETS

REGISTRY_PATH = Path("initiative/audits/correspondence-release-registry.json")
MULTI_ASSESSOR_STAGE = "verified_multi_assessor_correspondence_source_binding"
TEMPORAL_STAGE = "verified_temporal_multi_assessor_correspondence_binding"
TEMPORAL_ACTION = "exclude_whole_study_report_supervision_non_gold"
SOURCE_HEADER_STAGE = "verified_source_header_multi_assessor_correspondence_binding"

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
    "independent_selected_cached_representative_visual_assessment": "/technical_interpretation",
}
_CACHE_EXCLUSION_ACTION = "exclude_whole_study_report_supervision_preserve_originals_and_gold"


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


def _validate_cache_action_lineage(
    record: dict,
    proof: dict,
    root: dict,
    selected: dict,
    independent: dict,
    handoff: dict,
    report: str,
    artifact_root: Path,
) -> None:
    """Verify preserved source/visual freezes; do not rerun image interpretation."""
    visual_ref = proof.get("root_visual_assessment")
    if not isinstance(visual_ref, dict):
        raise ValueError("Cache action proof requires its root visual reference")
    visual = _object(_bound_path(visual_ref, artifact_root))
    visual_claim = _claim(visual, proof.get("root_visual_claim"))
    if (
        selected.get("frozen_root_visual", {}).get("sha256") != visual_ref["sha256"]
        or handoff.get("original_root_visual_assessment", {}).get("sha256") != visual_ref["sha256"]
        or handoff.get("original_independent_visual_assessment", {}).get("sha256")
        != proof["independent_assessment"]["sha256"]
        or _assessor_id(visual.get("agent_id")) != "/root"
        or visual.get("stage") != "root_independent_selected_cache_visual_assessment"
        or visual.get("case") != record["case"]
        or visual.get("study_uid") != record["study_uid"]
        or proof["root_visual_claim"]["pointer"] != "/bounded_disposition"
        or not isinstance(visual_claim, str)
        or not visual_claim.strip()
        or not isinstance(visual.get("observations"), list)
        or not visual["observations"]
        or any(not isinstance(item, str) or not item.strip() for item in visual["observations"])
    ):
        raise ValueError("Cache action proof lacks its original root visual freeze")
    sources = []
    for role, stage, batch_key, text_key, quote_key in (
        (
            "root",
            "root_independent_complete_source_structure_preread",
            "source_batch",
            "root_observations",
            "exact_source_quotes",
        ),
        (
            "independent",
            "independent_source_only_correspondence_preread",
            "input_batch",
            "source_structure",
            "exact_structure_quotes",
        ),
    ):
        reference = proof.get(f"{role}_source_assessment")
        source = _object(_bound_path(reference, artifact_root))
        pointer = proof.get(f"{role}_source_selector")
        source_record = _selected(source, pointer)
        quotes = source_record.get(quote_key) if isinstance(source_record, dict) else None
        if (
            not isinstance(pointer, str)
            or not pointer.startswith("/records/")
            or pointer.count("/") != 2
            or source.get("stage") != stage
            or _assessor_id(source.get("agent_id"))
            != ("/root" if role == "root" else independent["agent_id"])
            or not isinstance(source_record, dict)
            or source_record.get("case") != record["case"]
            or source_record.get("study_uid") != record["study_uid"]
            or source_record.get("source_report_sha256") != record["source_report_sha256"]
            or source_record.get("report") != report
            or source_record.get(batch_key, {}).get("sha256") != record["source_batch_sha256"]
            or source_record.get(batch_key, {}).get("path") != "recovered/" + record["source_path"]
            or not isinstance(source_record.get(text_key), str)
            or not source_record[text_key].strip()
            or not isinstance(quotes, list)
            or not quotes
            or any(not isinstance(q, str) or not q or q not in report for q in quotes)
        ):
            raise ValueError("Cache action source freeze differs from its original full report")
        sources.append(source)
    expected_hashes = {
        proof[f"{role}_source_assessment"]["sha256"] for role in ("root", "independent")
    }
    for document in (independent, handoff):
        lineage = document.get("source_review_lineage", {})
        if not isinstance(lineage, dict):
            raise ValueError("Cache action source lineage must be an object")
        refs = lineage.get("source_assessments", [])
        if (
            lineage.get("case") != record["case"]
            or not isinstance(refs, list)
            or len(refs) != 2
            or any(not isinstance(ref, dict) for ref in refs)
            or {ref.get("sha256") for ref in refs} != expected_hashes
            or {_assessor_id(agent) for agent in lineage.get("independent_assessors", [])}
            != {"/root", independent["agent_id"]}
        ):
            raise ValueError("Cache action handoff does not bind both full source freezes")
    try:
        source_times = [datetime.fromisoformat(item["created_at_utc"]) for item in sources]
        visual_times = [
            datetime.fromisoformat(item["created_at_utc"]) for item in (visual, independent)
        ]
        handoff_time = datetime.fromisoformat(handoff["created_at_utc"])
        replay_time = datetime.fromisoformat(root["completed_at_utc"])
        times = source_times + visual_times + [handoff_time, replay_time]
        ordered = (
            all(t.tzinfo is not None and t.utcoffset() is not None for t in times)
            and max(source_times)
            <= min(visual_times)
            <= max(visual_times)
            <= handoff_time
            <= replay_time
        )
    except (KeyError, TypeError, ValueError):
        ordered = False
    if not ordered:
        raise ValueError("Cache action source/visual freezes are not ordered before joint replay")


def _validate_multi_assessor_proof(record: dict, proof: dict, artifact_root: Path) -> None:
    source_ref = {
        "path": record["source_path"],
        "sha256": record["source_batch_sha256"],
    }
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
    if kind in {"reviewed_cache_aggregate", "reviewed_cache_action_aggregate"}:
        action_based = kind == "reviewed_cache_action_aggregate"
        exclusion = (
            (
                proof["root_disposition"]["pointer"] == "/required_export_action"
                and disposition == _CACHE_EXCLUSION_ACTION
            )
            if action_based
            else (
                proof["root_disposition"]["pointer"] == "/required_report_supervision_mask_all"
                and disposition is True
            )
        )
        if (
            not isinstance(selection, str)
            or not selection.startswith("/cases/")
            or selection.count("/") != 2
            or root.get("stage") != "independent_root_source_header_cache_display_replay"
            or root.get("status") != "verified"
            or selected.get("case") != record["case"]
            or not exclusion
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
        if action_based:
            _validate_cache_action_lineage(
                record,
                proof,
                root,
                selected,
                independent,
                handoff,
                batch[index]["report"],
                artifact_root,
            )
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


def _temporal_check(condition: bool, context: str) -> None:
    if not condition:
        raise ValueError("Temporal correspondence evidence " + context)


def _temporal_time(document: dict, key: str) -> datetime:
    try:
        value = datetime.fromisoformat(document[key])
        if value.tzinfo is not None and value.utcoffset() is not None:
            return value
    except (KeyError, TypeError, ValueError):
        pass
    raise ValueError("Temporal correspondence evidence has invalid recorded chronology")


def _validate_temporal_proof(record: dict, proof: dict, artifact_root: Path) -> None:
    """Authenticate actual temporal-review formats, without rerunning image review.

    These records preserve separate per-reader chronology, including an explicit
    correction of a historical source-link typo. They cannot prove past access order.
    """
    refs = proof.get("evidence")
    _temporal_check(isinstance(refs, dict), "requires bound evidence references")
    assert isinstance(refs, dict)
    keys = {
        "source_packet",
        "root_source",
        "root_visual",
        "root_lineage_correction",
        "root_numeric",
        "root_proposal",
        "root_acceptance",
        "independent_source",
        "independent_visual",
        "independent_numeric",
        "independent_policy",
        "independent_acceptance",
        "panel_manifest",
        "technical_metadata",
        "before_registry",
    }
    _temporal_check(set(refs) == keys, "has missing or unexpected evidence roles")
    docs = {
        key: _object(_bound_path(refs[key], artifact_root))
        for key in keys
        if key != "source_packet"
    }
    source_ref = {"path": record["source_path"], "sha256": record["source_batch_sha256"]}
    batch = _json(_bound_path(source_ref, artifact_root))
    index = proof.get("source_index")
    _temporal_check(
        type(proof.get("schema_version")) is int
        and proof["schema_version"] == 1
        and proof.get("source_batch") == source_ref
        and proof.get("source_report_sha256") == record["source_report_sha256"]
        and type(index) is int
        and isinstance(batch, list)
        and 0 <= index < len(batch)
        and batch[index].get("study_uid") == record["study_uid"],
        "differs from the exact original source",
    )
    assert isinstance(batch, list) and isinstance(index, int)
    report = batch[index]["report"]
    batch_number = Path(record["source_path"]).stem.removeprefix("batch-")
    _temporal_check(record["case"] == f"{batch_number}:{index:02d}", "has wrong source index")
    quotes = proof.get("source_evidence")
    _temporal_check(
        isinstance(quotes, list)
        and len(quotes) >= 2
        and all(isinstance(q, str) and q and q in report for q in quotes),
        "has changed source quotations",
    )

    def linked(document: dict, field: str, role: str) -> bool:
        ref = document.get(field)
        return isinstance(ref, dict) and ref.get("sha256") == refs[role]["sha256"]

    def case_matches(document: dict) -> bool:
        return (
            document.get("case") == record["case"]
            and document.get("study_uid") == record["study_uid"]
        )

    root = docs["root_acceptance"]
    accepted = docs["independent_acceptance"]
    independent = docs["independent_policy"]
    for key, stage in {
        "root_visual": "independent_root_visual_assessment_frozen_before_other_visual_findings",
        "root_numeric": "completed_bounded_root_source_visual_numeric_crosscheck_with_open_followups",
        "independent_visual": "independent_064046_metadata_and_visual_freeze_before_proposal",
        "panel_manifest": "bound_cached_representative_side_consistency_panels",
    }.items():
        _temporal_check(
            docs[key].get("stage") == stage, "has an unsupported original evidence format"
        )
    agent = _assessor_id(independent.get("agent_id"))
    _temporal_check(
        agent != "/root" and _assessor_id(proof.get("agent_id")) == agent,
        "requires distinct root and independent assessors",
    )
    for key in (
        "root_source",
        "root_visual",
        "root_numeric",
        "root_proposal",
        "root_acceptance",
        "root_lineage_correction",
    ):
        _temporal_check(
            _assessor_id(docs[key].get("agent_id")) == "/root", "has wrong root assessor"
        )
    for key in ("independent_visual", "independent_numeric", "independent_acceptance"):
        _temporal_check(
            _assessor_id(docs[key].get("agent_id")) == agent, "has wrong independent assessor"
        )
    source = docs["independent_source"]
    _temporal_check(
        _assessor_id(_selected(source, "/assessor/agent_id")) == agent,
        "has wrong independent source assessor",
    )
    _temporal_check(
        root.get("stage")
        == "root_accepted_temporal_correspondence_exclusion_after_independent_assessment"
        and root.get("status") == "accepted"
        and case_matches(root)
        and root.get("action") == TEMPORAL_ACTION
        and root.get("preserve_original_reports_annotations_images_and_gold") is True
        and root.get("source_index") == index
        and root.get("source_report_sha256") == record["source_report_sha256"]
        and root.get("source_batch", {}).get("path") == "recovered/" + record["source_path"]
        and root.get("source_batch", {}).get("sha256") == record["source_batch_sha256"]
        and root.get("independent_agent_id") == agent,
        "lacks the actual accepted root action",
    )
    for field, role in {
        "proposal": "root_proposal",
        "root_source_freeze": "root_source",
        "root_visual_freeze": "root_visual",
        "root_numeric_comparison": "root_numeric",
        "independent_source_freeze": "independent_source",
        "independent_metadata_visual_freeze": "independent_visual",
        "independent_policy_assessment": "independent_policy",
    }.items():
        _temporal_check(linked(root, field, role), "has altered root acceptance lineage")
    _temporal_check(
        accepted.get("stage") == "independent_accepted_temporal_correspondence_exclusion"
        and accepted.get("status") == "accepted"
        and case_matches(accepted)
        and accepted.get("action") == TEMPORAL_ACTION
        and accepted.get("required_report_loss_mask") == [False] * len(TARGETS)
        and all(v is False for v in accepted["required_report_loss_mask"])
        and linked(accepted, "root_acceptance", "root_acceptance")
        and linked(accepted, "independent_policy_assessment", "independent_policy")
        and independent.get("stage")
        == "independent_policy_assessment_complete_for_root_integration"
        and case_matches(independent)
        and _selected(independent, "/assessment_of_root_proposal/supported") is True
        and _selected(
            independent,
            "/assessment_of_root_proposal/proposed_action/report_derived_training_supervision_allowed",
        )
        is False
        and linked(independent, "source_first_freeze", "independent_source")
        and linked(independent, "independent_metadata_visual_freeze", "independent_visual")
        and linked(independent, "numeric_and_exact_before_policy_context", "independent_numeric"),
        "lacks the actual independent accepted action and policy content",
    )
    packet = _json(_bound_path(refs["source_packet"], artifact_root))
    selected_packet = _selected(packet, proof.get("source_packet_selector"))
    root_source = _selected(docs["root_source"], proof.get("root_source_selector"))
    independent_source = _selected(source, proof.get("independent_source_selector"))
    _temporal_check(
        all(isinstance(v, dict) for v in (selected_packet, root_source, independent_source)),
        "has invalid source selectors",
    )
    assert isinstance(selected_packet, dict)
    assert isinstance(root_source, dict)
    assert isinstance(independent_source, dict)
    for item in (selected_packet, root_source, independent_source):
        _temporal_check(
            item.get("study_uid") == record["study_uid"]
            and item.get("index") == index
            and item.get("batch") == batch_number
            and item.get("source_sha256") == record["source_batch_sha256"],
            "source selector identifies another record",
        )
    _temporal_check(
        selected_packet.get("report") == report
        and independent_source.get("report") == report
        and independent_source.get("report_sha256") == record["source_report_sha256"]
        and docs["root_source"].get("source_packet_sha256") == refs["source_packet"]["sha256"]
        and source.get("status") == "frozen_independent_source_first_assessment"
        and docs["root_source"].get("stage") == "source_first_technical_correspondence_freeze",
        "full source freeze differs from original report",
    )
    frozen_quotes = _selected(independent_source, "/assessment/literal_quotes")
    _temporal_check(frozen_quotes == quotes, "quotes differ from independent source freeze")
    root_visual = docs["root_visual"]
    correction = docs["root_lineage_correction"]
    _temporal_check(
        correction.get("stage") == "explicit_recorded_hash_binding_correction"
        and correction.get("status") == "accepted_record_correction"
        and correction.get("field") == "/source_preread_sha256"
        and linked(correction, "unchanged_original_visual_freeze", "root_visual")
        and linked(correction, "intended_unchanged_source_freeze", "root_source")
        and correction.get("recorded_wrong_sha256") == root_visual.get("source_preread_sha256")
        and correction.get("corrected_sha256") == refs["root_source"]["sha256"]
        and isinstance(correction.get("access_order_limit"), str)
        and bool(correction["access_order_limit"]),
        "lacks the explicit historical source-link correction",
    )
    visual = docs["independent_visual"]
    root_visual_case = _selected(root_visual, proof.get("root_visual_selector"))
    panel = docs["panel_manifest"]
    metadata = docs["technical_metadata"]
    numeric = docs["independent_numeric"]
    _temporal_check(
        isinstance(root_visual_case, dict)
        and case_matches(root_visual_case)
        and root_visual_case.get("manifest_sha256") == refs["panel_manifest"]["sha256"]
        and root_visual_case.get("panels_viewed") == 12
        and case_matches(visual)
        and visual.get("all_12_matrices_viewed") is True
        and visual.get("root_proposal_read") is False
        and linked(visual, "source_freeze", "independent_source")
        and linked(visual, "panel_manifest", "panel_manifest")
        and linked(visual, "technical_metadata", "technical_metadata")
        and case_matches(panel)
        and panel.get("cached_panels") == 12
        and panel.get("series_sampled") == 4
        and panel.get("all_source_membership_verified") == 130
        and case_matches(metadata)
        and metadata.get("series_count") == 4
        and metadata.get("slices") == 130
        and numeric.get("stage") == "independent_numeric_binding_checks_passed"
        and _selected(numeric, "/064_46/source_memberships") == 130
        and _selected(numeric, "/064_46/panel_manifest/sha256") == refs["panel_manifest"]["sha256"],
        "has inconsistent technical sample/metadata evidence",
    )
    checks = _selected(numeric, "/064_46/sample_matrices")
    _temporal_check(
        isinstance(checks, list)
        and len(checks) == 12
        and all(
            isinstance(c, dict)
            and c.get("cache_bytes_and_png_crop_equal") is True
            and c.get("source_sop_and_stored_native_fingerprint_bound") is True
            for c in checks
        ),
        "lacks exact numeric display checks",
    )
    _temporal_check(
        linked(docs["root_numeric"], "root_source_preread", "root_source")
        and linked(docs["root_numeric"], "root_visual_freeze", "root_visual"),
        "has changed root numeric lineage",
    )
    before = _selected(numeric, "/before_policy")
    _temporal_check(
        isinstance(before, dict)
        and before.get("064_46_present") is False
        and before.get("enforced_exclusions") == 16
        and docs["before_registry"].get("target_order") == list(TARGETS)
        and any(
            isinstance(v, dict)
            and v.get("snapshot", {}).get("sha256") == refs["before_registry"]["sha256"]
            for v in before.get("bound_files", [])
        ),
        "does not bind the exact before-policy context",
    )
    root_times = [
        _temporal_time(docs[k], field)
        for k, field in (
            ("root_source", "at_utc"),
            ("root_visual", "at_utc"),
            ("root_numeric", "at_utc"),
        )
    ]
    independent_times = [
        _temporal_time(docs[k], "created_at_utc")
        for k in (
            "independent_source",
            "independent_visual",
            "independent_numeric",
            "independent_policy",
        )
    ]
    accepted_time = _temporal_time(root, "completed_at_utc")
    _temporal_check(
        root_times == sorted(root_times)
        and independent_times == sorted(independent_times)
        and max(root_times + independent_times)
        <= accepted_time
        <= _temporal_time(accepted, "created_at_utc")
        <= _temporal_time(proof, "created_at_utc"),
        "has malformed recorded source/visual/acceptance ordering",
    )


def _validate_source_header_proof(record: dict, proof: dict, artifact_root: Path) -> None:
    """Bind the actual source/header formats and recorded review chronology.

    This authenticates preserved decisions, not image interpretation or past access
    order. Historical exposure disclosures remain part of the bound source records.
    """

    def check(condition: bool, context: str) -> None:
        if not condition:
            raise ValueError("Source/header correspondence evidence " + context)

    stages = {
        "root_source": "root_three_additional_full_source_structure_assessments",
        "root_metadata": "root_four_case_corrected_source_header_cache_identity_replay",
        "root_proposal": "root_source_metadata_correspondence_proposals_before_peer_policy_comparison",
        "root_acceptance": "root_accepted_four_case_source_metadata_correspondence_dispositions",
        "independent_source": "independent_four_case_full_source_structure_freeze",
        "independent_source_freeze": "source_only_before_new_raw_technical_and_policy_review",
        "independent_metadata": "post_source_freeze_raw_technical_metadata_read_and_binding_check",
        "independent_policy": "independent_post_source_freeze_correspondence_policy_suggestions",
        "independent_acceptance": "independent_accepted_source_header_correspondence_exclusion",
    }
    roles = set(stages) | {
        "source_packet",
        "independent_source_packet",
        "technical_metadata",
        "root_replay_script",
    }
    refs = proof.get("evidence")
    check(isinstance(refs, dict) and set(refs) == roles, "has invalid evidence roles")
    assert isinstance(refs, dict)
    paths = {key: _bound_path(refs[key], artifact_root) for key in roles}
    docs = {key: _object(path) for key, path in paths.items() if key != "root_replay_script"}
    for key, stage in stages.items():
        check(docs[key].get("stage") == stage, "has an unsupported original evidence format")
    agent = _assessor_id(docs["independent_policy"].get("agent_id"))
    check(
        agent != "/root" and _assessor_id(proof.get("agent_id")) == agent,
        "lacks distinct assessors",
    )
    for key in stages:
        check(
            _assessor_id(docs[key].get("agent_id"))
            == ("/root" if key.startswith("root_") else agent),
            "has changed assessor identity",
        )

    source_ref = {"path": record["source_path"], "sha256": record["source_batch_sha256"]}
    batch = _json(_bound_path(source_ref, artifact_root))
    index = proof.get("source_index")
    check(
        type(proof.get("schema_version")) is int
        and proof["schema_version"] == 1
        and proof.get("source_batch") == source_ref
        and proof.get("source_report_sha256") == record["source_report_sha256"]
        and isinstance(batch, list)
        and type(index) is int
        and 0 <= index < len(batch)
        and batch[index].get("study_uid") == record["study_uid"],
        "differs from the exact source/index",
    )
    assert isinstance(batch, list) and isinstance(index, int)
    report = batch[index]["report"]
    batch_number = Path(record["source_path"]).stem.removeprefix("batch-")
    check(
        record["case"] == f"{batch_number}:{index:02d}"
        and hashlib.sha256(report.encode()).hexdigest() == record["source_report_sha256"],
        "has changed full source identity",
    )

    def linked(document: dict, field: str, role: str) -> bool:
        ref = document.get(field)
        return isinstance(ref, dict) and ref.get("sha256") == refs[role]["sha256"]

    def identity(item: dict) -> bool:
        return item.get("case") == record["case"] and item.get("study_uid") == record["study_uid"]

    selectors = proof.get("selectors")
    selected_roles = set(stages) - {"independent_source_freeze", "independent_acceptance"}
    selected_roles |= {"source_packet", "independent_source_packet"}
    check(isinstance(selectors, dict) and set(selectors) == selected_roles, "has invalid selectors")
    assert isinstance(selectors, dict)
    selected = {}
    for role in selected_roles:
        item = _selected(docs[role], selectors[role])
        check(isinstance(item, dict) and identity(item), "selector identifies another case/study")
        assert isinstance(item, dict)
        selected[role] = item
    for role, item in selected.items():
        if role == "independent_metadata":
            continue
        index_key = (
            "source_index_zero_based"
            if role.startswith("root_")
            else "source_batch_index_zero_based"
        )
        check(
            item.get(index_key) == index
            and type(item.get(index_key)) is int
            and item.get("source_batch", {}).get("path") == "recovered/" + record["source_path"]
            and item.get("source_batch", {}).get("sha256") == record["source_batch_sha256"]
            and item.get("report_utf8_sha256") == record["source_report_sha256"],
            "source lineage identifies another original report",
        )
    for role in ("source_packet", "independent_source_packet"):
        check(selected[role].get("full_report_text") == report, "full packet source was changed")
    check(
        selected["root_source"].get("complete_source_read") is True
        and selected["root_source"].get("packet_sha256") == refs["source_packet"]["sha256"]
        and selected["independent_source"].get("complete_source_personally_read_this_stage") is True
        and linked(docs["independent_source"], "source_packet", "independent_source_packet")
        and linked(docs["independent_source_freeze"], "assessment", "independent_source")
        and linked(docs["independent_source_freeze"], "source_packet", "independent_source_packet")
        and isinstance(docs["root_source"].get("prior_exposure"), str)
        and bool(docs["root_source"]["prior_exposure"])
        and isinstance(docs["independent_source"].get("prior_exposure_disclosure"), list)
        and bool(docs["independent_source"]["prior_exposure_disclosure"]),
        "lacks the recorded full source freezes and exposure lineage",
    )
    for role, key in (
        ("root_source", "exact_source_quotes"),
        ("independent_source", "source_quotes"),
    ):
        quotes = selected[role].get(key)
        check(
            isinstance(quotes, list)
            and bool(quotes)
            and all(isinstance(q, str) and q and q in report for q in quotes),
            "has changed exact source quotations",
        )
    check(
        proof.get("source_evidence") == selected["independent_source"]["source_quotes"],
        "quotes differ from the independent source freeze",
    )
    for role in ("root_metadata", "root_proposal", "root_acceptance"):
        check(
            any(
                isinstance(ref, dict) and ref.get("sha256") == refs["root_source"]["sha256"]
                for ref in docs[role].get("root_source_freezes", [])
            ),
            "has changed root source lineage",
        )
    check(linked(docs["root_metadata"], "script", "root_replay_script"), "has changed replay code")
    for role in ("root_proposal", "root_acceptance"):
        check(
            linked(docs[role], "root_metadata_replay", "root_metadata"),
            "has changed root metadata lineage",
        )
    for field, role in {
        "root_proposals": "root_proposal",
        "independent_source": "independent_source",
        "independent_source_freeze": "independent_source_freeze",
        "independent_policy": "independent_policy",
        "independent_raw_technical_review": "independent_metadata",
    }.items():
        check(linked(docs["root_acceptance"], field, role), "has changed accepted evidence lineage")
    for document, fields in (
        (
            "independent_metadata",
            {
                "source_freeze": "independent_source_freeze",
                "source_assessment": "independent_source",
            },
        ),
        (
            "independent_policy",
            {
                "source_first_freeze": "independent_source_freeze",
                "source_first_assessment": "independent_source",
                "raw_technical_review": "independent_metadata",
            },
        ),
        (
            "independent_acceptance",
            {
                "root_acceptance": "root_acceptance",
                "independent_policy_assessment": "independent_policy",
            },
        ),
    ):
        for field, role in fields.items():
            check(linked(docs[document], field, role), "has changed independent evidence lineage")
    accepted = docs["independent_acceptance"]
    check(
        selected["root_proposal"].get("proposed_action") == TEMPORAL_ACTION
        and selected["root_acceptance"].get("accepted_action") == TEMPORAL_ACTION
        and selected["root_acceptance"].get("accepted") is True
        and selected["root_acceptance"].get("independent_assessor") == agent
        and selected["root_acceptance"].get("independent_policy_selector")
        == selectors["independent_policy"]
        and selected["independent_policy"].get("policy_suggestion")
        == "propose_whole_study_non_gold_report_supervision_exclusion_pending_exact_correspondence_binding"
        and identity(accepted)
        and accepted.get("status") == "accepted"
        and accepted.get("action") == TEMPORAL_ACTION
        and accepted.get("source_batch") == source_ref
        and accepted.get("source_index") == index
        and accepted.get("source_report_sha256") == record["source_report_sha256"]
        and accepted.get("required_report_loss_mask") == [False] * len(TARGETS)
        and all(v is False for v in accepted["required_report_loss_mask"]),
        "lacks actual distinct accepted exclusion actions",
    )
    metadata = docs["technical_metadata"]
    check(identity(metadata), "technical metadata identifies another study")
    check(
        linked(selected["root_metadata"], "technical_evidence", "technical_metadata")
        and any(
            ref.get("sha256") == refs["technical_metadata"]["sha256"]
            for ref in selected["independent_metadata"].get("metadata_files", [])
        )
        and any(
            ref.get("sha256") == refs["technical_metadata"]["sha256"]
            for ref in selected["independent_policy"].get("metadata_proofs", [])
        )
        and docs["root_metadata"].get("status") == "passed",
        "does not bind both metadata assessments",
    )
    series = metadata.get("series")
    check(isinstance(series, list) and bool(series), "lacks raw technical series records")
    assert isinstance(series, list)
    root_series = selected["root_metadata"].get("series", [])
    peer_series = selected["independent_metadata"].get("series", [])
    check(
        len(series) == len(root_series) == len(peer_series) == metadata.get("series_count")
        and len({s["series_uid"] for s in series}) == len(series),
        "has inconsistent series membership",
    )
    for source_series in series:
        uid = source_series["series_uid"]
        roots = [s for s in root_series if s.get("series_uid") == uid]
        peers = [s for s in peer_series if s.get("series_uid") == uid]
        check(len(roots) == len(peers) == 1, "has changed assessed series membership")
        root_row, peer_row = roots[0], peers[0]
        fields = source_series["header_endpoint_values"]
        check(
            root_row.get("n_sources")
            == peer_row.get("source_member_count")
            == source_series["n_slices"]
            and root_row.get("header_endpoints") == 2
            and peer_row.get("header_endpoint_values") == fields
            and root_row.get("literal_fields")
            == {
                key: fields[key]
                for key in (
                    "SeriesNumber",
                    "SeriesDescription",
                    "Laterality",
                    "ImageLaterality",
                    "FrameLaterality",
                    "SeriesTime",
                    "AcquisitionTime",
                )
            },
            "has altered literal headers or source membership counts",
        )
    scope = (len(series), sum(s["n_slices"] for s in series), 2 * len(series))
    for role in ("root_metadata", "root_proposal", "root_acceptance"):
        check(
            tuple(selected[role].get(k) for k in ("n_series", "n_sources", "n_endpoints")) == scope,
            "has inconsistent root technical scope",
        )
    check(
        tuple(
            selected["independent_metadata"].get(k)
            for k in ("series_count", "source_memberships", "lexical_endpoint_header_count")
        )
        == scope
        and tuple(
            selected["independent_policy"].get("technical_scope", {}).get(k)
            for k in ("series", "source_memberships", "lexical_endpoints")
        )
        == scope
        and metadata.get("slices") == scope[1]
        and metadata.get("endpoint_count") == scope[2],
        "has inconsistent independent technical scope",
    )

    def time(role: str) -> datetime:
        try:
            value = datetime.fromisoformat((proof if role == "proof" else docs[role])["at_utc"])
            if value.tzinfo is not None and value.utcoffset() is not None:
                return value
        except (KeyError, TypeError, ValueError):
            pass
        raise ValueError("Source/header correspondence evidence has invalid recorded chronology")

    root_times = [time(role) for role in ("root_source", "root_metadata", "root_proposal")]
    peer_times = [
        time(role)
        for role in (
            "independent_source",
            "independent_source_freeze",
            "independent_metadata",
            "independent_policy",
        )
    ]
    check(
        root_times == sorted(root_times)
        and peer_times == sorted(peer_times)
        and max(root_times + peer_times)
        <= time("root_acceptance")
        <= time("independent_acceptance")
        <= time("proof"),
        "has malformed recorded source/metadata/acceptance ordering",
    )


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
    if proof.get("stage") == TEMPORAL_STAGE:
        _validate_temporal_proof(record, proof, artifact_root)
    if proof.get("stage") == SOURCE_HEADER_STAGE:
        _validate_source_header_proof(record, proof, artifact_root)
    if proof.get("stage") == "root_verified_correspondence_source_identity_binding":
        review_path = _bound_path(proof.get("verified_review"), artifact_root)
        handoff_path = _bound_path(proof.get("source_handoff"), artifact_root)
        review, handoff = _object(review_path), _object(handoff_path)
        source = {
            "path": record["source_path"],
            "sha256": record["source_batch_sha256"],
        }
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
        if type(binding_version) is not int or binding_version not in {1, 2, 3, 4}:
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
                {
                    "path": record.get("source_path"),
                    "sha256": record.get("source_batch_sha256"),
                },
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
            if binding_version == 3:
                proof = _object(_bound_path(record.get("verified_technical_proof"), artifact_root))
                if proof.get("stage") != TEMPORAL_STAGE:
                    raise ValueError("Correspondence policy requires a temporal source binding")
            if binding_version == 4:
                proof = _object(_bound_path(record.get("verified_technical_proof"), artifact_root))
                if proof.get("stage") != SOURCE_HEADER_STAGE:
                    raise ValueError("Correspondence policy requires a source/header binding")
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
