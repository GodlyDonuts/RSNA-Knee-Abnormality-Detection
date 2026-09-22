import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna.adjudications import apply_adjudications, inspect_adjudications
from rsna.labels import (
    audit_labels,
    build_studies,
    file_sha256,
    read_reviews,
    source_reports_fingerprint,
)
from rsna.schema import TARGETS


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def reference(root: Path, path: Path) -> dict:
    return {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}


@pytest.fixture
def bundle(tmp_path: Path) -> dict:
    root = tmp_path
    base = root / "initiative/reviews-v2"
    reviews_folder = base / "reviewed"
    paths = {
        "rubric": root / "protocols/report_weak_v2.md",
        "source_batch": root / "initiative/reviews/inputs/batch-000.json",
        "base_review": reviews_folder / "batch-000-reviewed.jsonl",
        "proposal": root / "initiative/audits/correction/proposal.json",
        "independent_assessment": root / "initiative/audits/correction/assessment.json",
        "decision": root / "initiative/audits/correction/decision.json",
        "origin": root / "initiative/audits/correction/origin.json",
        "case": base / "adjudications/cases/case-001.json",
        "registry": base / "adjudications/registry.json",
        "qa": base / "adjudications/qa/cartilage.json",
    }
    reports = {uid: f"Grade 2 cartilage. Normal meniscus. Study {uid}." for uid in ("gold", "weak")}
    write(paths["rubric"], "Frozen test rubric")
    write(paths["origin"], {"finding": "No established grading convention"})
    write(
        paths["source_batch"],
        [{"study_uid": uid, "report": report} for uid, report in reports.items()],
    )
    context = {
        "source_reports_sha256": source_reports_fingerprint(reports),
        "rubric_sha256": file_sha256(paths["rubric"]),
    }
    rows, changes, proposed, assessed = [], [], [], []
    for uid in reports:
        row = {
            "study_uid": uid,
            "states": ["not_mentioned"] * 12,
            "evidence": [""] * 12,
            "notes": "Source review",
        }
        row["states"][TARGETS.index("PF OA")] = "negative"
        row["evidence"][TARGETS.index("PF OA")] = "Grade 2 cartilage."
        rows.append(row)
        changes.append(
            {
                "study_uid": uid,
                "target": "PF OA",
                "before": {"state": "negative", "evidence": "Grade 2 cartilage."},
                "after": {"state": "uncertain", "evidence": "Grade 2 cartilage."},
            }
        )
        proposed.append(
            {
                "study_uid": uid,
                "target": "PF OA",
                "published_state": "negative",
                "proposed_state": "uncertain",
            }
        )
        assessed.append(
            {
                "study_uid": uid,
                "target": "PF OA",
                "independent_source_state": "uncertain",
                "source_evidence": "Grade 2 cartilage.",
                "reason": "Grading convention remains unresolved",
            }
        )
    paths["base_review"].parent.mkdir(parents=True)
    paths["base_review"].write_text("".join(json.dumps(row) + "\n" for row in rows))
    write(
        paths["proposal"],
        {
            "agent_id": "reader",
            "session_id": "reader-session",
            "artifact_sha256": file_sha256(paths["base_review"]),
            "source_sha256": file_sha256(paths["source_batch"]),
            "proposed_masks": proposed,
        },
    )
    write(
        paths["independent_assessment"],
        {
            "stage": "independent_source_assessment",
            "agent_id": "adjudicator",
            "session_id": "adjudicator-session",
            "source_sha256": file_sha256(paths["source_batch"]),
            "rubric_sha256": context["rubric_sha256"],
            "records": assessed,
        },
    )
    write(
        paths["decision"],
        {
            **context,
            "status": "independently_adjudicated_pending_release_integration",
            "original_review_preserved": True,
            "original_review_sha256": file_sha256(paths["base_review"]),
            "proposal_sha256": file_sha256(paths["proposal"]),
            "independent_assessment_sha256": file_sha256(paths["independent_assessment"]),
            "origin_probe_summary_sha256": file_sha256(paths["origin"]),
            "proposer_agent_id": "reader",
            "adjudicator_agent_id": "adjudicator",
            "accepted_masks": assessed,
            "n_changes": len(changes),
        },
    )
    write(
        paths["case"],
        {
            **context,
            "schema_version": 1,
            "id": "cartilage-001",
            **{
                name: reference(root, paths[name])
                for name in (
                    "source_batch",
                    "base_review",
                    "proposal",
                    "independent_assessment",
                    "decision",
                )
            },
            "supporting_evidence": {
                "origin_probe_summary_sha256": reference(root, paths["origin"])
            },
            "changes": changes,
        },
    )
    write(
        paths["registry"],
        {
            **context,
            "schema_version": 1,
            "cases": [reference(root, paths["case"])],
            "semantic_qa": [],
        },
    )
    return {
        "root": root,
        "paths": paths,
        "reports": reports,
        "context": context,
        "reviews_folder": reviews_folder,
        "reviews": read_reviews(reviews_folder, reports, "-reviewed"),
    }


def inspect(bundle: dict) -> dict:
    return inspect_adjudications(
        bundle["reports"], bundle["reviews_folder"], bundle["paths"]["rubric"], bundle["root"]
    )


def apply(bundle: dict) -> tuple[pd.DataFrame, dict]:
    return apply_adjudications(
        bundle["reviews"],
        bundle["reports"],
        bundle["reviews_folder"],
        bundle["paths"]["rubric"],
        bundle["root"],
    )


def rebind_case(bundle: dict) -> None:
    paths = bundle["paths"]
    registry = read(paths["registry"])
    registry["cases"] = [reference(bundle["root"], paths["case"])]
    write(paths["registry"], registry)


def update_decision(bundle: dict, **values: object) -> None:
    paths = bundle["paths"]
    write(paths["decision"], {**read(paths["decision"]), **values})
    case = read(paths["case"])
    case["decision"] = reference(bundle["root"], paths["decision"])
    write(paths["case"], case)
    rebind_case(bundle)


def add_pending(bundle: dict) -> None:
    paths, root = bundle["paths"], bundle["root"]
    write(
        paths["qa"],
        {
            **bundle["context"],
            "schema_version": 1,
            "id": "cartilage-consistency",
            "status": "pending",
            "scope": "all_release_reviews",
            "reason": "Audit every completed batch",
            "evidence": [reference(root, paths["decision"])],
        },
    )
    registry = read(paths["registry"])
    registry["semantic_qa"] = [reference(root, paths["qa"])]
    write(paths["registry"], registry)


def test_valid_correction_masks_weak_and_preserves_originals_and_gold(bundle: dict) -> None:
    original_bytes = bundle["paths"]["base_review"].read_bytes()
    original_frame = bundle["reviews"].copy(deep=True)
    effective, trace = apply(bundle)
    assert trace["n_changes"] == 2
    assert effective["PF OA__state"].eq("uncertain").all()
    assert effective["PF OA__original_state"].eq("negative").all()
    assert json.loads(effective.iloc[0]["PF OA__correction"])["before"]["state"] == "negative"
    assert bundle["paths"]["base_review"].read_bytes() == original_bytes
    pd.testing.assert_frame_equal(bundle["reviews"], original_frame)
    train = pd.DataFrame(
        [
            {
                "StudyInstanceUID": uid,
                "Report": report,
                **{target: (1 if uid == "gold" else np.nan) for target in TARGETS},
            }
            for uid, report in bundle["reports"].items()
        ]
    )
    studies = build_studies(train, effective).set_index("study_uid")
    assert studies.loc["weak", "PF OA__mask"] == 0
    assert studies.loc["weak", "PF OA__provenance"] == "uncertain"
    assert studies.loc["gold", "split"] == "gold_evaluation"
    assert studies.loc["gold", list(TARGETS)].eq(1).all()
    assert studies.loc["gold", [f"{target}__mask" for target in TARGETS]].eq(1).all()
    assert (
        studies.loc["gold", [f"{target}__provenance" for target in TARGETS]]
        .eq("official_gold")
        .all()
    )
    assert studies.loc["gold", "PF OA__correction"]
    audit = audit_labels(studies.reset_index(), original_frame)
    assert audit["targets"]["PF OA"]["reviewer_state_changes"] == 0
    assert audit["targets"]["PF OA"]["adjudicated_state_changes"] == 2


@pytest.mark.parametrize(
    "name",
    ["source_batch", "base_review", "proposal", "independent_assessment", "decision", "origin"],
)
def test_changed_bound_artifact_is_rejected(bundle: dict, name: str) -> None:
    path = bundle["paths"][name]
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="checksum"):
        apply(bundle)


@pytest.mark.parametrize(
    "field,quote,message",
    [
        ("before", "Normal meniscus.", "before state/evidence"),
        ("after", "Normal meniscus.", "accepted assessment"),
        ("after", "Invented cartilage finding", "not verbatim"),
    ],
)
def test_changed_before_or_after_quote_is_rejected(
    bundle: dict, field: str, quote: str, message: str
) -> None:
    case = read(bundle["paths"]["case"])
    case["changes"][0][field]["evidence"] = quote
    write(bundle["paths"]["case"], case)
    rebind_case(bundle)
    with pytest.raises(ValueError, match=message):
        apply(bundle)


def test_source_and_rubric_context_cannot_change(bundle: dict) -> None:
    bundle["reports"]["weak"] += " Extra wording."
    with pytest.raises(ValueError, match="source reports/rubric"):
        apply(bundle)
    bundle["reports"]["weak"] = bundle["reports"]["weak"].removesuffix(" Extra wording.")
    bundle["paths"]["rubric"].write_text("Changed rubric")
    with pytest.raises(ValueError, match="source reports/rubric"):
        apply(bundle)


def test_self_approval_and_unaccepted_decision_are_rejected(bundle: dict) -> None:
    update_decision(bundle, adjudicator_agent_id="reader")
    with pytest.raises(ValueError, match="Self approval"):
        apply(bundle)
    update_decision(
        bundle,
        adjudicator_agent_id="adjudicator",
        status="proposal_pending_independent_adjudication",
    )
    with pytest.raises(ValueError, match="explicitly accepted"):
        apply(bundle)


def test_same_session_and_forged_identity_are_rejected(bundle: dict) -> None:
    paths = bundle["paths"]
    assessment = read(paths["independent_assessment"])
    assessment["session_id"] = "reader-session"
    write(paths["independent_assessment"], assessment)
    case = read(paths["case"])
    case["independent_assessment"] = reference(bundle["root"], paths["independent_assessment"])
    write(paths["case"], case)
    update_decision(
        bundle, independent_assessment_sha256=file_sha256(paths["independent_assessment"])
    )
    with pytest.raises(ValueError, match="proposer session"):
        apply(bundle)
    update_decision(bundle, adjudicator_agent_id="unrelated-agent")
    with pytest.raises(ValueError, match="identities do not match"):
        apply(bundle)


def test_duplicate_and_conflicting_targets_are_rejected(bundle: dict) -> None:
    case = read(bundle["paths"]["case"])
    case["changes"].append(copy.deepcopy(case["changes"][0]))
    write(bundle["paths"]["case"], case)
    rebind_case(bundle)
    with pytest.raises(ValueError, match="Duplicate/conflicting"):
        apply(bundle)
    case["changes"].pop()
    write(bundle["paths"]["case"], case)
    rebind_case(bundle)
    other = bundle["paths"]["case"].with_name("case-002.json")
    write(other, {**case, "id": "another-case"})
    registry = read(bundle["paths"]["registry"])
    registry["cases"].append(reference(bundle["root"], other))
    write(bundle["paths"]["registry"], registry)
    with pytest.raises(ValueError, match="Duplicate/conflicting correction target"):
        apply(bundle)


def test_partial_accepted_decision_cannot_be_silently_applied(bundle: dict) -> None:
    case = read(bundle["paths"]["case"])
    case["changes"].pop()
    write(bundle["paths"]["case"], case)
    rebind_case(bundle)
    with pytest.raises(ValueError, match="exactly cover"):
        apply(bundle)


def test_pending_qa_blocks_release_but_all_bindings_can_be_inspected(bundle: dict) -> None:
    add_pending(bundle)
    plan = inspect(bundle)
    assert plan["n_changes"] == 2
    assert plan["pending_semantic_qa"] == ["cartilage-consistency"]
    with pytest.raises(ValueError, match="Pending semantic QA"):
        apply(bundle)


@pytest.mark.parametrize("field", ["cases", "semantic_qa"])
def test_registry_omission_is_rejected(bundle: dict, field: str) -> None:
    add_pending(bundle)
    registry = read(bundle["paths"]["registry"])
    registry[field] = []
    write(bundle["paths"]["registry"], registry)
    with pytest.raises(ValueError, match="omitted"):
        apply(bundle)


def test_missing_registry_fails_and_explicit_empty_registry_passes(bundle: dict) -> None:
    paths = bundle["paths"]
    registry = read(paths["registry"])
    paths["registry"].unlink()
    with pytest.raises(ValueError, match="registry is missing"):
        apply(bundle)
    paths["case"].unlink()
    registry["cases"] = []
    write(paths["registry"], registry)
    effective, plan = apply(bundle)
    assert effective["PF OA__state"].eq("negative").all()
    assert plan["n_changes"] == 0


def test_semantic_resolution_is_bound_to_exact_review_artifacts(bundle: dict) -> None:
    add_pending(bundle)
    root, paths = bundle["root"], bundle["paths"]
    resolution_path = root / "initiative/audits/correction/qa-resolution.json"
    resolution = {
        **bundle["context"],
        "status": "resolved",
        "issue_id": "cartilage-consistency",
        "agent_id": "auditor",
        "session_id": "auditor-session",
        "notes": "All published cartilage decisions assessed",
        "review_artifacts": {
            str(paths["base_review"].relative_to(root)): file_sha256(paths["base_review"])
        },
        "correction_cases": {str(paths["case"].relative_to(root)): file_sha256(paths["case"])},
    }
    write(resolution_path, resolution)
    issue = read(paths["qa"])
    issue.update(status="resolved", resolution=reference(root, resolution_path))
    write(paths["qa"], issue)
    registry = read(paths["registry"])
    registry["semantic_qa"] = [reference(root, paths["qa"])]
    write(paths["registry"], registry)
    assert apply(bundle)[1]["pending_semantic_qa"] == []
    original_case = paths["case"].read_bytes()
    write(paths["case"], {**read(paths["case"]), "id": "changed-case"})
    rebind_case(bundle)
    with pytest.raises(ValueError, match="exact correction cases"):
        apply(bundle)
    paths["case"].write_bytes(original_case)
    rebind_case(bundle)
    later_review = bundle["reviews_folder"] / "batch-001-reviewed.jsonl"
    later_review.write_text(paths["base_review"].read_text())
    with pytest.raises(ValueError, match="exact release reviews"):
        apply(bundle)


def test_effective_frame_must_match_preserved_base(bundle: dict) -> None:
    bundle["reviews"].loc[0, "PF OA__state"] = "positive"
    with pytest.raises(ValueError, match="bound correction base"):
        apply(bundle)


def convert_to_multiple_batch_decision(bundle: dict) -> None:
    """Build the preserved accepted_changes format, covering different reviewed batches."""
    root, paths = bundle["root"], bundle["paths"]
    old_rows = [json.loads(line) for line in paths["base_review"].read_text().splitlines()]
    proposal = read(paths["proposal"])
    assessment = read(paths["independent_assessment"])
    decision = read(paths["decision"])
    case = read(paths["case"])
    proposed, assessed, accepted, changes = [], [], [], []
    for index, old_row in enumerate(old_rows):
        uid = old_row["study_uid"]
        old_row["states"][TARGETS.index("PF OA")] = "uncertain"
        base_path = paths["base_review"].with_name(f"batch-{index:03d}-reviewed.jsonl")
        base_path.write_text(json.dumps(old_row) + "\n")
        source_path = paths["source_batch"].with_name(f"batch-{index:03d}.json")
        write(source_path, [{"study_uid": uid, "report": bundle["reports"][uid]}])
        base_ref, source_ref = reference(root, base_path), reference(root, source_path)
        before = {"state": "uncertain", "evidence": "Grade 2 cartilage."}
        after = {
            "state": "negative" if index == 0 else "positive",
            "evidence": "Grade 2 cartilage.",
        }
        key = {"study_uid": uid, "target": "PF OA"}
        proposed.append(
            {
                **key,
                "before_state": before["state"],
                "proposed_state": after["state"],
                "source_quote": after["evidence"],
                "source_path": source_ref["path"],
                "source_sha256": source_ref["sha256"],
                "artifact_path": base_ref["path"],
                "artifact_sha256": base_ref["sha256"],
            }
        )
        assessment_row = {
            **key,
            "source_sha256": source_ref["sha256"],
            "source_evidence": after["evidence"],
            "independent_source_state": after["state"],
            "reason": "Explicit independently adjudicated test evidence",
        }
        assessed.append(assessment_row)
        accepted.append(
            {
                **assessment_row,
                "before_state": before["state"],
                "before_evidence": before["evidence"],
                "after_state": after["state"],
                "after_evidence": after["evidence"],
                "original_review_path": base_ref["path"],
                "original_review_sha256": base_ref["sha256"],
            }
        )
        changes.append(
            {
                **key,
                "before": before,
                "after": after,
                "base_review": base_ref,
                "source_batch": source_ref,
            }
        )
    for field in ("source_sha256", "artifact_sha256", "proposed_masks"):
        proposal.pop(field)
    proposal.update(bundle["context"], proposed_errata=proposed)
    assessment.pop("source_sha256")
    assessment.update(bundle["context"], records=assessed)
    for field in ("original_review_preserved", "original_review_sha256", "accepted_masks"):
        decision.pop(field)
    write(paths["proposal"], proposal)
    write(paths["independent_assessment"], assessment)
    decision.update(
        original_artifacts_preserved=True,
        accepted_changes=accepted,
        proposal_sha256=file_sha256(paths["proposal"]),
        independent_assessment_sha256=file_sha256(paths["independent_assessment"]),
    )
    write(paths["decision"], decision)
    case.pop("source_batch")
    case.pop("base_review")
    case.update(
        changes=changes,
        **{
            name: reference(root, paths[name])
            for name in ("proposal", "independent_assessment", "decision")
        },
    )
    write(paths["case"], case)
    rebind_case(bundle)
    bundle["reviews"] = read_reviews(bundle["reviews_folder"], bundle["reports"], "-reviewed")


def test_multiple_batch_decision_preserves_exact_before_after_and_gold(bundle: dict) -> None:
    convert_to_multiple_batch_decision(bundle)
    effective, plan = apply(bundle)
    assert plan["n_changes"] == 2
    effective = effective.set_index("study_uid")
    assert effective.loc["gold", "PF OA__state"] == "negative"
    assert effective.loc["weak", "PF OA__state"] == "positive"
    assert effective["PF OA__original_state"].eq("uncertain").all()
    train = pd.DataFrame(
        [
            {
                "StudyInstanceUID": uid,
                "Report": report,
                **{target: (1 if uid == "gold" else np.nan) for target in TARGETS},
            }
            for uid, report in bundle["reports"].items()
        ]
    )
    studies = build_studies(train, effective.reset_index()).set_index("study_uid")
    assert studies.loc["gold", "PF OA"] == 1
    assert studies.loc["gold", "PF OA__provenance"] == "official_gold"
    assert studies.loc["weak", "PF OA"] == 1
    assert studies.loc["weak", "PF OA__mask"] == 1
    assert studies.loc["weak", "PF OA__provenance"] == "report_adjudicated"


@pytest.mark.parametrize(
    "field,value,message",
    [
        ("original_review_sha256", "0" * 64, "source or base binding"),
        ("source_sha256", "0" * 64, "source or base binding"),
        ("before_evidence", "Normal meniscus.", "accepted exact annotation"),
    ],
)
def test_per_target_decision_binding_cannot_be_changed(
    bundle: dict, field: str, value: str, message: str
) -> None:
    convert_to_multiple_batch_decision(bundle)
    accepted = read(bundle["paths"]["decision"])["accepted_changes"]
    accepted[0][field] = value
    update_decision(bundle, accepted_changes=accepted)
    with pytest.raises(ValueError, match=message):
        apply(bundle)


def test_labels_cli_cannot_release_without_registry(
    bundle: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsna import labels

    root, paths = bundle["root"], bundle["paths"]
    paths["registry"].unlink()
    train_path = root / "train.csv"
    pd.DataFrame(
        [{"StudyInstanceUID": uid, "Report": report} for uid, report in bundle["reports"].items()]
    ).to_csv(train_path, index=False)
    drafts = root / "drafts"
    drafts.mkdir()
    (drafts / "batch-000-draft.jsonl").write_text(paths["base_review"].read_text())
    monkeypatch.setattr(labels, "validate_review_provenance", lambda *args: {})
    arguments = {
        "train": train_path,
        "reviews": bundle["reviews_folder"],
        "drafts": drafts,
        "output": root / "release",
        "dicom-audit": root / "unused-audit",
        "inventory": root / "unused-inventory",
        "source-revision": "a" * 40,
        "review-provenance": root / "unused-provenance",
        "rubric": paths["rubric"],
        "artifact-root": root,
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rsna.labels",
            *[item for key, value in arguments.items() for item in (f"--{key}", str(value))],
        ],
    )
    with pytest.raises(ValueError, match="registry is missing"):
        labels.main()
    assert not (root / "release").exists()


@pytest.mark.usefixtures("isolate_semantic_release")
def test_labels_cli_exports_effective_original_and_correction_lineage(
    bundle: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    from rsna import labels
    from rsna.correspondence import REGISTRY_PATH
    from rsna.report_similarity import REGISTRY_PATH as SIMILARITY_REGISTRY

    root, paths = bundle["root"], bundle["paths"]
    write(root / SIMILARITY_REGISTRY, {"schema_version": 1, "policies": []})
    write(
        root / REGISTRY_PATH,
        {"schema_version": 1, "target_order": list(TARGETS), "policies": []},
    )
    train_path = root / "train.csv"
    pd.DataFrame(
        [
            {
                "StudyInstanceUID": uid,
                "Report": report,
                **{target: 1 if uid == "gold" else np.nan for target in TARGETS},
            }
            for uid, report in bundle["reports"].items()
        ]
    ).to_csv(train_path, index=False)
    drafts = root / "drafts"
    drafts.mkdir()
    (drafts / "batch-000-draft.jsonl").write_text(paths["base_review"].read_text())
    inventory = root / "inventory.parquet"
    pd.DataFrame({"fixture": [1]}).to_parquet(inventory)
    audit_path = root / "audit.json"
    write(
        audit_path, {"fixture": "DICOM validation is independently covered by test_label_release"}
    )
    monkeypatch.setattr(labels, "validate_review_provenance", lambda *args: {})
    monkeypatch.setattr(labels, "validate_dicom_audit", lambda *args: [])
    output = root / "release"
    arguments = {
        "train": train_path,
        "reviews": bundle["reviews_folder"],
        "drafts": drafts,
        "output": output,
        "dicom-audit": audit_path,
        "inventory": inventory,
        "source-revision": "a" * 40,
        "review-provenance": root / "unused-provenance",
        "rubric": paths["rubric"],
        "artifact-root": root,
    }
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rsna.labels",
            *[item for key, value in arguments.items() for item in (f"--{key}", str(value))],
        ],
    )
    # A late release-boundary rejection must not expose a partial export.
    from rsna import release

    def reject_release(*args):
        raise ValueError("fixture late release rejection")

    with monkeypatch.context() as late:
        late.setattr(release, "write_verified_release", reject_release)
        with pytest.raises(ValueError, match="fixture late release rejection"):
            labels.main()
        assert not output.exists()
        assert not list(output.parent.glob(".release-*"))
    labels.main()
    original = pd.read_parquet(output / "report_annotations_original.parquet")
    effective = pd.read_parquet(output / "report_annotations.parquet")
    studies = pd.read_parquet(output / "studies.parquet").set_index("study_uid")
    csv = pd.read_csv(output / "study_labels.csv").set_index("study_uid")
    trace = read(output / "adjudication_trace.json")
    audit = read(output / "label_audit.json")
    assert original["PF OA__state"].eq("negative").all()
    assert effective["PF OA__state"].eq("uncertain").all()
    assert effective["PF OA__original_state"].eq("negative").all()
    assert studies.loc["weak", "PF OA__mask"] == 0
    assert studies.loc["gold", "PF OA"] == 1
    assert json.loads(csv.loc["weak", "PF OA__correction"])["decision"]["sha256"] == file_sha256(
        paths["decision"]
    )
    assert trace == audit["adjudications"]
    assert trace["n_changes"] == 2
    assert audit["release_inputs"]["adjudication_registry_sha256"] == file_sha256(paths["registry"])
