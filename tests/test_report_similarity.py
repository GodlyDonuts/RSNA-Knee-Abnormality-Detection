from __future__ import annotations

import copy
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna import labels
from rsna.labels import build_studies, file_sha256, report_group
from rsna.report_similarity import (
    REGISTRY_PATH,
    RELATION,
    bind_similarity_splits,
    inspect_report_similarity,
    similarity_pairs,
)
from rsna.schema import TARGETS


def _write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def _ref(root: Path, path: Path) -> dict:
    return {"path": str(path.relative_to(root)), "sha256": file_sha256(path)}


def _report_hash(report: str) -> str:
    return hashlib.sha256(report.encode()).hexdigest()


def _save_policy(bundle: dict) -> None:
    root = bundle["root"]
    _write(bundle["paths"]["policy"], bundle["policy"])
    _write(
        root / REGISTRY_PATH,
        {
            "schema_version": 1,
            "policies": [_ref(root, bundle["paths"]["policy"])],
        },
    )


def _rebind_evidence(bundle: dict, key: str) -> None:
    """Rehash a changed inner document through every outer dependency."""
    root, paths, docs = bundle["root"], bundle["paths"], bundle["docs"]
    _write(paths[key], docs[key])
    edge = bundle["policy"]["records"][0]
    if key in {"root", "independent"}:
        edge["source_assessments"][key] = _ref(root, paths[key])
        field = "root_source_freeze" if key == "root" else "own_source_freeze"
        docs["joint"][field] = _ref(root, paths[key])
        proof_field = (
            "root_source_freeze_sha256" if key == "root" else "independent_source_freeze_sha256"
        )
        docs["proof"][proof_field] = file_sha256(paths[key])
        _rebind_evidence(bundle, "joint")
    elif key == "joint":
        edge["joint_source_comparison"] = _ref(root, paths[key])
        docs["proof"]["joint_source_comparison_sha256"] = file_sha256(paths[key])
        _rebind_evidence(bundle, "proof")
    elif key == "proof":
        edge["difference_proof"] = _ref(root, paths[key])
        _save_policy(bundle)
    else:
        raise AssertionError(key)


def _bundle(
    root: Path, left: str = "Finding. . Impression.", right: str = "Finding. Impression."
) -> dict:
    reports = {
        "a": left,
        "b": right,
        "c": right,
        "d": "Finding. Different report.",
        "e": "Finding. Unrelated report.",
    }
    paths = {
        key: root / "evidence" / f"{key}.json" for key in ("root", "independent", "joint", "proof")
    }
    paths["batch"] = root / "initiative/reviews/inputs/batch-000.json"
    paths["policy"] = root / "initiative/audits/report-similarity-dispositions-v1.json"
    batch = [{"study_uid": uid, "report": reports[uid]} for uid in ("a", "b")]
    _write(paths["batch"], batch)
    endpoints = [
        {
            "study_uid": uid,
            "source_batch": _ref(root, paths["batch"]),
            "source_report_sha256": _report_hash(reports[uid]),
        }
        for uid in ("a", "b")
    ]
    docs = {}
    for role, stage in (
        ("root", "root_independent_complete_source_structure_preread"),
        ("independent", "independent_source_only_correspondence_preread"),
    ):
        docs[role] = {
            "stage": stage,
            "agent_id": f"agent-{role}",
            "session_id": f"session-{role}",
            "source_first": True,
            "independence": "Complete original reports read before other assessor findings.",
            "records": [
                {
                    "case": f"000:{index:02}",
                    "study_uid": endpoint["study_uid"],
                    "report": reports[endpoint["study_uid"]],
                    "source_report_sha256": endpoint["source_report_sha256"],
                    "source_batch" if role == "root" else "input_batch": copy.deepcopy(
                        endpoint["source_batch"]
                    ),
                }
                for index, endpoint in enumerate(endpoints)
            ],
        }
        _write(paths[role], docs[role])
    docs["joint"] = {
        "stage": "separate_independent_source_structure_comparison",
        "agent_id": "agent-independent",
        "root_source_freeze": _ref(root, paths["root"]),
        "own_source_freeze": _ref(root, paths["independent"]),
        "records": [
            {
                "case": f"000:{index:02}",
                "study_uid": endpoint["study_uid"],
                "source_report_sha256": endpoint["source_report_sha256"],
                "material_structural_disagreement": False,
                "source_structure_agreement": "Both independently read the complete original source.",
            }
            for index, endpoint in enumerate(endpoints)
        ],
    }
    _write(paths["joint"], docs["joint"])
    docs["proof"] = {
        "stage": "source_bound_report_similarity_edge_for_final_leakage_QA",
        "assessor": "agent-root",
        "study_uids": [endpoint["study_uid"] for endpoint in endpoints],
        "source_report_sha256": [endpoint["source_report_sha256"] for endpoint in endpoints],
        "root_source_freeze_sha256": file_sha256(paths["root"]),
        "independent_source_freeze_sha256": file_sha256(paths["independent"]),
        "joint_source_comparison_sha256": file_sha256(paths["joint"]),
    }
    _write(paths["proof"], docs["proof"])
    policy = {
        "schema_version": 1,
        "stage": "reviewed_report_similarity_leakage_edges",
        "status": "accepted",
        "records": [
            {
                "relation": RELATION,
                "status": "accepted",
                "source_assessment_order": {
                    "root": "full_sources_before_other_assessor_findings",
                    "independent": "full_sources_before_other_assessor_findings",
                },
                "left": endpoints[0],
                "right": endpoints[1],
                "source_assessments": {
                    role: _ref(root, paths[role]) for role in ("root", "independent")
                },
                "difference_proof": _ref(root, paths["proof"]),
                "joint_source_comparison": _ref(root, paths["joint"]),
            }
        ],
    }
    bundle = {
        "root": root,
        "reports": reports,
        "paths": paths,
        "docs": docs,
        "policy": policy,
        "batch": batch,
    }
    _save_policy(bundle)
    return bundle


@pytest.fixture
def bundle(tmp_path: Path) -> dict:
    return _bundle(tmp_path)


def _inspect(bundle: dict) -> dict:
    return inspect_report_similarity(bundle["reports"], bundle["root"])


def _frames(reports: dict[str, str], gold_value: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.DataFrame(
        [
            {
                "StudyInstanceUID": uid,
                "Report": report,
                **{target: gold_value if uid == "d" else np.nan for target in TARGETS},
            }
            for uid, report in reports.items()
        ]
    )
    reviews = pd.DataFrame(
        [
            {
                "study_uid": uid,
                **{f"{target}__state": "negative" for target in TARGETS},
                **{f"{target}__evidence": "Finding." for target in TARGETS},
            }
            for uid in reports
        ]
    )
    return train, reviews


def test_valid_edge_preserves_original_reports_and_exact_report_groups(bundle: dict) -> None:
    before = copy.deepcopy(bundle["reports"])
    result = _inspect(bundle)
    assert result["edge_count"] == result["policy_count"] == 1
    assert similarity_pairs(result) == [("a", "b")]
    assert result["decisions"][0]["assessor_ids"] == {
        "root": "agent-root",
        "independent": "agent-independent",
    }
    assert bundle["reports"] == before
    assert report_group(before["a"]) != report_group(before["b"])


def test_explicit_empty_registry_is_valid_only_without_preserved_policies(tmp_path: Path) -> None:
    _write(tmp_path / REGISTRY_PATH, {"schema_version": 1, "policies": []})
    assert inspect_report_similarity({"a": "Finding."}, tmp_path)["edge_count"] == 0
    _write(
        tmp_path / "initiative/audits/report-similarity-dispositions-v2.json",
        {"schema_version": 1, "records": []},
    )
    with pytest.raises(ValueError, match="every policy"):
        inspect_report_similarity({"a": "Finding."}, tmp_path)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "omitted",
        "duplicate",
        "changed_hash",
        "unregistered_policy",
        "duplicate_json_key",
    ],
)
def test_required_registry_cannot_silently_drop_or_substitute_edges(
    bundle: dict, mutation: str
) -> None:
    path = bundle["root"] / REGISTRY_PATH
    obj = json.loads(path.read_text())
    if mutation == "missing":
        path.unlink()
    elif mutation == "omitted":
        obj["policies"] = []
        _write(path, obj)
    elif mutation == "duplicate":
        obj["policies"] *= 2
        _write(path, obj)
    elif mutation == "changed_hash":
        obj["policies"][0]["sha256"] = "0" * 64
        _write(path, obj)
    elif mutation == "unregistered_policy":
        _write(
            bundle["paths"]["policy"].with_name("report-similarity-dispositions-v2.json"),
            bundle["policy"],
        )
    else:
        path.write_text('{"schema_version":1,"policies":[],"policies":[]}')
    with pytest.raises(ValueError):
        _inspect(bundle)


@pytest.mark.parametrize(
    "left,right",
    [
        ("Finding. Impression.", "Finding. . Impression."),
        ("Finding. . Impression.", "Finding. Impression."),
        ("Résumé. . Impression.", "Résumé. Impression."),
    ],
)
def test_only_explicit_single_local_redundant_stop_is_accepted(
    tmp_path: Path, left: str, right: str
) -> None:
    assert _inspect(_bundle(tmp_path, left, right))["edge_count"] == 1


@pytest.mark.parametrize(
    "left,right",
    [
        ("Size 1.5 mm.", "Size 15 mm."),
        ("Finding. . Not present.", "Finding. Present."),
        ("Finding. . Impression. . End.", "Finding. Impression. End."),
        ("Finding Impression.", "Finding. Impression."),
        ("Finding.  Impression.", "Finding. Impression."),
        ("Finding. .. Impression.", "Finding. Impression."),
        ("Finding. . Impression.", "finding. Impression."),
        ("Finding. . Impression.", "Finding. Impression. Extra."),
    ],
)
def test_similarity_is_not_a_fuzzy_or_global_punctuation_matcher(
    tmp_path: Path, left: str, right: str
) -> None:
    with pytest.raises(ValueError, match="beyond one redundant"):
        _inspect(_bundle(tmp_path, left, right))


@pytest.mark.parametrize(
    "mutation",
    [
        "unknown",
        "self",
        "duplicate",
        "reverse_duplicate",
        "wrong_report_hash",
        "unsupported_relation",
    ],
)
def test_edge_identity_and_relation_are_fail_closed(bundle: dict, mutation: str) -> None:
    edge = bundle["policy"]["records"][0]
    if mutation == "unknown":
        edge["right"]["study_uid"] = "test-only"
    elif mutation == "self":
        edge["right"] = copy.deepcopy(edge["left"])
    elif mutation == "duplicate":
        bundle["policy"]["records"].append(copy.deepcopy(edge))
    elif mutation == "reverse_duplicate":
        extra = copy.deepcopy(edge)
        extra["left"], extra["right"] = extra["right"], extra["left"]
        bundle["policy"]["records"].append(extra)
    elif mutation == "wrong_report_hash":
        edge["left"]["source_report_sha256"] = "0" * 64
    else:
        edge["relation"] = "edit_distance_small"
    _save_policy(bundle)
    with pytest.raises(ValueError):
        _inspect(bundle)


@pytest.mark.parametrize(
    "mutation",
    ["alter_text", "duplicate_uid", "extra_gold_column", "noncanonical_path", "unrelated_uid"],
)
def test_blinded_source_requires_exact_text_membership_and_canonical_path(
    bundle: dict, mutation: str
) -> None:
    batch = bundle["batch"]
    if mutation == "alter_text":
        batch[0]["report"] = "Different report."
    elif mutation == "duplicate_uid":
        batch.append(copy.deepcopy(batch[0]))
    elif mutation == "extra_gold_column":
        batch[0]["ACL"] = 1
    elif mutation == "unrelated_uid":
        batch.append({"study_uid": "unknown", "report": "Unrelated."})
    old = bundle["paths"]["batch"]
    new = bundle["root"] / "evidence/source-copy.json" if mutation == "noncanonical_path" else old
    _write(new, batch)
    for side in ("left", "right"):
        bundle["policy"]["records"][0][side]["source_batch"] = _ref(bundle["root"], new)
    _save_policy(bundle)
    with pytest.raises(ValueError):
        _inspect(bundle)


@pytest.mark.parametrize("role", ["root", "independent"])
@pytest.mark.parametrize(
    "mutation",
    [
        "missing_endpoint",
        "duplicate_endpoint",
        "wrong_text",
        "wrong_hash",
        "wrong_source_path",
        "wrong_source_hash",
        "wrong_stage",
    ],
)
def test_each_independent_assessor_must_bind_both_complete_originals(
    bundle: dict, role: str, mutation: str
) -> None:
    doc = bundle["docs"][role]
    row = doc["records"][0]
    field = "source_batch" if role == "root" else "input_batch"
    if mutation == "missing_endpoint":
        doc["records"].pop()
    elif mutation == "duplicate_endpoint":
        doc["records"].append(copy.deepcopy(row))
    elif mutation == "wrong_text":
        row["report"] = "Finding. Different statement."
    elif mutation == "wrong_hash":
        row["source_report_sha256"] = "0" * 64
    elif mutation == "wrong_source_path":
        row[field]["path"] = "initiative/reviews/inputs/batch-999.json"
    elif mutation == "wrong_source_hash":
        row[field]["sha256"] = "0" * 64
    else:
        doc["stage"] = "posthoc_copy_of_other_assessment"
    _rebind_evidence(bundle, role)
    with pytest.raises(ValueError):
        _inspect(bundle)


@pytest.mark.parametrize("identity", ["agent-root", "agent-root ", " agent-root", "\tagent-root\n"])
def test_assessor_identity_cannot_be_copied_with_whitespace_alias(
    bundle: dict, identity: str
) -> None:
    bundle["docs"]["independent"]["agent_id"] = identity
    bundle["docs"]["joint"]["agent_id"] = identity
    _rebind_evidence(bundle, "independent")
    with pytest.raises(ValueError, match="independent"):
        _inspect(bundle)


@pytest.mark.parametrize(
    "key,field,value",
    [
        ("proof", "study_uids", ["a", "c"]),
        ("proof", "source_report_sha256", ["0" * 64, "1" * 64]),
        ("proof", "root_source_freeze_sha256", "0" * 64),
        ("proof", "independent_source_freeze_sha256", "0" * 64),
        ("proof", "joint_source_comparison_sha256", "0" * 64),
        ("proof", "assessor", "unrelated-agent"),
        ("joint", "root_source_freeze", {"path": "wrong.json", "sha256": "0" * 64}),
        ("joint", "own_source_freeze", {"path": "wrong.json", "sha256": "0" * 64}),
    ],
)
def test_rehashed_proof_must_still_bind_exact_pair_and_assessments(
    bundle: dict, key: str, field: str, value: object
) -> None:
    bundle["docs"][key][field] = value
    _rebind_evidence(bundle, key)
    with pytest.raises(ValueError, match="proof"):
        _inspect(bundle)


@pytest.mark.parametrize(
    "mutation", ["missing", "unrelated_uid", "wrong_hash", "duplicate", "disagreement"]
)
def test_joint_comparison_cannot_be_rebound_without_agreement_on_both_pair_members(
    bundle: dict, mutation: str
) -> None:
    rows = bundle["docs"]["joint"]["records"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "unrelated_uid":
        rows[0]["study_uid"] = "c"
    elif mutation == "wrong_hash":
        rows[0]["source_report_sha256"] = "0" * 64
    elif mutation == "duplicate":
        rows.append(copy.deepcopy(rows[0]))
    else:
        rows[0]["material_structural_disagreement"] = True
    _rebind_evidence(bundle, "joint")
    with pytest.raises(ValueError):
        _inspect(bundle)


def test_similarity_exact_report_and_pixel_edges_form_one_gold_isolated_component(
    bundle: dict,
) -> None:
    audit = _inspect(bundle)
    train, reviews = _frames(bundle["reports"])
    before_train, before_reviews = train.copy(deep=True), reviews.copy(deep=True)
    no_similarity = build_studies(train, reviews, [("c", "d")]).set_index("study_uid")
    assert no_similarity.loc["a", "split"] != "quarantine"
    result = build_studies(train, reviews, [("c", "d"), *similarity_pairs(audit)])
    indexed = result.set_index("study_uid")
    assert indexed.loc[list("abcd"), "leakage_group"].nunique() == 1
    assert indexed.loc[list("abc"), "split"].eq("quarantine").all()
    assert indexed.loc["d", "split"] == "gold_evaluation"
    assert indexed.loc["e", "leakage_group"] != indexed.loc["a", "leakage_group"]
    assert indexed.loc["a", "report_group"] != indexed.loc["b", "report_group"]
    assert indexed.loc["d", list(TARGETS)].eq(1).all()
    assert all(indexed.loc["d", f"{target}__provenance"] == "official_gold" for target in TARGETS)
    # A report-loss exclusion must not remove a graph bridge after grouping.
    result["report_supervision_allowed"] = True
    result.loc[result.study_uid.eq("b"), "report_supervision_allowed"] = False
    for target in TARGETS:
        result.loc[result.study_uid.eq("b"), f"{target}__mask"] = 0.0
    trace = bind_similarity_splits(audit, result)
    assert trace["edges_enforced"] is True
    assert trace["decisions"][0]["component_splits"] == [
        {"study_uid": uid, "split": "gold_evaluation" if uid == "d" else "quarantine"}
        for uid in "abcd"
    ]
    pd.testing.assert_frame_equal(train, before_train)
    pd.testing.assert_frame_equal(reviews, before_reviews)
    for target in TARGETS:
        assert indexed[f"{target}__state"].eq("negative").all()
        assert indexed[f"{target}__evidence"].eq("Finding.").all()


def test_group_membership_and_splits_are_order_and_gold_value_independent(bundle: dict) -> None:
    audit = _inspect(bundle)
    edges = [("c", "d"), *similarity_pairs(audit)]
    train, reviews = _frames(bundle["reports"])
    first = build_studies(train, reviews, edges)
    train0, reviews0 = _frames(bundle["reports"], 0)
    second = build_studies(
        train0.sample(frac=1, random_state=7),
        reviews0.sample(frac=1, random_state=9),
        [(right, left) for left, right in reversed(edges)],
    )
    pd.testing.assert_frame_equal(
        first[["study_uid", "leakage_group", "report_group", "split"]],
        second[["study_uid", "leakage_group", "report_group", "split"]],
    )
    assert _inspect(bundle)["decisions"] == audit["decisions"]
    train0.loc[:, list(TARGETS)] = np.nan
    nongold = build_studies(train0, reviews0, edges).set_index("study_uid")
    assert nongold.loc[list("abcd"), "leakage_group"].nunique() == 1
    assert nongold.loc[list("abcd"), "split"].nunique() == 1
    assert nongold.loc[list("abcd"), "split"].isin(["train", "weak_validation"]).all()


def test_binding_detects_required_edge_not_enforced(bundle: dict) -> None:
    audit = _inspect(bundle)
    train, reviews = _frames(bundle["reports"])
    studies = build_studies(train, reviews, [("c", "d")])
    with pytest.raises(ValueError, match="not enforced"):
        bind_similarity_splits(audit, studies)


def _mock_cli_inputs(monkeypatch: pytest.MonkeyPatch, bundle: dict, output: Path) -> None:
    """Keep unrelated release gates synthetic while exercising actual similarity/union/export."""
    from rsna import adjudications, correspondence

    train, reviews = _frames(bundle["reports"])
    train_path = bundle["root"] / "train.csv"
    train.to_csv(train_path, index=False)
    inventory_path = bundle["root"] / "inventory.parquet"
    pd.DataFrame({"synthetic": [1]}).to_parquet(inventory_path)
    audit_path = bundle["root"] / "audit.json"
    _write(audit_path, {"synthetic": True})
    monkeypatch.setattr(labels, "read_reviews", lambda *args: reviews.copy(deep=True))
    monkeypatch.setattr(labels, "validate_review_provenance", lambda *args: {})
    monkeypatch.setattr(labels, "validate_dicom_audit", lambda *args: [("c", "d")])
    monkeypatch.setattr(
        adjudications,
        "apply_adjudications",
        lambda frame, *args: (frame, {"registry_sha256": "synthetic"}),
    )
    monkeypatch.setattr(
        correspondence,
        "apply_correspondence",
        lambda frame, *args: (frame, {"registry_sha256": "synthetic"}),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "labels",
            "--train",
            str(train_path),
            "--reviews",
            "unused",
            "--drafts",
            "unused",
            "--output",
            str(output),
            "--dicom-audit",
            str(audit_path),
            "--inventory",
            str(inventory_path),
            "--source-revision",
            "a" * 40,
            "--review-provenance",
            "unused",
            "--rubric",
            "unused",
            "--artifact-root",
            str(bundle["root"]),
        ],
    )


def test_cli_enforces_similarity_before_gold_quarantine_and_exports_bound_trace(
    bundle: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = bundle["root"] / "release"
    _mock_cli_inputs(monkeypatch, bundle, output)
    labels.main()
    studies = pd.read_parquet(output / "studies.parquet").set_index("study_uid")
    assert studies.loc[list("abc"), "split"].eq("quarantine").all()
    trace = json.loads((output / "report_similarity_trace.json").read_text())
    audit = json.loads((output / "label_audit.json").read_text())
    assert trace["edge_count"] == 1 and trace["edges_enforced"] is True
    assert audit["release_inputs"]["report_similarity_registry_sha256"] == trace["registry_sha256"]
    assert audit["report_similarity"] == trace


@pytest.mark.parametrize("mutation", ["missing_registry", "changed_proof", "omitted_policy"])
def test_cli_invalid_similarity_blocks_before_any_release_artifact(
    bundle: dict, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    output = bundle["root"] / "must-not-exist"
    _mock_cli_inputs(monkeypatch, bundle, output)
    if mutation == "missing_registry":
        (bundle["root"] / REGISTRY_PATH).unlink()
    elif mutation == "changed_proof":
        bundle["paths"]["proof"].write_text("{}")
    else:
        _write(bundle["root"] / REGISTRY_PATH, {"schema_version": 1, "policies": []})
    with pytest.raises(ValueError):
        labels.main()
    assert not output.exists()


@pytest.mark.parametrize(
    "mutation",
    [
        "missing_status",
        "pending_status",
        "rejected_status",
        "missing_order",
        "contrary_root",
        "contrary_independent",
        "missing_role",
    ],
)
def test_explicit_accepted_source_first_admission_is_required(bundle: dict, mutation: str) -> None:
    edge = bundle["policy"]["records"][0]
    if mutation == "missing_status":
        del edge["status"]
    elif mutation == "pending_status":
        edge["status"] = "pending"
    elif mutation == "rejected_status":
        edge["status"] = "rejected"
    elif mutation == "missing_order":
        del edge["source_assessment_order"]
    elif mutation == "contrary_root":
        edge["source_assessment_order"]["root"] = "source_read_after_other_assessor_findings"
    elif mutation == "contrary_independent":
        edge["source_assessment_order"]["independent"] = "source_read_after_other_assessor_findings"
    else:
        del edge["source_assessment_order"]["root"]
    _save_policy(bundle)
    with pytest.raises(ValueError):
        _inspect(bundle)
