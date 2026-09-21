"""Exact-action correspondence regressions using original sources and synthetic labels."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import pytest

from rsna import correspondence as m
from rsna.labels import file_sha256, source_reports_fingerprint
from rsna.schema import TARGETS

ROOT = Path(os.environ.get("RSNA_CORRESPONDENCE_BASE", str(Path(__file__).resolve().parents[1])))
OVERLAY = Path(os.environ.get("RSNA_CORRESPONDENCE_OVERLAY", str(ROOT)))
POLICY = Path("initiative/audits/correspondence-release-dispositions-v4.json")
CASES = {"048:09", "048:38", "053:47"}


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


@pytest.fixture
def bundle(tmp_path):
    overlay = OVERLAY

    def copy(ref):
        rel = Path(ref["path"])
        src = overlay / rel if (overlay / rel).exists() else ROOT / rel
        assert file_sha256(src) == ref["sha256"]
        dest = tmp_path / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        return read(dest)

    # Preserve this historical v4 regression; live registry coverage is tested
    # by the temporal-policy tests as subsequent policies are added.
    registry = read(
        ROOT / "initiative/audits/correspondence-extension-v4/overlay" / m.REGISTRY_PATH
    )
    write(tmp_path / m.REGISTRY_PATH, registry)
    for ref in registry["policies"]:
        policy = copy(ref)
        for record in policy["records"]:
            copy({"path": record["source_path"], "sha256": record["source_batch_sha256"]})
            proof = copy(record["verified_technical_proof"])
            for key in (
                "verified_review",
                "source_handoff",
                "root_decision",
                "independent_assessment",
                "root_visual_assessment",
                "root_source_assessment",
                "independent_source_assessment",
            ):
                if key in proof:
                    copy(proof[key])
    reports = {}
    for path in sorted((ROOT / "initiative/reviews/inputs").glob("batch-*.json")):
        for row in read(path):
            assert row["study_uid"] not in reports
            reports[row["study_uid"]] = row["report"]
    return {"root": tmp_path, "reports": reports, "policy": read(tmp_path / POLICY)}


def record(bundle, case="048:09"):
    return next(r for r in bundle["policy"]["records"] if r["case"] == case)


def proof(bundle, case="048:09"):
    return read(bundle["root"] / record(bundle, case)["verified_technical_proof"]["path"])


def evidence(bundle, p, key):
    path = bundle["root"] / p[key]["path"]
    return path, read(path)


def changed(path, obj, p, key):
    write(path, obj)
    p[key]["sha256"] = file_sha256(path)


def validate(bundle, p, case="048:09"):
    m._validate_multi_assessor_proof(record(bundle, case), p, bundle["root"])


def test_all_4407_sources_sixteen_cases_and_exact_actions(bundle):
    assert len(bundle["reports"]) == 4407
    assert (
        source_reports_fingerprint(bundle["reports"])
        == "5c18838a09e9eec6522966555694fca83a9bae67e4d7c5c68592207ee4b60ed3"
    )
    out = m.inspect_correspondence(bundle["reports"], bundle["root"])
    assert out["policy_count"] == 4 and out["study_count"] == 16
    assert {r["case"] for r in bundle["policy"]["records"]} == CASES
    assert "048:31" not in {r["case"] for r in out["decisions"]}
    for case in CASES:
        p = proof(bundle, case)
        root = read(bundle["root"] / p["root_decision"]["path"])
        selected = m._selected(root, p["root_selector"])
        assert "required_report_supervision_mask_all" not in selected
        assert selected["required_export_action"] == m._CACHE_EXCLUSION_ACTION
        assert p["root_disposition"] == {
            "pointer": "/required_export_action",
            "value": m._CACHE_EXCLUSION_ACTION,
        }


def test_synthetic_masks_preserve_sources_states_labels_groups_and_gold(bundle):
    gold = record(bundle, "048:38")["study_uid"]
    rows = []
    for uid, report in bundle["reports"].items():
        rows.append(
            {
                "study_uid": uid,
                "Report": report,
                "split": "gold_evaluation" if uid == gold else "train",
                "group_id": "synthetic-" + uid,
                **{t: 0.37 for t in TARGETS},
                **{t + "__mask": 1.0 for t in TARGETS},
                **{t + "__state": "synthetic" for t in TARGETS},
                **{t + "__evidence": "synthetic exact evidence" for t in TARGETS},
                **{t + "__provenance": "synthetic input" for t in TARGETS},
            }
        )
    before = pd.DataFrame(rows)
    snapshot = before.copy(deep=True)
    after, audit = m.apply_correspondence(before, bundle["reports"], bundle["root"])
    pd.testing.assert_frame_equal(before, snapshot)
    assert len(audit["report_masked_studies"]) == 15 and audit[
        "official_gold_preserved_studies"
    ] == [gold]
    masked = after.study_uid.isin(audit["report_masked_studies"])
    assert after.loc[masked, [t + "__mask" for t in TARGETS]].eq(0).all().all()
    assert after.loc[~masked, [t + "__mask" for t in TARGETS]].eq(1).all().all()
    keep = [
        c
        for c in before
        if c not in {t + "__" + suffix for t in TARGETS for suffix in ["mask", "provenance"]}
    ]
    pd.testing.assert_frame_equal(after[keep], before[keep])


def test_action_policy_rejects_status_only_wrapper_after_outer_hash_rebinding(bundle):
    p = proof(bundle)
    p["stage"] = "bare_verified_assertion"
    path = bundle["root"] / record(bundle)["verified_technical_proof"]["path"]
    write(path, p)
    record(bundle)["verified_technical_proof"]["sha256"] = file_sha256(path)
    write(bundle["root"] / POLICY, bundle["policy"])
    registry_path = bundle["root"] / m.REGISTRY_PATH
    registry = read(registry_path)
    for reference in registry["policies"]:
        if reference["path"] == str(POLICY):
            reference["sha256"] = file_sha256(bundle["root"] / POLICY)
    write(registry_path, registry)
    with pytest.raises(ValueError, match="multi-assessor source binding"):
        m.inspect_correspondence(bundle["reports"], bundle["root"])


@pytest.mark.parametrize(
    "action",
    [
        None,
        False,
        True,
        "verified",
        "retain_source_and_series_no_exclusion_from_this_trigger_alone",
        "exclude_whole_study_report_supervision",
        m._CACHE_EXCLUSION_ACTION + " ",
    ],
)
def test_only_exact_original_action_is_accepted_after_rebinding(bundle, action):
    p = proof(bundle)
    path, obj = evidence(bundle, p, "root_decision")
    m._selected(obj, p["root_selector"])["required_export_action"] = action
    p["root_disposition"]["value"] = action
    changed(path, obj, p, "root_decision")
    with pytest.raises(ValueError, match="explicit bound whole-study exclusion"):
        validate(bundle, p)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_selector",
        "boolean_invented",
        "stage_only",
        "wrong_kind",
        "source_quote",
        "source_index",
        "independent_stage",
        "identity_claim",
        "whitespace_assessor",
    ],
)
def test_proof_shortcuts_fail(bundle, mutation):
    p = proof(bundle)
    if mutation == "wrong_selector":
        p["root_selector"] = "/cases/1"
    elif mutation == "boolean_invented":
        p["root_disposition"] = {
            "pointer": "/required_report_supervision_mask_all",
            "value": True,
        }
    elif mutation == "stage_only":
        p["root_disposition"] = {
            "pointer": "/study_uid",
            "value": record(bundle)["study_uid"],
        }
    elif mutation == "wrong_kind":
        p["proof_kind"] = "reviewed_cache_aggregate"
    elif mutation == "source_quote":
        p["source_evidence"] = ["invented source"]
    elif mutation == "source_index":
        p["source_index"] = 38
    elif mutation == "identity_claim":
        p["independent_claim"] = {
            "pointer": "/agent_id",
            "value": "/root/image_pipeline_audit",
        }
    else:
        path, obj = evidence(bundle, p, "independent_assessment")
        obj["stage" if mutation == "independent_stage" else "agent_id"] = (
            "verified" if mutation == "independent_stage" else " /root/image_pipeline_audit"
        )
        changed(path, obj, p, "independent_assessment")
    with pytest.raises(ValueError):
        validate(bundle, p)


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_visual_case",
        "wrong_visual_agent",
        "empty_visual",
        "wrong_visual_hash",
        "source_text",
        "source_sha",
        "source_case",
        "source_batch",
        "source_selector",
        "source_agent",
        "source_quote",
        "source_claim",
        "source_late",
        "source_time_naive",
        "visual_late",
        "lineage_missing",
        "lineage_wrong_hash",
        "handoff_original_visual",
    ],
)
def test_original_source_visual_lineage_is_enforced_after_hash_rebinding(bundle, mutation):
    p = proof(bundle)
    key = (
        "root_visual_assessment"
        if mutation.startswith("wrong_visual") or mutation == "empty_visual"
        else "root_source_assessment"
    )
    if mutation in {"visual_late"}:
        key = "root_visual_assessment"
    if mutation.startswith("lineage") or mutation == "handoff_original_visual":
        key = "source_handoff"
    path, obj = evidence(bundle, p, key)
    if key == "root_visual_assessment":
        if mutation == "wrong_visual_case":
            obj["case"] = "048:31"
        elif mutation == "wrong_visual_agent":
            obj["agent_id"] = "/root/image_pipeline_audit"
        elif mutation == "empty_visual":
            obj["observations"] = []
        elif mutation == "visual_late":
            obj["created_at_utc"] = "2099-01-01T00:00:00+00:00"
        else:
            obj["extra_change"] = True
    elif key == "source_handoff":
        if mutation == "lineage_missing":
            obj["source_review_lineage"]["source_assessments"] = []
        elif mutation == "lineage_wrong_hash":
            obj["source_review_lineage"]["source_assessments"][0]["sha256"] = "0" * 64
        else:
            obj["original_root_visual_assessment"]["sha256"] = "0" * 64
    else:
        row = m._selected(obj, p["root_source_selector"])
        if mutation == "source_text":
            row["report"] += " changed"
        elif mutation == "source_sha":
            row["source_report_sha256"] = "0" * 64
        elif mutation == "source_case":
            row["case"] = "048:31"
        elif mutation == "source_batch":
            row["source_batch"]["sha256"] = "0" * 64
        elif mutation == "source_selector":
            p["root_source_selector"] = "/records/0"
        elif mutation == "source_agent":
            obj["agent_id"] = "/root/image_pipeline_audit"
        elif mutation == "source_quote":
            row["exact_source_quotes"] = ["invented quote"]
        elif mutation == "source_claim":
            row["root_observations"] = ""
        elif mutation == "source_late":
            obj["created_at_utc"] = "2099-01-01T00:00:00+00:00"
        else:
            obj["created_at_utc"] = "2026-09-21T13:01:20"
    changed(path, obj, p, key)
    # Rebind upstream hash attestations to reach semantic validation where applicable.
    root_path, root = evidence(bundle, p, "root_decision")
    selected = m._selected(root, p["root_selector"])
    if key == "root_visual_assessment" and mutation != "wrong_visual_hash":
        selected["frozen_root_visual"]["sha256"] = p[key]["sha256"]
        hp, h = evidence(bundle, p, "source_handoff")
        h["original_root_visual_assessment"]["sha256"] = p[key]["sha256"]
        changed(hp, h, p, "source_handoff")
        selected["joint_handoff"]["sha256"] = p["source_handoff"]["sha256"]
    if key == "source_handoff":
        selected["joint_handoff"]["sha256"] = p[key]["sha256"]
    if key == "root_source_assessment":
        for parentkey in ["source_handoff", "independent_assessment"]:
            pp, po = evidence(bundle, p, parentkey)
            for item in po["source_review_lineage"]["source_assessments"]:
                if item["path"].endswith("/root-source-preread.json"):
                    item["sha256"] = p[key]["sha256"]
            changed(pp, po, p, parentkey)
        selected["joint_handoff"]["sha256"] = p["source_handoff"]["sha256"]
        selected["agent_visual"]["sha256"] = p["independent_assessment"]["sha256"]
        hp, h = evidence(bundle, p, "source_handoff")
        h["original_independent_visual_assessment"]["sha256"] = p["independent_assessment"][
            "sha256"
        ]
        changed(hp, h, p, "source_handoff")
        selected["joint_handoff"]["sha256"] = p["source_handoff"]["sha256"]
    changed(root_path, root, p, "root_decision")
    with pytest.raises(ValueError):
        validate(bundle, p)
