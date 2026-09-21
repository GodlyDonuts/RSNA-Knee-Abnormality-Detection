"""Actual temporal evidence with synthetic target values; no official gold reads."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rsna import correspondence as m
from rsna.labels import file_sha256
from rsna.schema import TARGETS

ROOT = Path(__file__).resolve().parents[1]
POLICY = Path("initiative/audits/correspondence-release-dispositions-v5.json")


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


@pytest.fixture
def temporal(tmp_path):
    policy = read(ROOT / POLICY)
    record = policy["records"][0]
    proof = read(ROOT / record["verified_technical_proof"]["path"])
    references = [
        {"path": record["source_path"], "sha256": record["source_batch_sha256"]},
        *proof["evidence"].values(),
    ]
    for ref in references:
        src = ROOT / ref["path"]
        assert file_sha256(src) == ref["sha256"]
        dest = tmp_path / ref["path"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
    reports = {row["study_uid"]: row["report"] for row in read(tmp_path / record["source_path"])}
    return {
        "root": tmp_path,
        "policy": policy,
        "record": record,
        "proof": proof,
        "reports": reports,
    }


def validate(bundle):
    m._validate_temporal_proof(bundle["record"], bundle["proof"], bundle["root"])


def replace_hash(obj, old, new):
    if isinstance(obj, dict):
        return {key: replace_hash(value, old, new) for key, value in obj.items()}
    if isinstance(obj, list):
        return [replace_hash(value, old, new) for value in obj]
    return new if obj == old else obj


def mutate_and_rebind(bundle, role, mutate):
    """Rebind the evidence DAG, so inner semantic checks must reject the change."""
    proof = bundle["proof"]
    docs = {key: read(bundle["root"] / ref["path"]) for key, ref in proof["evidence"].items()}
    mutate(docs[role])
    for _ in range(30):
        changed = False
        for key, doc in list(docs.items()):
            ref = proof["evidence"][key]
            path = bundle["root"] / ref["path"]
            write(path, doc)
            new = file_sha256(path)
            old = ref["sha256"]
            if new != old:
                changed = True
                docs = {k: replace_hash(v, old, new) for k, v in docs.items()}
                proof = replace_hash(proof, old, new)
        if not changed:
            bundle["proof"] = proof
            return
    raise AssertionError("Evidence bindings did not converge")


def install_policy(bundle):
    root = bundle["root"]
    proof_path = root / bundle["record"]["verified_technical_proof"]["path"]
    write(proof_path, bundle["proof"])
    bundle["record"]["verified_technical_proof"]["sha256"] = file_sha256(proof_path)
    write(root / POLICY, bundle["policy"])
    write(
        root / m.REGISTRY_PATH,
        {
            "schema_version": 1,
            "target_order": list(TARGETS),
            "policies": [{"path": str(POLICY), "sha256": file_sha256(root / POLICY)}],
        },
    )


def test_live_registry_binds_all_4407_reports_and_seventeen_exclusions():
    reports = {
        row["study_uid"]: row["report"]
        for path in (ROOT / "initiative/reviews/inputs").glob("batch-*.json")
        for row in read(path)
    }
    audit = m.inspect_correspondence(reports, ROOT)
    assert len(reports) == 4407
    assert audit["policy_count"] == 5 and audit["study_count"] == 17
    assert [r["case"] for r in audit["decisions"] if r["policy"]["path"] == str(POLICY)] == [
        "064:46"
    ]


@pytest.mark.parametrize("split", ["train", "gold_evaluation"])
def test_temporal_mask_preserves_all_other_fields_and_gold(temporal, split):
    install_policy(temporal)
    uid = temporal["record"]["study_uid"]
    rows = [
        {
            "study_uid": key,
            "Report": report,
            "split": split if key == uid else "train",
            "group_id": "synthetic-" + key,
            **{t: 0.37 for t in TARGETS},
            **{t + "__mask": 1.0 for t in TARGETS},
            **{t + "__state": "synthetic" for t in TARGETS},
            **{t + "__evidence": "synthetic evidence" for t in TARGETS},
            **{t + "__provenance": "synthetic input" for t in TARGETS},
        }
        for key, report in temporal["reports"].items()
    ]
    before = pd.DataFrame(rows)
    snapshot = before.copy(deep=True)
    after, audit = m.apply_correspondence(before, temporal["reports"], temporal["root"])
    pd.testing.assert_frame_equal(before, snapshot)
    masks = after.loc[after.study_uid.eq(uid), [t + "__mask" for t in TARGETS]]
    assert masks.eq(0 if split == "train" else 1).all().all()
    assert audit["report_masked_studies"] == ([uid] if split == "train" else [])
    assert audit["official_gold_preserved_studies"] == ([uid] if split == "gold_evaluation" else [])
    keep = [
        c
        for c in before
        if c not in {t + "__" + s for t in TARGETS for s in ("mask", "provenance")}
    ]
    pd.testing.assert_frame_equal(after[keep], before[keep])


@pytest.mark.parametrize(
    "mutation",
    ["wrong_index", "wrong_quote", "boolean_version", "wrong_selector", "missing_correction"],
)
def test_wrapper_shortcuts_rejected(temporal, mutation):
    proof = temporal["proof"]
    if mutation == "wrong_index":
        proof["source_index"] = 14
    elif mutation == "wrong_quote":
        proof["source_evidence"] = ["Invented temporal statement", "Another invented statement"]
    elif mutation == "boolean_version":
        proof["schema_version"] = True
    elif mutation == "wrong_selector":
        proof["independent_source_selector"] = "/cases/0"
    else:
        proof["evidence"].pop("root_lineage_correction")
    with pytest.raises(ValueError, match="Temporal"):
        validate(temporal)


@pytest.mark.parametrize(
    "role,field,value",
    [
        ("root_acceptance", "action", None),
        ("root_acceptance", "action", "verified"),
        ("root_acceptance", "status", "pending"),
        ("independent_acceptance", "action", None),
        ("independent_acceptance", "required_report_loss_mask", [0] * 12),
        ("independent_policy", "agent_id", "/root"),
        ("independent_visual", "all_12_matrices_viewed", False),
        ("independent_visual", "created_at_utc", "2000-01-01T00:00:00+00:00"),
        ("independent_visual", "created_at_utc", "2026-09-21T19:00:00"),
        ("root_source", "at_utc", "2099-01-01T00:00:00+00:00"),
        ("root_lineage_correction", "corrected_sha256", "0" * 64),
        ("technical_metadata", "study_uid", "wrong-study"),
    ],
)
def test_rebound_evidence_mutations_rejected(temporal, role, field, value):
    mutate_and_rebind(temporal, role, lambda obj: obj.__setitem__(field, value))
    with pytest.raises(ValueError, match="Temporal|assessor"):
        validate(temporal)


def test_changed_full_source_rejected_after_all_outer_hashes_rebound(temporal):
    def mutate(obj):
        obj["cases"][1]["report"] += " Fabricated addendum."

    mutate_and_rebind(temporal, "independent_source", mutate)
    with pytest.raises(ValueError, match="full source freeze"):
        validate(temporal)


def test_policy_cannot_downgrade_to_status_only_proof(temporal):
    temporal["proof"]["stage"] = "bare_verified_assertion"
    install_policy(temporal)
    with pytest.raises(ValueError, match="temporal source binding"):
        m.inspect_correspondence(temporal["reports"], temporal["root"])
