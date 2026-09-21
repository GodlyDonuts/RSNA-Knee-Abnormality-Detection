"""Preserved source/header evidence; synthetic targets and gold role only."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from rsna import correspondence as m
from rsna.labels import file_sha256
from rsna.schema import TARGETS

ROOT = Path(__file__).resolve().parents[1]
POLICY = Path("initiative/audits/correspondence-release-dispositions-v6.json")


def read(path):
    return json.loads(path.read_text())


def write(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


@pytest.fixture
def bundle(tmp_path):
    policy = read(ROOT / POLICY)
    record = policy["records"][0]
    proof = read(ROOT / record["verified_technical_proof"]["path"])
    refs = [
        {"path": record["source_path"], "sha256": record["source_batch_sha256"]},
        *proof["evidence"].values(),
    ]
    for ref in refs:
        source = ROOT / ref["path"]
        assert file_sha256(source) == ref["sha256"]
        destination = tmp_path / ref["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
    return {
        "root": tmp_path,
        "policy": policy,
        "record": record,
        "proof": proof,
        "reports": {r["study_uid"]: r["report"] for r in read(tmp_path / record["source_path"])},
    }


def validate(bundle):
    m._validate_source_header_proof(bundle["record"], bundle["proof"], bundle["root"])


def replace_hash(obj, old, new):
    if isinstance(obj, dict):
        return {k: replace_hash(v, old, new) for k, v in obj.items()}
    if isinstance(obj, list):
        return [replace_hash(v, old, new) for v in obj]
    return new if obj == old else obj


def mutate_and_rebind(bundle, role, mutate):
    """Rehash the evidence DAG so semantic checks, not only file hashes, are tested."""
    proof = bundle["proof"]
    docs = {
        k: read(bundle["root"] / ref["path"])
        for k, ref in proof["evidence"].items()
        if k != "root_replay_script"
    }
    mutate(docs[role])
    for _ in range(30):
        changed = False
        for key, document in list(docs.items()):
            ref = proof["evidence"][key]
            path = bundle["root"] / ref["path"]
            write(path, document)
            old, new = ref["sha256"], file_sha256(path)
            if old != new:
                changed = True
                docs = {k: replace_hash(v, old, new) for k, v in docs.items()}
                proof = replace_hash(proof, old, new)
        if not changed:
            bundle["proof"] = proof
            return
    raise AssertionError("Evidence hashes did not converge")


def install(bundle):
    root = bundle["root"]
    proof = root / bundle["record"]["verified_technical_proof"]["path"]
    write(proof, bundle["proof"])
    bundle["record"]["verified_technical_proof"]["sha256"] = file_sha256(proof)
    write(root / POLICY, bundle["policy"])
    write(
        root / m.REGISTRY_PATH,
        {
            "schema_version": 1,
            "target_order": list(TARGETS),
            "policies": [{"path": str(POLICY), "sha256": file_sha256(root / POLICY)}],
        },
    )


def test_actual_source_header_proof_and_prior_policies_unchanged(bundle):
    validate(bundle)
    before = read(ROOT / "initiative/audits/correspondence-source-header-v1/before-registry.json")
    current = read(ROOT / m.REGISTRY_PATH)
    assert current["policies"][:-1] == before["policies"]
    records = []
    for ref in before["policies"]:
        assert file_sha256(ROOT / ref["path"]) == ref["sha256"]
        records.extend(read(ROOT / ref["path"])["records"])
    assert len(before["policies"]) == 5 and len(records) == 17
    assert bundle["record"]["case"] == "069:10"
    assert bundle["policy"]["proof_binding_version"] == 4


@pytest.mark.parametrize("split", ["train", "weak_validation", "quarantine", "gold_evaluation"])
def test_only_non_gold_report_masks_and_provenance_change(bundle, split):
    install(bundle)
    uid = bundle["record"]["study_uid"]
    before = pd.DataFrame(
        [
            {
                "study_uid": key,
                "Report": report,
                "split": split if key == uid else "train",
                "report_group": "synthetic report group " + key,
                "leakage_group": "synthetic leakage group " + key,
                **{t: 0.37 for t in TARGETS},
                **{t + "__mask": 1.0 for t in TARGETS},
                **{t + "__state": "synthetic state" for t in TARGETS},
                **{t + "__evidence": "synthetic evidence" for t in TARGETS},
                **{t + "__provenance": "synthetic source" for t in TARGETS},
            }
            for key, report in bundle["reports"].items()
        ]
    )
    snapshot = before.copy(deep=True)
    after, audit = m.apply_correspondence(before, bundle["reports"], bundle["root"])
    pd.testing.assert_frame_equal(before, snapshot)
    selected = before.study_uid.eq(uid)
    mutable = [t + "__" + suffix for t in TARGETS for suffix in ("mask", "provenance")]
    pd.testing.assert_frame_equal(after.loc[~selected, before.columns], before.loc[~selected])
    keep = [c for c in before if c not in mutable]
    pd.testing.assert_frame_equal(after[keep], before[keep])
    masks = after.loc[selected, [t + "__mask" for t in TARGETS]]
    assert masks.eq(1 if split == "gold_evaluation" else 0).all().all()
    if split == "gold_evaluation":
        pd.testing.assert_frame_equal(after[before.columns], before)
        assert audit["official_gold_preserved_studies"] == [uid]
        assert audit["report_masked_studies"] == []
    else:
        assert audit["newly_masked_targets"] == dict.fromkeys(TARGETS, 1)
        assert audit["report_masked_studies"] == [uid]


@pytest.mark.parametrize(
    "mutation", ["index", "boolean_index", "quote", "selector", "role", "version"]
)
def test_wrapper_shortcuts_rejected(bundle, mutation):
    p = bundle["proof"]
    if mutation == "index":
        p["source_index"] = 11
    elif mutation == "boolean_index":
        p["source_index"] = True
    elif mutation == "quote":
        p["source_evidence"] = ["invented source quotation"]
    elif mutation == "selector":
        p["selectors"]["root_acceptance"] = "/cases/0"
    elif mutation == "role":
        p["evidence"].pop("root_replay_script")
    else:
        p["schema_version"] = True
    with pytest.raises(ValueError, match="Source/header"):
        validate(bundle)


@pytest.mark.parametrize(
    "role,pointer,value",
    [
        ("root_acceptance", "/cases/3/accepted", False),
        ("root_acceptance", "/cases/3/accepted_action", "retain"),
        ("root_proposal", "/cases/3/proposed_action", "retain"),
        ("independent_policy", "/cases/3/policy_suggestion", "retain"),
        ("independent_acceptance", "/action", "verified"),
        ("independent_acceptance", "/required_report_loss_mask", [0] * 12),
        ("independent_acceptance", "/source_index", 11),
        ("independent_source", "/agent_id", "/root"),
        ("independent_metadata", "/agent_id", "/root/last_twelve_visual_review/"),
        ("independent_source", "/cases/3/source_quotes", ["invented exact quote"]),
        ("source_packet", "/cases/3/full_report_text", "invented complete report"),
        ("independent_source_packet", "/cases/3/full_report_text", "invented complete report"),
        ("root_source", "/cases/2/source_index_zero_based", 11),
        ("root_source", "/prior_exposure", ""),
        ("root_metadata", "/status", "pending"),
        ("root_metadata", "/cases/3/series/0/literal_fields/Laterality", ["R"]),
        ("technical_metadata", "/series/0/n_slices", 999),
        ("root_acceptance", "/independent_source/sha256", "0" * 64),
        ("independent_policy", "/source_first_freeze/sha256", "0" * 64),
        ("root_source", "/at_utc", "2099-01-01T00:00:00+00:00"),
        ("independent_source_freeze", "/at_utc", "2026-01-01T00:00:00"),
        ("independent_policy", "/at_utc", "2000-01-01T00:00:00+00:00"),
        ("independent_acceptance", "/at_utc", "2000-01-01T00:00:00+00:00"),
    ],
)
def test_rebound_semantic_or_lineage_tampering_rejected(bundle, role, pointer, value):
    parent, _, key = pointer.rpartition("/")
    mutate_and_rebind(bundle, role, lambda obj: m._selected(obj, parent).__setitem__(key, value))
    with pytest.raises(ValueError, match="Source/header|assessor"):
        validate(bundle)


@pytest.mark.parametrize("stage", [m.TEMPORAL_STAGE, m.MULTI_ASSESSOR_STAGE, "bare_verified"])
def test_policy_requires_its_honest_source_header_format(bundle, stage):
    bundle["proof"]["stage"] = stage
    install(bundle)
    with pytest.raises(ValueError, match="source/header binding"):
        m.inspect_correspondence(bundle["reports"], bundle["root"])


def test_unbound_replay_script_edit_rejected(bundle):
    path = bundle["root"] / bundle["proof"]["evidence"]["root_replay_script"]["path"]
    path.write_bytes(path.read_bytes() + b"\n# changed\n")
    with pytest.raises(ValueError, match="checksum"):
        validate(bundle)
