import copy
import json
import sqlite3

import pytest

from scripts.kaggle.inference_variants import KIND, identity, read, save, validate


def record(policy="edge-repeat"):
    return {
        "schema_version": 1,
        "kind": KIND,
        "weights_sha256": "a" * 64,
        "source_sha256": "b" * 64,
        "parent_source_sha256": "c" * 64,
        "policy": policy,
        "variant_sha256": identity("a" * 64, "b" * 64, policy),
        "kernel_slug": "godlydonuts/rsna-series-" + policy + "-" + "b" * 12,
        "own_training": False,
        "independent_internal_validation": False,
        "state": "prepared",
    }


def ledger():
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE events(at TEXT,sha TEXT,event TEXT)")
    db.execute("CREATE TABLE entries(sha TEXT PRIMARY KEY,record TEXT)")
    db.execute(
        "INSERT INTO entries VALUES (?,?)",
        ("a" * 64, json.dumps({"state": "scored", "public_score": "0.949"})),
    )
    return db


def test_two_inference_variants_share_real_weights_without_overwriting_parent():
    db = ledger()
    for policy in ("within-series", "edge-repeat"):
        r = record(policy)
        save(db, r, previous_state=None, at="now")
        assert read(db, r["variant_sha256"]) == r
        with pytest.raises(ValueError, match="re-created"):
            save(db, r, previous_state=None, at="later")
    assert db.execute("SELECT COUNT(*) FROM inference_variants").fetchone()[0] == 2
    assert (
        json.loads(db.execute("SELECT record FROM entries").fetchone()[0])["public_score"]
        == "0.949"
    )


def test_uncertainty_and_terminal_states_cannot_automatically_be_retried():
    db = ledger()
    r = record()
    save(db, r, previous_state=None, at="t0")
    for old, new in (("prepared", "push_intent"), ("push_intent", "push_uncertain")):
        r["state"] = new
        save(db, r, previous_state=old, at="t1")
    r["state"] = "pushed"
    with pytest.raises(ValueError, match="transition"):
        save(db, r, previous_state="push_uncertain", at="t2")


@pytest.mark.parametrize(
    "field,value",
    [
        ("weights_sha256", "d" * 64),
        ("source_sha256", "e" * 64),
        ("own_training", True),
        ("independent_internal_validation", True),
        ("checkpoint_sha256", "a" * 64),
        ("variant_sha256", "a" * 64),
    ],
)
def test_no_false_weight_identity_or_claims(field, value):
    r = record()
    r[field] = value
    with pytest.raises(ValueError):
        validate(r)


def test_valid_public_and_code_submission_lifecycle_requires_actual_score_fields():
    db = ledger()
    r = record()
    save(db, r, previous_state=None, at="t0")
    states = ["prepared", "push_intent", "pushed", "public_accepted", "submit_intent", "submitted"]
    for before, after in zip(states, states[1:]):
        r["state"] = after
        save(db, r, previous_state=before, at="t1")
    bad = copy.deepcopy(r)
    bad["state"] = "scored"
    with pytest.raises(ValueError, match="Actual scored"):
        save(db, bad, previous_state="submitted", at="t2")
    r.update(state="scored", public_score="0.950", submission_id=123)
    save(db, r, previous_state="submitted", at="t2")
    assert read(db, r["variant_sha256"])["public_score"] == "0.950"
