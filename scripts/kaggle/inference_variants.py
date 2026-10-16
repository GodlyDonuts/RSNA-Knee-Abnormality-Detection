"""Policy/source identities in the same canonical account ledger as checkpoints.

An inference adaptation must not masquerade as new checkpoint bytes or overwrite
the already-scored parent weight entry. All cloud operations still use the sole
account coordinator claim; this table adds explicit inference-variant provenance.
"""

from __future__ import annotations

import hashlib
import json

KIND = "licensed_released_coatnet_series_boundary_v1"
IMMUTABLE = (
    "schema_version",
    "kind",
    "variant_sha256",
    "weights_sha256",
    "source_sha256",
    "parent_source_sha256",
    "policy",
    "kernel_slug",
    "own_training",
    "independent_internal_validation",
)
TRANSITIONS = {
    "prepared": {"push_intent"},
    "push_intent": {"pushed", "push_uncertain"},
    "pushed": {"public_accepted", "public_failed"},
    "public_accepted": {"submit_intent"},
    "submit_intent": {"submitted", "submit_uncertain"},
    "submitted": {"scored", "failed"},
    "scored": set(),
    "failed": set(),
    "public_failed": set(),
    "push_uncertain": set(),
    "submit_uncertain": set(),
}


def is_sha(value):
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def identity(weights_sha256, source_sha256, policy):
    if (
        not is_sha(weights_sha256)
        or not is_sha(source_sha256)
        or policy not in {"within-series", "edge-repeat"}
    ):
        raise ValueError("Exact weight/source hashes and explicit boundary policy required")
    return hashlib.sha256(
        json.dumps(
            {
                "kind": KIND,
                "weights_sha256": weights_sha256,
                "source_sha256": source_sha256,
                "policy": policy,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def validate(record):
    if (
        record.get("schema_version") != 1
        or record.get("kind") != KIND
        or record.get("state") not in TRANSITIONS
        or record.get("own_training") is not False
        or record.get("independent_internal_validation") is not False
        or not is_sha(record.get("parent_source_sha256"))
        or record.get("variant_sha256")
        != identity(record.get("weights_sha256"), record.get("source_sha256"), record.get("policy"))
        or record.get("kernel_slug")
        != "godlydonuts/rsna-series-" + record["policy"] + "-" + record["source_sha256"][:12]
        or "checkpoint_sha256" in record
    ):
        raise ValueError("Distinct inference identity and accurate parent provenance required")
    if record["state"] == "scored" and (
        not isinstance(record.get("public_score"), str) or record.get("submission_id", 0) <= 0
    ):
        raise ValueError("Actual scored submission identity required")


def read(db, variant_sha256):
    row = db.execute(
        "SELECT record FROM inference_variants WHERE variant_sha=?", (variant_sha256,)
    ).fetchone()
    if row is None:
        raise ValueError("Unknown inference variant; reconcile original operation")
    record = json.loads(row[0])
    validate(record)
    return record


def save(db, record, *, previous_state, at):
    validate(record)
    db.execute(
        "CREATE TABLE IF NOT EXISTS inference_variants (variant_sha TEXT PRIMARY KEY, record TEXT NOT NULL)"
    )
    current = db.execute(
        "SELECT record FROM inference_variants WHERE variant_sha=?", (record["variant_sha256"],)
    ).fetchone()
    if previous_state is None:
        if current is not None or record["state"] != "prepared":
            raise ValueError("Existing inference variant must not be re-created")
        db.execute(
            "INSERT INTO inference_variants(variant_sha,record) VALUES (?,?)",
            (record["variant_sha256"], json.dumps(record, sort_keys=True)),
        )
    else:
        if current is None:
            raise ValueError("Missing original variant")
        old = json.loads(current[0])
        validate(old)
        if (
            old["state"] != previous_state
            or any(old[k] != record[k] for k in IMMUTABLE)
            or record["state"] not in TRANSITIONS[previous_state]
        ):
            raise ValueError("Invalid transition or changed immutable inference identity")
        db.execute(
            "UPDATE inference_variants SET record=? WHERE variant_sha=?",
            (json.dumps(record, sort_keys=True), record["variant_sha256"]),
        )
    db.execute(
        "INSERT INTO events(at,sha,event) VALUES (?,?,?)",
        (
            at,
            record["variant_sha256"],
            json.dumps(
                {
                    "kind": KIND,
                    "state": record["state"],
                    "parent_weights_sha256": record["weights_sha256"],
                    "new_checkpoint_bytes": False,
                }
            ),
        ),
    )
    db.commit()
