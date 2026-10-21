"""Small original Colab completion evidence; never MRI or model/checkpoint bytes."""

import base64
import hashlib
import json
import zlib
from pathlib import Path

SCHEMA = "actual_Colab_C100_original_final_handoff_v1"
MARKER = "RSNA_COLAB_FINAL_HANDOFF_ZLIB_V1 "
FILENAME = "rsna-colab-C100-final-evidence-bce57e766fe4.json"
CAP = 2 * 2**20
KEYS = {
    "binding_text",
    "terminal_text",
    "epoch_publications",
    "terminal_publication",
    "own_source_text",
}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_small(path):
    p = Path(path)
    require(
        p.is_file() and not p.is_symlink() and 0 <= p.stat().st_size <= CAP,
        "Original bounded regular evidence file required",
    )
    return p.read_text()


def validate(bundle, contract):
    require(
        set(bundle) == {"schema", "evidence", "evidence_sha256", "weights_or_pixels_included"}
        and bundle["schema"] == SCHEMA
        and bundle["weights_or_pixels_included"] is False,
        "Exact small original evidence envelope required",
    )
    evidence = bundle["evidence"]
    require(
        set(evidence) == KEYS
        and len(canonical(bundle)) <= CAP
        and sha(canonical(evidence)) == bundle["evidence_sha256"],
        "Evidence identity differs",
    )
    b, t = json.loads(evidence["binding_text"]), json.loads(evidence["terminal_text"])
    require(
        t["status"] == "actual_full_C100_seed20261021_complete"
        and t["binding"] == b
        and t["epochs"] == 6
        and t["updates"] == 2916
        and contract.terminal_storage_state(t)
        and t["clinical_or_Kaggle_score"] is None
        and b["source_git"] == contract.SCIENCE
        and b["branch"] == contract.BRANCH
        and b["input_manifest_sha256"] == contract.MANIFEST
        and b["recipe"]["seed"] == 20261021,
        "Only the actual complete independent Colab final",
    )
    own = json.loads(evidence["own_source_text"])
    require(
        set(own) == set(contract.MODULE_SHA256)
        and all(sha(own[n].encode()) == h for n, h in contract.MODULE_SHA256.items()),
        "Only exact own numerical serving source; no foreign files",
    )
    pubs = evidence["epoch_publications"]
    require(len(pubs) == 6, "Six original privately verified epochs required")
    continuation = contract.execution_continuation(t, b, pubs[0])
    contract.recovery_checkpoint_binding(continuation, pubs)
    require(continuation is not None, "Actually executed epoch001 recovery must be disclosed")
    names = {
        n + ".bin"
        for n in (
            "model.pt",
            "full-state.private.pt",
            "run-start.json",
            "source-permission.txt",
            "colab_train.py",
            "colab_storage.py",
            "kaggle_io.py",
        )
    }
    identities = []
    for i, p in enumerate(pubs, 1):
        extra = {"resume-control.py.bin", "continuation.json.bin"} if i >= 2 else set()
        contract.publication(p, f"epoch-{i:03d}", names | extra)
        contract.matches(p["files"]["run-start.json.bin"], evidence["binding_text"].encode())
        if i >= 2:
            executed = contract.epoch_continuation(continuation, i)
            require(
                p["files"]["resume-control.py.bin"]["sha256"] == executed["control_sha256"],
                "Original recovery source differs",
            )
            contract.matches(p["files"]["continuation.json.bin"], canonical(executed) + b"\n")
        identities.append(p["id"])
    terminal = evidence["terminal_publication"]
    contract.publication(terminal, "terminal", {"terminal.json.bin"})
    contract.matches(terminal["files"]["terminal.json.bin"], evidence["terminal_text"].encode())
    require(
        len(set(identities + [terminal["id"]])) == 7, "Distinct original private Datasets required"
    )
    return evidence


def collect(root, contract):
    root = Path(root)
    old = root / "run-C100-seed20261021"
    run = root / "run-C100-seed20261021-recovery-v1"
    second = root / "run-C100-seed20261021-recovery-v2"
    if second.exists():
        run = second
    first = json.loads(read_small(old / "publication-001/reconciled-original.json"))["publication"]
    evidence = dict(
        binding_text=read_small(old / "run-start.json"),
        terminal_text=read_small(run / "terminal.json"),
        epoch_publications=[first]
        + [
            json.loads(
                read_small(
                    run
                    / (
                        "publication-network-002/verified.json"
                        if i == 2 and run == second
                        else f"publication-{i:03d}/verified.json"
                    )
                )
            )
            for i in range(2, 7)
        ],
        terminal_publication=json.loads(read_small(run / "terminal-publication/verified.json")),
        own_source_text=json.dumps(
            {n: read_small(root / "code" / n) for n in contract.MODULE_SHA256}, sort_keys=True
        ),
    )
    bundle = dict(
        schema=SCHEMA,
        evidence=evidence,
        evidence_sha256=sha(canonical(evidence)),
        weights_or_pixels_included=False,
    )
    validate(bundle, contract)
    return bundle


def decode_output(encoded, contract):
    require(isinstance(encoded, str) and len(encoded) <= CAP, "Bounded compressed handoff required")
    compressed = base64.b64decode(encoded, validate=True)
    decoder = zlib.decompressobj()
    raw = decoder.decompress(compressed, CAP + 1)
    require(
        len(raw) <= CAP and decoder.eof and not decoder.unused_data and not decoder.unconsumed_tail,
        "Complete single bounded compressed envelope required",
    )
    bundle = json.loads(raw)
    validate(bundle, contract)
    return bundle


def from_saved_output(terminal, completion, own_source, contract):
    """Recover exact original serialized metadata from existing notebook output.

    The queued cell already prints all seven original publication proofs and
    the trainer prints its terminal. Every reconstructed byte string must match
    those original hashes; no new claim of checkpoint verification is made.
    """
    require(
        completion["phase"]
        in (
            "COLAB_FULL_RECOVERY_COMPLETE_PRIVATE_PRESERVED",
            "COLAB_FULL_RECOVERY2_COMPLETE_PRIVATE_PRESERVED",
        )
        and completion["updates"] == 2916
        and completion["epochs"] == 6
        and completion["seed"] == 20261021
        and completion["continuation"] == terminal["execution_continuation"],
        "Actual original queued completion output required",
    )
    proofs = completion["publication_proofs"]
    require(len(proofs) == 7, "Seven original printed publication proofs required")
    evidence = dict(
        binding_text=(canonical(terminal["binding"]) + b"\n").decode(),
        terminal_text=(canonical(terminal) + b"\n").decode(),
        epoch_publications=proofs[:6],
        terminal_publication=proofs[6],
        own_source_text=json.dumps(own_source, sort_keys=True),
    )
    bundle = dict(
        schema=SCHEMA,
        evidence=evidence,
        evidence_sha256=sha(canonical(evidence)),
        weights_or_pixels_included=False,
    )
    validate(bundle, contract)
    return bundle
