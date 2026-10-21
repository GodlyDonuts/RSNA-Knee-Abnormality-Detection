import base64
import copy
import importlib.util
import json
import zlib
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


H = load("Colab_handoff_tests", HERE / "handoff.py")
OWNER = load("Colab_owner_tests", HERE / "owner.py")
F = load(
    "original_Colab_fixtures", HERE.parent / "colab-C100-final-serving-20261021-v1/test_serving.py"
)


@pytest.fixture
def bundle():
    _, e = F.case.__wrapped__()
    F.recovered(e)
    e.pop("license_bytes")
    return dict(
        schema=H.SCHEMA,
        evidence=e,
        evidence_sha256=H.sha(H.canonical(e)),
        weights_or_pixels_included=False,
    )


@pytest.mark.parametrize("second", [False, True])
def test_full_original_collection_and_compressed_roundtrip(bundle, tmp_path, second):
    e = bundle["evidence"]
    if second:
        F.recovered2(e)
    old = tmp_path / "run-C100-seed20261021"
    run = tmp_path / (
        "run-C100-seed20261021-recovery-v2" if second else "run-C100-seed20261021-recovery-v1"
    )

    def put(path, text):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    put(old / "run-start.json", e["binding_text"])
    put(
        old / "publication-001/reconciled-original.json",
        json.dumps({"publication": e["epoch_publications"][0]}),
    )
    put(run / "terminal.json", e["terminal_text"])
    put(run / "terminal-publication/verified.json", json.dumps(e["terminal_publication"]))
    for i in range(2, 7):
        name = (
            "publication-network-002/verified.json"
            if i == 2 and second
            else f"publication-{i:03d}/verified.json"
        )
        put(run / name, json.dumps(e["epoch_publications"][i - 1]))
    for name, body in json.loads(e["own_source_text"]).items():
        put(tmp_path / "code" / name, body)
    result = H.collect(tmp_path, F.C)
    assert H.validate(result, F.C) == result["evidence"]
    encoded = base64.b64encode(zlib.compress(H.canonical(result))).decode()
    assert H.decode_output(encoded, F.C) == result
    assert len(H.canonical(result)) < H.CAP
    assert not any("license_bytes" in k or "token" in k for k in result["evidence"])


@pytest.mark.parametrize(
    "fault",
    [
        "hash",
        "source",
        "not_final",
        "wrong_branch",
        "unverified",
        "recovery",
        "missing_epoch",
        "extra_file",
        "wrong_terminal",
        "duplicate_id",
    ],
)
def test_bad_completion_never_admitted(bundle, fault):
    e = bundle["evidence"]
    if fault == "hash":
        bundle["evidence_sha256"] = "0" * 64
    if fault == "source":
        own = json.loads(e["own_source_text"])
        own[next(iter(own))] += "\n# changed"
        e["own_source_text"] = json.dumps(own)
    if fault in ["not_final", "wrong_branch", "unverified", "recovery"]:
        t = json.loads(e["terminal_text"])
        if fault == "not_final":
            t["updates"] = 2800
        if fault == "wrong_branch":
            t["binding"]["branch"] = "Newton-C100"
        if fault == "unverified":
            t["all_epochs_private_verified"] = False
        if fault == "recovery":
            t["execution_continuation"]["resume_update"] = 489
        e["terminal_text"] = json.dumps(t)
    if fault == "missing_epoch":
        e["epoch_publications"].pop()
    if fault == "extra_file":
        e["epoch_publications"][2]["files"]["unexpected.bin"] = {"bytes": 1, "sha256": "0" * 64}
    if fault == "wrong_terminal":
        e["terminal_publication"]["files"]["terminal.json.bin"]["sha256"] = "0" * 64
    if fault == "duplicate_id":
        e["terminal_publication"]["id"] = e["epoch_publications"][2]["id"]
        e["terminal_publication"]["revision"] = e["epoch_publications"][2]["revision"]
    if fault != "hash":
        bundle["evidence_sha256"] = H.sha(H.canonical(e))
    with pytest.raises((ValueError, KeyError)):
        H.validate(bundle, F.C)


def test_compressed_size_and_trailing_bytes_rejected(bundle):
    raw = zlib.compress(H.canonical(bundle))
    with pytest.raises(ValueError):
        H.decode_output(base64.b64encode(raw + b"extra").decode(), F.C)
    with pytest.raises(ValueError):
        H.decode_output(base64.b64encode(raw[:-2]).decode(), F.C)
    with pytest.raises(ValueError):
        H.decode_output(base64.b64encode(zlib.compress(b" " * (H.CAP + 1))).decode(), F.C)


def test_readonly_collector_rejects_symlink(tmp_path):
    p = tmp_path / "original"
    p.write_text("{}")
    q = tmp_path / "alias"
    q.symlink_to(p)
    with pytest.raises(ValueError):
        H.read_small(q)


def test_existing_queued_output_reconstructs_only_exact_original_bytes(bundle):
    evidence = bundle["evidence"]
    terminal = json.loads(evidence["terminal_text"])
    completion = dict(
        phase="COLAB_FULL_RECOVERY_COMPLETE_PRIVATE_PRESERVED",
        updates=2916,
        epochs=6,
        seed=20261021,
        continuation=terminal["execution_continuation"],
        publication_proofs=evidence["epoch_publications"] + [evidence["terminal_publication"]],
    )
    rebuilt = H.from_saved_output(
        terminal, completion, json.loads(evidence["own_source_text"]), F.C
    )
    assert H.validate(rebuilt, F.C)["terminal_text"] == evidence["terminal_text"]
    completion["publication_proofs"][6]["files"]["terminal.json.bin"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        H.from_saved_output(terminal, completion, json.loads(evidence["own_source_text"]), F.C)


def test_epoch2_paced_publication_handoff_requires_original_proofs():
    _, e = F.case.__wrapped__()
    F.recovered2(e)
    terminal = json.loads(e["terminal_text"])
    completion = dict(
        phase="COLAB_FULL_RECOVERY2_COMPLETE_PRIVATE_PRESERVED",
        updates=2916,
        epochs=6,
        seed=20261021,
        continuation=terminal["execution_continuation"],
        publication_proofs=e["epoch_publications"] + [e["terminal_publication"]],
    )
    result = H.from_saved_output(terminal, completion, json.loads(e["own_source_text"]), F.C)
    assert H.validate(result, F.C)["terminal_text"] == e["terminal_text"]
    completion["publication_proofs"][1]["files"]["full-state.private.pt.bin"]["sha256"] = "0" * 64
    with pytest.raises(ValueError):
        H.from_saved_output(terminal, completion, json.loads(e["own_source_text"]), F.C)


def test_adopt_is_an_exclusive_mutation_and_requires_actual_proof():
    _, b = OWNER.base()
    assert "adopt" in b.MUTATIONS and b.DEPLOY == OWNER.DEPLOY
    assert (
        b.phase_result(
            "adopt", "qualification", {"status": "waiting_existing_canonical_claim_no_mutation"}
        )
        == "wait"
    )
    good = dict(
        status="original_Colab_bytes_verified_existing_Dataset_admitted",
        private=True,
        version=1,
        original_Colab_downloaded_all_bytes=True,
        full_optimizer_downloaded_again=False,
    )
    assert b.phase_result("adopt", "qualification", good) == "done"
    bad = copy.deepcopy(good)
    bad["private"] = False
    with pytest.raises(ValueError):
        b.phase_result("adopt", "qualification", bad)


def test_frozen_single_model_observer_includes_canonical_Newton_identity():
    from types import SimpleNamespace

    m, _ = OWNER.base()
    c = m.training()
    f = c.frozen(SimpleNamespace(source_git=m.SOURCE))
    body = OWNER.observer_body(m, f, c.original_jobs())
    compile(body, "observer", "exec")
    assert "mode=ro" in body and "prepared['bundle_identity']" in body
    assert "row.get('model_files') == {model['model_filename']:model['model']}" in body
