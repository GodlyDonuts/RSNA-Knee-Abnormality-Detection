import copy
import hashlib
import importlib.util
import json
import types
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "Colab_readonly_reconciliation", HERE / "reconcile_existing.py"
)
R = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(R)


def fixture(tmp_path, key="epoch-001"):
    lease = json.loads((HERE / "actual-A10040-checkpoint-lease.json").read_text())["exact_lease"]
    names = (
        ["terminal.json"]
        if key == "terminal"
        else [
            "model.pt",
            "full-state.private.pt",
            "run-start.json",
            "source-permission.txt",
            "colab_train.py",
            "colab_storage.py",
            "kaggle_io.py",
        ]
    )
    payloads = {n + ".bin": (n + " original private fixture").encode() for n in names}
    files = {
        n: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in payloads.items()
    }
    intent = {"slug": lease["slugs"][key], "lease": lease, "files": files, "automatic_retry": False}
    journal = tmp_path / "original"
    journal.mkdir()
    (journal / "Kaggle-intent.json").write_text(json.dumps(intent))
    metadata = {"id": 123, "version": 1, "private": True, "slug": intent["slug"]}
    calls = []

    def call(service, method, body):
        assert service == "datasets.DatasetApiService" and method == "GetDatasetStatus"
        calls.append(method)
        return {"status": "READY"}

    def stream(_remote, slug, name, proof, **kwargs):
        assert slug == intent["slug"]
        actual = payloads[name]
        assert {"bytes": len(actual), "sha256": hashlib.sha256(actual).hexdigest()} == proof
        calls.append(name)

    # No create/upload methods are available on either injected API object.
    remote = types.SimpleNamespace(call=call)
    pub = types.SimpleNamespace(
        metadata=lambda *a: copy.deepcopy(metadata),
        list_files=lambda *a: {n: p["bytes"] for n, p in files.items()},
        stream_file=stream,
    )
    return lease, journal, remote, pub, metadata, calls, payloads


@pytest.mark.parametrize("key", ["epoch-001", "terminal"])
def test_uncertain_original_creation_reconciles_without_any_mutation(tmp_path, key):
    lease, journal, remote, pub, _, calls, payloads = fixture(tmp_path, key)
    before = (journal / "Kaggle-intent.json").read_bytes()
    receipt = tmp_path / "reconciled.json"
    result = R.reconcile(remote, pub, journal, key, lease, receipt, stopped_observation=True)
    assert result["all_original_bytes_downloaded_in_this_reconciliation"] is True
    assert set(calls) == {"GetDatasetStatus", *payloads}
    assert result["new_Dataset_create_upload_GPU_or_competition_calls"] == 0
    assert not result["training_resumed"] and not (journal / "verified.json").exists()
    assert (journal / "Kaggle-intent.json").read_bytes() == before
    with pytest.raises(ValueError, match="overwrite"):
        R.reconcile(remote, pub, journal, key, lease, receipt, stopped_observation=True)


def test_completed_original_proof_is_not_downloaded_again(tmp_path):
    lease, journal, remote, pub, _, calls, _ = fixture(tmp_path)
    result = R.reconcile(
        remote, pub, journal, "epoch-001", lease, tmp_path / "first.json", stopped_observation=True
    )
    (journal / "verified.json").write_text(json.dumps(result["publication"]))
    calls.clear()
    reused = R.reconcile(
        remote, pub, journal, "epoch-001", lease, tmp_path / "second.json", stopped_observation=True
    )
    assert reused["original_verified_receipt_reused"] and calls == ["GetDatasetStatus"]
    assert not reused["all_original_bytes_downloaded_in_this_reconciliation"]


@pytest.mark.parametrize(
    "fault",
    [
        "active",
        "lease",
        "public",
        "version",
        "pending",
        "listing",
        "hash",
        "changing_identity",
        "original_proof",
    ],
)
def test_refuses_unsafe_or_uncertain_originals_without_publishing(tmp_path, fault):
    lease, journal, remote, pub, meta, _, payloads = fixture(tmp_path)
    if fault == "lease":
        lease["CPU_job"] = "not-the-original"
    elif fault == "public":
        meta["private"] = False
    elif fault == "version":
        meta["version"] = 2
    elif fault == "pending":
        remote.call = lambda *a: {"status": "PROCESSING"}
    elif fault == "listing":
        pub.list_files = lambda *a: {}
    elif fault == "hash":
        payloads["model.pt.bin"] = b"changed"
    elif fault == "changing_identity":
        first = copy.deepcopy(meta)

        def changing(*args):
            nonlocal first
            if first is not None:
                answer, first = first, None
                return answer
            return {**meta, "id": 999}

        pub.metadata = changing
    elif fault == "original_proof":
        (journal / "verified.json").write_text("{}")
    receipt = tmp_path / "reconciled.json"
    with pytest.raises((ValueError, AssertionError)):
        R.reconcile(
            remote, pub, journal, "epoch-001", lease, receipt, stopped_observation=fault != "active"
        )
    assert not receipt.exists()
