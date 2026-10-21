"""Preserve completed local epochs once, spaced, without coupling HTTP to fitting."""

import importlib.util
import json
import os
import sys
import time
from pathlib import Path

BASE = Path("/content/rsna-independent-C100-20261021-v1")
CONTROL = Path("/content/rsna-colab-private-control-20261021-v2")
FIRST = BASE / "run-C100-seed20261021"
PRIOR = BASE / "run-C100-seed20261021-recovery-v1"
RUN = BASE / "run-C100-seed20261021-recovery-v2"
RESUME_SHA = "9576fb26cea626754bc7cc4d94be0c25e0b545d41bf3ef428066a6b7d6f0eb28"
CHECKPOINT_SHA = "0fab361de8da414968391b8f7a596190d68cb1e2966e72e96429bcc8b5dae516"
MIN_SPACING = 900


def read(path):
    return json.loads(Path(path).read_text())


def file_plan(epoch):
    assert epoch in range(2, 7)
    run = PRIOR if epoch == 2 else RUN
    publication = run / f"publication-{epoch:03d}"
    files = {
        "model.pt": publication / "model.pt",
        "full-state.private.pt": run / f"checkpoints/epoch-{epoch:03d}.pt",
        "run-start.json": run / "run-start.json",
        "source-permission.txt": BASE / "source-permission.txt",
        "colab_train.py": CONTROL / "actual_colab_train.py",
        "colab_storage.py": CONTROL / "colab_storage.py",
        "kaggle_io.py": CONTROL / "kaggle_io.py",
        "resume-control.py": run / "resume-control.py",
        "continuation.json": run / "continuation.json",
    }
    destination = RUN / "publication-network-002" if epoch == 2 else publication
    return files, destination


def validate_terminal(t):
    assert t["status"] == "actual_full_C100_seed20261021_complete"
    assert t["updates"] == 2916 and t["epochs"] == 6 and t["clinical_or_Kaggle_score"] is None
    assert t["binding"]["branch"] == "Colab-C100-seed20261021"
    assert t["all_epochs_private_verified"] is False and t["all_epochs_local_immutable"] is True
    c = t["execution_continuation"]
    assert c["status"] == "exact_epoch002_resume_with_network_independent_local_preservation"
    assert c["resume_update"] == 972 and c["source_checkpoint_sha256"] == CHECKPOINT_SHA
    assert c["control_sha256"] == RESUME_SHA and c["remote_publication_deferred"] is True
    assert c["numerical_functions_or_recipe_changed"] is False


def main(token):
    os.umask(0o077)
    sys.path.insert(0, str(CONTROL))
    import colab_storage as storage

    terminal = read(RUN / "terminal.json")
    validate_terminal(terminal)
    assert not (RUN / "postfit-publication-start.json").exists(), (
        "Reconcile the original publication owner"
    )
    lease = read(CONTROL / "checkpoint-lease.private.json")
    assert storage.sha(CONTROL / "colab_storage.py") == lease["code_sha256"]["colab_storage.py"]
    assert storage.sha(RUN / "resume-control.py") == RESUME_SHA
    first = read(FIRST / "publication-001/reconciled-original.json")["publication"]
    assert first["id"] == 12756621 and first["private"] and first["version"] == 1
    old_intent = read(PRIOR / "publication-002/Kaggle-intent.json")
    assert old_intent["slug"] == lease["slugs"]["epoch-002"]
    assert old_intent["files"]["full-state.private.pt.bin"]["sha256"] == CHECKPOINT_SHA
    assert not (PRIOR / "publication-002/Kaggle-response.json").exists()
    store = storage.Store(token, lease)
    # The failed CreateDataset returned HTTP429. Reconcile its namespace again
    # before the separately authorized post-fit attempt; never create if found.
    absence = store.pub.require_absent(store.remote, lease["slugs"]["epoch-002"])
    old = Path("/content/rsna-colab-epoch001-recovery-20261021-v1/resume.py")
    assert storage.sha(old) == "27b87a02e899fda775bb9985841b0354c614805bbdc6ea08ea92855cf6bfea0f"
    spec = importlib.util.spec_from_file_location("original_readonly_retry", old)
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    original_call = store.remote.call
    store.remote.call = lambda service, method, body: previous.read_retry(
        original_call, service, method, body
    )
    storage.once(
        RUN / "postfit-publication-start.json",
        {
            "status": "one_postfit_writer_admitted",
            "source_sha256": storage.sha(__file__),
            "original_epoch002_429_namespace_absent": absence,
            "minimum_create_spacing_seconds": MIN_SPACING,
            "successful_epoch001_repeated": False,
            "epochs": [2, 3, 4, 5, 6],
            "maximum_new_successful_Datasets": 6,
            "automatic_write_retry": False,
            "original_lease": lease["canonical_ledger_key"],
            "bytes_within_original_80GiB_scope": True,
        },
    )
    proofs = [first]
    last_started = None

    def spacing():
        nonlocal last_started
        if last_started is not None:
            time.sleep(max(0, MIN_SPACING - (time.monotonic() - last_started)))
        last_started = time.monotonic()

    try:
        for epoch in range(2, 7):
            files, dest = file_plan(epoch)
            if epoch == 2:
                assert not dest.exists()
                dest.mkdir(mode=0o700)
                expected = old_intent["files"]
            else:
                expected = read(dest / "local-preserved.json")["files"]
            assert set(expected) == {n + ".bin" for n in files}
            assert all(
                p.is_file()
                and p.stat().st_size == expected[n + ".bin"]["bytes"]
                and storage.sha(p) == expected[n + ".bin"]["sha256"]
                for n, p in files.items()
            )
            spacing()
            proof = store.publish(f"epoch-{epoch:03d}", files, dest)
            assert proof["files"] == expected
            proofs.append(proof)
        spacing()
        proofs.append(
            store.publish(
                "terminal", {"terminal.json": RUN / "terminal.json"}, RUN / "terminal-publication"
            )
        )
        assert len(proofs) == 7 and len({p["id"] for p in proofs}) == 7
        assert all(
            p["private"] is True
            and p["version"] == 1
            and p["status"] == "private_epoch_full_download_size_SHA_verified"
            for p in proofs
        )
        completion = {
            "phase": "COLAB_FULL_RECOVERY2_COMPLETE_PRIVATE_PRESERVED",
            "updates": 2916,
            "epochs": 6,
            "seed": 20261021,
            "publication_proofs": proofs,
            "continuation": terminal["execution_continuation"],
            "clinical_or_Kaggle_score": None,
            "releasing_paid_runtime": True,
            "source_control_sha256": storage.sha(__file__),
        }
        storage.once(RUN / "postfit-completion.json", completion)
        print(
            json.dumps({"terminal": terminal, "completion": completion}, sort_keys=True), flush=True
        )
    except Exception as exc:
        response = getattr(exc, "response", None)
        failure = {
            "status": "postfit_publication_stopped_original_retained",
            "error_type": type(exc).__name__,
            "HTTP_status": getattr(response, "status_code", None),
            "runtime_must_remain_assigned": True,
            "no_automatic_mutation_retry": True,
            "completed_private_proofs": len(proofs),
        }
        storage.once(RUN / "postfit-publication-failed.json", failure)
        # The exact response is private diagnostic data; do not print tokens/URLs.
        if response is not None:
            with (RUN / "postfit-HTTP-response.private.txt").open("x") as f:
                f.write(response.text[:65536])
        print(json.dumps(failure), flush=True)
        raise


if __name__ == "__main__":
    main(json.load(sys.stdin)["token"])
