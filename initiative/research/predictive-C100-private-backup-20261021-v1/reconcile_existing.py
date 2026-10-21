"""Finish the original returned backup after its read-only status HTTP403.

No CreateDataset, file upload, checkpoint export, archive rebuild or training.
Downloads the original 16 objects once and finishes the original local journal.
"""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

RUN = Path("/tmp/rsna-predictive-full-C100-20261020-v1")
ROOT = Path("/tmp/rsna-C100-backup456-20261021-v1")
LEDGER = Path("/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3")
SCIENCE = "8f1e8f2bb9f2ded8a2e4a582718188b68639db8b"
ORIGINAL = "00512114b20c70e4f840d7a8394303ceca93e2c7"
SLUG = "godlydonuts/rsna-c100-backup456-6df0b38cfc79"
DATASET = 12762120


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read(path):
    return json.loads(path.read_text())


def once(path, value):
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def main(value):
    import requests

    os.umask(0o077)
    assert os.uname().nodename == "evc27" and not os.environ.get("SLURM_JOB_GPUS")
    assert shutil.disk_usage("/tmp").free > (518 + 20) * 2**30
    q = subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True)
    row = next(x.split() for x in q.splitlines() if x.strip().startswith("/home/sa305415"))
    assert (int(row[2]) - int(row[1])) * 1024 > 20 * 2**30 + 123 * 2**20
    assert int(row[6]) - int(row[5]) >= 5370
    states = subprocess.check_output(
        ["sacct", "-X", "-j", "995351,1000318", "-n", "-P", "--format=JobIDRaw,State,ExitCode"],
        text=True,
    )
    rows = {r.split("|")[0]: r.split("|")[1:3] for r in states.splitlines() if r.strip()}
    assert rows["995351"] == ["COMPLETED", "0:0"]
    assert rows["1000318"][0] == "FAILED"
    assert not subprocess.check_output(["squeue", "-h", "-j", "1000318"], text=True).strip()
    terminal = read(RUN / "terminal.json")
    assert (
        terminal["source_git"] == SCIENCE
        and terminal["updates"] == 2916
        and terminal["epochs"] == 6
    )
    original = read(ROOT / "intent.json")
    returned = read(ROOT / "create-returned.json")
    assert original["slug"] == SLUG and original["source_git"] == ORIGINAL
    assert (
        original["CPU_job"] == "1000318"
        and original["version"] == 1
        and original["private"] is True
    )
    assert (
        returned["returned"] is True
        and returned["status"] == "Ok"
        and returned["final_create_rpc_calls"] == 1
    )
    assert not (ROOT / "dataset-verified.json").exists() and not (ROOT / "complete.json").exists()
    assert not (ROOT / "existing-dataset-reconciliation-intent.json").exists()
    for name, expected in value["helper_sha256"].items():
        path = ROOT / "control" / name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    w = load("backup_reconciliation_workflow", ROOT / "control/workflow.py")
    pub = load("backup_reconciliation_publication", ROOT / "control/publication.py")
    remote = w.Remote.__new__(w.Remote)
    remote.session = requests.Session()
    remote.session.headers["Authorization"] = "Bearer " + value["token"]
    identity = remote.call("security.OAuthService", "IntrospectToken", {"token": value["token"]})
    assert identity.get("active") and identity.get("username") == "godlydonuts"
    remote.username = identity["username"]
    meta = pub.metadata(remote, SLUG)
    assert meta["id"] == DATASET and meta["version"] == 1 and meta["private"] is True
    claim = LEDGER.parent / "coordinator-claim"
    owner = read(claim / "owner.json")
    assert owner == {
        k: original[k] for k in ("kind", "CPU_job", "source_git", "slug", "new_GPU_or_entries")
    }
    lock_runner = Path("/home/sa305415/rsna-knee/native-full-cache-two-lanes-v1/common/runner.py")
    assert (
        hashlib.sha256(lock_runner.read_bytes()).hexdigest()
        == "3575a18ecd7a13a2becc9348c48a9ebda95f8934069e67dd1912737e6194118c"
    )
    common = load("original_backup_shared_lock", lock_runner)
    key = "private-C100-backup456:" + SCIENCE
    with (
        common.publication_lock(lock_runner.parent.parent / "publication.lock"),
        w.ledger(LEDGER, "godlydonuts") as db,
    ):
        record = json.loads(db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()[0])
        assert record == {**original, "state": "create_returned", "response": returned}
        records = {}
        for epoch in (4, 5, 6):
            folder = RUN / f"publication/epoch-{epoch:03d}"
            assert not (folder / "verified.json").exists()
            r = read(folder / ("intent.json" if epoch == 4 else "export-ready.json"))
            assert r["epoch"] == epoch and r["source_git"] == SCIENCE and r["GPU_job"] == "995351"
            assert {f"epoch-{epoch:03d}/" + n: proof for n, proof in r["files"].items()} == {
                n: p for n, p in original["files"].items() if n.startswith(f"epoch-{epoch:03d}/")
            }
            records[epoch] = r
        assert len(original["files"]) == 16
        once(
            ROOT / "existing-dataset-reconciliation-intent.json",
            {
                "source_git": value["source_git"],
                "CPU_job": os.environ["SLURM_JOB_ID"],
                "original_CPU_job": "1000318",
                "Dataset_id": DATASET,
                "slug": SLUG,
                "new_uploads": 0,
                "new_CreateDataset_calls": 0,
                "original_claim_retained": owner,
            },
        )
        verified = pub.reconcile(remote, SLUG, original["files"])
        assert (
            verified["id"] == DATASET and verified["private"] is True and verified["version"] == 1
        )
        assert verified["downloaded_all_bytes"] is True and verified["files"] == original["files"]
        once(ROOT / "dataset-verified.json", verified)
        for epoch, r in records.items():
            proof = {
                k: z
                for k, z in r.items()
                if k
                not in ("status", "parent", "automatic_retry", "independent_private_copy_verified")
            }
            proof.update(
                status="actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified",
                objects=5,
                storage_provider="Kaggle",
                revision=f"Kaggle:{DATASET}:v1:epoch-{epoch:03d}",
                canonical_shared_publication_lock_used=True,
                private_backup={
                    k: verified[k]
                    for k in ("id", "slug", "version", "private", "downloaded_all_bytes")
                },
            )
            proof["private_backup"].update(
                archive=original["archive"],
                original_members_verified_before_upload=True,
                members={f"epoch-{epoch:03d}/" + n: z for n, z in r["files"].items()},
                remote_names={
                    n: z
                    for n, z in verified["remote_names"].items()
                    if n.startswith(f"epoch-{epoch:03d}/")
                },
            )
            once(RUN / f"publication/epoch-{epoch:03d}/verified.json", proof)
        record.update(
            state="all_original_epochs_privately_download_verified",
            verified=verified,
            status_read_403_reconciliation={
                "CPU_job": os.environ["SLURM_JOB_ID"],
                "source_git": value["source_git"],
                "new_CreateDataset_calls": 0,
            },
        )
        db.execute("UPDATE meta SET value=? WHERE key=?", (json.dumps(record), key))
        db.commit()
        result = {
            "status": "actual_C100_epochs456_privately_download_SHA_verified",
            "CPU_job": os.environ["SLURM_JOB_ID"],
            "Dataset": {k: verified[k] for k in ("id", "slug", "version", "private")},
            "archive": original["archive"],
            "objects": 16,
            "epochs": [4, 5, 6],
            "new_GPU_or_entries": 0,
            "source_git": value["source_git"],
            "new_CreateDataset_calls": 0,
            "original_create_CPU_job": "1000318",
            "original_status_HTTP403_preserved": True,
        }
        once(ROOT / "complete.json", result)
        assert read(claim / "owner.json") == owner
        (claim / "owner.json").unlink()
        claim.rmdir()
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main(json.load(sys.stdin))
