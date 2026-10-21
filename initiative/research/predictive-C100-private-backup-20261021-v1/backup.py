"""One private Kaggle backup of original epochs4–6 under the canonical claim."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

RUN = Path("/tmp/rsna-predictive-full-C100-20261020-v1")
ROOT = Path("/tmp/rsna-C100-backup456-20261021-v1")
LEDGER = Path("/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3")
SCIENCE = "8f1e8f2bb9f2ded8a2e4a582718188b68639db8b"


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def once(path, value):
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def main(v):
    import requests

    os.umask(0o077)
    assert os.uname().nodename == "evc27" and not os.environ.get("SLURM_JOB_GPUS")
    assert not ROOT.exists()
    assert shutil.disk_usage("/tmp").free > (518 + 20) * 2**30
    q = subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True)
    row = next(x.split() for x in q.splitlines() if x.strip().startswith("/home/sa305415"))
    assert (int(row[2]) - int(row[1])) * 1024 > 20 * 2**30 + 123 * 2**20 and int(row[6]) - int(
        row[5]
    ) >= 5370
    terminal = json.loads((RUN / "terminal.json").read_text())
    assert (
        terminal["GPU_job"] == "995351"
        and terminal["source_git"] == SCIENCE
        and terminal["updates"] == 2916
        and terminal["epochs"] == 6
    )
    state = subprocess.check_output(
        ["sacct", "-X", "-j", "995351", "-n", "-P", "--format=JobIDRaw,State,ExitCode"], text=True
    )
    assert [x.split("|")[:3] for x in state.splitlines() if x.startswith("995351|")] == [
        ["995351", "COMPLETED", "0:0"]
    ]
    # Keep the original failure: no HF retry or billing mutation.
    failed = json.loads(
        (
            RUN / "publication/epoch-004/commit-completion-canonical-path-20261021-v1/failed.json"
        ).read_text()
    )
    assert failed["HTTP"] == 400 and "automatic credit recharge" in failed["error"]
    binding = json.loads(
        Path("/tmp/rsna-predictive-density100-full-20261020-v1/binding.private.json").read_text()
    )
    records = {}
    for epoch in (4, 5, 6):
        p = RUN / f"publication/epoch-{epoch:03d}"
        assert not (p / "verified.json").exists()
        name = "intent.json" if epoch == 4 else "export-ready.json"
        r = json.loads((p / name).read_text())
        assert r["epoch"] == epoch and r["source_git"] == SCIENCE and r["GPU_job"] == "995351"
        assert set(r["files"]) == {
            "model.pt",
            "provenance.json",
            "own-source.json",
            "source-permission.txt",
            "manifest.json",
        }
        records[epoch] = r
    ROOT.mkdir(mode=0o700)
    control = ROOT / "control"
    control.mkdir()
    for name, source in v["helpers"].items():
        assert name in ("workflow.py", "components.py", "publication.py")
        (control / name).write_text(source)
    w = load("backup_canonical_workflow", control / "workflow.py")
    pub = load("backup_canonical_publication", control / "publication.py")
    remote = w.Remote.__new__(w.Remote)
    remote.session = requests.Session()
    remote.session.headers.update({"Authorization": "Bearer " + v["token"]})
    ident = remote.call("security.OAuthService", "IntrospectToken", {"token": v["token"]})
    assert ident.get("active") is True and ident.get("username") == "godlydonuts"
    remote.username = ident["username"]
    bundle = ROOT / "bundle"
    bundle.mkdir()
    files = {}
    for epoch, r in records.items():
        target = bundle / f"epoch-{epoch:03d}"
        target.mkdir()
        for name, proof in r["files"].items():
            source = (
                Path(binding["license"]["path"])
                if name == "source-permission.txt"
                else RUN / f"publication/epoch-{epoch:03d}" / name
            )
            assert (
                source.is_file()
                and not source.is_symlink()
                and source.stat().st_size == proof["bytes"]
                and sha(source) == proof["sha256"]
            )
            os.link(source, target / name)
            files[f"epoch-{epoch:03d}/" + name] = proof
    once(
        bundle / "bundle-manifest.json",
        {
            "files": files,
            "science_git": SCIENCE,
            "GPU_job": "995351",
            "epochs": [4, 5, 6],
            "private": True,
            "no_originals_deleted": True,
        },
    )
    files["bundle-manifest.json"] = {
        "bytes": (bundle / "bundle-manifest.json").stat().st_size,
        "sha256": sha(bundle / "bundle-manifest.json"),
    }
    upload = ROOT / "upload"
    upload.mkdir()
    archive = pub.build_zip(bundle, files, upload / "bundle.zip")
    assert archive["bytes"] < 2**30
    slug = "godlydonuts/rsna-c100-backup456-" + archive["sha256"][:12]
    claim = LEDGER.parent / "coordinator-claim"
    claim.mkdir(mode=0o700)
    owner = {
        "kind": "original_C100_private_checkpoint_backup",
        "CPU_job": os.environ["SLURM_JOB_ID"],
        "source_git": v["source_git"],
        "slug": slug,
        "new_GPU_or_entries": 0,
    }
    w.new_json(claim / "owner.json", owner)
    retain = False
    # Match the same common private publication lock used by the original HF
    # exporter, in addition to the account claim. Original jobs are terminal.
    lock_runner = Path("/home/sa305415/rsna-knee/native-full-cache-two-lanes-v1/common/runner.py")
    assert sha(lock_runner) == "3575a18ecd7a13a2becc9348c48a9ebda95f8934069e67dd1912737e6194118c"
    common = load("original_backup_shared_lock", lock_runner)
    try:
        with (
            common.publication_lock(lock_runner.parent.parent / "publication.lock"),
            w.ledger(LEDGER, "godlydonuts") as db,
        ):
            w.verify_account(db, remote)
            key = "private-C100-backup456:" + SCIENCE
            assert db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone() is None
            absence = pub.require_absent(remote, slug)
            record = {
                **owner,
                "state": "create_intent",
                "private": True,
                "version": 1,
                "files": files,
                "archive": archive,
                "absence": absence,
                "automatic_retry": False,
            }
            once(ROOT / "intent.json", record)
            db.execute("INSERT INTO meta VALUES (?,?)", (key, json.dumps(record)))
            db.commit()
            retain = True
            returned = pub.sdk_create_once(
                v["token"], slug, "Private original C100 epochs4–6", upload
            )
            once(ROOT / "create-returned.json", returned)
            record.update(state="create_returned", response=returned)
            db.execute("UPDATE meta SET value=? WHERE key=?", (json.dumps(record), key))
            db.commit()
            deadline = time.monotonic() + 1200
            while True:
                status = remote.call(
                    "datasets.DatasetApiService",
                    "GetDatasetStatus",
                    {"ownerSlug": "godlydonuts", "datasetSlug": slug.split("/")[1]},
                )
                if status.get("status") in ("READY", 5):
                    break
                assert time.monotonic() < deadline
                time.sleep(15)
            verified = pub.reconcile(remote, slug, files)
            assert (
                verified["private"] is True
                and verified["version"] == 1
                and verified["downloaded_all_bytes"] is True
            )
            once(ROOT / "dataset-verified.json", verified)
            for epoch, r in records.items():
                proof = {
                    k: z
                    for k, z in r.items()
                    if k
                    not in (
                        "status",
                        "parent",
                        "automatic_retry",
                        "independent_private_copy_verified",
                    )
                }
                proof.update(
                    status="actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified",
                    objects=5,
                    storage_provider="Kaggle",
                    revision=f"Kaggle:{verified['id']}:v1:epoch-{epoch:03d}",
                    canonical_shared_publication_lock_used=True,
                    private_backup={
                        k: verified[k]
                        for k in ("id", "slug", "version", "private", "downloaded_all_bytes")
                    },
                )
                proof["private_backup"].update(
                    archive=archive,
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
                state="all_original_epochs_privately_download_verified", verified=verified
            )
            db.execute("UPDATE meta SET value=? WHERE key=?", (json.dumps(record), key))
            db.commit()
            retain = False
            result = {
                "status": "actual_C100_epochs456_privately_download_SHA_verified",
                "CPU_job": os.environ["SLURM_JOB_ID"],
                "Dataset": {k: verified[k] for k in ("id", "slug", "version", "private")},
                "archive": archive,
                "objects": len(files),
                "epochs": [4, 5, 6],
                "new_GPU_or_entries": 0,
                "source_git": v["source_git"],
            }
            once(ROOT / "complete.json", result)
            print(json.dumps(result))
    finally:
        if not retain:
            assert json.loads((claim / "owner.json").read_text()) == owner
            (claim / "owner.json").unlink()
            claim.rmdir()


if __name__ == "__main__":
    main(json.load(sys.stdin))
