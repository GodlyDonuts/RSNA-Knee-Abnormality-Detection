"""One private input Dataset, under the canonical Newton account claim.

Only opaque files are uploaded. The Colab consumer performs the independent
download/hash verification; creation does not claim that verification occurred.
"""

import importlib.util
import inspect
import json
import os
import subprocess
import time
from pathlib import Path

LEDGER = Path("/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3")


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def private_multifile_publisher(publication):
    source = inspect.getsource(publication.sdk_create_once)
    old = """    if {x.name for x in upload.iterdir()} != {"bundle.zip"}:
        raise ValueError("Upload directory must contain only the explicit ZIP")"""
    new = """    paths = list(upload.iterdir())
    if not paths or any(not p.is_file() or p.is_symlink() or not p.name.endswith('.bin') for p in paths):
        raise ValueError("Only explicit opaque private files may be uploaded")"""
    assert source.count(old) == 1
    ns = dict(publication.__dict__)
    exec(compile(source.replace(old, new), "original_private_multifile_API", "exec"), ns)
    return ns["sdk_create_once"]


def create_private_inputs(token, slug, upload, proofs, journal, source_git):
    root = Path(__file__).resolve().parent
    w = load("Colab_transport_canonical_workflow", root / "workflow.py")
    pub = load("Colab_transport_canonical_publication", root / "publication.py")
    import requests

    remote = w.Remote.__new__(w.Remote)
    remote.session = requests.Session()
    remote.session.headers.update({"Authorization": "Bearer " + token})
    identity = remote.call("security.OAuthService", "IntrospectToken", {"token": token})
    assert identity.get("active") is True and identity.get("username") == "godlydonuts"
    remote.username = identity["username"]
    assert set(proofs) == {p.name for p in upload.iterdir()}
    assert "input-manifest.bin" in proofs
    assert sum(p["bytes"] for p in proofs.values()) < 48 * 2**30
    # Actual UI quota was 92.97 / 214.75 GB. Reserve 80 GiB for this entire
    # transport + six immutable checkpoints, leaving >35 GB for other work.
    quota = json.loads((root / "private-storage-observed.json").read_text())
    assert quota["private_dataset_used_GB"] == 92.97 and quota["private_dataset_limit_GB"] == 214.75
    assert time.time() - quota["observed_unix"] < 2 * 3600
    assert quota["private_dataset_limit_GB"] * 10**9 - 94 * 10**9 > 80 * 2**30
    usage = subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True)
    row = next(r.split() for r in usage.splitlines() if r.strip().startswith("/home/sa305415"))
    free_bytes, free_files = (int(row[2]) - int(row[1])) * 1024, int(row[6]) - int(row[5])
    for arm, job in (("B", "989036"), ("C", "989035"), ("C0", "989037")):
        for extension in ("out", "err"):
            log = LEDGER.parent / f"rsna-predictive-full2-{arm}-{job}.{extension}"
            assert log.is_file() and not log.is_symlink() and log.stat().st_uid == os.getuid()
    assert free_bytes > 20 * 2**30 + 123 * 2**20 and free_files >= 5000 + 376 - 6
    claim = LEDGER.parent / "coordinator-claim"
    claim.mkdir(mode=0o700)
    owner = {
        "kind": "private_Colab_input_handoff",
        "CPU_job": os.environ["SLURM_JOB_ID"],
        "source_git": source_git,
        "slug": slug,
        "new_GPU_or_competition_calls": 0,
    }
    w.new_json(claim / "owner.json", owner)
    retain = False
    try:
        with w.ledger(LEDGER, "godlydonuts") as db:
            w.verify_account(db, remote)
            key = "private-colab-input-handoff:" + slug
            assert db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone() is None
            assert not (journal / "dataset-intent.json").exists()
            absence = pub.require_absent(remote, slug)
            record = {
                **owner,
                "state": "create_intent",
                "private": True,
                "version": 1,
                "files": proofs,
                "absence": absence,
                "automatic_retry": False,
            }
            w.new_json(journal / "dataset-intent.json", record)
            db.execute("INSERT INTO meta VALUES (?,?)", (key, json.dumps(record)))
            db.commit()
            retain = True
            returned = private_multifile_publisher(pub)(
                token, slug, "Private RSNA Colab contexts", upload
            )
            w.new_json(journal / "dataset-response.json", returned)
            record.update(state="private_create_returned_download_pending", response=returned)
            db.execute("UPDATE meta SET value=? WHERE key=?", (json.dumps(record), key))
            db.commit()
            retain = False
            return {
                "status": "private_Kaggle_input_creation_returned_download_pending",
                "CPU_job": os.environ["SLURM_JOB_ID"],
                "source_git": source_git,
                "slug": slug,
                "private_requested": True,
                "version": 1,
                "new_Kaggle_dataset_calls": 1,
                "new_GPU_or_competition_calls": 0,
                "files": proofs,
                "total_bytes": sum(p["bytes"] for p in proofs.values()),
                "HF_completed_uploads_repeated": 0,
                "Colab_independent_download_pending": True,
            }
    finally:
        if not retain:
            assert json.loads((claim / "owner.json").read_text()) == owner
            (claim / "owner.json").unlink()
            claim.rmdir()
