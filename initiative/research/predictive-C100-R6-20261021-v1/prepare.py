"""One CPU preparation on evc27. No GPU, upload, notebook push or submission."""

import importlib.util
import json
import os
import shutil
import sqlite3
import subprocess
from pathlib import Path

import requests

ROOT = Path("/tmp/rsna-C100-R6-20261021-v1")
ORIGINAL = Path("/tmp/rsna-predictive-final-deployment-20261020-v1-C100-selected-20261021-v1")
LEDGER = Path("/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3")


def load(name, path):
    s = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(s)
    s.loader.exec_module(m)
    return m


def main(v):
    os.umask(0o077)
    assert os.uname().nodename == "evc27" and not os.environ.get("SLURM_JOB_GPUS")
    assert not os.environ.get("CUDA_VISIBLE_DEVICES") and not ROOT.exists()
    quota = subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True)
    q = next(
        line.split() for line in quota.splitlines() if line.strip().startswith("/home/sa305415")
    )
    funding = dict(
        node_free_bytes=shutil.disk_usage("/tmp").free,
        shared_free_bytes=(int(q[2]) - int(q[1])) * 1024,
        shared_free_files=int(q[6]) - int(q[5]),
        prior_node_bytes=518 * 2**30 + 128 * 2**20,
        new_node_bytes=16 * 2**20,
        prior_shared_bytes=125 * 2**20,
        prior_shared_files=370,
        new_shared_bytes=0,
        new_shared_files=0,
        new_model_bytes=0,
    )
    assert funding["node_free_bytes"] >= (518 + 20) * 2**30 + 144 * 2**20
    assert (
        funding["shared_free_bytes"] >= 20 * 2**30 + 125 * 2**20
        and funding["shared_free_files"] >= 5370
    )
    ROOT.mkdir(mode=0o700)
    for name, text in v["files"].items():
        assert name in {"compile.py", "supervisor.py", "fusion.py", "workflow.py", "components.py"}
        with (ROOT / name).open("x") as f:
            f.write(text)
    c = load("C100_R6_private_compiler", ROOT / "compile.py")
    w = load("C100_R6_readonly_account", ROOT / "workflow.py")
    remote = w.Remote.__new__(w.Remote)
    remote.session = requests.Session()
    remote.session.headers["Authorization"] = "Bearer " + v["token"]
    remote.username = "godlydonuts"
    identity = remote.call("security.OAuthService", "IntrospectToken", {"token": v["token"]})
    assert identity.get("active") and identity.get("username") == remote.username
    c_source = (ORIGINAL / "compiled-C/source.private.py").read_bytes()
    assert c.digest(c_source) == c.C100
    original = remote.kernel("GetKernel", "godlydonuts/rsna-public0958-script-b472091d7a32", 2)
    r6_source = original["blob"]["source"].encode()
    assert c.digest(r6_source) == c.R6 and original["metadata"]["id"] == 139271768
    key = "licensed-parent-full-epoch-serving:exact-public0958-script:b472091d7a323e0165958ca01e6f023b21db9c3d2b10d3f79ba8962fb83efc5b"
    with sqlite3.connect("file:" + str(LEDGER) + "?mode=ro", uri=True) as db:
        row = json.loads(db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()[0])
    assert row["submission_id"] == 57333150 and row["source_sha256"] == c.R6
    r6_public = row["actual_public_accepted"]["receipt"]
    assert r6_public["R6_fallbacks"] == 0 and r6_public["completed_original_cells"] == list(
        range(8)
    )
    assert (
        r6_public["small_output_proofs"]["submission.csv"]["sha256"]
        == "5e9ba665f3e30642e18dbb555eb7cbd16ab8a659ffb5f7226c72aaac44c9df1b"
    )
    source, receipt = c.build(
        c_source,
        r6_source,
        v["qualification"],
        v["public"],
        supervisor_text=v["files"]["supervisor.py"],
        fusion_text=v["files"]["fusion.py"],
    )
    for name, data in (
        ("source.private.py", source.encode()),
        ("R6-source.private.py", r6_source),
        ("R6-original-metadata.private.json", c.canonical(original["metadata"])),
    ):
        with (ROOT / name).open("xb") as f:
            f.write(data)
    assert sum(p.stat().st_size for p in ROOT.iterdir()) < 16 * 2**20
    receipt.update(
        source_git=v["pin"],
        funding=funding,
        CPU_job=os.environ["SLURM_JOB_ID"],
        current_C100_public_score_known=False,
        old_conditional_C_branch_unchanged=True,
        original_R6_public=r6_public,
        new_account_mutations=0,
        original_R6_metadata_sha256=c.digest(c.canonical(original["metadata"])),
    )
    with (ROOT / "prepared.json").open("x") as f:
        json.dump(receipt, f, indent=2)
    return receipt
