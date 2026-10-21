"""One canonical CPU-only notebook action; never GPU, Dataset or competition submit."""

import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

import requests

ROOT = Path("/tmp/rsna-predictive-hidden-input-audit-20261021-v1")
LEDGER = Path("/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def request_for(source, compiled):
    assert sha(source.encode()) == compiled["source_sha256"]
    assert compiled["GPU"] is False and compiled["competition_entry"] is False
    slug = "godlydonuts/rsna-predictive-headercpu-" + compiled["source_sha256"][:12]
    return {
        "slug": slug,
        "newTitle": slug.split("/")[1].replace("-", " "),
        "text": source,
        "language": "python",
        "kernelType": "script",
        "isPrivate": True,
        "enableGpu": False,
        "enableTpu": False,
        "enableInternet": False,
        "sessionTimeoutSeconds": 1200,
        "dockerImage": compiled["binding"]["pinned_Docker"],
        "dockerImagePinningType": "original",
        "datasetDataSources": [],
        "kernelDataSources": [],
        "competitionDataSources": ["rsna-knee-abnormality-detection"],
    }


def accept(receipt, compiled):
    assert receipt["status"] == "actual_accessible_DICOM_assumptions_audited"
    assert receipt["binding"] == compiled["binding"]
    assert receipt["GPU_count"] == 0 and receipt["model_or_label_reads"] is False
    assert (
        receipt["submission_csv_created"] is False and receipt["hidden_error_cause_proven"] is False
    )
    assert len(receipt["public"]) == 3 and 0 < receipt["elapsed_seconds"] < 900
    assert receipt["headers"] == sum(receipt["header_categories"].values())
    return receipt


def main(v):
    os.umask(0o077)
    assert os.uname().nodename == "evc27"
    assert os.environ.get("CUDA_VISIBLE_DEVICES", "") == "" and not os.environ.get("SLURM_JOB_GPUS")
    assert v["phase"] in ("push", "status")
    quota = subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True)
    q = next(x.split() for x in quota.splitlines() if x.strip().startswith("/home/sa305415"))
    funding = {
        "node_free_bytes": shutil.disk_usage("/tmp").free,
        "shared_free_bytes": (int(q[2]) - int(q[1])) * 1024,
        "shared_free_files": int(q[6]) - int(q[5]),
        "other_node_bytes": 518 * 2**30,
        "new_node_bytes": 32 * 2**20,
        "other_shared_bytes": 123 * 2**20,
        "other_shared_files": 370,
        "new_shared_bytes": 0,
        "new_shared_files": 0,
    }
    assert funding["node_free_bytes"] >= (518 + 20) * 2**30 + 32 * 2**20
    assert funding["shared_free_bytes"] >= 20 * 2**30 + 123 * 2**20
    assert funding["shared_free_files"] >= 5000 + 370
    assert not ROOT.is_symlink()
    ROOT.mkdir(mode=0o700, exist_ok=True)
    for name, body in v["common"].items():
        assert name in ("workflow.py", "components.py")
        path = ROOT / name
        if path.exists():
            assert path.read_text() == body
        else:
            with path.open("x") as f:
                f.write(body)
    spec = importlib.util.spec_from_file_location(
        "CPU_only_canonical_workflow", ROOT / "workflow.py"
    )
    w = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(w)
    remote = w.Remote.__new__(w.Remote)
    remote.session = requests.Session()
    remote.session.headers["Authorization"] = "Bearer " + v["token"]
    remote.username = "godlydonuts"
    who = remote.call("security.OAuthService", "IntrospectToken", {"token": v["token"]})
    assert who.get("active") and who.get("username") == remote.username
    request = request_for(v["source"], v["compiled"])
    claim = LEDGER.parent / "coordinator-claim"
    claim_deadline = time.monotonic() + 45
    while True:
        try:
            claim.mkdir(mode=0o700)
            break
        except FileExistsError:
            if time.monotonic() >= claim_deadline:
                return {"status": "waiting_existing_claim_no_mutation"}
            time.sleep(1)
    w.new_json(
        claim / "owner.json",
        {"kind": "CPU_only_input_preparation", "CPU_job": os.environ["SLURM_JOB_ID"]},
    )
    retain = False
    try:
        with w.ledger(LEDGER, "godlydonuts") as db:
            key = "research:predictive-hidden-input-audit-v1:" + v["compiled"]["source_sha256"]
            found = db.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
            row = json.loads(found[0]) if found else {}

            def save():
                db.execute("INSERT OR REPLACE INTO meta VALUES (?,?)", (key, json.dumps(row)))
                db.commit()

            if v["phase"] == "push":
                assert not row and not (ROOT / "push-intent.json").exists()
                absence = remote.require_absent_kernel(request["slug"])
                row = {
                    "state": "CPU_only_push_intent",
                    "source_git": v["pin"],
                    "source_sha256": v["compiled"]["source_sha256"],
                    "slug": request["slug"],
                    "version": 1,
                    "request_sha256": w.object_hash(request),
                    "GPU": False,
                    "competition_entry": False,
                    "absence": absence,
                    "funding": funding,
                    "CPU_job": os.environ["SLURM_JOB_ID"],
                    "automatic_retry": False,
                }
                w.new_json(ROOT / "push-intent.json", row)
                save()
                retain = True
                response = remote.call("kernels.KernelsApiService", "SaveKernel", request)
                w.new_json(ROOT / "push-response.json", response)
                assert response.get("ref") == "/code/" + request["slug"]
                assert response.get("versionNumber") == 1 and type(response.get("kernelId")) is int
                assert not any(
                    response.get(k)
                    for k in (
                        "error",
                        "invalidDatasetSources",
                        "invalidKernelSources",
                        "invalidCompetitionSources",
                    )
                )
                row.update(state="CPU_only_notebook_pushed_once", kernel_id=response["kernelId"])
                save()
                retain = False
            assert row.get("state") in (
                "CPU_only_notebook_pushed_once",
                "CPU_only_preparation_complete",
            )
            actual = remote.kernel("GetKernel", row["slug"], 1)
            metadata = actual["metadata"]
            assert metadata["id"] == row["kernel_id"] and actual["blob"]["source"] == v["source"]
            assert metadata["isPrivate"] and not metadata.get("enableGpu", False)
            assert not metadata.get("enableTpu", False) and not metadata.get(
                "enableInternet", False
            )
            assert metadata["dockerImage"] == request["dockerImage"]
            for name in ("datasetDataSources", "kernelDataSources", "competitionDataSources"):
                assert sorted(metadata.get(name, [])) == sorted(request[name])
            result = {
                "slug": row["slug"],
                "version": 1,
                "kernel_id": row["kernel_id"],
                "source_sha256": row["source_sha256"],
                "actual_GPU_disabled": True,
                "competition_entry": False,
                "new_Dataset": False,
                "status": row["state"],
                "CPU_job": os.environ["SLURM_JOB_ID"],
            }
            if v["phase"] == "status":
                status = remote.kernel("GetKernelSessionStatus", row["slug"], 1).get("status")
                result["status"] = status
                if status == "COMPLETE":
                    receipt = accept(
                        remote.output_json(row["slug"], 1, "predictive-hidden-input-audit.json"),
                        v["compiled"],
                    )
                    result["receipt"] = receipt
                    if row["state"] != "CPU_only_preparation_complete":
                        w.new_json(ROOT / "actual-complete.json", result)
                        row.update(
                            state="CPU_only_preparation_complete",
                            receipt_sha256=w.object_hash(receipt),
                        )
                        save()
            return result
    finally:
        if not retain:
            (claim / "owner.json").unlink()
            claim.rmdir()
