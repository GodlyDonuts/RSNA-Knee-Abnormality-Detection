"""One explicitly reconciled metadata completion; never re-upload model bytes.

The original process failed after its blob upload and before any observed
commit. Recheck the immutable parent and complete path absence under the same
publication lock, then send one small original five-object commit. Any uncertain
outcome stops. This is not an automatic retry and changes no checkpoint bytes.
"""

import base64
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

RUN = Path("/tmp/rsna-predictive-full-C100-20261020-v1")
ORIGINAL = RUN / "publication/epoch-004"
ROOT = ORIGINAL / "commit-completion-canonical-path-20261021-v1"
REPO = "GodlyDonuts/RSNA-Data"
PARENT = "1ac9dfab79b1de09ba11fb184767293ad96f0ec2"
PREFIX = "checkpoints/predictive-C100-full3887-8f1e8f2bb9f2/C/epoch-004"
MODEL = "5e38e551b68b4d1b13638ce0407c83dea51491f5c11cedbf3c18f5fc34749a4a"


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def once(path, value):
    with path.open("x") as f:
        json.dump(value, f, sort_keys=True, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def payload(intent, paths):
    assert intent["prefix"] == PREFIX and intent["parent"] == PARENT
    assert intent["files"]["model.pt"]["sha256"] == MODEL
    assert set(intent["files"]) == {
        "model.pt",
        "provenance.json",
        "own-source.json",
        "source-permission.txt",
        "manifest.json",
    }
    records = [
        {
            "key": "header",
            "value": {
                "summary": "Complete reconciled original C100 epoch4 publication",
                "description": "Original blob upload succeeded; one metadata-only completion after unchanged-parent and full-path-absence reconciliation.",
                "parentCommit": PARENT,
            },
        }
    ]
    for name, item in intent["files"].items():
        p = paths[name]
        assert p.is_file() and not p.is_symlink() and p.stat().st_size == item["bytes"]
        if name == "model.pt":
            records.append(
                {
                    "key": "lfsFile",
                    "value": {
                        "path": PREFIX + "/" + name,
                        "algo": "sha256",
                        "oid": MODEL,
                        "size": item["bytes"],
                    },
                }
            )
        else:
            assert item["bytes"] < 2 * 2**20 and sha(p) == item["sha256"]
            records.append(
                {
                    "key": "file",
                    "value": {
                        "path": PREFIX + "/" + name,
                        "content": base64.b64encode(p.read_bytes()).decode(),
                        "encoding": "base64",
                    },
                }
            )
    data = b"".join(json.dumps(r).encode() + b"\n" for r in records)
    assert len(data) < 2**20
    return data


def main(v):
    import requests
    from huggingface_hub import HfApi

    os.umask(0o077)
    assert os.uname().nodename == "evc27" and not os.environ.get("SLURM_JOB_GPUS")
    assert (
        not ROOT.exists()
        and not (ORIGINAL / "commit.json").exists()
        and not (ORIGINAL / "verified.json").exists()
    )
    assert shutil.disk_usage("/tmp").free > (516 + 20) * 2**30
    q = subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True)
    row = next(x.split() for x in q.splitlines() if x.strip().startswith("/home/sa305415"))
    assert (int(row[2]) - int(row[1])) * 1024 > 20 * 2**30 + 123 * 2**20
    assert int(row[6]) - int(row[5]) >= 5000 + 370
    state = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", "998781", "--format=JobIDRaw,State,ExitCode"], text=True
    )
    rows = [line.strip().split("|") for line in state.splitlines()]
    assert [r[:3] for r in rows if r[0] == "998781"] == [["998781", "FAILED", "1:0"]]
    binding = json.loads(
        Path("/tmp/rsna-predictive-density100-full-20261020-v1/binding.private.json").read_text()
    )
    publisher = Path(binding["code_root"]) / "publish_epoch.py"
    assert sha(publisher) == binding["code_sha256"]["publish_epoch.py"]
    ns = {"__file__": str(publisher), "__name__": "original_publisher_readonly"}
    exec(compile(publisher.read_text(), str(publisher), "exec"), ns)
    assert sha(ns["CORE"]) == ns["CORE_SHA"]
    spec = importlib.util.spec_from_file_location("original_shared_publication_lock", ns["CORE"])
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    api = HfApi(token=v["token"])
    with common.shared_publication_lock():
        info = api.repo_info(REPO, repo_type="dataset")
        assert info.private is True and info.sha == PARENT
        assert info.id.lower() == REPO.lower()
        assert not list(
            api.get_paths_info(REPO, paths=[PREFIX], repo_type="dataset", revision=PARENT)
        )
        intent = json.loads((ORIGINAL / "intent.json").read_text())
        paths = {n: ORIGINAL / n for n in intent["files"]}
        paths["source-permission.txt"] = Path(binding["license"]["path"])
        # Original full model bytes were already hashed by its upload. The
        # server accepts only that SHA-addressed blob; no second upload occurs.
        data = payload(intent, paths)
        ROOT.mkdir(mode=0o700)
        once(
            ROOT / "intent.json",
            {
                "CPU_job": os.environ["SLURM_JOB_ID"],
                "original_failed_CPU": "998781",
                "parent": PARENT,
                "prefix": PREFIX,
                "metadata_bytes": len(data),
                "metadata_sha256": hashlib.sha256(data).hexdigest(),
                "automatic_retry": False,
                "model_bytes_uploaded": 0,
            },
        )
        response = requests.post(
            "https://huggingface.co/api/datasets/" + info.id + "/commit/main",
            headers={
                "Authorization": "Bearer " + v["token"],
                "Content-Type": "application/x-ndjson",
            },
            data=data,
            timeout=(20, 180),
            allow_redirects=False,
        )
        try:
            body = response.json()
        except ValueError:
            body = {
                "non_json_body": response.text[:2000],
                "location": response.headers.get("Location"),
            }
        # API body contains no input credential; keep it private and emit only
        # status/error/message/commit identifier fields.
        once(ROOT / "response.private.json", {"HTTP": response.status_code, "body": body})
        if response.status_code != 200:
            result = {
                "status": "metadata_completion_rejected_no_retry",
                "HTTP": response.status_code,
                "error": body.get("error"),
                "message": body.get("message"),
                "CPU_job": os.environ["SLURM_JOB_ID"],
                "model_bytes_uploaded": 0,
            }
            once(ROOT / "failed.json", result)
            print(json.dumps(result))
            return
        revision = body["commitOid"]
        assert len(revision) == 40 and revision != PARENT
        once(ROOT / "commit.json", {"revision": revision})
        for name, item in intent["files"].items():
            with tempfile.TemporaryDirectory(prefix="independent-download-", dir=ROOT) as temp:
                p = Path(
                    api.hf_hub_download(
                        REPO,
                        PREFIX + "/" + name,
                        repo_type="dataset",
                        revision=revision,
                        local_dir=temp,
                        force_download=True,
                    )
                )
                assert p.stat().st_size == item["bytes"] and sha(p) == item["sha256"]
        result = {k: z for k, z in intent.items() if k not in ("parent", "automatic_retry")}
        result.update(
            status="actual_predictive_model_only_epoch_private_all_objects_download_size_SHA_verified",
            revision=revision,
            objects=5,
            canonical_shared_publication_lock_used=True,
            metadata_completion_CPU_job=os.environ["SLURM_JOB_ID"],
            original_failed_CPU_job="998781",
            no_model_reupload=True,
        )
        once(ROOT / "verified.json", result)
        once(ORIGINAL / "commit.json", {"revision": revision, "reconciled_completion": str(ROOT)})
        once(ORIGINAL / "verified.json", result)
        print(json.dumps(result))


if __name__ == "__main__":
    main(json.load(sys.stdin))
