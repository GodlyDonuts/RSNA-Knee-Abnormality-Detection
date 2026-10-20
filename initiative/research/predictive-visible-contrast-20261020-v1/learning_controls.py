"""Bounded storage and original-input checks shared by CPU admission and GPU run."""

import hashlib
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/tmp/rsna-predictive-contrast-learning-20261020-v1")
ORIGINAL = Path("/tmp/rsna-predictive-stable-warmup-20261020-v1")
PORTABLE = Path("/tmp/rsna-predictive-portable-recovery-20261020-v1")
PRIMARY = Path("/tmp/rsna-predictive-evc27-deadline-placement-20261020-v1")
BINDING_SHA = "43304b78b34b407f55088d4f8e610af8d1cbb112a9960f75311689a0078a75cc"
ENGINEERING_SHA = "66bf73182bd4aefa6dbf7444a65bd1f841ab4279cabb1bec3ea058ebbbb82b10"
INDEX_SHA = "526f087ae214c38d0cffdfcc4ba2196d7cc5fd03171080ebd8c48575f1041bf1"


def sha(body):
    return hashlib.sha256(body).hexdigest()


def signature(path):
    st = path.stat()
    assert path.is_file() and not path.is_symlink() and st.st_uid == os.getuid()
    assert not st.st_mode & 0o077
    return [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def read(path, expected=None):
    before = signature(path)
    assert before[1] <= 4 * 2**20
    body = path.read_bytes()
    assert signature(path) == before
    assert expected is None or sha(body) == expected
    return json.loads(body), body


def primary_paid_logs(primary=PRIMARY, logs=Path("/home/sa305415/rsna-knee/kaggle")):
    """Only original sealed dispatches and physically existing logs reduce liabilities."""
    dispatch = primary / "dispatch"
    if not dispatch.exists():
        assert not dispatch.is_symlink()
        return 0
    intent, _ = read(dispatch / "intent.json")
    assert intent["source_git"] == "44cc559e947560a635c34fee7b1f2a3b7be3a241"
    assert intent["owner_git"] == "2a224ae05fc7c2fc78d4d508b6407344ac523a1c"
    assert intent["arms"] == ["C", "B", "C0"] and intent["automatic_retry"] is False
    inodes = set()
    for arm in intent["arms"]:
        path = dispatch / (arm + "-submitted.json")
        if not path.exists():
            assert not path.is_symlink()
            continue
        job, _ = read(path)
        single, _ = read(dispatch / (arm + "-intent.json"))
        returned, _ = read(dispatch / (arm + "-sbatch.json"))
        identity, name = job["GPU_job"], "rsna-predictive-full2-" + arm
        assert re.fullmatch(r"\d+", identity) and returned["returncode"] == 0
        assert re.fullmatch(r"\d+(?:;[^\s]+)?\s*", returned["stdout"])
        assert returned["stdout"].strip().split(";")[0] == identity
        assert job["arm"] == single["arm"] == arm and job["node"] == "evc27"
        assert job["source_git"] == single["source_git"] == intent["source_git"]
        assert job["job_name"] == name and single["automatic_retry"] is False
        assert job["run_root"] == f"/tmp/rsna-predictive-full-{arm}-20261020-v2"
        _, binding = read(primary / f"full-admission/binding-{arm}.private.json")
        script = dispatch / (arm + ".sh")
        signature(script)
        assert job["binding_sha256"] == single["binding_sha256"] == sha(binding)
        assert job["script_sha256"] == single["script_sha256"] == sha(script.read_bytes())
        size = 0
        for field, suffix in (("stdout", "out"), ("stderr", "err")):
            log = logs / f"{name}-{identity}.{suffix}"
            assert job[field] == str(log) and not log.is_symlink()
            if log.exists():
                signature(log)
                st = log.stat()
                key = (st.st_dev, st.st_ino)
                assert key not in inodes
                inodes.add(key)
                size += st.st_size
        assert size <= 8 * 2**20
    assert len(inodes) <= 6
    return len(inodes)


def capacity(root=ROOT):
    files = [p for p in root.rglob("*") if p.is_file()]
    assert all(not p.is_symlink() and p.stat().st_uid == os.getuid() for p in files)
    cache = sum(p.stat().st_size for p in files if root / "contexts" in p.parents)
    checkpoints = sum(p.stat().st_size for p in files if root / "engineering" in p.parents)
    controls = sum(p.stat().st_size for p in files if root / "contexts" not in p.parents
                   and root / "engineering" not in p.parents)
    assert cache <= 2**30 and checkpoints <= 6 * 2**30 and controls <= 4 * 2**20
    required = 394815078400 + 7 * 2**30 + 4 * 2**20 - cache - checkpoints - controls
    free = shutil.disk_usage(root).free
    assert free > required
    paid_logs = primary_paid_logs()
    quota = subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True)
    row = next(x.split() for x in quota.splitlines() if x.strip().startswith("/home/sa305415"))
    free_shared, free_files = (int(row[2]) - int(row[1])) * 1024, int(row[6]) - int(row[5])
    assert free_shared > 20 * 2**30 + 123 * 2**20
    assert free_files >= 5000 + 376 - paid_logs
    return {"free_bytes": free, "required_after_exact_paid_bytes": required,
            "paid_cache_bytes": cache, "paid_checkpoint_bytes": checkpoints,
            "paid_control_bytes": controls, "exact_paid_original_log_files": paid_logs,
            "shared_files_after_future_liabilities": free_files - 376 + paid_logs,
            "shared_bytes_after_future_liabilities": free_shared - 123 * 2**20}


def terminal_jobs(text):
    found = {}
    for line in text.splitlines():
        if not line.strip():
            continue
        identity, status, code, node = line.split("|")
        assert identity not in found
        found[identity] = (status, code, node)
    assert found == {"984645": ("COMPLETED", "0:0", "evc33"),
                     "984690": ("COMPLETED", "0:0", "evc33"),
                     "984691": ("COMPLETED", "0:0", "evc27")}
    return found
