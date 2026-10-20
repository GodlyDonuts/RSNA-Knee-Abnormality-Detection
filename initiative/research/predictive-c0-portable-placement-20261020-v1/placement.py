"""Portable C0 controls. Importing this module does not allocate, copy or dispatch."""

import copy
import hashlib
import json
import os
import re
from pathlib import Path

GIB, MIB = 2**30, 2**20
SCIENCE = "44cc559e947560a635c34fee7b1f2a3b7be3a241"
BASE = "initiative/research/predictive-full-fitting-20261020-v2"
ROOT33 = Path("/tmp/rsna-predictive-full-fitting-20261020-v2")
ROOT27 = Path("/tmp/rsna-predictive-c0-portable-placement-20261020-v1")
RETURN33 = Path("/tmp/rsna-predictive-C0-return-20261020-v1")
OLD27 = Path("/tmp/rsna-predictive-portable-recovery-20261020-v1")
PY33 = "/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python"
PY27 = str(OLD27 / "runtime/bin/python")
INITIAL = "dbf69ff87261f7a0657c866ef8e41db8b125a62c6245d80017945c634e60158b"
ROSTER = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"
TARGETS = "48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285"
CONTEXT_CAP = 96 * GIB
CONTROL_CAP = 64 * MIB
FIT_CAP = 12 * GIB
OTHER27 = 191 * GIB + 564 * MIB
EOF = b"RSNA_C0_EXACT_CONTEXT_EOF_V1"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha(data):
    return hashlib.sha256(data).hexdigest()


def file_sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def signature(path):
    path = Path(path)
    st = path.stat()
    require(
        path.is_absolute()
        and path.is_file()
        and not path.is_symlink()
        and st.st_uid == os.getuid()
        and not st.st_mode & 0o077,
        "Owned private regular file required",
    )
    return [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def once(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("xb") as stream:
        stream.write(canonical(value) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        os.link(temporary, path)
    finally:
        temporary.unlink()


def read_ref(ref):
    path = Path(ref["path"])
    before = signature(path)
    raw = path.read_bytes()
    require(sha(raw) == ref["sha256"] and signature(path) == before, "Reference changed")
    return raw


def reference(path):
    return {"path": str(path), "sha256": file_sha(path)}


def handoff(receipt):
    require(
        receipt["status"] == "actual_idle_68792_stopped_before_full_fitting_intent"
        and receipt["old_owner_pid"] == 68792
        and receipt["old_owner_source_git"] == SCIENCE
        and receipt["old_owner_children"] == []
        and receipt["old_local_phase_intents"] == []
        and receipt["old_remote_admission_or_dispatch_exists"] is False
        and receipt["predictive_full_GPU_jobs"] == []
        and receipt["owner_process_absent_after_stop"] is True
        and receipt["remote_jobs_cancelled"] == []
        and receipt["ownership"] == {"B": "evc33", "C": "evc33", "C0": "evc27"}
        and receipt["original_C0_shared_log_envelope_reassigned"] is True
        and re.fullmatch("[0-9a-f]{64}", receipt["remote_read_only_snapshot_sha256"]),
        "Actual exclusive idle-owner handoff required; no live-owner override",
    )


def capacity(free, *, contexts_paid=0, controls_paid=0, fit_paid=0):
    for paid, cap in (
        (contexts_paid, CONTEXT_CAP),
        (controls_paid, CONTROL_CAP),
        (fit_paid, FIT_CAP),
    ):
        require(type(paid) is int and 0 <= paid <= cap, "Exact bounded paid bytes required")
    required = (
        OTHER27
        + 20 * GIB
        + CONTEXT_CAP
        - contexts_paid
        + CONTROL_CAP
        - controls_paid
        + FIT_CAP
        - fit_paid
    )
    require(type(free) is int and free > required, "Destination reserve is not funded")
    return {"free_bytes": free, "required_bytes": required, "remaining_bytes": free - required}


def check_rows(rows, terminal):
    require(
        terminal["status"] == "actual_missing_only3887_exact_FP32_context_alias_recovery_accepted"
        and terminal["studies"] == 3887
        and len(rows) == 3887,
        "Actual complete original context producer required",
    )
    uids = [row["uid"] for row in rows]
    require(
        uids == sorted(set(uids)) and sha(canonical(uids)) == ROSTER,
        "Exact complete fitting order required",
    )
    require(len({row["path"] for row in rows}) == 3887, "Duplicate context paths")
    total = 0
    for row in rows:
        proof = row["proof"]
        require(
            type(proof["bytes"]) is int
            and proof["bytes"] > 0
            and re.fullmatch("[0-9a-f]{64}", proof["sha256"])
            and len(proof["stat"]) == 4
            and proof["stat"][1] == proof["bytes"],
            "Original whole-byte context proof required",
        )
        total += proof["bytes"]
    require(total <= CONTEXT_CAP, "Complete contexts exceed 96GiB")
    return total


def frame_read(stream, count):
    chunks, remaining = [], count
    while remaining:
        block = stream.read(remaining)
        require(block, "Interrupted original stream; retain partials and reconcile")
        chunks.append(block)
        remaining -= len(block)
    return b"".join(chunks)


def read_json_frame(stream, limit=CONTROL_CAP):
    size = int.from_bytes(frame_read(stream, 8), "big")
    require(0 < size <= limit, "Private metadata frame exceeds bound")
    return json.loads(frame_read(stream, size))


def write_json_frame(stream, value, limit=CONTROL_CAP):
    raw = canonical(value)
    require(0 < len(raw) <= limit, "Private metadata exceeds bound")
    stream.write(len(raw).to_bytes(8, "big"))
    stream.write(raw)
    stream.flush()


def context_name(index):
    require(type(index) is int and 0 <= index < 3887, "Invalid context ordinal")
    return f"context-{index:04d}.private.npz"


def missing_contexts(root, rows):
    """Reuse only already-complete verified members; never repeat uncertain transfers."""
    missing = []
    for i, row in enumerate(rows):
        path = root / "contexts" / context_name(i)
        receipt = path.with_suffix(".verified.json")
        if not path.exists() and not receipt.exists():
            require(not path.with_suffix(".partial").exists(), "Unresolved partial context")
            missing.append(i)
            continue
        require(path.exists() and receipt.exists(), "Uncertain context needs explicit review")
        actual = json.loads(receipt.read_text())
        require(
            actual["sha256"] == row["proof"]["sha256"]
            and actual["bytes"] == row["proof"]["bytes"]
            and actual["stat"] == signature(path),
            "Previously verified destination context changed",
        )
    return missing


def send_contexts(stream, rows, missing):
    require(
        missing == sorted(set(missing))
        and all(type(i) is int and 0 <= i < len(rows) for i in missing),
        "Unique ordered missing-only selection required",
    )
    for i in missing:
        row = rows[i]
        require(signature(row["path"]) == row["proof"]["stat"], "Source context changed")
        with Path(row["path"]).open("rb") as source:
            left = row["proof"]["bytes"]
            while left:
                block = source.read(min(left, MIB))
                require(block, "Original source context truncated")
                stream.write(block)
                left -= len(block)
            require(source.read(1) == b"", "Original source grew")
        require(signature(row["path"]) == row["proof"]["stat"], "Source changed during stream")
    stream.write(EOF)
    stream.flush()


def receive_contexts(stream, root, rows, missing):
    """Whole-byte verify in the owning-node stream; original provenance stays untouched."""
    transferred = 0
    for i in missing:
        row = rows[i]
        path = root / "contexts" / context_name(i)
        partial = path.with_suffix(".partial")
        require(not path.exists(), "Never replace a context")
        digest, left = hashlib.sha256(), row["proof"]["bytes"]
        with partial.open("xb") as target:
            while left:
                block = frame_read(stream, min(left, MIB))
                target.write(block)
                digest.update(block)
                left -= len(block)
            target.flush()
            os.fsync(target.fileno())
        require(digest.hexdigest() == row["proof"]["sha256"], "Transferred context SHA differs")
        os.link(partial, path)
        partial.unlink()
        once(
            path.with_suffix(".verified.json"),
            {"bytes": row["proof"]["bytes"], "sha256": digest.hexdigest(), "stat": signature(path)},
        )
        transferred += row["proof"]["bytes"]
    require(
        frame_read(stream, len(EOF)) == EOF and stream.read(1) == b"", "Exact stream EOF required"
    )
    result = copy.deepcopy(rows)
    for i, row in enumerate(result):
        path = root / "contexts" / context_name(i)
        row["path"] = str(path)
        row["proof"]["stat"] = signature(path)
        require(row["proof"]["stat"][1] == row["proof"]["bytes"], "Destination bytes differ")
        original = rows[i]
        require(
            row["provenance"] == original["provenance"]
            and {k: v for k, v in row["proof"].items() if k != "stat"}
            == {k: v for k, v in original["proof"].items() if k != "stat"},
            "Original numerical provenance cannot be rewritten",
        )
    return result, transferred


def paid_shared_logs(jobs, root=Path("/home/sa305415/rsna-knee/kaggle")):
    """Retire only original job-bound log files already charged to actual quota."""
    require(set(jobs) <= {"B", "C", "C0"}, "Unknown predictive log owner")
    paths, identities = [], set()
    for arm, job in jobs.items():
        identity = job["GPU_job"]
        name = "rsna-predictive-full2-" + arm
        require(
            re.fullmatch(r"[0-9]+", identity)
            and job["arm"] == arm
            and job["source_git"] == SCIENCE
            and job["node"] == ("evc27" if arm == "C0" else "evc33")
            and job["job_name"] == name,
            "Exact original dispatch binding required for paid logs",
        )
        total = 0
        for field, extension in (("stdout", "out"), ("stderr", "err")):
            path = root / (name + "-" + identity + "." + extension)
            require(job[field] == str(path), "Log path differs from original dispatch")
            if not path.exists():
                require(not path.is_symlink(), "Dangling log link is not a paid original")
                continue
            stat = signature(path)
            st = path.stat()
            key = (st.st_dev, st.st_ino)
            require(key not in identities, "Duplicate paid log inode")
            identities.add(key)
            total += stat[1]
            paths.append(str(path))
        require(total <= 8 * MIB, "Original per-arm combined log bound exceeded")
    require(len(paths) <= 6, "Only original six GPU logs can retire file obligations")
    return {
        "files": len(paths),
        "paths": sorted(paths),
        "originals_deleted": False,
        "basis": "Exact original dispatch logs already present in current quota",
    }


def derived_remote(source, node, shared_bytes, shared_files):
    """Only owner control literals change; worker/publisher bytes never pass here."""
    require(node in ("evc33", "evc27"), "Only accepted ordinary two-H100 placements")
    require(type(shared_bytes) is int and shared_bytes >= 123 * MIB, "Fresh shared liabilities")
    require(type(shared_files) is int and shared_files >= 376, "Fresh shared file liabilities")
    source = source.replace(
        'ARMS = ("B", "C", "C0")', 'ARMS = ("B", "C")' if node == "evc33" else 'ARMS = ("C0",)'
    )
    source = source.replace(
        '"status": "actual_three_predictive_full_jobs_submitted_once",',
        '"status": "actual_predictive_placement_jobs_submitted_once", '
        '"submitted_arm_count": len(ARMS),',
    )
    source = source.replace(
        '"physical_concurrent_GPU_limit": 2,',
        '"ordinary_node_physical_GPU_limit": 2, "placement_total_GPU_limit": 3,',
    )
    source = source.replace("base_other(v) + 36 * GIB", "base_other(v) + len(ARMS) * 12 * GIB")
    source = source.replace("SHARED_FUTURE = 121 * MIB", f"SHARED_FUTURE = {shared_bytes}")
    source = source.replace('"shared_unpaid_files": 376', f'"shared_unpaid_files": {shared_files}')
    if node == "evc27":
        source = source.replace(str(ROOT33), str(ROOT27)).replace(PY33, PY27)
        source = source.replace('"evc33"', '"evc27"').replace(
            "--nodelist=evc33", "--nodelist=evc27"
        )
        # The portable base_other includes64MiB controls minus actual paid bytes.
        source = source.replace("+ 16 * MIB\n", "+ 0\n")
    compile(source, "distinct_placement_owner_control", "exec")
    return source
