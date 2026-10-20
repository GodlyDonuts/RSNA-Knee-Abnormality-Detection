"""Read-only terminal/prefix/source-buffer inventory; aggregate output only."""

import csv
import gzip
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

SOURCE = "e9bdd1249fa7cf75f792b3762c2028bd7beb4337"
JOBS = {"raw": "981958", "context": "981960"}
INTENTS = {
    "raw": "7ad45b9e78c7fc987bd913356b3f91c615f9ab446f92dc626b411da154c9d27d",
    "context": "e32a42c84d2b99a8185d19d18178a77fa972665e6008940e983f15e6aac7425c",
}
ROOTS = {r: Path(f"/tmp/rsna-predictive-full-{r}-data-recovery-20261020-v1") for r in JOBS}
MIB = 2**20


def signature(path):
    s = path.lstat()
    assert stat.S_ISREG(s.st_mode) and s.st_uid == os.getuid() and not s.st_mode & 0o077
    return [s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def metadata(path, cap=4*MIB):
    before = signature(path)
    assert before[1] <= cap
    raw = path.read_bytes()
    assert signature(path) == before
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def directory(path):
    s = path.lstat()
    assert stat.S_ISDIR(s.st_mode) and s.st_uid == os.getuid() and not s.st_mode & 0o077


def scheduler():
    body = subprocess.check_output([
        "sacct", "-X", "-n", "-P", "-j", ",".join(JOBS.values()),
        "--format=JobIDRaw,State,ExitCode,NodeList,Start,End,Elapsed",
    ], text=True, env={**os.environ, "TZ": "UTC"})
    queue = subprocess.check_output(["squeue", "-h", "-u", "sa305415", "-o", "%i"], text=True)
    result = {}
    for role, job in JOBS.items():
        rows = [r.split("|") for r in body.splitlines() if r.split("|")[0] == job]
        assert len(rows) == 1 and rows[0][:4] == [job, "FAILED", "1:0", "evc33"]
        assert job not in queue.split()
        row = rows[0]
        result[role] = dict(job=job, state=row[1], exit_code=row[2], node=row[3],
                            start_UTC=row[4], end_UTC=row[5], elapsed=row[6], queue_empty=True)
    return result


def chain_entries(chain):
    levels = []
    while chain:
        assert len(levels) < 8 and set(chain) == {"raw", "context"}
        levels.append(chain)
        left = chain["raw"].get("ancestor_chain")
        right = chain["context"].get("ancestor_chain")
        assert left == right
        chain = left
    return levels


def bounded_progress(root):
    path = root / "progress.private.json"
    if not path.exists():
        return None
    value, _ = metadata(path)
    assert type(value["completed"]) is int and 0 <= value["completed"] <= 3887
    assert value["total"] == 3887 and 0 <= value["elapsed_seconds"] < 48*3600
    return {k: value[k] for k in ("completed", "total", "elapsed_seconds")}


def proof_inventory(role, roots, uids):
    initial = Path(f"/tmp/rsna-predictive-{'raw' if role == 'raw' else 'context'}16-cache-20261020-v1")
    name = "cache-index.private.json" if role == "raw" else "context-index.private.json"
    rows, _ = metadata(initial / name)
    assert len(rows) == 16
    lookup = {uid: i for i, uid in enumerate(uids)}
    complete, physical, incomplete_json = {}, {}, 0

    def accept(row, expected=None):
        i = lookup[row["uid"]]
        assert expected is None or i == expected
        path = Path(row["path"])
        assert path.is_absolute() and any(path.is_relative_to(r) for r in [initial, *roots])
        sig = signature(path)
        assert sig == row["proof"]["stat"] and sig[1] == row["proof"]["bytes"]
        identity = (str(path), row["proof"])
        assert i not in complete or complete[i] == identity
        complete[i] = identity
        s = path.stat()
        physical[(s.st_dev, s.st_ino)] = s.st_size

    for row in rows:
        accept(row)
    by_generation = []
    for root in roots:
        pattern = "proofs/case-*.verified.private.json" if role == "raw" else "context-*.verified.private.json"
        proofs = sorted(root.glob(pattern))
        assert len(proofs) <= 3887
        accepted = 0
        for path in proofs:
            match = re.fullmatch(r"(?:case|context)-(\d{4})\.verified\.private\.json", path.name)
            assert match and int(match[1]) < 3887
            try:
                row, _ = metadata(path)
            except json.JSONDecodeError:
                incomplete_json += 1
                continue
            accept(row, int(match[1]))
            accepted += 1
        caches = list((root / "cache").glob("*.npz")) if role == "raw" else list(root.glob("context-*.npz"))
        orphan_count = 0
        for path in caches:
            s = path.lstat()
            signature(path)
            if (s.st_dev, s.st_ino) not in physical:
                orphan_count += 1
            physical[(s.st_dev, s.st_ino)] = s.st_size
        by_generation.append(dict(verified_sidecars=accepted, cache_files=len(caches),
                                  orphan_or_later_generation_referenced_files=orphan_count))
    return dict(completed_distinct_studies=len(complete), pending_studies=3887-len(complete),
                cache_physical_bytes=sum(physical.values()), incomplete_verified_sidecars=incomplete_json,
                generations=by_generation, completed_cache_bodies_rehashed=False), set(complete)


def buffer_identity(root):
    directory(root)
    data = root / "data"
    directory(data)
    csv_path = data / "train.csv"
    before = signature(csv_path)
    assert before[1] <= 4096
    with csv_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["StudyInstanceUID"]
        rows = list(reader)
    assert len(rows) == 1 and signature(csv_path) == before
    return rows[0]["StudyInstanceUID"]


def classify_buffer(root, record, completed):
    data = root / "data"
    uid = record["uid"]
    expected = set()
    counts = dict(complete_size=0, partial=0, missing=0, oversize=0)
    retained, physical = 0, {}
    for row in record["source_files"]:
        path = data / "train_series" / uid / row["series_uid"] / (row["sop_uid"] + ".dcm")
        expected.add(path)
        if not path.exists():
            counts["missing"] += 1
            continue
        size = signature(path)[1]
        retained += size
        s = path.stat()
        physical[(s.st_dev, s.st_ino)] = size
        counts["complete_size" if size == row["bytes"] else "partial" if size < row["bytes"] else "oversize"] += 1
    actual = set((data / "train_series").rglob("*.dcm"))
    assert actual <= expected
    journal = root / "source-failure.private.json"
    failure = None
    if journal.exists():
        value, digest = metadata(journal)
        kind = value["error_type"]
        assert re.fullmatch(r"[A-Za-z][A-Za-z0-9_]{0,79}", kind)
        failure = {"error_type": kind, "journal_sha256": digest}
        # Historical helpers usually retain only error_type. Never infer an
        # HTTP status from its class or a later successful request.
        for key in ("status", "response_status", "attempt"):
            if type(value.get(key)) is int:
                failure[key] = value[key]
    proof = root / "source-proof.private.json"
    source_proof_present = proof.is_file()
    if source_proof_present:
        signature(proof)
    return dict(source_members=len(record["source_files"]), counts=counts,
                retained_logical_source_bytes=retained, distinct_physical_source_bytes=sum(physical.values()),
                source_failure=failure, source_proof_present=source_proof_present,
                source_size_comparison_only_not_hash_verification=True,
                study_already_has_verified_raw_cache=completed)


def main(v):
    assert v == {} and os.uname().nodename == "evc33"
    assert not os.environ.get("SLURM_JOB_GPUS") and os.environ.get("CUDA_VISIBLE_DEVICES", "") == ""
    assert os.environ["SLURM_CPUS_PER_TASK"] == "1" and sys.dont_write_bytecode
    terminals = scheduler()
    intents, role_results = {}, {}
    for role, root in ROOTS.items():
        directory(root)
        intent, digest = metadata(root / "intent.json")
        assert digest == INTENTS[role] and intent["CPU_job"] == JOBS[role] and intent["source_git"] == SOURCE
        assert not (root / "actual.json").exists()
        intents[role] = intent
        role_results[role] = dict(intent_sha256=digest, source_git=SOURCE,
                                  scheduler=terminals[role], progress=bounded_progress(root), final_present=False)
    assert intents["raw"]["continuation_chain"] == intents["context"]["continuation_chain"]
    levels = chain_entries(intents["raw"]["continuation_chain"])
    roots, ancestry = {}, []
    for role in JOBS:
        roots[role] = [Path(f"/tmp/rsna-predictive-full-{role}-cache-20261020-v1")]
    for level in reversed(levels):
        safe = {}
        for role, entry in level.items():
            root = Path(entry["root"])
            assert root.parent == Path("/tmp") and root.name.startswith("rsna-predictive-full-"+role+"-")
            intent, digest = metadata(root / "intent.json")
            assert digest == entry["intent_sha256"] and intent["CPU_job"] == entry["job"]
            assert intent["source_git"] == entry["source_git"]
            roots[role].append(root)
            safe[role] = {k: entry[k] for k in ("job", "intent_sha256", "source_git")}
        ancestry.append(safe)
    for role in JOBS:
        roots[role].append(ROOTS[role])
        assert len(roots[role]) == len(set(roots[role])) <= 8
    buffers = sorted((ROOTS["raw"] / "scratch").iterdir())
    assert len(buffers) <= 4
    wanted = {buffer_identity(path) for path in buffers}
    location, _ = metadata(Path("/tmp/rsna-S6-fitting-teacher-stream-20261018-v1/RAM-location.private.json"))
    index = roots["raw"][0] / "original-fitting-source-index.private.jsonl.gz"
    before = signature(index)
    assert intents["raw"]["original_source_index_sha256"] == location["sha256"]
    uids, selected = [], {}
    with gzip.open(index, "rt") as stream:
        while line := stream.readline(8*MIB+1):
            assert len(line) <= 8*MIB and len(uids) < 3887
            row = json.loads(line)
            uids.append(row["uid"])
            if row["uid"] in wanted:
                # The corrected1740/1850 cases already have sealed caches. A
                # surprising retained buffer there needs its effective table.
                assert len(uids)-1 not in {1740, 1850}
                selected[row["uid"]] = row
    assert len(uids) == len(set(uids)) == 3887 and set(selected) == wanted
    assert signature(index) == before
    raw_complete = None
    for role in JOBS:
        inventory, complete = proof_inventory(role, roots[role], uids)
        role_results[role]["proof_inventory"] = inventory
        if role == "raw":
            raw_complete = complete
    lookup = {uid: i for i, uid in enumerate(uids)}
    failures = [classify_buffer(path, selected[buffer_identity(path)],
                               lookup[buffer_identity(path)] in raw_complete) for path in buffers]
    physical_scratch = {}
    scratch_generations = []
    for root in roots["raw"]:
        directory(root / "scratch")
        directories = list((root / "scratch").iterdir())
        assert len(directories) <= 8
        count, size = 0, 0
        for owned in directories:
            directory(owned)
            for path in owned.rglob("*"):
                if path.is_file():
                    signature(path)
                    s = path.stat()
                    count += 1
                    size += s.st_size
                    physical_scratch[(s.st_dev, s.st_ino)] = s.st_size
                else:
                    directory(path)
        scratch_generations.append(dict(buffers=len(directories), files=count, logical_bytes=size))
    quota = subprocess.check_output(["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True)
    q = next(r.split() for r in quota.splitlines() if r.strip().startswith("/home/sa305415"))
    out = dict(status="actual_HTTP499_generation_terminal_prefix_and_buffers_readonly",
               CPU_job=os.environ["SLURM_JOB_ID"], observed_at_UTC=datetime.now(timezone.utc).isoformat(),
               roles=role_results, ancestor_intent_chain=ancestry,
               current_source_buffers=failures, retained_scratch_generations=scratch_generations,
               retained_distinct_physical_scratch_bytes=sum(physical_scratch.values()),
               original_source_index_sha256_from_prior_proof=location["sha256"],
               source_index_stat_unchanged=True, source_index_rehashed=False,
               node_free_bytes=shutil.disk_usage("/tmp").free,
               shared_quota=dict(used_kB=int(q[1]), limit_kB=int(q[2]), used_files=int(q[5]),
                                 limit_files=int(q[6]), free_bytes=(int(q[2])-int(q[1]))*1024,
                                 free_files=int(q[6])-int(q[5])),
               source_HTTP_requests=0, pixel_decodes=0, completed_cache_body_reads=0,
               labels_read=False, writes=0)
    print(json.dumps(out, allow_nan=False), flush=True)


if __name__ == "__main__":
    try:
        main(json.load(sys.stdin))
    except Exception as error:
        print(json.dumps({"status": "readonly_inventory_failed", "error_type": type(error).__name__}),
              file=sys.stderr, flush=True)
        raise SystemExit(1) from None
