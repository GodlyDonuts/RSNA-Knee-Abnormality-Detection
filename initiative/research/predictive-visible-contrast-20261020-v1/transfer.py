"""One bounded private node-to-node pilot context stream; no pixels reach the Mac."""

import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

SOURCE = Path("/tmp/rsna-predictive-visible-contrast16-20261020-v1")
DESTINATION = Path("/tmp/rsna-predictive-contrast-learning-20261020-v1")
ORIGINAL = Path("/tmp/rsna-predictive-stable-warmup-20261020-v1")
ORIGINAL_BINDING_SHA = "43304b78b34b407f55088d4f8e610af8d1cbb112a9960f75311689a0078a75cc"
GIB, MIB = 2**30, 2**20
CACHE_CAP, CONTROL_CAP, CHECKPOINT_CAP = GIB, 4 * MIB, 6 * GIB
# Entire all-evc27 requirement already includes the20GiB floor and all optional
# C100/G100 scopes. None is reassigned to this new independent experiment.
OLD_NODE_REQUIREMENT = 394815078400


def sha(body):
    return hashlib.sha256(body).hexdigest()


def signature(path):
    st = path.stat()
    assert path.is_file() and not path.is_symlink() and st.st_uid == os.getuid()
    return [st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def once(path, value):
    with path.open("x") as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def capacity(free, quota):
    required = OLD_NODE_REQUIREMENT + CACHE_CAP + CONTROL_CAP + CHECKPOINT_CAP
    row = next(line.split() for line in quota.splitlines() if line.strip().startswith("/home/sa305415"))
    free_shared = (int(row[2]) - int(row[1])) * 1024
    free_files = int(row[6]) - int(row[5])
    assert free > required and free_shared > 20 * GIB + 123 * MIB and free_files >= 5000 + 376
    return {"free_bytes_before": free, "old_complete_requirement_including_floor": OLD_NODE_REQUIREMENT,
            "new_cache_cap_bytes": CACHE_CAP, "new_checkpoint_cap_bytes": CHECKPOINT_CAP,
            "new_control_cap_bytes": CONTROL_CAP, "total_required_bytes": required,
            "margin_after_all_liabilities_bytes": free - required,
            "shared_free_after_old_liabilities_bytes": free_shared - 123 * MIB,
            "shared_files_after_old_liabilities": free_files - 376,
            "new_shared_bytes_or_files": 0, "no_prior_scope_reassigned": True}


def write_header(stream, value):
    body = json.dumps(value, sort_keys=True, allow_nan=False).encode() + b"\n"
    assert len(body) < 8192
    stream.write(body)


def read_header(stream):
    line = stream.readline(8192)
    assert line.endswith(b"\n") and len(line) < 8192
    return json.loads(line)


def read_exact(stream, size):
    assert type(size) is int and 0 <= size <= CONTROL_CAP
    result = bytearray()
    while len(result) < size:
        part = stream.read(min(MIB, size - len(result)))
        assert part, "Truncated private control body"
        result.extend(part)
    return bytes(result)


def roles(index):
    assert len(index) == 16 and len({row["uid"] for row in index}) == 16
    for i, row in enumerate(index):
        assert set(row["variants"]) == {"control", "normalized"}
        for role in ("control", "normalized"):
            item = row["variants"][role]
            assert item["provenance"]["role"] == role and item["provenance"]["metadata_width"] == 10
            yield i, role, item


def send(v):
    assert os.uname().nodename == "evc33"
    actual_body = (SOURCE / "actual.json").read_bytes()
    actual = json.loads(actual_body)
    assert sha(actual_body) == v["remote_actual_sha256"]
    assert actual == v["accepted"] and actual["CPU_job"] == "984645"
    body = (SOURCE / "context-index.private.json").read_bytes()
    assert sha(body) == actual["private_index_sha256"]
    index = json.loads(body)
    members = list(roles(index))
    assert sum(item["proof"]["bytes"] for _, _, item in members) == actual["cache_bytes"] <= CACHE_CAP
    output = sys.stdout.buffer
    write_header(output, {"kind": "original_index", "bytes": len(body), "sha256": sha(body)})
    output.write(body)
    for i, role, item in members:
        path = SOURCE / f"{role}-{i:03d}.private.npz"
        assert item["path"] == str(path) and signature(path) == item["proof"]["stat"]
        assert path.stat().st_size == item["proof"]["bytes"]
        write_header(output, {"kind": "context", "ordinal": i, "role": role,
                              "bytes": item["proof"]["bytes"], "sha256": item["proof"]["sha256"]})
        with path.open("rb") as stream:
            while chunk := stream.read(4 * MIB):
                output.write(chunk)
        assert signature(path) == item["proof"]["stat"]
    output.flush()


def receive(v):
    assert os.uname().nodename == "evc27" and not DESTINATION.exists()
    assert v["accepted"]["CPU_job"] == "984645" and v["accepted"]["all_roundtrips_bitwise_equal"]
    old_body = (ORIGINAL / "binding.private.json").read_bytes()
    assert sha(old_body) == ORIGINAL_BINDING_SHA
    old = json.loads(old_body)
    original_context_bytes = Path(old["context_index_private_path"]).read_bytes()
    assert sha(original_context_bytes) == old["context_index_sha256"]
    old_context = json.loads(original_context_bytes)
    funding = capacity(shutil.disk_usage("/tmp").free, subprocess.check_output(
        ["lfs", "quota", "-u", "sa305415", "/home/sa305415"], text=True))
    source = sys.stdin.buffer
    header = read_header(source)
    assert header["kind"] == "original_index" and header["sha256"] == v["accepted"]["private_index_sha256"]
    body = read_exact(source, header["bytes"])
    assert sha(body) == header["sha256"]
    index = json.loads(body)
    members = list(roles(index))
    assert [row["uid"] for row in index] == [row["uid"] for row in old_context]
    assert sum(row["proof"]["bytes"] for _, _, row in members) == v["accepted"]["cache_bytes"] <= CACHE_CAP
    DESTINATION.mkdir(mode=0o700)
    (DESTINATION / "contexts").mkdir(mode=0o700)
    once(DESTINATION / "transfer-intent.json", {"CPU_job": os.environ["SLURM_JOB_ID"],
         "source_git": v["source_git"], "funding": funding, "source_actual_sha256": v["remote_actual_sha256"],
         "original_binding_sha256": ORIGINAL_BINDING_SHA, "automatic_retry": False})
    copied, total = copy.deepcopy(index), 0
    for i, role, item in members:
        header = read_header(source)
        assert header == {"kind": "context", "ordinal": i, "role": role,
                          "bytes": item["proof"]["bytes"], "sha256": item["proof"]["sha256"]}
        assert type(header["bytes"]) is int and 0 < header["bytes"] <= 128 * MIB
        assert total + header["bytes"] <= CACHE_CAP
        path = DESTINATION / "contexts" / f"{role}-{i:03d}.private.npz"
        digest, remaining = hashlib.sha256(), header["bytes"]
        with path.open("xb") as output:
            while remaining:
                chunk = source.read(min(4 * MIB, remaining))
                assert chunk, "Truncated original context stream; retain partial output"
                output.write(chunk)
                digest.update(chunk)
                remaining -= len(chunk)
        assert digest.hexdigest() == header["sha256"]
        proof = {**item["proof"], "stat": signature(path)}
        copied[i]["variants"][role] = {**item, "path": str(path), "proof": proof}
        total += header["bytes"]
    assert not source.read(1), "Unexpected trailing bytes after32 original contexts"
    once(DESTINATION / "context-index.private.json", copied)
    once(DESTINATION / "source-context-acceptance.json", v["accepted"])
    result = {"status": "actual_contrast_matched16_private_transfer_accepted",
              "CPU_job": os.environ["SLURM_JOB_ID"], "source_git": v["source_git"],
              "source_CPU_job": "984645", "context_files": 32, "bytes": total,
              "source_actual_sha256": v["remote_actual_sha256"],
              "source_index_sha256": v["accepted"]["private_index_sha256"],
              "destination_index_sha256": sha((DESTINATION / "context-index.private.json").read_bytes()),
              "original_binding_sha256": ORIGINAL_BINDING_SHA, "all_original_sizes_SHA_EOF_verified": True,
              "original_target_row_order_equal": True, "funding": funding,
              "pixels_IDs_or_private_index_exported_to_Mac_or_Git": False,
              "GPU_dispatched_or_training_accepted": False}
    once(DESTINATION / "transfer-accepted.json", result)
    print(json.dumps(result, sort_keys=True, allow_nan=False), flush=True)


if __name__ == "__main__":
    os.umask(0o077)
    assert not os.environ.get("SLURM_JOB_GPUS") and os.environ.get("CUDA_VISIBLE_DEVICES") == ""
    assert sys.dont_write_bytecode and os.environ["SLURM_CPUS_PER_TASK"] == "1"
    mode, payload = sys.argv[1:]
    assert mode in ("send", "receive")
    (send if mode == "send" else receive)(json.loads(payload))
