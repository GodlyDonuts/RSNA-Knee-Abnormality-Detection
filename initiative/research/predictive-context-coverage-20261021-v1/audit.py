"""Read only existing fitting-cache integer metadata; never decode MRI or labels."""

import ast
import hashlib
import json
import os
import statistics
import struct
import time
import zipfile
from pathlib import Path

INDEX = Path(
    "/tmp/rsna-predictive-full-raw-http499-recovery-20261020-v1/full-cache-index.private.json"
)
INDEX_SHA = "4cbabc6b954d16cc4f9d764b03be3fdef2c4dfc60f396f9d50bf4eb3d64626e4"
ROSTER_SHA = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"


def signature(p):
    s = p.stat()
    return [s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def integers(raw):
    assert raw[:6] == b"\x93NUMPY"
    major, minor = raw[6:8]
    assert (major, minor) in ((1, 0), (2, 0))
    width, fmt = (2, "<H") if major == 1 else (4, "<I")
    n = struct.unpack(fmt, raw[8 : 8 + width])[0]
    offset = 8 + width + n
    header = ast.literal_eval(raw[8 + width : offset].decode("latin1"))
    assert header["descr"] == "<i8" and header["fortran_order"] is False
    shape = header["shape"]
    assert len(shape) == 1 and 0 < shape[0] <= 10000
    assert len(raw) == offset + shape[0] * 8
    values = struct.unpack("<" + "q" * shape[0], raw[offset:])
    assert min(values) >= 0
    return values


def centers(count, cap):
    """Exact nearest-integer uniform spacing; even caps have no half ties."""
    assert 0 < count <= 10000 and cap in (24, 96)
    if count <= cap:
        return list(range(count))
    return [(2 * i * (count - 1) + cap - 1) // (2 * (cap - 1)) for i in range(cap)]


def coverage(groups, cap):
    roster, seen, neighbors = [], set(), {}
    for g, (aliases, acquisition, ordinal) in enumerate(groups):
        assert len(aliases) == len(acquisition) == len(ordinal)
        assert not (set(aliases) & seen)
        seen.update(aliases)
        first, by_acq = {}, {}
        for i, alias in enumerate(aliases):
            first.setdefault(alias, i)
            by_acq.setdefault(acquisition[i], []).append(i)
        for i in first.values():
            roster.append((acquisition[i], ordinal[i], g, i))
        for rows in by_acq.values():
            rows.sort(key=lambda i: ordinal[i])
            for pos, i in enumerate(rows):
                neighbors[g, i] = tuple(
                    rows[max(0, min(len(rows) - 1, pos + d))] for d in (-1, 0, 1)
                )
    roster.sort()
    assert len(set(x[:2] for x in roster)) == len(roster)
    chosen = [roster[i] for i in centers(len(roster), cap)]
    covered = set()
    for _, _, g, i in chosen:
        covered.update(groups[g][0][j] for j in neighbors[g, i])
    all_acq, selected_acq = {x[0] for x in roster}, {x[0] for x in chosen}
    return dict(
        unique_frames=len(roster),
        centers=len(chosen),
        triplet_unique_frames=len(covered),
        acquisitions=len(all_acq),
        acquisitions_without_center=len(all_acq - selected_acq),
        center_fraction=len(chosen) / len(roster),
        triplet_fraction=len(covered) / len(roster),
    )


def summary(rows):
    out = dict(
        studies=len(rows),
        unique_frames=sum(r["unique_frames"] for r in rows),
        center_frames=sum(r["centers"] for r in rows),
        triplet_unique_frames=sum(r["triplet_unique_frames"] for r in rows),
        studies_with_missed_acquisition=sum(r["acquisitions_without_center"] > 0 for r in rows),
        missed_acquisitions=sum(r["acquisitions_without_center"] for r in rows),
        acquisitions=sum(r["acquisitions"] for r in rows),
    )
    for name in ("center_fraction", "triplet_fraction"):
        values = sorted(r[name] for r in rows)
        out[name] = dict(
            mean=statistics.mean(values),
            median=statistics.median(values),
            minimum=values[0],
            p10=values[int(0.1 * (len(values) - 1))],
            p90=values[int(0.9 * (len(values) - 1))],
            maximum=values[-1],
        )
    return out


def main():
    assert os.uname().nodename == "evc33" and not os.environ.get("SLURM_JOB_GPUS")
    start = time.monotonic()
    body = INDEX.read_bytes()
    assert hashlib.sha256(body).hexdigest() == INDEX_SHA
    rows = json.loads(body)
    assert len(rows) == 3887
    assert (
        hashlib.sha256(
            json.dumps([r["uid"] for r in rows], separators=(",", ":")).encode()
        ).hexdigest()
        == ROSTER_SHA
    )
    measurements = {24: [], 96: []}
    metadata_bytes = 0
    for row in rows:
        p = Path(row["path"])
        s = p.lstat()
        assert (
            p.is_file() and not p.is_symlink() and s.st_uid == os.getuid() and not s.st_mode & 0o077
        )
        assert signature(p) == row["proof"]["stat"]
        groups = []
        with zipfile.ZipFile(p) as z:
            for g in range(row["proof"]["groups"]):
                arrays = []
                for name in ("aliases", "acquisition", "ordinal"):
                    raw = z.read(f"g{g:04d}_{name}.npy")
                    metadata_bytes += len(raw)
                    arrays.append(integers(raw))
                groups.append(arrays)
        assert signature(p) == row["proof"]["stat"]
        for cap in measurements:
            measurements[cap].append(coverage(groups, cap))
    result = dict(
        status="actual_original_fitting_cache_context_coverage_readonly",
        CPU_job=os.environ["SLURM_JOB_ID"],
        observed_at=time.time(),
        index_sha256=INDEX_SHA,
        fitting_roster_sha256=ROSTER_SHA,
        studies=3887,
        metadata_array_bytes=metadata_bytes,
        context_caps={str(c): summary(v) for c, v in measurements.items()},
        elapsed_seconds=time.monotonic() - start,
        pixel_or_descriptor_reads=0,
        labels_read=0,
        new_GPU_jobs=0,
        model_inference=0,
        clinical_accuracy_claim=False,
        shared_files_created=0,
        private_case_ids_collected=False,
    )
    print(json.dumps(result, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
