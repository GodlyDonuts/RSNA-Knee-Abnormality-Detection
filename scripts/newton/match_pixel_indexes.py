"""Private CPU advisory matching of two original pixel-only source indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

from rsna.pixel_overlap import overlap_candidates


def read_exact_index(stream, expected_bytes, expected_sha256):
    if not 0 < expected_bytes <= 64 * 1024**2:
        raise ValueError("Bounded exact private metadata required")
    result = bytearray()
    digest = hashlib.sha256()
    remaining = expected_bytes
    while remaining:
        chunk = stream.read(min(4 * 1024**2, remaining))
        if not chunk:
            raise ValueError("Incomplete original private index stream")
        digest.update(chunk)
        result.extend(chunk)
        remaining -= len(chunk)
    if stream.read(1) or digest.hexdigest() != expected_sha256:
        raise ValueError("Extra or changed original metadata bytes")
    return result


def save_private_candidates(path, pairs, cap_bytes=64 * 1024**2):
    size = 0
    with path.open("xb") as stream:
        for chunk in json.JSONEncoder(sort_keys=True).iterencode(pairs):
            encoded = chunk.encode()
            if size + len(encoded) > cap_bytes:
                raise ValueError("Private candidate retention cap exceeded")
            stream.write(encoded)
            size += len(encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    control = args.control
    source = json.loads((control / "source.json").read_text())
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
        or os.uname().nodename != source["left_node"]
    ):
        raise ValueError("Private one-CPU zero-GPU original left-owning-node match required")
    started = time.monotonic()
    raw_left = Path(source["left_index_path"]).read_bytes()
    right = read_exact_index(
        sys.stdin.buffer, source["right_index_bytes"], source["right_index_sha256"]
    )
    if (
        len(raw_left) != source["left_index_bytes"]
        or hashlib.sha256(raw_left).hexdigest() != source["left_index_sha256"]
    ):
        raise ValueError("Changed original private Stanford metadata bytes")
    left_rows = [json.loads(line) for line in raw_left.splitlines()]
    right_rows = [json.loads(line) for line in right.splitlines()]
    if (
        len(left_rows) != 3750
        or len({r["study"] for r in left_rows}) != 1250
        or len(right_rows) != 18906
        or len({r["study"] for r in right_rows}) != 3403
    ):
        raise ValueError("Original complete source pixel-only memberships required")
    pairs = overlap_candidates(left_rows, right_rows, max_hamming=3)
    if shutil.disk_usage("/tmp").free < 20 * 1024**3 + 64 * 1024**2:
        raise ValueError("Private advisory results must preserve local disk reserve")
    directory = Path(tempfile.mkdtemp(prefix="rsna-private-pixel-matches-retained-", dir="/tmp"))
    output = directory / "candidates.json"
    save_private_candidates(output, pairs)
    receipt = {
        "status": "private_coarse_pixel_match_completed_original_review_required",
        "job_id": os.environ["SLURM_JOB_ID"],
        "node": os.uname().nodename,
        "source_git": source["source_git"],
        "left_index_sha256": source["left_index_sha256"],
        "right_index_sha256": source["right_index_sha256"],
        "Stanford_volumes": len(left_rows),
        "RSNA_volumes": len(right_rows),
        "candidate_volume_pairs": len(pairs),
        "Stanford_candidate_exams": len({p["left"]["study"] for p in pairs}),
        "RSNA_candidate_exams": len({p["right"]["study"] for p in pairs}),
        "private_candidates_path": str(output),
        "private_candidates_bytes": output.stat().st_size,
        "private_candidates_sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
        "max_hamming": 3,
        "minimum_distinct_slices_and_patterns_in_both": 3,
        "elapsed_s": time.monotonic() - started,
        "independent_patients_or_absence_of_overlap_proven": False,
        "clinical_labels_read": False,
        "protected_RSNA_cohorts_read": False,
        "raw_pixels_or_private_signatures_published": False,
    }
    with (control / "match-verified.json").open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(receipt), flush=True)


if __name__ == "__main__":
    main()
