"""CPU-only survey of accepted training geometry; never extracts images or labels."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import tarfile
from collections import Counter
from pathlib import Path


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def signature(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Immutable regular cache archive required")
    s = path.stat()
    return dict(
        dev=s.st_dev, ino=s.st_ino, bytes=s.st_size, mtime_ns=s.st_mtime_ns, ctime_ns=s.st_ctime_ns
    )


def coordinates(metadata):
    count = metadata["n_slices"]
    if metadata.get("ordering") != "patient_geometry":
        return None
    basis, points = metadata.get("orientation", []), metadata.get("positions", [])
    if len(basis) != 6 or len(points) != count or any(len(p) != 3 for p in points):
        raise ValueError("Incomplete physical coordinates")
    if not all(math.isfinite(x) for x in basis + list(itertools.chain.from_iterable(points))):
        raise ValueError("Nonfinite physical coordinates")
    row, col = basis[:3], basis[3:]

    def dot(a, b):
        return sum(x * y for x, y in zip(a, b))

    if any(
        abs(x - y) > 1e-3
        for x, y in zip(
            [math.sqrt(dot(row, row)), math.sqrt(dot(col, col)), dot(row, col)], [1, 1, 0]
        )
    ):
        raise ValueError("Invalid orientation basis")
    normal = [
        row[1] * col[2] - row[2] * col[1],
        row[2] * col[0] - row[0] * col[2],
        row[0] * col[1] - row[1] * col[0],
    ]
    norm = math.sqrt(dot(normal, normal))
    projected = [dot(p, normal) / norm for p in points]
    if any(b <= a for a, b in zip(projected, projected[1:])):
        raise ValueError("Physical coordinates must follow accepted increasing order")
    return projected


def survey(request, archive, proof, run):
    if signature(archive) != proof["assets"][str(archive)]["stat"]:
        raise ValueError("Archive differs from full-byte-verified CPU acceptance")
    if (
        request["training_uids_sha256"] != run["binding"]["training_uids_sha256"]
        or request["training_uids_sha256"] != digest(sorted(request["training_uids"]))
        or len(request["training_uids"]) != run["training_studies"]
        or len(set(request["training_uids"])) != len(request["training_uids"])
        or request["source_revision"] != proof["data"]["release_binding"]["source_revision"]
        or request["series_table_sha256"] != run["binding"]["release"]["series_sha256"]
    ):
        raise ValueError("Request differs from actual accepted training membership/source")
    rows = request["series"]
    expected = {f"cache/train_series/{r['study_uid']}/{r['series_uid']}.json": r for r in rows}
    if len(expected) != len(rows) or set(r["study_uid"] for r in rows) != set(
        request["training_uids"]
    ):
        raise ValueError("Duplicate series or incomplete/foreign training membership")
    counts, by_plane, studies, gaps, identities = Counter(), {}, set(), [], hashlib.sha256()
    seen = set()
    with tarfile.open(archive, "r:") as source:
        for member in source:
            if member.name not in expected:
                continue
            if member.name in seen or not member.isfile() or member.size > 2 * 1024**2:
                raise ValueError("Invalid/duplicate selected metadata member")
            raw = source.extractfile(member).read()
            meta, row = json.loads(raw), expected[member.name]
            count = row["n_slices"]
            if (
                meta["study_uid"] != row["study_uid"]
                or meta["series_uid"] != row["series_uid"]
                or meta["source_revision"] != request["source_revision"]
                or meta["side"] != 256
                or meta["n_slices"] != count
                or meta["cache_path"]
                != member.name.removeprefix("cache/").removesuffix(".json") + ".npz"
            ):
                raise ValueError("Selected metadata identity/source/count mismatch")
            seen.add(member.name)
            identities.update(member.name.encode() + b"\0" + hashlib.sha256(raw).digest())
            plane = by_plane.setdefault(row["plane"], Counter())
            counts["selected_series"] += 1
            plane["selected_series"] += 1
            projected = coordinates(meta)
            if projected is None:
                counts["unknown_geometry_series"] += 1
                plane["unknown_geometry_series"] += 1
                continue
            counts["physical_geometry_series"] += 1
            plane["physical_geometry_series"] += 1
            indices = row["uniform_indices"]
            if (
                indices != sorted(set(indices))
                or not indices
                or indices[0] < 0
                or indices[-1] >= count
            ):
                raise ValueError("Invalid exact uniform selected slice indices")
            if len(indices) < 2:
                counts["single_slice_series"] += 1
                continue
            studies.add(row["study_uid"])
            counts["eligible_series"] += 1
            plane["eligible_series"] += 1
            distances = [projected[j] - projected[i] for i, j in itertools.combinations(indices, 2)]
            gaps.extend(distances)
            counts["possible_uniform_pairs"] += len(distances)
            counts["capped_pairs_per_epoch_upper_bound"] += min(32, len(distances))
            plane["possible_uniform_pairs"] += len(distances)
            if len({round(x, 6) for x in distances}) < 2:
                counts["series_without_shuffled_distance_information"] += 1
    if seen != set(expected) or signature(archive) != proof["assets"][str(archive)]["stat"]:
        raise ValueError("Selected metadata missing or cache changed during survey")
    return {
        "status": "actual_training_only_metadata_geometry_survey_completed",
        "training_studies": len(request["training_uids"]),
        "training_uids_sha256": request["training_uids_sha256"],
        "studies_with_eligible_relations": len(studies),
        "counts": dict(counts),
        "by_plane": {k: dict(v) for k, v in by_plane.items()},
        "unsigned_uniform_pair_mm": {
            "minimum": min(gaps),
            "maximum": max(gaps),
            "median": statistics.median(gaps),
            "deciles": statistics.quantiles(gaps, n=10),
        }
        if len(gaps) > 1
        else None,
        "metadata_member_order_digest_sha256": identities.hexdigest(),
        "cache_archive_sha256_previously_verified": proof["assets"][str(archive)]["sha256"],
        "cache_stat_unchanged_before_after": True,
        "reports_or_target_values_read": False,
        "outer_geometry_read": False,
        "source_images_decoded": False,
        "clinical_accuracy_claim": False,
        "sampling_scope": "all possible pairs on exact fixed uniform sixteen-slice grid; not actual stratified epoch replay",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("request", "archive", "proof", "run", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    result = survey(
        request, args.archive, json.loads(args.proof.read_text()), json.loads(args.run.read_text())
    )
    result["request_sha256"] = hashlib.sha256(args.request.read_bytes()).hexdigest()
    result["survey_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
