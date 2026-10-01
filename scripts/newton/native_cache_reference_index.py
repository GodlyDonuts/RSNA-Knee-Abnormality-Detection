"""Index accepted cache tar headers once; selected payload bytes still need hashing."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tarfile
from pathlib import Path, PurePosixPath

PLAN_SHA = "44e6775c5413e42d8744ead0f7d7bb0de258893a10867123e47fbd770f84793f"
ARCHIVE_BYTES = 82834493440


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def signature(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("regular_reference_archive_required")
    s = path.stat()
    return {
        "dev": s.st_dev,
        "ino": s.st_ino,
        "bytes": s.st_size,
        "mtime_ns": s.st_mtime_ns,
        "ctime_ns": s.st_ctime_ns,
    }


def planned_members(manifest_path):
    if manifest_path.is_symlink() or digest(manifest_path) != PLAN_SHA:
        raise ValueError("pinned_cohort_manifest_required")
    manifest = json.loads(manifest_path.read_text())
    if set(manifest["shards"]) != {f"shard-{i:03d}.json" for i in range(75)}:
        raise ValueError("all_cohort_shards_required")
    members, studies, series = {}, set(), set()
    for n, binding in manifest["shards"].items():
        p = manifest_path.parent / n
        if p.is_symlink() or p.stat().st_size != binding["bytes"] or digest(p) != binding["sha256"]:
            raise ValueError("immutable_shard_binding_changed")
        shard = json.loads(p.read_text())
        if shard["selection_sha256"] != manifest["selection_sha256"] or studies & set(
            shard["study_uids"]
        ):
            raise ValueError("shard_selection_or_study_membership_changed")
        studies.update(shard["study_uids"])
        for row in shard["series"]:
            path = f"train_series/{row['study_uid']}/{row['series_uid']}.npz"
            if (
                row["cache_path"] != path
                or row["study_uid"] not in shard["study_uids"]
                or row["series_uid"] in series
            ):
                raise ValueError("accepted_series_identity_changed")
            series.add(row["series_uid"])
            stem = "cache/" + path
            members[stem] = {"bytes": row["cache_bytes"], "sha256": row["cache_sha256"]}
            members[str(PurePosixPath(stem).with_suffix(".json"))] = {
                "bytes": row["cache_metadata_bytes"],
                "sha256": row["cache_metadata_sha256"],
            }
    if (
        len(series) != 18906
        or len(studies) != 3403
        or studies != set(manifest["training_uids"]) | set(manifest["outer_uids"])
    ):
        raise ValueError("full_cohort_membership_changed")
    return members


def build_index(archive, members, manifest_sha):
    before = signature(archive)
    offsets = {}
    for name, record in members.items():
        p = PurePosixPath(name)
        if (
            not name.startswith("cache/")
            or p.is_absolute()
            or ".." in p.parts
            or "\\" in name
            or p.as_posix() != name
            or p.suffix not in {".npz", ".json"}
        ):
            raise ValueError("unsafe_expected_reference_member")
        if (
            type(record["bytes"]) is not int
            or record["bytes"] < 1
            or not re.fullmatch("[0-9a-f]{64}", record["sha256"])
        ):
            raise ValueError("complete_expected_reference_bytes_required")
    with tarfile.open(archive, "r:") as tar:
        for member in tar:
            if member.name not in members:
                continue
            expected = members[member.name]
            if not member.isfile() or member.size != expected["bytes"] or member.name in offsets:
                raise ValueError("reference_header_membership_changed")
            if (
                member.offset_data < 0
                or member.offset_data % 512
                or member.offset_data + member.size > before["bytes"]
            ):
                raise ValueError("reference_member_outside_archive")
            offsets[member.name] = {**expected, "offset": member.offset_data}
    if offsets.keys() != members.keys() or signature(archive) != before:
        raise ValueError("missing_reference_member_or_changed_archive")
    return {
        "schema_version": 1,
        "status": "accepted_reference_member_headers_indexed",
        "plan_manifest_sha256": manifest_sha,
        "archive_signature": before,
        "members": offsets,
        "regular_members": len(offsets),
        "fresh_whole_archive_sha256_verified": False,
        "selected_member_bytes_verified": False,
        "target_values_or_reports_read": False,
    }


def load_offsets(archive, path, sha, expected):
    if path.is_symlink() or digest(path) != sha:
        raise ValueError("pinned_reference_index_bytes_changed")
    index = json.loads(path.read_text())
    if (
        index.get("status") != "accepted_reference_member_headers_indexed"
        or index.get("plan_manifest_sha256") != PLAN_SHA
        or index.get("archive_signature") != signature(archive)
    ):
        raise ValueError("reference_index_archive_or_plan_changed")
    if (
        index.get("regular_members") != len(index["members"])
        or index.get("fresh_whole_archive_sha256_verified") is not False
        or index.get("selected_member_bytes_verified") is not False
    ):
        raise ValueError("reference_index_scope_changed")
    offsets = {}
    for name, binding in expected.items():
        row = index["members"].get(name)
        if (
            not row
            or any(row.get(k) != v for k, v in binding.items())
            or type(row.get("offset")) is not int
            or row["offset"] < 0
            or row["offset"] % 512
            or row["offset"] + row["bytes"] > index["archive_signature"]["bytes"]
        ):
            raise ValueError("indexed_reference_member_binding_changed")
        offsets[name] = (row["offset"], row["bytes"])
    return offsets


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
        or os.environ.get("SLURM_JOB_GPUS")
    ):
        raise ValueError("one_cpu_zero_gpu_required")
    if signature(args.archive)["bytes"] != ARCHIVE_BYTES:
        raise ValueError("accepted_archive_size_changed")
    if args.output.exists():
        raise FileExistsError("existing_index_requires_readonly_review")
    index = build_index(args.archive, planned_members(args.manifest), PLAN_SHA)
    index["job_id"] = os.environ["SLURM_JOB_ID"]
    with args.output.open("x") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    print(
        json.dumps(
            {
                "status": index["status"],
                "index_sha256": digest(args.output),
                "regular_members": index["regular_members"],
                "job_id": index["job_id"],
                "fresh_whole_archive_sha256_verified": False,
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
