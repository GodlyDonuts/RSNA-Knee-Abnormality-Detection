"""Assemble the complete accepted native384 cohort from pinned private shard receipts.

No GPU, labels, or model fitting is involved. Partial cohorts and existing output
attempts are refused; interrupted data are retained for read-only reconciliation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

REPO = "GodlyDonuts/RSNA-Data"
SCIENCE = "c9fcc7eefb09103942be89bf4fa87410eb9ec5ab"
SOURCE = "0b5e410288232f570c69ba4f44723356e6ae6ab7"
PLAN_SHA = "44e6775c5413e42d8744ead0f7d7bb0de258893a10867123e47fbd770f84793f"
WORKERS = {0: "62a879a7c97d118163b3e9769f18f691e9a971f7ee3cf0133c5130e9ad5fb173"}
DEFAULT_WORKER = "b8942961f5d79fb21f603692b6d40a528325efae60c627ff100f703e9fa0178d"
NAMES = {
    "cache-side-256.tar",
    "cache-side-384.tar",
    "cache-verified.json",
    "series-proofs.jsonl",
    "attribution.json",
    "manifest.json",
}
BASE = Path("/home/sa305415/rsna-knee")


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_bytes(data, record):
    if len(data) != record["bytes"] or hashlib.sha256(data).hexdigest() != record["sha256"]:
        raise ValueError("bound_payload_bytes_changed")
    return data


def file_bytes(path, record, maximum=None):
    if (
        path.is_symlink()
        or not path.is_file()
        or (maximum is not None and path.stat().st_size > maximum)
    ):
        raise ValueError("bounded_regular_artifact_required")
    return verify_bytes(path.read_bytes(), record)


def bind_shard(shard, receipt, report, proofs):
    """Bind existing complete byte evidence to every accepted study and series."""
    n = shard["shard"]
    prefix = (
        "experiments/dinov3-v1/native-original384-v1/selection-"
        + shard["selection_sha256"]
        + f"/shard-{n:03d}"
    )
    if (
        receipt.get("status") != "private_native_shard_full_download_size_sha_verified"
        or receipt.get("repo_id") != REPO
        or receipt.get("prefix") != prefix
        or receipt.get("shard") != n
        or not re.fullmatch("[0-9a-f]{40}", receipt.get("revision", ""))
        or set(receipt.get("files", {})) != NAMES
        or receipt.get("model_training_or_accuracy_claim") is not False
    ):
        raise ValueError("complete_exact_private_shard_receipt_required")
    for row in receipt["files"].values():
        if (
            type(row.get("bytes")) is not int
            or row["bytes"] < 1
            or not re.fullmatch("[0-9a-f]{64}", row.get("sha256", ""))
        ):
            raise ValueError("complete_private_byte_bindings_required")
    execution = receipt["execution_provenance"]
    if (
        report.get("execution_provenance") != execution
        or not re.fullmatch("[1-9][0-9]*", execution.get("job_id", ""))
        or any(
            execution.get(k) != v
            for k, v in {
                "cpus": 1,
                "gpus": 0,
                "worker_sha256": WORKERS.get(n, DEFAULT_WORKER),
                "plan_manifest_sha256": PLAN_SHA,
            }.items()
        )
    ):
        raise ValueError("accepted_source_execution_required")
    expected = {
        "schema_version": 1,
        "status": "native_paired_shard_cache_and_loader_verified",
        "selection_sha256": shard["selection_sha256"],
        "shard": n,
        "scientific_source_git": SCIENCE,
        "source_revision": SOURCE,
        "studies": len(shard["study_uids"]),
        "series": shard["series_count"],
        "slices": shard["slices"],
        "source_bytes_verified": shard["source_bytes"],
        "target_values_or_reports_read": False,
        "accepted_cache_labels_masks_folds_mutated": False,
        "full_cohort_built": False,
        "model_training_or_accuracy_claim": False,
    }
    if any(report.get(k) != v for k, v in expected.items()):
        raise ValueError("source_report_scope_changed")
    rows = {r["series_uid"]: r for r in shard["series"]}
    if (
        len(rows) != shard["series_count"]
        or len(proofs) != len(rows)
        or {p["series_uid"] for p in proofs} != rows.keys()
    ):
        raise ValueError("exact_source_proof_series_membership_required")
    if (
        {r["study_uid"] for r in rows.values()} != set(shard["study_uids"])
        or sum(r["n_slices"] for r in rows.values()) != shard["slices"]
        or sum(r["source_bytes"] for r in rows.values()) != shard["source_bytes"]
    ):
        raise ValueError("exact_source_proof_study_counts_required")
    members = {}
    for side in (256, 384):
        loader = report["loaders"][str(side)]
        if any(
            loader.get(k) != v
            for k, v in {
                "studies": len(shard["study_uids"]),
                "source_cache_side": side,
                "input_side": 224 if side == 256 else 384,
                "original_source_revision": SOURCE,
                "labels_read": False,
            }.items()
        ) or not 0 < loader["valid_sampled_slices"] <= 96 * len(shard["study_uids"]):
            raise ValueError("actual_source_loader_acceptance_changed")
        selected = {}
        for proof in proofs:
            row = rows[proof["series_uid"]]
            expected_path = f"train_series/{row['study_uid']}/{row['series_uid']}.npz"
            difference = proof["accepted256_difference"]["max_absolute_uint16_difference"]
            if (
                row["cache_path"] != expected_path
                or (proof["study_uid"], proof["slices"], proof["source_bytes"])
                != (row["study_uid"], row["n_slices"], row["source_bytes"])
                or type(difference) is not int
                or not 0 <= difference <= 1
                or not re.fullmatch("[0-9a-f]{64}", proof.get("source_membership_sha256", ""))
            ):
                raise ValueError("accepted_source_series_identity_changed")
            name = "cache/" + expected_path
            selected[name] = proof["files"][str(side)]["npz"]
            selected[str(PurePosixPath(name).with_suffix(".json"))] = proof["files"][str(side)][
                "metadata"
            ]
        archive = report["archives"][f"cache-side-{side}.tar"]
        for record in selected.values():
            if (
                type(record.get("bytes")) is not int
                or record["bytes"] < 1
                or not re.fullmatch("[0-9a-f]{64}", record.get("sha256", ""))
            ):
                raise ValueError("source_member_byte_bindings_required")
        if (
            archive["members"] != selected
            or archive["regular_members"] != len(selected)
            or len(selected) != shard["regular_members_per_side"]
            or archive["unpacked_file_bytes"] != sum(v["bytes"] for v in selected.values())
            or archive["bytes"] > shard["archive_upper_bytes_by_side"][str(side)]
            or {k: archive[k] for k in ("bytes", "sha256")}
            != receipt["files"][f"cache-side-{side}.tar"]
        ):
            raise ValueError("archive_and_private_proof_membership_changed")
        members[side] = selected
    if report["max_reference_uint16_difference"] != max(
        p["accepted256_difference"]["max_absolute_uint16_difference"] for p in proofs
    ):
        raise ValueError("source_reference_difference_changed")
    return {
        "shard": n,
        "prefix": prefix,
        "revision": receipt["revision"],
        "publication_receipt_sha256": hashlib.sha256(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "files": receipt["files"],
        "members": members[384],
        "series": shard["series"],
        "execution_provenance": execution,
    }


def read_completed_shard(shard, control):
    result = json.loads((control / "result.json").read_text())
    receipt = json.loads((control / "publication/verified.json").read_text())
    accepted = json.loads((control / "private-preservation-accepted.json").read_text())
    if accepted.get("publication_receipt_sha256") != digest(control / "publication/verified.json"):
        raise ValueError("canonical_private_acceptance_changed")
    if (
        result.get("status") != "original_source_paired_shard_and_private_preservation_complete"
        or result.get("scratch_reclaimed_after_full_private_verification") is not True
        or result.get("shard") != shard["shard"]
        or result.get("revision") != receipt["revision"]
        or result.get("job_id") != receipt["execution_provenance"]["job_id"]
    ):
        raise ValueError("completed_source_shard_required")
    report = json.loads(
        file_bytes(
            control / "cache-verified.json", receipt["files"]["cache-verified.json"], 16 * 1024**2
        )
    )
    raw = file_bytes(
        control / "series-proofs.jsonl", receipt["files"]["series-proofs.jsonl"], 16 * 1024**2
    )
    proofs = [json.loads(line) for line in raw.decode().splitlines()]
    return bind_shard(shard, receipt, report, proofs)


def freeze_registry(manifest_path, first, batch):
    if manifest_path.is_symlink() or digest(manifest_path) != PLAN_SHA:
        raise ValueError("pinned_complete_cohort_plan_required")
    plan = json.loads(manifest_path.read_text())
    if set(plan["shards"]) != {f"shard-{i:03d}.json" for i in range(75)}:
        raise ValueError("all_75_plan_shards_required")
    for lane in (0, 1):
        result = json.loads((batch / f"lane-{lane}/complete.json").read_text())
        expected = list(range(lane + 1, 75, 2))
        if (
            result.get("status") != "all_assigned_shards_privately_preserved"
            or result.get("shards") != expected
        ):
            raise ValueError("both_complete_source_lanes_required")
    shards, studies, series = [], set(), set()
    for n in range(75):
        name = f"shard-{n:03d}.json"
        path = manifest_path.parent / name
        shard = json.loads(file_bytes(path, plan["shards"][name]))
        if shard["shard"] != n or any(
            shard[k] != plan[k]
            for k in (
                "scientific_source_git",
                "source_revision",
                "selection_sha256",
                "input_bindings",
            )
        ):
            raise ValueError("immutable_shard_selection_changed")
        if (
            set(shard["study_uids"]) & studies
            or {r["series_uid"] for r in shard["series"]} & series
        ):
            raise ValueError("overlapping_source_shards")
        studies.update(shard["study_uids"])
        series.update(r["series_uid"] for r in shard["series"])
        row = read_completed_shard(shard, first if n == 0 else batch / f"shard-{n:03d}")
        if row["execution_provenance"]["shard_file_sha256"] != plan["shards"][name]["sha256"]:
            raise ValueError("source_execution_plan_changed")
        shards.append(row)
    if (
        len(studies) != 3403
        or len(series) != 18906
        or sum(len(x["members"]) for x in shards) != 37812
        or studies != set(plan["training_uids"]) | set(plan["outer_uids"])
        or len(plan["training_uids"]) != 2247
        or len(plan["outer_uids"]) != 1156
    ):
        raise ValueError("full_development_cohort_membership_changed")
    return {
        "schema_version": 1,
        "status": "all_75_original_source_shards_privately_verified",
        "plan_manifest_sha256": PLAN_SHA,
        "selection_sha256": plan["selection_sha256"],
        "scientific_source_git": SCIENCE,
        "source_revision": SOURCE,
        "repo_id": REPO,
        "cache_side": 384,
        "study_count": len(studies),
        "series_count": len(series),
        "training_uids": plan["training_uids"],
        "outer_uids": plan["outer_uids"],
        "target_values_read": False,
        "full_cache_restored": False,
        "shards": shards,
    }


def restore_archive(path, record, members, output):
    if (
        path.is_symlink()
        or path.stat().st_size != record["bytes"]
        or digest(path) != record["sha256"]
    ):
        raise ValueError("downloaded_archive_bytes_changed")
    seen = set()
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            name = PurePosixPath(member.name)
            if (
                not member.isfile()
                or name.is_absolute()
                or ".." in name.parts
                or "\\" in member.name
                or name.as_posix() != member.name
                or not member.name.startswith("cache/train_series/")
                or member.name not in members
                or member.name in seen
                or member.size != members[member.name]["bytes"]
            ):
                raise ValueError("unsafe_or_changed_native_member")
            seen.add(member.name)
            destination = output / name
            destination.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
            if any(parent.is_symlink() for parent in (destination, *destination.parents)):
                raise ValueError("linked_cache_destination")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("unreadable_native_member")
            h = hashlib.sha256()
            with stream, destination.open("xb") as target:
                for block in iter(lambda: stream.read(4 * 1024**2), b""):
                    target.write(block)
                    h.update(block)
            if h.hexdigest() != members[member.name]["sha256"]:
                raise ValueError("restored_native_member_bytes_changed")
    if seen != members.keys():
        raise ValueError("missing_native_member")
    return len(seen)


def check_metadata(root, series):
    for row in series:
        volume = root / "cache" / row["cache_path"]
        meta = json.loads(volume.with_suffix(".json").read_text())
        if any(
            meta.get(k) != v
            for k, v in {
                "study_uid": row["study_uid"],
                "series_uid": row["series_uid"],
                "cache_path": row["cache_path"],
                "split": "train_series",
                "side": 384,
                "source_revision": SOURCE,
                "cache_version": "series-u16-v3",
                "n_slices": row["n_slices"],
                "cache_bytes": volume.stat().st_size,
                "cache_sha256": digest(volume),
            }.items()
        ):
            raise ValueError("restored_cache_metadata_binding_changed")
    return len(series)


def download(url, token, path, record, get):
    h, size = hashlib.sha256(), 0
    with get(
        url, headers={"Authorization": "Bearer " + token}, stream=True, timeout=(20, 180)
    ) as response:
        response.raise_for_status()
        with path.open("xb") as target:
            for block in response.iter_content(4 * 1024**2):
                size += len(block)
                if size > record["bytes"]:
                    raise ValueError("private_download_exceeds_bound")
                target.write(block)
                h.update(block)
    if size != record["bytes"] or h.hexdigest() != record["sha256"]:
        raise ValueError("private_restoration_download_changed")


def restore(registry, output, token, api, url, get):
    if output.exists() or output.is_symlink():
        raise FileExistsError("existing_native_assembly_requires_readonly_review")
    output.mkdir(mode=0o700)
    save_new(
        output / "intent.json",
        {
            "status": "native384_restore_started",
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "registry_sha256": hashlib.sha256(
                json.dumps(registry, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest(),
            "automatic_retry": False,
        },
    )
    for row in registry["shards"]:
        if api.repo_info(REPO, repo_type="dataset", revision=row["revision"]).private is not True:
            raise ValueError("private_pinned_shard_revision_required")
        path = output / f"shard-{row['shard']:03d}.tar"
        download(
            url(
                REPO,
                row["prefix"] + "/cache-side-384.tar",
                repo_type="dataset",
                revision=row["revision"],
            ),
            token,
            path,
            row["files"]["cache-side-384.tar"],
            get,
        )
        count = restore_archive(path, row["files"]["cache-side-384.tar"], row["members"], output)
        checked = check_metadata(output, row["series"])
        save_new(
            output / f"shard-{row['shard']:03d}-restored.json",
            {
                "shard": row["shard"],
                "revision": row["revision"],
                "members": count,
                "series": checked,
                "archive": row["files"]["cache-side-384.tar"],
                "all_restored_bytes_verified": True,
            },
        )
        path.unlink()
        print(json.dumps({"native384_shard_restored": row["shard"], "members": count}), flush=True)
    files = list((output / "cache").rglob("*"))
    if sum(p.is_file() for p in files) != 37812:
        raise ValueError("full_restored_cohort_file_count_changed")
    acceptance = {
        "status": "full_native384_cohort_bytes_metadata_restored",
        "studies": 3403,
        "series": 18906,
        "members": 37812,
        "target_values_read": False,
        "cache_side": 384,
        "fresh_gpu_or_accuracy_acceptance": False,
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "node": os.uname().nodename,
        "cache_root": str(output / "cache"),
    }
    save_new(output / "accepted.json", acceptance)
    return acceptance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    freeze = commands.add_parser("registry")
    freeze.add_argument("--manifest", type=Path, required=True)
    freeze.add_argument("--first-control", type=Path, required=True)
    freeze.add_argument("--batch-control", type=Path, required=True)
    freeze.add_argument("--output", type=Path, required=True)
    unpack = commands.add_parser("restore")
    unpack.add_argument("--registry", type=Path, required=True)
    unpack.add_argument("--registry-sha256", required=True)
    unpack.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
    ):
        raise ValueError("one_cpu_zero_gpu_required")
    if args.command == "registry":
        save_new(
            args.output, freeze_registry(args.manifest, args.first_control, args.batch_control)
        )
        return
    if digest(args.registry) != args.registry_sha256:
        raise ValueError("exact_native_registry_required")
    registry = json.loads(args.registry.read_text())
    if (
        registry.get("status") != "all_75_original_source_shards_privately_verified"
        or registry.get("plan_manifest_sha256") != PLAN_SHA
        or len(registry["shards"]) != 75
    ):
        raise ValueError("full_private_cohort_registry_required")
    if args.output.parent.is_symlink() or args.output.parent.stat().st_dev == BASE.stat().st_dev:
        raise ValueError("node_local_native_assembly_required")
    needed = sum(sum(x["bytes"] for x in row["members"].values()) for row in registry["shards"])
    needed += max(row["files"]["cache-side-384.tar"]["bytes"] for row in registry["shards"])
    if shutil.disk_usage(args.output.parent).free < needed + 20 * 1024**3:
        raise ValueError("full_native384_node_local_disk_required")
    payload = json.load(sys.stdin)
    if set(payload) != {"token"} or not payload["token"].startswith("hf_"):
        raise ValueError("private_credential_stdin_required")
    import requests
    from huggingface_hub import HfApi, hf_hub_url

    try:
        result = restore(
            registry,
            args.output,
            payload["token"],
            HfApi(token=payload["token"]),
            hf_hub_url,
            requests.get,
        )
        print(json.dumps(result))
    finally:
        payload.clear()


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print(
            json.dumps(
                {
                    "status": "native_assembly_failed",
                    "error_type": type(error).__name__,
                    "automatic_retry": False,
                }
            ),
            flush=True,
        )
        raise SystemExit(2) from None
