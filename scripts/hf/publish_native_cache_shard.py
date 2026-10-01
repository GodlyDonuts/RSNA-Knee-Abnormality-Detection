"""Preserve one verified original-DICOM cache shard privately, exactly once.

Archives stay on owned cluster scratch. Verification streams complete remote
bytes; no large local Hugging Face cache or credentials file is created.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tarfile
from pathlib import Path, PurePosixPath

REPO = "GodlyDonuts/RSNA-Data"
SCIENCE = "c9fcc7eefb09103942be89bf4fa87410eb9ec5ab"
SOURCE = "0b5e410288232f570c69ba4f44723356e6ae6ab7"
PLAN_SHA = "44e6775c5413e42d8744ead0f7d7bb0de258893a10867123e47fbd770f84793f"
WORKER_SHA = "62a879a7c97d118163b3e9769f18f691e9a971f7ee3cf0133c5130e9ad5fb173"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def save_new(path: Path, record: dict) -> None:
    with path.open("x") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def verify_file(path: Path, record: dict) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != record["bytes"]
        or digest(path) != record["sha256"]
    ):
        raise ValueError("artifact_bytes_changed:" + path.name)


def verify_archive(path: Path, record: dict, expected_members: dict) -> None:
    verify_file(path, record)
    if (
        record["members"] != expected_members
        or record["regular_members"] != len(expected_members)
        or record["unpacked_file_bytes"] != sum(x["bytes"] for x in expected_members.values())
    ):
        raise ValueError("archive_proof_membership_changed")
    seen = set()
    with tarfile.open(path, "r:") as archive:
        for member in archive:
            name = member.name
            parsed = PurePosixPath(name)
            if (
                not member.isfile()
                or parsed.is_absolute()
                or ".." in parsed.parts
                or parsed.as_posix() != name
                or "\\" in name
                or not name.startswith("cache/")
                or parsed.suffix not in {".npz", ".json"}
                or name in seen
                or name not in expected_members
                or member.size != expected_members[name]["bytes"]
            ):
                raise ValueError("unsafe_or_changed_archive_member")
            seen.add(name)
            h = hashlib.sha256()
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError("unreadable_archive_member")
            with stream:
                for block in iter(lambda: stream.read(4 * 1024**2), b""):
                    h.update(block)
            if h.hexdigest() != expected_members[name]["sha256"]:
                raise ValueError("archive_member_bytes_changed")
    if seen != expected_members.keys():
        raise ValueError("missing_archive_member")


def validate_bundle(root: Path, shard: dict, execution: dict) -> dict:
    report_path = root / "cache-verified.json"
    if report_path.is_symlink():
        raise ValueError("regular_report_required")
    report = json.loads(report_path.read_text())
    expected = {
        "schema_version": 1,
        "status": "native_paired_shard_cache_and_loader_verified",
        "selection_sha256": shard["selection_sha256"],
        "shard": shard["shard"],
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
        "execution_provenance": execution,
    }
    if any(report.get(k) != v for k, v in expected.items()) or (
        type(report.get("max_reference_uint16_difference")) is not int
        or not 0 <= report["max_reference_uint16_difference"] <= 1
    ):
        raise ValueError("worker_scope_or_binding_changed")
    if set(report["archives"]) != {"cache-side-256.tar", "cache-side-384.tar"}:
        raise ValueError("paired_archive_membership_required")
    verify_file(root / "series-proofs.jsonl", report["source_proofs"])
    proofs = [json.loads(line) for line in (root / "series-proofs.jsonl").read_text().splitlines()]
    rows = {r["series_uid"]: r for r in shard["series"]}
    if len(rows) != shard["series_count"] or (
        len(proofs) != len(rows) or {p["series_uid"] for p in proofs} != rows.keys()
    ):
        raise ValueError("source_proof_series_membership_changed")
    if {r["study_uid"] for r in rows.values()} != set(shard["study_uids"]):
        raise ValueError("source_proof_study_membership_changed")
    if sum(r["n_slices"] for r in rows.values()) != shard["slices"] or (
        sum(r["source_bytes"] for r in rows.values()) != shard["source_bytes"]
    ):
        raise ValueError("source_proof_counts_changed")
    if (
        max(p["accepted256_difference"]["max_absolute_uint16_difference"] for p in proofs)
        != (report["max_reference_uint16_difference"])
    ):
        raise ValueError("reference_difference_summary_changed")
    for side, input_side in ((256, 224), (384, 384)):
        loader = report["loaders"][str(side)]
        if any(
            loader[k] != v
            for k, v in {
                "studies": len(shard["study_uids"]),
                "source_cache_side": side,
                "input_side": input_side,
                "original_source_revision": SOURCE,
                "labels_read": False,
            }.items()
        ) or not 0 < loader["valid_sampled_slices"] <= 96 * len(shard["study_uids"]):
            raise ValueError("actual_loader_acceptance_changed")
        members = {}
        for proof in proofs:
            row = rows[proof["series_uid"]]
            if (proof["study_uid"], proof["slices"], proof["source_bytes"]) != (
                row["study_uid"],
                row["n_slices"],
                row["source_bytes"],
            ) or not 0 <= proof["accepted256_difference"]["max_absolute_uint16_difference"] <= 1:
                raise ValueError("source_proof_identity_or_reference_changed")
            cache = "cache/" + row["cache_path"]
            members[cache] = proof["files"][str(side)]["npz"]
            members[str(PurePosixPath(cache).with_suffix(".json"))] = proof["files"][str(side)][
                "metadata"
            ]
        record = report["archives"][f"cache-side-{side}.tar"]
        if len(members) != shard["regular_members_per_side"] or (
            record["bytes"] > shard["archive_upper_bytes_by_side"][str(side)]
        ):
            raise ValueError("archive_planned_bound_changed")
        verify_archive(root / f"cache-side-{side}.tar", record, members)
    return report


def verify_remote(api, prefix, revision, records, token, remote_url, get):
    if api.repo_info(REPO, repo_type="dataset", revision=revision).private is not True:
        raise ValueError("private_revision_required")
    for name, record in records.items():
        size = 0
        h = hashlib.sha256()
        with get(
            remote_url(REPO, prefix + "/" + name, repo_type="dataset", revision=revision),
            headers={"Authorization": "Bearer " + token},
            stream=True,
            timeout=(20, 180),
        ) as response:
            response.raise_for_status()
            for block in response.iter_content(4 * 1024**2):
                size += len(block)
                h.update(block)
        if size != record["bytes"] or h.hexdigest() != record["sha256"]:
            raise ValueError("private_download_bytes_changed:" + name)


def reconcile(journal, api, token, remote_url, get):
    """Read exact attempted paths; never rebuild, upload or retry a commit."""
    intent = json.loads((journal / "intent.json").read_text())
    if intent["repo_id"] != REPO or intent["automatic_retry"] is not False:
        raise ValueError("unexpected_canonical_intent")
    if (journal / "verified.json").exists():
        receipt = json.loads((journal / "verified.json").read_text())
        if receipt["files"] != intent["files"] or receipt["prefix"] != intent["prefix"]:
            raise ValueError("verified_intent_binding_changed")
        return {**receipt, "reconciliation": "existing_verified_receipt_no_mutation"}
    head = api.repo_info(REPO, repo_type="dataset")
    if head.private is not True:
        raise ValueError("private_repository_required")
    commit = journal / "commit.json"
    revision = json.loads(commit.read_text())["revision"] if commit.exists() else head.sha
    paths = {intent["prefix"] + "/" + n for n in intent["files"]}
    actual = api.get_paths_info(REPO, paths=sorted(paths), repo_type="dataset", revision=revision)
    if not actual:
        return {
            "status": "uncertain_intent_paths_absent_at_readonly_head",
            "revision": revision,
            "automatic_retry": False,
            "external_mutations": 0,
        }
    if {x.path for x in actual} != paths:
        raise ValueError("partial_remote_shard_requires_readonly_investigation")
    verify_remote(api, intent["prefix"], revision, intent["files"], token, remote_url, get)
    receipt = {
        "status": "private_native_shard_full_download_size_sha_verified",
        "repo_id": REPO,
        "prefix": intent["prefix"],
        "revision": revision,
        "files": intent["files"],
        "execution_provenance": intent["execution_provenance"],
        "shard": intent["shard"],
        "model_training_or_accuracy_claim": False,
        "reconciled_readonly": True,
        "external_mutations": 0,
    }
    save_new(journal / "verified.json", receipt)
    return receipt


def publish(root, shard, execution, journal, api, token, operation, remote_url, get):
    """Single external mutation; an existing intent always requires reconciliation."""
    if journal.exists():
        raise FileExistsError("Publication already attempted; reconcile the exact intent read-only")
    validate_bundle(root, shard, execution)
    prefix = (
        "experiments/dinov3-v1/native-original384-v1/selection-"
        + shard["selection_sha256"]
        + f"/shard-{shard['shard']:03d}"
    )
    files = {
        n: root / n
        for n in (
            "cache-side-256.tar",
            "cache-side-384.tar",
            "cache-verified.json",
            "series-proofs.jsonl",
        )
    }
    attribution = {
        "original_dataset": "RSNA Knee Abnormality Detection",
        "competition_url": "https://www.kaggle.com/competitions/rsna-knee-abnormality-detection",
        "source_private_repo": REPO,
        "original_source_revision": SOURCE,
        "scientific_source_git": SCIENCE,
        "use": "Private competition-data cache; original access and use restrictions retained",
        "relicensed": False,
        "model_weights_included": False,
    }
    data = {"attribution.json": (json.dumps(attribution, indent=2, sort_keys=True) + "\n").encode()}
    records = {n: {"bytes": p.stat().st_size, "sha256": digest(p)} for n, p in files.items()}
    records.update(
        {n: {"bytes": len(v), "sha256": hashlib.sha256(v).hexdigest()} for n, v in data.items()}
    )
    manifest = {
        "schema_version": 1,
        "prefix": prefix,
        "shard": shard,
        "execution_provenance": execution,
        "files": records.copy(),
        "full_cohort_built": False,
        "competition_submission": False,
    }
    data["manifest.json"] = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    records["manifest.json"] = {
        "bytes": len(data["manifest.json"]),
        "sha256": hashlib.sha256(data["manifest.json"]).hexdigest(),
    }
    head = api.repo_info(REPO, repo_type="dataset")
    if head.private is not True:
        raise ValueError("private_repository_required")
    if api.get_paths_info(
        REPO, paths=[prefix + "/" + n for n in records], repo_type="dataset", revision=head.sha
    ):
        raise FileExistsError("Private shard paths already exist; reconcile without re-upload")
    journal.mkdir(mode=0o700)
    save_new(
        journal / "intent.json",
        {
            "repo_id": REPO,
            "prefix": prefix,
            "parent_revision": head.sha,
            "files": records,
            "automatic_retry": False,
            "execution_provenance": execution,
            "shard": shard["shard"],
        },
    )
    additions = [
        operation(path_in_repo=prefix + "/" + n, path_or_fileobj=p) for n, p in files.items()
    ]
    additions += [
        operation(path_in_repo=prefix + "/" + n, path_or_fileobj=v) for n, v in data.items()
    ]
    commit = api.create_commit(
        REPO,
        repo_type="dataset",
        parent_commit=head.sha,
        operations=additions,
        commit_message=f"Preserve verified paired native cache shard {shard['shard']:03d}",
    )
    save_new(journal / "commit.json", {"repo_id": REPO, "prefix": prefix, "revision": commit.oid})
    verify_remote(api, prefix, commit.oid, records, token, remote_url, get)
    verified = {
        "status": "private_native_shard_full_download_size_sha_verified",
        "repo_id": REPO,
        "prefix": prefix,
        "revision": commit.oid,
        "files": records,
        "execution_provenance": execution,
        "shard": shard["shard"],
        "model_training_or_accuracy_claim": False,
    }
    save_new(journal / "verified.json", verified)
    return verified


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--plan-manifest", type=Path)
    parser.add_argument("--shard", type=int)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--reconcile", action="store_true")
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_GPUS")
        or (os.environ.get("SLURM_CPUS_PER_TASK") != "1")
    ):
        raise ValueError("one_cpu_zero_gpu_allocation_required")
    payload = json.load(sys.stdin)
    if set(payload) != {"token"} or not payload["token"].startswith("hf_"):
        raise ValueError("private_token_stdin_required")
    import requests
    from huggingface_hub import CommitOperationAdd, HfApi, hf_hub_url

    api = HfApi(token=payload["token"])
    if args.reconcile:
        print(json.dumps(reconcile(args.journal, api, payload["token"], hf_hub_url, requests.get)))
        return
    if args.root is None or args.plan_manifest is None or args.shard is None:
        raise ValueError("fresh_publication_requires_root_plan_and_shard")
    if digest(args.plan_manifest) != PLAN_SHA:
        raise ValueError("pinned_full_plan_required")
    plan = json.loads(args.plan_manifest.read_text())
    name = f"shard-{args.shard:03d}.json"
    path = args.plan_manifest.parent / name
    verify_file(path, plan["shards"][name])
    shard = json.loads(path.read_text())
    if any(
        shard[k] != plan[k]
        for k in ("scientific_source_git", "source_revision", "selection_sha256", "input_bindings")
    ):
        raise ValueError("shard_plan_binding_changed")
    report = json.loads((args.root / "cache-verified.json").read_text())
    execution = report["execution_provenance"]
    if any(
        execution.get(k) != v
        for k, v in {
            "job_id": os.environ["SLURM_JOB_ID"],
            "cpus": 1,
            "gpus": 0,
            "worker_sha256": WORKER_SHA,
            "plan_manifest_sha256": PLAN_SHA,
            "shard_file_sha256": digest(path),
        }.items()
    ):
        raise ValueError("same_allocated_worker_execution_required")

    receipt = publish(
        args.root,
        shard,
        execution,
        args.journal,
        api,
        payload["token"],
        CommitOperationAdd,
        hf_hub_url,
        requests.get,
    )
    print(
        json.dumps(
            {"status": receipt["status"], "shard": args.shard, "revision": receipt["revision"]}
        )
    )


if __name__ == "__main__":
    main()
