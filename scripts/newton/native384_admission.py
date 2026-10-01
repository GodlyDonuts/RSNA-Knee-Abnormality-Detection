"""Admit the full native cache on CPU, then bind unchanged inputs to the real GPU run.

CPU verification reads all restored bytes before a GPU is requested. The GPU
check compares the complete file inventory and stat signatures on the same node;
it also exercises the allocated CUDA device and strict deterministic runtime.
No transfer, model fitting, credential handling or scheduler dispatch occurs here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath

SCIENCE = "c9fcc7eefb09103942be89bf4fa87410eb9ec5ab"
ASSEMBLY_JOB = "866458"
ASSEMBLY_SHA = "dacb6c10fc2c47c07819e1ae7ce5077ea8e2ca11f5564bb004a53ceb7ff36891"
PLAN_SHA = "44e6775c5413e42d8744ead0f7d7bb0de258893a10867123e47fbd770f84793f"
SELECTION_SHA = "8bbf6ed6f2d9ec8e3ab9f39b0ac6f535865b6cb002915af0283c3cf458538ce2"
CONFIG_SHA = "e7ab923745a22995470ff68d98b46212acd1f4d12c10b07004851027ad488788"
ENV_SHA = "89597ed4d974e5874c7bb2ffb42545e6c0bb6a9ac796a5839a4b1902917ef338"
WRAPPER_SHA = "4141bedbe24cd87c31bbcffba4a34747c5228495c3e6f29b6b45a849196970a7"
TRAIN_UID_SHA = "e9d38ee21a114973177b89f4602129c7dc047927b8499991d3be30fd89737f3a"
OUTER_UID_SHA = "2f365bb0ce8e052f22dcab78bb2bbb7e03c0919ab28ba94ab9d3d8d6e08f0fdb"
NODE = "evc31"
BASE = Path("/home/sa305415/rsna-knee")
PILOT = BASE / "native384-recovery-c9-deterministic-evc31-v2/pilot-verified.json"
PILOT_SHA = "1e2febd0a08f4343b7184a589693d3028dc732b4501f281ad5a53e9ecd1454d2"
CODE = BASE / "code-delta-recovery-c9fcc7e"
INPUTS = BASE / "code-stopfix-1f486765"
ENVIRONMENT = (
    BASE / "kaggle-wheelhouse-v1/result-cp312-cu130-finalmain-v1/environment-cp312-cu130.tar"
)


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def save_new(path, value):
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def regular(path, root):
    if path.is_symlink() or not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("confined_regular_native_input_required")
    return path


def signature(path):
    st = path.stat()
    return [st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns]


def require_terminal_assembly(text, job=ASSEMBLY_JOB):
    if not re.fullmatch("[1-9][0-9]*", job):
        raise ValueError("real_cpu_job_required")
    rows = [line.strip().split("|") for line in text.splitlines() if line.strip()]
    if rows != [[job, "COMPLETED", "0:0"]]:
        raise ValueError("exact_cpu_assembly_terminal_success_required")


def validate_registry(registry):
    required = {
        "schema_version": 1,
        "status": "all_75_original_source_shards_privately_verified",
        "plan_manifest_sha256": PLAN_SHA,
        "selection_sha256": SELECTION_SHA,
        "scientific_source_git": SCIENCE,
        "source_revision": "0b5e410288232f570c69ba4f44723356e6ae6ab7",
        "repo_id": "GodlyDonuts/RSNA-Data",
        "cache_side": 384,
        "study_count": 3403,
        "series_count": 18906,
        "target_values_read": False,
        "full_cache_restored": False,
    }
    if any(type(registry.get(k)) is not type(v) or registry[k] != v for k, v in required.items()):
        raise ValueError("full_exact_native_cohort_registry_required")
    train, outer = registry["training_uids"], registry["outer_uids"]
    if (
        len(train) != 2247
        or len(outer) != 1156
        or len(set(train + outer)) != 3403
        or object_sha(sorted(train)) != TRAIN_UID_SHA
        or object_sha(sorted(outer)) != OUTER_UID_SHA
        or [row["shard"] for row in registry["shards"]] != list(range(75))
    ):
        raise ValueError("frozen_full_training_and_outer_membership_required")
    members, study_ids, series_ids = {}, set(), set()
    for row in registry["shards"]:
        if not re.fullmatch("[0-9a-f]{40}", row.get("revision", "")):
            raise ValueError("immutable_private_shard_revision_required")
        expected = set()
        for series in row["series"]:
            if series["series_uid"] in series_ids:
                raise ValueError("duplicate_native_series")
            study_ids.add(series["study_uid"])
            series_ids.add(series["series_uid"])
            name = f"cache/train_series/{series['study_uid']}/{series['series_uid']}.npz"
            if series["cache_path"] != name.removeprefix("cache/"):
                raise ValueError("native_series_path_identity_changed")
            expected.update((name, str(PurePosixPath(name).with_suffix(".json"))))
        if expected != set(row["members"]) or members.keys() & expected:
            raise ValueError("exact_disjoint_native_series_members_required")
        members.update(row["members"])
    if len(members) != 37812 or len(series_ids) != 18906 or study_ids != set(train + outer):
        raise ValueError("full_native_cohort_member_count_required")
    return members


def verify_members(restored, members, *, prior=None):
    """Full byte check on CPU; exact inventory and stat comparison on GPU."""
    actual = {str(p.relative_to(restored)) for p in (restored / "cache").rglob("*") if p.is_file()}
    if actual != set(members):
        raise ValueError("native_restored_inventory_changed")
    if prior is not None and set(prior) != set(members):
        raise ValueError("exact_cpu_verified_member_signatures_required")
    signatures = {}
    for name, row in sorted(members.items()):
        rel = PurePosixPath(name)
        if rel.is_absolute() or ".." in rel.parts or rel.parts[0] != "cache":
            raise ValueError("unsafe_native_member")
        path = regular(restored.joinpath(*rel.parts), restored)
        before = signature(path)
        if type(row.get("bytes")) is not int or before[2] != row["bytes"]:
            raise ValueError("native_member_size_changed")
        if prior is None:
            if digest(path) != row.get("sha256"):
                raise ValueError("native_member_hash_changed")
        elif before != prior.get(name):
            raise ValueError("cpu_verified_native_member_changed")
        if before != signature(path):
            raise ValueError("native_member_changed_during_acceptance")
        signatures[name] = before
    return signatures


def verify_assets(config_file, assembly, *, prior=None):
    """Bind the same accepted science, six data artifacts, model and environment."""
    if digest(config_file) != CONFIG_SHA or digest(PILOT) != PILOT_SHA:
        raise ValueError("exact_native_configuration_and_recovery_pilot_required")
    pilot = json.loads(PILOT.read_text())
    if (
        subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()
        != SCIENCE
    ):
        raise ValueError("pinned_native_scientific_checkout_required")
    code = {
        str(p.relative_to(CODE / "rsna")): digest(p) for p in sorted((CODE / "rsna").rglob("*.py"))
    }
    if code != pilot["binding"]["code_sha256"]:
        raise ValueError("native_scientific_source_files_changed")
    launch = json.loads((assembly / "launch.json").read_text())
    plan_path = Path(launch["manifest"])
    if digest(plan_path) != PLAN_SHA:
        raise ValueError("accepted_native_input_manifest_required")
    plan = json.loads(plan_path.read_text())
    config = json.loads(config_file.read_text())
    encoder = config["encoder"]
    files = {
        str(INPUTS / name): {"bytes": record["bytes"], "sha256": record["sha256"]}
        for name, record in plan["input_bindings"].items()
    }
    files.update(
        {
            encoder["weights_path"]: {
                "bytes": 1212559808,
                "sha256": "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
            },
            encoder["config_path"]: {
                "sha256": "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5"
            },
            str(ENVIRONMENT): {"sha256": ENV_SHA},
            str(INPUTS / "scripts/newton/runtime_environment.py"): {
                "sha256": "f1bfa0693a2d2c7afe9070ccd307b12206c24b89231223b93c2ccf3d7d97f8fd"
            },
            str(INPUTS / "scripts/newton/runtime_archive.py"): {
                "sha256": "ad3f0c6f77fc2ff53b7aafc0781668116bd36ba9fdb5f17e011cd2b3cd15592c"
            },
            str(CODE / "scripts/newton/supervise.py"): {
                "sha256": "e7631ca4139021d794923a0f05368d847e8dab08a5b124b754ffe51345db69e1"
            },
        }
    )
    if prior is not None and set(files) != set(prior):
        raise ValueError("exact_native_scientific_assets_required")
    checked = {}
    for name, expected in sorted(files.items()):
        path = regular(Path(name), BASE)
        before = signature(path)
        if before[2] != expected.get("bytes", before[2]):
            raise ValueError("native_scientific_asset_size_changed")
        if prior is None:
            if digest(path) != expected["sha256"]:
                raise ValueError("native_scientific_asset_hash_changed")
        elif prior[name] != {"sha256": expected["sha256"], "signature": before}:
            raise ValueError("cpu_verified_native_scientific_asset_changed")
        if signature(path) != before:
            raise ValueError("native_scientific_asset_changed_during_acceptance")
        checked[name] = {"sha256": expected["sha256"], "signature": before}
    return checked


def assembly_inputs(control):
    launch = regular(control / "launch.json", control)
    registry_file = regular(control / "registry.json", control)
    accepted_file = regular(control / "accepted.json", control)
    if digest(launch) != ASSEMBLY_SHA:
        raise ValueError("pinned_cpu_assembly_launch_required")
    registry = json.loads(registry_file.read_text())
    members = validate_registry(registry)
    accepted = json.loads(accepted_file.read_text())
    expected = {
        "status": "full_native384_cohort_bytes_metadata_restored",
        "job_id": ASSEMBLY_JOB,
        "node": NODE,
        "members": 37812,
        "studies": 3403,
        "series": 18906,
        "cache_side": 384,
        "target_values_read": False,
        "gpu_reserved": False,
        "fresh_training_or_accuracy_claim": False,
        "registry_file_sha256": digest(registry_file),
        "environment_archive_sha256": ENV_SHA,
        "source_git": "44dce81d86aa0971e41d827533a168130a982905",
    }
    if any(type(accepted.get(k)) is not type(v) or accepted[k] != v for k, v in expected.items()):
        raise ValueError("actual_full_cpu_assembly_acceptance_required")
    owned = Path(accepted["owned_root"])
    if (
        owned.is_symlink()
        or owned.parent != Path("/tmp")
        or not owned.name.startswith("rsna-native384-cohort-" + ASSEMBLY_JOB + "-")
        or owned.stat().st_uid != os.getuid()
        or os.uname().nodename != NODE
    ):
        raise ValueError("same_node_owned_native_cache_required")
    marker = regular(owned / ".native384-cohort-owner.json", owned)
    if json.loads(marker.read_text()) != {
        "kind": "native384-cohort-owned-v1",
        "job_id": ASSEMBLY_JOB,
        "uid": os.getuid(),
        "registry_sha256": digest(registry_file),
    }:
        raise ValueError("owned_native_cache_marker_changed")
    restored = owned / "restored"
    if Path(accepted["cache_root"]) != restored / "cache":
        raise ValueError("actual_assembled_cache_path_required")
    raw = json.loads(regular(restored / "accepted.json", restored).read_text())
    raw_required = {
        k: expected[k]
        for k in (
            "status",
            "job_id",
            "node",
            "members",
            "studies",
            "series",
            "cache_side",
            "target_values_read",
        )
    }
    if any(type(raw.get(k)) is not type(v) or raw[k] != v for k, v in raw_required.items()):
        raise ValueError("node_local_assembly_receipt_changed")
    require_terminal_assembly(
        subprocess.check_output(
            [
                "sacct",
                "-X",
                "-j",
                ASSEMBLY_JOB,
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,State,ExitCode",
            ],
            text=True,
        )
    )
    return accepted, registry, members, restored


def validate_provenance(value):
    expected = {
        "schema_version": 1,
        "status": "full_native384_training_input_and_cuda_accepted",
        "scientific_source_git": SCIENCE,
        "assembly_job_id": ASSEMBLY_JOB,
        "assembly_launch_sha256": ASSEMBLY_SHA,
        "plan_manifest_sha256": PLAN_SHA,
        "selection_sha256": SELECTION_SHA,
        "config_file_sha256": CONFIG_SHA,
        "environment_archive_sha256": ENV_SHA,
        "deterministic_wrapper_sha256": WRAPPER_SHA,
        "node": NODE,
        "cache_side": 384,
        "input_side": 384,
        "training_studies": 2247,
        "outer_studies": 1156,
        "members": 37812,
        "cpu_full_hash_verified": True,
        "gpu_all_signatures_unchanged": True,
        "strict_deterministic_algorithms": True,
        "cuda_context_verified": True,
        "torch_version": "2.14.0+cu130",
        "gpu_count": 1,
        "cpus": 2,
    }
    if any(type(value.get(k)) is not type(v) or value[k] != v for k, v in expected.items()):
        raise ValueError("exact_native_training_provenance_required")
    for key in (
        "cpu_preflight_sha256",
        "registry_file_sha256",
        "assembly_acceptance_sha256",
        "signatures_sha256",
        "admission_source_sha256",
    ):
        if not re.fullmatch("[0-9a-f]{64}", value.get(key, "")):
            raise ValueError("complete_native_input_hashes_required")
    if not re.fullmatch("[1-9][0-9]*", value.get("training_job_id", "")):
        raise ValueError("real_native_training_job_required")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("cpu", "check", "gpu"))
    parser.add_argument("--assembly", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cpu-proof", type=Path)
    parser.add_argument("--deterministic-wrapper", type=Path)
    args = parser.parse_args()
    if args.mode != "check" and args.output is None:
        raise ValueError("new_admission_output_required")
    if args.output is not None and (args.output.exists() or args.output.is_symlink()):
        raise FileExistsError("existing_admission_requires_readonly_review")
    job = os.environ.get("SLURM_JOB_ID", "")
    if not re.fullmatch("[1-9][0-9]*", job) or digest(args.config) != CONFIG_SHA:
        raise ValueError("real_job_and_exact_native_configuration_required")
    accepted, registry, members, restored = assembly_inputs(args.assembly)
    base = {
        "registry_file_sha256": digest(args.assembly / "registry.json"),
        "assembly_acceptance_sha256": digest(args.assembly / "accepted.json"),
        "cache_root": str(restored / "cache"),
        "config_file_sha256": CONFIG_SHA,
        "admission_source_sha256": digest(Path(__file__)),
    }
    if args.mode == "cpu":
        if os.environ.get("SLURM_JOB_GPUS") or os.environ.get("SLURM_CPUS_PER_TASK") != "1":
            raise ValueError("one_cpu_no_gpu_native_preflight_required")
        assets = verify_assets(args.config, args.assembly)
        signatures = verify_members(restored, members)
        proof = {
            **base,
            "schema_version": 1,
            "status": "full_native384_cpu_training_preflight_accepted",
            "job_id": job,
            "node": NODE,
            "members": len(members),
            "all_member_bytes_verified": True,
            "scientific_assets": assets,
            "signatures": signatures,
            "signatures_sha256": object_sha(signatures),
        }
        save_new(args.output, proof)
    else:
        if args.cpu_proof is None or args.deterministic_wrapper is None:
            raise ValueError("actual_cpu_proof_and_wrapper_required")
        proof = json.loads(args.cpu_proof.read_text())
        if (
            any(proof.get(k) != v for k, v in base.items())
            or proof.get("status") != "full_native384_cpu_training_preflight_accepted"
            or proof.get("node") != NODE
            or proof.get("members") != 37812
            or proof.get("all_member_bytes_verified") is not True
            or object_sha(proof["signatures"]) != proof.get("signatures_sha256")
            or digest(args.deterministic_wrapper) != WRAPPER_SHA
            or os.environ.get("SLURM_CPUS_PER_TASK") != ("1" if args.mode == "check" else "2")
            or (args.mode == "gpu" and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8")
            or (args.mode == "check" and os.environ.get("SLURM_JOB_GPUS"))
        ):
            raise ValueError("exact_full_cpu_preflight_and_runtime_required")
        require_terminal_assembly(
            subprocess.check_output(
                [
                    "sacct",
                    "-X",
                    "-j",
                    proof["job_id"],
                    "--noheader",
                    "--parsable2",
                    "--format=JobIDRaw,State,ExitCode",
                ],
                text=True,
            ),
            job=proof["job_id"],
        )
        verify_assets(args.config, args.assembly, prior=proof["scientific_assets"])
        signatures = verify_members(restored, members, prior=proof["signatures"])
        if args.mode == "check":
            print(
                json.dumps(
                    {
                        "status": "full_native384_cpu_prerequisites_still_valid",
                        "job_id": job,
                        "node": NODE,
                        "cpu_preflight_sha256": digest(args.cpu_proof),
                        "cache_root": accepted["cache_root"],
                        "members": len(members),
                        "gpu_reserved": False,
                    }
                ),
                flush=True,
            )
            return
        import torch

        if str(torch.__version__) != "2.14.0+cu130" or torch.cuda.device_count() != 1:
            raise ValueError("accepted_native_runtime_and_one_allocated_gpu_required")
        if (
            not torch.are_deterministic_algorithms_enabled()
            or torch.is_deterministic_algorithms_warn_only_enabled()
            or torch.backends.cudnn.benchmark
            or not torch.backends.cudnn.deterministic
        ):
            raise ValueError("strict_deterministic_native_runtime_required")
        torch.cuda.set_device(0)
        value = torch.ones(8, device="cuda")
        if value.sum().item() != 8.0:
            raise ValueError("real_allocated_cuda_computation_required")
        torch.cuda.synchronize()
        record = {
            **base,
            "schema_version": 1,
            "status": "full_native384_training_input_and_cuda_accepted",
            "scientific_source_git": SCIENCE,
            "assembly_job_id": ASSEMBLY_JOB,
            "assembly_launch_sha256": ASSEMBLY_SHA,
            "plan_manifest_sha256": PLAN_SHA,
            "selection_sha256": SELECTION_SHA,
            "environment_archive_sha256": ENV_SHA,
            "deterministic_wrapper_sha256": WRAPPER_SHA,
            "node": NODE,
            "cache_side": 384,
            "input_side": 384,
            "training_studies": len(registry["training_uids"]),
            "outer_studies": len(registry["outer_uids"]),
            "members": len(members),
            "cpu_full_hash_verified": True,
            "gpu_all_signatures_unchanged": True,
            "strict_deterministic_algorithms": True,
            "cuda_context_verified": True,
            "torch_version": str(torch.__version__),
            "gpu_count": 1,
            "cpus": 2,
            "cpu_preflight_sha256": digest(args.cpu_proof),
            "signatures_sha256": object_sha(signatures),
            "training_job_id": job,
        }
        save_new(args.output, validate_provenance(record))
    print(
        json.dumps(
            {
                "status": "native384_admission_completed",
                "mode": args.mode,
                "cache_root": accepted["cache_root"],
                "output_sha256": digest(args.output),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
