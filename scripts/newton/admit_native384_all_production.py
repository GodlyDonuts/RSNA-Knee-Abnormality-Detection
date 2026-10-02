"""Bind the admitted all-development roster to one fresh native384 production run."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

BASE = Path("/home/sa305415/rsna-knee")
CODE = BASE / "code-delta-recovery-c9fcc7e"
INPUTS = BASE / "code-stopfix-1f486765"
CONTROL = BASE / "production-launch-native384-all-c9-v1"
RUN = Path("/tmp/rsna-native384-all-production-20261002-v1")
INPUT_SHA = "dc70ab477aad8db956ea276ef656508f7cab3860586a1916f723ceab44800961"
RECOVERY_SHA = "b5e8d92e50792c217b59f7ebea959744e35e365e7a7af40dfe321db7891d57a4"
SCIENCE = "c9fcc7eefb09103942be89bf4fa87410eb9ec5ab"
CONFIG_SHA = "5a857ee296de625f15637a82ef8a2c1b04ad1995a160269c07ccd6b464ca36ca"
WRAPPER_SHA = "4141bedbe24cd87c31bbcffba4a34747c5228495c3e6f29b6b45a849196970a7"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_provenance(value):
    expected = {
        "status": "native384_all_development_production_inputs_and_cuda_accepted",
        "scientific_source_git": SCIENCE,
        "config_sha256": CONFIG_SHA,
        "input_receipt_sha256": INPUT_SHA,
        "recovery_receipt_sha256": RECOVERY_SHA,
        "deterministic_wrapper_sha256": WRAPPER_SHA,
        "node": "evc31",
        "training_studies": 3403,
        "outer_studies": 0,
        "members": 37812,
        "training_uids_sha256": "6d00d9aa90799aca0f32b5e0918f9716f00f7a3e516db3a4625aaed46b373844",
        "outer_uids_sha256": "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
        "steps_per_epoch": 426,
        "fixed_final_optimizer_steps": 4260,
        "original_all_byte_proof_reused": True,
        "all_original_signatures_unchanged": True,
        "strict_deterministic_algorithms": True,
        "cuda_context_verified": True,
        "torch_version": "2.14.0+cu130",
        "gpu_count": 1,
        "cpus": 2,
        "protected_evaluation_used": False,
    }
    if any(type(value.get(k)) is not type(v) or value[k] != v for k, v in expected.items()):
        raise ValueError("exact_all_development_native_provenance_required")
    if not re.fullmatch("[1-9][0-9]*", value.get("training_job_id", "")):
        raise ValueError("actual_production_job_required")
    for k in ("admission_source_sha256", "original_registry_sha256", "signatures_sha256"):
        if not re.fullmatch("[0-9a-f]{64}", value.get(k, "")):
            raise ValueError("complete_original_input_binding_required")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    cpus = "1" if args.check_only else "2"
    if os.uname().nodename != "evc31" or os.environ.get("SLURM_CPUS_PER_TASK") != cpus:
        raise ValueError("owning_node_minimum_cpu_allocation_required")
    if args.check_only and os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("cpu_precheck_must_not_reserve_gpu")
    input_file = BASE / "native384-all-development-20261002-v2/inputs-accepted.json"
    recovery_file = BASE / "native384-all-recovery-verification-20261002-v2/recovery-accepted.json"
    if (
        digest(input_file) != INPUT_SHA
        or digest(recovery_file) != RECOVERY_SHA
        or digest(CONTROL / "config.json") != CONFIG_SHA
    ):
        raise ValueError("actual_input_recovery_and_config_receipts_required")
    for job in ("868074", "868089"):
        actual = subprocess.check_output(
            [
                "sacct",
                "-X",
                "-j",
                job,
                "--noheader",
                "--parsable2",
                "--format=JobIDRaw,State,ExitCode",
            ],
            text=True,
        ).strip()
        if actual != job + "|COMPLETED|0:0":
            raise ValueError("original_prerequisite_terminal_changed")
    proof = json.loads(input_file.read_text())
    recovery = json.loads(recovery_file.read_text())
    if recovery.get("status") != "native384_all_development_actual_recovery_accepted" or not all(
        recovery.get(k) is True
        for k in (
            "full_stop_resume_decoded_payload_equal",
            "full_storage_reference_decoded_model_equal",
            "all_numerical_updates_equal",
            "all_rng_optimizer_lr_counters_equal",
        )
    ):
        raise ValueError("actual_complete_engine_recovery_required")
    head = subprocess.check_output(["git", "-C", str(CODE), "rev-parse", "HEAD"], text=True).strip()
    hashes = {
        str(p.relative_to(CODE / "rsna")): digest(p) for p in sorted((CODE / "rsna").rglob("*.py"))
    }
    if head != SCIENCE or hashes != proof["code_sha256"]:
        raise ValueError("unchanged_original_numerical_source_required")
    original = BASE / "production-launch-native384-a32-c9-v1"
    path = original / "native384_admission.py"
    if digest(path) != proof["source_admission_sha256"]:
        raise ValueError("original_native_admission_helper_changed")
    spec = importlib.util.spec_from_file_location("original_native_admission", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assembly = BASE / "native384-assembly-c9-v1"
    accepted, registry, members, restored = module.assembly_inputs(assembly)
    old = json.loads((original / "cpu-preflight.json").read_text())
    if (
        digest(original / "cpu-preflight.json") != proof["original_cpu_proof_sha256"]
        or digest(assembly / "registry.json") != proof["original_registry_sha256"]
    ):
        raise ValueError("original_whole_byte_proofs_changed")
    module.verify_assets(original / "config.json", assembly, prior=old["scientific_assets"])
    signatures = module.verify_members(restored, members, prior=old["signatures"])
    if object_sha(signatures) != proof["signatures_sha256"]:
        raise ValueError("original_member_signatures_changed")
    import sys

    sys.path.insert(0, str(CODE))
    from rsna.competitive.folds import load_folds
    from rsna.competitive.train import select_development
    from rsna.release import load_verified_release
    from rsna.schema import TARGETS

    manifest, release = load_verified_release(
        INPUTS / "initiative/releases/report-supervision-v1/studies.parquet", INPUTS
    )
    folds = load_folds(INPUTS / "initiative/experiments/competitive-v1/folds", manifest, release)
    train, outer = select_development(manifest, folds, 0, "all_development")
    expected = manifest[manifest.split.eq("train")]
    expected = (
        expected[expected[[f"{t}__mask" for t in TARGETS]].sum(axis=1) > 0]
        .sort_values("study_uid")
        .reset_index(drop=True)
    )
    cols = ["study_uid"] + [c for t in TARGETS for c in (t, f"{t}__mask")]
    if (
        len(train) != 3403
        or len(outer) != 0
        or not train[cols].reset_index(drop=True).equals(expected[cols])
        or object_sha(sorted(train.study_uid.tolist())) != proof["training_uids_sha256"]
        or release["binding_sha256"] != proof["release_binding_sha256"]
    ):
        raise ValueError("actual_original_population_labels_masks_and_release_required")
    if RUN.exists() or RUN.is_symlink() or shutil.disk_usage("/tmp").free < (20 + 3) * 2**30:
        raise ValueError("distinct_private_output_and_checkpoint_reserve_required")
    if args.check_only:
        print(
            json.dumps(
                {
                    "status": "native384_all_development_live_cpu_precheck_passed",
                    "local_free_bytes": shutil.disk_usage("/tmp").free,
                    "input_receipt_sha256": INPUT_SHA,
                    "recovery_receipt_sha256": RECOVERY_SHA,
                    "source_git": SCIENCE,
                    "cache_root": proof["cache_root"],
                }
            )
        )
        return
    import torch

    if (
        str(torch.__version__) != "2.14.0+cu130"
        or torch.cuda.device_count() != 1
        or not torch.are_deterministic_algorithms_enabled()
        or torch.is_deterministic_algorithms_warn_only_enabled()
        or torch.backends.cudnn.benchmark
        or not torch.backends.cudnn.deterministic
        or os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"
    ):
        raise ValueError("actual_single_gpu_strict_original_runtime_required")
    torch.cuda.set_device(0)
    value = torch.ones(8, device="cuda")
    assert value.sum().item() == 8
    torch.cuda.synchronize()
    record = {
        k: proof[k]
        for k in (
            "training_studies",
            "outer_studies",
            "training_uids_sha256",
            "outer_uids_sha256",
            "members",
            "steps_per_epoch",
            "fixed_final_optimizer_steps",
            "original_registry_sha256",
            "signatures_sha256",
            "cache_root",
        )
    }
    record.update(
        status="native384_all_development_production_inputs_and_cuda_accepted",
        scientific_source_git=SCIENCE,
        config_sha256=CONFIG_SHA,
        input_receipt_sha256=INPUT_SHA,
        recovery_receipt_sha256=RECOVERY_SHA,
        deterministic_wrapper_sha256=WRAPPER_SHA,
        node="evc31",
        original_all_byte_proof_reused=True,
        all_original_signatures_unchanged=True,
        strict_deterministic_algorithms=True,
        cuda_context_verified=True,
        torch_version=str(torch.__version__),
        gpu_count=1,
        cpus=2,
        protected_evaluation_used=False,
        training_job_id=os.environ["SLURM_JOB_ID"],
        admission_source_sha256=digest(Path(__file__)),
    )
    validate_provenance(record)
    RUN.mkdir(mode=0o700)
    for leaf, v in [
        ("native-input-provenance.json", record),
        ("all-development-input-provenance.json", proof),
        (
            "owner.json",
            {
                "uid": os.getuid(),
                "job_id": os.environ["SLURM_JOB_ID"],
                "checkpoint_cap_bytes": 3 * 2**30,
                "source_git": SCIENCE,
            },
        ),
    ]:
        with (RUN / leaf).open("x") as f:
            json.dump(v, f, sort_keys=True, indent=2)
            f.write("\n")
    print(
        json.dumps(
            {
                "status": record["status"],
                "output": str(RUN),
                "training_job_id": record["training_job_id"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
