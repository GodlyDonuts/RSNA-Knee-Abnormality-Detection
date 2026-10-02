"""Admit a new all-development population against the original native byte proof.

Runs on one CPU on the owning cache node. No encoder, optimizer, scheduler or
publication runs here; a successful receipt admits inputs, not GPU training.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

BASE = Path("/home/sa305415/rsna-knee")
ORIGINAL = BASE / "production-launch-native384-a32-c9-v1"
CODE = BASE / "code-delta-recovery-c9fcc7e"
INPUTS = BASE / "code-stopfix-1f486765"
SCIENCE = "c9fcc7eefb09103942be89bf4fa87410eb9ec5ab"
ADMISSION_SHA = "81060696c0fb4312d04e8f0a68b2c0eb515db743938af723a31f58e7ed962124"
RELEASE_SHA = "e2ecb283915a5b1810bfc9a2f6e1dda1f1d0d762f91cf2487140af3d05cb6c35"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def object_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def require_scientific_import(imported, code):
    if not Path(imported).resolve().is_relative_to(code.resolve()):
        raise ValueError("exact_original_scientific_import_required")


def validate_config(original, candidate):
    expected = {**original, "fit_scope": "all_development", "warmup_updates": 213}
    if candidate != expected or candidate.get("diagnostic_study_limit") is not None:
        raise ValueError("only_all_development_scope_and_schedule_may_change")
    if original.get("side") != 384 or original.get("fit_scope") != "outer_fold":
        raise ValueError("original_native384_reference_required")
    return {k: [original[k], candidate[k]] for k in expected if original[k] != candidate[k]}


def verify_population(manifest, train, outer, registry, targets):
    eligible = manifest[manifest.split.eq("train")].copy()
    observed = eligible[[f"{t}__mask" for t in targets]].sum(axis=1) > 0
    expected = eligible[observed].sort_values("study_uid").reset_index(drop=True)
    ids = train.study_uid.tolist()
    native_ids = registry["training_uids"] + registry["outer_uids"]
    if (
        len(train) != 3403
        or len(outer) != 0
        or ids != expected.study_uid.tolist()
        or len(set(ids)) != 3403
        or set(ids) != set(native_ids)
        or manifest.split.value_counts().to_dict()
        != {"train": 3904, "weak_validation": 441, "gold_evaluation": 58, "quarantine": 4}
        or int((~observed).sum()) != 501
    ):
        raise ValueError("exact_3403_accepted_studies_and_reserved_exclusions_required")
    columns = ["study_uid"] + [column for t in targets for column in (t, f"{t}__mask")]
    if not train[columns].reset_index(drop=True).equals(expected[columns]):
        raise ValueError("original_labels_and_masks_must_be_unchanged")
    return {
        "training_studies": 3403,
        "outer_studies": 0,
        "training_uids_sha256": object_sha(sorted(ids)),
        "outer_uids_sha256": object_sha([]),
        "fully_masked_train_excluded": 501,
        "reserved_confirmation_excluded": 441,
        "official_gold_excluded": 58,
        "quarantine_excluded": 4,
        "labels_and_masks_unchanged": True,
        "steps_per_epoch": 426,
        "fixed_final_optimizer_steps": 4260,
        "evaluation_or_selection_permitted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    args = parser.parse_args()
    control = args.control
    output = control / "inputs-accepted.json"
    if output.exists() or output.is_symlink():
        raise FileExistsError("existing_input_receipt_requires_readonly_reconciliation")
    if (
        os.uname().nodename != "evc31"
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
        or os.environ.get("SLURM_JOB_GPUS")
        or not os.environ.get("SLURM_JOB_ID", "").isdecimal()
    ):
        raise ValueError("one_cpu_zero_gpu_owning_node_job_required")
    source = ORIGINAL / "native384_admission.py"
    if digest(source) != ADMISSION_SHA:
        raise ValueError("original_native_admission_source_required")
    spec = importlib.util.spec_from_file_location("original_native_admission", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    proof_file = ORIGINAL / "cpu-preflight.json"
    proof = json.loads(proof_file.read_text())
    if (
        proof.get("status") != "full_native384_cpu_training_preflight_accepted"
        or proof.get("members") != 37812
        or proof.get("all_member_bytes_verified") is not True
        or proof.get("node") != "evc31"
        or object_sha(proof["signatures"]) != proof.get("signatures_sha256")
    ):
        raise ValueError("actual_original_full_byte_proof_required")
    module.require_terminal_assembly(
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
    assembly = BASE / "native384-assembly-c9-v1"
    accepted, registry, members, restored = module.assembly_inputs(assembly)
    if (
        digest(assembly / "registry.json") != proof["registry_file_sha256"]
        or digest(assembly / "accepted.json") != proof["assembly_acceptance_sha256"]
        or accepted["cache_root"] != proof["cache_root"]
    ):
        raise ValueError("original_cpu_byte_proof_binding_changed")
    module.verify_assets(ORIGINAL / "config.json", assembly, prior=proof["scientific_assets"])
    signatures = module.verify_members(restored, members, prior=proof["signatures"])
    original = json.loads((ORIGINAL / "config.json").read_text())
    candidate = json.loads((control / "config.json").read_text())
    changes = validate_config(original, candidate)
    sys.path.insert(0, str(CODE))
    from rsna.competitive.folds import load_folds
    from rsna.competitive.train import TrainConfig, select_development
    from rsna.release import load_verified_release
    from rsna.schema import TARGETS

    require_scientific_import(sys.modules["rsna.competitive.train"].__file__, CODE)
    TrainConfig(**candidate).validate()
    manifest, release = load_verified_release(
        INPUTS / "initiative/releases/report-supervision-v1/studies.parquet", INPUTS
    )
    if release["binding_sha256"] != RELEASE_SHA:
        raise ValueError("original_verified_release_required")
    folds = load_folds(INPUTS / "initiative/experiments/competitive-v1/folds", manifest, release)
    train, outer = select_development(manifest, folds, 0, "all_development")
    population = verify_population(manifest, train, outer, registry, TARGETS)
    free = shutil.disk_usage("/tmp").free
    required_local = (20 + 7 + 3 + 3) * 2**30
    if free < required_local:
        raise ValueError("node_reserve_environment_production_and_recovery_funding_required")
    record = {
        "status": "native384_all_development_inputs_accepted",
        "job_id": os.environ["SLURM_JOB_ID"],
        "node": "evc31",
        "scientific_source_git": SCIENCE,
        "code_sha256": {
            str(p.relative_to(CODE / "rsna")): digest(p)
            for p in sorted((CODE / "rsna").rglob("*.py"))
        },
        "source_admission_sha256": ADMISSION_SHA,
        "gate_sha256": digest(Path(__file__)),
        "config_sha256": digest(control / "config.json"),
        "original_cpu_proof_sha256": digest(proof_file),
        "release_binding_sha256": RELEASE_SHA,
        "original_registry_sha256": digest(assembly / "registry.json"),
        "cache_root": accepted["cache_root"],
        "members": len(members),
        "original_all_byte_proof_reused": True,
        "all_original_signatures_unchanged": True,
        "signatures_sha256": object_sha(signatures),
        "configuration_changes": changes,
        **population,
        "local_free_bytes": free,
        "local_funded_bytes": required_local,
        "allocated_gpus": 0,
        "encoder_or_optimizer_executed": False,
        "gpu_recovery_accepted": False,
        "production_dispatched": False,
        "checkpoint_publication_accepted": False,
    }
    with output.open("x") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    print(
        json.dumps(
            {
                "status": record["status"],
                "receipt_sha256": digest(output),
                "training_studies": 3403,
                "production_dispatched": False,
            }
        )
    )


if __name__ == "__main__":
    main()
