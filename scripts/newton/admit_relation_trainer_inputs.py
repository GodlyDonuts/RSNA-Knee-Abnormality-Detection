"""CPU-only admission of the actual 16+16 trainer-recovery fixture."""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import torch

from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.data import CompetitiveStudyDataset, collate_studies
from rsna.competitive.delta_checkpoint import _tensor_record
from rsna.competitive.folds import load_folds
from rsna.competitive.train import _hash, _uid_hash, select_development
from rsna.release import load_verified_release
from scripts.newton.audit_relation_geometry import digest
from scripts.newton.measure_relation_gradients import (
    admit_cpu_receipt,
    admitted_batches,
    matched_configs,
    validate_cache,
)


def verify_source(reference, expanded, root):
    extras = {
        "scripts/newton/admit_relation_trainer_inputs.py",
        "scripts/newton/launch_relation_trainer_inputs.py",
        "scripts/newton/probe_relation_trainer_recovery.py",
    }
    if (
        expanded.get("reference_source_git") != reference["source_git"]
        or set(expanded["files"]) != set(reference["files"]) | extras
        or any(expanded["files"][name] != row for name, row in reference["files"].items())
    ):
        raise ValueError("Recovery source must preserve every original numerical/input module")
    for name, row in expanded["files"].items():
        path = root / name
        if (
            path.is_symlink()
            or path.stat().st_size != row["bytes"]
            or file_sha256(path) != row["sha256"]
        ):
            raise ValueError("Materialized recovery source differs from its immutable pack")


def fixture_request(train, outer, release):
    if len(train) < 16 or len(outer) < 16 or set(train.study_uid) & set(outer.study_uid):
        raise ValueError("Disjoint accepted training/outer development pools required")
    return {
        "training_uids": train.head(16).study_uid.tolist(),
        "outer_uids": outer.head(16).study_uid.tolist(),
        "full_training_uids_sha256": _uid_hash(train),
        "full_outer_uids_sha256": _uid_hash(outer),
        "release_binding": release,
        "diagnostic_only": True,
        "clinical_selection_allowed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "config-root",
        "proof",
        "request",
        "run",
        "studies",
        "artifact-root",
        "folds",
        "series",
        "cache",
        "archive",
        "output",
        "source-manifest",
        "cpu-acceptance",
        "recovery-source-manifest",
        "fixture-request",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if (
        not os.environ.get("SLURM_JOB_ID", "").isdecimal()
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
        or os.environ.get("SLURM_JOB_GPUS")
        or torch.cuda.device_count() != 0
    ):
        raise ValueError("One CPU and zero GPUs required for input admission")
    if args.output.exists():
        raise FileExistsError("Original recovery input receipt exists; do not repeat admission")
    torch.set_num_threads(1)
    reference, source = (
        json.loads(path.read_text())
        for path in (args.source_manifest, args.recovery_source_manifest)
    )
    verify_source(reference, source, Path(__file__).resolve().parents[2])
    configs = matched_configs(args.config_root)
    batches, binding = admitted_batches(args, configs["physical"])
    admit_cpu_receipt(json.loads(args.cpu_acceptance.read_text()), binding, configs)
    del batches
    manifest, release = load_verified_release(args.studies, args.artifact_root)
    folds = load_folds(args.folds, manifest, release)
    folds_sha = _hash(folds.sort_values("study_uid").to_dict("records"))
    if folds_sha != json.loads(args.run.read_text())["binding"]["folds_sha256"]:
        raise ValueError("Full accepted fold membership differs from the original run")
    train, outer = select_development(manifest, folds, configs["physical"].fold)
    request = fixture_request(train, outer, release)
    if json.loads(args.fixture_request.read_text()) != request:
        raise ValueError("Fixture differs from the actual accepted training/development split")
    outer = outer.head(16)
    series = pd.read_csv(args.series)
    envelope = validate_cache(
        args.cache,
        args.archive,
        json.loads(args.proof.read_text()),
        outer.study_uid.tolist(),
        series,
    )
    config = configs["physical"]
    dataset = CompetitiveStudyDataset(
        outer,
        series,
        args.cache,
        **{
            name: getattr(config, name)
            for name in (
                "slices",
                "side",
                "max_series",
                "augment",
                "slice_sampling",
                "series_sampling",
            )
        },
    )
    records = []
    for index in range(len(dataset)):
        batch = collate_studies([dataset[index]])
        records.append(
            digest(
                {
                    name: _tensor_record(batch[name])
                    for name in (
                        "images",
                        "positions",
                        "features",
                        "series_mask",
                        "slice_mask",
                        "labels",
                        "label_mask",
                    )
                }
            )
        )
    output = {
        "status": "actual_32_study_relation_trainer_CPU_inputs_verified",
        "diagnostic_only": True,
        "optimizer_updates": 0,
        "GPU_acceptance": False,
        "clinical_accuracy_evidence": False,
        "training_studies": 16,
        "outer_engineering_studies": 16,
        "folds_sha256": folds_sha,
        "training_input_binding": binding,
        "outer_cache_binding": envelope,
        "outer_input_sha256": records,
        "fixture_request_sha256": file_sha256(args.fixture_request),
        "recovery_source_manifest_sha256": file_sha256(args.recovery_source_manifest),
        "source_git": source["source_git"],
        "reference_source_git": reference["source_git"],
        "original_CPU_acceptance_sha256": file_sha256(args.cpu_acceptance),
        "configs": {name: asdict(config) for name, config in configs.items()},
    }
    with args.output.open("x") as handle:
        json.dump(output, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"status": output["status"], "optimizer_updates": 0}), flush=True)


if __name__ == "__main__":
    main()
