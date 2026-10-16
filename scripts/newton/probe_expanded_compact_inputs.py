"""CPU-only actual expansion to2592 fitting studies, unchanged1156 endpoint.

No model, optimizer, GPU inference, accuracy metric or training dispatch. Private
metadata routes existing volumes; no original pixel payload is copied/rebuilt.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import pandas as pd
import torch

from rsna.competitive.cache_union import MetadataUnion
from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.competitive.expanded_fitting import build_overlay, select_population
from rsna.competitive.folds import load_folds
from rsna.competitive.label_source import load_fitting_label_source
from rsna.competitive.train import select_development
from rsna.release import load_verified_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "artifact-root", "cache", "additional", "source-csv", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if os.uname().nodename != "evc33" or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("Original owning node and zero allocated GPUs required")
    torch.set_num_threads(1)
    if args.output.exists():
        raise ValueError("Fresh owner-private expansion output required")
    original_root = Path("/tmp/rsna-compact-native-n3-production-evc33-20261016-v1")
    old_binding = json.loads((original_root / "production/run.json").read_text())["binding"]
    names = [
        "rsna/competitive/data.py",
        "rsna/competitive/cache_contrast.py",
        "rsna/competitive/label_source.py",
        "rsna/competitive/folds.py",
        "rsna/competitive/neighbor_input.py",
        "rsna/competitive/neighbor_geometry.py",
        "rsna/data.py",
        "rsna/release.py",
        "rsna/schema.py",
    ]
    for name in names:
        if file_sha256(args.source / name) != old_binding["source_sha256"][name]:
            raise ValueError("Original canonical input numerical code changed")
    manifest_path = args.artifact_root / "initiative/releases/report-supervision-v1/studies.parquet"
    frame, release = load_verified_release(manifest_path, args.artifact_root)
    fold_root = args.artifact_root / "initiative/experiments/competitive-v1/folds"
    folds = load_folds(fold_root, frame, release)
    baseline, outer = select_development(frame, folds, 0)
    expanded, new_outer, population = select_population(frame, folds)
    pd.testing.assert_frame_equal(outer, new_outer)
    if (len(baseline), len(expanded), len(outer), population["excluded_all_fold_studies"]) != (
        2247,
        2592,
        1156,
        1301,
    ):
        raise ValueError("Actual original and expanded population sizes differ")
    plan_path = args.additional / "plan/manifest.json"
    if file_sha256(plan_path) != "81d2f64c26594ec2c1ecc8f4cdb7adfabf799901ab51d8024563713d039bf0bd":
        raise ValueError("Original completed additional345 plan changed")
    plan = json.loads(plan_path.read_text())
    if set(expanded.study_uid) - set(baseline.study_uid) != set(plan["training_uids"]):
        raise ValueError("Added fitting rows differ from actual verified cache membership")
    preserved = json.loads((args.additional / "publication/all-verified.json").read_text())
    if (
        preserved["completed_shards"] != 7
        or preserved["objects_fully_download_size_SHA_verified"] != 42
    ):
        raise ValueError("Actual complete additional345 private preservation required")
    args.output.mkdir(mode=0o700)
    overlay = args.output / "expanded-overlay.json"
    old_overlay = Path("/tmp/rsna-public-label-repeat-inputs-20261012-v1/overlay.json")
    old_sha = "c8fd2e860d8a3913dbeab34a9d1078316a8f893360124eac088453dc2ed6cf51"
    source_sha = "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4"
    proof = build_overlay(
        frame,
        folds,
        old_overlay,
        old_sha,
        args.source_csv,
        source_sha,
        overlay,
        release_sha256=release["manifest_sha256"],
        folds_sha256=file_sha256(fold_root / "fold.csv"),
    )
    old_labels, _ = load_fitting_label_source(
        baseline,
        outer,
        old_overlay,
        old_sha,
        release_sha256=release["manifest_sha256"],
        folds_sha256=file_sha256(fold_root / "fold.csv"),
        fold=0,
    )
    labels, _ = load_fitting_label_source(
        expanded,
        outer,
        overlay,
        proof["overlay_sha256"],
        release_sha256=release["manifest_sha256"],
        folds_sha256=file_sha256(fold_root / "fold.csv"),
        fold=0,
    )
    series_path = (
        args.artifact_root
        / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv"
    )
    if file_sha256(series_path) != release["series_sha256"]:
        raise ValueError("Accepted original series sampling table changed")
    series = pd.read_csv(series_path)
    roots = [args.cache] + [
        args.additional / f"outputs/shard-{i:03d}/side-384/cache" for i in range(7)
    ]
    union = MetadataUnion.build(
        args.output / "metadata-cache",
        roots,
        series,
        set(expanded.study_uid) | set(outer.study_uid),
        max_bytes=960 * 2**20,
    )
    if (
        union.value["studies"] != 3748
        or union.value["series"] != 20675
        or union.value["side"] != 384
    ):
        raise ValueError("Complete expanded/legacy-outer native cache membership differs")
    kwargs = dict(
        slices=16,
        side=384,
        max_series=6,
        augment=False,
        slice_sampling="uniform",
        series_sampling="selected",
    )
    old_data = CompetitiveStudyDataset(old_labels, series, args.cache, soft_targets=True, **kwargs)
    old_outer = CompetitiveStudyDataset(outer, series, args.cache, soft_targets=False, **kwargs)
    fitting = CompetitiveStudyDataset(
        labels, series, union.cache, volume_provider=union, soft_targets=True, **kwargs
    )
    evaluation = CompetitiveStudyDataset(
        outer, series, union.cache, volume_provider=union, soft_targets=False, **kwargs
    )
    lookup = {uid: i for i, uid in enumerate(fitting.studies.study_uid)}
    pixel_hash = hashlib.sha256()
    label_hash = hashlib.sha256()
    for before, after, indices in [
        (old_data, fitting, [lookup[u] for u in old_labels.study_uid.head(16)]),
        (old_outer, evaluation, list(range(16))),
    ]:
        for i, j in enumerate(indices):
            left, right = before[i], after[j]
            if set(left) != set(right):
                raise ValueError("Native loader sample fields changed")
            for name, value in left.items():
                if isinstance(value, torch.Tensor):
                    if not torch.equal(value, right[name]):
                        raise ValueError("Actual old fitting/outer tensor changed: " + name)
                elif value != right[name]:
                    raise ValueError("Native sample identity differs")
            pixel_hash.update(right["images"].numpy().tobytes())
            for name in ["labels", "label_mask"]:
                label_hash.update(right[name].numpy().tobytes())
    added = sorted(
        set(expanded.study_uid) - set(baseline.study_uid),
        key=lambda x: hashlib.sha256(x.encode()).hexdigest(),
    )[:16]
    added_hash = hashlib.sha256()
    for uid in added:
        sample = fitting[lookup[uid]]
        if (
            not torch.isfinite(sample["images"]).all()
            or not sample["slice_mask"].any()
            or not sample["label_mask"].eq(1).all()
        ):
            raise ValueError("New actual native fitting input/source mask invalid")
        added_hash.update(sample["images"].numpy().tobytes())
        for name in ["labels", "label_mask"]:
            added_hash.update(sample[name].numpy().tobytes())
    if torch.cuda.is_initialized():
        raise ValueError("CPU-only admission initialized CUDA")
    owned = sum(
        p.stat().st_blocks * 512 for p in args.output.parent.rglob("*") if p.is_file() or p.is_dir()
    )
    if owned > 2**30:
        raise ValueError("Private source/metadata/output exceeded explicit1GiB cap")
    result = {
        "status": "actual_expanded2592_source_and_native_inputs_CPU_verified",
        "CPU_job": os.environ["SLURM_JOB_ID"],
        "node": "evc33",
        "population": population,
        "overlay": proof,
        "private_overlay": str(overlay),
        "metadata_union_index_sha256": file_sha256(union.cache / "index.json"),
        "private_metadata_union": str(union.cache / "index.json"),
        "union_studies": 3748,
        "union_series": 20675,
        "old_fitting_engineering_studies": 16,
        "old_outer_engineering_studies": 16,
        "old_image_tensor_parity": True,
        "all_old_labels_masks_and_outer_unchanged": True,
        "old_engineering_image_sha256": pixel_hash.hexdigest(),
        "old_engineering_labels_masks_sha256": label_hash.hexdigest(),
        "new_fitting_engineering_studies": 16,
        "new_fitting_engineering_tensor_sha256": added_hash.hexdigest(),
        "input_numerical_modules_unchanged": names,
        "source_MRI_payload_bytes_copied": 0,
        "actual_owned_physical_bytes": owned,
        "new_shared_bytes_files": 0,
        "encoder_optimizer_GPU_or_accuracy_executed": False,
        "new_training_dispatched": False,
        "fixed16_expanded_updates_if_later_admitted": 5184,
        "expanded16epochs_is_not_a_compute_matched4496_update_comparison": True,
    }
    with (args.output / "accepted.json").open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2)
    print(
        json.dumps({**result, "accepted_sha256": file_sha256(args.output / "accepted.json")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
