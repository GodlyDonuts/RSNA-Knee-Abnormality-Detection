"""CPU-only exact original-window parity and bounded dense-source MRI admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.compact_bag import candidate_windows
from rsna.competitive.compact_dense_infer import (
    POLICY,
    DenseSourceNeighborDataset,
    dense_plan,
)
from rsna.competitive.compact_source_neighbors import (
    MatchedSourceNeighborDataset,
    select_window_tensor,
)
from rsna.competitive.compact_train import admitted_datasets
from rsna.data import select_series


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "artifact-root", "cache", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if os.uname().nodename != "evc33" or os.environ.get("SLURM_JOB_GPUS") or args.output.exists():
        raise ValueError("Original owning-node zero-GPU fresh input admission required")
    torch.set_num_threads(1)
    A = args.artifact_root
    request = argparse.Namespace(
        studies=A / "initiative/releases/report-supervision-v1/studies.parquet",
        series=A
        / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv",
        folds=A / "initiative/experiments/competitive-v1/folds",
        artifact_root=A,
        cache=args.cache,
        fitting_supervision=Path("/tmp/rsna-public-label-repeat-inputs-20261012-v1/overlay.json"),
        engineering=False,
        input_policy="n3",
        loss_policy="unweighted",
    )
    train, outer, binding = admitted_datasets(request)
    original = json.loads(
        Path("/tmp/rsna-compact-matched-neighbor-inputs-20261016-v1/inputs/inputs.json").read_text()
    )["binding"]
    for k in (
        "release",
        "supervision",
        "fitting_uids_sha256",
        "outer_uids_sha256",
        "folds_file_sha256",
        "native_input",
    ):
        if binding[k] != original[k]:
            raise ValueError("Exact original matched-center source cohort/input binding differs")
    if (len(train), len(outer)) != (2247, 1156):
        raise ValueError("Original fitting and outer rosters differ")
    numerical = json.loads(
        Path(
            "/tmp/rsna-compact-native-n3-production-evc33-20261016-v1/production/run.json"
        ).read_text()
    )["binding"]["source_sha256"]
    names = [
        "rsna/competitive/data.py",
        "rsna/competitive/compact_source_neighbors.py",
        "rsna/competitive/neighbor_geometry.py",
        "rsna/competitive/neighbor_input.py",
        "rsna/data.py",
    ]
    # The source-neighbor module postdates original N3, and has its own frozen3803 proof.
    frozen = Path("/tmp/rsna-compact-matched-neighbor-inputs-20261016-v1/code")
    for name in names:
        if file_sha256(args.source / name) != file_sha256(frozen / name):
            raise ValueError("Frozen source-neighbor primitives changed")
        if (
            name in numerical
            and name != "rsna/competitive/compact_source_neighbors.py"
            and file_sha256(args.source / name) != numerical[name]
        ):
            raise ValueError("Original native input primitive changed")
    counts = {}
    digest = hashlib.sha256()
    total_old = total_dense = 0
    for role, base in [("fitting", train), ("outer", outer)]:
        old_count = dense_count = extra_count = legacy = 0
        for uid in base.studies.study_uid:
            for entry in select_series(base.series[uid], 6, False):
                meta = base._metadata[(uid, entry["SeriesInstanceUID"])][1]
                plan, index, fall = dense_plan(meta)
                old_count += len(index)
                dense_count += len(plan)
                extra_count += len(plan) - len(index)
                legacy += fall
        counts[role] = {
            "studies": len(base),
            "original_windows": old_count,
            "dense_windows": dense_count,
            "extra_physical_centers": extra_count,
            "unchanged_legacy_fallback_windows": legacy,
        }
        old_view = MatchedSourceNeighborDataset(base)
        new_view = DenseSourceNeighborDataset(base)
        for i in range(16):
            before = old_view[i]
            after = new_view[i]
            old_tensor = select_window_tensor(
                before, candidate_windows(before["series_mask"], before["slice_mask"])
            )
            if before["study_uid"] != after["study_uid"] or not torch.equal(
                old_tensor, after["dense_windows"][after["original_window_indices"]]
            ):
                raise ValueError("Actual MRI original-window subset tensor/identity changed")
            if (
                len(after["dense_windows"]) > 180
                or not torch.isfinite(after["dense_windows"]).all()
            ):
                raise ValueError("Actual dense source tensors violate budget or finite gate")
            total_old += len(old_tensor)
            total_dense += len(after["dense_windows"])
            digest.update(after["dense_windows"].numpy().tobytes())
            digest.update(after["original_window_indices"].numpy().tobytes())
            digest.update(after["source_plan_sha256"].encode())
    if torch.cuda.is_initialized():
        raise ValueError("CPU input admission must not initialize CUDA")
    args.output.mkdir(mode=0o700)
    proof = {
        "status": "actual_dense30_original_subset_MRI_CPU_verified",
        "CPU_job": os.environ["SLURM_JOB_ID"],
        "node": "evc33",
        "inference_policy": POLICY,
        "fitting_count": 2247,
        "outer_count": 1156,
        "all_metadata_census": counts,
        "engineering_studies": 32,
        "original_windows_bit_equal": True,
        "original_labels_masks_source_geometry_unchanged": True,
        "engineering_original_windows": total_old,
        "engineering_dense_windows": total_dense,
        "engineering_tensor_sha256": digest.hexdigest(),
        "model_GPU_optimizer_or_accuracy_executed": False,
        "source_MRI_payload_bytes_copied": 0,
        "target_runtime_qualified": False,
        "source_sha256": {
            name: file_sha256(args.source / name)
            for name in names + ["rsna/competitive/compact_dense_infer.py"]
        },
    }
    (args.output / "accepted.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(
        json.dumps({**proof, "accepted_sha256": file_sha256(args.output / "accepted.json")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
