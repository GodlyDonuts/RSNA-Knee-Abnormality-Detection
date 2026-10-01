"""Actual Stanford tensor/label admission, CPU-only and no model or optimizer."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from rsna.competitive.mrnet import MRNET_TARGETS, MRNetStudyDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--acquisition-receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if not os.environ.get("SLURM_JOB_ID") or torch.cuda.is_initialized():
        raise ValueError("Actual CPU Slurm admission required")
    if args.output.exists():
        raise FileExistsError("Original Stanford input probe already exists")
    records = []
    for split, count, missing in (("train", 1129, "0000"), ("valid", 119, "1130")):
        data = MRNetStudyDataset(args.root, args.acquisition_receipt, split)
        if (
            len(data) != count
            or missing in data.uids
            or data.training_permitted is not (split == "train")
        ):
            raise ValueError("Exact source supervised split required")
        for index in (0, len(data) // 3, 2 * len(data) // 3, len(data) - 1):
            item = data[index]
            assert item["images"].shape == (3, 16, 1, 256, 256)
            assert torch.isfinite(item["images"]).all()
            assert item["images"].min() >= 0 and item["images"].max() <= 1
            assert (
                item["labels"].shape == (3,)
                and ((item["labels"] == 0) | (item["labels"] == 1)).all()
            )
            records.append(
                {
                    "source_case": item["source_case_id"],
                    "source_split": split,
                    "shape": list(item["images"].shape),
                    "observed_slices_per_plane": item["slice_mask"].sum(1).tolist(),
                    "tensor_sha256": hashlib.sha256(item["images"].numpy().tobytes()).hexdigest(),
                    "labels_sha256": hashlib.sha256(item["labels"].numpy().tobytes()).hexdigest(),
                }
            )
    result = {
        "status": "actual_Stanford_three_plane_exam_tensors_verified",
        "job_id": os.environ["SLURM_JOB_ID"],
        "node": os.uname().nodename,
        "source_targets": list(MRNET_TARGETS),
        "source_train_exams": 1129,
        "source_valid_exams": 119,
        "validation_training_permitted": False,
        "cases_probed": 8,
        "model_ran": False,
        "optimizer_updates": 0,
        "source_weights_trained": False,
        "records": records,
        "acquisition_receipt_sha256": hashlib.sha256(
            args.acquisition_receipt.read_bytes()
        ).hexdigest(),
    }
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}))


if __name__ == "__main__":
    main()
