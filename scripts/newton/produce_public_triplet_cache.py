"""Prepared bounded producer of the source-bound public ordered-triplet cache.

NOT RUN by the authoring session and NOT a dispatch: it must be submitted only by the
existing coordinator after resource review. Requires evc31, CPU only, the existing
original fitting/evaluation MRI under <dicom-root>/train_series (never acquired here),
the exact v15 source, the admitted pixel group and the hashed adapter. It renders the
complete fold-0 2247/1156 roster once into a fresh cache directory, refusing up front
when the member cap, byte cap or free-space reserve could be exceeded. Labels are not
read; stdout/--out receive only hashes, counts, bytes and time, never study IDs.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rsna.competitive import public_input  # noqa: E402

RAW_SPLIT = "train_series"
FOLD = 0
ROSTER = (2247, 1156)
CONFIG = (
    ROOT / "initiative/experiments/competitive-v1/public-input-recipe-v1/public-triplet-config.json"
)


def run(args: argparse.Namespace) -> dict:
    if os.uname().nodename != "evc31" or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("Original owning-node CPU-only execution required")
    torch.set_num_threads(1)
    policy = json.loads(CONFIG.read_text())["public_input"]
    public_input.validate_public_input(policy)

    from rsna.competitive.folds import load_folds
    from rsna.competitive.train import select_development
    from rsna.release import load_verified_release

    frame, binding = load_verified_release(Path(args.studies), Path(args.artifact_root))
    folds = load_folds(Path(args.folds), frame, binding)
    fitting, outer = select_development(frame, folds, FOLD)
    if (len(fitting), len(outer)) != ROSTER:
        raise ValueError("Original fitting/evaluation population changed")
    series = pd.read_csv(args.series)
    plane_map = dict(zip(series.SeriesInstanceUID.astype(str), series.Anatomical_Plane))
    receipt = public_input.produce_public_triplet_cache(
        Path(args.cache),
        policy=policy,
        study_root=Path(args.dicom_root) / RAW_SPLIT,
        plane_map=plane_map,
        fitting_uids=list(map(str, fitting.study_uid)),
        evaluation_uids=list(map(str, outer.study_uid)),
        source_revision=binding["source_revision"],
        max_members=args.max_members,
        max_output_bytes=args.max_output_bytes,
        reserve_free_bytes=args.reserve_free_bytes,
    )
    return {
        **receipt,
        "fold": FOLD,
        "roster": list(ROSTER),
        "release_sha256": binding["manifest_sha256"],
        "source_revision": binding["source_revision"],
        "policy": policy,
        "GPU_used": False,
        "raw_images_acquired": False,
        "clinical_accuracy_claimed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("studies", "artifact-root", "folds", "series", "dicom-root", "cache"):
        parser.add_argument("--" + name, required=True)
    for name in ("max-members", "max-output-bytes", "reserve-free-bytes"):
        parser.add_argument("--" + name, type=int, required=True)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = json.dumps(run(args), indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
