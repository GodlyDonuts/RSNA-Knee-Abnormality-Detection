"""Bounded owning-node producer for the source-bound public ordered-triplet cache.

NOT RUN by the authoring session. Default mode only prints a sanitized plan (roster counts,
storage estimate, free space); `--execute` renders already-present original fold0 studies
through the hashed `dino_adapter.render_inputs` once each and writes the uint8 cache.
It never acquires images, dispatches jobs or reads labels; production still requires the
existing coordinator/resource review.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rsna.competitive import public_input as pi  # noqa: E402
from scripts.newton.compare_public_dinov2_inputs import (  # noqa: E402
    RAW_SPLIT,
    file_sha256,
    member_pixel_group,
)
from tools.research.public_dinov2_input_reference import provenance, reference  # noqa: E402

POPULATION = (2247, 1156)
MEMBER_BYTES = 6 * 12 * 336 * 336 + 4096


def run(args: argparse.Namespace) -> dict:
    if os.uname().nodename != "evc31" or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("Original owning-node CPU-only execution required")
    policy = json.loads(Path(args.config).read_text())["public_input"]
    pi.validate_public_input(policy)
    study_root = Path(args.dicom_root) / RAW_SPLIT
    if not study_root.is_dir():
        raise ValueError("Original raw MRI split is unavailable")
    if file_sha256(Path(args.checkpoint)) != provenance.CHECKPOINT_SHA256:
        raise ValueError("Checkpoint bytes differ from the audited public member")
    group = member_pixel_group(Path(args.member_manifest).read_bytes())["pixel_group"]
    if pi.canonical_sha256(group) != pi.ADMITTED_PIXEL_GROUP_SHA256:
        raise ValueError("Member pixel group differs from the admitted source-specific group")

    from rsna.competitive.folds import load_folds
    from rsna.competitive.train import select_development
    from rsna.release import load_verified_release

    frame, binding = load_verified_release(Path(args.studies), Path(args.artifact_root))
    folds = load_folds(Path(args.folds), frame, binding)
    fitting, outer = select_development(frame, folds, 0)
    if (len(fitting), len(outer)) != POPULATION:
        raise ValueError("Original fitting/evaluation population changed")
    fitting_uids = sorted(map(str, fitting.study_uid))
    evaluation_uids = sorted(map(str, outer.study_uid))
    total = len(fitting_uids) + len(evaluation_uids)
    out = Path(args.out_cache)
    free = shutil.disk_usage(out if out.exists() else out.parent).free
    estimate = total * MEMBER_BYTES
    plan = {
        "kind": pi.KIND,
        "fitting": len(fitting_uids),
        "evaluation": len(evaluation_uids),
        "max_studies": args.max_studies,
        "estimated_bytes": estimate,
        "free_bytes": free,
        "source_revision": binding["source_revision"],
        "executed": False,
    }
    if total > args.max_studies:
        raise ValueError("Declared roster exceeds --max-studies")
    if free - estimate < args.min_free_gib * 2**30:
        raise ValueError("Insufficient free space for the bounded cache")
    if not args.execute:
        return plan
    namespace, _ = reference.load(Path(args.source).read_bytes())
    reference.configure(namespace, group)
    series = pd.read_csv(args.series)
    plane_map = dict(zip(series.SeriesInstanceUID.astype(str), series.Anatomical_Plane))
    plan["index_sha256"] = pi.produce_public_triplet_cache(
        out,
        public_input=policy,
        namespace=namespace,
        study_root=study_root,
        plane_map=plane_map,
        source_revision=binding["source_revision"],
        fitting_uids=fitting_uids,
        evaluation_uids=evaluation_uids,
        max_studies=args.max_studies,
    )
    plan["executed"] = True
    return plan


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "studies",
        "artifact-root",
        "folds",
        "series",
        "dicom-root",
        "source",
        "member-manifest",
        "checkpoint",
        "out-cache",
    ):
        parser.add_argument(f"--{name}", required=True)
    parser.add_argument(
        "--config",
        default=str(
            ROOT / "initiative/experiments/competitive-v1/public-input-recipe-v1"
            "/public-triplet-config.json"
        ),
    )
    parser.add_argument("--max-studies", type=int, required=True)
    parser.add_argument("--min-free-gib", type=float, default=50.0)
    parser.add_argument("--execute", action="store_true")
    print(json.dumps(run(parser.parse_args(argv)), sort_keys=True))


if __name__ == "__main__":
    main()
