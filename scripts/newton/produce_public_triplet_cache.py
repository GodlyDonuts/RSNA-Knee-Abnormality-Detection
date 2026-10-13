"""Prepared streaming producer for the full-cohort public ordered-triplet cache.

NOT RUN by the authoring session; no job is dispatched here. CPU only. Renders every
fold-0 fitting and evaluation study of the accepted release through the exact hashed
Pilkwang v15 nodes and ``dino_adapter.render_inputs``, then commits each uint8 row
once via ``rsna.competitive.public_input.stream_public_triplet_cache``. Labels are
never read. Use ``--resume`` after an interruption; completed caches are never
overwritten. Stdout receives only aggregate provenance (no study IDs or pixels).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rsna.competitive import public_input  # noqa: E402
from rsna.competitive.folds import load_folds  # noqa: E402
from rsna.competitive.train import select_development  # noqa: E402
from rsna.release import load_verified_release  # noqa: E402
from tools.research.public_dinov2_input_reference import (  # noqa: E402
    cache_producer,
    provenance,
    reference,
)

CONFIG = ROOT / "initiative/experiments/competitive-v1/public-input-recipe-v1"
CONFIG = CONFIG / "public-triplet-config.json"
RAW_SPLIT = "train_series"
EXPECTED_POPULATION = (2247, 1156)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def member_pixel_group(manifest: bytes, checkpoint: str = provenance.CHECKPOINT) -> dict:
    members = [m for m in json.loads(manifest).get("members", []) if m.get("file") == checkpoint]
    if len(members) != 1:
        raise ValueError(f"Manifest must list {checkpoint} exactly once")
    group = members[0]["pixel_group"]
    return json.loads(group) if isinstance(group, str) else dict(group)


def run(args: argparse.Namespace) -> dict:
    if os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("CPU-only cache production required; no GPU allocation")
    torch.set_num_threads(1)
    config = json.loads(CONFIG.read_text())
    policy = config["public_input"]
    public_input.validate_public_input(policy)
    if config["fit_scope"] != "outer_fold":
        raise ValueError("Only the matched outer-fold comparison cohort is produced")
    split_root = Path(args.dicom_root) / RAW_SPLIT
    if not split_root.is_dir():
        raise ValueError("Original raw MRI split is unavailable")
    if file_sha256(Path(args.checkpoint)) != provenance.CHECKPOINT_SHA256:
        raise ValueError("Checkpoint bytes differ from the audited public member")
    pixel_group = member_pixel_group(Path(args.member_manifest).read_bytes())
    namespace, receipt = reference.load(Path(args.source).read_bytes(), [])
    reference.configure(namespace, pixel_group)
    frame, binding = load_verified_release(Path(args.studies), Path(args.artifact_root))
    folds = load_folds(Path(args.folds), frame, binding)
    fitting, outer = select_development(frame, folds, config["fold"], config["fit_scope"])
    if (len(fitting), len(outer)) != EXPECTED_POPULATION:
        raise ValueError("Original fitting/evaluation population changed")
    series_bytes = Path(args.series).read_bytes()
    if hashlib.sha256(series_bytes).hexdigest() != binding["series_sha256"]:
        raise ValueError("Series table differs from the accepted data release")
    series = pd.read_csv(Path(args.series))
    plane_map = dict(zip(series.SeriesInstanceUID.astype(str), series.Anatomical_Plane))
    render = cache_producer.source_row_renderer(
        policy, namespace, receipt, pixel_group, split_root, plane_map
    )
    started = time.monotonic()
    index_sha256 = public_input.stream_public_triplet_cache(
        Path(args.cache),
        render,
        pixel_group=pixel_group,
        source_revision=binding["source_revision"],
        fitting_uids=list(map(str, fitting.study_uid)),
        evaluation_uids=list(map(str, outer.study_uid)),
        resume=args.resume,
    )
    elapsed = time.monotonic() - started
    cohort = public_input.verify_public_cohort(
        Path(args.cache), policy, fitting, outer, binding["source_revision"]
    )
    if cohort["index_sha256"] != index_sha256:
        raise ValueError("Published index differs from the produced index")
    index, _ = public_input.load_public_index(Path(args.cache), policy, binding["source_revision"])
    return {
        "status": "produced_not_admitted",
        "resumed": bool(args.resume),
        "elapsed_seconds_this_invocation": round(elapsed, 3),
        "member_bytes": sum(m["bytes"] for m in index["members"].values()),
        "renderer": receipt["source_ref"],
        "source_sha256": receipt["source_sha256"],
        "node_text_sha256": receipt["node_text_sha256"],
        "source_revision": binding["source_revision"],
        **cohort,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    names = ("source", "member-manifest", "checkpoint", "studies", "artifact-root", "folds")
    for name in names + ("series", "cache", "dicom-root"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    text = json.dumps(run(args), indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
