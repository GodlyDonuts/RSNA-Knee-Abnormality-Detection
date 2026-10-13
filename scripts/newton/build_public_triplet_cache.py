"""Prepared owning-node CPU-only producer of the public ordered-triplet cohort cache.

NOT RUN by the authoring session; root reviews assets, quota and funding before any use.
Requires evc31, no GPU, the exact v15 source bytes, the private Pilkwang member manifest,
the verified release/folds, the archived original inventory and the original raw split.
Each study's exact inventory files are copied (bounded, hash-verified) into a private
transport scratch, rendered once via ``rsna.competitive.public_cache_producer`` and removed
before the next study. Pixels, the private index, receipts, case membership and clinical
values stay under ``--private-out`` on the node; stdout/--out receive sanitized aggregates
only. Never dispatches jobs, downloads data or re-acquires the retained eight-study packet.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from rsna.competitive import public_cache_producer as producer  # noqa: E402
from rsna.competitive import public_input as pi  # noqa: E402

NODE = "evc31"
RECIPE = ROOT / "initiative/experiments/competitive-v1/public-input-recipe-v1"
CONFIG = RECIPE / "public-triplet-config.json"
OVERLAY = ROOT / "initiative/experiments/competitive-v1/public-label-source-v1/fold0-overlay.json"
INVENTORY_PROVENANCE = ROOT / "initiative/source/inventory-provenance.json"
RETAINED_PACKET = Path("/tmp/rsna-public-dinov2-real-inputs-20261013-v2")
FUNDING_KEYS = {
    "schema_version",
    "node",
    "max_total_bytes",
    "min_free_bytes",
    "max_study_source_bytes",
    "max_studies_per_run",
}
MODES = ("stage", "reconcile", "finalize")


def file_sha256(path: Path) -> str:
    return producer._file_sha256(Path(path))


def check_runtime() -> dict:
    """Original owning node, CPU only: no scheduler GPU and an explicitly hidden CUDA device."""
    import torch

    if os.uname().nodename != NODE:
        raise ValueError("Original owning-node execution required")
    if os.environ.get("SLURM_JOB_GPUS") or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only execution requires CUDA_VISIBLE_DEVICES='' and no job GPU")
    if torch.cuda.is_available():
        raise ValueError("A CUDA device is visible; CPU-only execution required")
    torch.set_num_threads(1)
    import numpy
    import pydicom

    return {
        "node": NODE,
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "pydicom": pydicom.__version__,
        "threads": 1,
    }


def private_output(path: Path) -> Path:
    """Absolute, owner-only, non-symlinked directory outside the repository and retained packet."""
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("Private output must be an absolute non-symlink path")
    resolved = path.resolve()
    for protected in (ROOT.resolve(), RETAINED_PACKET):
        if resolved == protected or protected in resolved.parents:
            raise ValueError("Private output may not live in Git or the retained raw packet")
    if not path.exists():
        path.mkdir(mode=0o700)
    status = path.lstat()
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.getuid()
        or stat.S_IMODE(status.st_mode) & 0o077
    ):
        raise ValueError("Private output must be an owner-only directory")
    return path


def load_funding(path: Path, expected_sha256: str) -> dict:
    """Root-reviewed funding receipt bound by its exact bytes."""
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Funding receipt differs from its bound SHA256")
    funding = json.loads(payload)
    if (
        not isinstance(funding, dict)
        or set(funding) != FUNDING_KEYS
        or funding["schema_version"] != 1
        or funding["node"] != NODE
        or any(
            type(funding[key]) is not int or funding[key] < 1
            for key in FUNDING_KEYS - {"schema_version", "node"}
        )
    ):
        raise ValueError("Funding receipt fields differ")
    return funding


def member_pixel_group(manifest: bytes) -> dict:
    from tools.research.public_dinov2_input_reference import provenance

    members = [
        m for m in json.loads(manifest).get("members", []) if m.get("file") == provenance.CHECKPOINT
    ]
    if len(members) != 1:
        raise ValueError("Manifest must list the audited member exactly once")
    group = members[0]["pixel_group"]
    return json.loads(group) if isinstance(group, str) else dict(group)


def load_inventory(path: Path, source_revision: str) -> dict[str, list[dict]]:
    """Archived original inventory -> per-study train records (exact producer fields)."""
    import pandas as pd

    receipt = json.loads(INVENTORY_PROVENANCE.read_text())
    if receipt["source_revision"] != source_revision:
        raise ValueError("Original source revision differs from the archived inventory")
    inventory = pd.read_parquet(path)
    if len(inventory) != receipt["rows"]:
        raise ValueError("Inventory rows differ from the archived provenance")
    rows = inventory[inventory.split.eq(producer.SPLIT)][sorted(producer.RECORD_KEYS)]
    rows = rows.astype({"bytes": "int64"})
    return {
        str(uid): [{k: (int(v) if k == "bytes" else str(v)) for k, v in r.items()} for r in group]
        for uid, group in (
            (uid, frame.to_dict("records")) for uid, frame in rows.groupby("study_uid", sort=True)
        )
    }


def transport_study(
    source_root: Path, scratch: Path, uid: str, records: list[dict], funding: dict
) -> Path:
    """Copy exactly one study's inventory files, bounded before any read and hash-verified."""
    total = sum(r["bytes"] for r in records)
    if total > funding["max_study_source_bytes"]:
        raise ValueError("Study source exceeds the funded transport bound")
    if shutil.disk_usage(scratch).free - total < funding["min_free_bytes"]:
        raise ValueError("Study transport would breach the free-disk reserve")
    producer.study_source_sha256(uid, records)
    destination = scratch / producer.SPLIT / uid
    if destination.exists() or destination.is_symlink():
        raise ValueError("Transport scratch is not empty")
    for record in records:
        relative = PurePosixPath(record["path"])
        source = Path(source_root).joinpath(*relative.parts)
        if source.is_symlink() or any(p.is_symlink() for p in source.parents):
            raise ValueError("Original source paths may not traverse symlinks")
        target = scratch.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        digest, copied = hashlib.sha256(), 0
        with source.open("rb") as reader, target.open("xb") as writer:
            while block := reader.read(min(1 << 20, record["bytes"] + 1 - copied)):
                copied += len(block)
                if copied > record["bytes"]:
                    raise ValueError("Original source file exceeds its inventory bytes")
                digest.update(block)
                writer.write(block)
        if copied != record["bytes"] or digest.hexdigest() != record["sha256"]:
            raise ValueError("Original source bytes differ from the inventory")
    return destination


def aggregate(cache: Path, status: dict) -> dict:
    """Counts and digests only: no study/series UID, path, pixel or clinical value."""
    root = cache / pi.CACHE_DIRECTORY
    occupied: Counter = Counter()
    slots: Counter = Counter()
    laterality: Counter = Counter()
    roles: Counter = Counter()
    for name in sorted(status["states"].get("complete", [])):
        receipt = json.loads((root / producer.RECEIPTS / f"{name}.json").read_text())
        occupied[str(receipt["occupied_slots"])] += 1
        slots.update(str(i) for i, s in enumerate(receipt["slot_series"]) if s)
        laterality[str(receipt["laterality"])] += 1
        roles[receipt["role"]] += 1
    return {
        "intent_sha256": status["intent_sha256"],
        "counts": status["counts"],
        "unexpected_entries": len(status["unexpected"]),
        "finalized": status["finalized"],
        "committed_bytes": status["committed_bytes"],
        "complete_roles": dict(sorted(roles.items())),
        "occupied_slots_histogram": dict(sorted(occupied.items())),
        "per_slot_occupied": dict(sorted(slots.items())),
        "laterality": dict(sorted(laterality.items())),
    }


def run(args: argparse.Namespace) -> dict:
    runtime = check_runtime()
    out = private_output(Path(args.private_out))
    cache, scratch = out / "cache", out / "transport"
    for directory in (cache, scratch):
        directory.mkdir(mode=0o700, exist_ok=True)
    funding = load_funding(Path(args.funding), args.funding_sha256)
    config_bytes = CONFIG.read_bytes()
    policy = json.loads(config_bytes)["public_input"]
    pi.validate_public_input(policy)
    if args.mode == "reconcile":
        status = producer.reconcile(cache)
        return {"mode": args.mode, **aggregate(cache, status), "clinical_evidence": False}
    if args.mode == "finalize":
        index_sha = producer.finalize(cache, policy=policy)
        status = producer.reconcile(cache)
        return {
            "mode": args.mode,
            "index_sha256": index_sha,
            **aggregate(cache, status),
            "clinical_evidence": False,
        }

    from rsna.competitive.folds import load_folds
    from rsna.competitive.supervision import roster_sha
    from rsna.competitive.train import select_development
    from rsna.release import load_verified_release

    source_root = Path(args.dicom_root)
    if source_root.resolve() == RETAINED_PACKET or RETAINED_PACKET in source_root.resolve().parents:
        raise ValueError("The retained eight-study packet is not the cohort source")
    if not (source_root / producer.SPLIT).is_dir() or source_root.is_symlink():
        raise ValueError("Original raw MRI split is unavailable")
    source_bytes = Path(args.source).read_bytes()
    manifest = Path(args.member_manifest).read_bytes()
    pixel_group = member_pixel_group(manifest)
    renderer = producer.Renderer(policy, pixel_group, source_bytes)
    frame, release = load_verified_release(Path(args.studies), Path(args.artifact_root))
    folds = load_folds(Path(args.folds), frame, release)
    fitting, outer = select_development(frame, folds, 0)
    overlay = json.loads(OVERLAY.read_text())
    if (
        roster_sha(fitting) != overlay["fitting_uid_sha256"]
        or roster_sha(outer) != overlay["outer_uid_sha256"]
        or release["manifest_sha256"] != overlay["release_sha256"]
    ):
        raise ValueError("Fitting/development rosters differ from the accepted hashes")
    records = load_inventory(Path(args.inventory), args.source_revision)
    fitting_uids, evaluation_uids = (
        list(map(str, fitting.study_uid)),
        list(map(str, outer.study_uid)),
    )
    cohort = sorted(fitting_uids + evaluation_uids)
    if not set(cohort) <= set(records):
        raise ValueError("Cohort studies are absent from the original inventory")
    import pandas as pd

    series = pd.read_csv(args.series)
    plane_map = dict(zip(series.SeriesInstanceUID.astype(str), series.Anatomical_Plane))
    binding = {
        "runtime": runtime,
        "funding_sha256": args.funding_sha256,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "member_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "release_sha256": release["manifest_sha256"],
        "folds_sha256": overlay["folds_sha256"],
        "fitting_uid_sha256": overlay["fitting_uid_sha256"],
        "outer_uid_sha256": overlay["outer_uid_sha256"],
        "series_csv_sha256": file_sha256(Path(args.series)),
        "inventory_source_revision": args.source_revision,
    }
    try:
        intent, _ = producer.load_intent(cache, policy)
        if intent["binding"] != binding or intent["caps"] != {
            "max_total_bytes": funding["max_total_bytes"],
            "min_free_bytes": funding["min_free_bytes"],
        }:
            raise ValueError("Existing cache intent was bound to other inputs or funding")
    except producer.ProducerError as error:
        if (cache / pi.CACHE_DIRECTORY).exists():
            raise ValueError("Existing cache operation is unreadable; reconcile only") from error
        producer.create_intent(
            cache,
            policy=policy,
            pixel_group=pixel_group,
            source_revision=args.source_revision,
            fitting_uids=fitting_uids,
            evaluation_uids=evaluation_uids,
            source_studies={u: producer.study_source_sha256(u, records[u]) for u in cohort},
            caps={k: funding[k] for k in ("max_total_bytes", "min_free_bytes")},
            binding=binding,
        )
    status = producer.reconcile(cache, verify_bytes=False)
    pending = status["states"].get("pending", [])[: funding["max_studies_per_run"]]
    outcome: Counter = Counter()
    refusals: Counter = Counter()
    for uid in pending:
        for leftover in scratch.iterdir():
            shutil.rmtree(leftover) if leftover.is_dir() else leftover.unlink()
        try:
            transport_study(source_root, scratch, uid, records[uid], funding)
            receipt = producer.stage_member(
                cache, uid, scratch, records[uid], plane_map, renderer=renderer
            )
            outcome["reused" if receipt["reused"] else "rendered"] += 1
        except Exception as error:  # type only; no IDs or messages leave the node
            refusals[type(error).__name__] += 1
        finally:
            shutil.rmtree(scratch / producer.SPLIT, ignore_errors=True)
    status = producer.reconcile(cache)
    return {
        "mode": args.mode,
        "attempted": len(pending),
        "outcome": dict(sorted(outcome.items())),
        "refusals": dict(sorted(refusals.items())),
        "binding": {k: v for k, v in binding.items() if k != "runtime"} | {"runtime": runtime},
        **aggregate(cache, status),
        "clinical_evidence": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--private-out", required=True)
    parser.add_argument("--funding", required=True)
    parser.add_argument("--funding-sha256", required=True)
    for name in ("source", "member-manifest", "studies", "artifact-root", "folds", "series"):
        parser.add_argument("--" + name)
    for name in ("inventory", "dicom-root", "source-revision"):
        parser.add_argument("--" + name)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    required = ("source", "member_manifest", "studies", "artifact_root", "folds", "series")
    if args.mode == "stage" and any(
        getattr(args, k) is None for k in required + ("inventory", "dicom_root", "source_revision")
    ):
        parser.error("stage requires every source/input binding")
    text = json.dumps(run(args), indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
