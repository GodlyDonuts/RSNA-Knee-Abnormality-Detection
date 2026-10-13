"""Prepared owning-node CPU-only producer of the public ordered-triplet cohort cache.

NOT RUN by the authoring session; root reviews assets, quota and funding before any use.
Supported placement is an explicitly funded evc41 CPU-only run; evc31 is accepted only as an
independently funded alternate. Funding binds the node/placement, zero step GPUs and the exact
wrapper, producer, source-table and runtime hashes. Source objects come from
``public_source_table.accepted_source_objects`` (full baseline inventory, exact correction
plan/receipt, release series CSV, 2247/1156 rosters) and are converted to producer records
only after every identity is bound. Each study's ACCEPTED series are copied (bounded,
hash-verified, original ``os.scandir`` series order) into a private transport directory, so
unaccepted series never reach the renderer header/slot/laterality probe; a source-order
receipt binds that order and n_slices ties. The retained eight-study packet supplies verified
unchanged historical originals read-only; every other object comes from an already acquired
corrected-revision root. Rendering uses ``rsna.competitive.public_cache_producer`` unchanged,
the transport copy is removed after each study and stdout/--out carry sanitized aggregates
only. Never dispatches jobs, downloads or deletes/alters original data.
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
from rsna.competitive import public_source_table as pst  # noqa: E402

PLACEMENTS = {"evc41": "primary", "evc31": "independently_funded_alternate"}
RECIPE = ROOT / "initiative/experiments/competitive-v1/public-input-recipe-v1"
CONFIG = RECIPE / "public-triplet-config.json"
OVERLAY = ROOT / "initiative/experiments/competitive-v1/public-label-source-v1/fold0-overlay.json"
INVENTORY_DIR = ROOT / "initiative/source/inventory"
INVENTORY_PARTS = 9
CORRECTION = ROOT / "initiative/audits/source-pixel-representation-v1"
CORRECTION_PLAN = CORRECTION / "source_correction_plan.json"
CORRECTION_RECEIPT = CORRECTION / "hf-source-correction-receipt.json"
RETAINED_PACKET = Path("/tmp/rsna-public-dinov2-real-inputs-20261013-v2")
PACKET_ROSTER = "private-original-roster-and-files.json"
PACKET_PROOF = "plan-admitted.json"
PACKET_FILES = "original"
ORDER_DIRECTORY = "source-order"
DEVICE_LIST_VARIABLES = ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "GPU_DEVICE_ORDINAL")
DEVICE_COUNT_VARIABLES = ("SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_GPUS_PER_NODE")
LIMIT_KEYS = ("max_total_bytes", "min_free_bytes", "max_study_source_bytes", "max_studies_per_run")
HASH_KEYS = ("wrapper_sha256", "producer_sha256", "source_table_sha256", "runtime_sha256")
FUNDING_KEYS = {"schema_version", "node", "placement", "cpu_only", "step_gpus"} | set(
    LIMIT_KEYS + HASH_KEYS
)
MODES = ("stage", "reconcile", "finalize")


def file_sha256(path: Path) -> str:
    return producer._file_sha256(Path(path))


def code_hashes() -> dict:
    """Exact bytes of this wrapper, the producer and the accepted source-object binder."""
    return {
        "wrapper_sha256": file_sha256(Path(__file__)),
        "producer_sha256": file_sha256(Path(producer.__file__)),
        "source_table_sha256": file_sha256(Path(pst.__file__)),
    }


def check_runtime() -> dict:
    """Funded CPU-only node: no scheduler/step GPU and an explicitly hidden CUDA device."""
    import torch

    node = os.uname().nodename
    if node not in PLACEMENTS:
        raise ValueError("Execution requires the funded evc41 (or alternate evc31) node")
    if any(os.environ.get(name) for name in DEVICE_LIST_VARIABLES) or any(
        os.environ.get(name, "0") not in {"", "0"} for name in DEVICE_COUNT_VARIABLES
    ):
        raise ValueError("CPU-only execution requires zero job/step GPUs")
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("CPU-only execution requires CUDA_VISIBLE_DEVICES=''")
    if torch.cuda.is_available() or torch.cuda.device_count():
        raise ValueError("A CUDA device is visible; CPU-only execution required")
    torch.set_num_threads(1)
    import numpy
    import pandas
    import pydicom

    return {
        "node": node,
        "placement": PLACEMENTS[node],
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "pydicom": pydicom.__version__,
        "threads": 1,
        "cuda_visible_devices": "",
        "step_gpus": 0,
    }


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def private_output(
    path: Path, protected: list[Path] | tuple[Path, ...] = (), *, create: bool = True
) -> Path:
    """Absolute owner-only directory not overlapping Git, the retained packet or any source."""
    path = Path(path)
    if not path.is_absolute() or path.is_symlink():
        raise ValueError("Private output must be an absolute non-symlink path")
    resolved = path.resolve()
    for root in (ROOT, RETAINED_PACKET, *protected):
        other = Path(root).resolve()
        if _inside(resolved, other) or _inside(other, resolved):
            raise ValueError("Private output may not overlap Git, the retained packet or sources")
    if not path.exists():
        if not create:
            raise ValueError("Read-only reconciliation requires existing private output")
        path.mkdir(mode=0o700)
    status = path.lstat()
    if (
        not stat.S_ISDIR(status.st_mode)
        or status.st_uid != os.getuid()
        or stat.S_IMODE(status.st_mode) & 0o077
    ):
        raise ValueError("Private output must be an owner-only directory")
    return path


def load_funding(path: Path, expected_sha256: str, runtime: dict, hashes: dict) -> dict:
    """Root-reviewed funding receipt bound by exact bytes to this node, code and runtime."""
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise ValueError("Funding receipt differs from its bound SHA256")
    funding = json.loads(payload)
    if (
        not isinstance(funding, dict)
        or set(funding) != FUNDING_KEYS
        or funding["schema_version"] != 2
        or funding["cpu_only"] is not True
        or type(funding["step_gpus"]) is not int
        or funding["step_gpus"] != 0
        or any(type(funding[key]) is not int or funding[key] < 1 for key in LIMIT_KEYS)
    ):
        raise ValueError("Funding receipt fields differ")
    if funding["node"] != runtime["node"] or funding["placement"] != PLACEMENTS.get(
        funding["node"]
    ):
        raise ValueError("Funding node/placement differs from the executing node")
    observed = hashes | {"runtime_sha256": pi.canonical_sha256(runtime)}
    if any(funding[key] != observed[key] for key in HASH_KEYS):
        raise ValueError("Funding wrapper/producer/source-table/runtime hash differs")
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


def read_inventory(directory: Path, parts: int = INVENTORY_PARTS):
    """The complete historical baseline inventory (the correction plan binds its fingerprint)."""
    import pandas as pd

    files = sorted(Path(directory).glob("*.parquet"))
    if len(files) != parts:
        raise ValueError("Complete baseline inventory parts are required")
    frame = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    return frame[pst.COLUMNS] if set(pst.COLUMNS) <= set(frame.columns) else frame


def bind_sources(
    inventory,
    series_path: Path,
    release: dict,
    fitting_uids: list[str],
    evaluation_uids: list[str],
    plan_path: Path,
    receipt_path: Path,
    **binder_options,
) -> tuple[dict[str, list[dict]], dict, dict, set[str]]:
    """Accepted corrected source objects plus exact input hashes and corrected series."""
    import pandas as pd

    from rsna.audit import inventory_fingerprint

    series_bytes = Path(series_path).read_bytes()
    series_sha = hashlib.sha256(series_bytes).hexdigest()
    if series_sha != release["series_sha256"]:
        raise ValueError("Accepted series CSV differs from the verified release")
    plan_bytes, receipt_bytes = Path(plan_path).read_bytes(), Path(receipt_path).read_bytes()
    receipt = json.loads(receipt_bytes)
    objects, summary = pst.accepted_source_objects(
        inventory,
        pd.read_csv(series_path, dtype=str),
        fitting_uids,
        evaluation_uids,
        plan_bytes,
        receipt,
        source_revision=pst.CORRECTED_SOURCE_REVISION,
        **binder_options,
    )
    changes = pst.load_correction(
        plan_bytes,
        receipt,
        **{k: v for k, v in binder_options.items() if k in {"plan_file_sha256", "plan_sha256"}},
    )
    corrected_series = {str(change["before"]["series_uid"]) for change in changes.values()}
    hashes = {
        "inventory_fingerprint_sha256": inventory_fingerprint(inventory),
        "correction_plan_file_sha256": hashlib.sha256(plan_bytes).hexdigest(),
        "correction_receipt_sha256": hashlib.sha256(receipt_bytes).hexdigest(),
        "series_csv_sha256": series_sha,
        "release_sha256": release["manifest_sha256"],
    }
    return objects, summary, hashes, corrected_series


def producer_records(objects: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """Bound 8-column source objects -> producer 6-field records, after identity checks."""
    records = {}
    for uid, rows in objects.items():
        converted = []
        for row in rows:
            if set(row) != set(pst.COLUMNS) or row["split"] != producer.SPLIT:
                raise ValueError("Source object identity fields differ")
            expected = f"{producer.SPLIT}/{uid}/{row['series_uid']}/{row['sop_uid']}.dcm"
            if row["study_uid"] != uid or row["path"] != expected:
                raise ValueError("Source object path differs from its bound identities")
            converted.append({key: row[key] for key in sorted(producer.RECORD_KEYS)})
        producer.study_source_sha256(uid, converted)
        records[uid] = converted
    return records


def retained_studies(
    packet: Path, records: dict[str, list[dict]], corrected_series: set[str]
) -> tuple[dict[str, Path], dict]:
    """Retained eight-study packet studies whose accepted objects are all unchanged originals.

    Historical (52f995) bytes are admitted only where the corrected object equals the
    historical object and no accepted series of that study is touched by the correction plan.
    The packet is only read; nothing in it is moved, rewritten or deleted.
    """
    packet = Path(packet)
    if packet.is_symlink() or not packet.is_dir():
        raise ValueError("Retained packet must be a real directory")
    roster_bytes = (packet / PACKET_ROSTER).read_bytes()
    proof = json.loads((packet / PACKET_PROOF).read_text())
    if hashlib.sha256(roster_bytes).hexdigest() != proof.get("private_inventory_sha256"):
        raise ValueError("Retained packet roster differs from its admitted plan")
    roster = json.loads(roster_bytes)
    if roster.get("source_revision") != pst.HISTORICAL_SOURCE_REVISION:
        raise ValueError("Retained packet must be the historical original source")
    held = {str(row["path"]): row for row in roster["files"]}
    reusable: dict[str, Path] = {}
    refused: Counter = Counter()
    for uid in sorted({PurePosixPath(p).parts[1] for p in held if p.startswith("train_series/")}):
        accepted = records.get(uid)
        if accepted is None:
            refused["outside_cohort"] += 1
        elif {r["series_uid"] for r in accepted} & corrected_series:
            refused["corrected_series_overlap"] += 1
        elif any(
            r["path"] not in held
            or held[r["path"]]["sha256"] != r["sha256"]
            or int(held[r["path"]]["bytes"]) != r["bytes"]
            for r in accepted
        ):
            refused["changed_or_missing_object"] += 1
        else:
            reusable[uid] = packet / PACKET_FILES / producer.SPLIT / uid
    summary = {
        "retained_packet_inventory_sha256": proof["private_inventory_sha256"],
        "retained_studies": len(reusable),
        "retained_files": sum(len(records[u]) for u in reusable),
        "retained_bytes": sum(r["bytes"] for u in reusable for r in records[u]),
        "retained_refused": dict(sorted(refused.items())),
    }
    return reusable, summary


def _real_directory(path: Path) -> None:
    path = Path(path)
    if (
        not path.is_absolute()
        or any(p.is_symlink() for p in (path, *path.parents))
        or not path.is_dir()
    ):
        raise ValueError("Original source paths may not traverse symlinks")


def _copy(source: Path, target: Path, record: dict) -> None:
    digest, copied = hashlib.sha256(), 0
    descriptor = os.open(source, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as reader, target.open("xb") as writer:
        while block := reader.read(min(1 << 20, record["bytes"] + 1 - copied)):
            copied += len(block)
            if copied > record["bytes"]:
                raise ValueError("Original source file exceeds its bound bytes")
            digest.update(block)
            writer.write(block)
    target.chmod(0o400)
    if copied != record["bytes"] or digest.hexdigest() != record["sha256"]:
        raise ValueError("Original source bytes differ from the bound object")


def transport_study(
    source_study: Path,
    scratch: Path,
    uid: str,
    records: list[dict],
    funding: dict,
    *,
    origin: str,
) -> tuple[Path, dict]:
    """Copy only the accepted series of one study, in original scandir order, read-only.

    Unaccepted series present in the source are never opened, copied or probed. The copy's
    ``os.scandir`` series order must equal the filtered original order, which the original
    ``read_headers``/``pick_slots`` (first row wins n_slices ties) depend on.
    """
    total = sum(r["bytes"] for r in records)
    if total > funding["max_study_source_bytes"]:
        raise ValueError("Study source exceeds the funded transport bound")
    if shutil.disk_usage(scratch).free - total < funding["min_free_bytes"]:
        raise ValueError("Study transport would breach the free-disk reserve")
    accepted: dict[str, dict[str, dict]] = {}
    for record in records:
        parts = PurePosixPath(record["path"]).parts
        if (
            len(parts) != 4
            or parts[:2] != (producer.SPLIT, uid)
            or parts[2] != record["series_uid"]
        ):
            raise ValueError("Source record path differs from the original layout")
        accepted.setdefault(parts[2], {})[parts[3]] = record
    source_digest = producer.study_source_sha256(uid, records)
    source_study = Path(source_study)
    _real_directory(source_study)
    if source_study.name != uid:
        raise ValueError("Original study directory required")
    destination = scratch / producer.SPLIT / uid
    if destination.exists() or destination.is_symlink():
        raise ValueError("Transport scratch is not empty")
    order, excluded = [], 0
    for entry in os.scandir(source_study):
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            raise ValueError("Study directory holds only original series directories")
        if entry.name in accepted:
            order.append(entry.name)
        else:
            excluded += 1
    if sorted(order) != sorted(accepted):
        raise ValueError("Accepted series are missing from the original study")
    plan = []
    for series in order:
        names = {}
        for entry in os.scandir(source_study / series):
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise ValueError("Series directory holds only original regular files")
            names[entry.name] = Path(entry.path)
        if set(names) != set(accepted[series]):
            raise ValueError("Accepted series files differ from the bound objects")
        for name, path in names.items():
            status = path.lstat()
            plan.append((series, name, path, (status.st_size, status.st_mtime_ns, status.st_ino)))
    destination.mkdir(parents=True, mode=0o700)
    for series in order:
        (destination / series).mkdir(mode=0o700)
    for series, name, path, _ in plan:
        _copy(path, destination / series / name, accepted[series][name])
    for _, _, path, before in plan:
        status = path.lstat()
        if (status.st_size, status.st_mtime_ns, status.st_ino) != before:
            raise ValueError("Original source changed during transport")
    copied = [e.name for e in os.scandir(destination) if e.is_dir(follow_symlinks=False)]
    if sorted(copied) != sorted(order):
        raise ValueError("Transport series membership differs from the original source")
    slices = [len(accepted[s]) for s in order]
    ties = [[s for s, n in zip(order, slices) if n == count] for count in sorted(set(slices))]
    receipt = {
        "schema_version": 1,
        "study_uid": uid,
        "origin": origin,
        "source_revision": (
            pst.HISTORICAL_SOURCE_REVISION
            if origin == "retained"
            else pst.CORRECTED_SOURCE_REVISION
        ),
        "source_study_sha256": source_digest,
        "series_order": order,
        "n_slices": slices,
        "n_slices_tie_groups": [group for group in ties if len(group) > 1],
        "excluded_series_present": excluded,
        "copied_series_order": copied,
        "transport_order_equal": copied == order,
        "renderer_replays_original_series_order": True,
    }
    return destination, receipt


def bind_order_receipt(out: Path, uid: str, receipt: dict, intent_sha256: str) -> dict:
    """Write the study's source-order receipt once; any later order must be byte-identical."""
    bound = receipt | {"intent_sha256": intent_sha256}
    payload = producer._json_bytes(bound)
    directory = Path(out) / ORDER_DIRECTORY
    directory.mkdir(mode=0o700, exist_ok=True)
    path = directory / f"{producer._safe_uid(uid)}.json"
    if path.is_symlink():
        raise ValueError("Source-order receipt may not be a symlink")
    if path.exists():
        if path.read_bytes() != payload:
            raise ValueError("Source order differs from the bound source-order receipt")
    else:
        producer._write_exclusive(path, payload)
    return bound


def discard_unfinished_transport(scratch: Path) -> int:
    """Remove temporary copies left by an interrupted study; never follows a symlink."""
    removed = 0
    for leftover in list(Path(scratch).iterdir()):
        for path in [leftover, *leftover.rglob("*")]:
            if path.is_symlink():
                raise ValueError("Unfinished transport holds a symlink; review before staging")
        for path in [leftover, *leftover.rglob("*")]:
            if path.is_dir():
                path.chmod(0o700)
        shutil.rmtree(leftover) if leftover.is_dir() else leftover.unlink()
        removed += 1
    return removed


def stage_study(
    cache: Path,
    out: Path,
    scratch: Path,
    uid: str,
    source_study: Path,
    records: list[dict],
    plane_map: dict,
    funding: dict,
    renderer: producer.Renderer,
    *,
    origin: str,
) -> tuple[dict, dict]:
    """Transport, bind source order, render once through the producer, then delete the copy."""
    _, intent_sha = producer.load_intent(cache)
    try:
        _, order = transport_study(source_study, scratch, uid, records, funding, origin=origin)
        order = bind_order_receipt(out, uid, order, intent_sha)
        from tools.research.public_dinov2_input_reference import reference

        with reference.preserve_series_order(uid, order["series_order"]):
            receipt = producer.stage_member(
                cache, uid, scratch, records, plane_map, renderer=renderer
            )
    finally:
        transported = scratch / producer.SPLIT
        if transported.exists():
            for path in [transported, *transported.rglob("*")]:
                if path.is_dir() and not path.is_symlink():
                    path.chmod(0o700)
            shutil.rmtree(transported)
    return receipt, order


def require_clean(status: dict) -> dict:
    """Staging proceeds only over pending/complete members and no foreign cache entries."""
    if (
        status["finalized"]
        or status["unexpected"]
        or set(status["states"])
        - {
            "pending",
            "complete",
        }
    ):
        raise ValueError("Cache holds interrupted, foreign or finalized state; reconcile only")
    return status


def aggregate(cache: Path, status: dict, out: Path | None = None) -> dict:
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
    result = {
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
    if out is not None and (Path(out) / ORDER_DIRECTORY).is_dir():
        digest, ties, excluded = hashlib.sha256(), 0, 0
        origins: Counter = Counter()
        for path in sorted((Path(out) / ORDER_DIRECTORY).glob("*.json")):
            payload = path.read_bytes()
            digest.update(hashlib.sha256(payload).digest())
            order = json.loads(payload)
            origins[order["origin"]] += 1
            ties += bool(order["n_slices_tie_groups"])
            excluded += order["excluded_series_present"]
        result["source_order"] = {
            "receipts_sha256": digest.hexdigest(),
            "origins": dict(sorted(origins.items())),
            "studies_with_n_slices_ties": ties,
            "excluded_series_present": excluded,
        }
    return result


def run(args: argparse.Namespace) -> dict:
    runtime = check_runtime()
    hashes = code_hashes()
    funding = load_funding(Path(args.funding), args.funding_sha256, runtime, hashes)
    sources = [Path(p) for p in (args.dicom_root, args.retained_packet) if p]
    out = private_output(Path(args.private_out), sources, create=args.mode != "reconcile")
    cache, scratch = out / "cache", out / "transport"
    if args.mode != "reconcile":
        for directory in (cache, scratch):
            directory.mkdir(mode=0o700, exist_ok=True)
    config_bytes = CONFIG.read_bytes()
    policy = json.loads(config_bytes)["public_input"]
    pi.validate_public_input(policy)
    if args.mode == "reconcile":
        status = producer.reconcile(cache)
        return {"mode": args.mode, **aggregate(cache, status, out), "clinical_evidence": False}
    if args.mode == "finalize":
        index_sha = producer.finalize(cache, policy=policy)
        status = producer.reconcile(cache)
        return {
            "mode": args.mode,
            "index_sha256": index_sha,
            **aggregate(cache, status, out),
            "clinical_evidence": False,
        }

    from rsna.competitive.folds import load_folds
    from rsna.competitive.supervision import roster_sha
    from rsna.competitive.train import select_development
    from rsna.release import load_verified_release

    if args.dicom_root and args.source_revision != pst.CORRECTED_SOURCE_REVISION:
        raise ValueError("Newly acquired objects must be at the corrected source revision")
    if args.dicom_root:
        dicom_root = Path(args.dicom_root)
        _real_directory(dicom_root / producer.SPLIT)
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
    fitting_uids = list(map(str, fitting.study_uid))
    evaluation_uids = list(map(str, outer.study_uid))
    objects, table, input_hashes, corrected_series = bind_sources(
        read_inventory(INVENTORY_DIR),
        Path(args.series),
        release,
        fitting_uids,
        evaluation_uids,
        CORRECTION_PLAN,
        CORRECTION_RECEIPT,
    )
    records = producer_records(objects)
    retained, retained_summary = (
        retained_studies(Path(args.retained_packet), records, corrected_series)
        if args.retained_packet
        else ({}, {"retained_studies": 0})
    )
    import pandas as pd

    series = pd.read_csv(args.series, dtype=str)
    plane_map = dict(zip(series.SeriesInstanceUID, series.Anatomical_Plane))
    binding = {
        "runtime": runtime,
        "runtime_sha256": pi.canonical_sha256(runtime),
        "funding_sha256": args.funding_sha256,
        **hashes,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "member_manifest_sha256": hashlib.sha256(manifest).hexdigest(),
        "folds_sha256": overlay["folds_sha256"],
        "fitting_uid_sha256": overlay["fitting_uid_sha256"],
        "outer_uid_sha256": overlay["outer_uid_sha256"],
        **input_hashes,
        "source_table": table,
        "retained_packet_inventory_sha256": retained_summary.get(
            "retained_packet_inventory_sha256"
        ),
        "source_order_receipt_schema": 1,
    }
    cohort = sorted(fitting_uids + evaluation_uids)
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
            source_revision=pst.CORRECTED_SOURCE_REVISION,
            fitting_uids=fitting_uids,
            evaluation_uids=evaluation_uids,
            source_studies={u: producer.study_source_sha256(u, records[u]) for u in cohort},
            caps={k: funding[k] for k in ("max_total_bytes", "min_free_bytes")},
            binding=binding,
        )
    status = require_clean(producer.reconcile(cache, verify_bytes=False))
    discarded = discard_unfinished_transport(scratch)

    def source_of(uid: str) -> tuple[Path, str] | None:
        if uid in retained:
            return retained[uid], "retained"
        if args.dicom_root:
            return Path(args.dicom_root) / producer.SPLIT / uid, "corrected"
        return None

    waiting = status["states"].get("pending", [])
    available = [u for u in waiting if source_of(u)]
    pending = available[: funding["max_studies_per_run"]]
    outcome: Counter = Counter()
    refusals: Counter = Counter()
    for uid in pending:
        source_study, origin = source_of(uid) or (Path(), "")
        try:
            receipt, _ = stage_study(
                cache,
                out,
                scratch,
                uid,
                source_study,
                records[uid],
                plane_map,
                funding,
                renderer,
                origin=origin,
            )
            outcome[("reused_" if receipt["reused"] else "rendered_") + origin] += 1
        except Exception as error:  # type only; no IDs or messages leave the node
            refusals[type(error).__name__] += 1
    status = producer.reconcile(cache)
    return {
        "mode": args.mode,
        "attempted": len(pending),
        "unavailable_source_studies": len(waiting) - len(available),
        "discarded_unfinished_transport": discarded,
        "outcome": dict(sorted(outcome.items())),
        "refusals": dict(sorted(refusals.items())),
        "binding": binding,
        "retained": retained_summary,
        **aggregate(cache, status, out),
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
    for name in ("dicom-root", "source-revision", "retained-packet"):
        parser.add_argument("--" + name)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    required = ("source", "member_manifest", "studies", "artifact_root", "folds", "series")
    if args.mode == "stage" and (
        any(getattr(args, k) is None for k in required)
        or not (args.dicom_root or args.retained_packet)
        or bool(args.dicom_root) != bool(args.source_revision)
    ):
        parser.error("stage requires every input binding and a corrected root or retained packet")
    text = json.dumps(run(args), indent=2, sort_keys=True)
    if args.out:
        args.out.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
