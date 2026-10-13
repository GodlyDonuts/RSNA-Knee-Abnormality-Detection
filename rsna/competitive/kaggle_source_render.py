"""Kaggle CPU placement of the source-bound public triplet producer (prepared, not run).

The attached competition mount supplies the official training DICOMs, so the producer's
``render_member`` (exact source verification, original v15 render, bit-exact storage and
adapter replay) can run where the source already is. Each shard holds byte-exact producer
members, losslessly gzip-encoded, with immutable receipts and source/member/shard manifests.
The owning node verifies every shard against the complete-cohort manifest and its own
producer intent, restores the exact member bytes and commits them through
``public_cache_producer.import_member``; ``finalize`` then writes the unchanged trainer index.
Nothing here dispatches, pushes, downloads or mutates Kaggle, Hugging Face or Slurm state.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import zlib
from collections import Counter
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath

import numpy as np

from rsna.competitive import public_cache_producer as producer
from rsna.competitive import public_input as pi

SCHEMA_VERSION = 1
COMPETITION = "rsna-knee-abnormality-detection"
PLACEMENT = "kaggle_cpu_notebook"
ENCODING = "gzip9_mtime0_of_exact_member_npz"
PARITY_SALT = "public-dinov2-input-parity-v1"
PARITY_SIZE = 8
ACCEPTED_PARITY_TENSOR_SHA256 = "e602f1a976fe341a48794d5cec8919522ee6bdad473770661fb9c1d69db3c6d8"
PARITY_KEYS = ("images", "features", "positions", "series_mask", "slice_mask")
MEMBERS, RECEIPTS, ORDERS = "members", "receipts", "source-order"
SHARD_MANIFEST, COHORT_MANIFEST = "shard-manifest.json", "cohort-manifest.json"
RUNTIME_MODULES = ("numpy", "pandas", "pydicom", "torch", "cv2")
BUDGET_KEYS = {
    "schema_version",
    "placement",
    "competition",
    "cpu_only",
    "gpus",
    "internet",
    "max_studies",
    "max_study_source_bytes",
    "min_free_bytes",
    "max_output_bytes",
    "max_wall_seconds",
    "code_sha256",
    "runtime_sha256",
}


class PlacementError(ValueError):
    pass


def canonical_sha256(value: object) -> str:
    return pi.canonical_sha256(value)


def check_cpu_runtime(environ: Mapping[str, str] | None = None) -> dict:
    """Kaggle CPU session: no visible CUDA device; binds exact library versions and bytes."""
    import torch

    env = os.environ if environ is None else environ
    if env.get("CUDA_VISIBLE_DEVICES"):
        raise PlacementError("CPU-only placement requires no visible CUDA device")
    if torch.cuda.is_available() or torch.cuda.device_count():
        raise PlacementError("A CUDA device is visible; attach a CPU-only session")
    if any(Path("/dev").glob("nvidia[0-9]*")):
        raise PlacementError("An NVIDIA device node is present; attach a CPU-only session")
    torch.set_num_threads(1)
    import importlib
    import platform

    modules = {}
    for name in RUNTIME_MODULES:
        module = importlib.import_module(name)
        path = Path(str(getattr(module, "__file__", "")))
        modules[name] = {
            "version": str(getattr(module, "__version__", "")),
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None,
        }
    return {
        "placement": PLACEMENT,
        "kernel_run_type": env.get("KAGGLE_KERNEL_RUN_TYPE"),
        "python": platform.python_version(),
        "machine": platform.machine(),
        "modules": modules,
        "torch_threads": 1,
        "cuda_available": False,
        "gpus": 0,
    }


def resources(paths: Iterable[Path]) -> dict:
    """Actual CPU/RAM/disk envelope of this session (measured, never assumed)."""
    import resource
    import shutil

    meminfo = {}
    if Path("/proc/meminfo").exists():
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"MemTotal", "MemAvailable"}:
                meminfo[key] = int(value.split()[0]) * 1024
    limit = None
    for name in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        if Path(name).exists():
            text = Path(name).read_text().strip()
            limit = None if text == "max" else int(text)
            break
    return {
        "cpus": os.cpu_count(),
        "affinity_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "mem_total_bytes": meminfo.get("MemTotal"),
        "mem_available_bytes": meminfo.get("MemAvailable"),
        "cgroup_memory_limit_bytes": limit,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        "disk": {
            str(path): dict(zip(("total", "used", "free"), shutil.disk_usage(path)))
            for path in paths
        },
    }


def resolve_mount(input_root: Path, competition: str = COMPETITION) -> Path:
    """The attached competition source; real directories only, never a symlink."""
    candidates = [Path(input_root) / competition, Path(input_root) / "competitions" / competition]
    found = [c for c in candidates if (c / producer.SPLIT).is_dir()]
    if len(found) != 1:
        raise PlacementError("Exactly one attached competition training mount is required")
    mount = found[0]
    for path in (mount, mount / producer.SPLIT, mount / "train_series.csv"):
        if any(p.is_symlink() for p in (path, *path.parents)):
            raise PlacementError("Competition mount paths may not traverse symlinks")
    if not (mount / "train_series.csv").is_file():
        raise PlacementError("Mounted release series table is missing")
    return mount


def _hash_file(path: Path, limit: int) -> tuple[int, str]:
    digest, size = hashlib.sha256(), 0
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(descriptor, "rb") as reader:
        while block := reader.read(1 << 20):
            size += len(block)
            if size > limit:
                raise PlacementError("Mounted source object exceeds its bound bytes")
            digest.update(block)
    return size, digest.hexdigest()


def verify_corrections(mount: Path, changes: Mapping[str, Mapping]) -> dict:
    """Both officially corrected objects must hold their exact after bytes/SHA at the mount."""
    verified = 0
    for path, change in sorted(changes.items()):
        after = change["after"]
        relative = PurePosixPath(str(after["path"]))
        if relative.is_absolute() or ".." in relative.parts or str(relative) != path:
            raise PlacementError("Corrected source path differs from its plan")
        size, sha = _hash_file(Path(mount) / relative, int(after["bytes"]))
        if size != int(after["bytes"]) or sha != after["sha256"]:
            raise PlacementError("Mounted corrected object differs from the official after bytes")
        verified += 1
    return {
        "corrected_objects_verified": verified,
        "corrected_after_sha256": canonical_sha256(
            sorted((c["after"]["sha256"], int(c["after"]["bytes"])) for c in changes.values())
        ),
    }


def parity_roster(fitting_uids: Iterable[str], size: int = PARITY_SIZE) -> list[str]:
    """The retained outcome-free fitting roster (salted SHA rank, as the accepted receipt)."""
    uids = sorted(set(map(str, fitting_uids)))
    ranked = sorted(uids, key=lambda u: hashlib.sha256(f"{PARITY_SALT}:{u}".encode()).hexdigest())
    return ranked[:size]


def parity_study_bytes(inputs: Mapping, labels: np.ndarray, mask: np.ndarray) -> bytes:
    """One study's bytes in the accepted adapter receipt's tensor-hash order.

    The accepted receipt hashed the adapter row rendered with fitting-only overlay labels;
    labels never reach pixels, so the zero-label row plus the masked labels reproduce it.
    """
    labels = np.asarray(labels, dtype=np.float32)
    mask = np.asarray(mask, dtype=np.float32)
    parts = [np.asarray(inputs[key].numpy()).tobytes() for key in PARITY_KEYS]
    parts.append(np.where(mask.astype(bool), labels, np.float32(0)).astype(np.float32).tobytes())
    parts.append(mask.tobytes())
    return b"".join(parts)


def parity_digest(studies: Iterable[bytes]) -> str:
    digest = hashlib.sha256()
    for payload in studies:
        digest.update(payload)
    return digest.hexdigest()


def inputs_sha256(inputs: Mapping) -> str:
    """Digest of one adapter output: every tensor's name, dtype, shape and bytes."""
    digest = hashlib.sha256()
    for key in sorted(inputs):
        value = inputs[key]
        if isinstance(value, str):
            digest.update(f"{key}:str:{value}\n".encode())
            continue
        array = np.ascontiguousarray(value.numpy())
        digest.update(f"{key}:{array.dtype}:{list(array.shape)}\n".encode())
        digest.update(array.tobytes())
    return digest.hexdigest()


def encode_member(payload: bytes) -> bytes:
    """Lossless, deterministic encoding of the exact member bytes (restores the same SHA)."""
    return gzip.compress(payload, compresslevel=9, mtime=0)


def decode_member(blob: bytes, *, size: int, sha256: str) -> bytes:
    """Bounded decode: refuses anything but exactly ``size`` bytes with the receipted SHA."""
    decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    payload = decoder.decompress(blob, size + 1)
    if len(payload) != size or decoder.unconsumed_tail or not decoder.eof or decoder.unused_data:
        raise PlacementError("Encoded member does not decode to exactly its bound bytes")
    if hashlib.sha256(payload).hexdigest() != sha256:
        raise PlacementError("Decoded member differs from its receipted bytes")
    return payload


def measured_encodings(payload: bytes) -> dict:
    """Actual encoded sizes of one real member; NPZ-compressed keeps the array contract."""
    arrays = producer._load_arrays(payload)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{name: arrays[name] for name in pi.MEMBER_ARRAYS})
    compressed = buffer.getvalue()
    restored = producer._load_arrays(compressed)
    if not all(np.array_equal(restored[n], arrays[n]) for n in pi.MEMBER_ARRAYS):
        raise PlacementError("Compressed NPZ is not lossless for this member")
    return {"exact": len(payload), "gzip": len(encode_member(payload)), "npz_deflate": len(compressed)}


def assign_shards(cohort: Iterable[str], studies_per_shard: int) -> list[list[str]]:
    """Contiguous sorted-UID shards; the assignment is part of the cohort manifest."""
    uids = sorted(set(map(str, cohort)))
    if studies_per_shard < 1 or not uids:
        raise PlacementError("A nonempty cohort and positive shard size are required")
    return [uids[i : i + studies_per_shard] for i in range(0, len(uids), studies_per_shard)]


def cohort_manifest(
    *,
    fitting_uids: Iterable[str],
    evaluation_uids: Iterable[str],
    source_studies: Mapping[str, str],
    source_revision: str,
    binding: Mapping,
    studies_per_shard: int,
) -> dict:
    """Immutable complete-cohort manifest: rosters, per-study source digests, shard plan."""
    fitting, evaluation = sorted(fitting_uids), sorted(evaluation_uids)
    cohort = sorted(set(fitting) | set(evaluation))
    if set(source_studies) != set(cohort) or set(fitting) & set(evaluation):
        raise PlacementError("Every disjoint cohort member needs exactly one source digest")
    shards = assign_shards(cohort, studies_per_shard)
    return {
        "schema_version": SCHEMA_VERSION,
        "placement": PLACEMENT,
        "source_revision": source_revision,
        "fitting_uids_sha256": pi.roster_sha256(fitting),
        "evaluation_uids_sha256": pi.roster_sha256(evaluation),
        "cohort_sha256": pi.roster_sha256(cohort),
        "source_studies_sha256": canonical_sha256(dict(sorted(source_studies.items()))),
        "member_bytes": producer.member_bytes(),
        "encoding": ENCODING,
        "studies_per_shard": studies_per_shard,
        "shards": [pi.roster_sha256(shard) for shard in shards],
        "binding": dict(binding),
        "labels_stored": False,
    }


def load_budget(path: Path, expected_sha256: str, code_sha256: str, runtime: dict) -> dict:
    """Root-issued Kaggle CPU budget bound to the exact code and (after the probe) runtime."""
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise PlacementError("Budget bytes differ from the root-issued hash")
    budget = json.loads(payload)
    if (
        set(budget) != BUDGET_KEYS
        or budget["schema_version"] != SCHEMA_VERSION
        or budget["placement"] != PLACEMENT
        or budget["competition"] != COMPETITION
        or budget["cpu_only"] is not True
        or budget["gpus"] != 0
        or budget["internet"] is not False
        or budget["code_sha256"] != code_sha256
        or any(
            type(budget[k]) is not int or budget[k] < 1
            for k in BUDGET_KEYS
            if k.startswith(("max_", "min_"))
        )
    ):
        raise PlacementError("Budget differs from the CPU-only Kaggle placement contract")
    if budget["runtime_sha256"] not in {None, canonical_sha256(runtime)}:
        raise PlacementError("Runtime differs from the root-accepted Kaggle runtime")
    return budget


def write_shard_member(
    shard: Path, uid: str, payload: bytes, receipt: Mapping, order: Mapping, inputs: Mapping
) -> dict:
    """Write one encoded member, its producer receipt and source order, each exactly once."""
    producer._safe_uid(uid)
    if "intent_sha256" in receipt or receipt.get("study_uid") != uid:
        raise PlacementError("Shard receipts are unbound producer receipts of this study")
    if order.get("study_uid") != uid:
        raise PlacementError("Source-order receipt belongs to another study")
    blob = encode_member(payload)
    decode_member(blob, size=len(payload), sha256=receipt["member"]["sha256"])
    record = {
        "study_uid": uid,
        "encoded": {"path": f"{MEMBERS}/{uid}.npz.gz", "bytes": len(blob)},
        "member": dict(receipt["member"]),
        "receipt_sha256": hashlib.sha256(producer._json_bytes(dict(receipt))).hexdigest(),
        "order_sha256": hashlib.sha256(producer._json_bytes(dict(order))).hexdigest(),
        "inputs_sha256": inputs_sha256(inputs),
        "records_sha256": receipt["source"]["records_sha256"],
    }
    record["encoded"]["sha256"] = hashlib.sha256(blob).hexdigest()
    for name in (MEMBERS, RECEIPTS, ORDERS):
        (Path(shard) / name).mkdir(mode=0o700, parents=True, exist_ok=True)
    producer._write_exclusive(Path(shard) / record["encoded"]["path"], blob)
    producer._write_exclusive(
        Path(shard) / RECEIPTS / f"{uid}.json", producer._json_bytes(dict(receipt))
    )
    producer._write_exclusive(Path(shard) / ORDERS / f"{uid}.json", producer._json_bytes(dict(order)))
    return record


def seal_shard(
    shard: Path, cohort: Mapping, index: int | None, records: list[dict], runtime: dict
) -> str:
    """Write the immutable shard manifest once; returns its SHA. Probes (index None) never merge."""
    members = sorted(records, key=lambda r: r["study_uid"])
    uids = [r["study_uid"] for r in members]
    if len(set(uids)) != len(uids):
        raise PlacementError("Shard members must be unique")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "cohort_manifest_sha256": canonical_sha256(dict(cohort)),
        "shard_index": index,
        "shard_uids_sha256": pi.roster_sha256(uids),
        "members": members,
        "members_sha256": canonical_sha256(members),
        "runtime": runtime,
        "runtime_sha256": canonical_sha256(runtime),
        "encoded_bytes": sum(r["encoded"]["bytes"] for r in members),
        "exact_bytes": sum(r["member"]["bytes"] for r in members),
        "labels_stored": False,
    }
    payload = producer._json_bytes(manifest)
    producer._write_exclusive(Path(shard) / SHARD_MANIFEST, payload)
    return hashlib.sha256(payload).hexdigest()


def _regular(path: Path) -> bytes:
    if path.is_symlink() or not path.is_file():
        raise PlacementError("Shard entries must be regular files")
    return path.read_bytes()


def verify_shard(shard: Path, cohort: Mapping, expected_sha256: str | None = None) -> dict:
    """Read-only shard verification against the complete-cohort manifest; no foreign entries."""
    shard = Path(shard)
    payload = _regular(shard / SHARD_MANIFEST)
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise PlacementError("Shard manifest differs from its recorded hash")
    manifest = json.loads(payload)
    members = manifest["members"]
    uids = [m["study_uid"] for m in members]
    index = manifest["shard_index"]
    if (
        manifest["schema_version"] != SCHEMA_VERSION
        or manifest["cohort_manifest_sha256"] != canonical_sha256(dict(cohort))
        or type(index) is not int
        or not 0 <= index < len(cohort["shards"])
        or manifest["shard_uids_sha256"] != pi.roster_sha256(uids)
        or manifest["shard_uids_sha256"] != cohort["shards"][index]
        or manifest["members_sha256"] != canonical_sha256(members)
        or manifest["runtime_sha256"] != canonical_sha256(manifest["runtime"])
        or manifest["labels_stored"] is not False
    ):
        raise PlacementError("Shard manifest differs from the complete-cohort manifest")
    expected = {
        MEMBERS: {f"{u}.npz.gz" for u in uids},
        RECEIPTS: {f"{u}.json" for u in uids},
        ORDERS: {f"{u}.json" for u in uids},
    }
    found = {e.name for e in os.scandir(shard)}
    if found != {SHARD_MANIFEST, *expected}:
        raise PlacementError("Shard holds foreign or missing entries")
    for name, allowed in expected.items():
        if {e.name for e in os.scandir(shard / name)} != allowed:
            raise PlacementError("Shard members differ from its manifest")
    return manifest


def merge_shards(
    cache: Path,
    shards: Mapping[Path, str],
    cohort: Mapping,
    *,
    renderer: producer.Renderer | None = None,
) -> dict:
    """Owning-node merge: verify shards, restore exact members and import them once.

    ``shards`` maps each shard directory to its recorded manifest SHA. Already complete
    members must carry the same exact member SHA; anything else is refused. With a
    renderer, the owning runtime's adapter replay must equal the Kaggle-side digest.
    """
    intent, _ = producer.load_intent(cache)
    if (
        intent["cohort_sha256"] != cohort["cohort_sha256"]
        or intent["fitting_uids_sha256"] != cohort["fitting_uids_sha256"]
        or intent["evaluation_uids_sha256"] != cohort["evaluation_uids_sha256"]
        or intent["source_revision"] != cohort["source_revision"]
        or canonical_sha256(intent["source_studies"]) != cohort["source_studies_sha256"]
        or intent["member_bytes"] != cohort["member_bytes"]
    ):
        raise PlacementError("Owning cache intent differs from the complete-cohort manifest")
    status = producer.reconcile(cache, verify_bytes=False)
    if status["unexpected"] or set(status["states"]) - {"pending", "complete"}:
        raise PlacementError("Reconcile interrupted or foreign cache state before merging")
    complete = set(status["states"].get("complete", []))
    seen: set[str] = set()
    indices: set[int] = set()
    outcome: Counter = Counter()
    for path, sha in sorted(shards.items(), key=lambda item: str(item[0])):
        manifest = verify_shard(Path(path), cohort, sha)
        uids = [m["study_uid"] for m in manifest["members"]]
        if manifest["shard_index"] in indices or seen & set(uids):
            raise PlacementError("Shards overlap")
        indices.add(manifest["shard_index"])
        seen |= set(uids)
        for record in manifest["members"]:
            uid = record["study_uid"]
            receipt_bytes = _regular(Path(path) / RECEIPTS / f"{uid}.json")
            receipt = json.loads(receipt_bytes)
            if (
                hashlib.sha256(receipt_bytes).hexdigest() != record["receipt_sha256"]
                or hashlib.sha256(_regular(Path(path) / ORDERS / f"{uid}.json")).hexdigest()
                != record["order_sha256"]
                or receipt["member"] != record["member"]
                or record["records_sha256"] != intent["source_studies"].get(uid)
            ):
                raise PlacementError("Shard member records differ from their manifest")
            blob = _regular(Path(path) / record["encoded"]["path"])
            if (
                len(blob) != record["encoded"]["bytes"]
                or hashlib.sha256(blob).hexdigest() != record["encoded"]["sha256"]
            ):
                raise PlacementError("Encoded member bytes differ from the shard manifest")
            payload = decode_member(
                blob, size=record["member"]["bytes"], sha256=record["member"]["sha256"]
            )
            if renderer is not None:
                replay = renderer.roundtrip(producer._load_arrays(payload), uid)
                if inputs_sha256(replay) != record["inputs_sha256"]:
                    raise PlacementError("Owning-runtime adapter replay differs from the shard")
                outcome["replayed"] += 1
            if uid in complete:
                existing = json.loads(
                    (Path(cache) / pi.CACHE_DIRECTORY / producer.RECEIPTS / f"{uid}.json").read_bytes()
                )
                if existing["member"] != record["member"]:
                    raise PlacementError("Complete member differs from the shard member")
                outcome["already_complete"] += 1
                continue
            producer.import_member(cache, uid, payload, receipt)
            outcome["imported"] += 1
    status = producer.reconcile(cache)
    return {
        "shards": len(indices),
        "shard_members": len(seen),
        "outcome": dict(sorted(outcome.items())),
        "counts": status["counts"],
        "cohort_complete": set(status["states"]) == {"complete"},
        "labels_stored": False,
    }


def plan_shards(
    *,
    cohort_studies: int,
    seconds_per_study: float,
    encoded_bytes_per_study: int,
    scratch_bytes_per_study: int,
    max_wall_seconds: int,
    max_output_bytes: int,
    scratch_free_bytes: int,
    min_free_bytes: int,
    safety: float = 0.5,
) -> dict:
    """Bounded shard size from MEASURED per-study cost and root-funded session limits.

    ``safety`` keeps that fraction of wall time and output disk unused. Inputs must come
    from an actual probe; this function never assumes compression or speed.
    """
    if not 0 < safety < 1 or min(seconds_per_study, encoded_bytes_per_study) <= 0:
        raise PlacementError("Measured positive per-study costs and a safety fraction required")
    by_time = int(max_wall_seconds * (1 - safety) // seconds_per_study)
    by_output = int(max_output_bytes * (1 - safety) // encoded_bytes_per_study)
    by_scratch = 1 if scratch_free_bytes - scratch_bytes_per_study >= min_free_bytes else 0
    studies = min(by_time, by_output, cohort_studies) if by_scratch else 0
    if studies < 1:
        raise PlacementError("Measured costs do not fit one study inside the funded limits")
    shards = -(-cohort_studies // studies)
    return {
        "studies_per_shard": studies,
        "shards": shards,
        "bound": "time" if studies == by_time else "output" if studies == by_output else "cohort",
        "projected_shard_seconds": studies * seconds_per_study,
        "projected_shard_output_bytes": studies * encoded_bytes_per_study,
        "projected_total_output_bytes": cohort_studies * encoded_bytes_per_study,
        "projected_total_cpu_seconds": cohort_studies * seconds_per_study,
    }
