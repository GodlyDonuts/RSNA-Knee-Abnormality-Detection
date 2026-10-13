"""Streaming source-bound producer for the public ordered-triplet cohort cache.

Each original study is verified against its inventory records, rendered once by the
reviewed Pilkwang v15 renderer through the exact hashed ``dino_adapter.render_inputs``
bytes, staged under an immutable per-study receipt, and only then linked as a member.
The trainer-facing ``index.json`` (``public_input.load_public_index`` schema, unchanged)
is written once, atomically, after every declared member exists with its bound bytes.
No label, mask, native cache or float cache enters this cache; the trainer supplies labels.
Research comparison only; this module never dispatches jobs or downloads anything.
"""

from __future__ import annotations

import hashlib
import importlib
import io
import json
import os
import shutil
import types
from collections.abc import Iterable, Mapping
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np
import torch

from rsna.competitive import public_input as pi
from rsna.schema import TARGETS

SCHEMA_VERSION = 1
COHORT_SIZES = (2247, 1156)
SPLIT = "train_series"
INTENT, INDEX, FINALIZED = "intent.json", "index.json", "finalized.json"
MEMBERS, RECEIPTS, STAGING = "members", "receipts", "staging"
RECORD_KEYS = {"path", "sha256", "bytes", "study_uid", "series_uid", "sop_uid"}
ADAPTER_PACKAGE = "tools.research.public_dinov2_input_reference"
CAP_KEYS = {"max_total_bytes", "min_free_bytes"}


class ProducerError(ValueError):
    """A source, roster, cap or state check refused the operation; nothing was written."""


def _safe_uid(uid: object) -> str:
    if (
        not isinstance(uid, str)
        or not uid
        or uid in {".", ".."}
        or len(uid) > 128
        or any(char in uid for char in ("/", "\\", "\x00"))
        or uid != uid.strip()
    ):
        raise ProducerError("Study and series IDs must be safe individual filenames")
    return uid


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_exclusive(path: Path, payload: bytes) -> None:
    """Create-only write; an existing file (another writer or a prior run) is never replaced."""
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        view = memoryview(payload)
        while view:
            view = view[os.write(descriptor, view) :]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, indent=1, allow_nan=False).encode()


def _serialize(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    named: dict[str, Any] = {name: arrays[name] for name in pi.MEMBER_ARRAYS}
    np.savez(buffer, **named)
    return buffer.getvalue()


@lru_cache(maxsize=1)
def member_bytes() -> int:
    """Exact uncompressed member size; fixed dtypes/shapes make it content independent."""
    return len(
        _serialize(
            {name: np.zeros(shape, dtype) for name, (dtype, shape) in pi.MEMBER_ARRAYS.items()}
        )
    )


def _load_arrays(payload: bytes) -> dict[str, np.ndarray]:
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        if set(archive.files) != set(pi.MEMBER_ARRAYS):
            raise ProducerError("Member arrays differ from the source contract")
        arrays = {name: archive[name] for name in pi.MEMBER_ARRAYS}
    for name, (dtype, shape) in pi.MEMBER_ARRAYS.items():
        if arrays[name].dtype != dtype or arrays[name].shape != shape:
            raise ProducerError("Member dtype/shape differs from the source contract")
    return arrays


def study_source_sha256(uid: str, records: Iterable[Mapping]) -> str:
    """Validate one study's original inventory rows and return their canonical digest."""
    _safe_uid(uid)
    rows = []
    for record in records:
        if not isinstance(record, Mapping) or set(record) != RECORD_KEYS:
            raise ProducerError("Inventory record fields differ")
        path = PurePosixPath(str(record["path"]))
        if (
            path.is_absolute()
            or ".." in path.parts
            or len(path.parts) != 4
            or path.parts[0] != SPLIT
            or path.parts[1] != uid
            or record["study_uid"] != uid
            or path.parts[2] != _safe_uid(record["series_uid"])
            or path.name != f"{_safe_uid(record['sop_uid'])}.dcm"
            or str(path) != record["path"]
            or type(record["bytes"]) is not int
            or record["bytes"] < 1
            or not pi._is_sha(record["sha256"])
        ):
            raise ProducerError("Inventory record is not an original study/series/SOP DICOM")
        rows.append({key: record[key] for key in sorted(RECORD_KEYS)})
    rows.sort(key=lambda row: row["path"])
    if not rows or len({row["path"] for row in rows}) != len(rows):
        raise ProducerError("Each study needs unique nonempty original inventory rows")
    return pi.canonical_sha256(rows)


def _root(cache: Path) -> Path:
    return Path(cache) / pi.CACHE_DIRECTORY


def _validate_cohort(
    fitting: list[str], evaluation: list[str], sizes: tuple[int, int] = COHORT_SIZES
) -> None:
    for uid in (*fitting, *evaluation):
        _safe_uid(uid)
    if (
        len(set(fitting)) != len(fitting)
        or len(set(evaluation)) != len(evaluation)
        or set(fitting) & set(evaluation)
        or (len(fitting), len(evaluation)) != tuple(sizes)
    ):
        raise ProducerError(
            "Disjoint unique fitting/development membership of the declared size is required"
        )


class Renderer:
    """The reviewed original renderer plus the exact hashed adapter bytes, configured once."""

    def __init__(self, policy: dict, pixel_group: dict, source: bytes | None = None):
        reference = importlib.import_module(ADAPTER_PACKAGE + ".reference")
        pi.validate_public_input(policy)
        pi._validate_pixel_group(pixel_group)
        namespace, receipt = reference.load(source)
        if (
            receipt["source_sha256"] != pi.SOURCE_SHA256
            or receipt["node_text_sha256"] != pi.NODE_TEXT_SHA256
            or receipt["notebook_top_level_executed"] is not False
        ):
            raise ProducerError("Renderer source or reviewed node text differs")
        reference.configure(namespace, pixel_group)
        adapter_source = pi.ADAPTER_PATH.read_bytes()
        if hashlib.sha256(adapter_source).hexdigest() != policy["adapter_sha256"]:
            raise ProducerError("Adapter source differs from its bound hash")
        module = types.ModuleType(ADAPTER_PACKAGE + ".dino_adapter")
        module.__file__, module.__package__ = str(pi.ADAPTER_PATH), ADAPTER_PACKAGE
        exec(compile(adapter_source, str(pi.ADAPTER_PATH), "exec"), module.__dict__)
        self.namespace, self.adapter, self.study_inputs = namespace, module, module.study_inputs
        self.binding = {
            "kind": pi.KIND,
            "renderer": pi.RENDERER,
            "source_sha256": receipt["source_sha256"],
            "node_text_sha256": receipt["node_text_sha256"],
            "pixel_group_sha256": pi.canonical_sha256(pixel_group),
            "adapter_sha256": policy["adapter_sha256"],
        }

    def render(self, study_dir: Path, plane_map: dict) -> tuple[dict[str, np.ndarray], dict]:
        """One study through original ``render_inputs``; returns the cached row and slot facts."""
        captured: dict = {}
        namespace = self.namespace
        pick_slots, lat_of = namespace["pick_slots"], namespace["lat_of"]

        def capture_inputs(images, series_mask, features, positions, labels, mask, *, study_uid):
            out = self.study_inputs(
                images, series_mask, features, positions, labels, mask, study_uid=study_uid
            )
            captured.update(row=(images, series_mask, features, positions), inputs=out)
            return out

        def capture_slots(*args, **kwargs):
            captured["slots"] = pick_slots(*args, **kwargs)
            return captured["slots"]

        def capture_lat(*args, **kwargs):
            captured["lat"] = lat_of(*args, **kwargs)
            return captured["lat"]

        unlabeled = torch.zeros(len(TARGETS), dtype=torch.float32)
        setattr(self.adapter, "study_inputs", capture_inputs)
        namespace["pick_slots"], namespace["lat_of"] = capture_slots, capture_lat
        try:
            inputs = self.adapter.render_inputs(
                namespace, study_dir, plane_map, unlabeled, unlabeled.clone()
            )
        finally:
            setattr(self.adapter, "study_inputs", self.study_inputs)
            namespace["pick_slots"], namespace["lat_of"] = pick_slots, lat_of
        if "row" not in captured or captured["inputs"] is not inputs:
            raise ProducerError("Original renderer did not pass through the bound adapter")
        images, mask, features, positions = captured["row"]
        arrays = {
            "images": images.numpy().copy(),
            "slot_mask": mask.numpy().copy(),
            "features": features.numpy().copy(),
            "positions": positions.numpy().copy(),
        }
        uid = Path(study_dir).name
        chosen = captured["slots"][uid]
        names = [slot[0] for slot in namespace["SLOTS"]]
        facts = {
            "slot_series": [
                str(chosen[n]["SeriesInstanceUID"]) if n in chosen else None for n in names
            ],
            "slice_order": [
                list(chosen[n].get("ordered") or []) if n in chosen else None for n in names
            ],
            "slot_header": [
                [float(chosen[n]["px"]), int(chosen[n]["Rows"]), int(chosen[n]["Columns"])]
                if n in chosen
                else None
                for n in names
            ],
            "laterality": captured["lat"].get(uid),
            "inputs": inputs,
        }
        return arrays, facts

    def roundtrip(self, arrays: Mapping[str, np.ndarray], uid: str) -> dict:
        """The trainer's adapter call on stored arrays (labels absent, masks zero)."""
        unlabeled = torch.zeros(len(TARGETS), dtype=torch.float32)
        return self.study_inputs(
            torch.from_numpy(arrays["images"]),
            torch.from_numpy(arrays["slot_mask"]),
            torch.from_numpy(arrays["features"]),
            torch.from_numpy(arrays["positions"]),
            unlabeled,
            unlabeled.clone(),
            study_uid=uid,
        )


def create_intent(
    cache: Path,
    *,
    policy: dict,
    pixel_group: dict,
    source_revision: str,
    fitting_uids: Iterable[str],
    evaluation_uids: Iterable[str],
    source_studies: Mapping[str, str],
    caps: Mapping[str, int],
    binding: Mapping | None = None,
    cohort_sizes: tuple[int, int] = COHORT_SIZES,
) -> str:
    """Validate membership/source/caps and create the operation intent once; returns its SHA."""
    pi.validate_public_input(policy)
    pi._validate_pixel_group(pixel_group)
    if pi.canonical_sha256(pixel_group) != policy["pixel_group_sha256"]:
        raise ProducerError("Pixel group differs from the bound policy")
    if hashlib.sha256(pi.ADAPTER_PATH.read_bytes()).hexdigest() != policy["adapter_sha256"]:
        raise ProducerError("Adapter source differs from its bound hash")
    if not isinstance(source_revision, str) or not source_revision.strip():
        raise ProducerError("Original source revision required")
    fitting, evaluation = sorted(fitting_uids), sorted(evaluation_uids)
    _validate_cohort(fitting, evaluation, cohort_sizes)
    cohort = set(fitting) | set(evaluation)
    if (
        not isinstance(source_studies, Mapping)
        or set(source_studies) != cohort
        or not all(pi._is_sha(value) for value in source_studies.values())
    ):
        raise ProducerError("Every member needs exactly one bound original-source digest")
    if (
        not isinstance(caps, Mapping)
        or set(caps) != CAP_KEYS
        or any(type(value) is not int or value < 1 for value in caps.values())
        or caps["max_total_bytes"] < len(cohort) * member_bytes()
    ):
        raise ProducerError("Positive caps covering the complete exact member bytes required")
    root = _root(cache)
    if root.exists() or root.is_symlink():
        raise ProducerError("A cache operation already exists; reconcile it read-only")
    if shutil.disk_usage(Path(cache)).free < caps["min_free_bytes"] + len(cohort) * member_bytes():
        raise ProducerError("Free disk cannot fund complete members plus the reserve")
    intent = {
        "schema_version": SCHEMA_VERSION,
        **{key: policy[key] for key in sorted(pi.POLICY_KEYS)},
        "pixel_group": pixel_group,
        "source_revision": source_revision,
        "fitting_uids": fitting,
        "evaluation_uids": evaluation,
        "fitting_uids_sha256": pi.roster_sha256(fitting),
        "evaluation_uids_sha256": pi.roster_sha256(evaluation),
        "cohort_sha256": pi.roster_sha256(cohort),
        "source_studies": dict(sorted(source_studies.items())),
        "cohort_sizes": list(cohort_sizes),
        "member_bytes": member_bytes(),
        "caps": dict(sorted(caps.items())),
        "binding": dict(binding or {}),
        "labels_stored": False,
    }
    payload = _json_bytes(intent)
    root.mkdir(mode=0o700)
    for name in (MEMBERS, RECEIPTS, STAGING):
        (root / name).mkdir(mode=0o700)
    _write_exclusive(root / INTENT, payload)
    return hashlib.sha256(payload).hexdigest()


def load_intent(cache: Path, policy: dict | None = None) -> tuple[dict, str]:
    root = _root(cache)
    if root.is_symlink() or not (root / INTENT).is_file() or (root / INTENT).is_symlink():
        raise ProducerError("No cache intent exists")
    payload = (root / INTENT).read_bytes()
    intent = json.loads(payload)
    if (
        not isinstance(intent, dict)
        or intent.get("schema_version") != SCHEMA_VERSION
        or intent.get("labels_stored") is not False
        or intent.get("member_bytes") != member_bytes()
    ):
        raise ProducerError("Cache intent schema differs")
    stated = {key: intent.get(key) for key in pi.POLICY_KEYS}
    pi.validate_public_input(stated)
    if policy is not None and stated != policy:
        raise ProducerError("Cache intent differs from the bound policy")
    pi._validate_pixel_group(intent["pixel_group"])
    _validate_cohort(intent["fitting_uids"], intent["evaluation_uids"], intent["cohort_sizes"])
    if (
        intent["fitting_uids_sha256"] != pi.roster_sha256(intent["fitting_uids"])
        or intent["evaluation_uids_sha256"] != pi.roster_sha256(intent["evaluation_uids"])
        or set(intent["source_studies"])
        != set(intent["fitting_uids"]) | set(intent["evaluation_uids"])
    ):
        raise ProducerError("Cache intent roster digests differ")
    return intent, hashlib.sha256(payload).hexdigest()


def _paths(root: Path, uid: str) -> tuple[Path, Path, Path]:
    return (
        root / MEMBERS / f"{uid}.npz",
        root / RECEIPTS / f"{uid}.json",
        root / STAGING / f"{uid}.npz.partial",
    )


def _state(root: Path, uid: str) -> str:
    member, receipt, partial = (p.exists() or p.is_symlink() for p in _paths(root, uid))
    return {
        (False, False, False): "pending",
        (False, False, True): "stale_partial",
        (False, True, True): "receipt_unlinked",
        (True, True, True): "linked_with_partial",
        (True, True, False): "complete",
    }.get((member, receipt, partial), "corrupt")


def _read_receipt(root: Path, uid: str, intent_sha: str) -> dict:
    _, path, _ = _paths(root, uid)
    if path.is_symlink():
        raise ProducerError("Receipt must be a regular file")
    receipt = json.loads(path.read_bytes())
    if (
        receipt.get("study_uid") != uid
        or receipt.get("intent_sha256") != intent_sha
        or receipt.get("member", {}).get("path") != f"{MEMBERS}/{uid}.npz"
        or receipt["member"].get("bytes") != member_bytes()
        or not pi._is_sha(receipt["member"].get("sha256"))
    ):
        raise ProducerError("Per-study receipt differs from the intent")
    return receipt


def _verify_file(path: Path, receipt: dict) -> None:
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size != receipt["member"]["bytes"]
        or _file_sha256(path) != receipt["member"]["sha256"]
    ):
        raise ProducerError("Member bytes differ from their immutable receipt")


def _committed_bytes(root: Path) -> int:
    total = 0
    for directory in (MEMBERS, STAGING):
        total += sum(p.lstat().st_size for p in (root / directory).iterdir())
    return total


def _verify_source(study_dir: Path, uid: str, records: list[dict]) -> dict:
    """Exact original files only: no extra/missing file, symlink, size, hash or UID drift."""
    import pydicom

    expected = {
        PurePosixPath(r["path"]).relative_to(PurePosixPath(SPLIT, uid)).as_posix(): r
        for r in records
    }
    if study_dir.is_symlink() or not study_dir.is_dir() or study_dir.name != uid:
        raise ProducerError("Original study directory required")
    found = {}
    for series in os.scandir(study_dir):
        if series.is_symlink() or not series.is_dir(follow_symlinks=False):
            raise ProducerError("Study directory holds only original series directories")
        for entry in os.scandir(series.path):
            if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
                raise ProducerError("Series directory holds only original regular files")
            found[f"{series.name}/{entry.name}"] = Path(entry.path)
    if set(found) != set(expected):
        raise ProducerError("Source files differ from the bound original inventory")
    stats = {}
    for name, path in sorted(found.items()):
        record = expected[name]
        if path.stat().st_size != record["bytes"] or _file_sha256(path) != record["sha256"]:
            raise ProducerError("Original source bytes differ from the inventory")
        header = pydicom.dcmread(path, stop_before_pixels=True)
        if (
            str(getattr(header, "StudyInstanceUID", "")) != uid
            or str(getattr(header, "SeriesInstanceUID", "")) != record["series_uid"]
            or str(getattr(header, "SOPInstanceUID", "")) != record["sop_uid"]
        ):
            raise ProducerError("Original DICOM UIDs differ from the inventory")
        status = path.stat()
        stats[name] = (status.st_size, status.st_mtime_ns, status.st_ino)
    return stats


def _restat(study_dir: Path, stats: dict) -> None:
    for name, value in stats.items():
        status = (study_dir / name).stat()
        if (status.st_size, status.st_mtime_ns, status.st_ino) != value:
            raise ProducerError("Original source changed during rendering")


def stage_member(
    cache: Path,
    uid: str,
    source_root: Path,
    records: list[dict],
    plane_map: dict,
    *,
    renderer: Renderer,
) -> dict:
    """Render, validate and exclusively stage one member; completed members never re-render."""
    intent, intent_sha = load_intent(cache)
    root = _root(cache)
    if uid not in intent["source_studies"]:
        raise ProducerError("Study is outside the declared cohort")
    _safe_uid(uid)
    if (root / INDEX).exists() or (root / INDEX).is_symlink():
        raise ProducerError("Cache is finalized and immutable")
    expected = {key: intent[key] for key in pi.POLICY_KEYS}
    if renderer.binding != expected or expected["pixel_group_sha256"] != pi.canonical_sha256(
        intent["pixel_group"]
    ):
        raise ProducerError("Renderer binding differs from the cache intent")
    member, receipt_path, partial = _paths(root, uid)
    state = _state(root, uid)
    if state in {"complete", "receipt_unlinked", "linked_with_partial"}:
        receipt = _read_receipt(root, uid, intent_sha)
        if state == "receipt_unlinked":
            _verify_file(partial, receipt)
            os.link(partial, member)
            _fsync_directory(member.parent)
        _verify_file(member, receipt)
        if state != "complete":
            partial.unlink()
            _fsync_directory(partial.parent)
        return receipt | {"reused": True}
    if state != "pending":
        raise ProducerError(f"Interrupted member state {state}; reconcile before any write")
    if study_source_sha256(uid, records) != intent["source_studies"][uid]:
        raise ProducerError("Inventory rows differ from the bound original-source digest")
    _require_funded(root, intent)
    payload, receipt, _ = render_member(
        uid,
        source_root,
        records,
        plane_map,
        renderer=renderer,
        source_sha256=intent["source_studies"][uid],
        source_revision=intent["source_revision"],
        role="fitting" if uid in set(intent["fitting_uids"]) else "evaluation",
    )
    return _commit(root, intent, uid, payload, receipt | {"intent_sha256": intent_sha})


def _require_funded(root: Path, intent: dict) -> None:
    caps = intent["caps"]
    if _committed_bytes(root) + intent["member_bytes"] > caps["max_total_bytes"]:
        raise ProducerError("Member would exceed the funded cache bytes")
    if shutil.disk_usage(root).free - intent["member_bytes"] < caps["min_free_bytes"]:
        raise ProducerError("Member would breach the free-disk reserve")


def _commit(root: Path, intent: dict, uid: str, payload: bytes, receipt: dict) -> dict:
    member, receipt_path, partial = _paths(root, uid)
    _require_funded(root, intent)
    _write_exclusive(partial, payload)
    _write_exclusive(receipt_path, _json_bytes(receipt))
    os.link(partial, member)
    _fsync_directory(member.parent)
    partial.unlink()
    _fsync_directory(partial.parent)
    return receipt | {"reused": False}


def render_member(
    uid: str,
    source_root: Path,
    records: list[dict],
    plane_map: dict,
    *,
    renderer: Renderer,
    source_sha256: str,
    source_revision: str,
    role: str,
) -> tuple[bytes, dict, dict]:
    """Verify exact source, render once, prove bit-exact storage and adapter replay; no writes.

    Returns the exact member bytes, its receipt (without an intent binding) and the
    renderer-to-adapter inputs. Used by ``stage_member`` and by placements that stage
    elsewhere and import through ``import_member``.
    """
    _safe_uid(uid)
    if study_source_sha256(uid, records) != source_sha256:
        raise ProducerError("Inventory rows differ from the bound original-source digest")
    if role not in {"fitting", "evaluation"}:
        raise ProducerError("Member role must be fitting or evaluation")
    study_dir = Path(source_root) / SPLIT / uid
    if Path(source_root).is_symlink() or not Path(source_root).is_dir():
        raise ProducerError("Original source root required")
    stats = _verify_source(study_dir, uid, records)
    arrays, facts = renderer.render(study_dir, plane_map)
    _restat(study_dir, stats)
    payload = _serialize(arrays)
    if len(payload) != member_bytes():
        raise ProducerError("Serialized member bytes differ from the exact contract")
    stored = _load_arrays(payload)
    if not all(np.array_equal(stored[name], arrays[name]) for name in pi.MEMBER_ARRAYS):
        raise ProducerError("Stored member does not round-trip bit-exactly")
    replay, original = renderer.roundtrip(stored, uid), facts["inputs"]
    if not inputs_equal(replay, original):
        raise ProducerError("Stored member differs from the renderer-to-adapter output")
    receipt = {
        "schema_version": SCHEMA_VERSION,
        "study_uid": uid,
        "role": role,
        "member": {
            "path": f"{MEMBERS}/{uid}.npz",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        },
        "arrays": {
            name: {"dtype": str(arrays[name].dtype), "shape": list(arrays[name].shape)}
            for name in pi.MEMBER_ARRAYS
        },
        "occupied_slots": int(arrays["slot_mask"].sum()),
        "slot_series": facts["slot_series"],
        "slice_order": facts["slice_order"],
        "slot_header": facts["slot_header"],
        "laterality": facts["laterality"],
        "source": {
            "split": SPLIT,
            "source_revision": source_revision,
            "records_sha256": source_sha256,
            "files": len(records),
            "bytes": sum(r["bytes"] for r in records),
        },
        "labels_stored": False,
    }
    return payload, receipt, original


def inputs_equal(left: Mapping, right: Mapping) -> bool:
    """Exact equality of two adapter outputs (tensors bit-equal, strings equal)."""
    return set(left) == set(right) and all(
        left[key] == right[key]
        if isinstance(right[key], str)
        else torch.equal(left[key], right[key])
        for key in right
    )


def import_member(cache: Path, uid: str, payload: bytes, receipt: Mapping) -> dict:
    """Commit one member rendered by ``render_member`` elsewhere, under this cache's intent.

    The bytes must equal the receipted exact member, the source digest/revision/role must
    equal this intent's and the arrays must satisfy the contract; only pending members are
    written, through the same exclusive partial/receipt/link sequence as ``stage_member``.
    """
    intent, intent_sha = load_intent(cache)
    root = _root(cache)
    _safe_uid(uid)
    if uid not in intent["source_studies"]:
        raise ProducerError("Study is outside the declared cohort")
    if (root / INDEX).exists() or (root / INDEX).is_symlink():
        raise ProducerError("Cache is finalized and immutable")
    if _state(root, uid) != "pending":
        raise ProducerError("Only pending members may be imported")
    role = "fitting" if uid in set(intent["fitting_uids"]) else "evaluation"
    source = receipt.get("source", {})
    member = receipt.get("member", {})
    if (
        receipt.get("schema_version") != SCHEMA_VERSION
        or receipt.get("study_uid") != uid
        or receipt.get("role") != role
        or receipt.get("labels_stored") is not False
        or "intent_sha256" in receipt
        or source.get("split") != SPLIT
        or source.get("source_revision") != intent["source_revision"]
        or source.get("records_sha256") != intent["source_studies"][uid]
        or member.get("path") != f"{MEMBERS}/{uid}.npz"
        or member.get("bytes") != intent["member_bytes"]
        or len(payload) != intent["member_bytes"]
        or hashlib.sha256(payload).hexdigest() != member.get("sha256")
    ):
        raise ProducerError("Imported member differs from this intent or its receipt")
    arrays = _load_arrays(payload)
    if receipt.get("arrays") != {
        name: {"dtype": str(arrays[name].dtype), "shape": list(arrays[name].shape)}
        for name in pi.MEMBER_ARRAYS
    } or receipt.get("occupied_slots") != int(arrays["slot_mask"].sum()):
        raise ProducerError("Imported member arrays differ from their receipt")
    return _commit(root, intent, uid, payload, dict(receipt) | {"intent_sha256": intent_sha})


def release_stale_partial(cache: Path, uid: str) -> None:
    """Explicitly discard an unreceipted partial; completed or receipted members stay untouched."""
    root = _root(cache)
    load_intent(cache)
    if _state(root, _safe_uid(uid)) != "stale_partial":
        raise ProducerError("Only an unreceipted stale partial can be released")
    _paths(root, uid)[2].unlink()
    _fsync_directory(root / STAGING)


def reconcile(cache: Path, *, verify_bytes: bool = True) -> dict:
    """Read-only state of an interrupted or completed operation; never writes or renders."""
    intent, intent_sha = load_intent(cache)
    root = _root(cache)
    cohort = set(intent["source_studies"])
    states: dict[str, list[str]] = {}
    for uid in sorted(cohort):
        state = _state(root, uid)
        if state in {"complete", "linked_with_partial", "receipt_unlinked"}:
            try:
                receipt = _read_receipt(root, uid, intent_sha)
                member, _, partial = _paths(root, uid)
                if verify_bytes:
                    _verify_file(partial if state == "receipt_unlinked" else member, receipt)
            except (ProducerError, ValueError, KeyError, OSError):
                state = "corrupt"
        states.setdefault(state, []).append(uid)
    expected = {
        MEMBERS: {f"{uid}.npz" for uid in cohort},
        RECEIPTS: {f"{uid}.json" for uid in cohort},
        STAGING: {f"{uid}.npz.partial" for uid in cohort},
    }
    unexpected = sorted(
        f"{name}/{entry.name}"
        for name, allowed in expected.items()
        for entry in os.scandir(root / name)
        if entry.name not in allowed or entry.is_symlink()
    )
    unexpected += sorted(
        entry.name
        for entry in os.scandir(root)
        if entry.name not in {INTENT, INDEX, FINALIZED, *expected}
    )
    return {
        "intent_sha256": intent_sha,
        "states": states,
        "counts": {state: len(uids) for state, uids in sorted(states.items())},
        "unexpected": unexpected,
        "finalized": (root / INDEX).exists(),
        "committed_bytes": _committed_bytes(root),
    }


def finalize(cache: Path, *, policy: dict) -> str:
    """Write the trainer index once, only after complete membership with bound bytes."""
    intent, intent_sha = load_intent(cache, policy)
    root = _root(cache)
    if (root / INDEX).exists() or (root / INDEX).is_symlink():
        raise ProducerError("Cache is already finalized")
    status = reconcile(cache, verify_bytes=False)
    if status["unexpected"] or set(status["states"]) != {"complete"}:
        raise ProducerError("Complete membership with no interrupted or foreign entries required")
    members = {}
    for uid in sorted(intent["source_studies"]):
        receipt = _read_receipt(root, uid, intent_sha)
        path = root / receipt["member"]["path"]
        _verify_file(path, receipt)
        _load_arrays(path.read_bytes())
        members[uid] = dict(receipt["member"])
    total = sum(m["bytes"] for m in members.values())
    if total > intent["caps"]["max_total_bytes"] or total != len(members) * member_bytes():
        raise ProducerError("Member bytes exceed or differ from the funded bound")
    index = {
        "schema_version": 1,
        "kind": intent["kind"],
        "renderer": intent["renderer"],
        "source_sha256": intent["source_sha256"],
        "node_text_sha256": intent["node_text_sha256"],
        "pixel_group": intent["pixel_group"],
        "source_revision": intent["source_revision"],
        "fitting_uids_sha256": intent["fitting_uids_sha256"],
        "evaluation_uids_sha256": intent["evaluation_uids_sha256"],
        "cohort_sha256": pi.roster_sha256(members),
        "members": members,
    }
    payload = json.dumps(index, sort_keys=True, indent=1).encode()
    staged = root / STAGING / "index.json.partial"
    _write_exclusive(staged, payload)
    os.link(staged, root / INDEX)
    _fsync_directory(root)
    staged.unlink()
    _, index_sha = pi.load_public_index(Path(cache), policy, intent["source_revision"])
    if index_sha != hashlib.sha256(payload).hexdigest():
        raise ProducerError("Finalized index differs from the written bytes")
    _write_exclusive(
        root / FINALIZED,
        _json_bytes(
            {"intent_sha256": intent_sha, "index_sha256": index_sha, "members": len(members)}
        ),
    )
    return index_sha
