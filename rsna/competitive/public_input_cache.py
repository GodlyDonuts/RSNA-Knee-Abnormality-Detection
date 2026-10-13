"""Streaming source-bound producer for the public ordered-triplet cohort cache.

Research engineering only. One complete original study is fetched, rendered by the
reviewed Pilkwang v15 renderer through the exact hashed standalone adapter, stored once
as uint8 rows and released before the next study starts; no all-studies dictionary of
pixels is held. The accepted disjoint fitting/evaluation rosters and every original
object are validated before the intent is written. Members are written exactly once
through exclusive paths and journaled; the index is linked atomically only after every
member is complete. An interrupted cache is reconciled read-only: completed members are
verified, never overwritten or rendered again, and unjournaled files refuse resumption.
Raw objects, indices, journals and case files stay on the owning node; callers may only
export the sanitized summaries returned here (counts, hashes and provenance).
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import types
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote

import numpy as np
import torch

from rsna.competitive import public_input as pi
from rsna.schema import TARGETS

PRODUCER = "public_ordered_triplet_streaming_producer_v1"
ADAPTER_PACKAGE = "tools.research.public_dinov2_input_reference"
RAW_SPLIT = "train_series"
INTENT, JOURNAL, INDEX = "intent.json", "members.jsonl", "index.json"
UID = re.compile(r"[0-9]+(\.[0-9]+)*")
SHA = re.compile(r"[0-9a-f]{64}")
OBJECT_KEYS = {"split", "study_uid", "series_uid", "sop_uid", "path", "sha256", "bytes", "storage"}
JOURNAL_KEYS = {"uid", "path", "bytes", "sha256", "source_objects", "source_bytes"}
BLOCK = 1 << 20


class ProducerFailure(RuntimeError):
    """Sanitized stop: carries a stage and an exception type name, never IDs or text."""

    def __init__(self, stage: str, error_type: str, completed: int):
        super().__init__(f"{stage}:{error_type}")
        self.stage, self.error_type, self.completed = stage, error_type, completed


class TransportFailure(RuntimeError):
    """Original object retrieval refused; the message is a fixed category only."""


class Transport(Protocol):
    def fetch(self, record: Mapping[str, Any], target: Path) -> None: ...


@dataclass(frozen=True)
class Caps:
    max_study_objects: int
    max_study_source_bytes: int
    max_total_source_bytes: int
    max_member_bytes: int
    max_cache_bytes: int
    reserve_bytes: int

    def validate(self) -> None:
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError("Positive integer producer caps required")


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def file_sha256(path: Path) -> tuple[int, str]:
    digest, size = hashlib.sha256(), 0
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BLOCK), b""):
            digest.update(block)
            size += len(block)
    return size, digest.hexdigest()


def _uid(value: object) -> str:
    if not isinstance(value, str) or len(value) > 64 or not UID.fullmatch(value):
        raise ValueError("Original DICOM UID required")
    return value


def validate_rosters(fitting_uids: Iterable, evaluation_uids: Iterable, expected: tuple[int, int]):
    fitting, evaluation = list(fitting_uids), list(evaluation_uids)
    for uids in (fitting, evaluation):
        for uid in uids:
            _uid(uid)
    if (
        len(set(fitting)) != len(fitting)
        or len(set(evaluation)) != len(evaluation)
        or set(fitting) & set(evaluation)
        or (len(fitting), len(evaluation)) != expected
    ):
        raise ValueError("Exact disjoint accepted fitting/evaluation rosters required")
    return set(fitting), set(evaluation)


def validate_objects(objects: Mapping[str, list], cohort: set[str], caps: Caps) -> dict:
    """Bind every original object of the complete cohort before any intent is written."""
    if set(objects) != cohort:
        raise ValueError("Original inventory must cover the cohort exactly")
    paths, sops, total = set(), set(), 0
    for uid in sorted(cohort):
        records = objects[uid]
        if not records or len(records) > caps.max_study_objects:
            raise ValueError("Study object count outside the admitted cap")
        study = 0
        for record in records:
            if not isinstance(record, dict) or set(record) != OBJECT_KEYS:
                raise ValueError("Original object fields differ")
            series, sop = _uid(record["series_uid"]), _uid(record["sop_uid"])
            if (
                record["split"] != "train"
                or record["study_uid"] != uid
                or record["path"] != f"{RAW_SPLIT}/{uid}/{series}/{sop}.dcm"
                or record["storage"] not in ("lfs", "git")
                or type(record["bytes"]) is not int
                or record["bytes"] < 1
                or not isinstance(record["sha256"], str)
                or not SHA.fullmatch(record["sha256"])
                or record["path"] in paths
                or sop in sops
            ):
                raise ValueError("Invalid original object binding")
            paths.add(record["path"])
            sops.add(sop)
            study += record["bytes"]
        if study > caps.max_study_source_bytes:
            raise ValueError("Study source bytes exceed the admitted cap")
        total += study
    if total > caps.max_total_source_bytes:
        raise ValueError("Cohort source bytes exceed the admitted cap")
    ordered = {uid: sorted(objects[uid], key=lambda r: r["path"]) for uid in sorted(cohort)}
    return {
        "objects": len(paths),
        "source_bytes": total,
        "inventory_binding_sha256": hashlib.sha256(canonical(ordered)).hexdigest(),
    }


def validate_member(arrays: Mapping[str, np.ndarray]) -> None:
    if set(arrays) != set(pi.MEMBER_ARRAYS) or any(
        not isinstance(arrays[name], np.ndarray)
        or arrays[name].dtype != dtype
        or arrays[name].shape != shape
        for name, (dtype, shape) in pi.MEMBER_ARRAYS.items()
    ):
        raise ValueError("Rendered row differs from the source contract")
    mask, positions = arrays["slot_mask"], arrays["positions"]
    validity = positions[..., 1]
    if (
        not mask.any()
        or arrays["images"][~mask].any()
        or arrays["features"][~mask].any()
        or positions[~mask].any()
        or not np.isfinite(arrays["features"]).all()
        or not np.isfinite(positions).all()
        or not np.isin(validity, (0, 1)).all()
        or positions[..., 0][validity == 0].any()
    ):
        raise ValueError("Rendered row masks or geometry differ from the source contract")


def serialize_member(arrays: Mapping[str, np.ndarray]) -> bytes:
    validate_member(arrays)
    buffer = io.BytesIO()
    np.savez(
        buffer,
        images=arrays["images"],
        slot_mask=arrays["slot_mask"],
        features=arrays["features"],
        positions=arrays["positions"],
    )
    return buffer.getvalue()


def _write_new(path: Path, payload: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_dir(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def build_intent(
    *,
    renderer: dict,
    pixel_group: dict,
    source_revision: str,
    raw_source_revision: str,
    fitting: set[str],
    evaluation: set[str],
    binding: dict,
    caps: Caps,
    provenance: Mapping[str, Any],
) -> dict:
    pi._validate_pixel_group(pixel_group)
    if renderer.get("pixel_group_sha256") != pi.canonical_sha256(pixel_group):
        raise ValueError("Renderer pixel group differs from the cache pixel group")
    return {
        "producer": PRODUCER,
        "kind": pi.KIND,
        "renderer": pi.RENDERER,
        "source_sha256": pi.SOURCE_SHA256,
        "node_text_sha256": pi.NODE_TEXT_SHA256,
        "renderer_binding": dict(renderer),
        "pixel_group": pixel_group,
        "source_revision": source_revision,
        "raw_source_revision": raw_source_revision,
        "fitting_uids_sha256": pi.roster_sha256(fitting),
        "evaluation_uids_sha256": pi.roster_sha256(evaluation),
        "cohort_sha256": pi.roster_sha256(fitting | evaluation),
        "fitting_studies": len(fitting),
        "evaluation_studies": len(evaluation),
        **binding,
        "caps": asdict(caps),
        "provenance": dict(provenance),
        "write_order": "sorted_study_uid",
        "labels_or_masks_stored": False,
    }


def _journal(root: Path, cohort: list[str]) -> list[dict]:
    path = root / JOURNAL
    payload = path.read_bytes() if path.exists() else b""
    if payload and not payload.endswith(b"\n"):
        raise ValueError("Interrupted journal record; read-only review required")
    records = [json.loads(line) for line in payload.splitlines()]
    for position, record in enumerate(records):
        if (
            not isinstance(record, dict)
            or set(record) != JOURNAL_KEYS
            or position >= len(cohort)
            or record["uid"] != cohort[position]
            or record["path"] != f"members/{cohort[position]}.npz"
            or type(record["bytes"]) is not int
            or not isinstance(record["sha256"], str)
            or not SHA.fullmatch(record["sha256"])
        ):
            raise ValueError("Journal differs from the sorted cohort write order")
    return records


def reconcile(cache: Path, intent: dict) -> dict:
    """Read-only state of an existing cache; nothing is created, changed or removed."""
    root = Path(cache) / pi.CACHE_DIRECTORY
    if not root.exists():
        return {"state": "absent", "completed_members": 0}
    if root.is_symlink() or not (root / INTENT).is_file() or (root / INTENT).is_symlink():
        raise ValueError("Existing cache lacks a regular intent; read-only review required")
    if json.loads((root / INTENT).read_bytes()) != json.loads(canonical(intent)):
        raise ValueError("Existing cache intent differs; never overwrite another cache")
    cohort = intent.get("_cohort")
    if not isinstance(cohort, list) or cohort != sorted(cohort):
        raise ValueError("Reconciliation requires the private sorted cohort roster")
    records = _journal(root, cohort)
    members = root / "members"
    present = {p.name for p in members.iterdir()} if members.is_dir() else set()
    journaled = {Path(r["path"]).name for r in records}
    if present - journaled:
        return {
            "state": "interrupted_unjournaled",
            "completed_members": len(records),
            "unjournaled_files": len(present - journaled),
        }
    for record in records:
        path = root / record["path"]
        if path.is_symlink() or file_sha256(path) != (record["bytes"], record["sha256"]):
            raise ValueError("Completed member bytes differ from the journal")
    state = "complete" if (root / INDEX).exists() else "partial"
    if state == "partial" and any(p.name.startswith(INDEX) for p in root.iterdir()):
        state = "interrupted_finalization"
    return {"state": state, "completed_members": len(records)}


def _remove_scratch_study(scratch: Path, study: Path) -> None:
    if study.is_symlink() or study.parent != scratch or not study.is_dir():
        raise ValueError("Owned per-study scratch required")
    shutil.rmtree(study)


def produce(
    cache: Path,
    *,
    render: Callable[[Path], dict[str, np.ndarray]],
    renderer_binding: dict,
    transport: Transport,
    objects: Mapping[str, list],
    fitting_uids: Iterable,
    evaluation_uids: Iterable,
    expected_counts: tuple[int, int],
    pixel_group: dict,
    source_revision: str,
    raw_source_revision: str,
    scratch: Path,
    caps: Caps,
    provenance: Mapping[str, Any] | None = None,
    resume: bool = False,
    disk_usage: Callable[[Path], Any] = shutil.disk_usage,
    header_check: Callable[[Path, Mapping[str, Any]], None] | None = None,
) -> dict:
    """Stream the complete cohort into a source-bound cache; returns a sanitized summary."""
    caps.validate()
    fitting, evaluation = validate_rosters(fitting_uids, evaluation_uids, expected_counts)
    cohort = sorted(fitting | evaluation)
    binding = validate_objects(objects, set(cohort), caps)
    intent = build_intent(
        renderer=renderer_binding,
        pixel_group=pixel_group,
        source_revision=source_revision,
        raw_source_revision=raw_source_revision,
        fitting=fitting,
        evaluation=evaluation,
        binding=binding,
        caps=caps,
        provenance=provenance or {},
    )
    private_intent = {**intent, "_cohort": cohort}
    root = Path(cache) / pi.CACHE_DIRECTORY
    state = reconcile(cache, private_intent)
    if state["state"] != "absent" and not resume:
        raise ProducerFailure("existing_cache_requires_explicit_resume", state["state"], 0)
    if state["state"] in ("interrupted_unjournaled", "interrupted_finalization"):
        raise ProducerFailure(
            "read_only_reconciliation", state["state"], state["completed_members"]
        )
    if state["state"] == "complete":
        return _summary(root, intent, resumed=state["completed_members"], produced=0)
    if Path(scratch).exists():
        raise ValueError("Fresh private scratch required")
    if state["state"] == "absent":
        root.mkdir(mode=0o700, parents=True, exist_ok=False)
        (root / "members").mkdir(mode=0o700)
        _write_new(root / INTENT, canonical(private_intent))
        _fsync_dir(root)
    Path(scratch).mkdir(mode=0o700, parents=True, exist_ok=False)
    check = header_check or verify_header
    done = state["completed_members"]
    stored = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    produced = 0
    with (root / JOURNAL).open("ab") as journal:
        for position in range(done, len(cohort)):
            uid = cohort[position]
            records = sorted(objects[uid], key=lambda r: r["path"])
            source_bytes = sum(r["bytes"] for r in records)
            stage = "storage"
            study = Path(scratch) / uid
            try:
                if (
                    disk_usage(Path(scratch)).free < source_bytes + caps.reserve_bytes
                    or disk_usage(root).free < caps.max_member_bytes + caps.reserve_bytes
                    or stored + caps.max_member_bytes > caps.max_cache_bytes
                ):
                    raise ValueError("Storage envelope exhausted")
                study.mkdir(mode=0o700)
                stage = "transport"
                for record in records:
                    target = study / record["series_uid"] / f"{record['sop_uid']}.dcm"
                    target.parent.mkdir(mode=0o700, exist_ok=True)
                    transport.fetch(record, target)
                    if target.is_symlink() or file_sha256(target) != (
                        record["bytes"],
                        record["sha256"],
                    ):
                        raise TransportFailure("object_bytes_differ")
                    check(target, record)
                stage = "render"
                payload = serialize_member(render(study))
                if len(payload) > caps.max_member_bytes:
                    raise ValueError("Member bytes exceed the admitted cap")
                stage = "write"
                member = root / "members" / f"{uid}.npz"
                _write_new(member, payload)
                entry = {
                    "uid": uid,
                    "path": f"members/{uid}.npz",
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                    "source_objects": len(records),
                    "source_bytes": source_bytes,
                }
                del payload
                journal.write(canonical(entry) + b"\n")
                journal.flush()
                os.fsync(journal.fileno())
                stored += entry["bytes"] + len(canonical(entry)) + 1
                produced += 1
            except Exception as error:
                raise ProducerFailure(stage, type(error).__name__, position) from None
            finally:
                if study.exists():
                    _remove_scratch_study(Path(scratch), study)
    Path(scratch).rmdir()
    _finalize(root, intent, cohort)
    return _summary(root, intent, resumed=done, produced=produced)


def _finalize(root: Path, intent: dict, cohort: list[str]) -> None:
    records = _journal(root, cohort)
    if len(records) != len(cohort):
        raise ValueError("Index requires every cohort member")
    members = {}
    for record in records:
        path = root / record["path"]
        if path.is_symlink() or file_sha256(path) != (record["bytes"], record["sha256"]):
            raise ValueError("Completed member bytes differ from the journal")
        members[record["uid"]] = {k: record[k] for k in ("path", "bytes", "sha256")}
    index = {
        "schema_version": 1,
        **{k: intent[k] for k in ("kind", "renderer", "source_sha256", "node_text_sha256")},
        "pixel_group": intent["pixel_group"],
        "source_revision": intent["source_revision"],
        "fitting_uids_sha256": intent["fitting_uids_sha256"],
        "evaluation_uids_sha256": intent["evaluation_uids_sha256"],
        "cohort_sha256": pi.roster_sha256(members),
        "members": members,
    }
    if index["cohort_sha256"] != intent["cohort_sha256"]:
        raise ValueError("Finalized members differ from the declared cohort")
    staged = root / (INDEX + ".partial")
    _write_new(staged, json.dumps(index, sort_keys=True, indent=1).encode())
    os.link(staged, root / INDEX)
    staged.unlink()
    _fsync_dir(root)


def _summary(root: Path, intent: dict, *, resumed: int, produced: int) -> dict:
    policy = {
        "kind": pi.KIND,
        "renderer": pi.RENDERER,
        "source_sha256": pi.SOURCE_SHA256,
        "node_text_sha256": pi.NODE_TEXT_SHA256,
        "pixel_group_sha256": pi.canonical_sha256(intent["pixel_group"]),
        "adapter_sha256": intent["renderer_binding"]["adapter_sha256"],
    }
    index, index_sha = pi.load_public_index(root.parent, policy, intent["source_revision"])
    if (
        index["fitting_uids_sha256"] != intent["fitting_uids_sha256"]
        or index["evaluation_uids_sha256"] != intent["evaluation_uids_sha256"]
        or index["cohort_sha256"] != intent["cohort_sha256"]
    ):
        raise ValueError("Finalized index differs from the intent")
    return {
        "status": "public_triplet_cohort_cache_complete_private",
        "members": len(index["members"]),
        "fitting_studies": intent["fitting_studies"],
        "evaluation_studies": intent["evaluation_studies"],
        "resumed_verified_members": resumed,
        "newly_produced_members": produced,
        "member_bytes": sum(m["bytes"] for m in index["members"].values()),
        "source_objects": intent["objects"],
        "source_bytes": intent["source_bytes"],
        "index_sha256": index_sha,
        "intent_sha256": hashlib.sha256((root / INTENT).read_bytes()).hexdigest(),
        "cohort_sha256": intent["cohort_sha256"],
        "fitting_uids_sha256": intent["fitting_uids_sha256"],
        "evaluation_uids_sha256": intent["evaluation_uids_sha256"],
        "inventory_binding_sha256": intent["inventory_binding_sha256"],
        "policy": policy,
        "peak_resident_studies": 1,
        "labels_or_masks_stored": False,
        "clinical_accuracy_claim": False,
    }


def verify_header(path: Path, record: Mapping[str, Any]) -> None:
    import pydicom

    header = pydicom.dcmread(path, stop_before_pixels=True)
    if (
        str(header.StudyInstanceUID) != record["study_uid"]
        or str(header.SeriesInstanceUID) != record["series_uid"]
        or str(header.SOPInstanceUID) != record["sop_uid"]
    ):
        raise ValueError("Original DICOM header identity differs")


class ResolveTransport:
    """Bounded streaming original-object transport at one pinned private revision.

    One object at a time is streamed to its exclusive scratch path; the stream stops as
    soon as it exceeds the inventory size. No retry, and failures expose only a fixed
    category, never the URL, credential or study identity.
    """

    def __init__(
        self,
        session: Any,
        token: str,
        repo_id: str,
        revision: str,
        *,
        timeout: tuple[float, float] = (10, 60),
        block: int = 65536,
    ):
        if not re.fullmatch(r"[0-9a-f]{40}", revision) or not token:
            raise ValueError("Pinned revision and credential required")
        self.session, self.revision, self.repo_id = session, revision, repo_id
        self._headers = {"Authorization": "Bearer " + token}
        self.timeout, self.block = timeout, block
        self.objects = self.bytes = 0

    def fetch(self, record: Mapping[str, Any], target: Path) -> None:
        url = f"https://huggingface.co/datasets/{self.repo_id}/resolve/{self.revision}/" + quote(
            record["path"], safe="/"
        )
        digest, count = hashlib.sha256(), 0
        try:
            with target.open("xb") as stream:
                try:
                    response = self.session.get(
                        url, headers=self._headers, timeout=self.timeout, stream=True
                    )
                except Exception:
                    raise TransportFailure("transport") from None
                with response:
                    if response.status_code != 200:
                        raise TransportFailure("http_status")
                    try:
                        for block in response.iter_content(self.block):
                            count += len(block)
                            if count > record["bytes"]:
                                raise TransportFailure("size_exceeded")
                            digest.update(block)
                            stream.write(block)
                    except TransportFailure:
                        raise
                    except Exception:
                        raise TransportFailure("stream") from None
                stream.flush()
                os.fsync(stream.fileno())
            if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise TransportFailure("object_bytes_differ")
        except FileExistsError:
            raise
        except BaseException:
            if target.exists() and not target.is_symlink():
                target.unlink()
            raise
        self.objects += 1
        self.bytes += count


class LocalTransport:
    """Bounded copy of one retained original object from the owning-node inventory root.

    Reads only regular files inside the root, never follows a symlink, stops past the
    inventory size and leaves nothing behind on failure. No network is involved.
    """

    def __init__(self, root: Path, *, block: int = BLOCK):
        self.root = Path(root).resolve(strict=True)
        self.block, self.objects, self.bytes = block, 0, 0

    def fetch(self, record: Mapping[str, Any], target: Path) -> None:
        source = self.root / record["path"]
        digest, count = hashlib.sha256(), 0
        try:
            with target.open("xb") as stream:
                parts = Path(record["path"]).parts
                walk = [self.root.joinpath(*parts[:depth]) for depth in range(1, len(parts) + 1)]
                if (
                    len(parts) != 4
                    or any(part in ("", ".", "..") for part in parts)
                    or any(path.is_symlink() for path in walk)
                    or not source.is_file()
                ):
                    raise TransportFailure("source_path")
                with source.open("rb") as original:
                    for block in iter(lambda: original.read(self.block), b""):
                        count += len(block)
                        if count > record["bytes"]:
                            raise TransportFailure("size_exceeded")
                        digest.update(block)
                        stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            if count != record["bytes"] or digest.hexdigest() != record["sha256"]:
                raise TransportFailure("object_bytes_differ")
        except FileExistsError:
            raise
        except BaseException as error:
            if target.exists() and not target.is_symlink():
                target.unlink()
            if isinstance(error, TransportFailure):
                raise
            raise TransportFailure("source_read") from None
        self.objects += 1
        self.bytes += count


def load_bound_adapter(adapter_sha256: str) -> types.ModuleType:
    """Exact hashed adapter bytes, executed with its package context for the renderer."""
    source = pi.ADAPTER_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != adapter_sha256:
        raise ValueError("Public triplet adapter source differs from its bound hash")
    module = types.ModuleType(ADAPTER_PACKAGE + "._source_bound_dino_adapter")
    module.__file__ = str(pi.ADAPTER_PATH)
    module.__package__ = ADAPTER_PACKAGE
    exec(compile(source, str(pi.ADAPTER_PATH), "exec"), module.__dict__)
    return module


class BoundRenderer:
    """Original renderer -> label-free uint8 row through the exact adapter `render_inputs`.

    The adapter's own `study_inputs` still validates every row; this wrapper only
    records the source arrays it receives, so no pixel, feature or position is recomputed.
    Zero targets with an all-zero mask are passed because the cache never stores labels.
    """

    def __init__(
        self, namespace: dict, receipt: Mapping[str, Any], pixel_group: dict, plane_map: dict
    ):
        pi._validate_pixel_group(pixel_group)
        if (
            receipt.get("source_sha256") != pi.SOURCE_SHA256
            or receipt.get("node_text_sha256") != pi.NODE_TEXT_SHA256
            or [slot[0] for slot in namespace["SLOTS"]] != list(pixel_group["slots"])
        ):
            raise ValueError(
                "Reviewed v15 renderer configured with the member pixel group required"
            )
        self.adapter_sha256 = hashlib.sha256(pi.ADAPTER_PATH.read_bytes()).hexdigest()
        self.adapter = load_bound_adapter(self.adapter_sha256)
        self.original_study_inputs = getattr(self.adapter, "study_inputs")
        setattr(self.adapter, "study_inputs", self._capture)
        self.namespace, self.plane_map = namespace, plane_map
        self.binding = {
            "renderer": pi.RENDERER,
            "source_sha256": pi.SOURCE_SHA256,
            "node_text_sha256": pi.NODE_TEXT_SHA256,
            "pixel_group_sha256": pi.canonical_sha256(pixel_group),
            "adapter_sha256": self.adapter_sha256,
            "plane_map_sha256": hashlib.sha256(canonical(dict(plane_map))).hexdigest(),
        }
        self._captured: dict | None = None

    def _capture(self, images, series_mask, features, slice_positions, labels, label_mask, **kw):
        result = self.original_study_inputs(
            images, series_mask, features, slice_positions, labels, label_mask, **kw
        )
        self._captured = {
            "images": images.numpy().copy(),
            "slot_mask": series_mask.numpy().copy(),
            "features": features.numpy().copy(),
            "positions": slice_positions.numpy().copy(),
            "study_uid": kw["study_uid"],
        }
        return result

    def __call__(self, study_dir: Path) -> dict[str, np.ndarray]:
        self._captured = None
        zeros = torch.zeros(len(TARGETS), dtype=torch.float32)
        result = self.adapter.render_inputs(
            self.namespace, Path(study_dir), self.plane_map, zeros, zeros.clone()
        )
        captured, self._captured = self._captured, None
        if (
            captured is None
            or captured.pop("study_uid") != Path(study_dir).name
            or result["study_uid"] != Path(study_dir).name
            or result["label_mask"].any()
        ):
            raise ValueError("Adapter did not render this original study")
        del result
        return captured
