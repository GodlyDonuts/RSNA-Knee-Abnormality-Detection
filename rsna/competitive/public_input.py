"""Source-bound public ordered-triplet input policy; research comparison only.

Pixels come from the reviewed Pilkwang v15 renderer (pilkwang/rsna-knee-baseline-v1,
Kaggle notebook version 15, by Pilkwang), stored once as uint8 [6,12,336,336] rows.
The exact standalone ``dino_adapter.study_inputs`` bytes form the ten overlapping
three-slice views /255; the encoder alone applies ImageNet normalization. Labels
and masks always come from the unchanged fitting/evaluation frames. No native
cache, float cache, label, crop or geometry is read or reconstructed here.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import types
from collections.abc import Callable
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
import torch

from rsna.schema import TARGETS

KIND = "public_ordered_triplet_v1"
RENDERER = "pilkwang/rsna-knee-baseline-v1/v15"
SOURCE_SHA256 = "b32a9155fcc73c519e75e78c08e07d30393efc591c2798a3e40ca92f39832bb8"
NODE_TEXT_SHA256 = "6805a20dda177a1bb1ebeac82124acad2fd0732182b0576464f020e6eab9b599"
ADAPTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "tools/research/public_dinov2_input_reference/dino_adapter.py"
)
PIXEL_GROUP_SHA256 = "bbd7564015afc8a823fd122cd7150f5cf9025f3cdadf3396f0170406855eaa59"
CACHE_DIRECTORY = "public-ordered-triplet-v1"
PRODUCER_REQUEST, PRODUCER_JOURNAL = "producer-request.json", "producer-journal.jsonl"
SIDE, SLOTS, SLICES, VIEWS = 336, 6, 12, 10
MEMBER_ARRAYS = {
    "images": (np.uint8, (SLOTS, SLICES, SIDE, SIDE)),
    "slot_mask": (np.bool_, (SLOTS,)),
    "features": (np.float32, (SLOTS, 5)),
    "positions": (np.float32, (SLOTS, SLICES, 2)),
}
POLICY_KEYS = {
    "kind",
    "renderer",
    "source_sha256",
    "node_text_sha256",
    "pixel_group_sha256",
    "adapter_sha256",
}


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def roster_sha256(uids) -> str:
    return canonical_sha256(sorted(str(uid) for uid in uids))


def _is_sha(value: object) -> bool:
    return (
        isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)
    )


def validate_public_input(policy: object) -> None:
    if (
        not isinstance(policy, dict)
        or set(policy) != POLICY_KEYS
        or policy["kind"] != KIND
        or policy["renderer"] != RENDERER
        or policy["source_sha256"] != SOURCE_SHA256
        or policy["node_text_sha256"] != NODE_TEXT_SHA256
        or policy["pixel_group_sha256"] != PIXEL_GROUP_SHA256
        or not _is_sha(policy["adapter_sha256"])
    ):
        raise ValueError("Exact source-bound public ordered-triplet input policy required")


def _validate_pixel_group(group: object) -> None:
    if (
        not isinstance(group, dict)
        or {"img", "group", "slices", "crop_mm", "band", "rules", "slots"} != set(group)
        or group["img"] != SIDE
        or group["group"] != 3
        or group["slices"] != SLICES
        or group["crop_mm"] != 130.0
        or group["band"] != [0.2, 0.8]
        or not isinstance(group["rules"], dict)
        or not isinstance(group["slots"], list)
        or len(group["slots"]) != SLOTS
        or not all(isinstance(name, str) and name for name in group["slots"])
        or canonical_sha256(group) != PIXEL_GROUP_SHA256
    ):
        raise ValueError("Admitted public pixel group required")


def load_adapter(policy: dict) -> types.ModuleType:
    """Execute exactly the hashed standalone adapter bytes, never an importable copy."""
    validate_public_input(policy)
    source = ADAPTER_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != policy["adapter_sha256"]:
        raise ValueError("Public triplet adapter source differs from its bound hash")
    module = types.ModuleType("rsna_public_triplet_dino_adapter")
    module.__file__ = str(ADAPTER_PATH)
    exec(compile(source, str(ADAPTER_PATH), "exec"), module.__dict__)
    return module


def load_public_index(cache: Path, policy: dict, source_revision: str) -> tuple[dict, str]:
    validate_public_input(policy)
    payload = (Path(cache) / CACHE_DIRECTORY / "index.json").read_bytes()
    index = json.loads(payload)
    if not isinstance(index, dict) or set(index) != {
        "schema_version",
        "kind",
        "renderer",
        "source_sha256",
        "node_text_sha256",
        "pixel_group",
        "source_revision",
        "fitting_uids_sha256",
        "evaluation_uids_sha256",
        "cohort_sha256",
        "members",
    }:
        raise ValueError("Public triplet cache index fields differ")
    if index["schema_version"] != 1 or any(
        index[key] != policy[key]
        for key in ("kind", "renderer", "source_sha256", "node_text_sha256")
    ):
        raise ValueError("Public triplet cache renderer/source identity differs")
    _validate_pixel_group(index["pixel_group"])
    if canonical_sha256(index["pixel_group"]) != policy["pixel_group_sha256"]:
        raise ValueError("Public triplet cache pixel group differs from the bound policy")
    if index["source_revision"] != source_revision:
        raise ValueError("Public triplet cache source revision differs from the release")
    members = index["members"]
    if not isinstance(members, dict) or not members:
        raise ValueError("Public triplet cache has no members")
    for uid, member in members.items():
        if (
            not isinstance(uid, str)
            or not uid
            or not isinstance(member, dict)
            or set(member) != {"path", "bytes", "sha256"}
            or member["path"] != f"members/{uid}.npz"
            or PurePosixPath(member["path"]).name != f"{uid}.npz"
            or type(member["bytes"]) is not int
            or member["bytes"] < 1
            or not _is_sha(member["sha256"])
        ):
            raise ValueError("Invalid public triplet cache member binding")
    if index["cohort_sha256"] != roster_sha256(members):
        raise ValueError("Public triplet cache cohort differs from its members")
    return index, hashlib.sha256(payload).hexdigest()


def verify_public_cohort(
    cache: Path, policy: dict, train: pd.DataFrame, outer: pd.DataFrame, source_revision: str
) -> dict:
    """Bind the complete fitting/evaluation rosters before any diagnostic subsetting."""
    index, index_sha = load_public_index(cache, policy, source_revision)
    fitting, evaluation = set(train.study_uid), set(outer.study_uid)
    if (
        fitting & evaluation
        or index["fitting_uids_sha256"] != roster_sha256(fitting)
        or index["evaluation_uids_sha256"] != roster_sha256(evaluation)
        or set(index["members"]) != fitting | evaluation
    ):
        raise ValueError("Public triplet cache roster differs from the declared fold")
    return {
        "index_sha256": index_sha,
        "cohort_sha256": index["cohort_sha256"],
        "fitting_uids_sha256": index["fitting_uids_sha256"],
        "evaluation_uids_sha256": index["evaluation_uids_sha256"],
        "members": len(index["members"]),
        "adapter_sha256": policy["adapter_sha256"],
        "pixel_group_sha256": policy["pixel_group_sha256"],
    }


def _safe_uid(uid: object) -> bool:
    return (
        isinstance(uid, str)
        and bool(uid)
        and uid not in {".", ".."}
        and not any(char in uid for char in ("/", "\\", "\x00"))
    )


def _validate_row(arrays: object) -> None:
    """The adapter's source contract, checked before any member byte is written."""
    if (
        not isinstance(arrays, dict)
        or set(arrays) != set(MEMBER_ARRAYS)
        or any(
            not isinstance(arrays[name], np.ndarray)
            or arrays[name].dtype != dtype
            or arrays[name].shape != shape
            for name, (dtype, shape) in MEMBER_ARRAYS.items()
        )
    ):
        raise ValueError("Public triplet row arrays differ from the source contract")
    mask, features, positions = arrays["slot_mask"], arrays["features"], arrays["positions"]
    validity = positions[..., 1]
    if (
        not mask.any()
        or not np.isfinite(features).all()
        or not np.isfinite(positions).all()
        or not ((validity == 0) | (validity == 1)).all()
        or np.count_nonzero(positions[..., 0][validity == 0])
        or np.count_nonzero(arrays["images"][~mask])
        or np.count_nonzero(features[~mask])
        or np.count_nonzero(positions[~mask])
    ):
        raise ValueError("Public triplet row violates the source mask/geometry contract")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _durable_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(path.name + ".partial")
    with open(temporary, "wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _resume_members(partial: Path, request: bytes, order: list[str]) -> dict[str, dict]:
    """Re-verify every committed member; discard bytes never committed to the journal."""
    if (partial / PRODUCER_REQUEST).read_bytes() != request:
        raise ValueError("Public triplet producer request differs from the interrupted run")
    raw = (partial / PRODUCER_JOURNAL).read_bytes()
    lines = raw.split(b"\n")
    committed, torn = lines[:-1], lines[-1]
    members: dict[str, dict] = {}
    for uid, line in zip(order, committed, strict=False):
        entry = json.loads(line)
        if (
            not isinstance(entry, dict)
            or entry.get("uid") != uid
            or set(entry) != {"uid", "path", "bytes", "sha256"}
            or entry["path"] != f"members/{uid}.npz"
        ):
            raise ValueError("Public triplet producer journal differs from the roster order")
        payload = (partial / CACHE_DIRECTORY / entry["path"]).read_bytes()
        if len(payload) != entry["bytes"] or hashlib.sha256(payload).hexdigest() != entry["sha256"]:
            raise ValueError("Committed public triplet member bytes differ from the journal")
        members[uid] = {key: entry[key] for key in ("path", "bytes", "sha256")}
    if len(committed) > len(order):
        raise ValueError("Public triplet producer journal exceeds the declared roster")
    if torn:
        (partial / PRODUCER_JOURNAL).write_bytes(raw[: len(raw) - len(torn)])
    root = partial / CACHE_DIRECTORY / "members"
    for path in root.iterdir():
        if path.name not in {PurePosixPath(m["path"]).name for m in members.values()}:
            path.unlink()
    return members


def stream_public_triplet_cache(
    cache: Path,
    render: Callable[[str], dict[str, np.ndarray]],
    *,
    pixel_group: dict,
    source_revision: str,
    fitting_uids,
    evaluation_uids,
    resume: bool = False,
) -> str:
    """Render, verify and commit one original row at a time; returns the index SHA256.

    Members are written durably in sorted roster order and journaled under a private
    partial directory; ``resume=True`` re-verifies every committed member against the
    identical request and continues. The complete tree is published by one rename, so
    a readable cache index never describes a partial cohort.
    """
    _validate_pixel_group(pixel_group)
    fitting_uids, evaluation_uids = list(fitting_uids), list(evaluation_uids)
    if not all(_safe_uid(uid) for uid in [*fitting_uids, *evaluation_uids]):
        raise ValueError("Public triplet study IDs must be safe individual filenames")
    if (
        len(set(fitting_uids)) != len(fitting_uids)
        or len(set(evaluation_uids)) != len(evaluation_uids)
        or set(fitting_uids) & set(evaluation_uids)
        or not fitting_uids
    ):
        raise ValueError("Public triplet rows must cover disjoint unique rosters exactly")
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("Public triplet cache requires the release source revision")
    cache = Path(cache)
    final, partial = cache / CACHE_DIRECTORY, cache / f".{CACHE_DIRECTORY}.partial"
    if final.exists():
        raise ValueError("Public triplet cache already exists; it is never overwritten")
    order = sorted([*fitting_uids, *evaluation_uids])
    header = {
        "schema_version": 1,
        "kind": KIND,
        "renderer": RENDERER,
        "source_sha256": SOURCE_SHA256,
        "node_text_sha256": NODE_TEXT_SHA256,
        "pixel_group": pixel_group,
        "source_revision": source_revision,
        "fitting_uids_sha256": roster_sha256(fitting_uids),
        "evaluation_uids_sha256": roster_sha256(evaluation_uids),
        "cohort_sha256": roster_sha256(order),
    }
    request = json.dumps(header, sort_keys=True, indent=1).encode()
    if resume:
        if not partial.is_dir():
            raise ValueError("No interrupted public triplet producer run to resume")
        members = _resume_members(partial, request, order)
    else:
        if partial.exists():
            raise ValueError("Interrupted public triplet producer run exists; resume it")
        cache.mkdir(parents=True, exist_ok=True)
        (partial / CACHE_DIRECTORY / "members").mkdir(parents=True)
        _durable_write(partial / PRODUCER_REQUEST, request)
        _durable_write(partial / PRODUCER_JOURNAL, b"")
        _fsync_directory(partial)
        members = {}
    root = partial / CACHE_DIRECTORY
    with open(partial / PRODUCER_JOURNAL, "ab") as journal:
        for uid in order[len(members) :]:
            arrays = render(uid)
            _validate_row(arrays)
            buffer = io.BytesIO()
            np.savez(
                buffer,
                images=arrays["images"],
                slot_mask=arrays["slot_mask"],
                features=arrays["features"],
                positions=arrays["positions"],
            )
            payload = buffer.getvalue()
            _durable_write(root / "members" / f"{uid}.npz", payload)
            members[uid] = {
                "path": f"members/{uid}.npz",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
            journal.write(json.dumps({"uid": uid, **members[uid]}, sort_keys=True).encode())
            journal.write(b"\n")
            journal.flush()
            os.fsync(journal.fileno())
    index = {**header, "members": members}
    payload = json.dumps(index, sort_keys=True, indent=1).encode()
    _durable_write(root / "index.json", payload)
    _fsync_directory(root / "members")
    _fsync_directory(root)
    if final.exists():
        raise ValueError("Public triplet cache appeared during production; refusing overwrite")
    os.rename(root, final)
    _fsync_directory(cache)
    for name in (PRODUCER_REQUEST, PRODUCER_JOURNAL):
        (partial / name).unlink()
    partial.rmdir()
    _fsync_directory(cache)
    return hashlib.sha256(payload).hexdigest()


def write_public_triplet_cache(
    cache: Path,
    rows: dict[str, dict[str, np.ndarray]],
    *,
    pixel_group: dict,
    source_revision: str,
    fitting_uids,
    evaluation_uids,
) -> str:
    """In-memory rows through the same streaming producer; returns the index SHA256."""
    _validate_pixel_group(pixel_group)
    fitting_uids, evaluation_uids = list(fitting_uids), list(evaluation_uids)
    if not all(_safe_uid(uid) for uid in [*rows, *fitting_uids, *evaluation_uids]):
        raise ValueError("Public triplet study IDs must be safe individual filenames")
    if set(rows) != set(fitting_uids) | set(evaluation_uids):
        raise ValueError("Public triplet rows must cover disjoint unique rosters exactly")
    return stream_public_triplet_cache(
        cache,
        rows.__getitem__,
        pixel_group=pixel_group,
        source_revision=source_revision,
        fitting_uids=fitting_uids,
        evaluation_uids=evaluation_uids,
    )


class PublicTripletStudyDataset(torch.utils.data.Dataset):
    """Fixed source rows; the frame supplies every label and mask unchanged."""

    cache_side = SIDE

    def __init__(
        self,
        studies: pd.DataFrame,
        series: pd.DataFrame,
        cache: Path,
        augment: bool = False,
        *,
        public_input: dict,
        source_revision: str,
        slices: int,
        side: int,
        max_series: int,
        slice_sampling: str = "uniform",
        series_sampling: str = "selected",
        soft_targets: bool = False,
    ):
        if (
            augment is not False
            or (slices, side, max_series) != (VIEWS, SIDE, SLOTS)
            or (slice_sampling, series_sampling) != ("uniform", "selected")
            or type(soft_targets) is not bool
        ):
            raise ValueError("Public triplets require fixed 6-slot 10-view 336 inputs")
        if studies.study_uid.duplicated().any():
            raise ValueError("Study sampling requires unique study IDs")
        self.index, self.index_sha256 = load_public_index(cache, public_input, source_revision)
        self.adapter = load_adapter(public_input)
        self.studies = studies.reset_index(drop=True)
        if not set(self.studies.study_uid) <= set(self.index["members"]):
            raise ValueError("Requested studies are absent from the public triplet cohort")
        self.root = Path(cache) / CACHE_DIRECTORY
        self.source_revision = source_revision
        self.soft_targets = soft_targets

    def __len__(self) -> int:
        return len(self.studies)

    def _member(self, uid: str) -> dict[str, torch.Tensor]:
        member = self.index["members"][uid]
        payload = (self.root / member["path"]).read_bytes()
        if (
            len(payload) != member["bytes"]
            or hashlib.sha256(payload).hexdigest() != member["sha256"]
        ):
            raise ValueError("Public triplet member bytes differ from the cache index")
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != set(MEMBER_ARRAYS):
                raise ValueError("Public triplet member arrays differ from the source contract")
            arrays = {name: archive[name] for name in MEMBER_ARRAYS}
        for name, (dtype, shape) in MEMBER_ARRAYS.items():
            if arrays[name].dtype != dtype or arrays[name].shape != shape:
                raise ValueError("Public triplet member dtype/shape differs")
        return {name: torch.from_numpy(value) for name, value in arrays.items()}

    def __getitem__(self, index: int) -> dict:
        row = self.studies.iloc[index]
        uid = row.study_uid
        labels = torch.tensor(
            [float(row[t]) if t in row else 0 for t in TARGETS], dtype=torch.float32
        )
        label_mask = torch.tensor(
            [float(row[f"{t}__mask"]) if f"{t}__mask" in row else 0 for t in TARGETS],
            dtype=torch.float32,
        )
        observed = labels[label_mask == 1]
        if not self.soft_targets and not ((observed == 0) | (observed == 1)).all():
            raise ValueError("Supervised targets must be finite binary values")
        member = self._member(uid)
        return self.adapter.study_inputs(
            member["images"],
            member["slot_mask"],
            member["features"],
            member["positions"],
            labels,
            label_mask,
            study_uid=uid,
        )
