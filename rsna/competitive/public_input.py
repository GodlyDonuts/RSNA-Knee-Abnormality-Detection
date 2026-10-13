"""Source-bound public ordered-triplet input policy; research comparison only.

Pixels come from the reviewed Pilkwang v15 renderer (pilkwang/rsna-knee-baseline-v1,
Kaggle notebook version 15, by Pilkwang), stored once as uint8 [6,12,336,336] rows.
The exact standalone ``dino_adapter.study_inputs`` bytes form the ten overlapping
three-slice views /255; the encoder alone applies ImageNet normalization. Labels
and masks always come from the unchanged fitting/evaluation frames. No native
cache, float cache, label, crop or geometry is read or reconstructed here.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import io
import json
import os
import re
import shutil
import time
import types
from collections.abc import Callable, Iterable, Mapping
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
CACHE_DIRECTORY = "public-ordered-triplet-v1"
STAGING_DIRECTORY = CACHE_DIRECTORY + ".partial"
SIDE, SLOTS, SLICES, VIEWS = 336, 6, 12, 10
# The private member's exact manifest pixel group (actual-input-comparison receipt of
# m_e5427d6c21.pt): original SLOTS_RECOVERED names and RULES_NATIVE, nothing else.
ADMITTED_PIXEL_GROUP: dict = {
    "band": [0.2, 0.8],
    "crop_mm": 130.0,
    "group": 3,
    "img": 336,
    "rules": {
        "decode_fill": "nearest",
        "lat": "centre",
        "order": "normal",
        "slot_fallback": False,
    },
    "slices": 12,
    "slots": [
        "SAG_FLUID_FS",
        "COR_FLUID_FS",
        "AX_FLUID_FS",
        "SAG_FLUID_NOFS",
        "COR_T1",
        "SAG_T1",
    ],
}
UID_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]{0,127}")
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


ADMITTED_PIXEL_GROUP_SHA256 = canonical_sha256(ADMITTED_PIXEL_GROUP)


def roster_sha256(uids) -> str:
    return canonical_sha256(sorted(str(uid) for uid in uids))


def validate_uid(uid: object) -> str:
    """A study UID is a single safe file-name component: no separator, traversal or root."""
    if (
        not isinstance(uid, str)
        or not UID_PATTERN.fullmatch(uid)
        or ".." in uid
        or PurePosixPath(uid).name != uid
    ):
        raise ValueError("Public triplet study UIDs must be safe single path components")
    return uid


def validate_rosters(fitting_uids, evaluation_uids) -> tuple[frozenset, frozenset]:
    """Validated, duplicate-free, nonempty and disjoint fitting/evaluation rosters."""
    rosters = []
    for uids in (fitting_uids, evaluation_uids):
        if isinstance(uids, (str, bytes)):
            raise ValueError("Public triplet rosters must be collections of study UIDs")
        values = [validate_uid(uid) for uid in uids]
        if not values or len(set(values)) != len(values):
            raise ValueError("Public triplet rosters must be nonempty without duplicates")
        rosters.append(frozenset(values))
    if rosters[0] & rosters[1]:
        raise ValueError("Public triplet fitting and evaluation rosters overlap")
    return rosters[0], rosters[1]


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
        or policy["pixel_group_sha256"] != ADMITTED_PIXEL_GROUP_SHA256
        or not _is_sha(policy["adapter_sha256"])
    ):
        raise ValueError("Exact source-bound public ordered-triplet input policy required")


def _validate_pixel_group(group: object) -> None:
    """Exactly the admitted slot list, renderer rules and recipe, with JSON-exact types."""
    if (
        not isinstance(group, dict)
        or set(group) != set(ADMITTED_PIXEL_GROUP)
        or canonical_sha256(group) != ADMITTED_PIXEL_GROUP_SHA256
    ):
        raise ValueError("Admitted public pixel group required")


def _validate_row(arrays: object) -> None:
    """The adapter's source-row contract, checked before any member byte is written."""
    if (
        not isinstance(arrays, Mapping)
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
        validate_uid(uid)
        if (
            not isinstance(member, dict)
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


def write_public_triplet_cache(
    cache: Path,
    rows: Mapping[str, Mapping[str, np.ndarray]] | Callable[[str], Mapping[str, np.ndarray]],
    *,
    pixel_group: dict,
    source_revision: str,
    fitting_uids,
    evaluation_uids,
    max_output_bytes: int | None = None,
) -> str:
    """Store original renderer rows once as uint8; returns the index SHA256.

    UIDs, rosters, pixel group and destination are validated before any directory or
    member is created. Members are written to a sibling staging directory that is only
    renamed into place after every row and the index are complete; any refusal removes it.
    ``rows`` is either a complete mapping or a per-UID callable that renders on demand.
    """
    _validate_pixel_group(pixel_group)
    fitting, evaluation = validate_rosters(fitting_uids, evaluation_uids)
    cohort = sorted(fitting | evaluation)
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("Public triplet cache requires the release source revision")
    render: Callable[[str], Mapping[str, np.ndarray]]
    if isinstance(rows, Mapping):
        if set(rows) != set(cohort):
            raise ValueError("Public triplet rows must cover the declared roster exactly")
        for uid in cohort:
            _validate_row(rows[uid])
        render = rows.__getitem__
    elif callable(rows):
        render = rows
    else:
        raise ValueError("Public triplet rows must be a mapping or a per-study renderer")
    if max_output_bytes is not None and (type(max_output_bytes) is not int or max_output_bytes < 1):
        raise ValueError("Public triplet output byte cap must be a positive integer")
    root = Path(cache) / CACHE_DIRECTORY
    staging = Path(cache) / STAGING_DIRECTORY
    if root.exists() or root.is_symlink() or staging.exists() or staging.is_symlink():
        raise FileExistsError("Public triplet cache or its staging directory already exists")
    staging.mkdir(parents=True)
    try:
        (staging / "members").mkdir()
        members, written = {}, 0
        for uid in cohort:
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
            written += len(payload)
            if max_output_bytes is not None and written > max_output_bytes:
                raise ValueError("Public triplet cache exceeds its admitted output byte cap")
            path = staging / "members" / f"{uid}.npz"
            if path.parent != staging / "members":
                raise ValueError("Public triplet member path escapes the cache")
            path.write_bytes(payload)
            members[uid] = {
                "path": f"members/{uid}.npz",
                "bytes": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        index = {
            "schema_version": 1,
            "kind": KIND,
            "renderer": RENDERER,
            "source_sha256": SOURCE_SHA256,
            "node_text_sha256": NODE_TEXT_SHA256,
            "pixel_group": pixel_group,
            "source_revision": source_revision,
            "fitting_uids_sha256": roster_sha256(fitting),
            "evaluation_uids_sha256": roster_sha256(evaluation),
            "cohort_sha256": roster_sha256(members),
            "members": members,
        }
        payload = json.dumps(index, sort_keys=True, indent=1).encode()
        (staging / "index.json").write_bytes(payload)
        os.rename(staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return hashlib.sha256(payload).hexdigest()


# Upper bound for one uncompressed npz member: the four arrays plus generous zip headers.
MEMBER_BYTES_BOUND = sum(
    int(np.prod(shape)) * np.dtype(dtype).itemsize for dtype, shape in MEMBER_ARRAYS.values()
) + (1 << 16)
ADAPTER_PACKAGE = "tools.research.public_dinov2_input_reference"


def _configured_renderer(namespace: dict | None) -> dict:
    """The reviewed v15 nodes configured with exactly the admitted member pixel group."""
    provenance = importlib.import_module(ADAPTER_PACKAGE + ".provenance")
    reference = importlib.import_module(ADAPTER_PACKAGE + ".reference")
    if namespace is None:
        raw = reference.SOURCE_PATH.read_bytes()
        if hashlib.sha256(raw).hexdigest() != SOURCE_SHA256:
            raise ValueError("Public renderer source bytes differ from the bound source")
        namespace, receipt = reference.load(raw)
        if (
            receipt["source_sha256"] != SOURCE_SHA256
            or receipt["node_text_sha256"] != NODE_TEXT_SHA256
            or provenance.SOURCE_SHA256 != SOURCE_SHA256
        ):
            raise ValueError("Public renderer receipt differs from the bound source")
    reference.configure(namespace, ADMITTED_PIXEL_GROUP)
    rules, slots = ADMITTED_PIXEL_GROUP["rules"], ADMITTED_PIXEL_GROUP["slots"]
    if (
        namespace["RULES"] != rules
        or [slot[0] for slot in namespace["SLOTS"]] != slots
        or namespace["SLOTS"] is not namespace["SLOTS_RECOVERED"]
    ):
        raise ValueError("Configured renderer differs from the admitted rules/slots")
    return namespace


def produce_public_triplet_cache(
    cache: Path,
    *,
    policy: dict,
    study_root: Path,
    plane_map: Mapping[str, str],
    fitting_uids: Iterable[str],
    evaluation_uids: Iterable[str],
    source_revision: str,
    max_members: int,
    max_output_bytes: int,
    reserve_free_bytes: int = 0,
    namespace: dict | None = None,
) -> dict:
    """Bounded source-bound producer: original renderer -> hashed adapter -> uint8 cache.

    Every study is rendered by the reviewed v15 nodes under the admitted pixel group and
    its geometry/features come from the exact hashed ``dino_adapter.render_inputs`` bytes;
    the row passed to its ``study_inputs`` is stored verbatim. Labels are never read. The
    full declared rosters, member/byte caps and free space are checked before any render
    or write, and any refused study aborts the whole cache (no partial cohort).
    """
    validate_public_input(policy)
    fitting, evaluation = validate_rosters(fitting_uids, evaluation_uids)
    cohort = fitting | evaluation
    if type(max_members) is not int or not 0 < len(cohort) <= max_members:
        raise ValueError("Public triplet cohort exceeds its admitted member cap")
    if type(max_output_bytes) is not int or len(cohort) * MEMBER_BYTES_BOUND > max_output_bytes:
        raise ValueError("Public triplet cohort could exceed its admitted output byte cap")
    if type(reserve_free_bytes) is not int or reserve_free_bytes < 0:
        raise ValueError("Free-space reserve must be a nonnegative integer")
    study_root = Path(study_root)
    if not study_root.is_dir():
        raise ValueError("Original study root is unavailable")
    missing = [uid for uid in sorted(cohort) if not (study_root / uid).is_dir()]
    if missing:
        raise ValueError(f"{len(missing)} declared studies have no original source directory")
    probe = Path(cache)
    while not probe.exists():
        probe = probe.parent
    if shutil.disk_usage(probe).free < max_output_bytes + reserve_free_bytes:
        raise ValueError("Insufficient free space for the admitted cache byte cap and reserve")
    namespace = _configured_renderer(namespace)
    adapter = load_adapter(policy)
    adapter.__package__ = ADAPTER_PACKAGE
    original_study_inputs = getattr(adapter, "study_inputs")
    captured: list[tuple] = []

    def capture(images, series_mask, features, slice_positions, labels, label_mask, **kw):
        captured.append((images, series_mask, features, slice_positions))
        return original_study_inputs(
            images, series_mask, features, slice_positions, labels, label_mask, **kw
        )

    setattr(adapter, "study_inputs", capture)
    unread_labels, unread_mask = torch.zeros(len(TARGETS)), torch.ones(len(TARGETS))

    def render(uid: str) -> dict[str, np.ndarray]:
        captured.clear()
        item = adapter.render_inputs(
            namespace, study_root / uid, dict(plane_map), unread_labels, unread_mask
        )
        if len(captured) != 1 or item["study_uid"] != uid:
            raise ValueError("Public adapter did not render exactly the requested study")
        images, mask, features, positions = captured[0]
        return {
            "images": images.numpy().copy(),
            "slot_mask": mask.numpy().copy(),
            "features": features.numpy().copy(),
            "positions": positions.numpy().copy(),
        }

    started = time.monotonic()
    index_sha256 = write_public_triplet_cache(
        cache,
        render,
        pixel_group=copy.deepcopy(ADMITTED_PIXEL_GROUP),
        source_revision=source_revision,
        fitting_uids=fitting,
        evaluation_uids=evaluation,
        max_output_bytes=max_output_bytes,
    )
    index, _ = load_public_index(cache, policy, source_revision)
    return {
        "index_sha256": index_sha256,
        "members": len(index["members"]),
        "member_bytes": sum(member["bytes"] for member in index["members"].values()),
        "fitting_uids_sha256": index["fitting_uids_sha256"],
        "evaluation_uids_sha256": index["evaluation_uids_sha256"],
        "cohort_sha256": index["cohort_sha256"],
        "pixel_group_sha256": ADMITTED_PIXEL_GROUP_SHA256,
        "adapter_sha256": policy["adapter_sha256"],
        "wall_seconds": round(time.monotonic() - started, 3),
        "labels_read": False,
    }


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
