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
import re
import types
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

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
SIDE, SLOTS, SLICES, VIEWS = 336, 6, 12, 10
MEMBER_ARRAYS = {
    "images": (np.uint8, (SLOTS, SLICES, SIDE, SIDE)),
    "slot_mask": (np.bool_, (SLOTS,)),
    "features": (np.float32, (SLOTS, 5)),
    "positions": (np.float32, (SLOTS, SLICES, 2)),
}
# The private Pilkwang member's exact manifest pixel group, as bound by CPU 905973.
ADMITTED_PIXEL_GROUP: dict[str, Any] = {
    "band": [0.2, 0.8],
    "crop_mm": 130.0,
    "group": 3,
    "img": 336,
    "rules": {"decode_fill": "nearest", "lat": "centre", "order": "normal", "slot_fallback": False},
    "slices": 12,
    "slots": ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"],
}
ADMITTED_PIXEL_GROUP_SHA256 = "bbd7564015afc8a823fd122cd7150f5cf9025f3cdadf3396f0170406855eaa59"
PIXEL_GROUP_SHA256 = ADMITTED_PIXEL_GROUP_SHA256
UID_PATTERN = re.compile(r"[0-9A-Za-z][0-9A-Za-z._-]{0,127}")
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
        or policy["pixel_group_sha256"] != ADMITTED_PIXEL_GROUP_SHA256
        or not _is_sha(policy["adapter_sha256"])
    ):
        raise ValueError("Exact source-bound public ordered-triplet input policy required")


def _validate_pixel_group(group: object) -> None:
    if not isinstance(group, dict) or canonical_sha256(group) != ADMITTED_PIXEL_GROUP_SHA256:
        raise ValueError("Exact admitted public pixel group, slots and rules required")


def _validate_uid(uid: object) -> str:
    if not isinstance(uid, str) or not UID_PATTERN.fullmatch(uid) or ".." in uid:
        raise ValueError("Public triplet study IDs must be safe individual filenames")
    return uid


def _declared_rosters(fitting_uids, evaluation_uids) -> tuple[set[str], set[str]]:
    fitting_list = [_validate_uid(uid) for uid in fitting_uids]
    evaluation_list = [_validate_uid(uid) for uid in evaluation_uids]
    fitting, evaluation = set(fitting_list), set(evaluation_list)
    if (
        not fitting
        or len(fitting) != len(fitting_list)
        or len(evaluation) != len(evaluation_list)
        or fitting & evaluation
    ):
        raise ValueError("Public triplet rosters must be disjoint unique and nonempty")
    return fitting, evaluation


def load_adapter(policy: dict) -> types.ModuleType:
    """Execute exactly the hashed standalone adapter bytes, never an importable copy."""
    validate_public_input(policy)
    source = ADAPTER_PATH.read_bytes()
    if hashlib.sha256(source).hexdigest() != policy["adapter_sha256"]:
        raise ValueError("Public triplet adapter source differs from its bound hash")
    module = types.ModuleType("rsna_public_triplet_dino_adapter")
    module.__file__ = str(ADAPTER_PATH)
    # `render_inputs` imports the reviewed renderer relative to its own package.
    module.__package__ = "tools.research.public_dinov2_input_reference"
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
            or not UID_PATTERN.fullmatch(uid)
            or ".." in uid
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


def write_public_triplet_cache(
    cache: Path,
    rows: Mapping[str, dict[str, np.ndarray]] | Iterable[tuple[str, dict[str, np.ndarray]]],
    *,
    public_input: dict,
    source_revision: str,
    fitting_uids,
    evaluation_uids,
) -> str:
    """Store original renderer rows once as uint8; returns the index SHA256.

    Every UID and both disjoint rosters are validated before anything is created. Rows
    stream into a sibling `.partial` directory that is renamed only once complete.
    """
    fitting, evaluation = _declared_rosters(fitting_uids, evaluation_uids)
    if not isinstance(source_revision, str) or not source_revision:
        raise ValueError("Public triplet cache requires its release source revision")
    adapter = load_adapter(public_input)
    root = Path(cache) / CACHE_DIRECTORY
    partial = Path(cache) / f"{CACHE_DIRECTORY}.partial"
    if root.exists() or partial.exists():
        raise FileExistsError("Public triplet cache output already exists")
    declared = fitting | evaluation
    items = rows.items() if isinstance(rows, Mapping) else rows
    (partial / "members").mkdir(parents=True)
    members = {}
    for uid, arrays in items:
        if uid not in declared or uid in members:
            raise ValueError("Public triplet row is outside the declared roster or repeated")
        if set(arrays) != set(MEMBER_ARRAYS) or any(
            arrays[name].dtype != dtype or arrays[name].shape != shape
            for name, (dtype, shape) in MEMBER_ARRAYS.items()
        ):
            raise ValueError("Public triplet row arrays differ from the source contract")
        adapter.study_inputs(
            *(torch.from_numpy(arrays[name]) for name in MEMBER_ARRAYS),
            torch.zeros(len(TARGETS)),
            torch.zeros(len(TARGETS)),
            study_uid=uid,
        )
        buffer = io.BytesIO()
        np.savez(
            buffer,
            images=arrays["images"],
            slot_mask=arrays["slot_mask"],
            features=arrays["features"],
            positions=arrays["positions"],
        )
        payload = buffer.getvalue()
        with open(partial / "members" / f"{uid}.npz", "xb") as handle:
            handle.write(payload)
        members[uid] = {
            "path": f"members/{uid}.npz",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
    if set(members) != declared:
        raise ValueError("Public triplet rows do not cover the declared roster")
    index = {
        "schema_version": 1,
        "kind": KIND,
        "renderer": RENDERER,
        "source_sha256": SOURCE_SHA256,
        "node_text_sha256": NODE_TEXT_SHA256,
        "pixel_group": ADMITTED_PIXEL_GROUP,
        "source_revision": source_revision,
        "fitting_uids_sha256": roster_sha256(fitting),
        "evaluation_uids_sha256": roster_sha256(evaluation),
        "cohort_sha256": roster_sha256(members),
        "members": dict(sorted(members.items())),
    }
    payload = json.dumps(index, sort_keys=True, indent=1).encode()
    (partial / "index.json").write_bytes(payload)
    partial.rename(root)
    return hashlib.sha256(payload).hexdigest()


def _validate_namespace(namespace: dict) -> None:
    group = ADMITTED_PIXEL_GROUP
    if (
        namespace.get("IMG") != group["img"]
        or namespace.get("CACHE_SLICES") != group["slices"]
        or namespace.get("GROUP") != group["group"]
        or namespace.get("CROP_MM") != group["crop_mm"]
        or tuple(namespace.get("SLICE_BAND", ())) != tuple(group["band"])
        or namespace.get("RULES") != group["rules"]
        or namespace.get("N_SLOT") != len(group["slots"])
        or [slot[0] for slot in namespace.get("SLOTS", ())] != group["slots"]
    ):
        raise ValueError("Renderer namespace is not configured with the admitted pixel group")


def produce_public_triplet_cache(
    cache: Path,
    *,
    public_input: dict,
    namespace: dict,
    study_root: Path,
    plane_map: dict,
    source_revision: str,
    fitting_uids,
    evaluation_uids,
    max_studies: int,
) -> str:
    """Bounded producer: render each already-present original study once and store it.

    Pixels, slot mask, features and positions are exactly the arguments the hashed
    `render_inputs` passes to `study_inputs`; nothing is recropped, acquired or labeled.
    """
    fitting, evaluation = _declared_rosters(fitting_uids, evaluation_uids)
    declared = sorted(fitting | evaluation)
    if type(max_studies) is not int or not 1 <= len(declared) <= max_studies:
        raise ValueError("Public triplet production exceeds its explicit study bound")
    _validate_namespace(namespace)
    directories = {uid: Path(study_root) / uid for uid in declared}
    if not all(path.is_dir() for path in directories.values()):
        raise FileNotFoundError("Original study directories are absent; no acquisition here")
    adapter = load_adapter(public_input)
    original = getattr(adapter, "study_inputs")
    captured: dict[str, dict[str, np.ndarray]] = {}

    def capture(images, series_mask, features, slice_positions, labels, label_mask, *, study_uid):
        captured[study_uid] = {
            "images": images.numpy().copy(),
            "slot_mask": series_mask.numpy().copy(),
            "features": features.numpy().copy(),
            "positions": slice_positions.numpy().copy(),
        }
        return original(
            images, series_mask, features, slice_positions, labels, label_mask, study_uid=study_uid
        )

    setattr(adapter, "study_inputs", capture)

    def rows():
        for uid, directory in directories.items():
            adapter.render_inputs(
                namespace,
                directory,
                plane_map,
                torch.zeros(len(TARGETS)),
                torch.zeros(len(TARGETS)),
            )
            if set(captured) != {uid}:
                raise ValueError("Original renderer produced a different study")
            yield uid, captured.pop(uid)

    return write_public_triplet_cache(
        cache,
        rows(),
        public_input=public_input,
        source_revision=source_revision,
        fitting_uids=fitting,
        evaluation_uids=evaluation,
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
