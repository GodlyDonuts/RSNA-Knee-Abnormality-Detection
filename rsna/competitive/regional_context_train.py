"""Same-graph spatial-input contrast using the unchanged FP32 context trainer.

Global, regional and parent readouts must come from one admitted CNN forward.
No CNN fitting, clinical evaluation or old/new embedding substitution occurs.
"""

from __future__ import annotations

import hashlib
import io
import tarfile
from pathlib import Path

import numpy as np
import torch

from rsna.competitive.released_context import window_metadata
from rsna.competitive.released_context_train import PARENT_SHA

ENGINE = "licensed_coherent_regional_input_context_v1"
REPRESENTATION = "same_forward_global_regional_v1"
ROLES = ("global_repeat", "regional")
PARENT_SOURCE = "493c7b0d44fe5244b02024448a93f5393cdb073781a068951badb3ffd53fcbde"
GENERIC_TRAINER_SHA = "5addea8212501ba31021c739543a7fa627cf70066ad439fa0889ba5c5b3fa59f"
REGIONAL_HELPER_SHA = "d2c45353e56b8c7a86a0b83a15edb35395341867d72ec68b91d77e59b705be17"
POPULATIONS = {
    "engineering32": (32, "6ca592004ae972a93a8bb96dc7024b46c7ad86525388c02542dca844a6b1605f"),
    "full3887": (3887, "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"),
}


def feature_vector(global_features, regions, role):
    """Keep identical graph/parameter dimensions; only the input content differs."""
    if (
        role not in ROLES
        or global_features.ndim != 3
        or global_features.dtype != torch.float32
        or regions.dtype != torch.float16
        or regions.shape != (*global_features.shape[:2], 7, global_features.shape[-1])
        or not torch.isfinite(global_features).all()
        or not torch.isfinite(regions).all()
    ):
        raise ValueError("Coherent finite FP32 globals and seven FP16 regional summaries required")
    if role == "global_repeat":
        return global_features.repeat(1, 1, 8)
    return torch.cat((global_features, regions.float().flatten(-2)), dim=-1)


class RegionalFeatures:
    """Read immutable coherent NPZ records directly from original private tars.

    Caller verifies whole archives, source eligibility and label provenance before
    construction. Exact individual bytes are verified on first access; each later
    access requires the same device/inode/size/mtime/ctime signature. Production
    wrappers enforce1024channels/94windows/12targets; smaller dimensions support
    synthetic recovery tests, never competition acceptance.
    """

    def __init__(
        self, records, roots, labels, masks, binding, role, *, channels=1024, windows=94, targets=12
    ):
        if (
            role not in ROLES
            or not records
            or any(type(n) is not int or n < 1 for n in (channels, windows, targets))
            or not isinstance(binding, dict)
            or binding.get("representation") != REPRESENTATION
            or binding.get("parent_weights_sha256") != PARENT_SHA
            or binding.get("parent_source_sha256") != PARENT_SOURCE
        ):
            raise ValueError(
                "Admitted coherent representation, exact parent and explicit role required"
            )
        self.records, self.roots = records, {k: Path(p) for k, p in roots.items()}
        self.role, self.channels, self.windows, self.targets = role, channels, windows, targets
        self.labels = torch.as_tensor(labels, dtype=torch.float32).clone()
        raw_masks = torch.as_tensor(masks)
        if not ((raw_masks == 0) | (raw_masks == 1)).all():
            raise ValueError("Original binary source observation masks required")
        self.masks = raw_masks.bool().clone()
        if (
            self.labels.shape != (len(records), targets)
            or self.masks.shape != self.labels.shape
            or not torch.isfinite(self.labels[self.masks]).all()
            or ((self.labels[self.masks] < 0) | (self.labels[self.masks] > 1)).any()
            or len({r["study_uid"] for r in records}) != len(records)
        ):
            raise ValueError(
                "Distinct fitting records and aligned observed source probabilities required"
            )
        self.binding = dict(binding)
        for key, value in {
            "regional_engine": ENGINE,
            "regional_role": role,
            "window_feature_dimension": channels * 8,
        }.items():
            if key in binding and binding[key] != value:
                raise ValueError("Role or input dimension conflicts with admitted binding")
            self.binding[key] = value
        self.signatures = {}
        self.members: dict[Path, dict[str, tarfile.TarInfo]] = {}
        self.verified = set()
        for row in records:
            uid, name = row["study_uid"], row["file"]
            if (
                not isinstance(uid, str)
                or not uid
                or name != hashlib.sha256(uid.encode()).hexdigest() + ".npz"
                or row.get("archive") != "features.tar"
                or row.get("source_key") not in self.roots
                or type(row["bytes"]) is not int
                or not 0 < row["bytes"] < 4 * 2**20
                or not isinstance(row["sha256"], str)
                or len(row["sha256"]) != 64
                or any(c not in "0123456789abcdef" for c in row["sha256"])
            ):
                raise ValueError("Exact safe archived coherent member identity required")
            path = self.roots[row["source_key"]] / "features.tar"
            self._stat(path)
            if path not in self.members:
                with tarfile.open(path, "r:") as handle:
                    members = handle.getmembers()
                mapped = {m.name: m for m in members}
                if len(mapped) != len(members) or any(
                    not m.isfile()
                    or Path(m.name).name != m.name
                    or len(m.name) != 68
                    or not m.name.endswith(".npz")
                    or not 0 < m.size < 4 * 2**20
                    for m in members
                ):
                    raise ValueError("Unsafe or duplicate coherent archive member")
                self.members[path] = mapped
            member = self.members[path].get(name)
            if member is None or member.size != row["bytes"]:
                raise ValueError("Missing or wrongly sized coherent feature member")

    def _stat(self, path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Original regular coherent archive required")
        s = path.stat()
        signature = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if path in self.signatures and self.signatures[path] != signature:
            raise ValueError("Admitted coherent archive changed")
        self.signatures[path] = signature

    def __len__(self):
        return len(self.records)

    def __getitem__(self, position):
        row = self.records[position]
        path = self.roots[row["source_key"]] / "features.tar"
        self._stat(path)
        with path.open("rb") as handle:
            handle.seek(self.members[path][row["file"]].offset_data)
            raw = handle.read(row["bytes"])
        key = (path, row["file"])
        if key not in self.verified:
            if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Original coherent NPZ bytes differ from admission")
            self.verified.add(key)
        with np.load(io.BytesIO(raw), allow_pickle=False) as a:
            if set(a.files) != {
                "features",
                "regional",
                "parent_logits",
                "centers",
                "slots",
                "depth",
                "present",
            }:
                raise ValueError("Exact same-forward global/regional/parent schema required")
            f, r, p, c, slots, depth, present = (
                a[k]
                for k in (
                    "features",
                    "regional",
                    "parent_logits",
                    "centers",
                    "slots",
                    "depth",
                    "present",
                )
            )
            k = self.windows
            if (
                f.dtype != np.float32
                or f.shape != (2, k, self.channels)
                or r.dtype != np.float16
                or r.shape != (2, k, 7, self.channels)
                or p.dtype != np.float32
                or p.shape != (2, self.targets)
                or c.dtype != np.int64
                or c.shape != (k,)
                or slots.dtype != np.int64
                or slots.shape != (k,)
                or depth.dtype != np.float32
                or depth.shape != (2, k)
                or present.dtype != np.bool_
                or present.shape != (k,)
                or any(not np.isfinite(x).all() for x in (f, r, p, depth))
            ):
                raise ValueError(
                    "Original finite coherent storage dtype and complete dimensions required"
                )
            expected, plain = window_metadata(torch.from_numpy(c))
            _, mirror = window_metadata(torch.from_numpy(c), mirrored=True)
            if not np.array_equal(slots, expected.numpy()) or not np.array_equal(
                depth, torch.stack((plain, mirror)).numpy()
            ):
                raise ValueError(
                    "Original center, slot and anatomical-mirror depth metadata required"
                )
            return (
                feature_vector(torch.from_numpy(f.copy()), torch.from_numpy(r.copy()), self.role),
                torch.from_numpy(p.copy()),
                torch.from_numpy(np.broadcast_to(slots, (2, k)).copy()),
                torch.from_numpy(depth.copy()),
                torch.from_numpy(np.broadcast_to(present, (2, k)).copy()),
                torch.ones((2, k), dtype=torch.bool),
                self.labels[position],
                self.masks[position],
            )
