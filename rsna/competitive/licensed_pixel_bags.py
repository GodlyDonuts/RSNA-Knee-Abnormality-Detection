"""Read immutable original uint8 MRI bags without a second pixel extraction tree.

Private records/targets and the exact original CPU renderer are supplied by an
admitted wrapper. No external source, image, label or identity is embedded here.
"""

from __future__ import annotations

import hashlib
import json
import tarfile
from pathlib import Path

import torch

from rsna.competitive.licensed_pixel_cache import decode

POPULATION_SHA = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"


class PixelBags:
    def __init__(self, records, roots, targets, render, *, diagnostic_only=False):
        self.records = [dict(row) for row in records]
        records = self.records
        self.roots, self.render = {k: Path(p) for k, p in roots.items()}, render
        self.uids = [r["study_uid"] for r in records]
        if (
            not records
            or len(set(self.uids)) != len(records)
            or list(targets["uids"]) != self.uids
            or (
                not diagnostic_only
                and (
                    len(records) != 3887
                    or hashlib.sha256(
                        json.dumps(self.uids, separators=(",", ":")).encode()
                    ).hexdigest()
                    != POPULATION_SHA
                )
            )
        ):
            raise ValueError("Exact ordered original fitting pixel/target population required")
        self.labels = torch.as_tensor(targets["labels"], dtype=torch.float32).clone()
        self.mask = torch.as_tensor(targets["masks"]).clone()
        self.teacher = torch.as_tensor(targets["parent_logits"]).clone()
        n = len(records)
        if (
            self.labels.shape != (n, 12)
            or self.mask.shape != (n, 12)
            or self.mask.dtype != torch.bool
            or self.teacher.shape != (n, 2, 12)
            or self.teacher.dtype != torch.float32
            or not torch.isfinite(self.teacher).all()
            or not torch.isfinite(self.labels[self.mask]).all()
            or ((self.labels[self.mask] < 0) | (self.labels[self.mask] > 1)).any()
        ):
            raise ValueError(
                "Original aligned FP32 source/teacher probabilities and bool masks required"
            )
        self.signatures = {}
        self.members = {}
        self.verified_inputs = set()
        if {r["holder"] for r in records} != set(self.roots):
            raise ValueError("Exact original archive holders required")
        for holder, path in self.roots.items():
            self.stat(path)
            with tarfile.open(path, "r:") as f:
                members = f.getmembers()
            mapped = {m.name: m for m in members}
            if len(mapped) != len(members) or any(
                not m.isreg()
                or Path(m.name).name != m.name
                or len(m.name) != 68
                or not m.name.endswith(".npz")
                or not 0 < m.size <= 16 * 2**20
                for m in members
            ):
                raise ValueError("Safe unique bounded original pixel members required")
            if set(mapped) != {r["file"] for r in records if r["holder"] == holder}:
                raise ValueError("Exact archive member population required")
            self.members[holder] = mapped
        for row in records:
            if (
                row["file"] != hashlib.sha256(row["study_uid"].encode()).hexdigest() + ".npz"
                or row["file"] not in self.members[row["holder"]]
                or row["bytes"] != self.members[row["holder"]][row["file"]].size
            ):
                raise ValueError("Exact original pixel member identity/size required")

    def stat(self, path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Original regular private pixel archive required")
        s = path.stat()
        signature = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if path in self.signatures and self.signatures[path] != signature:
            raise ValueError("Original pixel archive changed after admission")
        self.signatures[path] = signature

    def __len__(self):
        return len(self.records)

    def case(self, index, positions, mirror):
        if (
            type(index) is not int
            or not 0 <= index < len(self)
            or type(mirror) is not bool
            or not isinstance(positions, list)
            or not 1 <= len(positions) <= 94
            or any(type(i) is not int or not 0 <= i < 94 for i in positions)
            or len(set(positions)) != len(positions)
        ):
            raise ValueError(
                "Exact case, original unique window indices and anatomical view required"
            )
        row = self.records[index]
        path = self.roots[row["holder"]]
        self.stat(path)
        member = self.members[row["holder"]][row["file"]]
        with path.open("rb") as f:
            f.seek(member.offset_data)
            raw = f.read(member.size)
        volume, mask = decode(raw, row["sha256"])
        if (
            not mask.any()
            or hashlib.sha256(volume.tobytes()).hexdigest() != row["volume_sha256"]
            or hashlib.sha256(mask.tobytes()).hexdigest() != row["mask_sha256"]
        ):
            raise ValueError("Original nonempty decoded volume/mask bytes required")
        images = self.render(volume, mask, mirror)
        if (
            not isinstance(images, torch.Tensor)
            or images.dtype != torch.float32
            or images.shape != (94, 3, 320, 320)
            or not torch.isfinite(images).all()
        ):
            raise ValueError("Exact finite original94-window FP32 image tensor required")
        key = (path, row["file"], mirror)
        if key not in self.verified_inputs:
            expected = row[
                "original_mirrored_input_sha256" if mirror else "original_plain_input_sha256"
            ]
            if hashlib.sha256(images.contiguous().numpy().tobytes()).hexdigest() != expected:
                raise ValueError("Original rendered window input tensor differs")
            self.verified_inputs.add(key)
        return {
            "images": images[positions].contiguous(),
            "labels": self.labels[index],
            "mask": self.mask[index],
            "parent_logits": self.teacher[index, int(mirror)],
        }
