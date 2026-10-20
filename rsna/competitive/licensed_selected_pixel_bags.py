"""Selected original MRI windows checked against a private full-input manifest.

The caller must bind the actual original full-cohort manifest receipt and exact
selected renderer source. This reader changes no labels, views or sampling; it
avoids rendering unused windows while checking every returned window's bytes.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from rsna.competitive.licensed_pixel_bags import POPULATION_SHA, PixelBags
from rsna.competitive.licensed_pixel_cache import decode

STUDIES = 3887
STATUS = "actual_full_fitting_original94_both_views_and_private_per_window_integrity_verified"


def record_digest(records):
    return hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def signature(path):
    if path.is_symlink() or not path.is_file():
        raise ValueError("Original private regular integrity file required")
    s = path.stat()
    if s.st_uid != os.getuid() or s.st_mode & 0o077:
        raise ValueError("Owner-only private window integrity file required")
    return (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)


class SelectedPixelBags:
    def __init__(self, original, manifest, receipt, render_selected):
        if (
            not isinstance(original, PixelBags)
            or len(original) != STUDIES
            or receipt.get("status") != STATUS
            or receipt.get("studies") != STUDIES
            or receipt.get("views") != 2
            or receipt.get("windows_per_view") != 94
            or receipt.get("cohort_sha256") != POPULATION_SHA
            or receipt.get("record_binding_sha256") != record_digest(original.records)
            or receipt.get("original_full94_tensor_hash_checks") != STUDIES * 2
            or not callable(render_selected)
        ):
            raise ValueError("Actual original full-cohort both-view window integrity required")
        self.path = Path(manifest)
        before = signature(self.path)
        if before[2] != receipt.get("manifest_bytes") or not 0 < before[2] < 32 * 2**20:
            raise ValueError("Exact bounded private manifest bytes required")
        if hashlib.sha256(self.path.read_bytes()).hexdigest() != receipt.get("manifest_sha256"):
            raise ValueError("Exact original private manifest hash required")
        values = np.load(self.path, allow_pickle=False)
        if (
            not isinstance(values, np.ndarray)
            or values.dtype != np.uint8
            or values.shape != (STUDIES, 2, 94, 32)
        ):
            raise ValueError("Exact per-study/view/window SHA256 byte array required")
        self.hashes = values.copy()
        self.hashes.flags.writeable = False
        self.file_signature = before
        if signature(self.path) != before:
            raise ValueError("Integrity file changed during admission")
        self.original = original
        self.uids = list(original.uids)
        self.render_selected = render_selected
        self.record_signatures = tuple(record_digest(row) for row in original.records)
        self.roots = dict(original.roots)

    def __len__(self):
        return len(self.original)

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
            raise ValueError("Original case, unique window indices and anatomical view required")
        if signature(self.path) != self.file_signature:
            raise ValueError("Original private manifest changed after admission")
        data = self.original
        row = data.records[index]
        if (
            record_digest(row) != self.record_signatures[index]
            or data.uids[index] != self.uids[index]
        ):
            raise ValueError("Original admitted pixel record binding changed")
        path = self.roots[row["holder"]]
        data.stat(path)
        member = data.members[row["holder"]][row["file"]]
        with path.open("rb") as handle:
            handle.seek(member.offset_data)
            raw = handle.read(member.size)
        volume, mask = decode(raw, row["sha256"])
        if (
            not mask.any()
            or hashlib.sha256(volume.tobytes()).hexdigest() != row["volume_sha256"]
            or hashlib.sha256(mask.tobytes()).hexdigest() != row["mask_sha256"]
        ):
            raise ValueError("Original nonempty decoded volume/mask bytes required")
        images = self.render_selected(volume, mask, positions, mirror)
        if (
            not isinstance(images, torch.Tensor)
            or images.dtype != torch.float32
            or images.device.type != "cpu"
            or images.shape != (len(positions), 3, 320, 320)
            or not torch.isfinite(images).all()
        ):
            raise ValueError("Exact finite selected original FP32 CPU windows required")
        for j, position in enumerate(positions):
            actual = hashlib.sha256(images[j].contiguous().numpy().tobytes()).digest()
            if actual != self.hashes[index, int(mirror), position].tobytes():
                raise ValueError("Selected window differs from verified original full94 input")
        return {
            "images": images.contiguous(),
            "labels": data.labels[index],
            "mask": data.mask[index],
            "parent_logits": data.teacher[index, int(mirror)],
        }
