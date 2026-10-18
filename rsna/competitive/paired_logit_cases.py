"""Source-bound training corrections; no images/labels or mutable epoch cache."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path
from typing import Any

import numpy as np
import torch

from rsna.competitive.licensed_importance import ProposalCases, Recipe, sample
from rsna.competitive.paired_logit_control import lookup

VERSION = "paired_parent_logit_IID16_training_v1"


class PairedCases(ProposalCases):
    def __init__(self, original, table, table_sha256, recipe):
        if not isinstance(original, ProposalCases) or not isinstance(recipe, Recipe):
            raise ValueError("Admitted proposal population and recipe required")
        if recipe.role != "influence" or recipe.epochs != 3:
            raise ValueError("Original fixed three-epoch influence schedule required")
        path = Path(table)
        if not 0 < path.stat().st_size <= 8 * 1024 * 1024:
            raise ValueError("Bounded original correction table required")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != table_sha256:
            raise ValueError("Exact correction table SHA required")
        shapes = {
            "study_keys": ((len(original), 32), np.uint8),
            "positions": ((len(original), 3, 16), np.uint8),
            "mirror": ((len(original), 3), np.bool_),
            "log_proposal": ((len(original), 3, 16), np.float32),
            "shift": ((len(original), 3, 12), np.float32),
        }
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != set(shapes) or len(archive.files) != len(shapes):
                raise ValueError("Complete unique correction arrays required")
            arrays = {}
            for name, (shape, dtype) in shapes.items():
                value = archive[name]
                if value.shape != shape or value.dtype != dtype or not np.isfinite(value).all():
                    raise ValueError("Exact finite source-aligned correction arrays required")
                arrays[name] = value.copy()
        super().__init__(
            original.original, torch.stack([original.proposal(i) for i in range(len(original))])
        )
        self.recipe = recipe
        self.correction_sha256 = table_sha256
        self._entries = []
        for index, uid in enumerate(self.uids):
            digest = hashlib.sha256(uid.encode()).digest()
            if arrays["study_keys"][index].tobytes() != digest:
                raise ValueError("Exact original fitting order and identity required")
            entry: dict[str, Any] = {
                "study_uid_sha256": digest.hex(),
                **{
                    name: torch.from_numpy(arrays[name][index].copy())
                    for name in ("positions", "mirror", "log_proposal", "shift")
                },
            }
            for epoch in range(3):
                selected, mirror, logq = sample(uid, recipe, epoch, self.proposal(index))
                if (
                    entry["positions"][epoch].tolist() != selected
                    or bool(entry["mirror"][epoch]) != mirror
                    or not torch.equal(entry["log_proposal"][epoch], logq)
                ):
                    raise ValueError("All exact original epoch draws must match")
            self._entries.append(entry)

    def corrected_case(self, index, positions, mirror, log_proposal):
        # Lookup precedes pixels: wrong source/draw cannot invoke a reader.
        shift = lookup(self._entries[index], self.uids[index], positions, mirror, log_proposal)
        result = self.case(index, positions, mirror)
        result["baseline_shift"] = shift
        return result
