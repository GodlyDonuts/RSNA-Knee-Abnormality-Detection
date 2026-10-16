"""Deterministic FP32 context-head fitting on immutable licensed MRI features.

No CNN fitting or independent OOF claim. Private indexes, labels and predictions
must stay on authorized storage. The same engine serves recovery and full fitting.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import tarfile
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from rsna.competitive.released_context import (
    ContextResidual,
    anchored_loss,
    tta_probabilities,
    window_metadata,
)

VERSION = "licensed_frozen_feature_context_trainer_v1"
PARENT_SHA = "7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef"


def digest(path):
    with Path(path).open("rb") as handle:
        value = hashlib.sha256()
        for block in iter(lambda: handle.read(2**20), b""):
            value.update(block)
        return value.hexdigest()


def object_sha(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Recipe:
    epochs: int = 10
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    warmup_epochs: int = 1
    clip: float = 1.0
    source_weight: float = 0.25
    seed: int = 20261016
    embedding_dim: int = 1024
    width: int = 128
    targets: int = 12
    layers: int = 2
    limit: float = 0.5

    def __post_init__(self):
        for name in ("epochs", "batch_size", "embedding_dim", "width", "targets"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError("Positive integer recipe dimensions required")
        if (
            type(self.warmup_epochs) is not int
            or not 0 <= self.warmup_epochs < self.epochs
            or type(self.seed) is not int
            or not 0 <= self.seed < 2**63
            or type(self.layers) is not int
            or not 0 <= self.layers <= 2
            or self.width % 4
        ):
            raise ValueError("Invalid warmup, seed or head geometry")
        for name in ("learning_rate", "clip", "source_weight", "limit"):
            x = getattr(self, name)
            if type(x) not in (float, int) or not math.isfinite(x) or x <= 0:
                raise ValueError("Finite positive recipe values required")
        if (
            self.source_weight > 1
            or self.limit > 0.5
            or type(self.weight_decay) not in (float, int)
            or not math.isfinite(self.weight_decay)
            or self.weight_decay < 0
        ):
            raise ValueError("Invalid source weight, correction bound or decay")


def learning_rate(recipe, completed, batches_per_epoch):
    total = recipe.epochs * batches_per_epoch
    if not 0 <= completed < total:
        raise ValueError("Update cursor outside fixed recipe")
    warmup = recipe.warmup_epochs * batches_per_epoch
    if completed < warmup:
        return recipe.learning_rate * (completed + 1) / warmup
    elapsed = completed - warmup
    span = max(1, total - warmup - 1)
    return recipe.learning_rate * 0.5 * (1 + math.cos(math.pi * elapsed / span))


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"]:
        if not torch.cuda.is_initialized():
            raise ValueError("Checkpoint requires an initialized original CUDA runtime")
        torch.cuda.set_rng_state_all(state["cuda"])


class FrozenFeatures:
    """Read exact NPZ members without extracting a second feature tree.

    Caller admits source, whole archive/index bytes and eligible label membership
    once before construction. Individual member bytes are checked on their first
    read; subsequent reads require the same original file stat signature.
    """

    def __init__(self, records, roots, labels, masks, binding, *, embedding_dim=1024, targets=12):
        self.records, self.roots = records, {k: Path(v) for k, v in roots.items()}
        self.binding, self.embedding_dim, self.targets = binding, embedding_dim, targets
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.masks = torch.as_tensor(masks, dtype=torch.bool)
        if (
            not records
            or self.labels.shape != (len(records), targets)
            or self.masks.shape != self.labels.shape
            or not torch.isfinite(self.labels[self.masks]).all()
            or ((self.labels[self.masks] < 0) | (self.labels[self.masks] > 1)).any()
            or len({r["study_uid"] for r in records}) != len(records)
            or not isinstance(binding, dict)
            or not binding
        ):
            raise ValueError(
                "Distinct admitted fitting records and aligned observed probabilities required"
            )
        self.signatures = {}
        self.members: dict[Path, dict[str, tarfile.TarInfo]] = {}
        self.verified = set()
        for row in records:
            uid, name = row["study_uid"], row["file"]
            if (
                not isinstance(uid, str)
                or name != hashlib.sha256(uid.encode()).hexdigest() + ".npz"
                or type(row["bytes"]) is not int
                or not 0 < row["bytes"] <= 2**20
                or len(row["sha256"]) != 64
                or row["source_kernel"] not in self.roots
            ):
                raise ValueError("Exact safe feature member identity required")
            root = self.roots[row["source_kernel"]]
            archive = row.get("archive")
            if archive is not None and archive != "features.tar":
                raise ValueError("Only the admitted streaming features archive is accepted")
            path = root / (archive or name)
            self._stat(path)
            if archive and path not in self.members:
                with tarfile.open(path, "r:") as handle:
                    members = handle.getmembers()
                mapped = {m.name: m for m in members}
                if len(mapped) != len(members) or any(
                    not m.isreg()
                    or Path(m.name).name != m.name
                    or len(m.name) != 68
                    or not m.name.endswith(".npz")
                    or not 0 < m.size <= 2**20
                    for m in members
                ):
                    raise ValueError("Unsafe or duplicated archive feature member")
                self.members[path] = mapped
            if archive:
                member = self.members[path].get(name)
                if member is None or member.size != row["bytes"]:
                    raise ValueError("Missing or wrongly sized admitted archive member")

    def _stat(self, path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("Original regular feature file required")
        s = path.stat()
        signature = (s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns)
        if path in self.signatures and signature != self.signatures[path]:
            raise ValueError("Admitted feature bytes changed after validation")
        self.signatures[path] = signature

    def __len__(self):
        return len(self.records)

    def __getitem__(self, position):
        row = self.records[position]
        root = self.roots[row["source_kernel"]]
        path = root / row.get("archive", row["file"])
        self._stat(path)
        with path.open("rb") as handle:
            if row.get("archive"):
                handle.seek(self.members[path][row["file"]].offset_data)
            raw = handle.read(row["bytes"])
        key = (path, row["file"])
        if key not in self.verified:
            if len(raw) != row["bytes"] or hashlib.sha256(raw).hexdigest() != row["sha256"]:
                raise ValueError("Original NPZ feature bytes failed admission")
            self.verified.add(key)
        with np.load(io.BytesIO(raw), allow_pickle=False) as arrays:
            if set(arrays.files) != {
                "features",
                "parent_logits",
                "centers",
                "slots",
                "depth",
                "present",
            }:
                raise ValueError("Exact accepted feature schema required")
            f, p, c, slot, depth, present = (
                arrays[n]
                for n in ("features", "parent_logits", "centers", "slots", "depth", "present")
            )
            k = len(c)
            if (
                f.dtype != np.float32
                or p.dtype != np.float32
                or c.dtype != np.int64
                or slot.dtype != np.int64
                or depth.dtype != np.float32
                or present.dtype != np.bool_
                or f.shape != (2, k, self.embedding_dim)
                or p.shape != (2, self.targets)
                or c.shape != (k,)
                or slot.shape != (k,)
                or depth.shape != (2, k)
                or present.shape != (k,)
                or not k
                or not np.isfinite(f).all()
                or not np.isfinite(p).all()
            ):
                raise ValueError("Original FP32 feature/metadata shapes required")
            expected_slot, plain = window_metadata(torch.from_numpy(c))
            _, mirror = window_metadata(torch.from_numpy(c), mirrored=True)
            if not np.array_equal(slot, expected_slot.numpy()) or not np.array_equal(
                depth, torch.stack([plain, mirror]).numpy()
            ):
                raise ValueError("Original selected-depth and anatomical-mirror metadata required")
            return (
                torch.from_numpy(f.copy()),
                torch.from_numpy(p.copy()),
                torch.from_numpy(np.broadcast_to(slot, (2, k)).copy()),
                torch.from_numpy(depth.copy()),
                torch.from_numpy(np.broadcast_to(present, (2, k)).copy()),
                torch.ones((2, k), dtype=torch.bool),
                self.labels[position],
                self.masks[position],
            )


def batch(dataset, indices):
    examples = [dataset[i] for i in indices]
    if not examples:
        raise ValueError("Nonempty study batch required")
    tensors = [torch.stack([example[n] for example in examples]) for n in range(8)]
    x = [t.flatten(0, 1) for t in tensors[:6]]
    labels, masks = (t.repeat_interleave(2, dim=0) for t in tensors[6:])
    return x, labels, masks


class Trainer:
    """One fixed head engine with complete update-boundary recovery state."""

    def __init__(self, dataset, recipe):
        if not len(dataset):
            raise ValueError("Nonempty admitted fitting dataset required")
        self.data, self.recipe = dataset, recipe
        self.steps = math.ceil(len(dataset) / recipe.batch_size)
        torch.use_deterministic_algorithms(True)
        self.model = (
            ContextResidual(
                recipe.embedding_dim,
                recipe.width,
                recipe.targets,
                recipe.layers,
                recipe.limit,
                recipe.seed,
            )
            .float()
            .cpu()
        )
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=recipe.learning_rate,
            weight_decay=recipe.weight_decay,
            foreach=False,
            fused=False,
        )
        random.seed(recipe.seed)
        np.random.seed(recipe.seed % 2**32)
        torch.manual_seed(recipe.seed)
        self.order_rng = torch.Generator(device="cpu").manual_seed(recipe.seed)
        self.order = None
        self.epoch, self.cursor, self.completed = 0, 0, 0
        self.history = []

    def state(self):
        return {
            "version": VERSION,
            "recipe": asdict(self.recipe),
            "binding": self.data.binding,
            "population": len(self.data),
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": {
                "kind": "linear_warmup_cosine_v1",
                "batches_per_epoch": self.steps,
                "completed": self.completed,
            },
            "epoch": self.epoch,
            "cursor": self.cursor,
            "completed": self.completed,
            "order": self.order,
            "order_rng": self.order_rng.get_state(),
            "rng": rng_state(),
            "history": self.history,
        }

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if path.exists():
            raise ValueError("Immutable context checkpoint already exists")
        with path.open("xb") as handle:
            os.fchmod(handle.fileno(), 0o600)
            torch.save(self.state(), handle)

    def load(self, path):
        value = torch.load(path, map_location="cpu", weights_only=False)
        if (
            value["version"] != VERSION
            or value["recipe"] != asdict(self.recipe)
            or value["binding"] != self.data.binding
            or value["population"] != len(self.data)
            or value["scheduler"]
            != {
                "kind": "linear_warmup_cosine_v1",
                "batches_per_epoch": self.steps,
                "completed": value["completed"],
            }
            or value["completed"] != value["epoch"] * self.steps + value["cursor"]
            or not 0 <= value["epoch"] <= self.recipe.epochs
            or not 0 <= value["cursor"] < self.steps
            or len(value["history"]) != value["completed"]
        ):
            raise ValueError("Checkpoint recipe, input binding or update cursor changed")
        order = value["order"]
        if (value["cursor"] and order is None) or (
            order is not None and sorted(order) != list(range(len(self.data)))
        ):
            raise ValueError("Complete original epoch permutation required")
        self.model.load_state_dict(value["model"], strict=True)
        self.optimizer.load_state_dict(value["optimizer"])
        self.order_rng.set_state(value["order_rng"])
        self.epoch, self.cursor, self.completed = (
            value["epoch"],
            value["cursor"],
            value["completed"],
        )
        self.order, self.history = order, value["history"]
        restore_rng(value["rng"])

    def run(self, *, stop_after=None, epoch_directory=None, progress=None):
        if (
            stop_after is not None
            and not self.completed < stop_after <= self.steps * self.recipe.epochs
        ):
            raise ValueError("Future fixed update boundary required")
        self.model.train()
        while self.epoch < self.recipe.epochs:
            if self.order is None:
                self.order = torch.randperm(len(self.data), generator=self.order_rng).tolist()
            start = self.cursor * self.recipe.batch_size
            indices = self.order[start : start + self.recipe.batch_size]
            x, labels, masks = batch(self.data, indices)
            self.optimizer.zero_grad(set_to_none=True)
            lr = learning_rate(self.recipe, self.completed, self.steps)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            logits, correction = self.model(*x)
            loss = anchored_loss(
                logits, x[1], labels, masks, source_weight=self.recipe.source_weight
            )
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.recipe.clip)
            if not torch.isfinite(loss) or not torch.isfinite(norm):
                raise ValueError("Nonfinite context optimizer update")
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            if any(not torch.isfinite(p).all() for p in self.model.parameters()):
                raise ValueError("Nonfinite learned context parameter")
            self.completed += 1
            self.cursor += 1
            record = {
                "update": self.completed,
                "epoch": self.epoch + 1,
                "batch": self.cursor,
                "studies": len(indices),
                "lr": lr,
                "loss": float(loss.detach()),
                "gradient_norm": float(norm),
                "max_abs_correction": float(correction.detach().abs().max()),
            }
            self.history.append(record)
            if self.cursor == self.steps:
                self.epoch += 1
                self.cursor, self.order = 0, None
                if epoch_directory is not None:
                    self.save(Path(epoch_directory) / f"epoch-{self.epoch:03d}.pt")
            if progress is not None:
                progress(record)
            if stop_after == self.completed:
                break

    @torch.no_grad()
    def predictions(self):
        previous = self.model.training
        self.model.eval()
        result = []
        try:
            for start in range(0, len(self.data), self.recipe.batch_size):
                x, _, _ = batch(
                    self.data,
                    list(range(start, min(start + self.recipe.batch_size, len(self.data)))),
                )
                logits, _ = self.model(*x)
                result.append(tta_probabilities(logits.view(-1, 2, self.recipe.targets)).numpy())
        finally:
            self.model.train(previous)
        return np.concatenate(result)
