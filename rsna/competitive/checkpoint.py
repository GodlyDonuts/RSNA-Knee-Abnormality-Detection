"""Atomic, source-bound optimizer-boundary recovery for one fixed world size."""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import random
import shutil
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@contextlib.contextmanager
def run_lock(output: Path) -> Iterator[None]:
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".run.lock").open("a") as handle:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another process owns this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", dir=path.parent, delete=False) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, sort_keys=True, indent=2, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _tensor_bytes(value: object) -> int:
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value, (list, tuple)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def save_checkpoint(path: Path, state: dict, *, min_free_gib: float = 20.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Old checkpoint stays intact while writing the successor.
    required = int(min_free_gib * 1024**3) + _tensor_bytes(state) + 1024**2
    if shutil.disk_usage(path.parent).free < required:
        raise OSError("Insufficient free disk to write a checkpoint while preserving the reserve")
    descriptor, name = tempfile.mkstemp(prefix="checkpoint-", suffix=".partial", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(state, handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        temporary.unlink(missing_ok=True)


def capture_rng(device: torch.device) -> dict:
    numpy_state: Any = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state(device).cpu() if device.type == "cuda" else None,
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
    }


def restore_rng(state: dict, device: torch.device) -> None:
    torch.set_rng_state(state["torch"].cpu())
    if device.type == "cuda":
        torch.cuda.set_rng_state(state["cuda"].cpu(), device)
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32), *numpy_state[2:])
    )


def load_checkpoint(path: Path, binding: dict) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    expected_schema = (
        2 if binding.get("config", {}).get("checkpoint_format") == "frozen_base_delta" else 1
    )
    if checkpoint.get("schema_version") != expected_schema or checkpoint.get("binding") != binding:
        raise ValueError("Checkpoint source, configuration, code, device or world size differs")
    if not checkpoint.get("optimizer_boundary"):
        raise ValueError("Checkpoint is not a complete effective-batch boundary")
    if len(checkpoint.get("rng_by_rank", [])) != binding["world_size"]:
        raise ValueError("Checkpoint is missing per-rank RNG state")
    return checkpoint
