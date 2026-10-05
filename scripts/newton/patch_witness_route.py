"""Bound the one-time private 32-study transfer before writing destination data."""

from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath

HEADER_CAP = 2 * 1024**2
INPUT_CAP = 2 * 1024**3


def parse_header(raw: bytes) -> dict:
    if len(raw) > HEADER_CAP or not raw.endswith(b"\n"):
        raise ValueError("Complete bounded private transfer header required")
    value = json.loads(raw)
    if value.get("source_node") != "evc31" or value.get("destination_node") != "evc48":
        raise ValueError("Exact admitted source and destination required")
    members = value.get("members")
    if not isinstance(members, dict) or len(members) != 370:
        raise ValueError("Exact original32-study370-member copy required")
    for name, row in members.items():
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or str(path) != name
            or len(path.parts) != 4
            or path.parts[:2] != ("cache", "train_series")
            or path.suffix not in {".npz", ".json"}
            or type(row.get("bytes")) is not int
            or row["bytes"] <= 0
            or len(row.get("sha256", "")) != 64
            or any(c not in "0123456789abcdef" for c in row["sha256"])
        ):
            raise ValueError("Unsafe or unbound original input member")
    if sum(row["bytes"] for row in members.values()) > INPUT_CAP:
        raise ValueError("Private engineering input copy exceeds funded cap")
    if (
        hashlib.sha256(value["original_CPU_raw"].encode()).hexdigest()
        != value["seed"]["original_CPU_receipt_sha256"]
    ):
        raise ValueError("Exact original CPU admission required")
    return value
