"""Compile the reviewed v15 training-window nodes (``take_group``/``augment``) for tests.

Only exact original source segments listed in ``review.json`` are compiled, through the
same gate as the renderer and teacher namespaces. The canonical trainer never imports
this module; it exists so tests can compare ``rsna.competitive.source_slot`` against the
original functions on identical inputs and random draws.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from tools.research.public_dinov2_input_reference import provenance

REVIEW_PATH = Path(__file__).with_name("review.json")


def load_review(path: Path = REVIEW_PATH) -> dict:
    review = json.loads(Path(path).read_text())
    if (
        review.get("source_sha256") != provenance.SOURCE_SHA256
        or review.get("status") != "reviewed"
        or review.get("node_text_format") != "original_source_segments_utf8"
    ):
        raise ValueError("Source-matched slot review is not bound to the reviewed source")
    return review


def load_namespace(raw: bytes, group: int = 3) -> dict[str, Any]:
    review = load_review()
    exceptions = review["reviewed_exceptions"]
    text, digest = provenance.extract_nodes(
        provenance.verify_source(raw), list(review["nodes"]), exceptions
    )
    if digest != review["node_text_sha256"]:
        raise ValueError("Source-matched slot node text digest changed")
    preamble = {"np": np, "torch": torch, "F": F, "GROUP": group}
    return provenance.compile_nodes(text, preamble, exceptions)
