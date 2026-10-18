"""Exact original cached-input helper with explicit metadata-only node aliases.

No parent weights/native MRI volume is copied or instantiated. Only the source
parent's run metadata is present; online mode is deliberately unavailable.
"""

import hashlib
import importlib.util
import json
from pathlib import Path

R = Path("/tmp/rsna-path-recovery-evc26-20261018-v2")
T = R / "reader-engine-v1"
spec = importlib.util.spec_from_file_location(
    "unchanged_original_input_and_recovery_checks", T / "code/original_input_helper.py"
)
original = importlib.util.module_from_spec(spec)
spec.loader.exec_module(original)
original.R = R
identical, parent_digest = original.identical, original.parent_digest


def inputs(admission, mode):
    if mode != "cached":
        raise ValueError("Only original fully transferred cached MRI maps; no encoder on this node")
    proof = json.loads((R / "fixture-verified.json").read_text())
    if proof["feature_files_fully_stream_download_size_SHA_verified"] != 32:
        raise ValueError("Original completed whole-copy proof required")
    run = R / "parent/run.json"
    if hashlib.sha256(run.read_bytes()).hexdigest() != proof["parent_run_binding_sha256"]:
        raise ValueError("Original source parent metadata changed")
    pointer = R / "parent/epoch-checkpoints/epoch-016.pt"
    if pointer.exists():
        raise ValueError("This path is metadata-only; parent weights are deliberately not copied")
    return original.inputs(admission, "cached")
