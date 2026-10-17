"""Compile exact full-MRI inference for preserved frozen-encoder factorials."""

from __future__ import annotations

import ast
import hashlib
from pathlib import Path

from rsna.competitive import licensed_cached_serving as scope
from rsna.schema import TARGETS
from scripts.kaggle.series_boundary_notebook import GUARD, PARENT_SOURCE_SHA256

ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = (
    ROOT
    / "initiative/experiments/competitive-v1/cached-full-study-factorial-v1/serving-runtime.py.txt"
)
BOUNDED_SHA = "5576dff1348ebb7fa322c0ce78f7fdf8aa4b9ef1f2a6be2b757765fcc733cd42"
CODEC_SHA = "7bdaf41a864b499401ff4a62e6288d30eead3849c0fda5c49cc67b3b2d0e2ab5"


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def standalone(helper):
    if digest(helper) != digest(Path(scope.__file__).read_text()):
        raise ValueError("Exact current cached-readout validator and serving source required")
    parts = []
    removed = 0
    for node in ast.parse(helper).body:
        if isinstance(node, ast.ImportFrom) and node.module == "__future__":
            continue
        if isinstance(node, ast.ImportFrom) and node.module == "rsna.schema":
            removed += 1
            continue
        text = ast.get_source_segment(helper, node)
        if text is None:
            raise ValueError("Complete validated source spans required")
        parts.append(text)
    if removed != 1:
        raise ValueError("One original target-order import required")
    return "TARGETS=" + repr(TARGETS) + "\n" + "\n\n".join(parts) + "\n"


def head_filename(role, windows):
    scope.fixed_recipe(role, windows)
    return f"cached-{role}-{windows}-epoch-003.pt"


def build(
    parent, bounded, helper, codec, engine, full, publication, recovery, role, windows, *, mode
):
    model_sha = scope.FINAL_SHAS.get((role, windows))
    scope.provenance(full, publication, recovery, role, windows, model_sha)
    if (
        mode not in ("qualification", "competition")
        or digest(parent) != PARENT_SOURCE_SHA256
        or digest(bounded) != BOUNDED_SHA
        or digest(codec) != CODEC_SHA
        or digest(engine) != scope.ENGINE
        or parent.count(GUARD) != 1
        or parent[parent.index(GUARD) + len(GUARD) :].strip()
    ):
        raise ValueError("Exact original image graph/helpers and admitted serving mode required")
    validators = standalone(helper)
    readouts = [
        n for n in ast.parse(engine).body if isinstance(n, ast.ClassDef) and n.name == "Readout"
    ]
    if len(readouts) != 1:
        raise ValueError("One exact original training readout graph required")
    readout = ast.get_source_segment(engine, readouts[0])
    if readout is None:
        raise ValueError("Complete accepted training graph required")
    training_class = (
        "import torch\nfrom torch import nn\n"
        f"PARENT_SHA={scope.PARENT!r}\nPARENT_SOURCE={scope.PARENT_SOURCE!r}\n" + readout + "\n"
    )
    source = parent[parent.index("# %% [code]") : parent.index(GUARD)]
    replacements = {
        'torch.load(pt_path, map_location="cpu", weights_only=False)': 'torch.load(pt_path, map_location="cpu", weights_only=True)',
        "nproc = max(4, min(96, (os.cpu_count() or 8) - 8))": "nproc = 2",
        "pool.imap_unordered(_pre_one, test_ids, chunksize=2)": "bounded_results(pool, _pre_one, test_ids, limit=4)",
        "q = queue.Queue(maxsize=QUEUE_MAX)": "q = queue.Queue(maxsize=2)",
        'print(f"  [gpu {d}] study {i} {sid[:16]} FALLBACK ({type(e).__name__}: {e})", flush=True)': 'print(json.dumps({"cached_MRI_fallback_index":i,"error_type":type(e).__name__}),flush=True)',
    }
    for old, new in replacements.items():
        if source.count(old) != 1:
            raise ValueError("Unique original loader/preparation/queue/fallback sites required")
        source = source.replace(old, new)
    specification = {
        "role": role,
        "training_windows": windows,
        "head_filename": head_filename(role, windows),
        "head_bytes": 1126029,
        "head_sha256": model_sha,
        "parent_sha256": scope.PARENT,
    }
    runtime = TEMPLATE.read_text()
    for key, value in {
        "MODE_VALUE": mode,
        "HEAD_SPEC_VALUE": specification,
        "FULL_VALUE": full,
        "PUB_VALUE": publication,
        "RECOVERY_VALUE": recovery,
        "VALIDATOR_VALUE": validators,
        "TRAINING_CLASS_VALUE": training_class,
        "CODEC_VALUE": codec,
    }.items():
        if runtime.count(key) != 1:
            raise ValueError("Unique immutable runtime parameter sites required")
        runtime = runtime.replace(key, repr(value))
    header = (
        "# Licensed fixed-encoder source-label/coverage factorial, fixed epoch3.\n"
        "# Original source comments describe its upstream model, not this head training.\n"
        "import os\nos.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'\n"
    )
    result = header + source + "\n" + bounded + "\n" + runtime
    compile(result, "complete_MRI_cached_readout_serving", "exec")
    return result, {
        **specification,
        "mode": mode,
        "source_sha256": digest(result),
        "model_source_git": scope.SOURCE,
        "helper_sha256": digest(helper),
        "runtime_template_sha256": digest(TEMPLATE.read_text()),
        "parent_source_sha256": PARENT_SOURCE_SHA256,
        "serving_windows": 94,
        "FP32_head_outside_encoder_autocast": True,
        "encoder_weights_changed": False,
        "qualification_cannot_submit": mode == "qualification",
        "new_private_small_head_Dataset_required": True,
        "independent_internal_validation": False,
    }
