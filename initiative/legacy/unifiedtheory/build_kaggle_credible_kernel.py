#!/usr/bin/env python3
"""Build, but do not publish, the audited heterogeneous Kaggle kernel."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--owner", default="tomdifiore")
    parser.add_argument("--slug", default="rsna-knee-credible-heterogeneous-route")
    parser.add_argument("--code-dataset", default="tomdifiore/rsna-knee-unifiedtheory-code")
    parser.add_argument("--checkpoint-dataset", default="tomdifiore/rsna-knee-credible-checkpoints")
    parser.add_argument("--dino-model-source", default="metaresearch/dinov2/pyTorch/base/1")
    return parser.parse_args()


def checkpoint_from_oof(path: str) -> Path:
    source = Path(path)
    if not source.name.endswith("_oof.csv"):
        raise ValueError(f"cannot derive checkpoint from OOF source {source}")
    return source.with_name(source.name.removesuffix("_oof.csv") + ".pt")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_checkpoint_contract(blend: dict[str, Any]) -> dict[str, list[dict[str, str]]]:
    contract = {}
    for name in blend["members"]:
        checkpoints = blend.get("checkpoint_source_files", {}).get(name)
        if isinstance(checkpoints, list) and checkpoints:
            sources = [Path(source) for source in checkpoints]
        else:
            oof_sources = blend.get("source_files", {}).get(name)
            if not isinstance(oof_sources, list) or not oof_sources:
                raise ValueError(f"blend has no source files for {name!r}")
            sources = [checkpoint_from_oof(source) for source in oof_sources]
        rows = []
        for checkpoint in sources:
            if not checkpoint.is_file():
                raise FileNotFoundError(checkpoint)
            rows.append({"name": checkpoint.name, "sha256": sha256(checkpoint)})
        contract[name] = rows
    return contract


def portable_families(blend: dict[str, Any], registry: dict[str, Any]) -> dict[str, Any]:
    result = {}
    for name in blend["members"]:
        if name == blend["anchor"]:
            continue
        family = registry.get("families", {}).get(name)
        if not isinstance(family, dict):
            raise ValueError(f"selected raw member {name!r} is absent from registry")
        arguments = list(family.get("train_args", []))
        if "--backbone" not in arguments:
            raise ValueError(f"family {name!r} has no backbone declaration")
        backbone = arguments[arguments.index("--backbone") + 1]
        result[name] = {
            "inference_slices": int(family.get("inference_slices", 16)),
            "requires_dino_config": backbone == "dinov2",
        }
    return result


CELL_TEMPLATE = r'''
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import time

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
STARTED = time.monotonic()
BUDGET_SECONDS = 8.75 * 3600
BLEND = __BLEND__
CHECKPOINTS = __CHECKPOINTS__
RAW_FAMILIES = __RAW_FAMILIES__
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

wanted_names = {
    "run_selected_raw_inference.py", "sample_submission.csv", "config.json",
    *(row["name"] for rows in CHECKPOINTS.values() for row in rows),
}
file_index = {name: [] for name in wanted_names}
pruned_directories = {
    "train_series", "test_series", "train_images", "test_images",
    "train", "test",
}
for root, directories, files in os.walk(INPUT):
    directories[:] = [name for name in directories if name not in pruned_directories]
    for name in wanted_names.intersection(files):
        file_index[name].append(Path(root) / name)

code_hits = file_index["run_selected_raw_inference.py"]
if len(code_hits) != 1:
    raise RuntimeError(f"expected exactly one code entry point, found {code_hits}")
code_dir = code_hits[0].parent
sample_hits = file_index["sample_submission.csv"]
if len(sample_hits) != 1:
    raise RuntimeError(f"expected exactly one sample submission, found {sample_hits}")
COMPETITION_INPUT = sample_hits[0].parent

digest_cache = {}

def digest(path):
    key = str(path)
    if key in digest_cache:
        return digest_cache[key]
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            value.update(chunk)
    digest_cache[key] = value.hexdigest()
    return digest_cache[key]

def locate_checkpoint(row):
    named_hits = file_index[row["name"]]
    hits = [path for path in named_hits if digest(path) == row["sha256"]]
    if len(hits) != 1:
        raise RuntimeError(
            f"expected one hash-matching attached {row['name']}, "
            f"found {hits} among {named_hits}"
        )
    return hits[0]

resolved = {
    member: [locate_checkpoint(row) for row in rows]
    for member, rows in CHECKPOINTS.items()
}

model_candidates = []
for config in file_index["config.json"]:
    try:
        payload = json.loads(config.read_text())
    except Exception:
        continue
    if payload.get("model_type") == "dinov2" and int(payload.get("hidden_size", 0)) == 768:
        model_candidates.append(config.parent)
needs_dino = True  # the consensus patch anchor always uses DINOv2-base
if needs_dino and len(model_candidates) != 1:
    raise RuntimeError(f"expected one local DINOv2-base config, found {model_candidates}")
dino_model = model_candidates[0]

blend_path = WORKING / "credible_blend.json"
blend_path.write_text(json.dumps(BLEND, indent=2) + "\n")
member_csv = {}
anchor = BLEND["anchor"]
anchor_csv = WORKING / f"{anchor}.csv"
patch_command = [
    sys.executable, str(code_dir / "kaggle_offline_infer.py"),
    "--data-root", str(COMPETITION_INPUT),
    "--dino-model", str(dino_model),
    "--work-dir", str(WORKING / "patch_cache"),
    "--output", str(anchor_csv),
    "--runtime-json", str(WORKING / "patch_runtime.json"),
    "--batch-size", "16", "--inference-batch-size", "3",
    "--workers", "4", "--max-slices", "64", "--time-budget-hours", "8.5",
]
for checkpoint in resolved[anchor]:
    patch_command.extend(["--checkpoint-glob", str(checkpoint)])
subprocess.run(patch_command, check=True)
member_csv[anchor] = anchor_csv

for name, family in RAW_FAMILIES.items():
    if time.monotonic() - STARTED >= BUDGET_SECONDS:
        raise RuntimeError("credible route exhausted the Kaggle runtime budget")
    destination = WORKING / f"{name}.csv"
    command = [
        sys.executable, str(code_dir / "infer_raw_mil.py"),
        "--data-root", str(COMPETITION_INPUT),
        "--sample-submission", str(COMPETITION_INPUT / "sample_submission.csv"),
        "--output", str(destination),
        "--slices-per-plane", str(family["inference_slices"]),
        "--workers", "4",
    ]
    if family["requires_dino_config"]:
        command.extend(["--model-name", str(dino_model)])
    for checkpoint in resolved[name]:
        command.extend(["--checkpoint-glob", str(checkpoint)])
    subprocess.run(command, check=True)
    member_csv[name] = destination

if time.monotonic() - STARTED >= BUDGET_SECONDS:
    raise RuntimeError("credible route exhausted the Kaggle runtime budget")
blend_command = [
    sys.executable, str(code_dir / "blend_submission_files.py"),
    "--blend", str(blend_path),
    "--sample-submission", str(COMPETITION_INPUT / "sample_submission.csv"),
    "--output", str(WORKING / "submission.csv"),
]
for name in BLEND["members"]:
    blend_command.extend(["--member", f"{name}={member_csv[name]}"])
subprocess.run(blend_command, check=True)
if not (WORKING / "submission.csv").is_file():
    raise RuntimeError("inference completed without submission.csv")
(WORKING / "credible_runtime.json").write_text(json.dumps({
    "elapsed_seconds": time.monotonic() - STARTED,
    "budget_seconds": BUDGET_SECONDS,
    "members": BLEND["members"],
    "checkpoint_count": sum(map(len, resolved.values())),
}, indent=2) + "\n")
'''.strip()


def main() -> None:
    args = parse_args()
    blend = json.loads(args.blend.read_text())
    registry = json.loads(args.registry.read_text())
    contract = selected_checkpoint_contract(blend)
    families = portable_families(blend, registry)
    portable_blend = {
        "schema_version": int(blend.get("schema_version", 1)),
        "method": str(blend.get("method", "audited heterogeneous rank blend")),
        "anchor": str(blend["anchor"]),
        "members": list(blend["members"]),
        "weights": blend["weights"],
    }
    cell = (
        CELL_TEMPLATE.replace("__BLEND__", repr(portable_blend))
        .replace("__CHECKPOINTS__", repr(contract))
        .replace("__RAW_FAMILIES__", repr(families))
    )
    args.output.mkdir(parents=True, exist_ok=True)
    notebook_path = args.output / "credible_route_inference.ipynb"
    notebook = {
        "cells": [
            {
                "cell_type": "markdown", "id": "route-description", "metadata": {},
                "source": [
                    "# RSNA knee credible heterogeneous route\n",
                    "Audited consensus DINO anchor plus only nested-OOF-promoted raw specialists.\n",
                ],
            },
            {
                "cell_type": "code", "id": "credible-inference", "execution_count": None,
                "metadata": {}, "outputs": [],
                "source": [line + "\n" for line in cell.splitlines()],
            },
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    notebook_path.write_text(json.dumps(notebook, indent=1) + "\n")
    metadata = {
        "id": f"{args.owner}/{args.slug}",
        "title": "RSNA Knee Credible Heterogeneous Route",
        "code_file": notebook_path.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": False,
        "enable_gpu": True,
        "enable_internet": False,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "dataset_sources": [args.code_dataset, args.checkpoint_dataset],
        "model_sources": [args.dino_model_source],
    }
    (args.output / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (args.output / "checkpoint_contract.json").write_text(
        json.dumps(contract, indent=2) + "\n"
    )
    print(json.dumps({"notebook": str(notebook_path), "metadata": metadata, "members": blend["members"]}, indent=2))


if __name__ == "__main__":
    main()
