#!/usr/bin/env python3
"""Run offline inference only after the fixed family audit passes.

This is deliberately fail-closed: it checks the preregistered audit contract,
the complete seed/fold checkpoint matrix, and checkpoint metadata before it
delegates to :mod:`kaggle_offline_infer`.  It creates a submission *file* but
never uploads or submits it to Kaggle.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch

try:
    from .constants import TARGETS
except ImportError:
    from constants import TARGETS


AUDIT_CONTRACT = (
    "all seeds and both families receive equal rank weight; "
    "no fitted or target-wise weights"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-summary", type=Path, required=True)
    parser.add_argument("--summary-runs", type=Path, required=True)
    parser.add_argument("--patch-runs", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--dino-model", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[2026, 2027, 2028])
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-json", type=Path)
    parser.add_argument("--decision-json", type=Path)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--inference-batch-size", type=int, default=3)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--max-slices", type=int, default=64)
    parser.add_argument("--time-budget-hours", type=float, default=8.5)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="validate promotion and checkpoints, but do not extract or infer",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def validate_checkpoint(
    path: Path, *, model_type: str, seed: int, fold: int
) -> None:
    if not path.is_file():
        raise FileNotFoundError(path)
    checkpoint = load_checkpoint(path)
    if checkpoint.get("model_type") != model_type:
        raise ValueError(f"{path} is not a {model_type} checkpoint")
    if int(checkpoint.get("seed", -1)) != seed or int(checkpoint.get("fold", -1)) != fold:
        raise ValueError(f"{path} has mismatched seed/fold metadata")
    config = checkpoint.get("model_config", {})
    if config.get("aggregator") != "mean":
        raise ValueError(f"{path} is not a mean-aggregation checkpoint")
    if int(config.get("num_targets", -1)) != len(TARGETS):
        raise ValueError(f"{path} has the wrong target count")
    if model_type == "patch" and int(config.get("token_adapter_bottleneck", -1)) != 0:
        raise ValueError(f"{path} is not from the fixed zero-adapter patch family")


def audited_checkpoints(args: argparse.Namespace) -> list[Path]:
    paths: list[Path] = []
    for seed in args.seeds:
        for fold in range(args.folds):
            summary = args.summary_runs / f"seed{seed}" / f"mean_fold{fold}.pt"
            patch = args.patch_runs / f"seed{seed}" / f"patch_mean_fold{fold}.pt"
            validate_checkpoint(summary, model_type="summary", seed=seed, fold=fold)
            validate_checkpoint(patch, model_type="patch", seed=seed, fold=fold)
            paths.extend([summary, patch])
    if len(set(paths)) != 2 * len(args.seeds) * args.folds:
        raise ValueError("checkpoint matrix contains duplicates")
    return paths


def validate_audit(path: Path, folds: int) -> dict:
    audit = json.loads(path.read_text())
    if audit.get("contract") != AUDIT_CONTRACT:
        raise ValueError("audit contract differs from the preregistered fixed-family contract")
    if int(audit.get("paired_fold_target_cells", -1)) != folds * len(TARGETS):
        raise ValueError("audit does not cover every fold-target cell")
    gain = float(audit["combined_gain"])
    minimum = float(audit["minimum_preregistered_gain"])
    promoted = bool(audit.get("promote_combined"))
    if promoted != (gain >= minimum):
        raise ValueError("audit promotion flag disagrees with its numeric threshold")
    return audit


def inference_command(args: argparse.Namespace, paths: list[Path]) -> list[str]:
    runtime = args.runtime_json or args.output.with_suffix(".runtime.json")
    command = [
        sys.executable,
        str(Path(__file__).resolve().parent / "kaggle_offline_infer.py"),
        "--data-root",
        str(args.data_root),
        "--dino-model",
        str(args.dino_model),
        "--work-dir",
        str(args.work_dir),
        "--output",
        str(args.output),
        "--runtime-json",
        str(runtime),
        "--batch-size",
        str(args.batch_size),
        "--inference-batch-size",
        str(args.inference_batch_size),
        "--workers",
        str(args.workers),
        "--max-slices",
        str(args.max_slices),
        "--time-budget-hours",
        str(args.time_budget_hours),
    ]
    for path in paths:
        command.extend(["--checkpoint-glob", str(path)])
    return command


def main() -> None:
    args = parse_args()
    audit = validate_audit(args.audit_summary, args.folds)
    decision_path = args.decision_json or args.output.with_suffix(".decision.json")
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "audit_summary": str(args.audit_summary),
        "combined_gain": float(audit["combined_gain"]),
        "minimum_preregistered_gain": float(audit["minimum_preregistered_gain"]),
        "promote_combined": bool(audit["promote_combined"]),
        "submission_uploaded": False,
    }
    if not audit["promote_combined"]:
        record["status"] = "declined_by_preregistered_gate"
        decision_path.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps(record, indent=2))
        return

    paths = audited_checkpoints(args)
    command = inference_command(args, paths)
    record.update(
        {
            "status": "validated_dry_run" if args.dry_run else "approved_for_offline_inference",
            "checkpoint_count": len(paths),
            "checkpoint_paths": [str(path) for path in paths],
            "command": command,
        }
    )
    decision_path.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2), flush=True)
    if args.dry_run:
        return
    subprocess.run(command, check=True)
    record["status"] = "offline_inference_complete"
    record["output"] = str(args.output)
    decision_path.write_text(json.dumps(record, indent=2) + "\n")


if __name__ == "__main__":
    main()
