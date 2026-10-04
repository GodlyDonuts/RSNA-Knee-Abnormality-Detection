"""Replay fitting-only comparison support; never decode MRI or score a model."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rsna.competitive.sampling import EffectiveBatchSampler
from rsna.schema import TARGETS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def fitting_inputs(
    studies: Path, folds_path: Path, original_run: Path
) -> tuple[pd.DataFrame, dict]:
    """Project masks for membership, then decode target values for fitting UIDs only."""
    run = json.loads(original_run.read_text())
    binding = run["binding"]
    config = binding["config"]
    if (
        run["status"] != "completed"
        or config["fit_scope"] != "outer_fold"
        or config["diagnostic_study_limit"] is not None
        or binding["world_size"] != 1
        or binding["torch_version"].split("+")[0] != torch.__version__.split("+")[0]
        or binding["targets"] != list(TARGETS)
        or binding.get("experimental_fitting_supervision", {}).get("enabled") is not False
        or binding["experimental_fitting_supervision"]["applied_cells"] != 0
        or digest(studies) != binding["release"]["manifest_sha256"]
    ):
        raise ValueError(
            "Exact completed unmodified control release and full fitting scope required"
        )
    sampler_path = Path(inspect.getfile(EffectiveBatchSampler))
    if digest(sampler_path) != binding["code_sha256"]["competitive/sampling.py"]:
        raise ValueError("Actual original sampler changed")
    folds = pd.read_csv(folds_path)
    if (
        folds.study_uid.duplicated().any()
        or canonical(folds.sort_values("study_uid").to_dict("records")) != binding["folds_sha256"]
    ):
        raise ValueError("Exact original full fold membership required")
    mask_columns = [f"{target}__mask" for target in TARGETS]
    metadata = pd.read_parquet(
        studies,
        columns=[
            "study_uid",
            "split",
            "leakage_group",
            "report_supervision_allowed",
            *mask_columns,
        ],
        filters=[("split", "==", "train")],
    )
    if metadata.study_uid.duplicated().any() or set(metadata.study_uid) != set(folds.study_uid):
        raise ValueError("Original accepted development membership changed")
    joined = metadata.merge(folds[["study_uid", "fold"]], on="study_uid", validate="one_to_one")
    observed = joined[mask_columns].sum(axis=1) > 0
    fitting = joined[observed & (joined.fold != config["fold"])].sort_values("study_uid")
    outer = joined[observed & (joined.fold == config["fold"])].sort_values("study_uid")
    if (
        fitting.empty
        or outer.empty
        or canonical(fitting.study_uid.tolist()) != binding["training_uids_sha256"]
        or canonical(outer.study_uid.tolist()) != binding["outer_uids_sha256"]
        or set(fitting.leakage_group) & set(outer.leakage_group)
        or not fitting.report_supervision_allowed.eq(True).all()  # noqa: E712
    ):
        raise ValueError("Exact fitting/outer separation and original roster bindings required")
    # No outer, confirmation or official-gold target values are decoded here.
    labels = pd.read_parquet(
        studies,
        columns=["study_uid", *TARGETS],
        filters=[("split", "==", "train"), ("study_uid", "in", fitting.study_uid.tolist())],
    )
    if labels.study_uid.duplicated().any() or set(labels.study_uid) != set(fitting.study_uid):
        raise ValueError("Fitting-only target projection changed membership")
    frame = fitting.merge(labels, on="study_uid", validate="one_to_one").reset_index(drop=True)
    return frame, {
        "release_sha256": digest(studies),
        "original_completed_run_sha256": digest(original_run),
        "folds_sha256": binding["folds_sha256"],
        "training_uids_sha256": binding["training_uids_sha256"],
        "outer_uids_sha256": binding["outer_uids_sha256"],
        "sampler_sha256": digest(sampler_path),
        "CPU_torch_version": str(torch.__version__),
        "original_torch_version": binding["torch_version"],
        "fitting_studies": len(frame),
        "outer_roster_studies": len(outer),
        "outer_confirmation_gold_target_values_decoded": False,
        "batch_size": config["batch_size"],
        "accumulation_steps": config["accumulation_steps"],
        "world_size": 1,
        "epochs": config["epochs"],
        "parent_seed": config["seed"],
        "original_masks_and_folds_changed": False,
    }


def comparison_pairs(
    positive: np.ndarray, negative: np.ndarray, groups: np.ndarray
) -> tuple[int, int]:
    """Count strict-polarity comparisons; same-component contrasts stay excluded."""
    matrix = positive[:, None] & negative[None, :]
    same = groups[:, None] == groups[None, :]
    return int((matrix & ~same).sum()), int((matrix & same).sum())


def summarize_batches(
    labels: np.ndarray,
    masks: np.ndarray,
    groups: np.ndarray,
    *,
    batch_size: int,
    accumulation_steps: int,
    seed: int,
    epochs: int,
) -> dict:
    """Summarize the exact ordered batches without changing their sampling policy."""
    if (
        labels.ndim != 2
        or masks.shape != labels.shape
        or groups.shape != labels.shape[:1]
        or labels.shape[1] != len(TARGETS)
        or not len(labels)
        or epochs < 1
        or not np.isfinite(masks).all()
        or not np.isin(masks, [0, 1]).all()
        or not np.isfinite(labels[masks > 0]).all()
        or not np.isin(labels[masks > 0], [0, 1]).all()
    ):
        raise ValueError("Aligned original binary observed labels/masks and components required")
    positive = (masks > 0) & (labels == 1)
    negative = (masks > 0) & (labels == 0)
    per_epoch = []
    observed_cells_sequence = []
    for epoch in range(epochs):
        sampler = EffectiveBatchSampler(
            len(labels), batch_size, accumulation_steps, seed=seed, epoch=epoch
        )
        microbatches = iter(sampler)
        seen = []
        counters = np.zeros((len(TARGETS), 7), dtype=np.int64)
        for _ in range(sampler.updates):
            effective_indices = []
            for _ in range(accumulation_steps):
                indices = [index for index, valid, _ in next(microbatches) if valid]
                seen.extend(indices)
                effective_indices.extend(indices)
                for target in range(len(TARGETS)):
                    pairs, _ = comparison_pairs(
                        positive[indices, target], negative[indices, target], groups[indices]
                    )
                    counters[target, 6] += pairs
            observed_cells_sequence.append(int(masks[effective_indices].sum()))
            for target in range(len(TARGETS)):
                pos, neg = positive[effective_indices, target], negative[effective_indices, target]
                pairs, excluded = comparison_pairs(pos, neg, groups[effective_indices])
                counters[target] += np.array(
                    [
                        int(pos.any()),
                        int(neg.any() and not pos.any()),
                        int((pos | neg).any()),
                        int(pairs > 0),
                        pairs,
                        excluded,
                        0,
                    ]
                )
        if sorted(seen) != list(range(len(labels))) or next(microbatches, None) is not None:
            raise ValueError("Original sampler must expose every real fitting row once per epoch")
        per_epoch.append(
            {
                "epoch": epoch + 1,
                "updates": sampler.updates,
                "real_fitting_rows": len(seen),
                "targets": {
                    target: dict(
                        zip(
                            [
                                "updates_with_positive",
                                "updates_negative_only",
                                "updates_observed",
                                "updates_with_cross_component_pair",
                                "cross_component_pairs",
                                "same_component_pairs_excluded",
                                "within_microbatch_pairs",
                            ],
                            map(int, counters[t]),
                            strict=True,
                        )
                    )
                    for t, target in enumerate(TARGETS)
                },
            }
        )
    updates = sum(row["updates"] for row in per_epoch)
    return {
        "seed": seed,
        "epochs": epochs,
        "optimizer_updates_replayed": updates,
        "effective_batch": batch_size * accumulation_steps,
        "microbatch": batch_size,
        "observed_cells_sequence_sha256": canonical(observed_cells_sequence),
        "per_epoch": per_epoch,
        "targets": {
            target: {
                "fitting_observed_positive": int(positive[:, t].sum()),
                "fitting_observed_negative": int(negative[:, t].sum()),
                **{
                    key: sum(epoch["targets"][target][key] for epoch in per_epoch)
                    for key in per_epoch[0]["targets"][target]
                },
                "fraction_updates_with_effective_batch_pair": sum(
                    epoch["targets"][target]["updates_with_cross_component_pair"]
                    for epoch in per_epoch
                )
                / updates,
            }
            for t, target in enumerate(TARGETS)
        },
    }


def verify_original_observed_counts(summary: dict, original: Path, resumed: Path) -> dict:
    """Check CPU replay against the actual uninterrupted/resumed GPU update logs."""

    def rows(path: Path) -> dict[int, dict]:
        result = {}
        for line in path.read_text().splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if not isinstance(value, dict) or "optimizer_steps" not in value:
                continue
            step = value["optimizer_steps"]
            if type(step) is not int or step < 1 or step in result:
                raise ValueError("Duplicate or invalid original training counter")
            result[step] = value
        if not result:
            raise ValueError("Actual training update records required")
        return result

    old, new = rows(original), rows(resumed)
    if any(old[step]["observed_cells"] != new[step]["observed_cells"] for step in old.keys() & new):
        raise ValueError("Observed-mask interruption overlap changed")
    start = min(new)
    joined = {step: row for step, row in old.items() if step < start} | new
    if sorted(joined) != list(range(1, summary["optimizer_updates_replayed"] + 1)):
        raise ValueError("Original and resumed counters do not cover the exact replay")
    counts = [joined[step]["observed_cells"] for step in sorted(joined)]
    if any(not np.isfinite(n) or n < 0 or n != int(n) for n in counts):
        raise ValueError("Original observed-cell counters must be finite nonnegative integers")
    if canonical([int(n) for n in counts]) != summary["observed_cells_sequence_sha256"]:
        raise ValueError("CPU sampler replay differs from actual GPU observed-mask sequence")
    return {
        "original_training_log_sha256": digest(original),
        "resumed_training_log_sha256": digest(resumed),
        "all_actual_GPU_update_observed_cells_match_CPU_replay": True,
        "updates_verified": len(counts),
        "overlap_updates_verified": len(old.keys() & new),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studies", type=Path, required=True)
    parser.add_argument("--folds", type=Path, required=True)
    parser.add_argument("--original-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve original diagnostic output; do not overwrite")
    frame, binding = fitting_inputs(args.studies, args.folds, args.original_run)
    labels = frame[list(TARGETS)].to_numpy()
    masks = frame[[f"{target}__mask" for target in TARGETS]].to_numpy()
    groups = frame.leakage_group.to_numpy()
    result = {
        "status": "original_fitting_only_sampler_comparison_support_replayed",
        "binding": binding,
        "seeds": [
            summarize_batches(
                labels,
                masks,
                groups,
                batch_size=binding["batch_size"],
                accumulation_steps=binding["accumulation_steps"],
                seed=seed,
                epochs=binding["epochs"],
            )
            for seed in [binding["parent_seed"], 20261004]
        ],
        "labels_are_report_derived_not_MRI_consensus": True,
        "concept_or_partial_order_labels_admitted": False,
        "MRI_model_optimizer_or_GPU_run": False,
        "accuracy_or_root_cause_claimed": False,
        "ranking_objective_implemented_or_accepted": False,
    }
    result["actual_parent_training_mask_sequence_verification"] = verify_original_observed_counts(
        result["seeds"][0],
        args.original_run.parent / "original-training.out",
        args.original_run.parent / "resumed-training.out",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as out:
        json.dump(result, out, indent=2, sort_keys=True, allow_nan=False)
        out.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "fitting_studies": len(frame),
                "seeds": [item["seed"] for item in result["seeds"]],
            }
        )
    )


if __name__ == "__main__":
    main()
