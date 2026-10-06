"""Fixed-final two-arm audit; engineering checkpoints and partial runs are rejected."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from rsna.schema import TARGETS
from scripts.rank_learning import auc_plan, component_weights, interval, weighted_auc

ROLES = ("control", "candidate")


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def validate_final(
    role,
    run,
    result,
    cpu,
    publication,
    *,
    publication_status="private_original_smooth_witness_epoch_full_download_size_SHA_verified",
):
    if publication_status not in {
        "private_original_smooth_witness_epoch_full_download_size_SHA_verified",
        "private_original_plane_memory_epoch_full_download_size_SHA_verified",
    }:
        raise ValueError("Known fixed-final readout preservation contract required")
    if role not in ROLES or run.get("status") != "completed" or result.get("status") != "completed":
        raise ValueError("Both actual completed fixed-final roles are required")
    binding = run["binding"]
    if (result["optimizer_steps"], result["training_studies"], result["outer_studies"]) != (
        843,
        2247,
        1156,
    ):
        raise ValueError("Exact full fixed3-epoch/843-update cohort required")
    if run["diagnostic_run"] or run["held_out_confirmation_and_gold_used"]:
        raise ValueError("Engineering or protected-evaluation models are not eligible")
    if binding["config"] != cpu["expected_bound_config"][role] or binding["code_sha256"] != {
        n.removeprefix("rsna/"): h for n, h in cpu["source_sha256"].items() if n.startswith("rsna/")
    }:
        raise ValueError("Actual accepted model/source/configuration differs")
    for key in ("training_uids_sha256", "outer_uids_sha256", "folds_sha256"):
        if binding[key] != cpu[key]:
            raise ValueError("Actual frozen membership differs")
    if binding["release"]["binding_sha256"] != cpu["release_binding_sha256"] or result[
        "binding_sha256"
    ] != digest(binding):
        raise ValueError("Final release/model binding differs")
    if (
        publication.get("status") != publication_status
        or publication["file_count"] != 5
        or publication["epoch"] != 3
        or publication["mode"] != role
        or publication["checkpoint_sha256"] != result["checkpoint_sha256"]
    ):
        raise ValueError("Actual fixed-final private whole-byte preservation required")


def paired_metrics(labels, models, *, draws=4000, seed=20261001):
    if type(draws) is not int or draws < 100:
        raise ValueError("At least100 predeclared paired component draws required")
    if set(models) != set(ROLES) or not labels.index.is_unique or labels.leakage_group.isna().any():
        raise ValueError("Aligned unique studies and whole leakage components required")
    if any(
        not frame.index.is_unique or set(frame.index) != set(labels.index)
        for frame in models.values()
    ):
        raise ValueError("Both model outputs require the exact paired study set")
    arrays = {
        role: models[role].loc[labels.index, list(TARGETS)].to_numpy(dtype=float) for role in ROLES
    }
    if any(not np.isfinite(v).all() or (v < 0).any() or (v > 1).any() for v in arrays.values()):
        raise ValueError("Finite probabilities required")
    plans, records = [], []
    for index, target in enumerate(TARGETS):
        masks = labels[target + "__mask"].to_numpy()
        if not np.isin(masks, (0, 1)).all():
            raise ValueError("Original binary observation masks required")
        mask = masks == 1
        truth = labels.loc[mask, target].to_numpy(dtype=int)
        role_plans = {role: auc_plan(truth, values[mask, index]) for role, values in arrays.items()}
        point = {role: weighted_auc(plan, np.ones(len(truth))) for role, plan in role_plans.items()}
        records.append(
            {
                "target": target,
                "auc": point,
                "observed_studies": len(truth),
                "positive_studies": int(truth.sum()),
                "negative_studies": int(len(truth) - truth.sum()),
                "positive_components": len(
                    set(labels.loc[mask & labels[target].eq(1), "leakage_group"])
                ),
                "delta": None
                if any(v is None for v in point.values())
                else point["candidate"] - point["control"],
            }
        )
        plans.append((mask, role_plans))
    macro = {
        role: float(np.mean([r["auc"][role] for r in records]))
        if all(r["auc"][role] is not None for r in records)
        else None
        for role in ROLES
    }
    rng = np.random.default_rng(seed)
    values = [[] for _ in TARGETS]
    macros = []
    draw_sha = hashlib.sha256()
    for _ in range(draws):
        weights = component_weights(labels.leakage_group.to_list(), rng)
        draw_sha.update(weights.astype("<i8").tobytes())
        deltas = []
        for index, (mask, role_plans) in enumerate(plans):
            a, b = (weighted_auc(role_plans[role], weights[mask]) for role in ROLES)
            delta = None if a is None or b is None else b - a
            values[index].append(delta)
            deltas.append(delta)
        macros.append(float(np.mean(deltas)) if all(v is not None for v in deltas) else None)
    for row, samples in zip(records, values, strict=True):
        row["paired_delta_uncertainty"] = interval(samples, draws)
    return {
        "macro_auc": macro,
        "macro_delta": None
        if any(v is None for v in macro.values())
        else macro["candidate"] - macro["control"],
        "macro_delta_uncertainty": interval(macros, draws),
        "primary_meniscus": next(r for r in records if r["target"] == "Medial Meniscus"),
        "targets": records,
        "draws": draws,
        "bootstrap_seed": seed,
        "paired_component_draws_sha256": draw_sha.hexdigest(),
        "components": labels.leakage_group.nunique(),
        "undefined_policy": "retain all12 targets; explicitly count undefined draws, never repair/re-draw them",
        "clinical_or_Kaggle_gain_promoted": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--experiment", choices=("smooth_witness", "plane_memory"), default="smooth_witness"
    )
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve prior results; use an unattempted output path")
    cpu = read(args.audit / "actual-CPU-acceptance.json")
    protocol = read(args.audit / "protocol.json")
    collection = read(args.collection / "collection.json")
    if (
        collection["Slurm"]["state"] != "COMPLETED"
        or collection["Slurm"]["exitcode"] != "0:0"
        or collection["terminal"]["status"] != f"two_original_full_{args.experiment}_arms_completed"
    ):
        raise ValueError("Actual original complete worker/Slurm terminal required")
    if (
        cpu["initial_learned_state"]["control"] != cpu["initial_learned_state"]["candidate"]
        or protocol["scientific_source_git"] != cpu["source_git"]
        or collection["source_git"] != cpu["source_git"]
    ):
        raise ValueError("Predeclared scientific source and matched actual initialization required")
    for name, row in collection["files"].items():
        p = args.collection / name
        if p.stat().st_size != row["bytes"] or sha(p) != row["sha256"]:
            raise ValueError("Original whole collected result bytes differ")
    models = {}
    result_hash = {}
    for role in ROLES:
        run = read(args.collection / (role + "-run.json"))
        result = read(args.collection / (role + "-result.json"))
        publication = read(args.audit / "epochs" / f"{role}-003" / "verified.json")
        validate_final(
            role,
            run,
            result,
            cpu,
            publication,
            publication_status=f"private_original_{args.experiment}_epoch_full_download_size_SHA_verified",
        )
        progress = collection["roles"][role]
        if (
            progress["optimizer_record_count"] != 843
            or not progress["contiguous_optimizer_steps"]
            or not progress["all_progress_finite"]
            or progress["role_terminal"]["returncode"] != 0
            or progress["stderr_bytes"] != 0
        ):
            raise ValueError("Actual complete finite843-step history required")
        path = args.collection / (role + "-outer_predictions.csv")
        if sha(path) != result["predictions_sha256"]:
            raise ValueError("Actual final predictions differ")
        frame = pd.read_csv(path)
        if (
            list(frame.columns) != ["study_uid", "fold", "seed", *TARGETS]
            or len(frame) != 1156
            or frame.study_uid.duplicated().any()
            or not frame.fold.eq(0).all()
            or not frame.seed.eq(20260922).all()
        ):
            raise ValueError("Exact original1156-study MRI-only final output required")
        models[role] = frame.set_index("study_uid")
        result_hash[role] = result
    uids = sorted(models["control"].index)
    if set(uids) != set(models["candidate"].index) or digest(uids) != cpu["outer_uids_sha256"]:
        raise ValueError("Final paired frozen outer membership differs")
    labels = (
        ds.dataset(args.release, format="parquet")
        .to_table(
            columns=[
                "study_uid",
                "leakage_group",
                "split",
                *TARGETS,
                *[t + "__mask" for t in TARGETS],
            ],
            filter=ds.field("study_uid").isin(uids),
        )
        .to_pandas()
        .set_index("study_uid")
        .loc[uids]
    )
    if not labels.split.eq("train").all() or len(labels) != 1156:
        raise ValueError("Only frozen development observations are allowed")
    metrics = paired_metrics(labels, models)
    for role in ROLES:
        if abs(metrics["macro_auc"][role] - result_hash[role]["metrics"]["macro_auc"]) > 1e-9:
            raise ValueError("Independent fixed-final aggregate metric differs")
        for row in metrics["targets"]:
            if (
                abs(
                    row["auc"][role] - result_hash[role]["metrics"]["per_target_auc"][row["target"]]
                )
                > 1e-9
            ):
                raise ValueError("Independent target metric differs")
    metrics.update(
        status=f"actual_original_fixed_final_{args.experiment}_comparison_recomputed",
        source_git=cpu["source_git"],
        protocol_sha256=sha(args.audit / "protocol.json"),
        release_sha256=sha(args.release),
        outer_roster_sha256=cpu["outer_uids_sha256"],
        checkpoint_sha256={r: v["checkpoint_sha256"] for r, v in result_hash.items()},
        predictions_sha256={r: v["predictions_sha256"] for r, v in result_hash.items()},
        gold_or_confirmation_used=False,
        kaggle_score=None,
        primary_endpoint=protocol["primary_endpoint"],
    )
    if args.experiment == "plane_memory":
        if protocol["primary_endpoint"] != "12-target masked macro AUC candidate-minus-control":
            raise ValueError("Predeclared all-target memory endpoint required")
        metrics["medial_meniscus"] = metrics.pop("primary_meniscus")
    with args.output.open("x") as f:
        json.dump(metrics, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    print(
        json.dumps(
            {
                "macro_auc": metrics["macro_auc"],
                "primary_endpoint": metrics["primary_endpoint"],
                "medial_meniscus": metrics.get("primary_meniscus", metrics.get("medial_meniscus")),
                "macro_delta_uncertainty": metrics["macro_delta_uncertainty"],
            }
        )
    )


if __name__ == "__main__":
    main()
