"""Replay native384 fixed-final evidence against the predeclared uniform reference."""

from __future__ import annotations

import csv
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from rsna.schema import TARGETS

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def main():
    runpy.run_path(
        str(ROOT / "initiative/audits/native384-reference-selection-v1/replay.py"),
        run_name="__main__",
    )
    collection = read(AUDIT / "collection.json")
    for name, row in collection["files"].items():
        path = AUDIT / name
        assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]
    result, run = read(AUDIT / "result.json"), read(AUDIT / "run.json")
    initial = read(
        ROOT / "initiative/audits/native384-production-preparation-v1/gpu-initial-progress/run.json"
    )
    assert run["binding"] == initial["binding"]
    assert run["status"] == result["status"] == "completed"
    assert run["diagnostic_run"] is result["diagnostic_run"] is False
    assert run["held_out_confirmation_and_gold_used"] is False
    assert result["checkpoint_selection"] == "fixed_final_epoch_no_outer_selection"
    assert canonical(run["binding"]) == result["binding_sha256"]
    assert result["optimizer_steps"] == 2810 and result["outer_fold"] == 0
    assert result["training_studies"] == 2247 and result["outer_studies"] == 1156
    invocation = read(AUDIT / "invocation.json")
    assert invocation["status"] == "completed" and invocation["exit_code"] == 0
    assert invocation["binding_sha256"] == result["binding_sha256"]
    updates = invocation["updates"]
    assert len(updates) == 2810
    assert [x["optimizer_steps"] for x in updates] == list(range(1, 2811))
    assert all(np.isfinite(x["loss"]) and np.isfinite(x["gradient_norm"]) for x in updates)
    assert all(sum(x["epoch"] == e for x in updates) == 281 for e in range(10))
    assert (AUDIT / "training.err").stat().st_size == 0
    assert "867367|COMPLETED|0:0|" in (AUDIT / "slurm-terminal.txt").read_text()
    inputs = read(AUDIT / "native-input-provenance.json")
    assert inputs["scientific_source_git"] == collection["source_git"]
    assert inputs["strict_deterministic_algorithms"] is True
    assert inputs["input_side"] == inputs["cache_side"] == 384
    assert inputs["config_file_sha256"] == digest(AUDIT / "control/config.json")
    assert inputs["cpu_preflight_sha256"] == digest(AUDIT / "control/cpu-preflight.json")
    assert str(inputs["training_job_id"]) == "867367"
    snap = read(AUDIT / "epoch-checkpoints/epoch-010.json")
    assert snap["checkpoint_sha256"] == result["checkpoint_sha256"]
    assert snap["binding_sha256"] == result["binding_sha256"]
    assert snap["epoch"] == 10 and snap["optimizer_steps"] == 2810
    proof = read(AUDIT / "publication/epoch-010/verified.json")
    assert proof["checkpoint_sha256"] == snap["checkpoint_sha256"]
    assert proof["files_verified_by_download"] == 5
    assert proof["status"] == "private_epoch_fully_verified"
    reference = ROOT / "initiative/audits/sampling-fixed-final-v1/uniform"
    results = {"native384": result, "uniform224": read(reference / "result.json")}
    models = {}
    for role, folder in (("native384", AUDIT), ("uniform224", reference)):
        path = folder / "outer_predictions.csv"
        assert digest(path) == results[role]["predictions_sha256"]
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        models[role] = {r["study_uid"]: r for r in rows}
        assert len(rows) == len(models[role]) == 1156
        assert all(r["fold"] == "0" and r["seed"] == "20260922" for r in rows)
        assert all(0 <= float(r[t]) <= 1 for r in rows for t in TARGETS)
    assert models["native384"].keys() == models["uniform224"].keys()
    outer = sorted(models["native384"])
    assert canonical(outer) == run["binding"]["outer_uids_sha256"]
    label_path = ROOT / "initiative/releases/report-supervision-v1/study_labels.csv"
    prior = read(ROOT / "initiative/audits/sampling-fixed-final-v1/comparison.json")
    assert digest(label_path) == prior["labels_sha256"]
    with label_path.open(newline="") as stream:
        labels = {r["study_uid"]: r for r in csv.DictReader(stream) if r["study_uid"] in set(outer)}
    assert labels.keys() == models["native384"].keys()
    assert all(r["split"] == "train" and r["leakage_group"] for r in labels.values())
    library = runpy.run_path(str(ROOT / "scripts/rank_learning.py"))
    groups = [labels[u]["leakage_group"] for u in outer]
    plans, targets = [], []
    for target in TARGETS:
        mask = np.array([float(labels[u][target + "__mask"]) == 1 for u in outer])
        selected = [u for u, take in zip(outer, mask, strict=True) if take]
        truth = [int(float(labels[u][target])) for u in selected]
        scores, pair = {}, []
        for role in ("uniform224", "native384"):
            values = [float(models[role][u][target]) for u in selected]
            scores[role] = float(roc_auc_score(truth, values))
            assert abs(scores[role] - results[role]["metrics"]["per_target_auc"][target]) < 1e-9
            assert len(selected) == results[role]["metrics"]["observed_studies"][target]
            pair.append(library["auc_plan"](truth, values))
        plans.append((mask, pair))
        targets.append(
            dict(
                target=target,
                **scores,
                delta=scores["native384"] - scores["uniform224"],
                observed_studies=len(truth),
                positive_studies=sum(truth),
                negative_studies=len(truth) - sum(truth),
                positive_components=len(
                    {labels[u]["leakage_group"] for u, y in zip(selected, truth, strict=True) if y}
                ),
            )
        )
    macro = {r: float(np.mean([row[r] for row in targets])) for r in models}
    assert all(abs(macro[r] - results[r]["metrics"]["macro_auc"]) < 1e-9 for r in models)
    draws, seed = 4000, 20261001
    rng = np.random.default_rng(seed)
    per_target, deltas = [[] for _ in TARGETS], []
    for _ in range(draws):
        weights = library["component_weights"](groups, rng)
        differences = []
        for i, (mask, pair) in enumerate(plans):
            values = [library["weighted_auc"](p, weights[mask]) for p in pair]
            delta = None if any(v is None for v in values) else values[1] - values[0]
            per_target[i].append(delta)
            differences.append(delta)
        deltas.append(
            float(np.mean(differences)) if all(v is not None for v in differences) else None
        )
    record = dict(
        status="fixed_final_native384_recomputed_against_predeclared_uniform_reference",
        native_source_git=collection["source_git"],
        native_job=867367,
        native_macro_auc=macro["native384"],
        uniform_macro_auc=macro["uniform224"],
        native_minus_uniform=macro["native384"] - macro["uniform224"],
        studies=1156,
        training_studies=2247,
        epochs=10,
        optimizer_steps=2810,
        checkpoint_sha256=result["checkpoint_sha256"],
        native_predictions_sha256=result["predictions_sha256"],
        reference_predictions_sha256=results["uniform224"]["predictions_sha256"],
        labels_sha256=digest(label_path),
        macro_delta_uncertainty=library["interval"](deltas, draws),
        bootstrap_draws=draws,
        bootstrap_seed=seed,
        bootstrap_method="paired whole-leakage-component resampling with replacement",
        undefined_macro_policy="undefined if any of all12 targets lacks either class; no redrawing/target dropping",
        uncertainty_caveat="Conditional on defined draws in one weak-label development pool; no cross-seed/fold evidence.",
        causal_caveat="Native initial-state equality was not collected and independently reconstructed caches differ.",
        targets=[
            dict(row, paired_delta_uncertainty=library["interval"](values, draws))
            for row, values in zip(targets, per_target, strict=True)
        ],
        accepted_masks_or_folds_changed=False,
        protected_gold_or_confirmation_used=False,
        kaggle_score=None,
    )
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output = AUDIT / "comparison.json"
    if output.exists():
        assert output.read_text() == text
    else:
        output.write_text(text)
    print(
        json.dumps(
            {
                k: record[k]
                for k in [
                    "native_macro_auc",
                    "uniform_macro_auc",
                    "native_minus_uniform",
                    "macro_delta_uncertainty",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
