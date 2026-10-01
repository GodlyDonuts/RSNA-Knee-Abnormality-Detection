"""Replay matched fixed-final sampling results and paired component uncertainty."""

from __future__ import annotations

import csv
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from rsna.schema import TARGETS

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
SOURCE = "7ac7a9779d3c8bfd756a44eceb6414e9bb763b85"
LABELS = ROOT / "initiative/releases/report-supervision-v1/study_labels.csv"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def main():
    originals = json.loads((AUDIT / "collection.json").read_text())
    for name, row in originals["files"].items():
        path = AUDIT / name
        assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]
    roles = ("uniform", "stratified")
    models, results, bindings, initializations = {}, {}, {}, []
    for role, job in (("uniform", 866719), ("stratified", 866716)):
        folder = AUDIT / role
        result = results[role] = json.loads((folder / "result.json").read_text())
        run = json.loads((folder / "run.json").read_text())
        binding = bindings[role] = run["binding"]
        assert canonical(binding) == result["binding_sha256"]
        assert result["status"] == run["status"] == "completed"
        assert result["diagnostic_run"] is run["diagnostic_run"] is False
        assert run["held_out_confirmation_and_gold_used"] is False
        assert result["outer_fold"] == 0 and result["optimizer_steps"] == 2810
        assert result["outer_studies"] == 1156 and result["training_studies"] == 2247
        assert result["checkpoint_selection"] == "fixed_final_epoch_no_outer_selection"
        snapshot = json.loads((folder / "epoch-checkpoints/epoch-010.json").read_text())
        assert snapshot["epoch"] == 10 and snapshot["optimizer_steps"] == 2810
        assert snapshot["checkpoint_sha256"] == result["checkpoint_sha256"]
        assert snapshot["binding_sha256"] == result["binding_sha256"]
        assert snapshot["release_binding_sha256"] == binding["release"]["binding_sha256"]
        source = json.loads((folder / "control/source.json").read_text())
        provenance = json.loads((folder / "sampling-input-provenance.json").read_text())
        assert source["source_git"] == provenance["source_git"] == SOURCE
        assert source["files"] == provenance["source_files"]
        assert digest(folder / "control/source.json") == provenance["source_manifest_sha256"]
        assert digest(folder / "control/files.json") == provenance["payload_manifest_sha256"]
        assert digest(folder / f"control/{role}-config.json") == provenance["config_sha256"]
        assert provenance["strict_determinism"] is True
        assert provenance["accepted_data_masks_or_folds_changed"] is False
        assert provenance["base_weights_reused_not_diagnostic_checkpoint"] is True
        assert str(provenance["job_id"]) == str(job)
        init = json.loads((folder / f"control/{role}-initialization-{job}.json").read_text())
        assert init == provenance["initialization"]
        initializations.append(init)
        for name, value in binding["code_sha256"].items():
            assert source["files"]["rsna/" + name]["sha256"] == value
        assert (
            digest(folder / f"control/node-header-verified-{job}.json")
            == provenance["GPU_header_receipt_sha256"]
        )
        terminal = json.loads((folder / f"control/{role}-terminal-{job}.json").read_text())
        assert terminal["child_exit"] == 0 and terminal["stop_requested"] is False
        assert terminal["sampling_input_provenance_sha256"] == digest(
            folder / "sampling-input-provenance.json"
        )
        assert f"{job}|COMPLETED|0:0|" in (AUDIT / "slurm-terminal.txt").read_text()
        invocation = json.loads((folder / "invocation.json").read_text())
        assert invocation["binding_sha256"] == result["binding_sha256"]
        assert invocation["exit_code"] == 0 and invocation["status"] == "completed"
        updates = invocation["updates"]
        assert len(updates) == 2810
        assert [x["optimizer_steps"] for x in updates] == list(range(1, 2811))
        assert all(np.isfinite(x["loss"]) and np.isfinite(x["gradient_norm"]) for x in updates)
        assert all(sum(x["epoch"] == e for x in updates) == 281 for e in range(10))
        path = folder / "outer_predictions.csv"
        assert digest(path) == result["predictions_sha256"]
        with path.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        model = models[role] = {row["study_uid"]: row for row in rows}
        assert len(rows) == len(model) == 1156
        assert all(r["fold"] == "0" and r["seed"] == "20260922" for r in rows)
        assert all(0 <= float(r[t]) <= 1 for r in rows for t in TARGETS)
        assert canonical(sorted(model)) == binding["outer_uids_sha256"]
    first, second = (bindings[r] for r in roles)
    assert {k for k in first if first[k] != second[k]} == {"config"}
    assert {k for k in first["config"] if first["config"][k] != second["config"][k]} == {
        "slice_sampling"
    }
    assert [bindings[r]["config"]["slice_sampling"] for r in roles] == list(roles)
    assert initializations[0]["initial_state"] == initializations[1]["initial_state"]
    assert (
        initializations[0]["initial_logits_sha256"] == initializations[1]["initial_logits_sha256"]
    )
    assert models["uniform"].keys() == models["stratified"].keys()
    folds = pd.read_csv(ROOT / "initiative/experiments/competitive-v1/folds/fold.csv")
    assert canonical(folds.sort_values("study_uid").to_dict("records")) == first["folds_sha256"]
    with LABELS.open(newline="") as stream:
        development = {r["study_uid"]: r for r in csv.DictReader(stream) if r["split"] == "train"}
    assert set(development) == set(folds.study_uid)
    observed = {
        u for u, r in development.items() if any(float(r[t + "__mask"]) == 1 for t in TARGETS)
    }
    eligible = folds.loc[folds.study_uid.isin(observed)]
    outer = sorted(eligible.loc[eligible["fold"] == 0, "study_uid"])
    train = sorted(eligible.loc[eligible["fold"] != 0, "study_uid"])
    assert not {development[u]["leakage_group"] for u in train} & {
        development[u]["leakage_group"] for u in outer
    }
    assert len(outer) == 1156 and len(train) == 2247 and not set(outer) & set(train)
    assert outer == sorted(models["uniform"])
    assert canonical(train) == first["training_uids_sha256"]
    prior = json.loads(
        (ROOT / "initiative/audits/dinov3-adapter-a32-fold0-result-v1/comparison.json").read_text()
    )
    assert digest(LABELS) == prior["labels_sha256"]
    with LABELS.open(newline="") as stream:
        labels = {r["study_uid"]: r for r in csv.DictReader(stream) if r["study_uid"] in set(outer)}
    assert labels.keys() == models["uniform"].keys()
    assert all(r["split"] == "train" and r["leakage_group"] for r in labels.values())
    library = runpy.run_path(str(ROOT / "scripts/rank_learning.py"))
    groups = [labels[u]["leakage_group"] for u in outer]
    plans, targets = [], []
    for target in TARGETS:
        mask = np.array([float(labels[u][target + "__mask"]) == 1 for u in outer])
        selected = [u for u, take in zip(outer, mask, strict=True) if take]
        truth = [int(float(labels[u][target])) for u in selected]
        scores, pair = {}, []
        for role in roles:
            values = [float(models[role][u][target]) for u in selected]
            scores[role] = float(roc_auc_score(truth, values))
            assert abs(scores[role] - results[role]["metrics"]["per_target_auc"][target]) < 1e-9
            assert len(selected) == results[role]["metrics"]["observed_studies"][target]
            plan = library["auc_plan"](truth, values)
            assert abs(library["weighted_auc"](plan, np.ones(len(selected))) - scores[role]) < 1e-12
            pair.append(plan)
        plans.append((mask, pair))
        targets.append(
            dict(
                target=target,
                **scores,
                delta=scores["stratified"] - scores["uniform"],
                observed_studies=len(truth),
                positive_studies=sum(truth),
                negative_studies=len(truth) - sum(truth),
                positive_components=len(
                    {labels[u]["leakage_group"] for u, y in zip(selected, truth, strict=True) if y}
                ),
                negative_components=len(
                    {
                        labels[u]["leakage_group"]
                        for u, y in zip(selected, truth, strict=True)
                        if not y
                    }
                ),
            )
        )
    macro = {r: float(np.mean([row[r] for row in targets])) for r in roles}
    assert all(abs(macro[r] - results[r]["metrics"]["macro_auc"]) < 1e-9 for r in roles)
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
        status="matched_fixed_final_sampling_recomputed_with_paired_component_uncertainty",
        source_git=SOURCE,
        uniform_macro_auc=macro["uniform"],
        stratified_macro_auc=macro["stratified"],
        stratified_minus_uniform=macro["stratified"] - macro["uniform"],
        studies=1156,
        training_studies=2247,
        epochs=10,
        optimizer_updates_per_role=2810,
        components=len(set(groups)),
        labels_sha256=digest(LABELS),
        uniform_predictions_sha256=results["uniform"]["predictions_sha256"],
        stratified_predictions_sha256=results["stratified"]["predictions_sha256"],
        equal_initial_learned_states_and_logits=True,
        recipe_difference_only="slice_sampling during training",
        evaluation_sampling="uniform",
        macro_delta_uncertainty=library["interval"](deltas, draws),
        bootstrap_draws=draws,
        bootstrap_seed=seed,
        bootstrap_method="unstratified paired whole-leakage-component resampling with replacement",
        undefined_macro_policy="undefined whenever any of twelve targets lacks either class; no redrawing or target dropping",
        uncertainty_caveat="Intervals condition on defined draws and one weak-label development pool; no cross-seed/fold evidence.",
        targets=[
            dict(row, paired_delta_uncertainty=library["interval"](values, draws))
            for row, values in zip(targets, per_target, strict=True)
        ],
        protected_gold_or_confirmation_used=False,
        kaggle_score=None,
        accepted_masks_or_folds_changed=False,
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
                    "uniform_macro_auc",
                    "stratified_macro_auc",
                    "stratified_minus_uniform",
                    "macro_delta_uncertainty",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
