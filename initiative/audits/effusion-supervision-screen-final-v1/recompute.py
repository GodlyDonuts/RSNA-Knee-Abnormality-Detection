"""Replay the fixed three-epoch Effusion contrast and paired component uncertainty."""

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
SCREEN = ROOT / "initiative/audits/effusion-supervision-screen-v1"


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def updates(path):
    records = []
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if "optimizer_steps" in row:
            records.append(row)
    return records


def main():
    plan = read(SCREEN / "final-analysis-plan.json")
    assert plan["primary"] == "candidate_minus_control_Effusion_outer_AUC"
    assert plan["epochs"] == 3 and plan["optimizer_steps"] == 843
    assert plan["bootstrap_draws"] == 4000 and plan["bootstrap_seed"] == 20261001
    terminal = {
        fields[0]: fields[1:]
        for line in (AUDIT / "slurm-terminal.txt").read_text().splitlines()
        if (fields := line.split("|"))
    }
    for job in ("869815", "869816", "869954", "869957", "869966"):
        assert terminal[job][:2] == ["COMPLETED", "0:0"]
    assert terminal["869956"][:2] == ["FAILED", "1:0"]
    decision = read(AUDIT / "validation-decision.json")
    assert decision["original_analysis_plan_sha256"] == digest(SCREEN / "final-analysis-plan.json")
    assert decision["training_or_predictions_rerun"] is False
    assert decision["epochs_updates_primary_bootstrap_labels_or_masks_changed"] is False
    collection = read(AUDIT / "collection.json")
    assert collection["scientific_source_git"] == plan["scientific_source_git"]
    for name, row in collection["files"].items():
        path = AUDIT / name
        assert path.stat().st_size == row["bytes"] and digest(path) == row["sha256"]
    initial = read(SCREEN / "evc32-independent-jobs/gpu-progress-002.json")
    results, models, bindings = {}, {}, {}
    numeric = ("epoch", "update", "optimizer_steps", "loss", "gradient_norm", "observed_cells")
    for role in ("control", "candidate"):
        folder = AUDIT / role
        run, result, invocation = (
            read(folder / (name + ".json")) for name in ("run", "result", "invocation")
        )
        binding = run["binding"]
        assert binding == initial["roles"][role]["run"]["binding"]
        assert run["status"] == result["status"] == invocation["status"] == "completed"
        assert invocation["exit_code"] == 0
        assert canonical(binding) == result["binding_sha256"] == invocation["binding_sha256"]
        assert result["optimizer_steps"] == 843 and result["outer_fold"] == 0
        assert result["training_studies"] == 2247 and result["outer_studies"] == 1156
        assert (
            result["diagnostic_run"] is False
            and run["held_out_confirmation_and_gold_used"] is False
        )
        assert result["checkpoint_selection"] == "fixed_final_epoch_no_outer_selection"
        policy = binding["experimental_fitting_supervision"]
        assert policy["enabled"] is (role == "candidate")
        assert policy["applied_cells"] == (31 if role == "candidate" else 0)
        assert policy["evaluation_masks_changed"] is False and policy["experimental_only"] is True
        original = updates(folder / "original-training.out")
        resumed = updates(folder / "resumed-training.out")
        assert [r["optimizer_steps"] for r in resumed] == list(range(176, 844))
        assert invocation["updates"] == resumed
        joined = [r for r in original if r["optimizer_steps"] <= 175] + resumed
        assert [r["optimizer_steps"] for r in joined] == list(range(1, 844))
        assert all(np.isfinite(r["loss"]) and np.isfinite(r["gradient_norm"]) for r in joined)
        assert all(sum(r["epoch"] == epoch for r in joined) == 281 for epoch in range(3))
        old, new = ({r["optimizer_steps"]: r for r in rows} for rows in (original, resumed))
        overlap = sorted(set(old) & set(new))
        assert overlap == list(range(176, 184 if role == "control" else 183))
        assert all(
            {k: old[n][k] for k in numeric} == {k: new[n][k] for k in numeric} for n in overlap
        )
        assert (folder / "resumed-training.err").stat().st_size == 0
        assert "DUE TO PREEMPTION" in (folder / "original-training.err").read_text()
        snapshot = read(folder / "epoch-checkpoints/epoch-003.json")
        proof = read(SCREEN / f"{role}-epoch-003-verified.json")
        assert snapshot["epoch"] == 3 and snapshot["optimizer_steps"] == 843
        assert (
            snapshot["checkpoint_sha256"]
            == result["checkpoint_sha256"]
            == proof["checkpoint_sha256"]
        )
        assert (
            proof["status"] == "private_epoch_fully_verified"
            and proof["files_verified_by_download"] == 5
        )
        path = folder / "outer_predictions.csv"
        assert digest(path) == result["predictions_sha256"]
        with path.open(newline="") as f:
            rows = list(csv.DictReader(f))
        predictions = {r["study_uid"]: r for r in rows}
        assert len(rows) == len(predictions) == 1156
        assert all(r["fold"] == "0" and r["seed"] == "20260922" for r in rows)
        assert all(
            np.isfinite(float(r[t])) and 0 <= float(r[t]) <= 1 for r in rows for t in TARGETS
        )
        assert canonical(sorted(predictions)) == binding["outer_uids_sha256"]
        results[role], models[role], bindings[role] = result, predictions, binding
    assert models["control"].keys() == models["candidate"].keys()
    for name in (
        "outer_uids_sha256",
        "training_uids_sha256",
        "release",
        "folds_sha256",
        "code_sha256",
        "targets",
    ):
        assert bindings["control"][name] == bindings["candidate"][name]
    outer = sorted(models["control"])
    allowed = set(outer)
    labels_path = ROOT / "initiative/releases/report-supervision-v1/study_labels.csv"
    reference = read(ROOT / "initiative/audits/sampling-fixed-final-v1/comparison.json")
    assert digest(labels_path) == reference["labels_sha256"]
    with labels_path.open(newline="") as f:
        labels = {r["study_uid"]: r for r in csv.DictReader(f) if r["study_uid"] in allowed}
    assert labels.keys() == models["control"].keys()
    assert all(r["split"] == "train" and r["leakage_group"] for r in labels.values())
    library = runpy.run_path(str(ROOT / "scripts/rank_learning.py"))
    groups = [labels[u]["leakage_group"] for u in outer]
    targets, plans = [], []
    for target in TARGETS:
        mask = np.array([float(labels[u][target + "__mask"]) == 1 for u in outer])
        selected = [u for u, take in zip(outer, mask, strict=True) if take]
        truth = [int(float(labels[u][target])) for u in selected]
        scores, pair = {}, []
        for role in ("control", "candidate"):
            probabilities = [float(models[role][u][target]) for u in selected]
            scores[role] = float(roc_auc_score(truth, probabilities))
            assert abs(scores[role] - results[role]["metrics"]["per_target_auc"][target]) < 1e-9
            assert len(selected) == results[role]["metrics"]["observed_studies"][target]
            pair.append(library["auc_plan"](truth, probabilities))
        plans.append((mask, pair))
        targets.append(
            dict(
                target=target,
                **scores,
                delta=scores["candidate"] - scores["control"],
                observed_studies=len(truth),
                positive_studies=sum(truth),
                negative_studies=len(truth) - sum(truth),
                positive_components=len(
                    {labels[u]["leakage_group"] for u, y in zip(selected, truth, strict=True) if y}
                ),
            )
        )
    macro = {role: float(np.mean([r[role] for r in targets])) for role in models}
    assert all(abs(macro[role] - results[role]["metrics"]["macro_auc"]) < 1e-9 for role in models)
    draws, seed = plan["bootstrap_draws"], plan["bootstrap_seed"]
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
    target_records = [
        dict(row, paired_delta_uncertainty=library["interval"](values, draws))
        for row, values in zip(targets, per_target, strict=True)
    ]
    primary = next(row for row in target_records if row["target"] == "Effusion")
    record = dict(
        status="fixed_final_effusion_supervision_contrast_independently_recomputed",
        primary=primary,
        control_macro_auc=macro["control"],
        candidate_macro_auc=macro["candidate"],
        macro_delta=macro["candidate"] - macro["control"],
        macro_delta_uncertainty=library["interval"](deltas, draws),
        targets=target_records,
        bootstrap_draws=draws,
        bootstrap_seed=seed,
        bootstrap_method=plan["bootstrap_method"],
        undefined_policy=plan["undefined_policy"],
        studies=1156,
        training_studies=2247,
        epochs=3,
        optimizer_steps=843,
        scientific_source_git=collection["scientific_source_git"],
        labels_sha256=digest(labels_path),
        analysis_plan_sha256=digest(SCREEN / "final-analysis-plan.json"),
        checkpoint_SHA={role: results[role]["checkpoint_sha256"] for role in models},
        prediction_SHA={role: results[role]["predictions_sha256"] for role in models},
        trajectory_semantics="join immutable original updates1..175 with resumed invocation176..843; exact replayed overlap; original per-invocation records unchanged",
        accepted_masks_or_folds_changed=False,
        protected_gold_or_confirmation_used=False,
        kaggle_score=None,
        caveat=plan["caveat"],
        replication_required_before_gain_promotion=True,
    )
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output = AUDIT / "comparison.json"
    if output.exists():
        assert output.read_text() == text
    else:
        with output.open("x") as f:
            f.write(text)
    print(
        json.dumps(
            {
                "primary": primary,
                "control_macro_auc": macro["control"],
                "candidate_macro_auc": macro["candidate"],
                "macro_delta": record["macro_delta"],
                "macro_delta_uncertainty": record["macro_delta_uncertainty"],
            }
        )
    )


if __name__ == "__main__":
    main()
