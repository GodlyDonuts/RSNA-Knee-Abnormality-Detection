"""Replay the original fixed-final neighbor experiment without selecting a recipe."""

from __future__ import annotations

import csv
import hashlib
import json
import runpy
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
from sklearn.metrics import roc_auc_score

from rsna.schema import TARGETS

ROOT = Path(__file__).resolve().parents[4]
AUDIT = Path(__file__).resolve().parent
PARENT = AUDIT.parent
GRAY = ROOT / "initiative/audits/effusion-supervision-screen-final-v1/control"
PHYSICAL = PARENT / "physical-fixed-final-v1"
CONTRASTS = {
    "physical_minus_gray": ("gray", "physical"),
    "physical_minus_distant": ("distant", "physical"),
    "distant_minus_gray": ("gray", "distant"),
}


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def main():
    cpu = read(PARENT / "actual-CPU-acceptance.json")
    collection = read(AUDIT / "collection.json")
    terminal = read(AUDIT / "terminal-collection.json")
    old_progress = read(PARENT / "progress-1319.json")
    assert collection["source_git"] == terminal["source_git"] == cpu["source_git"]
    assert terminal["source_and_config_bindings_verified"]
    assert terminal["Slurm"]["state"] == "COMPLETED"
    assert terminal["Slurm"]["exitcode"] == "0:0"
    assert terminal["terminal"]["status"] == "three_original_full_neighbor_arms_completed"
    assert terminal["GPU_stderr_bytes"] == 0
    assert cpu["initial_learned_state"]["gray"] == cpu["initial_learned_state"]["physical"]
    assert cpu["initial_learned_state"]["gray"] == cpu["initial_learned_state"]["distant"]
    for name, row in collection["files"].items():
        path = AUDIT / name
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]
    for name, row in read(PHYSICAL / "collection.json")["files"].items():
        path = PHYSICAL / name
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]
    paths = {
        "gray": (
            AUDIT / "gray-run.json",
            AUDIT / "gray-result.json",
            GRAY / "outer_predictions.csv",
        ),
        "physical": (
            PHYSICAL / "run.json",
            PHYSICAL / "result.json",
            PHYSICAL / "outer_predictions.csv",
        ),
        "distant": (
            AUDIT / "distant-run.json",
            AUDIT / "distant-result.json",
            AUDIT / "distant-outer_predictions.csv",
        ),
    }
    equality = read(PARENT / "gray-fixed-final-equality.json")
    assert sha(paths["gray"][2]) == equality["new_and_original_predictions_sha256"]
    models, results = {}, {}
    for role, (run_path, result_path, predictions) in paths.items():
        result = read(result_path)
        run = read(run_path)
        binding = run["binding"]
        assert run["status"] == result["status"] == "completed"
        assert binding["config"] == cpu["expected_bound_config"][role]
        assert binding["code_sha256"] == {
            n.removeprefix("rsna/"): h
            for n, h in cpu["source_sha256"].items()
            if n.startswith("rsna/")
        }
        for key in ("training_uids_sha256", "outer_uids_sha256", "folds_sha256"):
            assert binding[key] == cpu[key]
        assert binding["release"]["binding_sha256"] == cpu["release_binding_sha256"]
        assert result["binding_sha256"] == digest(binding)
        assert not run["held_out_confirmation_and_gold_used"]
        progress = terminal["distant"] if role == "distant" else old_progress["roles"][role]
        assert progress["optimizer_record_count"] == 843
        assert progress["all_progress_finite"] and progress["contiguous_optimizer_steps"]
        assert progress["role_terminal"]["returncode"] == 0 and progress["stderr_bytes"] == 0
        assert result["optimizer_steps"] == 843 and result["training_studies"] == 2247
        assert result["outer_studies"] == 1156
        publication = read(PARENT / f"epochs/{role}-003/verified.json")
        assert publication["file_count"] == 5
        assert result["checkpoint_sha256"] == publication["checkpoint_sha256"]
        assert sha(predictions) == result["predictions_sha256"]
        with predictions.open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        models[role] = {row["study_uid"]: row for row in rows}
        assert len(rows) == len(models[role]) == 1156
        assert all(row["fold"] == "0" and row["seed"] == "20260922" for row in rows)
        assert all(
            np.isfinite(float(row[t])) and 0 <= float(row[t]) <= 1 for row in rows for t in TARGETS
        )
        results[role] = result
    assert models["gray"].keys() == models["physical"].keys() == models["distant"].keys()
    uids = sorted(models["gray"])
    assert digest(uids) == cpu["outer_uids_sha256"]
    release = ROOT / "initiative/releases/report-supervision-v1/studies.parquet"
    labels = (
        ds.dataset(release, format="parquet")
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
    assert len(labels) == 1156 and labels.index.is_unique and labels.split.eq("train").all()
    assert labels.leakage_group.notna().all()
    library = runpy.run_path(str(ROOT / "scripts/rank_learning.py"))
    plans, records = [], []
    for target in TARGETS:
        mask = labels[target + "__mask"].to_numpy() == 1
        truth = labels.loc[mask, target].to_numpy(dtype=int)
        selected = [u for u, take in zip(uids, mask, strict=True) if take]
        values, target_plans = {}, {}
        for role in models:
            scores = [float(models[role][u][target]) for u in selected]
            values[role] = float(roc_auc_score(truth, scores))
            assert abs(values[role] - results[role]["metrics"]["per_target_auc"][target]) < 1e-9
            assert len(selected) == results[role]["metrics"]["observed_studies"][target]
            target_plans[role] = library["auc_plan"](truth, scores)
        plans.append((mask, target_plans))
        records.append(
            dict(
                target=target,
                auc=values,
                observed_studies=len(truth),
                positive_studies=int(truth.sum()),
                negative_studies=int(len(truth) - truth.sum()),
                positive_components=len(
                    set(labels.loc[mask & labels[target].eq(1), "leakage_group"])
                ),
            )
        )
    macros = {role: float(np.mean([x["auc"][role] for x in records])) for role in models}
    for role in models:
        assert abs(macros[role] - results[role]["metrics"]["macro_auc"]) < 1e-9
    draws, seed = 4000, 20261001
    rng = np.random.default_rng(seed)
    per_target = {name: [[] for _ in TARGETS] for name in CONTRASTS}
    macro_draws = {name: [] for name in CONTRASTS}
    for _ in range(draws):
        weights = library["component_weights"](labels.leakage_group.to_list(), rng)
        differences = {name: [] for name in CONTRASTS}
        for i, (mask, target_plans) in enumerate(plans):
            values = {
                role: library["weighted_auc"](p, weights[mask]) for role, p in target_plans.items()
            }
            for name, (control, candidate) in CONTRASTS.items():
                a, b = values[control], values[candidate]
                delta = None if a is None or b is None else b - a
                per_target[name][i].append(delta)
                differences[name].append(delta)
        for name, values in differences.items():
            macro_draws[name].append(
                float(np.mean(values)) if all(x is not None for x in values) else None
            )
    comparisons = {}
    for name, (control, candidate) in CONTRASTS.items():
        targets = [
            dict(
                row,
                delta=row["auc"][candidate] - row["auc"][control],
                paired_delta_uncertainty=library["interval"](values, draws),
            )
            for row, values in zip(records, per_target[name], strict=True)
        ]
        comparisons[name] = dict(
            control=control,
            candidate=candidate,
            primary_meniscus=next(x for x in targets if x["target"] == "Medial Meniscus"),
            macro_delta=macros[candidate] - macros[control],
            macro_delta_uncertainty=library["interval"](macro_draws[name], draws),
            targets=targets,
        )
    old = read(PHYSICAL / "comparison.json")
    assert (
        comparisons["physical_minus_gray"]["macro_delta_uncertainty"]
        == old["macro_delta_uncertainty"]
    )
    assert (
        comparisons["physical_minus_gray"]["primary_meniscus"]["paired_delta_uncertainty"]
        == old["primary"]["paired_delta_uncertainty"]
    )
    output = dict(
        status="three_original_fixed_final_neighbor_arms_independently_recomputed",
        macro_auc=macros,
        comparisons=comparisons,
        source_git=cpu["source_git"],
        checkpoint_sha256={role: result["checkpoint_sha256"] for role, result in results.items()},
        prediction_sha256={role: result["predictions_sha256"] for role, result in results.items()},
        release_parquet_sha256=sha(release),
        original_protocol_sha256=sha(
            ROOT / "initiative/experiments/competitive-v1/neighbor-channels-v1/protocol.json"
        ),
        outer_roster_sha256=cpu["outer_uids_sha256"],
        bootstrap_draws=draws,
        bootstrap_seed=seed,
        bootstrap_method="paired unstratified whole leakage component resampling",
        undefined_policy="retain undefined whenever any of all12 target AUCs is undefined",
        primary_contrast="physical_minus_gray Medial Meniscus",
        specificity_contrast="physical_minus_distant Medial Meniscus",
        distant_minus_gray_is_secondary=True,
        studies=1156,
        training_studies=2247,
        epochs_per_arm=3,
        optimizer_updates_per_arm=843,
        protected_gold_confirmation_used=False,
        accepted_labels_masks_or_folds_changed=False,
        epoch_or_strength_selection=False,
        gain_promoted=False,
        kaggle_score=None,
        caveat="One seed and weak-label pool; conditional intervals omit undefined draws explicitly. Sparse secondary targets cannot select a recipe. Distant perturbation magnitudes are unmatched, limiting anatomical specificity claims.",
    )
    text = json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path = AUDIT / "comparison.json"
    if path.exists():
        assert path.read_text() == text
    else:
        with path.open("x") as stream:
            stream.write(text)
    print(
        json.dumps(
            {
                "macro_auc": macros,
                "contrasts": {
                    name: {
                        k: value[k]
                        for k in ["primary_meniscus", "macro_delta", "macro_delta_uncertainty"]
                    }
                    for name, value in comparisons.items()
                },
            }
        )
    )


if __name__ == "__main__":
    main()
