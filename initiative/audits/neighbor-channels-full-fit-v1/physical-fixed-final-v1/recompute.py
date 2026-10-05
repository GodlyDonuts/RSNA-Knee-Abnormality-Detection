"""Fixed-final physical-minus-gray comparison; distant specificity remains pending."""

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


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def main():
    collection = read(AUDIT / "collection.json")
    cpu = read(PARENT / "actual-CPU-acceptance.json")
    progress = read(PARENT / "progress-1319.json")
    assert collection["source_git"] == cpu["source_git"] == progress["source_git"]
    assert progress["source_and_config_bindings_verified"]
    assert cpu["initial_learned_state"]["gray"] == cpu["initial_learned_state"]["physical"]
    assert cpu["initial_learned_state"]["gray"] == cpu["initial_learned_state"]["distant"]
    for name, row in collection["files"].items():
        path = AUDIT / name
        assert path.stat().st_size == row["bytes"] and sha(path) == row["sha256"]
    run, result = read(AUDIT / "run.json"), read(AUDIT / "result.json")
    assert run["status"] == result["status"] == "completed"
    assert run["binding"]["config"] == cpu["expected_bound_config"]["physical"]
    assert result["binding_sha256"] == digest(run["binding"])
    assert result["optimizer_steps"] == 843 and result["training_studies"] == 2247
    assert result["outer_studies"] == 1156 and not run["held_out_confirmation_and_gold_used"]
    proof = read(PARENT / "epochs/physical-003/verified.json")
    assert result["checkpoint_sha256"] == proof["checkpoint_sha256"]
    assert proof["file_count"] == 5
    for role in ("gray", "physical"):
        p = progress["roles"][role]
        assert p["optimizer_record_count"] == 843
        assert p["all_progress_finite"] and p["contiguous_optimizer_steps"]
        assert p["role_terminal"]["returncode"] == 0 and p["stderr_bytes"] == 0
    equality = read(PARENT / "gray-fixed-final-equality.json")
    assert sha(GRAY / "outer_predictions.csv") == equality["new_and_original_predictions_sha256"]
    assert sha(AUDIT / "outer_predictions.csv") == result["predictions_sha256"]
    models = {}
    for role, folder in (("gray", GRAY), ("physical", AUDIT)):
        with (folder / "outer_predictions.csv").open(newline="") as stream:
            rows = list(csv.DictReader(stream))
        models[role] = {row["study_uid"]: row for row in rows}
        assert len(rows) == len(models[role]) == 1156
        assert all(row["fold"] == "0" and row["seed"] == "20260922" for row in rows)
        assert all(
            np.isfinite(float(row[t])) and 0 <= float(row[t]) <= 1 for row in rows for t in TARGETS
        )
    assert models["gray"].keys() == models["physical"].keys()
    uids = sorted(models["physical"])
    assert digest(uids) == cpu["outer_uids_sha256"]
    path = ROOT / "initiative/releases/report-supervision-v1/studies.parquet"
    labels = (
        ds.dataset(path, format="parquet")
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
    gray_result = read(GRAY / "result.json")
    for target in TARGETS:
        mask = labels[target + "__mask"].to_numpy() == 1
        truth = labels.loc[mask, target].to_numpy(dtype=int)
        selected = [u for u, take in zip(uids, mask, strict=True) if take]
        values, pair = {}, []
        for role, source in (("gray", gray_result), ("physical", result)):
            scores = [float(models[role][u][target]) for u in selected]
            values[role] = float(roc_auc_score(truth, scores))
            assert abs(values[role] - source["metrics"]["per_target_auc"][target]) < 1e-9
            assert len(selected) == source["metrics"]["observed_studies"][target]
            pair.append(library["auc_plan"](truth, scores))
        plans.append((mask, pair))
        records.append(
            dict(
                target=target,
                **values,
                delta=values["physical"] - values["gray"],
                observed_studies=len(truth),
                positive_studies=int(truth.sum()),
                negative_studies=int(len(truth) - truth.sum()),
                positive_components=len(
                    set(labels.loc[mask & labels[target].eq(1), "leakage_group"])
                ),
            )
        )
    macro = {role: float(np.mean([x[role] for x in records])) for role in models}
    assert abs(macro["physical"] - result["metrics"]["macro_auc"]) < 1e-9
    assert abs(macro["gray"] - equality["fixed_final_internal_macro_AUC"]) < 1e-9
    draws, seed = 4000, 20261001
    rng = np.random.default_rng(seed)
    per_target, macros = [[] for _ in TARGETS], []
    for _ in range(draws):
        weights = library["component_weights"](labels.leakage_group.to_list(), rng)
        differences = []
        for i, (mask, pair) in enumerate(plans):
            a, b = (library["weighted_auc"](p, weights[mask]) for p in pair)
            delta = None if a is None or b is None else b - a
            per_target[i].append(delta)
            differences.append(delta)
        macros.append(
            float(np.mean(differences)) if all(x is not None for x in differences) else None
        )
    targets = [
        dict(row, paired_delta_uncertainty=library["interval"](values, draws))
        for row, values in zip(records, per_target, strict=True)
    ]
    output = dict(
        status="fixed_final_physical_minus_gray_independently_recomputed_distant_pending",
        primary=next(x for x in targets if x["target"] == "Medial Meniscus"),
        gray_macro_auc=macro["gray"],
        physical_macro_auc=macro["physical"],
        macro_delta=macro["physical"] - macro["gray"],
        macro_delta_uncertainty=library["interval"](macros, draws),
        targets=targets,
        bootstrap_draws=draws,
        bootstrap_seed=seed,
        bootstrap_method="paired unstratified whole leakage component resampling",
        undefined_policy="retain undefined whenever any of all12 target AUCs is undefined",
        studies=1156,
        training_studies=2247,
        epochs=3,
        optimizer_updates=843,
        source_git=cpu["source_git"],
        release_parquet_sha256=sha(path),
        fixed_original_protocol_sha256=sha(
            ROOT / "initiative/experiments/competitive-v1/neighbor-channels-v1/protocol.json"
        ),
        checkpoint_sha256=result["checkpoint_sha256"],
        physical_prediction_sha256=result["predictions_sha256"],
        gray_prediction_sha256=equality["new_and_original_predictions_sha256"],
        protected_gold_confirmation_used=False,
        accepted_labels_masks_or_folds_changed=False,
        distant_specificity_comparison_complete=False,
        kaggle_score=None,
        gain_promoted=False,
        caveat="Conditional on defined draws, one fitted seed and weak-label pool; sparse targets and repeated-seed uncertainty remain. Distant perturbation magnitudes are unmatched.",
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
                k: output[k]
                for k in [
                    "primary",
                    "gray_macro_auc",
                    "physical_macro_auc",
                    "macro_delta",
                    "macro_delta_uncertainty",
                ]
            }
        )
    )


if __name__ == "__main__":
    main()
