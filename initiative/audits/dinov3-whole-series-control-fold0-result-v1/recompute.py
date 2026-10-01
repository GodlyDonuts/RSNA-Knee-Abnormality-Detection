"""Replay fixed-final whole-series development AUCs; preserve rare-label sensitivity."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from sklearn.metrics import roc_auc_score

from rsna.schema import TARGETS

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
PREVIOUS = ROOT / "initiative/audits/dinov3-adapter-a32-fold0-result-v1"
LABELS = ROOT / "initiative/releases/report-supervision-v1/study_labels.csv"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def predictions(folder, result):
    path = folder / "outer_predictions.csv"
    assert sha(path) == result["predictions_sha256"]
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    index = {r["study_uid"]: r for r in rows}
    assert len(index) == len(rows) == 1156 and all(r["fold"] == "0" for r in rows)
    return index


def main():
    current = json.loads((AUDIT / "result.json").read_text())
    prior = json.loads((PREVIOUS / "result.json").read_text())
    binding = json.loads((AUDIT / "run.json").read_text())["binding"]
    prior_binding = json.loads((PREVIOUS / "run.json").read_text())["binding"]
    keys = (
        "training_uids_sha256",
        "outer_uids_sha256",
        "folds_sha256",
        "cache_side",
        "cache_version",
        "source_revision",
        "targets",
        "release",
    )
    assert all(binding[k] == prior_binding[k] for k in keys)
    for result, run in ((current, binding), (prior, prior_binding)):
        assert (
            hashlib.sha256(
                json.dumps(run, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
            ).hexdigest()
            == result["binding_sha256"]
        )
        assert result["status"] == "completed" and result["diagnostic_run"] is False
        assert result["outer_fold"] == 0 and result["optimizer_steps"] == 2810
        assert result["outer_studies"] == 1156 and result["training_studies"] == 2247
        assert result["checkpoint_selection"] == "fixed_final_epoch_no_outer_selection"
    adapted, baseline = predictions(AUDIT, current), predictions(PREVIOUS, prior)
    assert adapted.keys() == baseline.keys()
    accepted_release = binding["release"]
    assert sha(LABELS) == json.loads((PREVIOUS / "comparison.json").read_text())["labels_sha256"]
    # Restrict lookup immediately to the fixed development prediction membership;
    # never inspect gold or reserved confirmation targets to tune this comparison.
    labels = {}
    with LABELS.open(newline="") as stream:
        for row in csv.DictReader(stream):
            if row["study_uid"] in adapted:
                assert row["split"] == "train"
                labels[row["study_uid"]] = row
    assert labels.keys() == adapted.keys()
    targets = []
    for target in TARGETS:
        uids = sorted(u for u in adapted if float(labels[u][target + "__mask"]) == 1)
        y = [int(float(labels[u][target])) for u in uids]
        assert set(y) == {0, 1}
        scores = {}
        for name, model, result in (
            ("whole_series", adapted, current),
            ("rank32", baseline, prior),
        ):
            value = float(roc_auc_score(y, [float(model[u][target]) for u in uids]))
            assert abs(value - result["metrics"]["per_target_auc"][target]) < 1e-9
            scores[name] = value
        positive = {labels[u]["leakage_group"] for u, truth in zip(uids, y) if truth}
        negative = {labels[u]["leakage_group"] for u, truth in zip(uids, y) if not truth}
        rare = positive if len(positive) <= len(negative) else negative
        sensitivity = {}
        if len(rare) <= 9:
            for name, model in (("whole_series", adapted), ("rank32", baseline)):
                values = []
                for group in sorted(rare):
                    kept = [u for u in uids if labels[u]["leakage_group"] != group]
                    truth = [int(float(labels[u][target])) for u in kept]
                    if set(truth) == {0, 1}:
                        values.append(
                            float(roc_auc_score(truth, [float(model[u][target]) for u in kept]))
                        )
                sensitivity[name] = {
                    "defined_recomputations": len(values),
                    "min_auc": min(values) if values else None,
                    "max_auc": max(values) if values else None,
                }
        targets.append(
            {
                "target": target,
                **scores,
                "delta": scores["whole_series"] - scores["rank32"],
                "positive_studies": sum(y),
                "negative_studies": len(y) - sum(y),
                "positive_components": len(positive),
                "negative_components": len(negative),
                "leave_one_rare_component": sensitivity,
            }
        )
    new_macro = sum(t["whole_series"] for t in targets) / 12
    prior_macro = sum(t["rank32"] for t in targets) / 12
    assert abs(new_macro - current["metrics"]["macro_auc"]) < 1e-9
    assert abs(prior_macro - prior["metrics"]["macro_auc"]) < 1e-9
    record = {
        "status": "same_holdout_fixed_final_whole_series_comparison_recomputed",
        "matched_binding_fields": list(keys),
        "release": accepted_release,
        "labels_sha256": sha(LABELS),
        "whole_series_result_sha256": sha(AUDIT / "result.json"),
        "rank32_result_sha256": sha(PREVIOUS / "result.json"),
        "whole_series_predictions_sha256": current["predictions_sha256"],
        "rank32_predictions_sha256": prior["predictions_sha256"],
        "whole_series_macro_auc": new_macro,
        "rank32_macro_auc": prior_macro,
        "macro_delta": new_macro - prior_macro,
        "targets": targets,
        "gold_or_confirmation_used": False,
        "kaggle_score": None,
        "causal_rank_isolation": False,
        "recipe_changes": {
            key: {"rank32": prior_binding["config"].get(key), "whole_series": value}
            for key, value in binding["config"].items()
            if value != prior_binding["config"].get(key)
        },
        "initialization_caveat": "Whole-series coverage changes the forward/reduction graph; one seed/fold cannot establish a generalization gain.",
    }
    text = json.dumps(record, indent=2, sort_keys=True, allow_nan=False) + "\n"
    # Keep the first replay unchanged; v2 also pins the accepted label bytes
    # and records configuration differences rather than claiming rank isolation.
    output = AUDIT / "comparison.json"
    if output.exists():
        assert output.read_text() == text
    else:
        output.write_text(text)
    print(
        json.dumps(
            {
                "whole_series": new_macro,
                "rank32": prior_macro,
                "delta": new_macro - prior_macro,
                "MCL": next(t for t in targets if t["target"] == "MCL"),
            }
        )
    )


if __name__ == "__main__":
    main()
