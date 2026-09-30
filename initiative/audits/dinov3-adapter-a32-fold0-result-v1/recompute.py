"""Recompute the fixed-final A32 result against the identical baseline holdout."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from sklearn import __version__ as sklearn_version
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
BASE = ROOT / "initiative/audits/epoch1-baseline-and-epoch-preservation-v1"
LABELS = ROOT / "initiative/releases/report-supervision-v1/study_labels.csv"
TARGETS = (
    "ACL",
    "MCL",
    "Medial Meniscus",
    "Lateral Meniscus",
    "Medial OA",
    "Lateral OA",
    "PF OA",
    "Effusion",
    "Synovitis",
    "Baker's",
    "Contusion",
    "Fracture",
)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def predictions(path: Path, result: dict) -> dict[str, dict[str, str]]:
    if sha(path) != result["predictions_sha256"]:
        raise ValueError("Frozen prediction hash differs")
    data = rows(path)
    index = {row["study_uid"]: row for row in data}
    if len(index) != len(data) or len(data) != 1156:
        raise ValueError("Duplicate or incomplete outer predictions")
    if any(row["fold"] != "0" for row in data):
        raise ValueError("Different outer fold")
    return index


def main() -> None:
    actual = json.loads((AUDIT / "result.json").read_text())
    previous = json.loads((BASE / "final-result.json").read_text())
    new_binding = json.loads((AUDIT / "run-start.json").read_text())["binding"]
    old_binding = json.loads((BASE / "final-run.json").read_text())["binding"]
    matched_fields = (
        "training_uids_sha256",
        "outer_uids_sha256",
        "folds_sha256",
        "cache_side",
        "cache_version",
        "source_revision",
        "targets",
        "release",
    )
    if any(new_binding[key] != old_binding[key] for key in matched_fields):
        raise ValueError("Training, evaluation, data or supervision binding differs")
    for result, binding in ((actual, new_binding), (previous, old_binding)):
        binding_sha = hashlib.sha256(
            json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
        ).hexdigest()
        if binding_sha != result["binding_sha256"]:
            raise ValueError("Result and exact run binding differ")
    for result in (actual, previous):
        if (
            result["status"] != "completed"
            or result["diagnostic_run"] is not False
            or result["outer_fold"] != 0
            or result["training_studies"] != 2247
            or result["outer_studies"] != 1156
            or result["optimizer_steps"] != 2810
            or result["checkpoint_selection"] != "fixed_final_epoch_no_outer_selection"
        ):
            raise ValueError("Different run scope or checkpoint selection")
    adapted = predictions(AUDIT / "outer_predictions.csv", actual)
    baseline = predictions(BASE / "final-outer_predictions.csv", previous)
    if adapted.keys() != baseline.keys():
        raise ValueError("Baseline and adapter evaluation memberships differ")
    labels = {row["study_uid"]: row for row in rows(LABELS)}
    if any(labels[uid]["split"] != "train" for uid in adapted):
        raise ValueError("Gold or reserved confirmation included")
    targets = []
    for target in TARGETS:
        uids = sorted(uid for uid in adapted if labels[uid][target + "__mask"] == "1.0")
        truth = [int(float(labels[uid][target])) for uid in uids]
        if set(truth) != {0, 1}:
            raise ValueError("Target is undefined or nonbinary")
        adapted_auc = float(roc_auc_score(truth, [float(adapted[u][target]) for u in uids]))
        baseline_auc = float(roc_auc_score(truth, [float(baseline[u][target]) for u in uids]))
        if (
            abs(adapted_auc - actual["metrics"]["per_target_auc"][target]) > 1e-9
            or abs(baseline_auc - previous["metrics"]["per_target_auc"][target]) > 1e-9
        ):
            raise ValueError("Trainer and independent AUC differ")
        positive = {labels[u]["leakage_group"] for u, y in zip(uids, truth) if y == 1}
        negative = {labels[u]["leakage_group"] for u, y in zip(uids, truth) if y == 0}
        rare = positive if len(positive) <= len(negative) else negative
        sensitivity = None
        if len(rare) <= 9:
            recomputed = []
            for group in sorted(rare):
                kept = [u for u in uids if labels[u]["leakage_group"] != group]
                kept_truth = [int(float(labels[u][target])) for u in kept]
                if set(kept_truth) == {0, 1}:
                    recomputed.append(
                        float(roc_auc_score(kept_truth, [float(adapted[u][target]) for u in kept]))
                    )
            sensitivity = {
                "defined_recomputations": len(recomputed),
                "min_auc": min(recomputed) if recomputed else None,
                "max_auc": max(recomputed) if recomputed else None,
            }
        targets.append(
            {
                "target": target,
                "baseline_auc": baseline_auc,
                "adapter_auc": adapted_auc,
                "delta": adapted_auc - baseline_auc,
                "positive_studies": sum(truth),
                "negative_studies": len(truth) - sum(truth),
                "positive_components": len(positive),
                "negative_components": len(negative),
                "adapter_leave_one_rare_component": sensitivity,
            }
        )
    macro = sum(row["adapter_auc"] for row in targets) / len(targets)
    old_macro = sum(row["baseline_auc"] for row in targets) / len(targets)
    if abs(macro - actual["metrics"]["macro_auc"]) > 1e-9:
        raise ValueError("Macro AUC differs")
    record = {
        "schema_version": 1,
        "status": "identical_holdout_fixed_final_comparison_recomputed",
        "evaluation_source": "accepted report-derived development labels",
        "causal_isolation": False,
        "recipe_changes": {
            key: {"baseline": old_binding["config"].get(key), "adapter": value}
            for key, value in new_binding["config"].items()
            if value != old_binding["config"].get(key)
        },
        "world_sizes": {
            "baseline": old_binding["world_size"],
            "adapter": new_binding["world_size"],
        },
        "matched_binding_fields": list(matched_fields),
        "matched_frozen_control_completed": False,
        "kaggle_score": None,
        "training_studies": 2247,
        "outer_studies": 1156,
        "labels_sha256": sha(LABELS),
        "adapter_predictions_sha256": actual["predictions_sha256"],
        "baseline_predictions_sha256": previous["predictions_sha256"],
        "adapter_result_sha256": sha(AUDIT / "result.json"),
        "baseline_result_sha256": sha(BASE / "final-result.json"),
        "sklearn_version": sklearn_version,
        "baseline_macro_auc": old_macro,
        "adapter_macro_auc": macro,
        "macro_delta": macro - old_macro,
        "targets": targets,
    }
    text = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output = AUDIT / "comparison.json"
    if output.exists():
        if output.read_text() != text:
            raise ValueError("Archived comparison differs")
    else:
        output.write_text(text)
    print(json.dumps({k: record[k] for k in ("status", "adapter_macro_auc", "macro_delta")}))


if __name__ == "__main__":
    main()
