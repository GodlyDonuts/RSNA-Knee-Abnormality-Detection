"""Recompute the frozen fold-0 score and rare-component sensitivity."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from sklearn import __version__ as sklearn_version
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parents[3]
AUDIT = Path(__file__).resolve().parent
LABELS = ROOT / "initiative/releases/report-supervision-v1/study_labels.csv"
RUN = ROOT / "initiative/audits/epoch1-baseline-and-epoch-preservation-v1"
PREDICTIONS = RUN / "final-outer_predictions.csv"
RESULT = RUN / "final-result.json"
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
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def main() -> None:
    result = json.loads(RESULT.read_text())
    if sha(PREDICTIONS) != result["predictions_sha256"]:
        raise ValueError("Frozen prediction file differs")
    labels = {row["study_uid"]: row for row in read_csv(LABELS)}
    predictions = read_csv(PREDICTIONS)
    if len(predictions) != result["outer_studies"] or len(
        {p["study_uid"] for p in predictions}
    ) != len(predictions):
        raise ValueError("Outer-fold prediction membership differs")
    if any(
        p["study_uid"] not in labels
        or labels[p["study_uid"]]["split"] != "train"
        or p["fold"] != "0"
        for p in predictions
    ):
        raise ValueError("Prediction row is outside the accepted training outer fold")
    metrics = []
    for target in TARGETS:
        rows = [
            (labels[p["study_uid"]], p)
            for p in predictions
            if labels[p["study_uid"]][target + "__mask"] == "1.0"
        ]
        values = [int(float(label[target])) for label, _ in rows]
        if len(set(values)) != 2 or any(value not in (0, 1) for value in values):
            raise ValueError("Frozen target is undefined or nonbinary")
        auc = roc_auc_score(values, [float(p[target]) for _, p in rows])
        if abs(auc - result["metrics"]["per_target_auc"][target]) > 1e-9:
            raise ValueError("Trainer-reported target AUC differs")
        positive_groups = {
            label["leakage_group"] for label, _ in rows if int(float(label[target])) == 1
        }
        negative_groups = {
            label["leakage_group"] for label, _ in rows if int(float(label[target])) == 0
        }
        rare_class = 1 if len(positive_groups) <= len(negative_groups) else 0
        rare_groups = positive_groups if rare_class == 1 else negative_groups
        sensitivity = None
        if len(rare_groups) <= 9:
            leave_one = []
            for group in sorted(rare_groups):
                kept = [(label, p) for label, p in rows if label["leakage_group"] != group]
                kept_labels = [int(float(label[target])) for label, _ in kept]
                if len(set(kept_labels)) == 2:
                    leave_one.append(
                        roc_auc_score(kept_labels, [float(p[target]) for _, p in kept])
                    )
            sensitivity = {
                "rare_class": rare_class,
                "rare_components": len(rare_groups),
                "defined_recomputations": len(leave_one),
                "min_auc": min(leave_one) if leave_one else None,
                "max_auc": max(leave_one) if leave_one else None,
            }
        metrics.append(
            {
                "target": target,
                "auc": auc,
                "positive_studies": sum(values),
                "negative_studies": len(values) - sum(values),
                "positive_components": len(positive_groups),
                "negative_components": len(negative_groups),
                "leave_one_rare_component": sensitivity,
            }
        )
    macro = sum(row["auc"] for row in metrics) / len(metrics)
    if abs(macro - result["metrics"]["macro_auc"]) > 1e-9:
        raise ValueError("Trainer-reported macro AUC differs")
    record = {
        "schema_version": 1,
        "status": "exact_fold0_score_recomputed",
        "scope": "accepted report-derived labels, development outer fold only",
        "labels_sha256": sha(LABELS),
        "predictions_sha256": sha(PREDICTIONS),
        "result_sha256": sha(RESULT),
        "sklearn_version": sklearn_version,
        "outer_studies": len(predictions),
        "macro_auc": macro,
        "targets": metrics,
    }
    serialized = json.dumps(record, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output = AUDIT / "metrics.json"
    if output.exists():
        if output.read_text() != serialized:
            raise ValueError("Previously archived score analysis differs")
    else:
        output.write_text(serialized)
    print(json.dumps({"status": record["status"], "macro_auc": macro, "targets": len(metrics)}))


if __name__ == "__main__":
    main()
