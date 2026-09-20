#!/usr/bin/env python3
"""Create calibrated soft targets from multilingual radiology reports.

The deterministic rule channel is intentionally inspectable.  An optional NLI
CSV can add a second, independently produced probability per target.  Gold
labels override weak targets, while calibration is fitted out of fold whenever
a fold column is supplied.
"""

from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .constants import TARGETS
except ImportError:
    from constants import TARGETS


FINDING_PATTERNS = {
    "ACL": (r"\bacl\b", r"anterior cruciate", r"ligament croise anterieur", r"ligamento cruzado anterior"),
    "MCL": (r"\bmcl\b", r"medial collateral", r"ligament collateral medial", r"ligamento colateral medial"),
    "Medial Meniscus": (r"medial menisc", r"menisque medial", r"menisco medial", r"innenmenisk"),
    "Lateral Meniscus": (r"lateral menisc", r"menisque lateral", r"menisco lateral", r"aussenmenisk"),
    "Medial OA": (r"medial.{0,35}(osteoarth|arthros|cartilage loss|chondr)", r"medial compartment narrowing"),
    "Lateral OA": (r"lateral.{0,35}(osteoarth|arthros|cartilage loss|chondr)", r"lateral compartment narrowing"),
    "PF OA": (r"patellofemoral.{0,35}(osteoarth|arthros|chondr|cartilage loss)", r"retropatellar chondr"),
    "Effusion": (r"effusion", r"joint fluid", r"hydarthros", r"derrame articular", r"epanchement"),
    "Synovitis": (r"synovitis", r"synovial hypertrophy", r"sinovitis", r"synovite"),
    "Baker's": (r"baker.?s? cyst", r"popliteal cyst", r"quiste poplite", r"quiste de baker"),
    "Contusion": (r"bone (marrow )?(contusion|bruise)", r"marrow edema", r"bone oedema", r"contusion osse"),
    "Fracture": (r"fracture", r"cortical break", r"trabecular fracture", r"fraktur", r"fractura"),
}

NEGATION = re.compile(
    r"(?:\bno\b|\bnot\b|without|absence of|negative for|free of|intact|preserved|"
    r"kein(?:e|en|er)?|sans|aucun(?:e)?|sin evidencia de|no se observa)[^.;:]{0,55}$"
)
UNCERTAIN = re.compile(
    r"(?:possible|possibly|probable|may represent|cannot exclude|suspicious for|equivocal|"
    r"peut etre|possible de|posible|sospecha)[^.;:]{0,55}$"
)


def normalize_report(text: object) -> str:
    value = "" if pd.isna(text) else str(text)
    value = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("ascii")
    return re.sub(r"\s+", " ", value.lower()).strip()


def rule_probability(report: str, target: str) -> tuple[float, float, int]:
    """Return probability, confidence, and mention count for one finding."""

    matches = []
    for pattern in FINDING_PATTERNS[target]:
        matches.extend(re.finditer(pattern, report))
    if not matches:
        # Non-mention is weak negative evidence in a diagnostic impression,
        # never a hard negative.
        return 0.22, 0.22, 0
    states = []
    for match in matches:
        prefix = report[max(0, match.start() - 70) : match.start()]
        if NEGATION.search(prefix):
            states.append("negative")
        elif UNCERTAIN.search(prefix):
            states.append("uncertain")
        else:
            states.append("positive")
    if "positive" in states:
        return 0.93, min(0.95, 0.72 + 0.04 * len(matches)), len(matches)
    if "uncertain" in states:
        return 0.55, 0.35, len(matches)
    return 0.06, min(0.95, 0.72 + 0.04 * len(matches)), len(matches)


def _logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-4, 1 - 1e-4)
    return np.log(clipped / (1 - clipped))


def _sigmoid(value: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(value, -30, 30)))


def blend_channels(rule: np.ndarray, nli: np.ndarray | None, nli_weight: float) -> np.ndarray:
    if nli is None:
        return rule
    valid = np.isfinite(nli)
    blended = rule.copy()
    blended[valid] = _sigmoid(
        (1 - nli_weight) * _logit(rule[valid]) + nli_weight * _logit(nli[valid])
    )
    return blended


def calibrate_out_of_fold(
    raw: np.ndarray,
    gold: np.ndarray,
    folds: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calibrate raw scores with gold labels without crossing validation folds."""

    from sklearn.linear_model import LogisticRegression

    calibrated = raw.copy()
    reliability = np.full(len(raw), 0.35, dtype=float)
    valid_gold = np.isfinite(gold)
    applications = [np.ones(len(raw), dtype=bool)] if folds is None else [folds == f for f in np.unique(folds)]
    for application in applications:
        fitting = valid_gold & (~application if folds is not None else np.ones(len(raw), dtype=bool))
        if folds is None and (fitting.sum() < 8 or np.unique(gold[fitting]).size < 2):
            fitting = valid_gold
        if fitting.sum() < 4 or np.unique(gold[fitting]).size < 2:
            continue
        model = LogisticRegression(C=0.5, solver="lbfgs")
        model.fit(_logit(raw[fitting])[:, None], gold[fitting].astype(int))
        calibrated[application] = model.predict_proba(_logit(raw[application])[:, None])[:, 1]
        prediction = model.predict_proba(_logit(raw[fitting])[:, None])[:, 1]
        brier = float(np.mean((prediction - gold[fitting]) ** 2))
        reliability[application] = np.clip(1 - 2 * brier, 0.15, 0.9)
    return calibrated, reliability


def build_soft_labels(
    train: pd.DataFrame,
    nli: pd.DataFrame | None,
    fold_column: str | None,
    nli_weight: float,
) -> pd.DataFrame:
    if "StudyInstanceUID" not in train or "Report" not in train:
        raise ValueError("train CSV must contain StudyInstanceUID and Report")
    reports = train["Report"].map(normalize_report)
    folds = train[fold_column].to_numpy() if fold_column and fold_column in train else None
    output = pd.DataFrame({"StudyInstanceUID": train["StudyInstanceUID"].astype(str)})
    if folds is not None:
        output[fold_column] = folds
    nli_indexed = None
    if nli is not None:
        if "StudyInstanceUID" not in nli:
            raise ValueError("NLI CSV must contain StudyInstanceUID")
        nli = nli.copy()
        nli["StudyInstanceUID"] = nli["StudyInstanceUID"].astype(str)
        nli_indexed = nli.set_index("StudyInstanceUID")
    for target in TARGETS:
        scored = [rule_probability(report, target) for report in reports]
        rule = np.asarray([row[0] for row in scored])
        rule_conf = np.asarray([row[1] for row in scored])
        nli_values = None
        if nli_indexed is not None and f"{target}__nli" in nli_indexed:
            nli_values = nli_indexed.reindex(output["StudyInstanceUID"])[f"{target}__nli"].to_numpy(float)
        raw = blend_channels(rule, nli_values, nli_weight)
        gold = (
            pd.to_numeric(train[target], errors="coerce").to_numpy(float)
            if target in train
            else np.full(len(train), np.nan)
        )
        calibrated, calibration_reliability = calibrate_out_of_fold(raw, gold, folds)
        weak_conf = np.sqrt(rule_conf * calibration_reliability)
        if nli_values is not None:
            agreement = 1 - np.abs(rule - np.nan_to_num(nli_values, nan=rule))
            weak_conf *= np.clip(agreement, 0.25, 1)
        is_gold = np.isfinite(gold)
        output[target] = np.where(is_gold, gold, calibrated)
        output[f"{target}__conf"] = np.where(is_gold, 1.0, np.clip(weak_conf, 0.05, 0.9))
        output[f"{target}__gold"] = is_gold.astype(int)
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nli-csv", type=Path)
    parser.add_argument("--nli-weight", type=float, default=0.45)
    parser.add_argument("--fold-column", default="fold")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train = pd.read_csv(args.train_csv)
    nli = pd.read_csv(args.nli_csv) if args.nli_csv else None
    output = build_soft_labels(train, nli, args.fold_column, args.nli_weight)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output, index=False)
    gold = int(output[[f"{target}__gold" for target in TARGETS]].to_numpy().sum())
    print(f"wrote {args.output}: {len(output)} studies, {gold} gold target assignments")


if __name__ == "__main__":
    main()
