from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from rsna.schema import STATES, TARGETS


def report_group(report: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", report).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def read_reviews(folder: Path, reports: dict[str, str], suffix: str) -> pd.DataFrame:
    records = []
    seen: set[str] = set()
    for file in sorted(folder.glob(f"*{suffix}.jsonl")):
        for line in file.read_text().splitlines():
            row = json.loads(line)
            uid = row["study_uid"]
            if uid in seen or uid not in reports:
                raise ValueError(f"Duplicate/unknown study: {uid}")
            seen.add(uid)
            states, evidence = row["states"], row["evidence"]
            if len(states) != len(TARGETS) or len(evidence) != len(TARGETS):
                raise ValueError(f"Wrong target count: {uid}")
            for state, quote in zip(states, evidence):
                if state not in STATES or not isinstance(quote, str):
                    raise ValueError(f"Invalid annotation: {uid}")
                if state == "not_mentioned":
                    if quote:
                        raise ValueError(f"Evidence on a missing annotation: {uid}")
                elif not quote.strip() or quote not in reports[uid]:
                    raise ValueError(f"Unsupported quote: {uid}: {quote}")
            record = {"study_uid": uid, "notes": row["notes"], "artifact": file.name}
            for target, state, quote in zip(TARGETS, states, evidence):
                record[f"{target}__state"] = state
                record[f"{target}__evidence"] = quote
            records.append(record)
    if seen != set(reports):
        raise ValueError(f"Incomplete review: {len(set(reports) - seen)} missing studies")
    return pd.DataFrame(records).sort_values("study_uid").reset_index(drop=True)


def build_studies(
    train: pd.DataFrame,
    reviews: pd.DataFrame,
    duplicate_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    if train.StudyInstanceUID.duplicated().any() or reviews.study_uid.duplicated().any():
        raise ValueError("Study IDs must be unique")
    if set(train.StudyInstanceUID) != set(reviews.study_uid):
        raise ValueError("Review coverage does not match training metadata")
    if train.Report.isna().any() or train.Report.str.strip().eq("").any():
        raise ValueError("Missing report")
    train = train.rename(columns={"StudyInstanceUID": "study_uid"})
    studies = train.merge(reviews, on="study_uid", validate="one_to_one")
    gold = studies[list(TARGETS)].notna().all(axis=1)
    partial_gold = studies[list(TARGETS)].notna().any(axis=1) & ~gold
    if partial_gold.any():
        raise ValueError("Partial official labels need an explicit evaluation policy")
    if not studies.loc[gold, list(TARGETS)].isin([0, 1]).all().all():
        raise ValueError("Official labels must be binary")
    studies["report_group"] = studies.Report.map(report_group)
    groups = {uid: uid for uid in studies.study_uid}

    def root(uid: str) -> str:
        while uid != groups[uid]:
            groups[uid] = groups[groups[uid]]
            uid = groups[uid]
        return uid

    pairs = list(duplicate_pairs or [])
    for _, frame in studies.groupby("report_group"):
        ids = frame.study_uid.tolist()
        pairs.extend((ids[0], other) for other in ids[1:])
    for left, right in pairs:
        if left not in groups or right not in groups:
            raise ValueError("Duplicate pairs include an unknown/test study")
        first, second = sorted((root(left), root(right)))
        groups[second] = first
    studies["leakage_group"] = studies.study_uid.map(
        lambda uid: hashlib.sha256(root(uid).encode()).hexdigest()
    )
    gold_groups = set(studies.loc[gold, "leakage_group"])
    studies["split"] = studies.leakage_group.map(
        lambda group: "weak_validation" if int(group, 16) % 10 == 0 else "train"
    )
    studies.loc[studies.leakage_group.isin(gold_groups), "split"] = "quarantine"
    studies.loc[gold, "split"] = "gold_evaluation"
    for target in TARGETS:
        state = studies[f"{target}__state"]
        studies[f"{target}__mask"] = state.isin(["positive", "negative"]).astype(np.float32)
        studies[f"{target}__provenance"] = np.where(
            studies[f"{target}__mask"].eq(1), "report_reviewed", state
        )
        studies.loc[~gold, target] = state[~gold].eq("positive").astype(float)
        studies.loc[gold, f"{target}__mask"] = 1.0
        studies.loc[gold, f"{target}__provenance"] = "official_gold"
    if studies.groupby("report_group").split.nunique().max() > 1:
        nonquarantine = studies[studies.split != "quarantine"]
        if nonquarantine.groupby("report_group").split.nunique().max() > 1:
            raise ValueError("Report duplicates cross active splits")
    return studies.sort_values("study_uid").reset_index(drop=True)


def audit_labels(studies: pd.DataFrame, drafts: pd.DataFrame) -> dict:
    gold = studies[studies.split == "gold_evaluation"]
    weak = studies[~studies.split.isin(["gold_evaluation", "quarantine"])]
    drafts = drafts.set_index("study_uid").loc[studies.study_uid]
    targets = {}
    for target in TARGETS:
        state = gold[f"{target}__state"]
        observed = state.isin(["positive", "negative"])
        predictions = state[observed].eq("positive").astype(int)
        actual = gold.loc[observed, target].astype(int)
        tp = int(((predictions == 1) & (actual == 1)).sum())
        tn = int(((predictions == 0) & (actual == 0)).sum())
        fp = int(((predictions == 1) & (actual == 0)).sum())
        fn = int(((predictions == 0) & (actual == 1)).sum())
        targets[target] = {
            "weak_states": weak[f"{target}__state"].value_counts().to_dict(),
            "weak_observed_coverage": float(weak[f"{target}__mask"].mean()),
            "gold_review_coverage": float(observed.mean()),
            "gold_true_positive": tp,
            "gold_true_negative": tn,
            "gold_false_positive": fp,
            "gold_false_negative": fn,
            "gold_precision_observed": tp / (tp + fp) if tp + fp else None,
            "gold_recall_observed": tp / (tp + fn) if tp + fn else None,
            "gold_agreement_observed": (tp + tn) / int(observed.sum()) if observed.any() else None,
            "reviewer_state_changes": int(
                (
                    studies[f"{target}__state"].to_numpy() != drafts[f"{target}__state"].to_numpy()
                ).sum()
            ),
        }
    return {
        "studies": len(studies),
        "splits": studies.split.value_counts().to_dict(),
        "report_duplicate_studies": int(studies.report_group.duplicated(keep=False).sum()),
        "targets": targets,
        "policy": "Uncertain/unmentioned targets are masked. Gold is never used to fit report labels.",
        "probability_calibration": "Not applicable: categorical evidence states, not calibrated probabilities.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dicom-audit", type=Path, required=True)
    args = parser.parse_args()
    train = pd.read_csv(args.train)
    reports = dict(zip(train.StudyInstanceUID, train.Report))
    reviews = read_reviews(args.reviews, reports, "-reviewed")
    drafts = read_reviews(args.drafts, reports, "-draft")
    dicom_audit = json.loads(args.dicom_audit.read_text())
    if dicom_audit["train_test_duplicate_pairs"]:
        raise ValueError("Train/test image duplicates require review before release")
    pairs = [tuple(pair) for pair in dicom_audit["cross_study_pixel_duplicate_pairs"]]
    studies = build_studies(train, reviews, pairs)
    args.output.mkdir(parents=True, exist_ok=True)
    studies.to_parquet(args.output / "studies.parquet", index=False)
    reviews.to_parquet(args.output / "report_annotations.parquet", index=False)
    studies.drop(columns=["Report"]).to_csv(args.output / "study_labels.csv", index=False)
    report = audit_labels(studies, drafts)
    (args.output / "label_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
