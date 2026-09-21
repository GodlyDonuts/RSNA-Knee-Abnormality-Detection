from __future__ import annotations

import argparse
import hashlib
import json
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd

from rsna.schema import CACHE_VERSION, STATES, TARGETS


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_reports_fingerprint(reports: dict[str, str]) -> str:
    """Bind blinded report input without inspecting or including official targets."""
    records = [{"study_uid": uid, "report": reports[uid]} for uid in sorted(reports)]
    data = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def report_group(report: str) -> str:
    normalized = " ".join(unicodedata.normalize("NFKC", report).casefold().split())
    return hashlib.sha256(normalized.encode()).hexdigest()


def read_reviews(folder: Path, reports: dict[str, str], suffix: str) -> pd.DataFrame:
    if not reports or any(
        not isinstance(uid, str)
        or not uid.strip()
        or not isinstance(report, str)
        or not report.strip()
        for uid, report in reports.items()
    ):
        raise ValueError("Source reports require nonempty string study IDs and reports")
    records = []
    seen: set[str] = set()
    for file in sorted(folder.glob(f"*{suffix}.jsonl")):
        for line in file.read_text().splitlines():
            row = json.loads(line)
            if not isinstance(row, dict) or set(row) != {
                "study_uid",
                "states",
                "evidence",
                "notes",
            }:
                raise ValueError(f"Invalid annotation fields: {file.name}")
            uid = row["study_uid"]
            if not isinstance(uid, str) or uid in seen or uid not in reports:
                raise ValueError(f"Duplicate/unknown study: {uid}")
            seen.add(uid)
            states, evidence = row["states"], row["evidence"]
            if (
                not isinstance(states, list)
                or not isinstance(evidence, list)
                or len(states) != len(TARGETS)
                or len(evidence) != len(TARGETS)
            ):
                raise ValueError(f"Wrong target count: {uid}")
            if not isinstance(row["notes"], str):
                raise ValueError(f"Invalid annotation notes: {uid}")
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


def validate_review_provenance(
    provenance_path: Path,
    reports: dict[str, str],
    drafts: pd.DataFrame,
    reviews: pd.DataFrame,
    drafts_folder: Path,
    reviews_folder: Path,
    rubric_path: Path,
) -> dict:
    """Require two source-bound, separately identified passes for every fixed batch."""
    provenance = json.loads(provenance_path.read_text())
    expected_uids = sorted(reports)
    expected_batches = {
        f"batch-{start // 50:03d}": expected_uids[start : start + 50]
        for start in range(0, len(expected_uids), 50)
    }
    if not isinstance(provenance, dict) or set(provenance) != set(expected_batches):
        raise ValueError("Review provenance must cover exactly the source batches")
    rubric_hash = file_sha256(rubric_path)
    source_hash = source_reports_fingerprint(reports)
    for batch, uids in expected_batches.items():
        phases = provenance[batch]
        if not isinstance(phases, dict) or set(phases) != {"draft", "reviewed"}:
            raise ValueError(f"Both independently identified review stages are required: {batch}")
        for phase, frame, folder in (
            ("draft", drafts, drafts_folder),
            ("reviewed", reviews, reviews_folder),
        ):
            metadata = phases[phase]
            filename = f"{batch}-{phase}.jsonl"
            path = folder / filename
            if set(frame.loc[frame.artifact.eq(filename), "study_uid"]) != set(uids):
                raise ValueError(f"Review batch membership differs from source: {filename}")
            if not isinstance(metadata, dict) or any(
                not isinstance(metadata.get(field), str) or not metadata[field].strip()
                for field in ("agent_id", "session_id", "workflow_run_id", "notes")
            ):
                raise ValueError(f"Missing review identity/completion provenance: {filename}")
            if metadata.get("n_studies") != len(uids) or isinstance(metadata["n_studies"], bool):
                raise ValueError(f"Review provenance count mismatch: {filename}")
            if metadata.get("sha256") != file_sha256(path):
                raise ValueError(f"Review artifact checksum mismatch: {filename}")
            if metadata.get("rubric_sha256") != rubric_hash:
                raise ValueError(f"Review rubric is unbound or mismatched: {filename}")
            if metadata.get("source_reports_sha256") != source_hash:
                raise ValueError(f"Review source reports are unbound or mismatched: {filename}")
        if phases["reviewed"].get("parent_artifact_sha256") != phases["draft"]["sha256"]:
            raise ValueError(f"Reviewed artifact is not bound to its exact draft: {batch}")
        if any(
            phases["draft"][field] == phases["reviewed"][field]
            for field in ("agent_id", "session_id")
        ):
            raise ValueError(f"Reviewer must have a separate agent and session identity: {batch}")
    return {
        "rubric_sha256": rubric_hash,
        "source_reports_sha256": source_hash,
        "review_provenance_sha256": file_sha256(provenance_path),
        "independent_review_batches": len(expected_batches),
    }


def validate_dicom_audit(
    audit_path: Path, inventory: pd.DataFrame, train_uids: set[str], source_revision: str
) -> list[tuple[str, str]]:
    from rsna.audit import inventory_fingerprint

    audit = json.loads(audit_path.read_text())
    if audit.get("audit_schema_version") != 2 or audit.get("complete") is not True:
        raise ValueError("A complete versioned DICOM audit is required for release")
    if (
        audit.get("cache_version") != CACHE_VERSION
        or audit.get("source_revision") != source_revision
    ):
        raise ValueError("DICOM audit preprocessing/source revision mismatch")
    if audit.get("inventory_fingerprint_sha256") != inventory_fingerprint(inventory):
        raise ValueError("DICOM audit does not match the supplied source inventory")
    expected_counts = {
        "studies": inventory.study_uid.nunique(),
        "series": len(inventory[["split", "study_uid", "series_uid"]].drop_duplicates()),
        "decoded_slices": len(inventory),
        "source_bytes": int(inventory.bytes.sum()),
    }
    if any(audit.get(key) != value for key, value in expected_counts.items()):
        raise ValueError("DICOM audit coverage/counts differ from source inventory")
    by_split = {
        split: sorted(frame.study_uid.unique().tolist())
        for split, frame in inventory.groupby("split")
    }
    if (
        set(by_split) - {"train_series", "test_series"}
        or set(by_split.get("train_series", [])) != train_uids
    ):
        raise ValueError("DICOM training study coverage differs from report metadata")
    if set(by_split.get("train_series", [])) & set(by_split.get("test_series", [])):
        raise ValueError("Source study identities overlap train and test")
    if audit.get("study_uids_by_split") != by_split:
        raise ValueError("DICOM audit study identities differ from inventory")
    index_path = audit_path.parent / "series_cache_index.parquet"
    if not index_path.is_file() or audit.get("series_cache_index_sha256") != file_sha256(
        index_path
    ):
        raise ValueError("DICOM audit series index is missing or changed")
    if audit.get("train_test_duplicate_pairs") != []:
        raise ValueError("Train/test image duplicates require review before release")
    pairs = audit.get("cross_study_pixel_duplicate_pairs")
    if not isinstance(pairs, list):
        raise ValueError("Missing DICOM duplicate assessment")
    known_uids = set(inventory.study_uid)
    training_pairs = []
    for pair in pairs:
        if (
            not isinstance(pair, list)
            or len(pair) != 2
            or any(not isinstance(uid, str) or uid not in known_uids for uid in pair)
            or pair[0] == pair[1]
        ):
            raise ValueError("Invalid DICOM duplicate pair")
        left, right = pair
        if (left in train_uids) != (right in train_uids):
            raise ValueError("Train/test image duplicates require review before release")
        if left in train_uids:
            training_pairs.append((left, right))
    return training_pairs


def build_studies(
    train: pd.DataFrame,
    reviews: pd.DataFrame,
    duplicate_pairs: list[tuple[str, str]] | None = None,
) -> pd.DataFrame:
    if train.empty or any(
        not isinstance(uid, str) or not uid.strip() for uid in train.StudyInstanceUID
    ):
        raise ValueError("Study IDs must be nonempty strings")
    if train.StudyInstanceUID.duplicated().any() or reviews.study_uid.duplicated().any():
        raise ValueError("Study IDs must be unique")
    if set(train.StudyInstanceUID) != set(reviews.study_uid):
        raise ValueError("Review coverage does not match training metadata")
    if any(not isinstance(report, str) or not report.strip() for report in train.Report):
        raise ValueError("Missing report")
    for target in TARGETS:
        if not reviews[f"{target}__state"].isin(STATES).all():
            raise ValueError(f"Invalid annotation state: {target}")
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
        correction = f"{target}__correction"
        if correction in studies:
            studies.loc[
                studies[correction].ne("") & studies[f"{target}__mask"].eq(1),
                f"{target}__provenance",
            ] = "report_adjudicated"
        studies.loc[~gold, target] = state[~gold].eq("positive").astype(float)
        studies.loc[gold, f"{target}__mask"] = 1.0
        studies.loc[gold, f"{target}__provenance"] = "official_gold"
    nonquarantine = studies[studies.split != "quarantine"]
    if nonquarantine.groupby("leakage_group").split.nunique().max() > 1:
        raise ValueError("Report/pixel duplicates cross active splits")
    return studies.sort_values("study_uid").reset_index(drop=True)


def audit_labels(studies: pd.DataFrame, drafts: pd.DataFrame) -> dict:
    gold = studies[studies.split == "gold_evaluation"]
    weak = studies[~studies.split.isin(["gold_evaluation", "quarantine"])]
    drafts = drafts.set_index("study_uid").loc[studies.study_uid]
    targets = {}
    for target in TARGETS:
        original_state = studies.get(f"{target}__original_state", studies[f"{target}__state"])
        state = gold[f"{target}__state"]
        observed = state.isin(["positive", "negative"])
        if "report_supervision_allowed" in gold:
            observed &= gold.report_supervision_allowed
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
                (original_state.to_numpy() != drafts[f"{target}__state"].to_numpy()).sum()
            ),
            "adjudicated_state_changes": int(original_state.ne(studies[f"{target}__state"]).sum()),
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
    from rsna.adjudications import apply_adjudications
    from rsna.correspondence import apply_correspondence

    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--drafts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dicom-audit", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source-revision", required=True)
    parser.add_argument("--review-provenance", type=Path, required=True)
    parser.add_argument("--rubric", type=Path, required=True)
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("."),
        help="Repository root for relative adjudication evidence paths (default: current directory)",
    )
    args = parser.parse_args()
    train = pd.read_csv(args.train)
    if train.StudyInstanceUID.duplicated().any():
        raise ValueError("Study IDs must be unique")
    reports = dict(zip(train.StudyInstanceUID, train.Report))
    reviews = read_reviews(args.reviews, reports, "-reviewed")
    drafts = read_reviews(args.drafts, reports, "-draft")
    provenance = validate_review_provenance(
        args.review_provenance,
        reports,
        drafts,
        reviews,
        args.drafts,
        args.reviews,
        args.rubric,
    )
    original_reviews = reviews
    reviews, adjudications = apply_adjudications(
        reviews, reports, args.reviews, args.rubric, args.artifact_root
    )
    pairs = validate_dicom_audit(
        args.dicom_audit, pd.read_parquet(args.inventory), set(reports), args.source_revision
    )
    studies = build_studies(train, reviews, pairs)
    studies, correspondence = apply_correspondence(studies, reports, args.artifact_root)
    args.output.mkdir(parents=True, exist_ok=True)
    studies.to_parquet(args.output / "studies.parquet", index=False)
    reviews.to_parquet(args.output / "report_annotations.parquet", index=False)
    original_reviews.to_parquet(args.output / "report_annotations_original.parquet", index=False)
    (args.output / "adjudication_trace.json").write_text(
        json.dumps(adjudications, ensure_ascii=False, indent=2) + "\n"
    )
    (args.output / "correspondence_trace.json").write_text(
        json.dumps(correspondence, ensure_ascii=False, indent=2) + "\n"
    )
    studies.drop(columns=["Report"]).to_csv(args.output / "study_labels.csv", index=False)
    report = audit_labels(studies, drafts)
    report["adjudications"] = adjudications
    report["correspondence"] = correspondence
    report["release_inputs"] = {
        **provenance,
        "train_sha256": file_sha256(args.train),
        "dicom_audit_sha256": file_sha256(args.dicom_audit),
        "source_revision": args.source_revision,
        "cache_version": CACHE_VERSION,
        "adjudication_registry_sha256": adjudications["registry_sha256"],
        "correspondence_registry_sha256": correspondence["registry_sha256"],
    }
    (args.output / "label_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
