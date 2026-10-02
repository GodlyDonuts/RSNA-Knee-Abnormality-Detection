"""Compare public report verdicts on a fixed fitting cohort; never promote labels.

Only aggregate counts leave this diagnostic. Optional report-review records stay
in ignored artifacts, with owner-only permissions. Public scores are ranks, not
calibrated clinical probabilities or accepted target definitions.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

from rsna.schema import TARGETS

CAP = 8 * 1024 * 1024


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fitting_cohort(studies: Path, folds: Path, excluded_fold: int) -> pd.DataFrame:
    # Push the partition filter into the reader before decoding target values.
    frame = pd.read_parquet(studies, filters=[("split", "==", "train")])
    membership = pd.read_csv(folds, dtype={"study_uid": str, "leakage_group": str})
    if frame.study_uid.duplicated().any() or membership.study_uid.duplicated().any():
        raise ValueError("duplicate original membership")
    if membership.groupby("leakage_group").fold.nunique().gt(1).any():
        raise ValueError("leakage component crosses folds")
    joined = frame.merge(membership, on="study_uid", validate="one_to_one", suffixes=("", "_fold"))
    if len(joined) != len(frame) or not joined.leakage_group.eq(joined.leakage_group_fold).all():
        raise ValueError("original release/fold membership differs")
    masks = joined[[f"{t}__mask" for t in TARGETS]]
    if not masks.isin([0, 1]).all().all():
        raise ValueError("original masks must be binary")
    eligible = joined.report_supervision_allowed & masks.astype(bool).any(axis=1)
    return joined.loc[eligible & joined.fold.ne(excluded_fold)].copy().set_index("study_uid")


def csv_member(archive: Path, member: str) -> tuple[bytes, dict]:
    if archive.stat().st_size > CAP:
        raise ValueError("compressed archive exceeds cap")
    raw = archive.read_bytes()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        entries = z.infolist()
        names = [item.filename for item in entries]
        if len(names) != len(set(names)) or sum(item.file_size for item in entries) > CAP:
            raise ValueError("duplicate members or expanded archive exceeds cap")
        if any(Path(name).is_absolute() or ".." in Path(name).parts for name in names):
            raise ValueError("unsafe archive name")
        item = z.getinfo(member)
        if item.is_dir():
            raise ValueError("CSV member is a directory")
        with z.open(item) as stream:
            content = stream.read(CAP + 1)
        if len(content) > CAP or len(content) != item.file_size:
            raise ValueError("expanded member exceeds cap or size differs")
    return content, {"archive_sha256": sha(raw), "member": member, "member_sha256": sha(content)}


def compare(
    content: bytes, cohort: pd.DataFrame, review_per_category: int = 2
) -> tuple[dict, list]:
    if not 0 <= review_per_category <= 6:
        raise ValueError("review bound must be 0..6")
    counts = {t: Counter() for t in TARGETS}
    scores = {t: Counter() for t in TARGETS}
    confidences = {t: Counter() for t in TARGETS}
    masked_states = {t: Counter() for t in TARGETS}
    masked_reasons = {t: Counter() for t in TARGETS}
    candidates = defaultdict(list)
    seen = set()
    reader = csv.DictReader(io.StringIO(content.decode("utf-8-sig")))
    required = ["StudyInstanceUID"] + [
        c for t in TARGETS for c in (t, f"{t}__conf", f"{t}__verdict")
    ]
    if reader.fieldnames is None or len(set(reader.fieldnames)) != len(reader.fieldnames):
        raise ValueError("missing or duplicate CSV columns")
    if not set(required).issubset(reader.fieldnames):
        raise ValueError("required public columns missing")
    for row in reader:
        uid = row["StudyInstanceUID"]
        # Interpret no external targets for outer/protected/ineligible studies.
        if uid not in cohort.index:
            continue
        if uid in seen:
            raise ValueError("duplicate fitting study in public source")
        seen.add(uid)
        original = cohort.loc[uid]
        for target in TARGETS:
            verdict = row[f"{target}__verdict"]
            score, confidence = float(row[target]), float(row[f"{target}__conf"])
            if verdict not in {"YES", "NO", "UNK"} or any(
                not math.isfinite(x) or not 0 <= x <= 1 for x in (score, confidence)
            ):
                raise ValueError("invalid fitting verdict/score/confidence")
            mask, label = original[f"{target}__mask"], original[target]
            if mask and label not in (0, 1):
                raise ValueError("observed original label must be binary")
            state = "masked" if not mask else "positive" if label == 1 else "negative"
            if state == "masked":
                masked_states[target][str(original[f"{target}__state"])] += 1
                reasons = json.loads(str(original.get(f"{target}__semantic_reasons", "[]")))
                if not isinstance(reasons, list) or any(not isinstance(r, str) for r in reasons):
                    raise ValueError("original semantic reasons must be a list of strings")
                masked_reasons[target].update(reasons)
            counts[target][f"{state}:{verdict}"] += 1
            scores[target][f"{state}:{verdict}:{score:g}"] += 1
            confidences[target][f"{verdict}:{confidence:g}"] += 1
            category = None
            if verdict == "YES" and score >= 0.9 and state in {"masked", "negative"}:
                category = f"{state}_upper_score_yes"
            elif (state, verdict) in {("positive", "YES"), ("negative", "NO")}:
                category = f"{state}_agreement"
            if category and target in {"MCL", "Effusion", "Baker's", "Fracture"}:
                candidates[target, category].append(
                    {
                        "study_uid": uid,
                        "leakage_group": original["leakage_group"],
                        "target": target,
                        "category": category,
                        "Report": original["Report"],
                        "original_mask": int(mask),
                        "original_label": int(label) if mask else None,
                        "original_state": original[f"{target}__state"],
                        "original_evidence": original[f"{target}__evidence"],
                        "semantic_reasons": str(original.get(f"{target}__semantic_reasons", "")),
                        "public_verdict": verdict,
                        "public_score": score,
                        "public_confidence": confidence,
                    }
                )
    if seen != set(cohort.index):
        raise ValueError("public source does not cover exact fitting cohort")
    review = []
    for key in sorted(candidates):
        selected_groups = set()
        for record in sorted(candidates[key], key=lambda x: sha(x["study_uid"].encode())):
            if len(selected_groups) == review_per_category:
                break
            if record["leakage_group"] not in selected_groups:
                selected_groups.add(record["leakage_group"])
                review.append(record)
    result = {
        "schema_version": 1,
        "status": "training_only_source_disagreement_diagnostic",
        "fitting_studies": len(cohort),
        "fitting_leakage_components": cohort.leakage_group.nunique(),
        "fitting_uid_sha256": sha(("\n".join(sorted(cohort.index)) + "\n").encode()),
        "targets": {
            t: {
                "state_verdict_counts": dict(sorted(counts[t].items())),
                "observed_score_counts": dict(sorted(scores[t].items())),
                "observed_confidence_counts": dict(sorted(confidences[t].items())),
                "masked_original_state_counts": dict(sorted(masked_states[t].items())),
                "masked_semantic_reason_counts": dict(sorted(masked_reasons[t].items())),
            }
            for t in TARGETS
        },
        "private_review_counts": dict(Counter(f"{r['target']}:{r['category']}" for r in review)),
        "accepted_labels_changed": False,
        "source_verdicts_are_ground_truth": False,
        "public_scores_are_calibrated_probabilities": False,
        "clinical_accuracy_claimed": False,
        "scope": "fit rows only; no outer/protected target interpretation or selection",
    }
    return result, review


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    parser.add_argument("--metadata", required=True, type=Path)
    parser.add_argument("--member", default="report_labels_v2.csv")
    parser.add_argument("--studies", required=True, type=Path)
    parser.add_argument("--folds", required=True, type=Path)
    parser.add_argument("--rubric", required=True, type=Path)
    parser.add_argument("--exclude-fold", type=int, default=0)
    parser.add_argument("--expected-studies", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--private-review-output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("diagnostic output already exists")
    if args.private_review_output and "artifacts" not in args.private_review_output.resolve().parts:
        raise ValueError("private review must stay in ignored artifacts")
    cohort = fitting_cohort(args.studies, args.folds, args.exclude_fold)
    if len(cohort) != args.expected_studies:
        raise ValueError("unexpected fixed fitting cohort size")
    content, archive_proof = csv_member(args.archive, args.member)
    result, review = compare(content, cohort)
    metadata = json.loads(args.metadata.read_text())
    result["provenance"] = {
        **archive_proof,
        "dataset_id": metadata["id"],
        "dataset_ref": metadata["ref"],
        "dataset_version": metadata["currentVersionNumber"],
        "license": metadata["licenseName"],
        "metadata_sha256": sha(args.metadata.read_bytes()),
        "studies_sha256": sha(args.studies.read_bytes()),
        "folds_sha256": sha(args.folds.read_bytes()),
        "rubric_sha256": sha(args.rubric.read_bytes()),
        "excluded_fold": args.exclude_fold,
        "diagnostic_script_sha256": sha(Path(__file__).read_bytes()),
    }
    if args.private_review_output:
        args.private_review_output.parent.mkdir(parents=True, exist_ok=True)
        fd = os.open(args.private_review_output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as stream:
            json.dump(review, stream, indent=2, ensure_ascii=False, allow_nan=False)
            stream.write("\n")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {"status": result["status"], "studies": len(cohort), "review_records": len(review)}
        )
    )


if __name__ == "__main__":
    main()
