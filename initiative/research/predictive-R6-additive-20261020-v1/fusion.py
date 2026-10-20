"""Fixed exploratory rank fusion; no labels, calibration, selection or filesystem I/O."""

import csv
import hashlib
import io
import json
import math
from collections.abc import Sequence

# Exact local competition schema from rsna.schema.TARGETS; no external source body.
COLUMNS = (
    "StudyInstanceUID",
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
REFERENCE_WEIGHT = 0.75
CANDIDATE_WEIGHT = 0.25


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _roster(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError("official roster must be an ordered sequence of study identifiers")
    result = tuple(value)
    if not result or any(
        not isinstance(uid, str) or not uid or uid != uid.strip() for uid in result
    ):
        raise ValueError("official roster has an invalid study identifier")
    if len(set(result)) != len(result):
        raise ValueError("official roster contains duplicate study identifiers")
    return result


def _read_csv(data: bytes, roster: tuple[str, ...]) -> list[list[float]]:
    if not isinstance(data, bytes):
        raise ValueError("prediction CSV must be immutable bytes")
    try:
        reader = csv.reader(io.StringIO(data.decode("utf-8"), newline=""), strict=True)
        if next(reader, None) != list(COLUMNS):
            raise ValueError("prediction CSV header does not match the canonical schema")
        rows = {}
        for row in reader:
            if len(row) != len(COLUMNS):
                raise ValueError("prediction CSV row has the wrong number of columns")
            uid = row[0]
            if not uid or uid != uid.strip() or uid in rows:
                raise ValueError("prediction CSV has an invalid or duplicate study identifier")
            try:
                scores = [float(cell) for cell in row[1:]]
            except ValueError:
                raise ValueError("prediction CSV has a nonnumeric score") from None
            if not all(math.isfinite(score) and 0 <= score <= 1 for score in scores):
                raise ValueError("prediction CSV scores must be finite and within [0,1]")
            rows[uid] = scores
    except (UnicodeDecodeError, csv.Error):
        raise ValueError("prediction CSV is not valid UTF-8 CSV") from None
    if set(rows) != set(roster):
        raise ValueError("prediction CSV study set differs from the full official roster")
    return [rows[uid] for uid in roster]


def _normalized_midranks(values: Sequence[float]) -> list[float]:
    """Ascending ranks, average exact ties, normalized as (rank - 0.5) / N."""
    ordered = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        normalized = (start + end) / (2 * len(values))
        for index in ordered[start:end]:
            ranks[index] = normalized
        start = end
    return ranks


def fuse_csv(
    reference_csv: bytes,
    candidate_csv: bytes,
    official_study_roster: Sequence[str],
) -> tuple[bytes, dict]:
    """Fuse full R6 and C posterior CSVs in the caller's exact official roster order.

    Model/version provenance and C's complete two-mask inference must be bound by
    the caller; CSV scores cannot establish either fact. This function refuses
    incomplete inputs and never substitutes rows or reads labels.
    """
    roster = _roster(official_study_roster)
    reference = _read_csv(reference_csv, roster)
    candidate = _read_csv(candidate_csv, roster)
    fused = [[] for _ in roster]
    for column in range(len(COLUMNS) - 1):
        reference_ranks = _normalized_midranks([row[column] for row in reference])
        candidate_ranks = _normalized_midranks([row[column] for row in candidate])
        for row, first, second in zip(fused, reference_ranks, candidate_ranks, strict=True):
            row.append(REFERENCE_WEIGHT * first + CANDIDATE_WEIGHT * second)

    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(COLUMNS)
    for uid, scores in zip(roster, fused, strict=True):
        writer.writerow([uid, *(format(score, ".17g") for score in scores)])
    output = stream.getvalue().encode("utf-8")
    roster_bytes = json.dumps(roster, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    receipt = {
        "schema_version": 1,
        "status": "fixed_exploratory_rank_fusion_computed",
        "reference_csv_sha256": _sha256(reference_csv),
        "candidate_csv_sha256": _sha256(candidate_csv),
        "official_ordered_roster_sha256": _sha256(roster_bytes),
        "output_csv_sha256": _sha256(output),
        "study_count": len(roster),
        "target_count": len(COLUMNS) - 1,
        "columns": list(COLUMNS),
        "reference_weight": REFERENCE_WEIGHT,
        "candidate_weight": CANDIDATE_WEIGHT,
        "rank_rule": "ascending average exact ties; (rank - 0.5) / N; independently per target",
        "model_provenance_verified_by_helper": False,
        "labels_read": False,
        "fallback_rows": 0,
        "accuracy_claim": None,
    }
    return output, receipt
