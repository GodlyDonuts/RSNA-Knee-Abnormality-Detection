"""Fixed equal probability mean; no labels, calibration, selection or filesystem I/O."""

import csv
import hashlib
import io
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


def fuse_csv(first_csv: bytes, second_csv: bytes, official_study_roster: Sequence[str]):
    """Equal probability mean, aligned by UID; no model or cohort-dependent ranking."""
    roster = _roster(official_study_roster)
    first, second = (_read_csv(data, roster) for data in (first_csv, second_csv))
    stream = io.StringIO(newline="")
    writer = csv.writer(stream, lineterminator="\n")
    writer.writerow(COLUMNS)
    for uid, a, b in zip(roster, first, second, strict=True):
        writer.writerow([uid, *(format((x + y) / 2, ".17g") for x, y in zip(a, b, strict=True))])
    output = stream.getvalue().encode()
    return output, {
        "rule": "equal_arithmetic_probability_mean",
        "weights": [0.5, 0.5],
        "studies": len(roster),
        "targets": 12,
        "input_CSV_sha256": [_sha256(first_csv), _sha256(second_csv)],
        "output_CSV_sha256": _sha256(output),
        "labels_read": False,
        "fallback_rows": 0,
        "accuracy_claim": None,
    }
