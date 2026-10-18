"""Fitting-roster source coverage and size only, never clinical labels or pixels."""

from __future__ import annotations

import hashlib
import json

import numpy as np

POPULATION_SHA = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"


def summarize(uids, source):
    h = hashlib.sha256(json.dumps(uids, separators=(",", ":")).encode()).hexdigest()
    if len(uids) != 3887 or uids != sorted(set(uids)) or h != POPULATION_SHA:
        raise ValueError("Exact original fitting population required")
    by_uid = {}
    for r in source:
        u = r["study_uid"]
        if not isinstance(u, str) or u in by_uid:
            raise ValueError("Unique original source study metadata required")
        values = [r[k] for k in ("bytes_sum", "bytes_count", "series_uid_count_distinct")]
        if any(type(x) is not int or x < 1 for x in values) or values[1] < values[2]:
            raise ValueError("Positive integer byte, file and series counts required")
        by_uid[u] = values
    present = [by_uid[u] for u in uids if u in by_uid]
    out = {
        "fitting_studies": len(uids),
        "fitting_roster_sha256": h,
        "source_covered_fitting_studies": len(present),
        "source_missing_fitting_studies": len(uids) - len(present),
        "all_fitting_studies_have_original_source_metadata": len(present) == len(uids),
        "raw_pixels_or_clinical_labels_read": False,
        "teacher_generation_or_training_admitted": False,
    }
    if present:
        a = np.asarray(present, dtype=np.int64)
        out["covered_original_bytes"] = int(a[:, 0].sum())
        out["covered_original_files"] = int(a[:, 1].sum())
        out["covered_original_series"] = int(a[:, 2].sum())
        out["per_study_quantiles"] = {
            name: {str(q): float(np.quantile(a[:, i], q)) for q in (0, 0.25, 0.5, 0.95, 1)}
            for i, name in enumerate(("bytes", "files", "series"))
        }
    return out
