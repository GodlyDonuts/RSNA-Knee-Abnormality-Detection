"""Recover useful numerics from a terminal scout without accepting its resources."""

from __future__ import annotations

import hashlib
import json

import numpy as np

POPULATION_SHA = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"


def compare(
    selection,
    pixels,
    probabilities,
    logits,
    old_selection,
    old_rows,
    old_logits,
    parent_uids,
    parent_logits,
    classifier,
    summarize,
):
    """Require exact cohort/input/parity; never manufacture timing or RSS proof."""
    population_sha = hashlib.sha256(
        json.dumps(parent_uids, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        len(parent_uids) != 3887
        or parent_uids != sorted(set(parent_uids))
        or population_sha != POPULATION_SHA
        or selection.get("population_sha256") != population_sha
        or old_selection.get("ordered_fitting_uids") != parent_uids
    ):
        raise ValueError("Original complete fitting population required")
    counts = dict(old_selection["series_counts"])
    if set(counts) != set(parent_uids) or any(type(n) is not int or n < 1 for n in counts.values()):
        raise ValueError("Original series counts required")
    costs = sorted((counts[u], u) for u in parent_uids)
    p8 = np.linspace(0, 3886, 8).round().astype(int).tolist()
    positions = sorted(set(p8 + np.linspace(0, 3886, 26).round().astype(int).tolist()))
    selected = [costs[i][1] for i in positions]
    old8 = [costs[i][1] for i in p8]
    if (
        selection.get("selected_uids") != selected
        or len(selected) != 32
        or selection.get("positions") != positions
        or selection.get("old8_selected_uids") != old8
        or old_selection.get("selected_uids") != old8
        or selection.get("series_counts") != [costs[i][0] for i in positions]
        or set(pixels) != set(selected)
        or set(old_logits) != set(old8)
    ):
        raise ValueError("Exact original fixed32 and contained8 required")
    if (
        classifier.get("status") != "PASS"
        or classifier.get("studies") != 32
        or classifier.get("scored") != 32
        or classifier.get("fallback") != {}
        or classifier.get("timed_out") is not False
        or classifier.get("amp") != "fp32"
        or classifier.get("models") != ["E_prior_s42_fall.pt"]
        or classifier.get("spatial_models") != 1
    ):
        raise ValueError("Actual complete original classifier receipt required")
    old = {r["uid"]: r for r in old_rows}
    if len(old) != 8 or set(old) != set(old8):
        raise ValueError("Original eight input records required")
    for u in selected:
        row = pixels[u]
        if (
            row.get("error") is not None
            or row.get("hash_error")
            or type(row.get("n")) is not int
            or row["n"] < 1
            or any(
                not isinstance(row.get(k), str)
                or len(row[k]) != 64
                or any(c not in "0123456789abcdef" for c in row[k])
                for k in ("img_sha256", "meta_sha256")
            )
        ):
            raise ValueError("Complete original input fingerprints required")
        if u in old and (
            old[u]["windows"] != row["n"]
            or old[u]["images_sha256"] != row["img_sha256"]
            or old[u]["meta_sha256"] != row["meta_sha256"]
        ):
            raise ValueError("Original eight MRI inputs differ")
    windows = sum(pixels[u]["n"] for u in selected)
    if classifier.get("stats", {}).get("windows") != windows:
        raise ValueError("Original classifier window accounting required")
    probabilities, logits, teacher = map(np.asarray, (probabilities, logits, parent_logits))
    reference = np.asarray([old_logits[u] for u in old8], dtype=np.float64)
    if (
        probabilities.shape != (32, 12)
        or logits.shape != (32, 12)
        or teacher.shape != (3887, 2, 12)
        or teacher.dtype != np.float32
        or reference.shape != (8, 1, 12)
        or any(not np.isfinite(x).all() for x in (probabilities, logits, teacher, reference))
    ):
        raise ValueError("Complete finite teacher arrays required")
    with np.errstate(over="ignore"):
        expected = 1 / (1 + np.exp(-logits))
        error = float(
            np.max(
                np.abs(
                    probabilities[[selected.index(u) for u in old8]]
                    - 1 / (1 + np.exp(-reference[:, 0]))
                )
            )
        )
        a = (
            1 / (1 + np.exp(-teacher[[parent_uids.index(u) for u in selected]].astype(np.float64)))
        ).mean(axis=1)
    if not np.allclose(probabilities, expected, rtol=0, atol=1e-7) or error > 0.0001:
        raise ValueError("Original probability and predeclared FP32 parity required")
    return {
        "studies": 32,
        "windows": windows,
        "maximum_original8_probability_error": error,
        "original8_input_fingerprints_equal": True,
        "descriptive_cross_lineage": summarize(a, probabilities),
        "original_terminal_failure_preserved": True,
        "original_RSS_time_and_total_output_cap_accepted": False,
        "full_cohort_teacher_generation_or_training_accepted": False,
        "clinical_accuracy_claim": False,
    }
