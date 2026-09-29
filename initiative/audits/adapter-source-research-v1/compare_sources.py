"""Describe public label agreement on fold-0 training rows; never import supervision."""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from rsna.schema import TARGETS


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(root: Path, downloads: Path) -> dict:
    manifest = root / "initiative/releases/report-supervision-v1/studies.parquet"
    fold_path = root / "initiative/experiments/competitive-v1/folds/fold.csv"
    assert digest(manifest) == "2bf49fd2a8ae0b7c5af7e8c105fc3f2337845a92654d136fe67bc41a1e52fae7"
    assert digest(fold_path) == "fe590e5072a0d82abeb377598aeef0021bbe77fe4529a19b9826a7d40d322050"
    columns = ["study_uid", "split"] + [c for t in TARGETS for c in (t, t + "__mask")]
    studies = pd.read_parquet(manifest, columns=columns)
    folds = pd.read_csv(fold_path)
    joined = studies.merge(folds[["study_uid", "fold"]], on="study_uid", validate="one_to_one")
    selected = joined[(joined["split"] == "train") & (joined["fold"] != 0)].copy()
    selected = selected.set_index("study_uid")
    assert len(selected) == 2603 and selected.index.is_unique
    frames, sources = {}, {}
    receipts = json.loads((downloads / "download-receipts.json").read_text())
    for receipt in receipts:
        name = receipt["path"]
        path = downloads / name
        assert digest(path) == receipt["sha256"] and path.stat().st_size == receipt["bytes"]
        frame = pd.read_csv(path, dtype={"StudyInstanceUID": str})
        assert list(frame.columns) == ["StudyInstanceUID", *TARGETS]
        assert frame["StudyInstanceUID"].is_unique
        # Reserved and outer-fold values are excluded before numeric inspection.
        frame = frame.set_index("StudyInstanceUID").reindex(selected.index)[list(TARGETS)]
        frames[name] = frame
        values = frame.to_numpy(dtype=float)
        finite = np.isfinite(values)
        assert not np.isinf(values).any()
        targets = {}
        for target in TARGETS:
            public = frame[target]
            observed = (selected[target + "__mask"] == 1) & public.notna()
            positive = observed & (selected[target] == 1)
            negative = observed & (selected[target] == 0)
            # 0.5 may encode missing evidence; it is neither positive nor negative.
            decided = observed & (public != 0.5)
            targets[target] = {
                "observed_overlap": int(observed.sum()),
                "our_positive": int(positive.sum()),
                "our_negative": int(negative.sum()),
                "public_exact_half_on_observed": int((observed & (public == 0.5)).sum()),
                "our_positive_public_above_half": int((positive & (public > 0.5)).sum()),
                "our_positive_public_below_half": int((positive & (public < 0.5)).sum()),
                "our_negative_public_above_half": int((negative & (public > 0.5)).sum()),
                "our_negative_public_below_half": int((negative & (public < 0.5)).sum()),
                "threshold_agreement_excluding_half": (
                    float(((public[decided] > 0.5) == selected.loc[decided, target]).mean())
                    if decided.any()
                    else None
                ),
            }
        sources[name] = {
            "receipt": receipt,
            "covered_studies": int(frame.notna().all(axis=1).sum()),
            "finite_cells": int(finite.sum()),
            "missing_cells": int(np.isnan(values).sum()),
            "minimum": float(values[finite].min()),
            "maximum": float(values[finite].max()),
            "cells_outside_unit_interval": int(((values < 0) | (values > 1)).sum()),
            "targets": targets,
        }
    pairs = []
    for (left_name, left), (right_name, right) in combinations(frames.items(), 2):
        per_target = {}
        for target in TARGETS:
            mask = left[target].notna() & right[target].notna()
            a, b = left.loc[mask, target].to_numpy(), right.loc[mask, target].to_numpy()
            per_target[target] = {
                "cells": len(a),
                "exact_equal": int((a == b).sum()),
                "all_exact_equal": bool(np.array_equal(a, b)),
            }
        mask = left.notna() & right.notna()
        a = left.to_numpy()[mask.to_numpy()]
        b = right.to_numpy()[mask.to_numpy()]
        pairs.append(
            {
                "sources": [left_name, right_name],
                "cells": len(a),
                "exact_equal_fraction": float((a == b).mean()),
                "per_target": per_target,
            }
        )
    return {
        "schema_version": 1,
        "status": "descriptive_source_audit_not_training_labels",
        "scope": "accepted training split, fold != 0; no outer/gold/confirmation/quarantine",
        "studies": len(selected),
        "manifest_sha256": digest(manifest),
        "fold_csv_sha256": digest(fold_path),
        "selected_uid_sha256": hashlib.sha256(
            ("\n".join(sorted(selected.index)) + "\n").encode()
        ).hexdigest(),
        "threshold_semantics": "0.5 excluded; other thresholds descriptive, not calibrated confidence",
        "independent_source_count": None,
        "labels_promoted_to_training": False,
        "sources": sources,
        "pairwise_exact_values": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--downloads", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.root, args.downloads)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"studies": result["studies"], "sources": len(result["sources"])}))


if __name__ == "__main__":
    main()
