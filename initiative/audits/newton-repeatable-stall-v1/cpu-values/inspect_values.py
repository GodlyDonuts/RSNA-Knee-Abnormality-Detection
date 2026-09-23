"""Diagnose exact pixel bounds of the deterministic update-23 CPU inputs."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

SOURCE = Path("/home/sa305415/rsna-knee/code-stopfix-1f486765")
CONTROL = Path("/home/sa305415/rsna-knee/diagnostics/update23-values-v1")
REQUEST = Path("/home/sa305415/rsna-knee/diagnostics/update23-v1/request.json")
CACHE = Path("/home/sa305415/rsna-knee/diagnostics/update23-v1/cache")


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    result_path = CONTROL / "result.json"
    if result_path.exists():
        raise ValueError("Diagnostic result already exists")
    torch.set_num_threads(1)
    request = json.loads(REQUEST.read_text())
    assert request["source_git"] == "c58090e7c7a238461a5ac69293b32b64b74c6721"
    paths = {
        "studies.parquet": SOURCE / "initiative/releases/report-supervision-v1/studies.parquet",
        "fold.csv": SOURCE / "initiative/experiments/competitive-v1/folds/fold.csv",
        "series_sampling.csv": SOURCE
        / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv",
    }
    for name, path in paths.items():
        assert digest(path) == request["source_sha256"][name]
    assert sum(1 for path in CACHE.rglob("*") if path.is_file()) == 98

    sys.path.insert(0, str(SOURCE))
    from rsna.competitive.data import CompetitiveStudyDataset
    from rsna.competitive.train import select_development

    manifest = pd.read_parquet(paths["studies.parquet"])
    folds = pd.read_csv(paths["fold.csv"])
    series = pd.read_csv(paths["series_sampling.csv"])
    train, _ = select_development(manifest, folds, 0)
    assert len(train) == 2247
    order = torch.randperm(len(train), generator=torch.Generator().manual_seed(20260922)).tolist()
    selected = []
    for row in request["studies"]:
        index = order[22 * 8 + row["micro"] * 2 + row["rank"]]
        assert index == row["train_index"]
        assert train.iloc[index].study_uid == row["study_uid"]
        selected.append(train.iloc[index])
    dataset = CompetitiveStudyDataset(
        pd.DataFrame(selected), series, CACHE, slices=16, side=224, max_series=6
    )
    rows = []
    for index, row in enumerate(request["studies"]):
        sample = dataset[index]
        image = sample["images"]
        valid = image[sample["slice_mask"]]
        series_rows = []
        for slot in range(6):
            if not bool(sample["series_mask"][slot]):
                continue
            pixels = image[slot, sample["slice_mask"][slot]]
            series_rows.append(
                {
                    "slot": slot,
                    "min": float(pixels.min()),
                    "max": float(pixels.max()),
                    "nonfinite": int((~torch.isfinite(pixels)).sum()),
                    "below_zero": int((pixels < 0).sum()),
                    "above_one": int((pixels > 1).sum()),
                }
            )
        rows.append(
            {
                "study_uid": row["study_uid"],
                "rank": row["rank"],
                "micro": row["micro"],
                "image_sha256": hashlib.sha256(image.numpy().tobytes()).hexdigest(),
                "valid_slices": int(sample["slice_mask"].sum()),
                "min": float(valid.min()),
                "max": float(valid.max()),
                "nonfinite": int((~torch.isfinite(image)).sum()),
                "below_zero": int((image < 0).sum()),
                "above_one": int((image > 1).sum()),
                "series": series_rows,
            }
        )
    result = {
        "status": "exact_update23_pixel_bounds_inspected",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "request_sha256": digest(REQUEST),
        "source_git": request["source_git"],
        "study_rows": rows,
        "gpu_requested": False,
    }
    with result_path.open("x") as writer:
        json.dump(result, writer, sort_keys=True, indent=2)
        writer.write("\n")
        writer.flush()
    print(json.dumps({"status": result["status"], "studies": len(rows)}), flush=True)


if __name__ == "__main__":
    main()
