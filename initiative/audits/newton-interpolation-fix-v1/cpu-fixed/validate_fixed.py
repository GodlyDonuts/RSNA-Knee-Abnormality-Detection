"""Verify exact update-23 inputs with the committed range fix on Newton."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import torch

SOURCE = Path("/home/sa305415/rsna-knee/code-stopfix-1f486765")
CONTROL = Path("/home/sa305415/rsna-knee/diagnostics/update23-fixed-cpu-v1")
PRIOR = Path("/home/sa305415/rsna-knee/diagnostics/update23-values-v1/result.json")
REQUEST = Path("/home/sa305415/rsna-knee/diagnostics/update23-v1/request.json")
CACHE = Path("/home/sa305415/rsna-knee/diagnostics/update23-v1/cache")
COMMIT = "b74060b98f01bff1b924070ccba5ce44dd002f40"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = CONTROL / "result.json"
    if output.exists():
        raise ValueError("Fixed CPU result already exists")
    assert (
        subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"]).decode().strip()
        == COMMIT
    )
    assert (
        digest(SOURCE / "rsna/competitive/data.py")
        == "04f9b0e3c6e70c6a462775f130d8ec58c8d476233a690762bb2ea26097085aec"
    )
    assert digest(PRIOR) == "bd304d367b4c1f02ac48a58deb765fb635bc01405daa3104b093637ac85c0d69"
    request = json.loads(REQUEST.read_text())
    prior = json.loads(PRIOR.read_text())
    assert len(request["studies"]) == len(prior["study_rows"]) == 8
    paths = {
        "studies.parquet": SOURCE / "initiative/releases/report-supervision-v1/studies.parquet",
        "fold.csv": SOURCE / "initiative/experiments/competitive-v1/folds/fold.csv",
        "series_sampling.csv": SOURCE
        / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv",
    }
    for name, path in paths.items():
        assert digest(path) == request["source_sha256"][name]
    assert sum(1 for path in CACHE.rglob("*") if path.is_file()) == 98

    torch.set_num_threads(1)
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
    checked = []
    changed = []
    for index, row in enumerate(request["studies"]):
        sample = dataset[index]
        image = sample["images"]
        assert image.shape == (6, 16, 1, 224, 224)
        assert torch.isfinite(image).all() and (image >= 0).all() and (image <= 1).all()
        assert sample["series_mask"].any()
        assert torch.equal(sample["slice_mask"].any(dim=1), sample["series_mask"])
        image_sha256 = hashlib.sha256(image.numpy().tobytes()).hexdigest()
        previous = prior["study_rows"][index]
        assert (previous["rank"], previous["micro"], previous["study_uid"]) == (
            row["rank"],
            row["micro"],
            row["study_uid"],
        )
        if image_sha256 != previous["image_sha256"]:
            changed.append(index)
        checked.append(
            {
                "study_uid": row["study_uid"],
                "rank": row["rank"],
                "micro": row["micro"],
                "valid_slices": int(sample["slice_mask"].sum()),
                "min": float(image.min()),
                "max": float(image.max()),
                "image_sha256": image_sha256,
                "prior_image_sha256": previous["image_sha256"],
            }
        )
    assert changed == [5]
    assert checked[5]["max"] == 1.0 and prior["study_rows"][5]["above_one"] == 20
    result = {
        "status": "fixed_update23_cpu_inputs_passed",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": COMMIT,
        "request_sha256": digest(REQUEST),
        "prior_values_sha256": digest(PRIOR),
        "changed_study_indices": changed,
        "studies": checked,
        "gpu_requested": False,
    }
    with output.open("x") as writer:
        json.dump(result, writer, indent=2, sort_keys=True)
        writer.write("\n")
        writer.flush()
    print(json.dumps({"status": result["status"], "changed_study_indices": changed}))


if __name__ == "__main__":
    main()
