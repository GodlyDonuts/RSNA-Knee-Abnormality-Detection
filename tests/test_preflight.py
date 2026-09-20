from pathlib import Path

import pandas as pd

from rsna.preflight import FILES, preflight
from rsna.schema import TARGETS


def source_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    source, current = tmp_path / "source", tmp_path / "current"
    source.mkdir()
    current.mkdir()
    train = pd.DataFrame(
        [
            {"StudyInstanceUID": "study-a", "Report": "No effusion.", **dict.fromkeys(TARGETS)},
            {"StudyInstanceUID": "study-b", "Report": "NO  EFFUSION.", **dict.fromkeys(TARGETS)},
        ]
    )
    train.to_csv(source / "train.csv", index=False)
    pd.DataFrame({"StudyInstanceUID": ["study-test"]}).to_csv(source / "test.csv", index=False)
    pd.DataFrame([{"StudyInstanceUID": "study-test", **dict.fromkeys(TARGETS, 0.5)}]).to_csv(
        source / "sample_submission.csv", index=False
    )
    rows = []
    for split, ids in (("train", ["study-a", "study-b"]), ("test", ["study-test"])):
        pd.DataFrame(
            [
                {
                    "StudyInstanceUID": uid,
                    "SeriesInstanceUID": f"series-{uid}",
                    "Fluid_Sensitive": 1,
                    "Fat_Suppression": 0,
                    "Anatomical_Plane": "Sagittal",
                }
                for uid in ids
            ]
        ).to_csv(source / f"{split}_series.csv", index=False)
        for uid in ids:
            rows.append(
                {
                    "split": f"{split}_series",
                    "study_uid": uid,
                    "series_uid": f"series-{uid}",
                    "sop_uid": f"sop-{uid}",
                    "path": f"{split}_series/{uid}/series-{uid}/sop-{uid}.dcm",
                    "sha256": str(len(rows)) * 64,
                    "bytes": 100,
                    "storage": "lfs",
                }
            )
    for name in FILES:
        (current / name).write_bytes((source / name).read_bytes())
    inventory = tmp_path / "inventory.parquet"
    pd.DataFrame(rows).to_parquet(inventory, index=False)
    return source, current, inventory


def test_preflight_does_not_promote_metadata_to_training_readiness(tmp_path: Path) -> None:
    source, current, inventory = source_fixture(tmp_path)
    result = preflight(source, inventory, current, tmp_path / "result")
    assert result["source_metadata_valid"]
    assert not result["training_ready"]
    assert result["duplicate_report_groups"] == 1
    assert result["studies_in_duplicate_report_groups"] == 2
    # These flags are independent even when their real training values are correlated.
    assert result["checks"]["train_Fluid_Sensitive_binary"]
    assert result["checks"]["train_Fat_Suppression_binary"]


def test_preflight_blocks_stale_source_and_missing_series(tmp_path: Path) -> None:
    source, current, inventory = source_fixture(tmp_path)
    data = pd.read_csv(current / "train.csv")
    data.loc[0, "Report"] = "Updated official report."
    data.to_csv(current / "train.csv", index=False)
    pd.read_parquet(inventory).iloc[1:].to_parquet(inventory, index=False)
    result = preflight(source, inventory, current, tmp_path / "result")
    assert not result["source_metadata_valid"]
    assert not result["checks"]["current_train.csv_matches"]
    assert not result["checks"]["train_inventory_series_coverage"]
