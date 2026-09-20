from pathlib import Path

import pandas as pd
import pytest

from rsna.inventory import validate_metadata


def test_inventory_matches_study_and_series_metadata(tmp_path: Path) -> None:
    records = []
    for split, uid in [("train", "a"), ("test", "b")]:
        pd.DataFrame({"StudyInstanceUID": [uid]}).to_csv(tmp_path / f"{split}.csv", index=False)
        pd.DataFrame({"StudyInstanceUID": [uid], "SeriesInstanceUID": [f"{uid}-series"]}).to_csv(
            tmp_path / f"{split}_series.csv", index=False
        )
        records.append(
            {
                "split": f"{split}_series",
                "study_uid": uid,
                "series_uid": f"{uid}-series",
                "sop_uid": f"{uid}-sop",
                "path": f"{split}_series/{uid}/{uid}-series/{uid}-sop.dcm",
            }
        )
    inventory = pd.DataFrame(records)
    assert validate_metadata(inventory, tmp_path)["train"] == {
        "studies": 1,
        "series": 1,
        "dicoms": 1,
    }
    mismatch = inventory.copy()
    mismatch.loc[0, "series_uid"] = "unlisted-series"
    with pytest.raises(ValueError, match="original study/series metadata"):
        validate_metadata(mismatch, tmp_path)
    duplicate = pd.concat([inventory, inventory.iloc[:1]])
    with pytest.raises(ValueError, match="Duplicate source"):
        validate_metadata(duplicate, tmp_path)
