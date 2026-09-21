from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd
import pytest
import torch

from rsna.data import StudyDataset, select_series


def candidates() -> list[dict]:
    return [
        {
            "StudyInstanceUID": "study",
            "SeriesInstanceUID": f"{plane}-{kind}",
            "Anatomical_Plane": plane,
            "Fluid_Sensitive": int(kind == "a"),
            "Fat_Suppression": int(kind == "a"),
            "Selection_Tier": int(kind == "a"),
        }
        for plane in ("Sagittal", "Coronal", "Axial")
        for kind in ("a", "b")
    ]


@pytest.mark.parametrize("augment", [False, True])
def test_reviewed_preference_survives_random_sampling_and_feature_priority(augment: bool) -> None:
    rows = candidates()
    before = copy.deepcopy(rows)
    for seed in range(20):
        torch.manual_seed(seed)
        selected = select_series(rows, 3, augment)
        assert [r["SeriesInstanceUID"] for r in selected] == ["Sagittal-b", "Coronal-b", "Axial-b"]
    assert rows == before


@pytest.mark.parametrize("augment", [False, True])
def test_alternatives_remain_available_with_capacity_and_plane_coverage(augment: bool) -> None:
    rows = candidates()
    selected = select_series(rows, 6, augment)
    assert len(selected) == 6
    assert {r["SeriesInstanceUID"] for r in selected} == {r["SeriesInstanceUID"] for r in rows}
    only_axial = [r for r in rows if r["SeriesInstanceUID"] != "Axial-b"]
    limited = select_series(only_axial, 3, augment)
    assert {r["Anatomical_Plane"] for r in limited} == {"Sagittal", "Coronal", "Axial"}
    assert next(r for r in limited if r["Anatomical_Plane"] == "Axial")["Selection_Tier"] == 1


@pytest.mark.parametrize("augment", [False, True])
def test_zero_tier_matches_absent_tier_and_random_state(augment: bool) -> None:
    rows = candidates()
    absent = [{k: v for k, v in r.items() if k != "Selection_Tier"} for r in rows]
    zeros = [{**r, "Selection_Tier": 0} for r in absent]
    torch.manual_seed(47)
    first = select_series(absent, 4, augment)
    first_rng = torch.get_rng_state()
    torch.manual_seed(47)
    second = select_series(zeros, 4, augment)
    assert [r["SeriesInstanceUID"] for r in first] == [r["SeriesInstanceUID"] for r in second]
    assert torch.equal(first_rng, torch.get_rng_state())
    if not augment:
        assert [r["SeriesInstanceUID"] for r in first] == [
            "Sagittal-a",
            "Coronal-a",
            "Axial-a",
            "Axial-b",
        ]


@pytest.mark.parametrize("tier", [True, 0.5, -1, 2, "1", None])
def test_invalid_tier_rejected_before_cache_access(tmp_path: Path, tier: object) -> None:
    rows = candidates()
    frame = pd.DataFrame(rows)
    frame["Selection_Tier"] = tier
    with pytest.raises(ValueError, match="Selection_Tier"):
        StudyDataset(pd.DataFrame({"study_uid": ["study"]}), frame, tmp_path)
