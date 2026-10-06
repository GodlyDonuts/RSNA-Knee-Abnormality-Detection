"""SmMIL-style adjacent-slice smoothness arm: loss math, contract, real trainer."""

import argparse
import hashlib

import pandas as pd
import pytest
import torch
from torch.utils.data import Dataset

from rsna.competitive.loss import (
    adjacent_slice_pairs,
    slice_score_tv,
    validate_slice_smooth,
)
from rsna.competitive.train import TrainConfig
from rsna.schema import TARGETS


def _scores_mask(
    studies: int = 3, series: int = 2, slices: int = 4, targets: int = 12, seed: int = 0
):
    torch.manual_seed(seed)
    scores = torch.randn(studies, series, slices, targets)
    mask = torch.rand(studies, series, slices) > 0.25
    mask |= mask.new_ones(studies, series, 1).bool()  # at least one valid slice
    return scores, mask


def test_adjacent_slice_pairs_counts_consecutive_valid_slices():
    mask = torch.tensor(
        [
            [
                [True, True, True, False],
                [True, False, True, True],
            ]
        ]
    )
    # Series 0: pairs (0,1),(1,2) = 2; series 1: pair (2,3) = 1.
    assert adjacent_slice_pairs([mask]) == 3.0
    assert adjacent_slice_pairs([mask, mask]) == 6.0


def test_slice_score_tv_equals_mean_pairwise_squared_difference():
    scores, mask = _scores_mask()
    count = adjacent_slice_pairs([mask])
    term = slice_score_tv(scores, mask, count)
    pairs = mask[:, :, 1:] & mask[:, :, :-1]
    diffs = (scores[:, :, 1:, :].float() - scores[:, :, :-1, :].float()).pow(2)
    expected = ((diffs * pairs[..., None]).sum(dim=(0, 1, 2)) / count.to(diffs.dtype)).mean()
    torch.testing.assert_close(term, expected)


def test_slice_score_tv_is_zero_without_pairs():
    scores = torch.randn(2, 2, 3, 12)
    mask = torch.zeros(2, 2, 3, dtype=torch.bool)
    mask[:, :, 0] = True  # singleton slices only — no adjacent pairs
    count = adjacent_slice_pairs([mask])
    assert count == 0
    assert float(slice_score_tv(scores, mask, count)) == 0.0


def test_slice_score_tv_gradient_ignores_padded_slices():
    scores, mask = _scores_mask()
    mask[:, :, -1] = False  # last slice padded everywhere
    scores.requires_grad_(True)
    count = adjacent_slice_pairs([mask])
    slice_score_tv(scores, mask, count).backward()
    assert scores.grad is not None
    assert scores.grad[:, :, -1, :].abs().max() == 0
    assert scores.grad.abs().sum() > 0


def test_slice_score_tv_rejects_invalid_inputs():
    scores, mask = _scores_mask()
    count = adjacent_slice_pairs([mask])
    with pytest.raises(ValueError, match="aligned"):
        slice_score_tv(scores, mask[:, :1], count)
    with pytest.raises(ValueError, match="bool"):
        slice_score_tv(scores, mask.float(), count)
    with pytest.raises(ValueError, match="pair"):
        slice_score_tv(scores, mask, torch.tensor([-1.0]))
    with pytest.raises(ValueError, match="Non-finite"):
        bad = scores.clone()
        bad[0, 0, 0, 0] = float("nan")
        slice_score_tv(bad, mask, count)


def test_adjacent_slice_pairs_rejects_bad_masks():
    with pytest.raises(ValueError, match="microbatch"):
        adjacent_slice_pairs([])
    with pytest.raises(ValueError, match="boolean"):
        adjacent_slice_pairs([torch.ones(2, 2, 3)])


def test_validate_slice_smooth_accepts_valid_policy():
    validate_slice_smooth({"enabled": True, "kind": "attention_score_tv", "weight": 0.1})
    validate_slice_smooth({"enabled": False, "kind": "attention_score_tv", "weight": 1})


@pytest.mark.parametrize(
    "policy",
    [
        "smooth",
        {"enabled": True, "kind": "attention_score_tv"},
        {"enabled": True, "kind": "attention_score_tv", "weight": 0.1, "extra": 1},
        {"enabled": "yes", "kind": "attention_score_tv", "weight": 0.1},
        {"enabled": True, "kind": "logit_tv", "weight": 0.1},
        {"enabled": True, "kind": "attention_score_tv", "weight": 0},
        {"enabled": True, "kind": "attention_score_tv", "weight": 1.5},
        {"enabled": True, "kind": "attention_score_tv", "weight": "0.1"},
    ],
)
def test_validate_slice_smooth_rejects_invalid_policies(policy):
    with pytest.raises(ValueError, match="[Ss]lice-smooth"):
        validate_slice_smooth(policy)


def _config(**changes):
    return TrainConfig(
        diagnostic_study_limit=4,
        slices=2,
        side=32,
        max_series=3,
        **changes,
    )


def test_slice_smooth_config_must_be_an_isolated_attention_arm():
    arm = {"enabled": True, "kind": "attention_score_tv", "weight": 0.1}
    _config(slice_smooth=arm).validate()
    _config(slice_smooth={**arm, "enabled": False}).validate()
    with pytest.raises(ValueError, match="isolated outer-fold"):
        _config(slice_smooth=arm, pooling="max").validate()
    with pytest.raises(ValueError, match="isolated outer-fold"):
        _config(slice_smooth=arm, slice_context="neighbor3").validate()
    with pytest.raises(ValueError, match="isolated outer-fold"):
        _config(
            slice_smooth=arm,
            masked_cell={"enabled": True, "kind": "entropy_max", "weight": 0.05},
        ).validate()
    with pytest.raises(ValueError, match="isolated outer-fold"):
        _config(slice_smooth=arm, fit_scope="all_development").validate()


def test_slice_smooth_arm_trains_and_resumes_exactly(tmp_path, monkeypatch, capsys):
    from rsna.competitive import train
    from rsna.competitive.backbones import create_encoder
    from rsna.competitive.model import CompetitiveStudyNet

    class SyntheticStudies(Dataset):
        cache_side = 256
        source_revision = "synthetic"

        def __init__(self, studies, series, cache, augment=False, **kwargs):
            self.studies = studies.reset_index(drop=True)

        def __len__(self):
            return len(self.studies)

        def __getitem__(self, index):
            row = self.studies.iloc[index]
            return {
                "images": torch.rand(3, 4, 1, 16, 16),
                "features": torch.eye(3, 5),
                "series_mask": torch.ones(3, dtype=torch.bool),
                "slice_mask": torch.ones(3, 4, dtype=torch.bool),
                "positions": torch.zeros(3, 4, 2),
                "labels": torch.tensor([float(row[t]) for t in TARGETS]),
                "label_mask": torch.tensor([float(row[t + "__mask"]) for t in TARGETS]),
                "study_uid": row.study_uid,
            }

    rows = []
    for i in range(14):
        row = {"study_uid": f"study{i:02}", "leakage_group": f"group{i}", "split": "train"}
        for j, target in enumerate(TARGETS):
            row[target], row[target + "__mask"] = i % 2, int(j < 6 or i % 3 == 0)
            row[target + "__provenance"] = "synthetic_weak"
        rows.append(row)
    manifest = pd.DataFrame(rows)
    folds = manifest[["study_uid", "leakage_group"]].copy()
    folds["fold"] = [0] * 5 + [1] * 9
    series_path = tmp_path / "series.csv"
    series_path.write_text("SeriesInstanceUID\na\n")
    binding = {
        "series_sha256": hashlib.sha256(series_path.read_bytes()).hexdigest(),
        "source_revision": "synthetic",
    }
    monkeypatch.setattr(train, "load_verified_release", lambda *args: (manifest.copy(), binding))
    config = TrainConfig(
        slice_smooth={"enabled": True, "kind": "attention_score_tv", "weight": 0.1},
        diagnostic_study_limit=20,
        epochs=2,
        batch_size=2,
        accumulation_steps=2,
        slices=4,
        side=16,
        max_series=3,
        hidden_dim=16,
        freeze_encoder=False,
        augment=True,
        min_free_gib=0,
    )
    args = argparse.Namespace(
        studies=tmp_path / "unused",
        artifact_root=tmp_path,
        folds=tmp_path / "unused",
        series=series_path,
        cache=tmp_path,
        output=tmp_path / "full",
        resume=None,
        stop_after_updates=None,
    )

    def execute():
        return train._run_locked(
            args,
            config,
            [False],
            0,
            1,
            torch.device("cpu"),
            create_encoder,
            SyntheticStudies,
            CompetitiveStudyNet,
            lambda *a: folds.copy(),
        )

    assert execute() == 0
    full = torch.load(args.output / "checkpoint.pt", weights_only=True)
    out = capsys.readouterr().out
    assert '"slice_smooth"' in out and '"slice_pairs"' in out

    args.output = tmp_path / "resumed"
    args.stop_after_updates = 1
    assert execute() == 75
    args.resume = args.output / "checkpoint.pt"
    args.stop_after_updates = None
    assert execute() == 0
    resumed = torch.load(args.output / "checkpoint.pt", weights_only=True)
    assert full.keys() == resumed.keys()
    assert full["epoch"] == resumed["epoch"] and full["step"] == resumed["step"]
    for key in full["model"]:
        torch.testing.assert_close(full["model"][key], resumed["model"][key], rtol=0, atol=0)
