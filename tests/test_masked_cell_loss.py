"""Masked-cell entropy objective (ESA arm) tests."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F

from rsna.competitive.loss import (
    masked_cell_consistency,
    masked_cell_entropy,
    observed_counts,
    validate_masked_cell,
)
from rsna.competitive.train import TrainConfig


def _batch(studies: int = 4, targets: int = 12, seed: int = 0):
    torch.manual_seed(seed)
    logits = torch.randn(studies, targets)
    mask = (torch.rand(studies, targets) > 0.5).float()
    return logits, mask


def test_masked_entropy_equals_bce_against_point_five_on_masked_cells():
    logits, mask = _batch()
    counts = observed_counts([(mask == 0).float()])
    term = masked_cell_entropy(logits, mask, counts)
    unobserved = (mask == 0).float()
    sums = (
        F.binary_cross_entropy_with_logits(logits, torch.full_like(logits, 0.5), reduction="none")
        * unobserved
    ).sum(dim=0)
    included = counts > 0
    expected = (sums[included] / counts[included].to(sums.dtype)).mean()
    torch.testing.assert_close(term, expected)


def test_masked_entropy_is_zero_when_everything_is_supervised():
    logits, mask = _batch()
    mask = torch.ones_like(mask)
    counts = observed_counts([(mask == 0).float()])
    term = masked_cell_entropy(logits, mask, counts)
    assert float(term) == 0.0


def test_masked_entropy_global_normalization_across_accumulation():
    logits, mask = _batch(studies=6)
    pieces = [(logits[:2], mask[:2]), (logits[2:4], mask[2:4]), (logits[4:], mask[4:])]
    counts = observed_counts([(m == 0).float() for _, m in pieces])
    whole = masked_cell_entropy(logits, mask, counts)
    summed = sum(masked_cell_entropy(p_l, p_m, counts) for p_l, p_m in pieces)
    torch.testing.assert_close(whole, summed)


def test_masked_entropy_gradient_reaches_only_masked_logits():
    logits, mask = _batch()
    logits.requires_grad_(True)
    counts = observed_counts([(mask == 0).float()])
    masked_cell_entropy(logits, mask, counts).backward()
    assert logits.grad is not None
    observed_grad = logits.grad[mask == 1]
    masked_grad = logits.grad[mask == 0]
    assert observed_grad.abs().max() == 0
    assert (masked_grad.abs() > 0).any()


def test_masked_entropy_rejects_invalid_shapes():
    logits, mask = _batch()
    counts = observed_counts([(mask == 0).float()])
    with pytest.raises(ValueError, match="aligned"):
        masked_cell_entropy(logits, mask[:, :4], counts)
    with pytest.raises(ValueError, match="masked counts"):
        masked_cell_entropy(logits, mask, torch.zeros(3))
    with pytest.raises(ValueError, match="Non-finite"):
        bad = logits.clone()
        bad[0, 0] = float("nan")
        masked_cell_entropy(bad, mask, counts)


def test_masked_consistency_equals_mean_mse_on_masked_cells():
    logits, mask = _batch()
    teacher = torch.randn_like(logits)
    counts = observed_counts([(mask == 0).float()])
    term = masked_cell_consistency(logits, teacher, mask, counts)
    unobserved = (mask == 0).float()
    sums = ((logits.float() - teacher.float()).pow(2) * unobserved).sum(dim=0)
    included = counts > 0
    expected = (sums[included] / counts[included].to(sums.dtype)).mean()
    torch.testing.assert_close(term, expected)


def test_masked_consistency_is_zero_when_everything_is_supervised():
    logits, mask = _batch()
    mask = torch.ones_like(mask)
    counts = observed_counts([(mask == 0).float()])
    term = masked_cell_consistency(logits, torch.randn_like(logits), mask, counts)
    assert float(term) == 0.0


def test_masked_consistency_gradient_reaches_only_masked_student_logits():
    logits, mask = _batch()
    logits.requires_grad_(True)
    teacher = torch.randn_like(logits)
    counts = observed_counts([(mask == 0).float()])
    masked_cell_consistency(logits, teacher, mask, counts).backward()
    assert logits.grad is not None
    assert logits.grad[mask == 1].abs().max() == 0
    assert (logits.grad[mask == 0].abs() > 0).any()


def test_masked_consistency_rejects_invalid_inputs():
    logits, mask = _batch()
    counts = observed_counts([(mask == 0).float()])
    teacher = torch.randn_like(logits)
    with pytest.raises(ValueError, match="aligned"):
        masked_cell_consistency(logits, teacher[:, :4], mask, counts)
    with pytest.raises(ValueError, match="Non-finite"):
        bad = teacher.clone()
        bad[0, 0] = float("nan")
        masked_cell_consistency(logits, bad, mask, counts)


def test_validate_masked_cell_accepts_valid_policy():
    validate_masked_cell({"enabled": True, "kind": "entropy_max", "weight": 0.1})
    validate_masked_cell({"enabled": False, "kind": "entropy_max", "weight": 1})
    validate_masked_cell(
        {
            "enabled": True,
            "kind": "ema_consistency",
            "weight": 1.0,
            "alpha": 0.999,
            "ramp_fraction": 0.2,
        }
    )


@pytest.mark.parametrize(
    "policy",
    [
        "entropy",
        {"enabled": True, "kind": "entropy_max"},
        {"enabled": True, "kind": "entropy_max", "weight": 0.5, "extra": 1},
        {"enabled": "yes", "kind": "entropy_max", "weight": 0.5},
        {"enabled": True, "kind": "entropy_min", "weight": 0.5},
        {"enabled": True, "kind": "entropy_max", "weight": 0},
        {"enabled": True, "kind": "entropy_max", "weight": 1.5},
        {"enabled": True, "kind": "entropy_max", "weight": "0.5"},
        {"enabled": True, "kind": "ema_consistency", "weight": 0.5},
        {
            "enabled": True,
            "kind": "ema_consistency",
            "weight": 0.5,
            "alpha": 0.999,
            "ramp_fraction": 0.2,
            "extra": 1,
        },
        {
            "enabled": True,
            "kind": "ema_consistency",
            "weight": 0.5,
            "alpha": 1.0,
            "ramp_fraction": 0.2,
        },
        {
            "enabled": True,
            "kind": "ema_consistency",
            "weight": 0.5,
            "alpha": 0.999,
            "ramp_fraction": 0,
        },
    ],
)
def test_validate_masked_cell_rejects_invalid_policies(policy):
    with pytest.raises(ValueError, match="[Mm]asked-cell"):
        validate_masked_cell(policy)


def _config(**changes):
    return TrainConfig(
        diagnostic_study_limit=4,
        slices=2,
        side=32,
        max_series=3,
        **changes,
    )


def test_masked_cell_config_must_be_an_isolated_arm():
    arm = {"enabled": True, "kind": "entropy_max", "weight": 0.05}
    _config(masked_cell=arm).validate()
    _config(masked_cell={**arm, "enabled": False}).validate()
    with pytest.raises(ValueError, match="isolated outer-fold"):
        _config(
            masked_cell=arm,
            fitting_supervision={
                "kind": "label_source_v1",
                "path": "overlay.json",
                "sha256": "a" * 64,
            },
        ).validate()
    with pytest.raises(ValueError, match="isolated outer-fold"):
        _config(masked_cell=arm, fit_scope="all_development").validate()


def _assert_tree_equal(first, second):
    assert type(first) is type(second)
    if isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            _assert_tree_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for left, right in zip(first, second):
            _assert_tree_equal(left, right)
    else:
        assert first == second


def test_ema_consistency_arm_trains_checkpoints_teacher_and_resumes_exactly(
    tmp_path, monkeypatch, capsys
):
    import argparse
    import hashlib

    import pandas as pd
    from torch.utils.data import Dataset

    from rsna.competitive import train
    from rsna.competitive.backbones import create_encoder
    from rsna.competitive.model import CompetitiveStudyNet
    from rsna.schema import TARGETS

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
                "images": torch.rand(3, 2, 1, 16, 16),
                "features": torch.eye(3, 5),
                "series_mask": torch.ones(3, dtype=torch.bool),
                "slice_mask": torch.ones(3, 2, dtype=torch.bool),
                "positions": torch.zeros(3, 2, 2),
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
    arm = {
        "enabled": True,
        "kind": "ema_consistency",
        "weight": 1.0,
        "alpha": 0.9,
        "ramp_fraction": 0.2,
    }
    config = TrainConfig(
        masked_cell=arm,
        diagnostic_study_limit=20,
        epochs=2,
        batch_size=2,
        accumulation_steps=2,
        slices=2,
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
    assert "teacher" in full
    for key, value in full["teacher"].items():
        student = full["model"].get(key)
        if isinstance(student, torch.Tensor) and student.is_floating_point():
            # The EMA shadow lags the trained student after updates.
            assert not torch.equal(value, student), key
    out = capsys.readouterr().out
    assert '"masked_consistency"' in out

    args.output = tmp_path / "resumed"
    args.stop_after_updates = 1
    assert execute() == 75
    args.resume = args.output / "checkpoint.pt"
    args.stop_after_updates = None
    assert execute() == 0
    resumed = torch.load(args.output / "checkpoint.pt", weights_only=True)
    _assert_tree_equal(full, resumed)
