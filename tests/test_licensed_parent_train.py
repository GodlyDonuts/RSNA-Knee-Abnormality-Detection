from dataclasses import replace
from pathlib import Path

import pytest
import torch
from torch import nn

from rsna.competitive.checkpoint import capture_rng
from rsna.competitive.licensed_parent_train import (
    PARENT_SHA,
    Recipe,
    fit,
    lr_factor,
    sample,
)


class TinyMRI(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 4, 1),
            nn.BatchNorm2d(4),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(4, 12))

    def forward(self, x):
        b, k = x.shape[:2]
        f = self.backbone(x.flatten(0, 1)).view(b, k, 4).mean(1)
        return self.head(f)


class Cases:
    def __init__(self, n=15):
        self.uids = [f"synthetic-engineering-{i}" for i in range(n)]
        g = torch.Generator().manual_seed(71)
        self.images = torch.randn((n, 94, 3, 4, 4), generator=g)
        self.labels = torch.rand((n, 12), generator=g)
        self.teacher = torch.randn((n, 2, 12), generator=g)

    def __len__(self):
        return len(self.uids)

    def case(self, index, positions, mirror):
        x = self.images[index, positions]
        return {
            "images": x.flip(-1) if mirror else x,
            "labels": self.labels[index],
            "mask": torch.ones(12, dtype=torch.bool),
            "parent_logits": self.teacher[index, int(mirror)],
        }


def binding():
    return {"parent_checkpoint_sha256": PARENT_SHA, "diagnostic_only": True}


def equal(a, b):
    if isinstance(a, torch.Tensor):
        assert torch.equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    else:
        assert a == b


@pytest.mark.parametrize(
    "change",
    [
        {"epochs": True},
        {"windows": 95},
        {"windows": 0},
        {"effective_batch": 0},
        {"source_weight": 1.1},
        {"encoder_lr": float("nan")},
        {"warmup_fraction": 1},
        {"final_lr_fraction": 0},
    ],
)
def test_invalid_recipe_does_not_create_partial_fitting(change):
    with pytest.raises(ValueError):
        Recipe(**change)


def test_stateless_windows_preserve_original94_population_without_changing_global_rng():
    r = Recipe()
    before = capture_rng(torch.device("cpu"))
    a = sample("synthetic-MRI", r, 0)
    equal(before, capture_rng(torch.device("cpu")))
    assert a == sample("synthetic-MRI", r, 0)
    assert len(set(a[0])) == 12 and all(0 <= p < 94 for p in a[0])
    assert a != sample("synthetic-MRI", r, 1)
    assert a != sample("another-synthetic-MRI", r, 0)


def test_lr_has_finite_warmup_and_exact_low_final_endpoint():
    r = Recipe()
    rates = [lr_factor(i, 1458, r) for i in range(1458)]
    assert rates[0] == 1 / 146 and rates[145] == 1.0
    assert rates[-1] == r.final_lr_fraction
    assert all(0 < v <= 1 for v in rates)
    with pytest.raises(ValueError):
        lr_factor(1458, 1458, r)


def test_complete_model_adamw_history_and_all_rng_resume_exact_with_partial_batch(tmp_path):
    torch.set_num_threads(1)
    recipe = replace(Recipe(), epochs=2)
    data = Cases()  # 8 + 7 real cases: final batch must use its actual denominator.
    device = torch.device("cpu")
    torch.manual_seed(311)
    model = TinyMRI()
    initial = {k: v.clone() for k, v in model.state_dict().items()}
    result = fit(model, data, tmp_path / "full", binding(), recipe, device=device)
    assert result["status"] == "fixed_final_fit_complete" and result["completed"] == 4
    torch.manual_seed(311)
    stopped = TinyMRI()
    fit(stopped, data, tmp_path / "resumed", binding(), recipe, device=device, stop_after=1)
    # Construction/random work in a new worker may differ; loading restores all state.
    torch.manual_seed(9999)
    resumed = TinyMRI()
    fit(resumed, data, tmp_path / "resumed", binding(), recipe, device=device, resume=True)
    a = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    b = torch.load(tmp_path / "resumed/checkpoint.pt", weights_only=True)
    equal(a, b)
    assert [x["studies"] for x in a["history"]] == [8, 7, 8, 7]
    assert not torch.equal(initial["backbone.0.weight"], a["model"]["backbone.0.weight"])
    assert not torch.equal(initial["head.1.weight"], a["model"]["head.1.weight"])
    for name in (
        "backbone.1.running_mean",
        "backbone.1.running_var",
        "backbone.1.num_batches_tracked",
    ):
        assert torch.equal(initial[name], a["model"][name])
    for epoch in (1, 2):
        x = torch.load(tmp_path / f"full/epoch-checkpoints/epoch-{epoch:03d}.pt", weights_only=True)
        y = torch.load(
            tmp_path / f"resumed/epoch-checkpoints/epoch-{epoch:03d}.pt", weights_only=True
        )
        equal(x, y)
    with pytest.raises(ValueError, match="Existing fitting"):
        fit(TinyMRI(), data, tmp_path / "resumed", binding(), recipe, device=device)
    with pytest.raises(ValueError, match="Exact source"):
        fit(
            TinyMRI(),
            data,
            tmp_path / "resumed",
            {**binding(), "different": 1},
            recipe,
            device=device,
            resume=True,
        )


def test_rejects_unbound_parent_and_frozen_encoder_before_fitting(tmp_path):
    model = TinyMRI()
    with pytest.raises(ValueError, match="exact parent"):
        fit(model, Cases(), tmp_path, {}, Recipe(), device=torch.device("cpu"))
    model.backbone[0].weight.requires_grad_(False)
    with pytest.raises(ValueError, match="Every encoder"):
        fit(model, Cases(), tmp_path, binding(), Recipe(), device=torch.device("cpu"))
    assert not (Path(tmp_path) / "run.json").exists()


def test_actual_two_CPU_rank_fresh_process_recovery_including_odd_final_batch(tmp_path):
    import os
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "PYTHONNOUSERSITE": "1",
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }
    for mode in ("full", "stop", "resume"):
        x = subprocess.run(
            [
                sys.executable,
                str(root / "tests/fixtures/licensed_parent_ddp_worker.py"),
                str(tmp_path),
                mode,
                str(Path(__file__).resolve()),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert x.returncode == 0, x.stderr
    full = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    resumed = torch.load(tmp_path / "resumed/checkpoint.pt", weights_only=True)
    assert full["binding"]["world_size"] == 2
    equal(full, resumed)
    assert len(full["rng_by_rank"]) == len(full["scaler_by_rank"]) == 2
    assert [x["studies"] for x in full["history"]] == [8, 7, 8, 7]
