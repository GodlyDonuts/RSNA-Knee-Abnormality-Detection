import copy
import random

import numpy as np
import pytest
import torch
from torch import nn

from rsna.competitive.compact_bag import (
    CompactBagClassifier,
    candidate_windows,
    sample_windows,
    window_tensor,
)
from rsna.competitive.compact_source_neighbors import (
    ENGINE,
    POLICY,
    channel_plan,
    select_window_tensor,
    source_window_tensor,
)
from rsna.competitive.compact_train import Settings, fit, model_windows, predict
from scripts.newton.run_compact_bag_recovery import exact


def geometry(count):
    return {
        "n_slices": count,
        "side": 4,
        "ordering": "patient_geometry",
        "orientation": [1, 0, 0, 0, 1, 0],
        "pixel_spacing": [0.5, 0.5],
        "positions": [[0, 0, 3 * j] for j in range(count)],
    }


def test_donors_are_original_neighbors_while_centers_and_count_stay_fixed():
    meta = geometry(40)
    triples, fallback = channel_plan(meta)
    selected = np.linspace(0, 39, 16).round().astype(np.int64)
    assert triples.shape == (14, 3) and not fallback.any()
    np.testing.assert_array_equal(triples[:, 1], selected[1:-1])
    np.testing.assert_array_equal(np.diff(triples, axis=1), np.ones((14, 2)))
    assert not np.array_equal(triples[0], selected[:3])
    volume = np.stack([np.full((4, 4), j * 1000, dtype=np.uint16) for j in range(40)])
    images, used = source_window_tensor(volume, meta, 4)
    assert used == 0 and images.shape == (14, 3, 4, 4)
    torch.testing.assert_close(
        images[:, :, 0, 0],
        torch.from_numpy(triples.astype(np.float32)) * 1000 / 65535,
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("count", [1, 2, 3, 11, 16])
def test_short_series_is_exact_legacy_tensor_identity(count):
    meta = geometry(count)
    rng = np.random.default_rng(4)
    volume = rng.integers(0, 65536, size=(count, 4, 4), dtype=np.uint16)
    actual, fallbacks = source_window_tensor(volume, meta, 4)
    old = torch.from_numpy(volume.astype(np.float32))[:, None] / 65535
    expected = (
        torch.stack([old[j : j + 3, 0] for j in range(count - 2)])
        if count >= 3
        else torch.empty(0, 3, 4, 4)
    )
    assert torch.equal(actual, expected) and fallbacks == 0


def test_missing_or_gapped_geometry_explicitly_retains_original_context():
    meta = geometry(40)
    meta["ordering"] = "instance_number"
    triples, fallback = channel_plan(meta)
    selected = np.linspace(0, 39, 16).round().astype(np.int64)
    assert fallback.all()
    np.testing.assert_array_equal(triples, np.stack([selected[j : j + 3] for j in range(14)]))
    meta = geometry(40)
    meta["positions"][3] = [0.1, 0, 9]
    triples, fallback = channel_plan(meta)
    assert fallback[0]  # Old first center=3 has shifted in-plane origin.
    np.testing.assert_array_equal(triples[0], selected[:3])
    meta = geometry(40)
    meta["positions"][9] = [0, 0, 30]  # repeated adjacent coordinate
    _, fallback = channel_plan(meta)
    assert fallback.all()


class Fixture:
    def __init__(self, adjacent):
        if adjacent:
            self.input_policy = dict(POLICY)
        generator = torch.Generator().manual_seed(41)
        self.rows = []
        for i in range(3):
            images = torch.rand(2, 16, 1, 4, 4, generator=generator)
            sample = {
                "study_uid": f"fixture-{i}",
                "images": images,
                "series_mask": torch.ones(2, dtype=torch.bool),
                "slice_mask": torch.ones(2, 16, dtype=torch.bool),
                "labels": torch.rand(12, generator=generator),
                "label_mask": torch.ones(12),
            }
            if adjacent:
                sample["source_window_images"] = torch.stack(
                    [torch.stack([images[s, j : j + 3, 0] for j in range(14)]) for s in range(2)]
                )
                del sample["images"]
            self.rows.append(sample)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1)
        self.bn = nn.BatchNorm2d(4)

    def forward(self, x):
        return self.bn(self.conv(x)).mean((2, 3))


def fresh():
    torch.manual_seed(9)
    np.random.seed(9)
    random.seed(9)
    return CompactBagClassifier(Encoder(), dimension=4)


def test_original_bag_rng_and_window_identity_are_unchanged():
    left, right = Fixture(False), Fixture(True)
    for i in range(3):
        a, b = left[i], right[i]
        for epoch in (1, 2, 16):
            windows = sample_windows(
                a["series_mask"],
                a["slice_mask"],
                run_seed=20260922,
                epoch=epoch,
                study_uid=a["study_uid"],
            )
            assert torch.equal(
                window_tensor(a["images"], windows), select_window_tensor(b, windows)
            )
        with pytest.raises(ValueError, match="explicit"):
            model_windows(b, candidate_windows(b["series_mask"], b["slice_mask"]), None)
        with pytest.raises(ValueError, match="outside"):
            select_window_tensor(b, [(0, 14)])


def test_real_engine_identity_and_full_stop_resume_with_separate_policy_binding(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        settings = Settings(epochs=2, effective_batch=2)
        device = torch.device("cpu")
        binding = {"diagnostic_only": True}
        policy_binding = {**binding, "input_policy": dict(POLICY)}
        baseline = fresh()
        fit(baseline, Fixture(False), tmp_path / "legacy", binding, settings, device=device)
        full = fresh()
        fit(full, Fixture(True), tmp_path / "full", policy_binding, settings, device=device)
        resumed = fresh()
        fit(
            resumed,
            Fixture(True),
            tmp_path / "resumed",
            policy_binding,
            settings,
            device=device,
            stop_after_updates=1,
        )
        resumed = fresh()
        fit(
            resumed,
            Fixture(True),
            tmp_path / "resumed",
            policy_binding,
            settings,
            device=device,
            resume=True,
        )
        states = [
            torch.load(tmp_path / n / "checkpoint.pt", weights_only=True)
            for n in ["legacy", "full", "resumed"]
        ]
        assert states[0]["binding"]["engine"] != ENGINE and states[1]["binding"]["engine"] == ENGINE
        exact(states[1], states[2])
        for key in [
            "model",
            "optimizer",
            "scheduler",
            "rng",
            "history",
            "epoch",
            "cursor",
            "optimizer_steps",
            "order",
        ]:
            exact(states[0][key], states[1][key])
        a = predict(baseline, Fixture(False), device)
        b = predict(full, Fixture(True), device)
        c = predict(resumed, Fixture(True), device)
        assert a.to_csv(index=False) == b.to_csv(index=False) == c.to_csv(index=False)
        wrong = copy.deepcopy(policy_binding)
        wrong["input_policy"]["version"] = "different"
        with pytest.raises(ValueError, match="binding"):
            fit(fresh(), Fixture(True), tmp_path / "bad", wrong, settings, device=device)
        with pytest.raises(ValueError, match="binding"):
            fit(
                fresh(),
                Fixture(True),
                tmp_path / "legacy",
                policy_binding,
                settings,
                device=device,
                resume=True,
            )
    finally:
        torch.set_num_threads(old_threads)
