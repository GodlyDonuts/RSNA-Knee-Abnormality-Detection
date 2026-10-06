from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rsna.competitive.data import CompetitiveStudyDataset, _indices, collate_studies
from rsna.schema import CACHE_VERSION, TARGETS


def fixture(tmp_path: Path, counts=(3, 1), geometry=True):
    studies = pd.DataFrame(
        [
            {
                "study_uid": "1.2.3",
                TARGETS[0]: 1,
                f"{TARGETS[0]}__mask": 1,
                TARGETS[1]: float("nan"),
                f"{TARGETS[1]}__mask": 0,
            }
        ]
    )
    series = []
    paths = []
    for i, count in enumerate(counts):
        sid = f"1.2.3.{i + 1}"
        relative = f"train_series/1.2.3/{sid}.npz"
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        volume = np.stack(
            [np.full((16, 16), (j + 1) * 1000, dtype=np.uint16) for j in range(count)]
        )
        np.savez_compressed(path, volume=volume)
        sops = [f"{sid}.{j + 1}" for j in range(count)]
        rawpaths = sorted(f"train_series/1.2.3/{sid}/{sop}.dcm" for sop in sops)
        metadata = dict(
            cache_version=CACHE_VERSION,
            study_uid="1.2.3",
            series_uid=sid,
            split="train_series",
            side=16,
            source_revision="a" * 40,
            n_slices=count,
            original_shape=[count, 16, 16],
            sop_uids=sops,
            slice_sha256=[hashlib.sha256(sop.encode()).hexdigest() for sop in sops],
            path_fingerprint=hashlib.sha256("\n".join(rawpaths).encode()).hexdigest(),
            cache_path=relative,
            cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            cache_bytes=path.stat().st_size,
            ordering="patient_geometry" if geometry else "instance_number",
            positions=[[0, 0, j * 2] for j in range(count)] if geometry else [[]] * count,
            orientation=[1, 0, 0, 0, 1, 0] if geometry else [],
        )
        path.with_suffix(".json").write_text(json.dumps(metadata))
        paths.append(path)
        series.append(
            dict(
                StudyInstanceUID="1.2.3",
                SeriesInstanceUID=sid,
                Anatomical_Plane=["Sagittal", "Coronal", "Axial"][i % 3],
                Fluid_Sensitive=1,
                Fat_Suppression=0,
            )
        )
    return studies, pd.DataFrame(series), paths


def test_short_series_padding_true_positions_and_masked_labels(tmp_path):
    studies, series, _ = fixture(tmp_path)
    dataset = CompetitiveStudyDataset(studies, series, tmp_path, slices=5, side=16, max_series=3)
    row = dataset[0]
    assert row["images"].shape == (3, 5, 1, 16, 16)
    assert row["series_mask"].tolist() == [True, True, False]
    assert row["slice_mask"].sum(1).tolist() == [3, 1, 0]
    assert torch.equal(
        row["images"][~row["slice_mask"]], torch.zeros_like(row["images"][~row["slice_mask"]])
    )
    assert torch.allclose(row["positions"][0, :3, 0], torch.tensor([-0.02, 0, 0.02]))
    assert row["positions"][0, :3, 1].tolist() == [1, 1, 1]
    assert row["positions"][1, 0].tolist() == [0, 1]
    assert row["positions"][2].count_nonzero() == 0
    assert row["labels"][0] == 1 and row["label_mask"].sum() == 1
    assert torch.isfinite(row["labels"]).all() and row["labels"][1] == 0
    assert torch.allclose(row["images"][0, :3, 0, 0, 0], torch.tensor([1000, 2000, 3000]) / 65535)


def test_missing_geometry_does_not_invent_physical_positions(tmp_path):
    studies, series, _ = fixture(tmp_path, geometry=False)
    row = CompetitiveStudyDataset(studies, series, tmp_path, slices=2, side=16)[0]
    assert row["positions"].count_nonzero() == 0


def test_complete_coverage_uses_every_real_slice_and_series_even_above_caps(tmp_path):
    counts = (1, 2, 3, 5, 8, 11, 17)
    studies, series, _ = fixture(tmp_path, counts=counts)
    row = CompetitiveStudyDataset(
        studies,
        series,
        tmp_path,
        slices=2,
        max_series=3,
        side=16,
        slice_sampling="all",
        series_sampling="all",
    )[0]
    assert row["images"].shape == (7, 17, 1, 16, 16)
    assert row["series_mask"].all()
    assert sorted(row["slice_mask"].sum(1).tolist()) == sorted(counts)
    assert row["slice_mask"].sum() == sum(counts)
    for slot in range(7):
        count = row["slice_mask"][slot].sum().item()
        expected = torch.arange(1, count + 1) * 1000 / 65535
        torch.testing.assert_close(row["images"][slot, :count, 0, 0, 0], expected)
    assert row["label_mask"].sum() == 1 and row["labels"][0] == 1


def test_complete_coverage_padding_in_variable_batch_preserves_uid_masks_and_positions(tmp_path):
    first, first_series, _ = fixture(tmp_path / "first", counts=(2, 4))
    second, second_series, _ = fixture(tmp_path / "second", counts=(1, 3, 5, 7))
    second.loc[0, "study_uid"] = "1.2.3"  # separate fixture roots, no production UID overlap
    rows = [
        CompetitiveStudyDataset(
            studies, series, root, side=16, slice_sampling="all", series_sampling="all"
        )[0]
        for studies, series, root in [
            (first, first_series, tmp_path / "first"),
            (second, second_series, tmp_path / "second"),
        ]
    ]
    batch = collate_studies(rows)
    assert batch["images"].shape == (2, 4, 7, 1, 16, 16)
    assert batch["slice_mask"].sum((1, 2)).tolist() == [6, 16]
    for i, row in enumerate(rows):
        s, k = row["images"].shape[:2]
        for name in ["images", "positions", "slice_mask"]:
            torch.testing.assert_close(batch[name][i, :s, :k], row[name], rtol=0, atol=0)
        torch.testing.assert_close(batch["label_mask"][i], row["label_mask"], rtol=0, atol=0)
        assert batch["study_uid"][i] == row["study_uid"]
    assert not batch["images"][~batch["slice_mask"]].any()


def test_default_coverage_and_collation_remain_byte_equal(tmp_path):
    from torch.utils.data import default_collate

    studies, series, _ = fixture(tmp_path, counts=(5, 9))
    original = CompetitiveStudyDataset(studies, series, tmp_path, slices=3, side=16)[0]
    explicit = CompetitiveStudyDataset(
        studies,
        series,
        tmp_path,
        slices=3,
        side=16,
        slice_sampling="uniform",
        series_sampling="selected",
    )[0]
    for name, value in original.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(explicit[name], value, rtol=0, atol=0)
        else:
            assert explicit[name] == value
    expected, result = default_collate([original, explicit]), collate_studies([original, explicit])
    for name, value in expected.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(result[name], value, rtol=0, atol=0)
        else:
            assert result[name] == value


def test_interpolation_keeps_uint16_white_within_normalized_range(tmp_path):
    studies, series, paths = fixture(tmp_path, counts=(1,))
    volume = np.full((1, 256, 256), 65535, dtype=np.uint16)
    np.savez_compressed(paths[0], volume=volume)
    meta_path = paths[0].with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    meta.update(
        side=256,
        original_shape=[1, 256, 256],
        cache_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        cache_bytes=paths[0].stat().st_size,
    )
    meta_path.write_text(json.dumps(meta))
    image = CompetitiveStudyDataset(studies, series, tmp_path, slices=1, side=224)[0]["images"]
    assert torch.isfinite(image).all()
    assert image.min() >= 0 and image.max() <= 1
    assert torch.equal(image[0, 0], torch.ones_like(image[0, 0]))


def test_unique_sampling_for_all_short_and_long_series():
    for count in [1, 2, 3, 5, 17, 30]:
        for requested in [1, 2, 4, 16, 40]:
            for augment in [False, True]:
                values = _indices(count, requested, augment)
                assert len(values) == min(count, requested)
                assert len(set(values)) == len(values)
                assert (np.diff(values) > 0).all()
                assert values.min() >= 0 and values.max() < count


def test_augmentation_sampling_is_reproducible_and_does_not_flip(tmp_path):
    studies, series, paths = fixture(tmp_path, counts=(10,))
    volume = (
        np.broadcast_to(np.arange(16, dtype=np.uint16)[None, None, :], (10, 16, 16)).copy() * 1000
    )
    np.savez_compressed(paths[0], volume=volume)
    meta_path = paths[0].with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    meta.update(
        cache_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        cache_bytes=paths[0].stat().st_size,
    )
    meta_path.write_text(json.dumps(meta))
    dataset = CompetitiveStudyDataset(studies, series, tmp_path, slices=4, side=16, augment=True)
    torch.manual_seed(4)
    a = dataset[0]
    torch.manual_seed(4)
    b = dataset[0]
    assert torch.equal(a["images"], b["images"]) and torch.equal(a["positions"], b["positions"])
    assert a["images"][0, 0, 0, 0, 0] < a["images"][0, 0, 0, 0, -1]


@pytest.mark.parametrize("mutation", ["volume", "metadata"])
def test_changed_cache_fails_after_construction(tmp_path, mutation):
    studies, series, paths = fixture(tmp_path)
    dataset = CompetitiveStudyDataset(studies, series, tmp_path, slices=3, side=16)
    if mutation == "volume":
        paths[0].write_bytes(paths[0].read_bytes() + b"unexpected")
    else:
        path = paths[0].with_suffix(".json")
        data = json.loads(path.read_text())
        data["source_revision"] = "b" * 40
        path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="checksum|changed"):
        dataset[0]


@pytest.mark.parametrize(
    "mutation",
    [
        "wrong_uid",
        "wrong_revision",
        "wrong_split",
        "wrong_path",
        "duplicate_sop",
        "missing_hash",
        "wrong_fingerprint",
        "missing_geometry",
        "wrong_basis",
        "unordered",
    ],
)
def test_source_geometry_and_metadata_bindings_rejected(tmp_path, mutation):
    studies, series, paths = fixture(tmp_path)
    path = paths[0].with_suffix(".json")
    meta = json.loads(path.read_text())
    if mutation == "wrong_uid":
        meta["study_uid"] = "other"
    elif mutation == "wrong_revision":
        meta["source_revision"] = "b" * 40
    elif mutation == "wrong_split":
        meta["split"] = "test_series"
    elif mutation == "wrong_path":
        meta["cache_path"] = "other.npz"
    elif mutation == "duplicate_sop":
        meta["sop_uids"][1] = meta["sop_uids"][0]
    elif mutation == "missing_hash":
        meta["slice_sha256"].pop()
    elif mutation == "wrong_fingerprint":
        meta["path_fingerprint"] = "a" * 64
    elif mutation == "missing_geometry":
        meta["positions"].pop()
    elif mutation == "wrong_basis":
        meta["orientation"] = [1, 0, 0, 1, 0, 0]
    else:
        meta["positions"].reverse()
    path.write_text(json.dumps(meta))
    with pytest.raises(ValueError):
        CompetitiveStudyDataset(studies, series, tmp_path, side=16)


def test_bound_wrong_volume_shape_is_rejected(tmp_path):
    studies, series, paths = fixture(tmp_path)
    np.savez_compressed(paths[0], volume=np.ones((2, 16, 16), dtype=np.uint16))
    meta_path = paths[0].with_suffix(".json")
    meta = json.loads(meta_path.read_text())
    meta.update(
        cache_sha256=hashlib.sha256(paths[0].read_bytes()).hexdigest(),
        cache_bytes=paths[0].stat().st_size,
    )
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="shape"):
        CompetitiveStudyDataset(studies, series, tmp_path, side=16)[0]


def test_supervised_invalid_label_cannot_enter_loss(tmp_path):
    studies, series, _ = fixture(tmp_path)
    studies.loc[0, TARGETS[0]] = float("nan")
    with pytest.raises(ValueError, match="Supervised"):
        CompetitiveStudyDataset(studies, series, tmp_path, side=16)[0]


def test_graded_supervised_label_accepted(tmp_path):
    studies, series, _ = fixture(tmp_path)
    studies.loc[0, TARGETS[0]] = 0.5
    row = CompetitiveStudyDataset(studies, series, tmp_path, side=16)[0]
    assert row["labels"][0].item() == 0.5
    assert row["label_mask"][0].item() == 1.0


def test_out_of_range_supervised_label_rejected(tmp_path):
    studies, series, _ = fixture(tmp_path)
    studies.loc[0, TARGETS[0]] = 1.5
    with pytest.raises(ValueError, match="Supervised"):
        CompetitiveStudyDataset(studies, series, tmp_path, side=16)[0]
