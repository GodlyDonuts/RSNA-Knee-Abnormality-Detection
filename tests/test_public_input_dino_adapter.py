"""Source-public triplets feed the actual DINOv3 study-model interface."""

import pytest
import torch

from rsna.competitive.backbones import create_encoder
from rsna.competitive.data import collate_studies
from rsna.competitive.model import CompetitiveStudyNet
from tools.research.public_dinov2_input_reference.dino_adapter import study_inputs


def values():
    pixels = torch.zeros(6, 12, 336, 336, dtype=torch.uint8)
    for index in range(12):
        pixels[0, index] = index * 20
    mask = torch.tensor([True, False, False, False, False, False])
    features = torch.zeros(6, 5)
    features[0, 0] = 1
    positions = torch.zeros(6, 12, 2)
    positions[0, :, 0] = torch.arange(12) * 0.02
    positions[0, :, 1] = 1
    return [pixels, mask, features, positions, torch.linspace(0, 1, 12), torch.ones(12)]


def test_windows_preserve_source_channels_masks_and_geometry_without_double_normalization():
    v = values()
    result = study_inputs(*v, study_uid="synthetic")
    assert result["images"].shape == (6, 10, 3, 336, 336)
    assert torch.allclose(result["images"][0, 4, :, 0, 0], torch.tensor([80, 100, 120]) / 255)
    assert torch.equal(result["positions"], v[3][:, 1:11])
    assert result["slice_mask"][0].all() and not result["slice_mask"][1:].any()
    assert result["images"][1:].count_nonzero() == 0
    result["features"][0] = 0
    assert v[2][0, 0] == 1


@pytest.mark.parametrize(
    "change",
    [
        "wrong_shape",
        "float_pixels",
        "empty",
        "invented_position",
        "nonfinite",
        "absent_pixels",
        "absent_metadata",
        "bad_label",
        "bad_mask",
    ],
)
def test_incompatible_or_fabricated_source_inputs_fail_closed(change):
    v = values()
    if change == "wrong_shape":
        v[0] = v[0][:, :11]
    elif change == "float_pixels":
        v[0] = v[0].float()
    elif change == "empty":
        v[1].fill_(False)
    elif change == "invented_position":
        v[3][0, 0] = torch.tensor([1.0, 0.0])
    elif change == "nonfinite":
        v[2][0, 0] = float("nan")
    elif change == "absent_pixels":
        v[0][1, 0, 0, 0] = 1
    elif change == "absent_metadata":
        v[2][1, 0] = 1
    elif change == "bad_label":
        v[4][0] = float("nan")
    else:
        v[5][0] = 0.5
    with pytest.raises(ValueError):
        study_inputs(*v, study_uid="synthetic")


def test_actual_study_model_consumes_three_channel_views_and_backpropagates():
    torch.set_num_threads(1)
    torch.manual_seed(19)
    row = study_inputs(*values(), study_uid="synthetic")
    # Spatial reduction is only for this synthetic engineering test. The
    # source adapter's actual output remains336 pixels; no clinical cache uses16.
    row["images"] = torch.nn.functional.interpolate(
        row["images"].reshape(-1, 3, 336, 336), size=16
    ).reshape(6, 10, 3, 16, 16)
    batch = collate_studies([row])
    encoder, dim, _ = create_encoder(
        {"architecture": "diagnostic_tiny", "embedding_dim": 16}, input_channels=3
    )
    model = CompetitiveStudyNet(encoder, dim, hidden_dim=16, encode_chunk_size=4, input_channels=3)
    logits = model(
        batch["images"],
        batch["features"],
        series_mask=batch["series_mask"],
        slice_mask=batch["slice_mask"],
        positions=batch["positions"],
    )
    assert logits.shape == (1, 12) and torch.isfinite(logits).all()
    torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["labels"]).backward()
    grads = [p.grad for p in model.encoder.parameters()]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert sum(g.abs().sum() for g in grads) > 0


@pytest.mark.parametrize("side", ["left", "right"])
def test_original_render_pixels_and_physical_positions_are_aligned(tmp_path, side):
    from test_public_dinov2_input_reference import configured

    from tools.research.public_dinov2_input_reference import reference, synthetic
    from tools.research.public_dinov2_input_reference.dino_adapter import render_inputs

    ns, _ = configured(reference.SOURCE_PATH.read_bytes())
    study = synthetic.write_case(tmp_path, side, synthetic.MIRROR)
    plane_map = synthetic.plane_map(study, synthetic.MIRROR[side])
    public = reference.render_study(ns, study, plane_map)
    row = render_inputs(ns, study, plane_map, torch.zeros(12), torch.ones(12))
    expected = reference.windows(ns, public["images"]).permute(1, 0, 2, 3, 4).float() / 255
    assert torch.equal(row["images"], expected)
    assert torch.equal(row["series_mask"], public["slot_mask"])
    assert row["positions"][0, :, 1].eq(1).all()
    distances = row["positions"][0, :, 0]
    assert (
        (distances[1:] - distances[:-1]).ge(0).all()
        if side == "left"
        else (distances[1:] - distances[:-1]).le(0).all()
    )
    assert row["features"][0].tolist() == [1, 0, 0, 1, 1]
    assert row["features"][1].tolist() == [0, 1, 0, 1, 1]


def test_missing_physical_geometry_is_not_invented(tmp_path):
    import pydicom
    from test_public_dinov2_input_reference import configured

    from tools.research.public_dinov2_input_reference import reference, synthetic
    from tools.research.public_dinov2_input_reference.dino_adapter import render_inputs

    ns, _ = configured(reference.SOURCE_PATH.read_bytes())
    study = synthetic.write_case(tmp_path, "absent_slots")
    for path in study.rglob("*.dcm"):
        ds = pydicom.dcmread(path)
        del ds.ImagePositionPatient
        ds.save_as(path, enforce_file_format=True)
    row = render_inputs(
        ns,
        study,
        synthetic.plane_map(study, synthetic.CASES["absent_slots"]),
        torch.zeros(12),
        torch.ones(12),
    )
    assert row["positions"].count_nonzero() == 0


@pytest.mark.parametrize("index", [2, 3, 4, 5])
def test_source_metadata_and_targets_cannot_silently_change_precision(index):
    v = values()
    v[index] = v[index].double()
    with pytest.raises(ValueError):
        study_inputs(*v, study_uid="synthetic")
