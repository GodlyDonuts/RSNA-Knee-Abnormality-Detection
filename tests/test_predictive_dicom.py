"""Offline DICOM parity against the actual admitted source decoder/conversion."""

import hashlib
import json
import warnings
from pathlib import Path

import numpy as np
import pydicom
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

from rsna.competitive.predictive import inputs
from rsna.competitive.predictive.cache import normalized
from rsna.competitive.predictive.cases import RawGroup, assemble
from rsna.competitive.predictive.dicom import load_study

# Synthetic fixture namespace, deliberately unlike the source cohort's UID root.
STUDY = "2.25.202610200000000000000000000001"
SERIES_A = STUDY + ".1"
SERIES_B = STUDY + ".2"
REPO = Path(__file__).resolve().parents[1]


def write_dicom(root, series, ordinal, pixels, *, position=0, instance=1, **changes):
    sop = series + "." + str(ordinal)
    path = root / series / (sop + ".dcm")
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = sop
    ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = MRImageStorage
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID = STUDY, series, sop
    ds.Rows, ds.Columns = pixels.shape[-2:]
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
    ds.PixelRepresentation = int(np.issubdtype(pixels.dtype, np.signedinteger))
    ds.PhotometricInterpretation, ds.SamplesPerPixel = "MONOCHROME2", 1
    ds.InstanceNumber = instance
    ds.RescaleSlope, ds.RescaleIntercept = 2.5, -37.25
    if position is not None:
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.ImagePositionPatient = [0, 0, position]
    ds.PixelData = pixels.astype("<i2" if ds.PixelRepresentation else "<u2").tobytes()
    for name, value in changes.items():
        if value is None:
            del ds[name]
        else:
            setattr(ds, name, value)
    ds.save_as(path, enforce_file_format=True)
    return path


def fixture(root):
    a = (np.arange(80 * 80).reshape(80, 80) % 4096).astype(np.uint16)
    b = ((np.arange(64 * 64).reshape(64, 64) * 3) % 4096).astype(np.uint16)
    c = ((a.astype(np.uint32) + 12) % 4096).astype(np.uint16)
    d = (np.arange(64 * 64).reshape(64, 64) - 2048).astype(np.int16)
    return [
        write_dicom(root, SERIES_A, 1, a, position=20, instance=7),
        write_dicom(root, SERIES_A, 2, b, position=5, instance=9),
        write_dicom(root, SERIES_A, 3, c, position=0, instance=3),
        write_dicom(root, SERIES_A, 4, c, position=0, instance=3),
        write_dicom(
            root,
            SERIES_B,
            1,
            d,
            position=None,
            instance=-2,
            PhotometricInterpretation="MONOCHROME1",
        ),
    ]


def original_decoder(root, files):
    """Execute original admitted text; do not reimplement it using the adapter."""
    rows = []
    for path in files:
        raw = path.read_bytes()
        rows.append(
            {
                "study_uid": STUDY,
                "series_uid": path.parent.name,
                "sop_uid": path.stem,
                "split": "train_series",
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    source = (
        REPO / "initiative/research/predictive-raw-input-pilot-20261020-v1/program.py.txt"
    ).read_text()
    decoder = source[source.index("groups = {}") : source.index("roster = 0")]
    # Adapt only the source path/known one-case byte-count assertions.
    decoder = decoder.replace(
        'ram / "data/train_series" / uid / row["series_uid"]',
        'ram / row["series_uid"]',
    )
    decoder = decoder.replace(
        "assert file_bytes == 25131360", 'assert file_bytes == sum(r["bytes"] for r in rows)'
    )
    namespace = {
        "uid": STUDY,
        "rows": rows,
        "ram": root,
        "np": np,
        "pydicom": pydicom,
        "torch": torch,
        "hashlib": hashlib,
        "m": inputs,
    }
    exec(compile(decoder, "original_admitted_integer_decoder", "exec"), namespace)
    conversion = (
        REPO / "initiative/research/predictive-raw16-cache-20261020-v1/group_conversion.py.txt"
    ).read_text()
    exec(compile(conversion, "original_admitted_group_conversion", "exec"), namespace)
    return [
        RawGroup(normalized(g), g.aliases, g.acquisition, g.ordinal) for g in namespace["sources"]
    ]


def assert_same_groups(actual, expected):
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        for name in ("frames", "aliases", "acquisition", "ordinal"):
            assert torch.equal(getattr(left, name), getattr(right, name)), name
        assert left.frames.dtype == torch.float32
        assert left.frames.device.type == "cpu"
        assert left.descriptors is None


def assert_same_contexts(actual, expected):
    for left, right in zip(actual, expected):
        for key in ("images", "token_metadata", "region_metadata", "targets", "observed"):
            assert torch.equal(left[key], right[key]), key
        for key in ("indptr", "regions", "log_prior"):
            assert torch.equal(getattr(left["supports"], key), getattr(right["supports"], key))


@pytest.mark.parametrize("custom_source_order", [False, True])
def test_source_decoder_exact_mixed_matrix_alias_geometry_parity(tmp_path, custom_source_order):
    root = tmp_path / STUDY
    files = fixture(root)
    if custom_source_order:
        files = [files[4], files[2], files[0], files[3], files[1]]
    original_rng = torch.get_rng_state().clone()
    actual, stats = load_study(root, STUDY, ordered_files=files if custom_source_order else None)
    expected = original_decoder(root, files)
    assert_same_groups(actual, expected)
    # All complete target/support tensors match the real source path as well.
    assert_same_contexts(
        assemble(actual, side=32, halo=2, context_cap=3),
        assemble(expected, side=32, halo=2, context_cap=3),
    )
    assert torch.equal(original_rng, torch.get_rng_state())
    assert stats["series"] == 2
    assert stats["dicoms"] == 5
    assert stats["duplicate_frames"] == 1
    assert stats["unique_pixel_aliases"] == 4
    assert stats["matrix_groups"] == 2
    assert stats["geometry_frames"] == 4
    assert stats["instance_fallback_frames"] == 1
    assert stats["signed_frames"] == 1
    assert stats["source_bytes"] == sum(p.stat().st_size for p in files)
    assert STUDY not in json.dumps(stats) and str(tmp_path) not in json.dumps(stats)
    if not custom_source_order:
        # Critical: global acquisition ordinals, not the pilot's per-matrix ones.
        assert actual[0].ordinal.tolist() == [1, 1, 3]
        assert actual[1].ordinal.tolist() == [2, 0]
        signed = actual[1].frames[1]
        assert signed[0, 0].item() == 1.0 and signed[-1, -1].item() == 0.0
        # Positive rescale/intercept do not introduce image-derived normalization.
        original = pydicom.dcmread(files[0]).pixel_array.astype(np.float32)
        assert torch.equal(actual[0].frames[2], torch.from_numpy(original) / 4095)


def test_ordered_files_requires_complete_original_roster(tmp_path):
    root = tmp_path / STUDY
    files = fixture(root)
    with pytest.raises(ValueError, match="every DICOM"):
        load_study(root, STUDY, ordered_files=files[:-1])
    with pytest.raises(ValueError, match="every DICOM"):
        load_study(root, STUDY, ordered_files=files[:-1] + [files[0]])
    with pytest.raises(ValueError, match="complete study file roster"):
        load_study(root, STUDY, ordered_files=files[:-1] + [root / "unknown.dcm"])
    relative = [p.relative_to(root) for p in files]
    actual, _ = load_study(root, STUDY, ordered_files=relative)
    default, _ = load_study(root, STUDY)
    assert_same_groups(actual, default)


def test_geometry_bound_inputs_keep_physical_semantics_when_source_order_changes(tmp_path):
    root = tmp_path / STUDY
    files = fixture(root)
    default, _ = load_study(root, STUDY)
    reordered, _ = load_study(
        root, STUDY, ordered_files=[files[4], files[2], files[0], files[3], files[1]]
    )
    # Matrix/alias insertion IDs can differ, but this fixture has unambiguous
    # geometry (its only equal keys are true aliases within one acquisition).
    # Compare physical context and complete target regions by their metadata.
    a = assemble(default, side=32, halo=2, context_cap=3)
    b = assemble(reordered, side=32, halo=2, context_cap=3)
    for left, right in zip(a, b):
        for meta_name, value_name in (
            ("token_metadata", "images"),
            ("region_metadata", "targets"),
            ("region_metadata", "observed"),
        ):
            lmap = {
                tuple(meta.tolist()): value
                for meta, value in zip(left[meta_name], left[value_name])
            }
            rmap = {
                tuple(meta.tolist()): value
                for meta, value in zip(right[meta_name], right[value_name])
            }
            assert len(lmap) == len(left[meta_name])
            assert len(rmap) == len(right[meta_name])
            assert lmap.keys() == rmap.keys()
            assert all(torch.equal(value, rmap[key]) for key, value in lmap.items())


@pytest.mark.parametrize(
    "changes",
    [
        {"StudyInstanceUID": STUDY + ".9"},
        {"SeriesInstanceUID": SERIES_B},
        {"SOPInstanceUID": SERIES_A + ".99"},
        {"RescaleSlope": 0},
        {"RescaleSlope": -1},
        {"RescaleSlope": "nan"},
        {"RescaleIntercept": "inf"},
        {"BitsStored": 17, "HighBit": 16},
        {"PixelRepresentation": 2},
        {"PhotometricInterpretation": "PALETTE COLOR"},
        {"ImageOrientationPatient": [1, 0, 0]},
        {"ImagePositionPatient": None},
        {"ImageOrientationPatient": [0, 0, 0, 0, 0, 0]},
        {"ImagePositionPatient": [0, 0, "nan"]},
        {"InstanceNumber": None},
    ],
)
def test_rejects_malformed_identity_encoding_rescale_geometry(tmp_path, changes):
    root = tmp_path / STUDY
    pixels = np.arange(64 * 64).reshape(64, 64).astype(np.uint16)
    with warnings.catch_warnings():
        # pydicom may warn while writing deliberately nonconforming DS values.
        warnings.simplefilter("ignore", UserWarning)
        write_dicom(root, SERIES_A, 1, pixels, **changes)
    with pytest.raises(ValueError):
        load_study(root, STUDY)


def test_rejects_multiframe_float_and_unexpected_paths(tmp_path):
    root = tmp_path / STUDY
    pixels = np.ones((2, 64, 64), dtype=np.uint16)
    path = write_dicom(root, SERIES_A, 1, pixels, NumberOfFrames=2)
    with pytest.raises(ValueError, match="One monochrome"):
        load_study(root, STUDY)
    ds = pydicom.dcmread(path)
    del ds.PixelData
    del ds.NumberOfFrames
    ds.FloatPixelData = np.ones((64, 64), dtype="<f4").tobytes()
    ds.BitsAllocated = 32
    ds.save_as(path, enforce_file_format=True)
    with pytest.raises(ValueError, match="integer DICOM pixels"):
        load_study(root, STUDY)
    path.unlink()
    (path.parent / "unrecognized.txt").write_text("unexpected")
    with pytest.raises(ValueError, match="DICOM files only"):
        load_study(root, STUDY)
    with pytest.raises(ValueError, match="Valid DICOM UID"):
        load_study(root, "../invalid")


def test_exact_pixel_alias_with_inconsistent_encoding_is_rejected(tmp_path):
    root = tmp_path / STUDY
    pixels = np.ones((64, 64), dtype=np.uint16)
    write_dicom(root, SERIES_A, 1, pixels, PhotometricInterpretation="MONOCHROME1")
    write_dicom(root, SERIES_A, 2, pixels, PhotometricInterpretation="MONOCHROME2")
    with pytest.raises(ValueError, match="inconsistent encoding"):
        load_study(root, STUDY)
