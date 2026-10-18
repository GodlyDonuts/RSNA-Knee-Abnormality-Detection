"""New adapter integration with synthetic DICOMs; no clinical/GPU accuracy claim."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE), str(HERE.parent / "connected-witness-paths-20261017-v1")]
import competition_input as m


@pytest.fixture
def source(tmp_path):
    root = tmp_path / "competition"
    root.mkdir()
    study = "1.2.3"
    series = "1.2.3.4"
    folder = root / "test_series" / study / series
    folder.mkdir(parents=True)
    pd.DataFrame({"StudyInstanceUID": [study], "ACL": ["must-not-read"]}).to_csv(
        root / "test.csv", index=False
    )
    pd.DataFrame(
        {
            "StudyInstanceUID": [study],
            "SeriesInstanceUID": [series],
            "Anatomical_Plane": ["Sagittal"],
            "Fluid_Sensitive": [1],
            "Fat_Suppression": [1],
            "clinical_report": ["must-not-read"],
        }
    ).to_csv(root / "test_series.csv", index=False)
    for i in range(5):
        uid = series + "." + str(i + 1)
        path = folder / (uid + ".dcm")
        meta = FileMetaDataset()
        meta.TransferSyntaxUID = ExplicitVRLittleEndian
        meta.MediaStorageSOPClassUID = MRImageStorage
        meta.MediaStorageSOPInstanceUID = uid
        ds = FileDataset(str(path), {}, file_meta=meta, preamble=b"\0" * 128)
        ds.SOPClassUID = MRImageStorage
        ds.SOPInstanceUID = uid
        ds.StudyInstanceUID = study
        ds.SeriesInstanceUID = series
        ds.Modality = "MR"
        ds.Rows = 16
        ds.Columns = 16
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.BitsAllocated = 16
        ds.BitsStored = 16
        ds.HighBit = 15
        ds.PixelRepresentation = 0
        ds.ImagePositionPatient = [0, 0, float(i)]
        ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
        ds.PixelSpacing = [1, 1]
        ds.InstanceNumber = i + 1
        ds.PixelData = (np.arange(256, dtype=np.uint16).reshape(16, 16) + i).tobytes()
        ds.save_as(path, enforce_file_format=True)
    return root


class EncodedFixture:
    def __init__(self, dataset, parent, binding, **policy):
        assert list(dataset.studies) == ["study_uid"]
        assert all("clinical_report" not in r for rows in dataset.series.values() for r in rows)
        assert dataset.raw_split == "test_series" and dataset.cache_side == 384
        assert policy == {"sampling": "dense", "seed": 20261017, "chunk_size": 12}
        self.dataset = dataset

    def __getitem__(self, i):
        uid = self.dataset.studies.iloc[i].study_uid
        assert len(self.dataset._metadata) == 1
        return {
            "study_uid": uid,
            "maps": torch.zeros(3, 1),
            "base_logits": torch.zeros(12),
            "plan": None,
            "label_mask": torch.zeros(12),
        }


def test_new_adapter_actual_decoder_to_dataset_wiring(source, tmp_path, monkeypatch):
    monkeypatch.setattr(m, "StreamedFP32PathData", EncodedFixture)
    p = m.CompetitionPathProvider(source, tmp_path / "owned", object(), parent_sha256=m.PARENT)
    row = p[0]
    assert row["study_uid"] == "1.2.3" and not row["label_mask"].any()
    assert list((tmp_path / "owned").iterdir()) == []
    assert (
        p.statistics()["completed_image_studies"] == 1
        and p.statistics()["peak_owned_study_cache_bytes"] > 0
    )
    assert len(list((source / "test_series").rglob("*.dcm"))) == 5
    assert "clinical_report" not in p.frame and "ACL" not in p.frame


def test_adapter_cannot_replace_scratch(source, tmp_path):
    scratch = tmp_path / "occupied"
    scratch.mkdir()
    (scratch / "keep").write_text("original")
    with pytest.raises(FileExistsError):
        m.CompetitionPathProvider(source, scratch, object(), parent_sha256=m.PARENT)
    assert (scratch / "keep").read_text() == "original"


def test_adapter_bounds_decode_before_it_begins(source, tmp_path, monkeypatch):
    def unexpected(*a, **k):
        raise AssertionError("decoder must not run over budget")

    monkeypatch.setattr(m, "prepare_usable_series", unexpected)
    p = m.CompetitionPathProvider(
        source, tmp_path / "bounded", object(), parent_sha256=m.PARENT, capacity_bytes=2**20
    )
    with pytest.raises(OSError):
        p[0]
    assert list((tmp_path / "bounded").iterdir()) == []


def test_actual_test_roster_cannot_be_replaced_by_sample(source, tmp_path):
    (source / "test.csv").unlink()
    pd.DataFrame({"StudyInstanceUID": ["1.2.3"]}).to_csv(
        source / "sample_submission.csv", index=False
    )
    with pytest.raises(FileNotFoundError):
        m.CompetitionPathProvider(source, tmp_path / "no-roster", object(), parent_sha256=m.PARENT)


def test_reject_uid_path_traversal(source, tmp_path):
    pd.DataFrame({"StudyInstanceUID": ["../private"]}).to_csv(source / "test.csv", index=False)
    with pytest.raises(ValueError):
        m.CompetitionPathProvider(source, tmp_path / "unsafe", object(), parent_sha256=m.PARENT)
