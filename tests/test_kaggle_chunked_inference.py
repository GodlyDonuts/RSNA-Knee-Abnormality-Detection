"""Bounded Kaggle scratch and exact test-row ordering, without remote calls."""

from __future__ import annotations

import importlib.util
import io
import json
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
import torch
from test_image_integrity import modified_dicom

from rsna.competitive import infer
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.data import select_series
from rsna.schema import TARGETS

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/kaggle/chunked_infer.py"
SPEC = importlib.util.spec_from_file_location("kaggle_chunked_infer", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
chunked = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(chunked)


def fixture(tmp_path: Path) -> Namespace:
    identifiers = ["1.2.5", "1.2.1", "1.2.4", "1.2.2", "1.2.3"]
    template = tmp_path / "sample_submission.csv"
    pd.DataFrame(
        [dict(StudyInstanceUID=uid, **{target: 0 for target in TARGETS}) for uid in identifiers],
        columns=["StudyInstanceUID", *TARGETS],
    ).to_csv(template, index=False)
    series = tmp_path / "test_series.csv"
    pd.DataFrame(
        [
            {
                "StudyInstanceUID": uid,
                "SeriesInstanceUID": uid + ".1",
                "Anatomical_Plane": "Sagittal",
            }
            for uid in reversed(identifiers)
        ]
    ).to_csv(series, index=False)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"test-only checkpoint identity")
    encoder = tmp_path / "config.json"
    encoder.write_text("{}")
    return Namespace(
        checkpoint=checkpoint,
        encoder_config=encoder,
        series=series,
        template=template,
        cache_root=tmp_path / "scratch",
        output=tmp_path / "submission.csv",
        dicom_root=None,
        source_revision="test",
        chunk_size=2,
        workers=0,
        device="cpu",
        precision="fp32",
        allow_diagnostic=False,
    )


def fake_infer(command: list[str], *, check: bool, reverse: bool = False) -> None:
    assert check and command[1:3] == ["-m", "rsna.competitive.infer"]

    def option(name: str) -> Path:
        return Path(command[command.index(name) + 1])

    template = pd.read_csv(option("--template"))
    series = pd.read_csv(option("--series"))
    assert set(template.StudyInstanceUID) == set(series.StudyInstanceUID)
    output = option("--output")
    cache = option("--cache")
    cache.mkdir()
    (cache / "volume.npz").write_bytes(b"v" * (1024 * len(template)))
    rows = template.iloc[::-1] if reverse else template
    frame = pd.DataFrame(
        [
            {
                "StudyInstanceUID": uid,
                **{target: int(uid.split(".")[-1]) % 10 / 10 for target in TARGETS},
            }
            for uid in rows.StudyInstanceUID
        ],
        columns=["StudyInstanceUID", *TARGETS],
    )
    frame.to_csv(output, index=False)
    metrics = {
        "status": "passed",
        "checkpoint_sha256": chunked.sha256(option("--checkpoint")),
        "studies": len(template),
        "diagnostic_only": False,
        "submission_sha256": chunked.sha256(output),
        "test_inputs": {
            "series_sha256": chunked.sha256(option("--series")),
            "template_sha256": chunked.sha256(option("--template")),
            "metadata_binding": "separate_test_inputs_not_checkpoint_bound",
        },
        "scheduled_epoch_authorized": True,
        "initial_pretrained_weights_read": False,
        "cache_source_revision": "test",
        "training_release_binding": {"binding_sha256": "a" * 64},
        "cache_version": "test",
        "cache_side": 256,
        "encoder": {"architecture": "dinov3_l16"},
        "training_precision": "fp32",
        "inference_precision": "fp32",
        "device_type": "cpu",
        "checkpoint_epoch": 2,
        "elapsed_seconds": 0.1,
    }
    output.with_suffix(".metrics.json").write_text(json.dumps(metrics))


def test_chunking_preserves_template_order_and_cleans_each_cache(tmp_path):
    args = fixture(tmp_path)
    result = chunked.run(args, invoke=fake_infer)
    output = pd.read_csv(args.output)
    template = pd.read_csv(args.template)
    assert output.StudyInstanceUID.tolist() == template.StudyInstanceUID.tolist()
    assert output[TARGETS[0]].tolist() == [0.5, 0.1, 0.4, 0.2, 0.3]
    assert result["chunk_count"] == 3 and result["studies"] == 5
    assert [row["studies"] for row in result["chunks"]] == [2, 2, 1]
    assert result["peak_chunk_cache_bytes"] == 2048
    assert not list(args.cache_root.rglob("*.npz"))
    assert result["submission_sha256"] == chunked.sha256(args.output)


def test_duplicate_series_across_chunks_rejected_before_scratch(tmp_path):
    args = fixture(tmp_path)
    series = pd.read_csv(args.series)
    series.loc[4, "SeriesInstanceUID"] = series.loc[0, "SeriesInstanceUID"]
    series.to_csv(args.series, index=False)
    with pytest.raises(ValueError, match="duplicate test series"):
        chunked.run(args, invoke=fake_infer)
    assert not args.cache_root.exists() and not args.output.exists()


def test_misordered_chunk_output_never_becomes_a_submission(tmp_path):
    args = fixture(tmp_path)
    with pytest.raises(ValueError, match="order differ"):
        chunked.run(
            args, invoke=lambda command, **kwargs: fake_infer(command, reverse=True, **kwargs)
        )
    assert not args.output.exists()


def test_hidden_test_roster_replaces_three_row_public_sample(tmp_path):
    args = fixture(tmp_path)
    sample = pd.read_csv(args.template).iloc[:3]
    sample["StudyInstanceUID"] = ["9.9.1", "9.9.2", "9.9.3"]
    sample.to_csv(args.template, index=False)
    args.test_studies = tmp_path / "test.csv"
    expected_ids = ["1.2.5", "1.2.1", "1.2.4", "1.2.2", "1.2.3"]
    pd.DataFrame({"StudyInstanceUID": expected_ids}).to_csv(args.test_studies, index=False)
    result = chunked.run(args, invoke=fake_infer)
    assert pd.read_csv(args.output).StudyInstanceUID.tolist() == expected_ids
    assert result["studies"] == 5
    assert (
        result["test_inputs"]["metadata_binding"] == "test_csv_authoritative_not_checkpoint_bound"
    )
    assert result["test_inputs"]["sample_template_sha256"] == chunked.sha256(args.template)


def test_hidden_test_roster_works_without_sample_submission(tmp_path):
    args = fixture(tmp_path)
    args.test_studies = tmp_path / "test.csv"
    args.template = None
    expected_ids = ["1.2.5", "1.2.1", "1.2.4", "1.2.2", "1.2.3"]
    pd.DataFrame({"StudyInstanceUID": expected_ids}).to_csv(args.test_studies, index=False)
    result = chunked.run(args, invoke=fake_infer)
    assert pd.read_csv(args.output).StudyInstanceUID.tolist() == expected_ids
    assert result["test_inputs"]["sample_template_sha256"] is None


def test_thirteen_hundred_study_roster_stays_chunked(tmp_path):
    args = fixture(tmp_path)
    args.test_studies = tmp_path / "test.csv"
    args.chunk_size = 32
    expected_ids = [f"1.2.{i}" for i in range(1, 1302)]
    pd.DataFrame({"StudyInstanceUID": expected_ids}).to_csv(args.test_studies, index=False)
    pd.DataFrame(
        {
            "StudyInstanceUID": expected_ids,
            "SeriesInstanceUID": [uid + ".1" for uid in expected_ids],
            "Anatomical_Plane": "Sagittal",
        }
    ).to_csv(args.series, index=False)
    result = chunked.run(args, invoke=fake_infer)
    assert result["studies"] == 1301 and result["chunk_count"] == 41
    assert result["peak_chunk_cache_bytes"] == 32 * 1024
    assert pd.read_csv(args.output).StudyInstanceUID.tolist() == expected_ids
    assert not list(args.cache_root.rglob("*.npz"))


def test_actual_checkpoint_inference_matches_full_three_study_run(tmp_path, monkeypatch):
    inference_test = Path(__file__).with_name("test_competitive_inference.py")
    spec = importlib.util.spec_from_file_location("competitive_inference_fixture", inference_test)
    assert spec is not None and spec.loader is not None
    fixture_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fixture_module)
    original, _ = fixture_module._fixture(
        tmp_path, monkeypatch, checkpoint_only=True, precision="fp32", size=3
    )
    original.scheduled_epoch = True
    infer.run(original)
    expected = pd.read_csv(original.output)
    args = Namespace(
        checkpoint=original.checkpoint,
        encoder_config=original.encoder_config,
        series=original.series,
        template=original.template,
        cache_root=tmp_path / "chunk-scratch",
        output=tmp_path / "chunked-submission.csv",
        dicom_root=None,
        source_revision="synthetic-test",
        chunk_size=2,
        workers=0,
        device="cpu",
        precision="fp32",
        allow_diagnostic=True,
    )

    def invoke(command: list[str], *, check: bool) -> None:
        assert check

        def option(name: str) -> Path:
            return Path(command[command.index(name) + 1])

        cache = option("--cache")
        for uid in pd.read_csv(option("--template")).StudyInstanceUID:
            source = original.cache / "test_series" / uid
            destination = cache / "test_series" / uid
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, destination)
        selected = Namespace(**vars(original))
        selected.series = option("--series")
        selected.template = option("--template")
        selected.cache = cache
        selected.output = option("--output")
        selected.source_revision = args.source_revision
        infer.run(selected)

    result = chunked.run(args, invoke=invoke)
    actual = pd.read_csv(args.output)
    assert actual.StudyInstanceUID.tolist() == expected.StudyInstanceUID.tolist()
    assert np.allclose(actual[list(TARGETS)].to_numpy(), expected[list(TARGETS)].to_numpy())
    assert result["chunk_count"] == 2 and result["studies"] == 3
    assert result["diagnostic_only"] is True


def test_resilient_selection_ignores_unneeded_series_and_preserves_model_inputs(
    tmp_path, monkeypatch
):
    study = "1.2.5"
    rows = [
        {
            "StudyInstanceUID": study,
            "SeriesInstanceUID": f"{study}.{index}",
            "Anatomical_Plane": ("Sagittal", "Coronal", "Axial")[index % 3],
            "Fluid_Sensitive": index % 2,
            "Fat_Suppression": index % 2,
        }
        for index in range(1, 9)
    ]
    frame = pd.DataFrame(rows)
    root = tmp_path / "dicoms"
    for row in rows:
        folder = root / "test_series" / study / row["SeriesInstanceUID"]
        folder.mkdir(parents=True)
        (folder / "slice.dcm").write_bytes(b"synthetic")
    seen = []

    def cache(_source, paths, _cache, side, revision):
        assert side == 256 and revision == "test"
        seen.append(Path(paths[0]).parts[-2])

    monkeypatch.setattr(chunked, "cache_series", cache)
    selected, receipt = chunked.prepare_usable_series(
        frame,
        [study],
        dicom_root=root,
        cache=tmp_path / "cache",
        cache_side=256,
        max_series=6,
        source_revision="test",
    )
    expected = {row["SeriesInstanceUID"] for row in select_series(rows, 6, False)}
    assert set(selected.SeriesInstanceUID) == expected == set(seen)
    assert receipt == {
        "selected_series": 6,
        "skipped_series": {"missing_dicoms": 0, "invalid_dicoms": 0, "invalid_series": 0},
    }


def test_resilient_selection_uses_same_plane_alternate_after_bad_dicom(tmp_path, monkeypatch):
    study = "1.2.5"
    rows = [
        {
            "StudyInstanceUID": study,
            "SeriesInstanceUID": f"{study}.{index}",
            "Anatomical_Plane": plane,
            "Fluid_Sensitive": 1 if index == 1 else 0,
            "Fat_Suppression": 0,
        }
        for index, plane in enumerate(("Sagittal", "Sagittal", "Coronal", "Axial"), 1)
    ]
    frame = pd.DataFrame(rows)
    root = tmp_path / "dicoms"
    for row in rows:
        folder = root / "test_series" / study / row["SeriesInstanceUID"]
        folder.mkdir(parents=True)
        (folder / "slice.dcm").write_bytes(b"synthetic")

    def cache(_source, paths, *_args):
        if Path(paths[0]).parts[-2] == f"{study}.1":
            raise ValueError("Constant-intensity series")

    monkeypatch.setattr(chunked, "cache_series", cache)
    selected, receipt = chunked.prepare_usable_series(
        frame,
        [study],
        dicom_root=root,
        cache=tmp_path / "cache",
        cache_side=256,
        max_series=6,
        source_revision="test",
    )
    assert set(selected.SeriesInstanceUID) == {f"{study}.{i}" for i in (2, 3, 4)}
    assert receipt["skipped_series"]["invalid_series"] == 1


def test_resilient_selection_never_hides_cache_integrity_errors(tmp_path, monkeypatch):
    study = "1.2.5"
    sid = f"{study}.1"
    folder = tmp_path / "dicoms" / "test_series" / study / sid
    folder.mkdir(parents=True)
    (folder / "slice.dcm").write_bytes(b"synthetic")
    frame = pd.DataFrame(
        [
            {
                "StudyInstanceUID": study,
                "SeriesInstanceUID": sid,
                "Anatomical_Plane": "Sagittal",
                "Fluid_Sensitive": 0,
                "Fat_Suppression": 0,
            }
        ]
    )

    def cache(*_args):
        raise ValueError("Stale/corrupted cache: deliberate fault")

    monkeypatch.setattr(chunked, "cache_series", cache)
    with pytest.raises(ValueError, match="Stale/corrupted cache"):
        chunked.prepare_usable_series(
            frame,
            [study],
            dicom_root=tmp_path / "dicoms",
            cache=tmp_path / "cache",
            cache_side=256,
            max_series=6,
            source_revision="test",
        )


def test_real_dicom_fallback_produces_bound_model_cache(tmp_path):
    study = "2.3"
    planes = ("Sagittal", "Sagittal", "Coronal", "Axial")
    frame = pd.DataFrame(
        [
            {
                "StudyInstanceUID": study,
                "SeriesInstanceUID": f"{study}.{index}",
                "Anatomical_Plane": plane,
                "Fluid_Sensitive": 1 if index == 1 else 0,
                "Fat_Suppression": 0,
            }
            for index, plane in enumerate(planes, 1)
        ]
    )
    root = tmp_path / "dicoms"
    for row in frame.to_dict("records"):
        sid = row["SeriesInstanceUID"]
        folder = root / "test_series" / study / sid
        folder.mkdir(parents=True)
        data = (
            b"invalid DICOM"
            if sid == f"{study}.1"
            else modified_dicom(
                StudyInstanceUID=study,
                SeriesInstanceUID=sid,
                SOPInstanceUID=f"{sid}.1",
            )
        )
        (folder / f"{sid}.1.dcm").write_bytes(data)
    cache = tmp_path / "cache"
    selected, receipt = chunked.prepare_usable_series(
        frame,
        [study],
        dicom_root=root,
        cache=cache,
        cache_side=16,
        max_series=3,
        source_revision="test",
    )
    assert set(selected.SeriesInstanceUID) == {f"{study}.{i}" for i in (2, 3, 4)}
    assert receipt["skipped_series"]["invalid_dicoms"] == 1
    dataset = CompetitiveStudyDataset(
        pd.DataFrame({"study_uid": [study]}),
        selected,
        cache,
        slices=1,
        side=16,
        max_series=3,
        raw_split="test_series",
    )
    item = dataset[0]
    assert item["series_mask"].tolist() == [True, True, True]
    assert item["images"].shape == (3, 1, 1, 16, 16)


def test_truncated_pixel_payload_uses_healthy_same_plane_alternate(tmp_path):
    study = "2.3"
    frame = pd.DataFrame(
        [
            {
                "StudyInstanceUID": study,
                "SeriesInstanceUID": f"{study}.{index}",
                "Anatomical_Plane": plane,
                "Fluid_Sensitive": 1 if index == 1 else 0,
                "Fat_Suppression": 0,
            }
            for index, plane in enumerate(("Sagittal", "Sagittal", "Coronal", "Axial"), 1)
        ]
    )
    root = tmp_path / "dicoms"
    for row in frame.to_dict("records"):
        sid = row["SeriesInstanceUID"]
        folder = root / "test_series" / study / sid
        folder.mkdir(parents=True)
        attributes = {
            "StudyInstanceUID": study,
            "SeriesInstanceUID": sid,
            "SOPInstanceUID": f"{sid}.1",
        }
        if sid == f"{study}.1":
            attributes["PixelData"] = b"\0" * 512  # half the 16x32x16-bit payload
        (folder / f"{sid}.1.dcm").write_bytes(modified_dicom(**attributes))
    cache = tmp_path / "cache"
    selected, receipt = chunked.prepare_usable_series(
        frame,
        [study],
        dicom_root=root,
        cache=cache,
        cache_side=16,
        max_series=3,
        source_revision="test",
    )
    assert set(selected.SeriesInstanceUID) == {f"{study}.{i}" for i in (2, 3, 4)}
    assert receipt["skipped_series"]["invalid_series"] == 1
    dataset = CompetitiveStudyDataset(
        pd.DataFrame({"study_uid": [study]}),
        selected,
        cache,
        slices=1,
        side=16,
        max_series=3,
        raw_split="test_series",
    )
    assert dataset[0]["series_mask"].tolist() == [True, True, True]


def test_resilient_contract_checks_checkpoint_not_cli_assertions(tmp_path):
    path = tmp_path / "checkpoint.pt"
    torch.save({"binding": {"config": {"max_series": 6}, "cache_side": 256}}, path)
    chunked.verify_cache_contract(path, max_series=6, cache_side=256)
    with pytest.raises(ValueError, match="differs from the trained checkpoint"):
        chunked.verify_cache_contract(path, max_series=5, cache_side=256)
    with pytest.raises(ValueError, match="differs from the trained checkpoint"):
        chunked.verify_cache_contract(path, max_series=6, cache_side=224)


def test_resilient_run_uses_prepared_cache_without_redecoding(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    args.dicom_root = tmp_path / "dicoms"
    args.resilient_series = True
    args.cache_side = 256
    args.max_series = 6
    verified = []

    def verify(_checkpoint, *, max_series, cache_side):
        verified.append((max_series, cache_side))

    def prepare(frame, study_ids, **kwargs):
        assert kwargs["cache_side"] == 256
        assert set(frame.StudyInstanceUID) == set(study_ids)
        return frame, {"selected_series": len(frame), "skipped_series": {}}

    def invoke(command, *, check):
        assert "--dicom-root" not in command
        fake_infer(command, check=check)

    monkeypatch.setattr(chunked, "verify_cache_contract", verify)
    monkeypatch.setattr(chunked, "prepare_usable_series", prepare)
    result = chunked.run(args, invoke=invoke)
    assert verified == [(6, 256)]
    assert all("series_preparation" in row for row in result["chunks"])


@pytest.mark.parametrize("unusable_first_chunk", [("1.2.1",), ("1.2.5", "1.2.1")])
def test_unusable_mri_studies_get_neutral_rows_without_losing_other_predictions(
    tmp_path, monkeypatch, unusable_first_chunk
):
    args = fixture(tmp_path)
    args.dicom_root = tmp_path / "dicoms"
    args.resilient_series = True
    args.fallback_on_unusable_study = True
    args.cache_side = 256
    args.max_series = 6
    monkeypatch.setattr(chunked, "verify_cache_contract", lambda *_args, **_kwargs: None)

    def prepare(frame, study_ids, **kwargs):
        assert kwargs["allow_unusable_studies"] is True
        missing = [uid for uid in study_ids if uid in unusable_first_chunk]
        valid = frame.loc[~frame.StudyInstanceUID.isin(missing)].copy()
        return valid, {
            "selected_series": len(valid),
            "skipped_series": {"invalid_series": len(missing)},
            "unusable_study_ids": missing,
        }

    monkeypatch.setattr(chunked, "prepare_usable_series", prepare)
    result = chunked.run(args, invoke=fake_infer)
    output = pd.read_csv(args.output)
    assert output.StudyInstanceUID.tolist() == pd.read_csv(args.template).StudyInstanceUID.tolist()
    for uid in unusable_first_chunk:
        assert output.loc[output.StudyInstanceUID == uid, list(TARGETS)].eq(0.5).all().all()
    assert output.loc[output.StudyInstanceUID == "1.2.4", TARGETS[0]].item() == 0.4
    assert result["chunks"][0]["fallback_studies"] == len(unusable_first_chunk)
    assert result["chunks"][0]["inferred_studies"] == 2 - len(unusable_first_chunk)
    assert not list(args.cache_root.rglob("*.npz"))


def test_all_invalid_mri_studies_still_fail_instead_of_claiming_model_inference(
    tmp_path, monkeypatch
):
    args = fixture(tmp_path)
    args.dicom_root = tmp_path / "dicoms"
    args.resilient_series = True
    args.fallback_on_unusable_study = True
    args.cache_side = 256
    args.max_series = 6
    monkeypatch.setattr(chunked, "verify_cache_contract", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        chunked,
        "prepare_usable_series",
        lambda frame, study_ids, **kwargs: (
            frame.iloc[:0],
            {"selected_series": 0, "skipped_series": {}, "unusable_study_ids": study_ids},
        ),
    )
    with pytest.raises(ValueError, match="No predictions from MRI-backed studies"):
        chunked.run(args, invoke=fake_infer)
    assert not args.output.exists()


def test_missing_series_metadata_study_gets_neutral_row_when_fallback_enabled(
    tmp_path, monkeypatch
):
    args = fixture(tmp_path)
    args.dicom_root = tmp_path / "dicoms"
    args.resilient_series = True
    args.fallback_on_unusable_study = True
    args.cache_side = 256
    args.max_series = 6
    missing = "1.2.1"
    series = pd.read_csv(args.series)
    series.loc[series.StudyInstanceUID != missing].to_csv(args.series, index=False)
    monkeypatch.setattr(chunked, "verify_cache_contract", lambda *_args, **_kwargs: None)

    def prepare(frame, study_ids, **kwargs):
        unavailable = [uid for uid in study_ids if uid == missing]
        return frame, {
            "selected_series": len(frame),
            "skipped_series": {},
            "unusable_study_ids": unavailable,
        }

    monkeypatch.setattr(chunked, "prepare_usable_series", prepare)
    result = chunked.run(args, invoke=fake_infer)
    output = pd.read_csv(args.output)
    assert output.StudyInstanceUID.tolist() == pd.read_csv(args.template).StudyInstanceUID.tolist()
    assert output.loc[output.StudyInstanceUID == missing, list(TARGETS)].eq(0.5).all().all()
    assert sum(chunk["fallback_studies"] for chunk in result["chunks"]) == 1


def test_all_bad_real_dicoms_mark_only_that_study_unusable(tmp_path):
    good, bad = "2.3", "2.4"
    frame = pd.DataFrame(
        [
            {
                "StudyInstanceUID": study,
                "SeriesInstanceUID": f"{study}.1",
                "Anatomical_Plane": "Sagittal",
                "Fluid_Sensitive": 0,
                "Fat_Suppression": 0,
            }
            for study in (good, bad)
        ]
    )
    root = tmp_path / "dicoms"
    for study in (good, bad):
        folder = root / "test_series" / study / f"{study}.1"
        folder.mkdir(parents=True)
        data = (
            modified_dicom(
                StudyInstanceUID=study,
                SeriesInstanceUID=f"{study}.1",
                SOPInstanceUID=f"{study}.1.1",
            )
            if study == good
            else b"not a DICOM"
        )
        (folder / f"{study}.1.1.dcm").write_bytes(data)
    selected, receipt = chunked.prepare_usable_series(
        frame,
        [good, bad],
        dicom_root=root,
        cache=tmp_path / "cache",
        cache_side=16,
        max_series=3,
        source_revision="test",
        allow_unusable_studies=True,
    )
    assert selected.StudyInstanceUID.tolist() == [good]
    assert receipt["unusable_study_ids"] == [bad]
    assert receipt["skipped_series"]["invalid_dicoms"] == 1


@pytest.mark.parametrize("missing_field", ["StudyInstanceUID", "PixelData", "Rows"])
def test_missing_required_dicom_field_uses_healthy_series_alternate(tmp_path, missing_field):
    study = "2.3"
    frame = pd.DataFrame(
        [
            {
                "StudyInstanceUID": study,
                "SeriesInstanceUID": f"{study}.{index}",
                "Anatomical_Plane": "Sagittal",
                "Fluid_Sensitive": 1 if index == 1 else 0,
                "Fat_Suppression": 0,
            }
            for index in (1, 2)
        ]
    )
    root = tmp_path / "dicoms"
    for row in frame.to_dict("records"):
        sid = row["SeriesInstanceUID"]
        folder = root / "test_series" / study / sid
        folder.mkdir(parents=True)
        ds = pydicom.dcmread(
            io.BytesIO(
                modified_dicom(
                    StudyInstanceUID=study,
                    SeriesInstanceUID=sid,
                    SOPInstanceUID=f"{sid}.1",
                )
            )
        )
        if sid.endswith(".1"):
            delattr(ds, missing_field)
        ds.save_as(folder / f"{sid}.1.dcm", enforce_file_format=True)
    selected, receipt = chunked.prepare_usable_series(
        frame,
        [study],
        dicom_root=root,
        cache=tmp_path / "cache",
        cache_side=16,
        max_series=3,
        source_revision="test",
        allow_unusable_studies=True,
    )
    assert selected.SeriesInstanceUID.tolist() == [f"{study}.2"]
    assert receipt["skipped_series"]["invalid_dicoms"] == 1
    assert receipt["unusable_study_ids"] == []


def test_missing_dicom_transfer_syntax_uses_healthy_series_alternate(tmp_path):
    study = "2.3"
    frame = pd.DataFrame(
        [
            {
                "StudyInstanceUID": study,
                "SeriesInstanceUID": f"{study}.{index}",
                "Anatomical_Plane": "Sagittal",
                "Fluid_Sensitive": 1 if index == 1 else 0,
                "Fat_Suppression": 0,
            }
            for index in (1, 2)
        ]
    )
    root = tmp_path / "dicoms"
    for row in frame.to_dict("records"):
        sid = row["SeriesInstanceUID"]
        folder = root / "test_series" / study / sid
        folder.mkdir(parents=True)
        ds = pydicom.dcmread(
            io.BytesIO(
                modified_dicom(
                    StudyInstanceUID=study,
                    SeriesInstanceUID=sid,
                    SOPInstanceUID=f"{sid}.1",
                )
            )
        )
        if sid.endswith(".1"):
            del ds.file_meta.TransferSyntaxUID
        ds.save_as(folder / f"{sid}.1.dcm", enforce_file_format=False)
    selected, receipt = chunked.prepare_usable_series(
        frame,
        [study],
        dicom_root=root,
        cache=tmp_path / "cache",
        cache_side=16,
        max_series=3,
        source_revision="test",
        allow_unusable_studies=True,
    )
    assert selected.SeriesInstanceUID.tolist() == [f"{study}.2"]
    assert receipt["skipped_series"]["invalid_dicoms"] == 1
    assert receipt["unusable_study_ids"] == []


def test_unexpected_attribute_error_still_stops_inference(tmp_path, monkeypatch):
    study = "2.3"
    sid = f"{study}.1"
    frame = pd.DataFrame(
        [
            {
                "StudyInstanceUID": study,
                "SeriesInstanceUID": sid,
                "Anatomical_Plane": "Sagittal",
                "Fluid_Sensitive": 0,
                "Fat_Suppression": 0,
            }
        ]
    )
    folder = tmp_path / "dicoms" / "test_series" / study / sid
    folder.mkdir(parents=True)
    (folder / f"{sid}.1.dcm").write_bytes(b"synthetic")

    def crash(*_args):
        raise AttributeError("application bug")

    monkeypatch.setattr(chunked, "cache_series", crash)
    with pytest.raises(AttributeError, match="application bug"):
        chunked.prepare_usable_series(
            frame,
            [study],
            dicom_root=tmp_path / "dicoms",
            cache=tmp_path / "cache",
            cache_side=16,
            max_series=3,
            source_revision="test",
            allow_unusable_studies=True,
        )
