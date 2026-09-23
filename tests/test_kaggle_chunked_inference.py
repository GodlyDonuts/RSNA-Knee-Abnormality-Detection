"""Bounded Kaggle scratch and exact test-row ordering, without remote calls."""

from __future__ import annotations

import importlib.util
import json
import shutil
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rsna.competitive import infer
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
