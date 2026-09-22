from __future__ import annotations

import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import requests
import torch
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, MRImageStorage
from urllib3.response import HTTPResponse

from rsna import infer, train
from rsna.audit import audit_cache
from rsna.dicom import decode_slice, make_volume, order_slices
from rsna.infer import write_submission
from rsna.labels import build_studies, read_reviews
from rsna.model import StudyNet, auc_metrics, masked_bce
from rsna.preprocess import HubRetry, Source, cache_series
from rsna.schema import CACHE_VERSION, TARGETS


def dicom_bytes(instance: int, z: float, photo: str = "MONOCHROME2") -> bytes:
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = MRImageStorage
    meta.MediaStorageSOPInstanceUID = f"1.2.3.{instance}"
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID = (
        "1.2",
        "1.2.3",
        f"1.2.3.{instance}",
    )
    ds.SOPClassUID = MRImageStorage
    ds.Rows, ds.Columns = 16, 32
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = photo
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit, ds.PixelRepresentation = 15, 0
    ds.ImagePositionPatient = [0, 0, z]
    ds.ImageOrientationPatient = [1, 0, 0, 0, 1, 0]
    ds.PixelSpacing = [2, 1]
    ds.InstanceNumber = instance
    ds.RescaleSlope, ds.RescaleIntercept = 2, -10
    ds.PixelData = np.arange(512, dtype=np.uint16).reshape(16, 32).tobytes()
    buffer = io.BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def test_dicom_rescale_inversion_and_uid_check() -> None:
    normal = decode_slice(dicom_bytes(1, 0), "1.2", "1.2.3", "1.2.3.1")
    inverted = decode_slice(dicom_bytes(1, 0, "MONOCHROME1"), "1.2", "1.2.3", "1.2.3.1")
    assert normal.pixels[0, 0] == -10
    # Stored modality values remain comparable across slices; invert at volume level.
    np.testing.assert_array_equal(inverted.pixels, normal.pixels)
    normal_volume, _ = make_volume([normal], 32)
    inverted_volume, _ = make_volume([inverted], 32)
    np.testing.assert_allclose(
        inverted_volume.astype(float), 65535 - normal_volume.astype(float), atol=1
    )
    with pytest.raises(ValueError, match="UID mismatch"):
        decode_slice(dicom_bytes(1, 0), "9.9", "1.2.3", "1.2.3.1")


def test_geometry_overrides_instance_and_preserves_physical_aspect() -> None:
    slices = [
        decode_slice(dicom_bytes(i, z), "1.2", "1.2.3", f"1.2.3.{i}")
        for i, z in [(1, 3), (2, 1), (3, 2)]
    ]
    ordered, method, _ = order_slices(slices)
    assert [s.instance for s in ordered] == [2, 3, 1]
    assert method == "patient_geometry"
    volume, metadata = make_volume(slices, 32)
    assert volume.shape == (3, 32, 32) and volume.dtype == np.uint16
    assert metadata["resized_shape"] == [32, 32]
    slices[1].position = ()
    assert order_slices(slices)[1] == "instance_number"
    slices[1].instance = 1
    with pytest.raises(ValueError, match="Neither geometry"):
        order_slices(slices)


def test_cache_resume_detects_corruption_and_orphan(tmp_path: Path) -> None:
    source = tmp_path / "source"
    folder = source / "train_series/1.2/1.2.3"
    folder.mkdir(parents=True)
    paths = []
    for i in range(1, 4):
        path = folder / f"1.2.3.{i}.dcm"
        path.write_bytes(dicom_bytes(i, i))
        paths.append(str(path.relative_to(source)))
    output = tmp_path / "cache"
    first = cache_series(Source(source), paths, output, 32, "fixture")
    assert cache_series(Source(source), paths, output, 32, "fixture") == first
    with pytest.raises(ValueError, match="Stale/corrupted"):
        cache_series(Source(source), paths, output, 64, "fixture")
    cache_file = output / first["cache_path"]
    cache_file.write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="Stale/corrupted"):
        cache_series(Source(source), paths, output, 32, "fixture")
    cache_file.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="Incomplete cache pair"):
        cache_series(Source(source), paths, output, 32, "fixture")


def test_nonfinite_geometry_cannot_silently_define_slice_order() -> None:
    item = decode_slice(dicom_bytes(1, 0), "1.2", "1.2.3", "1.2.3.1")
    item.position = (0.0, 0.0, float("nan"))
    with pytest.raises(ValueError, match="Non-finite DICOM geometry"):
        order_slices([item])


def review_row(uid: str, state: str) -> dict:
    result = {"study_uid": uid}
    for target in TARGETS:
        result[f"{target}__state"] = state
        result[f"{target}__evidence"] = ""
    return result


def test_gold_isolation_duplicate_grouping_and_unknown_mask() -> None:
    train = pd.DataFrame(
        [
            {"StudyInstanceUID": uid, "Report": report, **{t: gold for t in TARGETS}}
            for uid, report, gold in [
                ("a", "Normal knee", 1),
                ("b", " NORMAL   KNEE ", np.nan),
                ("c", "Other knee", np.nan),
                ("d", "Other knee", np.nan),
            ]
        ]
    )
    reviews = pd.DataFrame([review_row(uid, "not_mentioned") for uid in "abcd"])
    result = build_studies(train, reviews).set_index("study_uid")
    assert result.loc["a", "split"] == "gold_evaluation"
    assert result.loc["b", "split"] == "quarantine"
    assert result.loc["c", "split"] == result.loc["d", "split"]
    assert result.loc["a", list(TARGETS)].eq(1).all()
    assert result.loc["c", [f"{t}__mask" for t in TARGETS]].eq(0).all()
    assert result.loc["a", [f"{t}__provenance" for t in TARGETS]].eq("official_gold").all()


def test_review_validation_rejects_missing_and_fabricated_evidence(tmp_path: Path) -> None:
    row = {"study_uid": "a", "states": ["negative"] * 12, "evidence": ["intact"] * 12, "notes": ""}
    file = tmp_path / "batch-000-reviewed.jsonl"
    file.write_text(json.dumps(row) + "\n")
    assert len(read_reviews(tmp_path, {"a": "Ligaments intact"}, "-reviewed")) == 1
    with pytest.raises(ValueError, match="Unsupported quote"):
        read_reviews(tmp_path, {"a": "Ligaments torn"}, "-reviewed")
    with pytest.raises(ValueError, match="missing studies"):
        read_reviews(tmp_path, {"a": "intact", "b": "intact"}, "-reviewed")


def test_unknown_labels_have_zero_gradient_and_valid_optimizer_step() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(42)
    model = StudyNet(width=8)
    images = torch.rand(2, 3, 2, 1, 32, 32)
    features = torch.rand(2, 3, 5)
    series_mask = torch.tensor([[True, True, False], [True, False, False]])
    logits = model(images, features, series_mask)
    logits.retain_grad()
    labels, mask = torch.zeros(2, 12), torch.zeros(2, 12)
    mask[:, 0] = 1
    loss = masked_bce(logits, labels, mask)
    loss.backward()
    assert logits.grad is not None and logits.grad[:, 1:].eq(0).all()
    old = model.classifier.weight.detach().clone()
    torch.optim.AdamW(model.parameters(), lr=0.01).step()
    assert not torch.equal(old, model.classifier.weight)
    model.eval()
    original = model(images, features, series_mask)
    images[~series_mask] = 999
    features[~series_mask] = 999
    torch.testing.assert_close(model(images, features, series_mask), original)


def test_auc_does_not_report_official_macro_when_classes_missing() -> None:
    labels = np.zeros((4, 12))
    labels[2:, 0] = 1
    scores = np.full((4, 12), 0.2)
    scores[2:, 0] = 0.8
    metrics = auc_metrics(labels, scores, np.ones_like(labels))
    assert metrics["per_target_auc"]["ACL"] == 1
    assert metrics["macro_auc"] is None
    assert metrics["n_scored_targets"] == 1


def test_submission_requires_exact_studies_and_valid_probabilities(tmp_path: Path) -> None:
    template = pd.DataFrame({"StudyInstanceUID": ["b", "a"], **{t: [0.5, 0.5] for t in TARGETS}})
    predictions = template.iloc[::-1].copy()
    output = tmp_path / "submission.csv"
    write_submission(predictions, template, output)
    assert pd.read_csv(output).StudyInstanceUID.tolist() == ["b", "a"]
    predictions.loc[predictions.index[0], "ACL"] = np.nan
    with pytest.raises(ValueError, match="probabilities"):
        write_submission(predictions, template, output)


@pytest.mark.parametrize("diagnostic_limit", [None, 2])
def test_offline_train_resume_predict_pipeline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, synthetic_release_binding, diagnostic_limit
) -> None:
    def deny_network(*args, **kwargs):
        raise AssertionError("Offline pipeline attempted network access")

    monkeypatch.setattr(requests.Session, "request", deny_network)
    studies, series = [], []
    cache = tmp_path / "cache"
    for index, uid in enumerate("abcdef"):
        split = "train" if index < 4 else "weak_validation"
        studies.append(
            {
                "study_uid": uid,
                "split": split,
                "report_group": uid,
                "leakage_group": uid,
                **{t: float(index % 2) for t in TARGETS},
                **{f"{t}__mask": 1.0 for t in TARGETS},
                **{f"{t}__provenance": "report_reviewed" for t in TARGETS},
            }
        )
        folder = cache / "train_series" / uid
        folder.mkdir(parents=True)
        series_uid = uid + "-series"
        path = folder / f"{series_uid}.npz"
        rng = np.random.default_rng(index)
        np.savez_compressed(path, volume=rng.integers(0, 65535, size=(3, 32, 32), dtype=np.uint16))
        path.with_suffix(".json").write_text(
            json.dumps(
                {
                    "cache_version": CACHE_VERSION,
                    "study_uid": uid,
                    "series_uid": series_uid,
                    "side": 32,
                    "source_revision": "synthetic-integration-test",
                }
            )
        )
        series.append(
            {
                "StudyInstanceUID": uid,
                "SeriesInstanceUID": series_uid,
                "Anatomical_Plane": "Sagittal",
                "Fluid_Sensitive": 1,
                "Fat_Suppression": 0,
            }
        )
    study_file, series_file = tmp_path / "studies.parquet", tmp_path / "series.csv"
    pd.DataFrame(studies).to_parquet(study_file, index=False)
    pd.DataFrame(series).to_csv(series_file, index=False)
    # This test exercises optimizer/resume/inference. Full evidence validation is
    # independently exercised by the release adapter's real roundtrip tests.
    binding = synthetic_release_binding(study_file, series_file, "synthetic-integration-test")

    def verified_manifest(path, artifact_root):
        assert path == study_file
        full_manifest = pd.read_parquet(path)
        assert len(full_manifest) == 6  # Validation always precedes diagnostic selection.
        return full_manifest, binding

    monkeypatch.setattr(train, "load_verified_release", verified_manifest)
    monkeypatch.setattr(infer, "load_verified_release", verified_manifest)
    base = [
        "train",
        "--studies",
        str(study_file),
        "--series",
        str(series_file),
        "--cache",
        str(cache),
        "--width",
        "8",
        "--slices",
        "2",
        "--side",
        "32",
        "--max-series",
        "3",
        "--workers",
        "0",
        "--device",
        "cpu",
    ]
    if diagnostic_limit is not None:
        base += ["--diagnostic-study-limit", str(diagnostic_limit)]
    continuous, resumed = tmp_path / "continuous", tmp_path / "resumed"
    monkeypatch.setattr(sys, "argv", base + ["--epochs", "2", "--output", str(continuous)])
    train.main()
    monkeypatch.setattr(sys, "argv", base + ["--epochs", "1", "--output", str(resumed)])
    train.main()
    monkeypatch.setattr(
        sys,
        "argv",
        base
        + [
            "--epochs",
            "2",
            "--output",
            str(resumed),
            "--resume",
            str(resumed / "checkpoint.pt"),
        ],
    )
    train.main()
    expected = torch.load(continuous / "checkpoint.pt", weights_only=True)
    actual = torch.load(resumed / "checkpoint.pt", weights_only=True)
    assert actual["run_config"]["training_studies"] == (diagnostic_limit or 4)
    assert actual["run_config"]["validation_studies"] == 2
    assert actual["release_binding"] == binding
    for key, value in expected["model"].items():
        torch.testing.assert_close(value, actual["model"][key], atol=0, rtol=0)
    predictions = tmp_path / "predictions.csv"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer",
            "--checkpoint",
            str(resumed / "checkpoint.pt"),
            "--studies",
            str(study_file),
            "--series",
            str(series_file),
            "--cache",
            str(cache),
            "--split",
            "weak_validation",
            "--output",
            str(predictions),
            "--device",
            "cpu",
            "--workers",
            "0",
        ],
    )
    infer.main()
    assert pd.read_csv(predictions).StudyInstanceUID.tolist() == ["e", "f"]
    altered = tmp_path / "altered.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        base + ["--epochs", "3", "--output", str(resumed), "--resume", str(altered)],
    )
    for field, value, message in [
        ("release_binding", {}, "supervision release differs"),
        ("targets", list(reversed(TARGETS)), "targets or training settings"),
        ("run_config", {**actual["run_config"], "seed": 1}, "targets or training settings"),
        ("cache_side", 64, "cache preprocessing"),
        ("cache_source_revision", "different-source", "cache preprocessing"),
        ("cache_version", "incompatible", "cache transform version"),
    ]:
        torch.save({**actual, field: value}, altered)
        with pytest.raises(ValueError, match=message):
            train.main()


def test_hub_retry_obeys_quota_reset() -> None:
    retry = HubRetry()
    response = HTTPResponse(status=429, headers={"RateLimit": '"api";r=0;t=123'})
    assert retry.get_retry_after(response) == 124
    response.headers["Retry-After"] = "42"
    assert retry.get_retry_after(response) == 42
    assert retry.get_retry_after(HTTPResponse(status=429)) == 300
    assert retry.get_retry_after(HTTPResponse(status=503)) is None


def test_cache_audit_requires_complete_source_verified_coverage(tmp_path: Path) -> None:
    source, cache, output = tmp_path / "source", tmp_path / "cache", tmp_path / "audit"
    rows = []
    for index in range(1, 4):
        path = f"train_series/1.2/1.2.3/1.2.3.{index}.dcm"
        file = source / path
        file.parent.mkdir(parents=True, exist_ok=True)
        data = dicom_bytes(index, index)
        file.write_bytes(data)
        rows.append(
            {
                "split": "train_series",
                "study_uid": "1.2",
                "series_uid": "1.2.3",
                "sop_uid": f"1.2.3.{index}",
                "path": path,
                "sha256": hashlib.sha256(data).hexdigest(),
                "bytes": len(data),
            }
        )
    inventory = pd.DataFrame(rows)
    cache_series(Source(source), inventory.path.tolist(), cache, 32, "fixture")
    report = audit_cache(inventory, cache, "fixture", output)
    assert report["decoded_slices"] == 3 and report["series"] == 1
    invalid = inventory.copy()
    invalid.loc[0, "sha256"] = "invalid"
    with pytest.raises(ValueError, match="pinned inventory"):
        audit_cache(invalid, cache, "fixture", output)
    with pytest.raises(ValueError, match="Missing/duplicate SOPs"):
        audit_cache(inventory.iloc[:2], cache, "fixture", output)


def test_transitive_image_report_duplicates_are_quarantined_with_gold() -> None:
    original = pd.DataFrame(
        [
            {"StudyInstanceUID": uid, "Report": report, **{t: gold for t in TARGETS}}
            for uid, report, gold in [
                ("a", "Report A", 1),
                ("b", "Report B", np.nan),
                ("c", "Report B", np.nan),
                ("d", "Report D", np.nan),
            ]
        ]
    )
    reviews = pd.DataFrame([review_row(uid, "not_mentioned") for uid in "abcd"])
    result = build_studies(original, reviews, [("a", "b"), ("c", "d")]).set_index("study_uid")
    assert result.loc["a", "split"] == "gold_evaluation"
    assert result.loc[list("bcd"), "split"].eq("quarantine").all()
    assert result.leakage_group.nunique() == 1


def test_diagnostic_training_still_requires_real_release_boundary(tmp_path, monkeypatch):
    output = tmp_path / "unapproved-run"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--studies",
            str(tmp_path / "studies.parquet"),
            "--series",
            str(tmp_path / "series.csv"),
            "--cache",
            str(tmp_path / "cache"),
            "--output",
            str(output),
            "--diagnostic-study-limit",
            "1",
            "--device",
            "cpu",
        ],
    )
    with pytest.raises(ValueError, match="verified release receipt is required"):
        train.main()
    assert not output.exists()
