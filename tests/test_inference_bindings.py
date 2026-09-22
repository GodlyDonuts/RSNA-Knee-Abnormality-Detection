from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from rsna import infer
from rsna.infer import load_inference_metadata
from rsna.model import StudyNet
from rsna.schema import CACHE_VERSION, TARGETS


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def case(tmp_path: Path, monkeypatch, synthetic_release_binding) -> dict:
    studies = pd.DataFrame(
        [
            {
                "study_uid": "1.2.4",
                "split": "gold_evaluation",
                **{target: 0.0 for target in TARGETS},
                **{f"{target}__mask": 1 for target in TARGETS},
            }
        ]
    )
    series = pd.DataFrame(
        [
            {
                "StudyInstanceUID": "1.2.4",
                "SeriesInstanceUID": "1.2.3",
                "Anatomical_Plane": "Sagittal",
                "Fluid_Sensitive": 1,
                "Fat_Suppression": 0,
                "Original_Anatomical_Plane": "Coronal",
                "Plane_Review_ID": "reviewed-fixture",
                "Plane_Review_SHA256": "d" * 64,
            }
        ]
    )
    manifest_path, series_path = tmp_path / "studies.parquet", tmp_path / "series.csv"
    studies.to_parquet(manifest_path, index=False)
    series.to_csv(series_path, index=False)
    cache = tmp_path / "cache"
    path = cache / "train_series/1.2.4/1.2.3.npz"
    path.parent.mkdir(parents=True)
    np.savez_compressed(path, volume=np.arange(2048, dtype=np.uint16).reshape(2, 32, 32))
    metadata = {
        "study_uid": "1.2.4",
        "series_uid": "1.2.3",
        "cache_version": CACHE_VERSION,
        "source_revision": "a" * 40,
        "side": 32,
    }
    path.with_suffix(".json").write_text(json.dumps(metadata))
    checkpoint = {
        "targets": list(TARGETS),
        "cache_version": CACHE_VERSION,
        "cache_side": 32,
        "cache_source_revision": "a" * 40,
        "series_sha256": sha(series_path),
        "manifest_sha256": sha(manifest_path),
        "width": 8,
        "config": {"slices": 2, "side": 32, "max_series": 3},
        "model": StudyNet(8).state_dict(),
    }
    checkpoint["release_binding"] = synthetic_release_binding(manifest_path, series_path, "a" * 40)
    # These cases target inference metadata/cache behavior; the complete release
    # boundary has separate real-validator roundtrip and tamper tests.
    monkeypatch.setattr(
        infer,
        "load_verified_release",
        lambda path, artifact_root: (pd.read_parquet(path), checkpoint["release_binding"]),
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    return {
        "studies": manifest_path,
        "series": series_path,
        "cache": cache,
        "checkpoint": checkpoint,
        "checkpoint_path": checkpoint_path,
        "output": tmp_path / "predictions.csv",
        "cache_metadata": path.with_suffix(".json"),
    }


def load(case):
    return load_inference_metadata(
        case["checkpoint"], case["series"], case["studies"], None, "gold_evaluation"
    )


def argv(case):
    return [
        "infer",
        "--checkpoint",
        str(case["checkpoint_path"]),
        "--studies",
        str(case["studies"]),
        "--series",
        str(case["series"]),
        "--cache",
        str(case["cache"]),
        "--output",
        str(case["output"]),
        "--device",
        "cpu",
        "--workers",
        "0",
    ]


def test_real_heldout_inference_uses_and_records_exact_checkpoint_inputs(case, monkeypatch):
    monkeypatch.setattr(sys, "argv", argv(case))
    infer.main()
    predictions = pd.read_csv(case["output"])
    assert predictions.StudyInstanceUID.tolist() == ["1.2.4"]
    assert list(predictions.columns) == ["StudyInstanceUID", *TARGETS]
    assert np.isfinite(predictions[list(TARGETS)].to_numpy()).all()
    metrics = json.loads(case["output"].with_suffix(".metrics.json").read_text())
    assert metrics["series_sha256"] == case["checkpoint"]["series_sha256"]
    assert metrics["manifest_sha256"] == case["checkpoint"]["manifest_sha256"]
    assert metrics["metadata_binding"] == "exact_checkpoint_series_and_manifest"


@pytest.mark.parametrize(
    "field,value",
    [
        ("Anatomical_Plane", "Axial"),
        ("Fluid_Sensitive", 0),
        ("Fat_Suppression", 1),
        ("StudyInstanceUID", "9.9"),
        ("SeriesInstanceUID", "9.9.1"),
        ("Plane_Review_SHA256", "e" * 64),
    ],
)
def test_effective_plane_features_selection_or_lineage_tampering_rejected(case, field, value):
    series = pd.read_csv(case["series"])
    series.loc[0, field] = value
    series.to_csv(case["series"], index=False)
    with pytest.raises(ValueError, match="series_sha256 differs"):
        load(case)


def test_omitted_series_and_reserialization_are_not_silent_subsets(case):
    original = case["series"].read_bytes()
    case["series"].write_bytes(original.split(b"\n", 1)[0] + b"\n")
    with pytest.raises(ValueError, match="series_sha256 differs"):
        load(case)
    case["series"].write_bytes(original + b"\n")
    with pytest.raises(ValueError, match="series_sha256 differs"):
        load(case)


@pytest.mark.parametrize("field", ["series_sha256", "manifest_sha256", "cache_source_revision"])
def test_unbound_legacy_heldout_checkpoint_is_rejected(case, field):
    del case["checkpoint"][field]
    with pytest.raises(ValueError, match="Checkpoint lacks"):
        load(case)


@pytest.mark.parametrize("digest", [None, "invalid", "A" * 64, "a" * 63])
def test_invalid_checkpoint_series_digest_is_rejected(case, digest):
    case["checkpoint"]["series_sha256"] = digest
    with pytest.raises(ValueError, match="valid series_sha256"):
        load(case)


def test_changed_manifest_split_or_labels_cannot_redefine_heldout_evaluation(case):
    manifest = pd.read_parquet(case["studies"])
    manifest.loc[0, TARGETS[0]] = 1
    manifest.to_parquet(case["studies"], index=False)
    with pytest.raises(ValueError, match="manifest_sha256 differs"):
        load(case)


def test_cache_revision_mismatch_fails_before_prediction_output(case, monkeypatch):
    meta = json.loads(case["cache_metadata"].read_text())
    meta["source_revision"] = "b" * 40
    case["cache_metadata"].write_text(json.dumps(meta))
    monkeypatch.setattr(sys, "argv", argv(case))
    with pytest.raises(ValueError, match="cache source revision differs"):
        infer.main()
    assert not case["output"].exists()


def test_wrong_metadata_fails_before_dicom_preprocessing(case, monkeypatch):
    case["series"].write_bytes(case["series"].read_bytes() + b"\n")
    monkeypatch.setattr(sys, "argv", argv(case) + ["--dicom-root", str(case["cache"] / "missing")])
    with pytest.raises(ValueError, match="series_sha256 differs"):
        infer.main()
    assert not case["output"].exists()


def test_wrong_heldout_raw_revision_fails_before_cache_mutation(case, monkeypatch):
    before = case["cache_metadata"].read_bytes()
    monkeypatch.setattr(sys, "argv", argv(case) + ["--dicom-root", str(case["cache"] / "missing")])
    with pytest.raises(ValueError, match="preprocessing source revision differs"):
        infer.main()
    assert case["cache_metadata"].read_bytes() == before


def test_test_template_mode_records_separate_inputs_without_false_training_equality(case):
    series = pd.read_csv(case["series"])
    series.loc[0, "StudyInstanceUID"] = "9.9"
    series.loc[0, "SeriesInstanceUID"] = "9.9.1"
    series.to_csv(case["series"], index=False)
    template = case["output"].with_name("sample_submission.csv")
    pd.DataFrame([{"StudyInstanceUID": "9.9", **{target: 0 for target in TARGETS}}]).to_csv(
        template, index=False
    )
    _, studies, loaded_template, bindings = load_inference_metadata(
        case["checkpoint"], case["series"], None, template, "gold_evaluation"
    )
    assert studies.study_uid.tolist() == [9.9] and loaded_template is not None
    assert bindings["series_sha256"] == sha(case["series"])
    assert bindings["series_sha256"] != case["checkpoint"]["series_sha256"]
    assert bindings["template_sha256"] == sha(template)
    assert bindings["metadata_binding"] == "separate_test_inputs_not_checkpoint_bound"
    assert "manifest_sha256" not in bindings


def test_template_and_heldout_modes_cannot_be_combined(case):
    with pytest.raises(ValueError, match="exactly one"):
        load_inference_metadata(
            case["checkpoint"], case["series"], case["studies"], case["output"], "gold_evaluation"
        )


def test_real_new_test_input_inference_preserves_separate_mode(case, monkeypatch):
    uid, series_uid = "9.9.9", "9.9.9.1"
    series = pd.read_csv(case["series"])
    series.loc[0, ["StudyInstanceUID", "SeriesInstanceUID"]] = [uid, series_uid]
    series.to_csv(case["series"], index=False)
    template = case["output"].with_name("template.csv")
    pd.DataFrame([{"StudyInstanceUID": uid, **{target: 0 for target in TARGETS}}]).to_csv(
        template, index=False
    )
    new_cache = case["cache"] / "test_series" / uid / f"{series_uid}.npz"
    new_cache.parent.mkdir(parents=True)
    new_cache.write_bytes(case["cache_metadata"].with_suffix(".npz").read_bytes())
    metadata = json.loads(case["cache_metadata"].read_text())
    metadata.update(study_uid=uid, series_uid=series_uid, source_revision="new-runtime-test")
    new_cache.with_suffix(".json").write_text(json.dumps(metadata))
    # New test inputs use separate provenance; missing training hashes must
    # never provide a fallback in held-out evaluation mode.
    for field in ["series_sha256", "manifest_sha256", "cache_source_revision"]:
        del case["checkpoint"][field]
    torch.save(case["checkpoint"], case["checkpoint_path"])
    args = argv(case)
    args[args.index("--studies")] = "--template"
    args[args.index(str(case["studies"]))] = str(template)
    monkeypatch.setattr(sys, "argv", args)
    infer.main()
    predictions = pd.read_csv(case["output"])
    assert predictions.StudyInstanceUID.tolist() == [uid]
    assert list(predictions.columns) == ["StudyInstanceUID", *TARGETS]
    metrics = json.loads(case["output"].with_suffix(".metrics.json").read_text())
    assert metrics["metadata_binding"] == "separate_test_inputs_not_checkpoint_bound"
    assert metrics["series_sha256"] == sha(case["series"])
    assert metrics["template_sha256"] == sha(template)
    assert metrics["cache_source_revision"] == "new-runtime-test"


def test_heldout_inference_requires_real_release_receipt(case, monkeypatch):
    from rsna.release import load_verified_release

    monkeypatch.setattr(infer, "load_verified_release", load_verified_release)
    monkeypatch.setattr(sys, "argv", argv(case))
    with pytest.raises(ValueError, match="verified release receipt is required"):
        infer.main()
    assert not case["output"].exists()


@pytest.mark.parametrize("mutation", ["missing", "changed_digest"])
def test_inference_rejects_unbound_checkpoint_before_output(case, monkeypatch, mutation):
    if mutation == "missing":
        del case["checkpoint"]["release_binding"]
    else:
        case["checkpoint"]["release_binding"]["manifest_sha256"] = "c" * 64
    torch.save(case["checkpoint"], case["checkpoint_path"])
    monkeypatch.setattr(sys, "argv", argv(case))
    with pytest.raises(ValueError):
        infer.main()
    assert not case["output"].exists()


@pytest.mark.parametrize("field", ["manifest_sha256", "series_sha256", "cache_source_revision"])
def test_public_inference_rejects_conflicting_training_identity(case, monkeypatch, field):
    template = case["output"].with_name("template.csv")
    pd.DataFrame([{"StudyInstanceUID": "1.2.4", **{t: 0 for t in TARGETS}}]).to_csv(
        template, index=False
    )
    case["checkpoint"][field] = "c" * 64
    torch.save(case["checkpoint"], case["checkpoint_path"])
    args = argv(case)
    args[args.index("--studies")] = "--template"
    args[args.index(str(case["studies"]))] = str(template)
    monkeypatch.setattr(sys, "argv", args)
    with pytest.raises(ValueError, match="conflicts with its verified release"):
        infer.main()
    assert not case["output"].exists()
