"""Real MRI input faults and bounded recovery without changing model probabilities."""

from __future__ import annotations

import json
import shutil
import subprocess
from argparse import Namespace
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from test_competitive_inference import _fixture as model_fixture
from test_image_integrity import modified_dicom
from test_kaggle_chunked_inference import chunked, fake_infer, fixture

from rsna.competitive import infer
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.preprocess import Source, cache_series
from rsna.schema import TARGETS


def write_series(root, uid, sid, positions=(0, 1, 2), *, corrupt_slice=None):
    folder = root / "test_series" / uid / sid
    folder.mkdir(parents=True)
    for instance, z in enumerate(positions, 1):
        sop = f"{sid}.{instance}"
        payload = modified_dicom(
            instance=instance,
            StudyInstanceUID=uid,
            SeriesInstanceUID=sid,
            SOPInstanceUID=sop,
            ImagePositionPatient=[0, 0, z],
        )
        if instance == corrupt_slice:
            payload = b"undecodable source slice"
        (folder / f"{sop}.dcm").write_bytes(payload)


def metadata(uid, sid, plane="Sagittal", fluid=1):
    return dict(
        StudyInstanceUID=uid,
        SeriesInstanceUID=sid,
        Anatomical_Plane=plane,
        Fluid_Sensitive=fluid,
        Fat_Suppression=0,
    )


def prepare(frame, root, cache):
    return chunked.prepare_usable_series(
        frame,
        frame.StudyInstanceUID.drop_duplicates().tolist(),
        dicom_root=root,
        cache=cache,
        cache_side=16,
        max_series=3,
        source_revision="test",
        allow_unusable_studies=True,
        recover_slices=True,
        validate_geometry=True,
    )


def test_duplicate_geometry_rejected_before_model_and_same_plane_alternate_used(tmp_path):
    uid = "2.3"
    root, cache = tmp_path / "dicoms", tmp_path / "cache"
    rows = []
    for number, plane in enumerate(("Sagittal", "Sagittal", "Coronal", "Axial"), 1):
        sid = f"{uid}.{number}"
        write_series(root, uid, sid, (0, 0, 2) if number == 1 else (0, 1, 2))
        rows.append(metadata(uid, sid, plane, fluid=int(number == 1)))
    selected, receipt = prepare(pd.DataFrame(rows), root, cache)
    assert set(selected.SeriesInstanceUID) == {"2.3.2", "2.3.3", "2.3.4"}
    assert receipt["skipped_series"]["invalid_series"] == 1
    assert receipt["source_recovery"][0]["rejected_model_geometry"].endswith("strictly ordered")
    ds = CompetitiveStudyDataset(
        pd.DataFrame({"study_uid": [uid]}),
        selected,
        cache,
        slices=3,
        side=16,
        max_series=3,
        raw_split="test_series",
    )
    item = ds[0]
    assert item["series_mask"].sum() == 3
    assert torch.isfinite(item["images"]).all()


def test_corrupt_slice_retains_real_healthy_sops_and_exact_trained_transform(tmp_path):
    uid, sid = "2.3", "2.3.1"
    root, cache = tmp_path / "dicoms", tmp_path / "cache"
    write_series(root, uid, sid, corrupt_slice=2)
    selected, receipt = prepare(pd.DataFrame([metadata(uid, sid)]), root, cache)
    assert len(selected) == 1
    recovery = receipt["source_recovery"][0]
    assert recovery["retained_slices"] == 2 and len(recovery["rejected_slices"]) == 1
    meta = json.loads((cache / "test_series" / uid / f"{sid}.json").read_text())
    assert meta["sop_uids"] == [sid + ".1", sid + ".3"]
    paths = sorted(
        str(p.relative_to(root))
        for p in (root / "test_series" / uid / sid).glob("*.dcm")
        if p.stem != sid + ".2"
    )
    clean_meta = cache_series(Source(root), paths, tmp_path / "reference", 16, "test")
    assert clean_meta == meta
    assert (tmp_path / "reference" / meta["cache_path"]).read_bytes() == (
        cache / meta["cache_path"]
    ).read_bytes()


def test_healthy_slice_recovery_is_byte_identical_to_original_cache(tmp_path):
    uid, sid = "2.3", "2.3.1"
    root, cache = tmp_path / "dicoms", tmp_path / "cache"
    write_series(root, uid, sid)
    _, receipt = prepare(pd.DataFrame([metadata(uid, sid)]), root, cache)
    paths = sorted(
        str(p.relative_to(root)) for p in (root / "test_series" / uid / sid).glob("*.dcm")
    )
    reference = cache_series(Source(root), paths, tmp_path / "reference", 16, "test")
    assert receipt["source_recovery"] == []
    assert (tmp_path / "reference" / reference["cache_path"]).read_bytes() == (
        cache / reference["cache_path"]
    ).read_bytes()


@pytest.mark.parametrize(
    "error",
    [
        OSError("disk unavailable"),
        RuntimeError("unrelated application bug"),
        ValueError("Stale/corrupted cache"),
    ],
)
def test_input_recovery_does_not_hide_cache_or_system_failures(tmp_path, monkeypatch, error):
    uid, sid = "2.3", "2.3.1"
    root = tmp_path / "dicoms"
    write_series(root, uid, sid)

    def broken(*args):
        raise error

    monkeypatch.setattr(chunked, "cache_series", broken)
    with pytest.raises(type(error), match=str(error)):
        prepare(pd.DataFrame([metadata(uid, sid)]), root, tmp_path / "cache")


def isolation_args(tmp_path, monkeypatch):
    args = fixture(tmp_path)
    args.fallback_on_unusable_study = True
    args.resilient_series = True
    args.isolate_study_errors = True
    args.cache_side, args.max_series = 16, 3
    args.dicom_root = tmp_path / "dicoms"
    monkeypatch.setattr(chunked, "verify_cache_contract", lambda *a, **k: None)
    monkeypatch.setattr(
        chunked,
        "prepare_usable_series",
        lambda frame, ids, **kw: (frame, {"unusable_study_ids": []}),
    )
    return args


def fake_child(command, *, bad_ids=(), trace=None, **kwargs):
    frame = pd.read_csv(command[command.index("--template") + 1])
    if set(frame.StudyInstanceUID) & set(bad_ids):
        raise subprocess.CalledProcessError(
            1,
            command,
            output="",
            stderr=trace
            or 'Traceback (most recent call last):\n  File "/code/rsna/competitive/data.py", line 115, in __getitem__\nValueError: malformed MRI positions\n',
        )
    cache = Path(command[command.index("--cache") + 1])
    if cache.exists():
        shutil.rmtree(cache)  # Only the stub's synthetic volume, never real cache data.
    fake_infer(command, check=kwargs["check"])
    return subprocess.CompletedProcess(command, 0, "", "")


def test_input_error_isolated_to_one_study_preserves_other_raw_probabilities(tmp_path, monkeypatch):
    args = isolation_args(tmp_path, monkeypatch)
    result = chunked.run(args, invoke=lambda c, **kw: fake_child(c, bad_ids={"1.2.1"}, **kw))
    frame = pd.read_csv(args.output)
    assert frame.StudyInstanceUID.tolist() == ["1.2.5", "1.2.1", "1.2.4", "1.2.2", "1.2.3"]
    assert frame[TARGETS[0]].tolist() == [0.5, 0.5, 0.4, 0.2, 0.3]
    assert sum(r["fallback_studies"] for r in result["chunks"]) == 1
    assert result["chunks"][0]["failed_study_ids"] == ["1.2.1"]
    assert len(result["chunks"][0]["inference_attempts"]) == 3
    assert not list(args.cache_root.rglob("*.npz"))
    assert not args.output.with_suffix(".partial.csv").exists()


def test_all_studies_failing_never_publishes_neutral_only_submission(tmp_path, monkeypatch):
    args = isolation_args(tmp_path, monkeypatch)
    ids = pd.read_csv(args.template).StudyInstanceUID.tolist()
    with pytest.raises(ValueError, match="No predictions from MRI-backed"):
        chunked.run(args, invoke=lambda c, **kw: fake_child(c, bad_ids=ids, **kw))
    assert not args.output.exists()


@pytest.mark.parametrize(
    "trace",
    [
        'Traceback\n  File "/code/rsna/competitive/infer.py", line 80\nValueError: code differs from training',
        'Traceback\n  File "/code/rsna/competitive/data.py", line 100\nValueError: Cache volume checksum/size mismatch',
        'Traceback\n  File "/code/rsna/competitive/data.py", line 100\nRuntimeError: CUDA out of memory',
    ],
)
def test_global_integrity_and_cuda_failures_remain_fatal(tmp_path, monkeypatch, trace):
    args = isolation_args(tmp_path, monkeypatch)
    with pytest.raises(subprocess.CalledProcessError):
        chunked.run(
            args, invoke=lambda c, **kw: fake_child(c, bad_ids={"1.2.5"}, trace=trace, **kw)
        )
    assert not args.output.exists()


def test_success_receipt_corruption_cannot_be_converted_to_neutral_fallback(tmp_path, monkeypatch):
    args = isolation_args(tmp_path, monkeypatch)

    def child(command, **kw):
        result = fake_child(command, **kw)
        path = Path(command[command.index("--output") + 1]).with_suffix(".metrics.json")
        payload = json.loads(path.read_text())
        payload["checkpoint_sha256"] = "0" * 64
        path.write_text(json.dumps(payload))
        return result

    with pytest.raises(ValueError, match="receipt differs"):
        chunked.run(args, invoke=child)
    assert not args.output.exists()


def test_actual_model_route_handles_duplicate_geometry_and_corrupt_slice(tmp_path, monkeypatch):
    base, _ = model_fixture(tmp_path, monkeypatch, size=2)
    root = tmp_path / "dicoms"
    rows = []
    for uid in pd.read_csv(base.template).StudyInstanceUID:
        sid, alternate = uid + ".1", uid + ".2"
        write_series(root, uid, sid, (0, 0, 2))
        write_series(root, uid, alternate, corrupt_slice=2)
        rows += [metadata(uid, sid), metadata(uid, alternate, fluid=0)]
    pd.DataFrame(rows).to_csv(base.series, index=False)
    args = Namespace(
        **{
            k: getattr(base, k)
            for k in (
                "checkpoint",
                "encoder_config",
                "series",
                "template",
                "source_revision",
                "device",
                "workers",
                "allow_diagnostic",
            )
        },
        output=base.output,
        cache_root=tmp_path / "scratch",
        dicom_root=root,
        chunk_size=2,
        precision="fp32",
        resilient_series=True,
        fallback_on_unusable_study=True,
        recover_slices=True,
        validate_model_geometry=True,
        isolate_study_errors=True,
        cache_side=16,
        max_series=3,
    )

    def actual_inference(command, **kwargs):
        def option(name):
            return Path(command[command.index(name) + 1])

        child = Namespace(**vars(base))
        child.series, child.template, child.cache, child.output = [
            option(name) for name in ("--series", "--template", "--cache", "--output")
        ]
        child.scheduled_epoch, child.precision = True, "fp32"
        infer.run(child)
        return subprocess.CompletedProcess(command, 0, "", "")

    result = chunked.run(args, invoke=actual_inference)
    assert result["status"] == "passed" and result["diagnostic_only"] is True
    frame = pd.read_csv(args.output)
    assert np.isfinite(frame[list(TARGETS)].to_numpy()).all()
    assert not (frame[list(TARGETS)].to_numpy() == 0.5).all()
    assert result["chunks"][0]["fallback_studies"] == 0
    assert result["chunks"][0]["series_preparation"]["skipped_series"]["invalid_series"] == 2
