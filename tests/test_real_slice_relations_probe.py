import hashlib
import io
import json
import tarfile

import numpy as np
import pandas as pd
import pytest
import torch

from scripts.newton.audit_relation_geometry import digest, signature
from scripts.newton.launch_relation_cpu_probe import unpack_source
from scripts.newton.probe_real_slice_relations import (
    real_batch,
    tree_equal,
    validate_source,
    volume_from_archive,
)


@pytest.fixture
def source(tmp_path):
    volume = np.arange(6 * 256 * 256, dtype=np.uint16).reshape(6, 256, 256)
    selected = dict(study_uid="train-a", series_uid="series-a", n_slices=6, plane="Sagittal")
    data = io.BytesIO()
    np.savez_compressed(data, volume=volume)
    payload = data.getvalue()
    metadata = dict(
        **{k: v for k, v in selected.items() if k != "plane"},
        source_revision="source",
        side=256,
        split="train_series",
        cache_path="train_series/train-a/series-a.npz",
        cache_bytes=len(payload),
        cache_sha256=hashlib.sha256(payload).hexdigest(),
        ordering="patient_geometry",
        orientation=[1, 0, 0, 0, 1, 0],
        positions=[[0, 0, 3 * i] for i in range(6)],
    )
    archive = tmp_path / "cache.tar"
    with tarfile.open(archive, "w:") as handle:
        for suffix, raw in (("json", json.dumps(metadata).encode()), ("npz", payload)):
            member = tarfile.TarInfo("cache/train_series/train-a/series-a." + suffix)
            member.size = len(raw)
            handle.addfile(member, io.BytesIO(raw))
    table = pd.DataFrame(
        [
            dict(
                StudyInstanceUID="train-a",
                SeriesInstanceUID="series-a",
                Anatomical_Plane="Sagittal",
                Fluid_Sensitive=1,
                Fat_Suppression=0,
            )
        ]
    )
    series = tmp_path / "series.csv"
    table.to_csv(series, index=False)
    request = dict(
        training_uids=["train-a"],
        training_uids_sha256=digest(["train-a"]),
        series_table_sha256=hashlib.sha256(series.read_bytes()).hexdigest(),
        source_revision="source",
        series=[selected],
    )
    run = dict(
        training_studies=1,
        binding=dict(
            training_uids_sha256=request["training_uids_sha256"],
            release=dict(series_sha256=request["series_table_sha256"]),
        ),
    )
    proof = dict(
        assets={str(archive): dict(stat=signature(archive))},
        data=dict(release_binding=dict(source_revision="source")),
    )
    return archive, series, table, request, run, proof, volume, metadata


def test_training_only_verified_bytes_and_real_image_layout(source):
    archive, series, table, request, run, proof, expected, _ = source
    selected = validate_source(request, proof, run, archive, series)
    volume, metadata, meta_sha = volume_from_archive(archive, selected, "source")
    assert np.array_equal(volume, expected) and len(meta_sha) == 64
    batch = real_batch(volume, metadata, selected, table)
    assert batch["images"].shape == (1, 3, 6, 1, 32, 32)
    assert not batch["images"][:, 1:].any() and not batch["slice_mask"][:, 1:].any()
    assert batch["positions"][0, 0, :, 1].eq(1).all()
    assert batch["features"][0, 0, 3:].tolist() == [1, 0]
    assert batch["source_slice_indices"] == list(range(6))
    assert batch["label_mask"].sum() == 0
    assert torch.isfinite(batch["images"]).all() and batch["images"].max() <= 1


@pytest.mark.parametrize(
    "fault", ["foreign_study", "changed_roster", "changed_csv", "changed_archive"]
)
def test_source_admission_refuses_unbound_training_inputs(source, fault):
    archive, series, _, request, run, proof, _, _ = source
    if fault == "foreign_study":
        request["series"][0]["study_uid"] = "outer"
    elif fault == "changed_roster":
        request["training_uids"].append("outer")
    elif fault == "changed_csv":
        series.write_text("changed")
    else:
        archive.write_bytes(archive.read_bytes() + b"changed")
    with pytest.raises(ValueError):
        validate_source(request, proof, run, archive, series)


def test_diagnostic_refuses_mismatched_cache_and_unknown_geometry(source):
    archive, _, table, request, _, _, volume, metadata = source
    with pytest.raises(ValueError):
        volume_from_archive(archive, request["series"][0], "wrong-revision")
    metadata["ordering"] = "ordinal"
    with pytest.raises(ValueError, match="physical"):
        real_batch(volume, metadata, request["series"][0], table)


def test_decoded_state_comparison_detects_nested_optimizer_or_rng_change():
    first = {"optimizer": {0: {"moment": torch.tensor([1.0])}}, "rng": [torch.arange(3), 7]}
    second = {"optimizer": {0: {"moment": torch.tensor([1.0])}}, "rng": [torch.arange(3), 7]}
    assert tree_equal(first, second)
    second["rng"][0][2] = 0
    assert not tree_equal(first, second)


@pytest.mark.parametrize("fault", [None, "path", "hash", "extra", "link"])
def test_source_pack_requires_exact_regular_hashed_members(tmp_path, fault):
    raw = b"verified scientific source"
    name = "rsna/probe.py"
    archive = tmp_path / "source.tar"
    expected = {name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}}
    with tarfile.open(archive, "w:") as handle:
        member = tarfile.TarInfo("../escaped.py" if fault == "path" else name)
        member.size = len(raw)
        if fault == "link":
            member.type = tarfile.SYMTYPE
            member.linkname = "/etc/passwd"
            member.size = 0
        handle.addfile(member, io.BytesIO(raw) if fault != "link" else None)
        if fault == "extra":
            extra = tarfile.TarInfo("extra.py")
            extra.size = len(raw)
            handle.addfile(extra, io.BytesIO(raw))
    if fault == "hash":
        expected[name]["sha256"] = "0" * 64
    output = tmp_path / "materialized"
    output.mkdir()
    if fault:
        with pytest.raises(ValueError):
            unpack_source(archive, output, expected)
    else:
        unpack_source(archive, output, expected)
        assert (output / name).read_bytes() == raw
