"""Independent inference integration regressions using synthetic labels/caches only."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from rsna.competitive import backbones, infer
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.relations import SliceDistanceHead, relation_model_kwargs
from rsna.competitive.snapshots import scheduled_epoch, write_epoch_snapshot
from rsna.competitive.train import TrainConfig
from rsna.schema import CACHE_VERSION, TARGETS
from rsna.semantic_supervision import digest


class SyntheticHF(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 768)

    def forward(self, pixel_values):
        return SimpleNamespace(pooler_output=self.projection(pixel_values.mean((2, 3))))


def _fixture(
    tmp_path,
    monkeypatch,
    *,
    checkpoint_only=False,
    precision="fp32",
    size=2,
    whole=False,
    relation=None,
    plane_memory=None,
):
    torch.manual_seed(123)
    config = TrainConfig(
        epochs=2,
        submission_every_epochs=1,
        diagnostic_study_limit=3,
        slices=2,
        side=16,
        max_series=3,
        hidden_dim=8,
        precision=precision,
        slice_context="neighbor3" if whole else "none",
        slice_sampling="all" if whole else "uniform",
        series_sampling="all" if whole else "selected",
        relation=relation,
        plane_memory=plane_memory,
    )
    spec_path = tmp_path / "encoder-spec.json"
    config_path = tmp_path / "encoder-config.json"
    if checkpoint_only:
        source_config = dict(
            model_type="dinov3_vit",
            hidden_size=768,
            intermediate_size=3072,
            num_hidden_layers=12,
            num_attention_heads=12,
            patch_size=16,
            num_register_tokens=4,
            num_channels=3,
        )
        config_path.write_text(json.dumps(source_config))
        source = dict(
            architecture="dinov3_b16",
            config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
            weights_sha256="f" * 64,
            source_id="synthetic/test",
            source_revision="a" * 40,
            license="test-only",
            state_dict_format="safetensors",
            transformers_version="5.6.1",
        )
        provenance = backbones._encoder_provenance(
            source, source_config, "ConvNextForImageClassification"
        )
        encoder, dim = backbones._GrayEncoder(SyntheticHF(), 16), 768
        monkeypatch.setattr(backbones, "_construct", lambda *args: SyntheticHF())
        monkeypatch.setattr(backbones.importlib.metadata, "version", lambda name: "5.6.1")
    else:
        spec = {"architecture": "diagnostic_tiny", "embedding_dim": 16}
        spec_path.write_text(json.dumps(spec))
        encoder, dim, provenance = backbones.create_encoder(spec)
    encoder.requires_grad_(False)
    model = CompetitiveStudyNet(
        encoder,
        dim,
        hidden_dim=8,
        slice_context=config.slice_context,
        **relation_model_kwargs(config.relation),
        plane_memory=config.plane_memory,
    )
    release = dict(
        schema_version=1,
        manifest_sha256="a" * 64,
        series_sha256="b" * 64,
        source_revision="synthetic-train",
        release_receipt_sha256="c" * 64,
        semantic_registry_sha256="d" * 64,
    )
    release["binding_sha256"] = digest(release)
    code_root = Path(infer.__file__).parents[1]
    binding = dict(
        release=release,
        config=dict(asdict(config), encoder=provenance),
        targets=list(TARGETS),
        cache_version=CACHE_VERSION,
        cache_side=16,
        source_revision=release["source_revision"],
        code_sha256={
            p.relative_to(code_root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in code_root.rglob("*.py")
        },
    )
    checkpoint = dict(
        schema_version=1,
        optimizer_boundary=True,
        binding=binding,
        epoch=2,
        next_update=0,
        scheduled_epoch=True,
        model=model.state_dict(),
    )
    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    rows = []
    ids = [f"1.2.{index}" for index in range(size)]
    cache = tmp_path / "cache"
    for uid in ids:
        sid = uid + ".1"
        count = 2 + int(uid.rsplit(".", 1)[1]) if whole else 2
        sops = [sid + f".{i + 1}" for i in range(count)]
        relative = f"test_series/{uid}/{sid}.npz"
        path = cache / relative
        path.parent.mkdir(parents=True)
        np.savez_compressed(
            path, volume=np.arange(count * 256, dtype=np.uint16).reshape(count, 16, 16)
        )
        source_paths = sorted(f"test_series/{uid}/{sid}/{sop}.dcm" for sop in sops)
        metadata = dict(
            cache_version=CACHE_VERSION,
            study_uid=uid,
            series_uid=sid,
            source_revision="synthetic-test",
            split="test_series",
            side=16,
            cache_path=relative,
            cache_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            cache_bytes=path.stat().st_size,
            n_slices=count,
            original_shape=[count, 16, 16],
            sop_uids=sops,
            slice_sha256=["a" * 64] * count if whole else ["a" * 64, "b" * 64],
            path_fingerprint=hashlib.sha256("\n".join(source_paths).encode()).hexdigest(),
            ordering="patient_geometry",
            orientation=[1, 0, 0, 0, 1, 0],
            positions=[[0, 0, i * 2] for i in range(count)],
        )
        path.with_suffix(".json").write_text(json.dumps(metadata))
        rows.append(
            dict(
                StudyInstanceUID=uid,
                SeriesInstanceUID=sid,
                Anatomical_Plane="Sagittal",
                Fluid_Sensitive=1,
                Fat_Suppression=0,
            )
        )
    series_path = tmp_path / "series.csv"
    pd.DataFrame(rows).to_csv(series_path, index=False)
    template_path = tmp_path / "template.csv"
    pd.DataFrame(
        [dict(StudyInstanceUID=uid, **{target: 0 for target in TARGETS}) for uid in reversed(ids)],
        columns=["StudyInstanceUID", *TARGETS],
    ).to_csv(template_path, index=False)
    args = argparse.Namespace(
        checkpoint=checkpoint_path,
        device="cpu",
        allow_diagnostic=True,
        scheduled_epoch=False,
        encoder_spec=None if checkpoint_only else spec_path,
        encoder_config=config_path if checkpoint_only else None,
        precision="checkpoint",
        series=series_path,
        template=template_path,
        cache=cache,
        dicom_root=None,
        source_revision="synthetic-test",
        batch_size=1,
        workers=0,
        output=tmp_path / "submission.csv",
    )
    return args, checkpoint


def test_full_coverage_context_checkpoint_infers_variable_batch_and_preserves_roster(
    tmp_path, monkeypatch
):
    args, state = _fixture(tmp_path, monkeypatch, size=3, whole=True)
    args.batch_size = 2
    infer.run(args)
    first = pd.read_csv(args.output)
    assert first.StudyInstanceUID.tolist() == pd.read_csv(args.template).StudyInstanceUID.tolist()
    assert list(first.columns) == ["StudyInstanceUID", *TARGETS]
    assert np.isfinite(first[list(TARGETS)].to_numpy()).all()
    assert ((first[list(TARGETS)] >= 0) & (first[list(TARGETS)] <= 1)).all().all()
    args.batch_size = 1
    args.output = tmp_path / "single-submission.csv"
    infer.run(args)
    second = pd.read_csv(args.output)
    np.testing.assert_allclose(first[list(TARGETS)], second[list(TARGETS)], rtol=0, atol=1e-6)
    assert state["binding"]["config"]["slice_context"] == "neighbor3"


def test_relation_training_checkpoint_infers_MRI_only_without_auxiliary_forward(
    tmp_path, monkeypatch
):
    policy = dict(weight=0.05, max_pairs=4, target_mode="physical", hidden_dim=12, head_seed=19)
    args, _ = _fixture(tmp_path, monkeypatch, relation=policy)

    def forbidden(*args, **kwargs):
        raise AssertionError("MRI-only classification must never execute the auxiliary head")

    monkeypatch.setattr(SliceDistanceHead, "forward", forbidden)
    infer.run(args)
    output = pd.read_csv(args.output)
    assert list(output) == ["StudyInstanceUID", *TARGETS]
    assert np.isfinite(output[list(TARGETS)].to_numpy()).all()


@pytest.mark.parametrize("checkpoint_only", [False, True])
def test_actual_cpu_inference_routes_and_explicit_fp32_override(
    tmp_path, monkeypatch, checkpoint_only
):
    args, state = _fixture(tmp_path, monkeypatch, checkpoint_only=checkpoint_only, precision="bf16")
    original_encoder = dict(state["binding"]["config"]["encoder"])
    with pytest.raises(ValueError, match="BF16"):
        infer.run(args)
    assert not args.output.exists()
    args.precision = "fp32"
    if checkpoint_only:

        def forbidden(*args, **kwargs):
            raise AssertionError(
                "Checkpoint-config route must not load original pretrained weights"
            )

        monkeypatch.setattr(infer, "create_encoder", forbidden)
    result = infer.run(args)
    assert result["training_precision"] == "bf16" and result["inference_precision"] == "fp32"
    assert result["encoder"] == original_encoder
    assert result["initial_pretrained_weights_read"] is False
    assert result["hidden_test_runtime_validated"] is False
    assert result["checkpoint_sha256"] == hashlib.sha256(args.checkpoint.read_bytes()).hexdigest()
    output = pd.read_csv(args.output)
    template = pd.read_csv(args.template)
    assert list(output.columns) == ["StudyInstanceUID", *TARGETS]
    assert output.StudyInstanceUID.tolist() == template.StudyInstanceUID.tolist()
    assert np.isfinite(output[list(TARGETS)].to_numpy()).all()
    persisted = torch.load(args.checkpoint, weights_only=True)
    assert persisted["binding"]["config"]["precision"] == "bf16"
    assert persisted["binding"]["config"]["encoder"] == original_encoder


@pytest.mark.parametrize(
    "field,value",
    [
        ("epoch", True),
        ("epoch", 2.0),
        ("next_update", False),
        ("next_update", 0.0),
        ("next_update", -1),
        ("optimizer_boundary", "yes"),
        ("optimizer_boundary", 1),
    ],
)
def test_invalid_completion_types_never_qualify(tmp_path, monkeypatch, field, value):
    args, state = _fixture(tmp_path, monkeypatch)
    if field == "epoch" and value is True:
        state["binding"]["config"]["epochs"] = 1
    state[field] = value
    with pytest.raises(ValueError):
        infer.validate_inference_checkpoint(
            state, allow_diagnostic=True, allow_scheduled_epoch=True
        )
    if field == "next_update":
        assert not scheduled_epoch(1, value, state["binding"]["config"])


def test_intermediate_requires_flag_marker_cadence_and_boundary(tmp_path, monkeypatch):
    args, state = _fixture(tmp_path, monkeypatch)
    state["epoch"] = 1
    with pytest.raises(ValueError, match="fixed final"):
        infer.validate_inference_checkpoint(state, allow_diagnostic=True)
    assert infer.validate_inference_checkpoint(
        state, allow_diagnostic=True, allow_scheduled_epoch=True
    )
    for change in [{"scheduled_epoch": False}, {"next_update": 1}, {"epoch": 3}]:
        with pytest.raises(ValueError):
            infer.validate_inference_checkpoint(
                dict(state, **change), allow_diagnostic=True, allow_scheduled_epoch=True
            )
    state["binding"]["config"]["submission_every_epochs"] = 2
    with pytest.raises(ValueError, match="fixed final"):
        infer.validate_inference_checkpoint(
            state, allow_diagnostic=True, allow_scheduled_epoch=True
        )


@pytest.mark.parametrize(
    "mutation",
    ["code", "provenance", "missing_encoder", "missing_head", "nonfinite_head", "config_bytes"],
)
def test_checkpoint_config_route_blocks_corruption_before_submission(
    tmp_path, monkeypatch, mutation
):
    args, state = _fixture(tmp_path, monkeypatch, checkpoint_only=True)
    if mutation == "code":
        state["binding"]["code_sha256"]["competitive/infer.py"] = "0" * 64
    elif mutation == "provenance":
        state["binding"]["config"]["encoder"]["embedding_dim"] = 1024
    elif mutation == "missing_encoder":
        state["model"].pop("encoder.backbone.projection.weight")
    elif mutation == "missing_head":
        state["model"].pop("output_bias")
    elif mutation == "nonfinite_head":
        state["model"]["output_bias"][0] = torch.nan
    else:
        args.encoder_config.write_bytes(args.encoder_config.read_bytes() + b" ")
    torch.save(state, args.checkpoint)
    with pytest.raises((ValueError, RuntimeError)):
        infer.run(args)
    assert not args.output.exists()


def test_diagnostic_permission_and_three_public_example_limit(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path, monkeypatch, size=4)
    args.allow_diagnostic = False
    with pytest.raises(ValueError, match="Diagnostic weights"):
        infer.run(args)
    args.allow_diagnostic = True
    with pytest.raises(ValueError, match="at most three"):
        infer.run(args)
    assert not args.output.exists()


def test_changed_original_spec_is_not_accepted_as_training_provenance(tmp_path, monkeypatch):
    args, _ = _fixture(tmp_path, monkeypatch)
    args.encoder_spec.write_text(
        json.dumps({"architecture": "diagnostic_tiny", "embedding_dim": 8})
    )
    with pytest.raises(ValueError, match="Encoder source"):
        infer.run(args)


def test_snapshot_metadata_conflict_never_overwrites_previous_record(tmp_path, monkeypatch):
    args, state = _fixture(tmp_path, monkeypatch)
    record = write_epoch_snapshot(args.checkpoint, tmp_path / "snapshots", state["binding"], 2, 8)
    before = (tmp_path / "snapshots/epoch-002.json").read_bytes()
    with pytest.raises(ValueError, match="metadata"):
        write_epoch_snapshot(args.checkpoint, tmp_path / "snapshots", state["binding"], 2, 9)
    assert (tmp_path / "snapshots/epoch-002.json").read_bytes() == before
    assert record["diagnostic_only"] and not record["submission_eligible"]


def test_cli_encoder_routes_are_mutually_exclusive(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "infer",
            "--checkpoint",
            "x",
            "--series",
            "x",
            "--template",
            "x",
            "--cache",
            "x",
            "--output",
            "x",
            "--encoder-spec",
            "x",
            "--encoder-config",
            "x",
        ],
    )
    with pytest.raises(SystemExit) as error:
        infer.main()
    assert error.value.code == 2
