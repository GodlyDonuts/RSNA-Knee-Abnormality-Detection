"""Offline competition-template inference for a frozen competitive checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rsna.competitive.adapters import configure_encoder_training
from rsna.competitive.backbones import create_encoder, create_encoder_from_training_checkpoint
from rsna.competitive.checkpoint import atomic_json, file_sha256
from rsna.competitive.data import (
    CompetitiveStudyDataset,
    collate_studies,
    evaluation_slice_sampling,
)
from rsna.competitive.delta_checkpoint import FrozenBaseDelta, restore_checkpoint_model
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.relations import relation_model_kwargs
from rsna.competitive.snapshots import scheduled_epoch
from rsna.competitive.train import TrainConfig, predict
from rsna.infer import load_inference_metadata, write_submission
from rsna.preprocess import Source, cache_series
from rsna.release import validate_checkpoint_binding
from rsna.schema import CACHE_VERSION, TARGETS


def validate_inference_checkpoint(
    checkpoint: dict, *, allow_diagnostic: bool = False, allow_scheduled_epoch: bool = False
) -> dict:
    if (
        checkpoint.get("schema_version") not in {1, 2}
        or checkpoint.get("optimizer_boundary") is not True
    ):
        raise ValueError("Unsupported competitive checkpoint")
    binding = checkpoint["binding"]
    if binding.get("external_initialization") is not None or binding.get("private_research_only"):
        raise ValueError("Private external-source research weights are not competition candidates")
    validate_checkpoint_binding(binding["release"])
    if binding.get("targets") != list(TARGETS) or binding.get("cache_version") != CACHE_VERSION:
        raise ValueError("Checkpoint targets or preprocessing version differ")
    if binding.get("source_revision") != binding["release"]["source_revision"]:
        raise ValueError("Checkpoint image source conflicts with supervision release")
    config = binding["config"]
    expected_schema = 2 if config.get("checkpoint_format") == "frozen_base_delta" else 1
    if checkpoint["schema_version"] != expected_schema:
        raise ValueError("Checkpoint schema conflicts with its bound storage format")
    if any(
        type(checkpoint.get(name)) is not int or checkpoint[name] < 0
        for name in ("epoch", "next_update")
    ):
        raise ValueError("Checkpoint progress counters must be nonnegative integers")
    final = checkpoint.get("epoch") == config["epochs"] and checkpoint.get("next_update") == 0
    scheduled = (
        allow_scheduled_epoch
        and checkpoint.get("scheduled_epoch") is True
        and scheduled_epoch(checkpoint["epoch"], checkpoint["next_update"], config)
    )
    if not final and not scheduled:
        raise ValueError(
            "Competition inference requires the fixed final checkpoint or an explicitly scheduled completed epoch"
        )
    if config.get("diagnostic_study_limit") is not None and not allow_diagnostic:
        raise ValueError("Diagnostic weights require explicit sample-test-only authorization")
    return binding


def run(args: argparse.Namespace) -> dict:
    started = time.monotonic()
    torch.set_num_threads(2)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA requested but unavailable; no automatic CPU fallback")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    binding = validate_inference_checkpoint(
        checkpoint,
        allow_diagnostic=args.allow_diagnostic,
        allow_scheduled_epoch=getattr(args, "scheduled_epoch", False),
    )
    package_root = Path(__file__).resolve().parents[1]
    current_code = {
        str(path.relative_to(package_root)): file_sha256(path)
        for path in sorted(package_root.rglob("*.py"))
    }
    if binding.get("code_sha256") != current_code:
        raise ValueError("Inference architecture/preprocessing code differs from training")
    if getattr(args, "encoder_config", None) is not None:
        if checkpoint["schema_version"] == 2:
            raise ValueError(
                "Delta inference requires --encoder-spec with the pinned original weights"
            )
        encoder_spec = dict(binding["config"]["encoder"])
        encoder, dimension, provenance = create_encoder_from_training_checkpoint(
            encoder_spec,
            args.encoder_config,
            encoder_state={
                name.removeprefix("encoder."): tensor
                for name, tensor in checkpoint["model"].items()
                if name.startswith("encoder.")
            },
        )
    else:
        encoder_spec = json.loads(args.encoder_spec.read_text())
        encoder, dimension, provenance = create_encoder(encoder_spec)
    if provenance != binding["config"]["encoder"]:
        raise ValueError("Encoder source/preprocessing differs from the training checkpoint")
    config_values = dict(binding["config"])
    config_values["encoder"] = encoder_spec
    config = TrainConfig(**config_values)
    config.validate()
    training_precision = config.precision
    requested_precision = getattr(args, "precision", "checkpoint")
    if requested_precision != "checkpoint":
        config = replace(config, precision=requested_precision)
        config.validate()
    if config.precision == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("Checkpoint BF16 inference needs a verified compatible CUDA GPU")
    series, studies, template, input_binding = load_inference_metadata(
        {}, args.series, None, args.template, "test"
    )
    assert template is not None
    if args.allow_diagnostic and len(studies) > 3:
        raise ValueError("Diagnostic inference is restricted to at most three public examples")
    if args.dicom_root:
        source = Source(args.dicom_root)
        for uid in studies.study_uid:
            for sid in series.loc[series.StudyInstanceUID == uid, "SeriesInstanceUID"]:
                folder = args.dicom_root / "test_series" / uid / sid
                paths = sorted(
                    str(path.relative_to(args.dicom_root)) for path in folder.glob("*.dcm")
                )
                if not paths:
                    raise ValueError("Test series has no local DICOMs")
                cache_series(source, paths, args.cache, binding["cache_side"], args.source_revision)
    dataset = CompetitiveStudyDataset(
        studies,
        series,
        args.cache,
        raw_split="test_series",
        slices=config.slices,
        side=config.side,
        max_series=config.max_series,
        slice_sampling=evaluation_slice_sampling(config.slice_sampling),
        series_sampling=config.series_sampling,
    )
    if dataset.cache_side != binding["cache_side"]:
        raise ValueError("Inference cache resolution differs from the trained transform")
    preprocessing_seconds = time.monotonic() - started
    model = CompetitiveStudyNet(
        encoder,
        dimension,
        pooling=config.pooling,
        hidden_dim=config.hidden_dim,
        encode_chunk_size=config.encode_chunk_size,
        slice_context=config.slice_context,
        witness=config.witness,
        **relation_model_kwargs(config.relation),
    ).to(device)
    configure_encoder_training(encoder, config.freeze_encoder)
    guard = FrozenBaseDelta(model, provenance) if checkpoint["schema_version"] == 2 else None
    restore_checkpoint_model(model, checkpoint, guard)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=collate_studies,
    )
    predictions, _ = predict(model, loader, device, config)
    predictions = predictions.rename(columns={"study_uid": "StudyInstanceUID"})[
        ["StudyInstanceUID", *TARGETS]
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_submission(predictions, template, args.output)
    elapsed = time.monotonic() - started
    result = {
        "schema_version": 1,
        "status": "passed",
        "diagnostic_only": config.diagnostic_study_limit is not None,
        "studies": len(studies),
        "device_type": device.type,
        "elapsed_seconds": elapsed,
        "preprocessing_seconds": preprocessing_seconds,
        "inference_studies_per_second": len(studies) / max(elapsed - preprocessing_seconds, 1e-9),
        "checkpoint_sha256": file_sha256(args.checkpoint),
        "submission_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "training_release_binding": binding["release"],
        "test_inputs": input_binding,
        "cache_version": CACHE_VERSION,
        "cache_side": dataset.cache_side,
        "cache_source_revision": dataset.source_revision,
        "encoder": provenance,
        "hidden_test_runtime_validated": False,
        "checkpoint_epoch": checkpoint["epoch"],
        "scheduled_epoch_authorized": bool(getattr(args, "scheduled_epoch", False)),
        "training_precision": training_precision,
        "inference_precision": config.precision,
        "training_slice_sampling": config.slice_sampling,
        "inference_slice_sampling": evaluation_slice_sampling(config.slice_sampling),
        "initial_pretrained_weights_read": (
            getattr(args, "encoder_config", None) is None and provenance.get("pretrained") is True
        ),
    }
    atomic_json(args.output.with_suffix(".metrics.json"), result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("checkpoint", "series", "template", "cache", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    encoder_group = parser.add_mutually_exclusive_group(required=True)
    encoder_group.add_argument("--encoder-spec", type=Path)
    encoder_group.add_argument(
        "--encoder-config",
        type=Path,
        help="Official architecture config; load all encoder weights from the bound trained checkpoint",
    )
    parser.add_argument("--dicom-root", type=Path)
    parser.add_argument("--source-revision", default="kaggle-runtime-test")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--allow-diagnostic", action="store_true")
    parser.add_argument(
        "--precision",
        choices=("checkpoint", "fp32", "bf16"),
        default="checkpoint",
        help="Explicit deployment precision; FP32 supports GPUs without BF16",
    )
    parser.add_argument(
        "--scheduled-epoch",
        action="store_true",
        help="Permit only a completed epoch fixed by the training submission cadence",
    )
    args = parser.parse_args()
    if args.batch_size < 1 or args.workers < 0:
        parser.error("Invalid batch size or worker count")
    print(json.dumps(run(args), sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
