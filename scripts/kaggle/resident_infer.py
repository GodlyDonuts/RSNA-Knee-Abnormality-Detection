"""Load one immutable trained model per device and reuse it across MRI chunks.

Standalone execution optimization: original rsna numerical modules, checkpoint,
preprocessing, source checks and sigmoid predictions stay unchanged. FP32 only;
this does not establish a nine-hour T4 deployment acceptance.
"""

from __future__ import annotations

import inspect
import json
import resource
import sys
import time
from dataclasses import replace
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from rsna.competitive import data as trained_data
from rsna.competitive import infer
from rsna.competitive.adapters import configure_encoder_training
from rsna.competitive.backbones import create_encoder, create_encoder_from_training_checkpoint
from rsna.competitive.delta_checkpoint import FrozenBaseDelta, restore_checkpoint_model
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.train import TrainConfig, predict
from rsna.infer import load_inference_metadata, write_submission
from rsna.schema import TARGETS

CompetitiveStudyDataset = trained_data.CompetitiveStudyDataset
collate_studies = trained_data.collate_studies


def evaluation_slice_sampling(policy):
    original = getattr(trained_data, "evaluation_slice_sampling", None)
    if original is not None:
        return original(policy)
    if policy not in {"uniform", "all"}:
        raise ValueError("Historical source does not admit this evaluation policy")
    return policy


def signature(path: Path) -> tuple:
    value = path.stat()
    if path.is_symlink() or not path.is_file():
        raise ValueError("Resident input must remain a regular immutable file")
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


class ResidentInference:
    def __init__(self, args, *, device_index: int = 0):
        started = time.monotonic()
        if args.precision != "fp32" or args.workers != 0 or args.batch_size != 1:
            raise ValueError("Initial resident admission fixes FP32/one-study/zero-worker views")
        if args.dicom_root is not None or getattr(args, "baseline_epoch_one", False):
            raise ValueError(
                "Resident execution needs prepared MRI and an admitted final checkpoint"
            )
        self.device = torch.device(f"cuda:{device_index}" if args.device == "cuda" else args.device)
        if self.device.type not in {"cuda", "cpu"}:
            raise ValueError("Unsupported resident device")
        if self.device.type == "cuda":
            if not torch.cuda.is_available() or device_index >= torch.cuda.device_count():
                raise ValueError("An assigned CUDA device is required")
            torch.cuda.set_device(self.device)
        torch.set_num_threads(1)
        self.checkpoint_sha = infer.file_sha256(args.checkpoint)
        self.saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        reviewed = getattr(args, "fitting_admission", None)
        if reviewed is None:
            binding = infer.validate_inference_checkpoint(
                self.saved,
                allow_diagnostic=args.allow_diagnostic,
                allow_scheduled_epoch=getattr(args, "scheduled_epoch", False),
            )
        else:
            if args.allow_diagnostic:
                raise ValueError("Reviewed admission cannot authorize diagnostic weights")
            if __package__:
                from .reviewed_infer import read_admission, validate_reviewed
            else:
                from reviewed_infer import read_admission, validate_reviewed

            binding = validate_reviewed(self.saved, read_admission(reviewed), self.checkpoint_sha)
        package = Path(infer.__file__).resolve().parents[1]
        hashes = {
            str(p.relative_to(package)): infer.file_sha256(p) for p in sorted(package.rglob("*.py"))
        }
        if binding["code_sha256"] != hashes:
            raise ValueError("Resident numerical source differs from the trained checkpoint")
        self.package, self.source_hashes = package, hashes
        channels = (
            3
            if (binding["config"].get("neighbor_channels") or {}).get("mode")
            in {"physical", "distant"}
            else 1
        )
        if args.encoder_config is not None:
            if self.saved["schema_version"] != 1:
                raise ValueError("Delta residency requires the pinned original encoder spec")
            spec = dict(binding["config"]["encoder"])
            encoder, dim, provenance = create_encoder_from_training_checkpoint(
                spec,
                args.encoder_config,
                encoder_state={
                    k.removeprefix("encoder."): v
                    for k, v in self.saved["model"].items()
                    if k.startswith("encoder.")
                },
            )
        else:
            spec = json.loads(args.encoder_spec.read_text())
            encoder, dim, provenance = create_encoder(
                spec, **({"input_channels": 3} if channels == 3 else {})
            )
        if provenance != binding["config"]["encoder"]:
            raise ValueError("Resident base/provenance differs from training")
        values = dict(binding["config"])
        values["encoder"] = spec
        self.config = TrainConfig(**values)
        self.config.validate()
        self.training_precision = self.config.precision
        self.config = replace(self.config, precision="fp32")
        self.config.validate()
        kwargs = dict(
            pooling=self.config.pooling,
            hidden_dim=self.config.hidden_dim,
            encode_chunk_size=self.config.encode_chunk_size,
        )
        supported = inspect.signature(CompetitiveStudyNet).parameters
        for name, value in [
            ("slice_context", getattr(self.config, "slice_context", "none")),
            ("witness", getattr(self.config, "witness", None)),
            ("input_channels", channels),
        ]:
            if name in supported:
                kwargs[name] = value
            elif value not in (None, "none", 1):
                raise ValueError("Frozen model cannot accept this policy")
        if getattr(self.config, "relation", None) is not None:
            raise ValueError("Relation checkpoints require separate resident admission")
        self.model = CompetitiveStudyNet(encoder, dim, **kwargs).to(self.device)
        configure_encoder_training(encoder, self.config.freeze_encoder)
        guard = (
            FrozenBaseDelta(self.model, provenance) if self.saved["schema_version"] == 2 else None
        )
        restore_checkpoint_model(self.model, self.saved, guard)
        self.model.eval()
        self.binding, self.provenance = binding, provenance
        paths = [args.checkpoint, args.encoder_config or args.encoder_spec]
        if args.encoder_spec is not None:
            paths += [Path(spec[k]) for k in ("weights_path", "config_path") if k in spec]
        if reviewed is not None:
            paths.append(reviewed)
        self.immutable = {str(p): signature(p) for p in paths}
        self.identity = self.options(args)
        self.saved_epoch = self.saved["epoch"]
        self.model_load_count = 1
        self.requests = 0
        self.startup_seconds = time.monotonic() - started
        # Inference never needs the AdamW state, RNG envelope or CPU learned copy.
        del self.saved

    @staticmethod
    def options(args):
        return {
            k: str(getattr(args, k, None))
            for k in (
                "checkpoint",
                "encoder_config",
                "encoder_spec",
                "device",
                "precision",
                "allow_diagnostic",
                "scheduled_epoch",
                "fitting_admission",
                "batch_size",
                "workers",
            )
        }

    def run(self, args):
        started = time.monotonic()
        if self.options(args) != self.identity or args.dicom_root is not None:
            raise ValueError("Resident model/runtime admission cannot change between chunks")
        if any(signature(Path(p)) != expected for p, expected in self.immutable.items()):
            raise ValueError("Immutable resident checkpoint/base/admission changed")
        if {
            str(p.relative_to(self.package)): infer.file_sha256(p)
            for p in sorted(self.package.rglob("*.py"))
        } != self.source_hashes:
            raise ValueError("Resident numerical source changed after admission")
        if args.output.exists():
            raise ValueError("Resident output must be new")
        series, studies, template, inputs = load_inference_metadata(
            {}, args.series, None, args.template, "test"
        )
        if args.allow_diagnostic and len(studies) > 3:
            raise ValueError("Diagnostic weights remain limited to three public examples")
        data = dict(
            raw_split="test_series",
            slices=self.config.slices,
            side=self.config.side,
            max_series=self.config.max_series,
            slice_sampling=evaluation_slice_sampling(self.config.slice_sampling),
            series_sampling=self.config.series_sampling,
        )
        if getattr(self.config, "neighbor_channels", None) is not None:
            data["neighbor_channels"] = self.config.neighbor_channels
        dataset = CompetitiveStudyDataset(studies, series, args.cache, **data)
        if dataset.source_revision != args.source_revision:
            raise ValueError("Resident cache source differs from admitted request")
        if dataset.cache_side != self.binding["cache_side"]:
            raise ValueError("Resident cache resolution differs")
        preprocessing = time.monotonic() - started
        loader = DataLoader(
            dataset,
            batch_size=1,
            num_workers=0,
            collate_fn=collate_studies,
            generator=torch.Generator().manual_seed(self.config.seed),
        )
        with torch.inference_mode():
            frame, _ = predict(self.model, loader, self.device, self.config)
        frame = frame.rename(columns={"study_uid": "StudyInstanceUID"})[
            ["StudyInstanceUID", *TARGETS]
        ]
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_submission(frame, template, args.output)
        self.requests += 1
        result = {
            "schema_version": 1,
            "status": "passed",
            "diagnostic_only": self.config.diagnostic_study_limit is not None,
            "studies": len(studies),
            "device_type": self.device.type,
            "elapsed_seconds": time.monotonic() - started,
            "preprocessing_seconds": preprocessing,
            "checkpoint_sha256": self.checkpoint_sha,
            "submission_sha256": infer.file_sha256(args.output),
            "training_release_binding": self.binding["release"],
            "test_inputs": inputs,
            "cache_version": self.binding["cache_version"],
            "cache_side": dataset.cache_side,
            "cache_source_revision": dataset.source_revision,
            "encoder": self.provenance,
            "hidden_test_runtime_validated": False,
            "checkpoint_epoch": self.saved_epoch,
            "scheduled_epoch_authorized": getattr(args, "scheduled_epoch", False),
            "training_precision": self.training_precision,
            "inference_precision": "fp32",
            "initial_pretrained_weights_read": args.encoder_spec is not None,
            "resident_model_load_count": 1,
            "resident_requests": self.requests,
            "resident_startup_seconds": self.startup_seconds,
            "resident_device_index": self.device.index,
            "host_peak_rss_bytes": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                * (1 if sys.platform == "darwin" else 1024)
            ),
            "GPU_peak_allocated_bytes": torch.cuda.max_memory_allocated(self.device)
            if self.device.type == "cuda"
            else None,
        }
        args.output.with_suffix(".metrics.json").write_text(
            json.dumps(result, sort_keys=True, indent=2) + "\n"
        )
        return result
