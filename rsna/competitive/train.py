"""Fixed-schedule outer-fold experiments; no gold or confirmation-set selection."""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import math
import os
import random
import signal
import stat
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from rsna.competitive.adapters import configure_encoder_training, validate_adapter
from rsna.competitive.cache_contrast import validate_cache_contrast
from rsna.competitive.canonical_side import (
    load_canonical_side_table,
    validate_canonical_side_policy,
)
from rsna.competitive.checkpoint import (
    atomic_json,
    capture_rng,
    file_sha256,
    load_checkpoint,
    restore_rng,
    run_lock,
    save_checkpoint,
)
from rsna.competitive.delta_checkpoint import FrozenBaseDelta, restore_checkpoint_model
from rsna.competitive.feature_summary import validate_patch_summary
from rsna.competitive.loss import (
    adjacent_slice_pairs,
    effective_batch_loss,
    masked_cell_consistency,
    masked_cell_entropy,
    observed_counts,
    slice_score_tv,
    validate_masked_cell,
    validate_slice_smooth,
)
from rsna.competitive.mrnet_initialization import initialize_private_mrnet
from rsna.competitive.neighbor_input import neighbor_input_channels
from rsna.competitive.plane_memory import validate_plane_memory
from rsna.competitive.relations import (
    effective_relation_loss,
    relation_model_kwargs,
    relation_study_count,
    relations_for_batch,
    validate_relation,
)
from rsna.competitive.sampling import EffectiveBatchSampler, SeededDataset
from rsna.competitive.snapshots import scheduled_epoch, write_epoch_snapshot
from rsna.competitive.supervision import apply_fitting_supervision, validate_fitting_supervision
from rsna.competitive.witness import validate_witness
from rsna.model import auc_metrics
from rsna.release import load_verified_release
from rsna.schema import CACHE_VERSION, TARGETS

STOP_REQUEST_NAME = ".checkpoint-stop-request.json"


def external_stop_requested(output: Path) -> bool:
    """Accept only this supervisor invocation's atomic request file."""
    session = os.environ.get("RSNA_STOP_SESSION")
    if not session:
        return False
    request = output / STOP_REQUEST_NAME
    try:
        metadata = request.lstat()
    except FileNotFoundError:
        return False
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > 4096:
        raise ValueError("Invalid checkpoint stop request")
    record = json.loads(request.read_text())
    if (
        not isinstance(record, dict)
        or record.get("status") != "checkpoint_requested"
        or record.get("checkpoint_name") != "checkpoint.pt"
        or not isinstance(record.get("session"), str)
        or not isinstance(record.get("requested_wall"), (float, int))
    ):
        raise ValueError("Malformed checkpoint stop request")
    return record["session"] == session


@dataclass(frozen=True)
class TrainConfig:
    encoder: dict = field(
        default_factory=lambda: {"architecture": "diagnostic_tiny", "embedding_dim": 16}
    )
    fold: int = 0
    fit_scope: str = "outer_fold"
    seed: int = 20260922
    epochs: int = 10
    batch_size: int = 1
    accumulation_steps: int = 4
    slices: int = 16
    side: int = 224
    max_series: int = 6
    workers: int = 0
    head_lr: float = 0.001
    encoder_lr: float = 0.00001
    weight_decay: float = 0.0001
    freeze_encoder: bool = True
    precision: str = "fp32"
    amp_init_scale: float = 128.0
    pooling: str = "attention"
    slice_context: str = "none"
    slice_sampling: str = "uniform"
    series_sampling: str = "selected"
    hidden_dim: int = 128
    encode_chunk_size: int = 32
    augment: bool = False
    warmup_updates: int = 0
    lr_schedule: str = "constant"
    clip_grad_norm: float = 1.0
    checkpoint_every_updates: int = 25
    checkpoint_format: str = "full"
    submission_every_epochs: int = 0
    min_free_gib: float = 20.0
    diagnostic_study_limit: int | None = None
    relation: dict | None = None
    witness: dict | None = None
    plane_memory: dict | None = None
    fitting_supervision: dict | None = None
    neighbor_channels: dict | None = None
    canonical_sides: dict | None = None
    masked_cell: dict | None = None
    slice_smooth: dict | None = None
    patch_summary: dict | None = None
    cache_contrast: dict | None = None

    def validate(self) -> None:
        if self.masked_cell is not None:
            validate_masked_cell(self.masked_cell)
            if (
                self.fit_scope != "outer_fold"
                or self.relation is not None
                or self.witness is not None
                or self.fitting_supervision is not None
                or self.neighbor_channels is not None
                or self.canonical_sides is not None
                or self.patch_summary is not None
                or self.cache_contrast is not None
            ):
                raise ValueError("Masked-cell contrast requires an isolated outer-fold arm")
        if self.slice_smooth is not None:
            validate_slice_smooth(self.slice_smooth)
            if (
                self.fit_scope != "outer_fold"
                or self.relation is not None
                or self.witness is not None
                or self.fitting_supervision is not None
                or self.neighbor_channels is not None
                or self.canonical_sides is not None
                or self.masked_cell is not None
                or self.patch_summary is not None
                or self.cache_contrast is not None
                or self.pooling != "attention"
                or self.slice_context != "none"
            ):
                raise ValueError("Slice-smooth contrast requires an isolated outer-fold arm")
        validate_cache_contrast(self.cache_contrast)
        if self.cache_contrast is not None and (
            self.fit_scope != "outer_fold"
            or self.augment
            or self.pooling != "attention"
            or self.slice_context != "none"
            or self.slice_sampling != "uniform"
            or self.series_sampling != "selected"
            or any(
                value is not None
                for value in (
                    self.neighbor_channels,
                    self.relation,
                    self.witness,
                    self.plane_memory,
                    self.patch_summary,
                    self.fitting_supervision,
                    self.canonical_sides,
                    self.masked_cell,
                    self.slice_smooth,
                )
            )
            or self.encoder.get("architecture") not in {"dinov3_l16", "diagnostic_tiny"}
        ):
            raise ValueError("Cache contrast must isolate its input change")
        if self.patch_summary is not None:
            validate_patch_summary(self.patch_summary)
            if (
                self.pooling != "attention"
                or self.slice_context != "none"
                or any(
                    value is not None
                    for value in (
                        self.witness,
                        self.plane_memory,
                        self.relation,
                        self.neighbor_channels,
                        self.fitting_supervision,
                        self.canonical_sides,
                        self.masked_cell,
                        self.slice_smooth,
                        self.cache_contrast,
                    )
                )
                or self.encoder.get("architecture") not in {"dinov3_l16", "diagnostic_tiny"}
            ):
                raise ValueError("CLS/patch-mean baseline must isolate its single readout change")
        neighbor_input_channels(self.neighbor_channels)
        if self.neighbor_channels is not None and (
            self.fit_scope != "outer_fold"
            or self.augment
            or self.pooling != "attention"
            or self.slice_context != "none"
            or self.slice_sampling != "uniform"
            or self.series_sampling != "selected"
            or self.relation is not None
            or self.witness is not None
            or self.fitting_supervision is not None
            or not isinstance(self.encoder, dict)
            or self.encoder.get("architecture") not in {"dinov3_l16", "diagnostic_tiny"}
        ):
            raise ValueError("Neighbor input comparison must isolate fixed uniform outer-fold arms")
        if self.fitting_supervision is not None:
            validate_fitting_supervision(self.fitting_supervision)
            if (
                self.fit_scope != "outer_fold"
                or self.relation is not None
                or self.witness is not None
            ):
                raise ValueError("Fitting-supervision contrast requires an isolated outer-fold arm")
            if self.fitting_supervision["kind"] == "label_source_v1" and (
                self.plane_memory is not None
                or self.pooling != "attention"
                or self.slice_context != "none"
            ):
                raise ValueError("Label-source contrast requires the plain attention readout")
        if self.canonical_sides is not None:
            validate_canonical_side_policy(self.canonical_sides)
            if (
                self.fit_scope != "outer_fold"
                or self.relation is not None
                or self.witness is not None
                or self.fitting_supervision is not None
                or self.neighbor_channels is not None
                or self.cache_contrast is not None
            ):
                raise ValueError("Canonical-side contrast requires an isolated outer-fold arm")
        positive = (
            "epochs",
            "batch_size",
            "accumulation_steps",
            "slices",
            "side",
            "max_series",
            "hidden_dim",
            "encode_chunk_size",
            "checkpoint_every_updates",
        )
        nonnegative = ("fold", "seed", "workers", "warmup_updates", "submission_every_epochs")
        for name in positive + nonnegative:
            value = getattr(self, name)
            if type(value) is not int or value < (1 if name in positive else 0):
                raise ValueError(f"Invalid integer configuration: {name}")
        for name in ("head_lr", "encoder_lr", "clip_grad_norm", "weight_decay", "min_free_gib"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (float, int))
                or not math.isfinite(value)
                or value < 0
                or (name in ("head_lr", "encoder_lr", "clip_grad_norm") and value == 0)
            ):
                raise ValueError(f"Invalid numeric configuration: {name}")
        if self.side < 16 or self.max_series < 3:
            raise ValueError("Need side >=16 and at least three series slots")
        if self.precision not in {"fp32", "bf16", "fp16"} or self.pooling not in {
            "attention",
            "max",
            "patch_attention",
        }:
            raise ValueError("Unknown precision or pooling policy")
        if (
            type(self.amp_init_scale) not in (int, float)
            or not math.isfinite(self.amp_init_scale)
            or not 1 <= self.amp_init_scale <= 65536
            or not math.log2(self.amp_init_scale).is_integer()
            or (self.precision != "fp16" and self.amp_init_scale != 128)
        ):
            raise ValueError("FP16 loss scale must be an explicit positive power of two")
        if any(type(getattr(self, name)) is not bool for name in ("freeze_encoder", "augment")):
            raise ValueError("Freeze and augmentation settings must be booleans")
        if self.slice_sampling not in {
            "uniform",
            "stratified",
            "all",
        } or self.series_sampling not in {
            "selected",
            "all",
        }:
            raise ValueError("Unknown examination coverage policy")
        if self.slice_sampling == "stratified" and self.augment:
            raise ValueError("Stratified slice experiment requires fixed series selection")
        if self.slice_context not in {"none", "neighbor3"} or (
            self.slice_context != "none" and self.pooling != "attention"
        ):
            raise ValueError("Unknown or incompatible neighboring-slice context")
        if self.relation is not None:
            validate_relation(self.relation)
            if self.seed >= 2**63:
                raise ValueError("Relation sampling requires seed below 2**63")
            if self.pooling != "attention" or self.slice_context != "none":
                raise ValueError(
                    "First relation experiment requires global attention without context"
                )
        if self.witness is not None:
            validate_witness(self.witness)
            if (
                self.pooling != "attention"
                or self.slice_context != "none"
                or self.relation is not None
                or self.encoder.get("architecture") not in {"dinov3_l16", "diagnostic_tiny"}
            ):
                raise ValueError("Witness comparison requires plain attention and spatial encoder")
        if self.plane_memory is not None:
            validate_plane_memory(self.plane_memory)
            if (
                self.pooling != "attention"
                or self.slice_context != "none"
                or self.witness is not None
                or self.relation is not None
                or self.neighbor_channels is not None
                or self.encoder.get("architecture") not in {"dinov3_l16", "diagnostic_tiny"}
            ):
                raise ValueError(
                    "Initial sequence-memory contrast requires plain single-channel spatial DINO"
                )
        if self.diagnostic_study_limit is not None and (
            type(self.diagnostic_study_limit) is not int or self.diagnostic_study_limit < 1
        ):
            raise ValueError("Diagnostic study limit must be positive")
        if not isinstance(self.encoder, dict):
            raise ValueError("Encoder specification must be an object")
        if self.lr_schedule not in {"constant", "cosine"}:
            raise ValueError("Unknown learning-rate schedule")
        if self.fit_scope not in {"outer_fold", "all_development"}:
            raise ValueError("Unknown training pool policy")
        if self.checkpoint_format not in {"full", "frozen_base_delta"} or (
            self.checkpoint_format == "frozen_base_delta"
            and (not self.freeze_encoder or "adapter" not in self.encoder)
        ):
            raise ValueError("Delta checkpoints require a frozen pretrained base and adapters")
        if "adapter" in self.encoder:
            if self.encoder.get("architecture") != "dinov3_l16" or not self.freeze_encoder:
                raise ValueError("DINOv3-L adapters require a frozen original encoder")
            policy = self.encoder["adapter"]
            if not isinstance(policy, dict):
                raise ValueError("Adapter policy must be an object")
            policy_keys = {"rank", "alpha", "dropout", "ffn_last_layers", "gradient_checkpointing"}
            derived_keys = {
                "method",
                "scaling",
                "base_frozen",
                "zero_initial_residual",
                "target_modules",
                "trainable_parameters",
            }
            if not policy_keys <= set(policy) or set(policy) - policy_keys - derived_keys:
                raise ValueError("Missing or unknown adapter policy fields")
            validate_adapter({key: policy[key] for key in policy_keys}, 24)
        if self.pooling == "patch_attention" and self.encoder.get("architecture") not in {
            "dinov3_l16",
            "diagnostic_tiny",
        }:
            raise ValueError("Patch attention requires the pinned DINOv3-L encoder")
        if (
            self.encoder.get("architecture") == "diagnostic_tiny"
            and self.diagnostic_study_limit is None
        ):
            raise ValueError("The diagnostic encoder requires an explicit diagnostic study limit")


def _hash(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def learning_rate_multiplier(step: int, total: int, warmup: int, policy: str) -> float:
    warm = min(1.0, (step + 1) / max(1, warmup))
    if policy == "constant" or step < warmup:
        return warm
    progress = min(1.0, max(0.0, (step - warmup) / max(1, total - warmup - 1)))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def _uid_hash(frame: pd.DataFrame) -> str:
    return _hash(sorted(frame.study_uid.tolist()))


def verified_job_release(
    studies: Path, artifact_root: Path, rank: int, world: int
) -> tuple[pd.DataFrame, dict]:
    """Replay full release evidence once per job, then share the verified table.

    Repeating the same large provenance replay on every GPU wastes host memory.
    Every rank still checks its local manifest against the verified byte hash.
    """
    if world == 1:
        return load_verified_release(studies, artifact_root)
    shared: list = [None]
    if rank == 0:
        try:
            manifest, binding = load_verified_release(studies, artifact_root)
            shared[0] = {"manifest": manifest, "binding": binding}
        except Exception as exc:
            shared[0] = {"error_type": type(exc).__name__}
    dist.broadcast_object_list(shared, src=0)
    result = shared[0]
    if "error_type" in result:
        raise ValueError("Full data-release validation failed on the coordinating rank")
    if file_sha256(studies) != result["binding"]["manifest_sha256"]:
        raise ValueError("Local manifest differs from the verified distributed job release")
    return result["manifest"], result["binding"]


def select_development(
    manifest: pd.DataFrame, folds: pd.DataFrame, fold: int, fit_scope: str = "outer_fold"
) -> tuple[pd.DataFrame, pd.DataFrame]:
    development = manifest[manifest.split == "train"].copy()
    if fold not in set(folds.fold) or development.study_uid.duplicated().any():
        raise ValueError("Invalid outer fold or duplicate studies")
    if set(development.study_uid) != set(folds.study_uid) or folds.study_uid.duplicated().any():
        raise ValueError("Folds must cover exactly the accepted training split")
    joined = development.merge(folds[["study_uid", "fold"]], on="study_uid", validate="one_to_one")
    observed = joined[[f"{target}__mask" for target in TARGETS]].sum(axis=1) > 0
    train = joined[observed & (joined.fold != fold)].sort_values("study_uid").reset_index(drop=True)
    outer = joined[observed & (joined.fold == fold)].sort_values("study_uid").reset_index(drop=True)
    forbidden = manifest[manifest.split != "train"]
    if train.empty or outer.empty or set(train.leakage_group) & set(outer.leakage_group):
        raise ValueError("Development folds are empty or share leakage groups")
    if set(joined.leakage_group) & set(forbidden.leakage_group):
        raise ValueError("Development folds overlap a held-out split")
    if any(joined[f"{target}__provenance"].eq("official_gold").any() for target in TARGETS):
        raise ValueError("Official labels cannot enter model development")
    if fit_scope == "all_development":
        # The same accepted development membership and masks, with no model
        # selection or evaluation on studies now entering the training pool.
        return joined[observed].sort_values("study_uid").reset_index(drop=True), outer.iloc[:0]
    if fit_scope != "outer_fold":
        raise ValueError("Unknown training pool policy")
    return train, outer


def _model_inputs(batch: dict, device: torch.device) -> tuple:
    return tuple(
        batch[name].to(device, non_blocking=True)
        for name in ("images", "features", "series_mask", "slice_mask", "positions")
    )


class PredictionInterrupted(Exception):
    """All evaluation ranks agreed to stop; the final training checkpoint exists."""


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    config: TrainConfig,
    stop_requested: list[bool] | None = None,
    output: Path | None = None,
) -> tuple[pd.DataFrame, dict]:
    model.eval()
    identifiers: list[str] = []
    scores, labels, masks = [], [], []
    for batch in loader:
        if stop_requested is not None:
            stopping = torch.tensor(
                int(stop_requested[0] or (output is not None and external_stop_requested(output))),
                device=device,
            )
            if dist.is_initialized():
                dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
            if stopping.item():
                raise PredictionInterrupted()
        with torch.autocast(
            device.type,
            dtype=torch.float16 if config.precision == "fp16" else torch.bfloat16,
            enabled=config.precision != "fp32",
        ):
            logits = model(*_model_inputs(batch, device))
        valid = batch.get("sample_valid", torch.ones(len(batch["study_uid"]), dtype=torch.bool))
        scores.append(logits.float().sigmoid().cpu().numpy()[valid.numpy()])
        labels.append(batch["labels"].numpy()[valid.numpy()])
        masks.append(batch["label_mask"].numpy()[valid.numpy()])
        identifiers.extend(uid for uid, keep in zip(batch["study_uid"], valid.tolist()) if keep)
    probabilities = np.concatenate(scores)
    frame = pd.DataFrame(probabilities, columns=TARGETS)
    frame.insert(0, "seed", config.seed)
    frame.insert(0, "fold", config.fold)
    frame.insert(0, "study_uid", identifiers)
    return frame, auc_metrics(np.concatenate(labels), probabilities, np.concatenate(masks))


def run(args: argparse.Namespace, stop_requested: list[bool]) -> int:
    # Imports are deliberately local: --help/config/loss tests do not load optional backbones.
    from rsna.competitive.backbones import create_encoder
    from rsna.competitive.data import CompetitiveStudyDataset
    from rsna.competitive.folds import load_folds
    from rsna.competitive.model import CompetitiveStudyNet

    config = TrainConfig(**json.loads(args.config.read_text()))
    config.validate()
    rank, world, local_rank = (
        int(os.environ.get(name, default))
        for name, default in (("RANK", "0"), ("WORLD_SIZE", "1"), ("LOCAL_RANK", "0"))
    )
    if world < 1 or not 0 <= rank < world or local_rank < 0:
        raise ValueError("Invalid distributed environment")
    device = torch.device(f"cuda:{local_rank}" if args.device == "cuda" else "cpu")
    if device.type == "cuda":
        if not torch.cuda.is_available() or local_rank >= torch.cuda.device_count():
            raise ValueError("CUDA was requested but no assigned CUDA device is available")
        torch.cuda.set_device(device)
        if config.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise ValueError("BF16 is unsupported on the assigned GPU")
    elif config.precision != "fp32":
        raise ValueError("CPU acceptance uses FP32; BF16 requires an actual GPU pilot")
    if world > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    try:
        with run_lock(args.output) if rank == 0 else contextlib.nullcontext():
            return _run_locked(
                args,
                config,
                stop_requested,
                rank,
                world,
                device,
                create_encoder,
                CompetitiveStudyDataset,
                CompetitiveStudyNet,
                load_folds,
            )
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run_locked(
    args,
    config,
    stop_requested,
    rank,
    world,
    device,
    create_encoder,
    dataset_class,
    model_class,
    load_folds,
) -> int:
    torch.set_num_threads(2)
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if (
        rank == 0
        and not args.resume
        and any(
            (args.output / name).exists()
            for name in ("checkpoint.pt", "run.json", "outer_predictions.csv")
        )
    ):
        raise ValueError(
            "Output already contains a run; resume explicitly or choose a new directory"
        )
    manifest, release_binding = verified_job_release(args.studies, args.artifact_root, rank, world)
    folds = load_folds(args.folds, manifest, release_binding)
    train, outer = select_development(manifest, folds, config.fold, config.fit_scope)
    supervision_path = getattr(args, "fitting_supervision", None)
    if (
        config.fitting_supervision is not None
        and getattr(args, "mrnet_checkpoint", None) is not None
    ):
        raise ValueError("First fitting-supervision contrast cannot mix external initialization")
    if config.fitting_supervision is not None or supervision_path is not None:
        train, supervision_receipt = apply_fitting_supervision(
            train,
            outer,
            supervision_path,
            config.fitting_supervision,
            release_sha256=release_binding["manifest_sha256"],
            folds_sha256=file_sha256(
                (args.folds if args.folds.is_dir() else args.folds.parent) / "fold.csv"
            ),
            fold=config.fold,
        )
    else:
        supervision_receipt = None
    canonical_sides_path = getattr(args, "canonical_sides", None)
    if config.canonical_sides is not None or canonical_sides_path is not None:
        if config.canonical_sides is None or canonical_sides_path is None:
            raise ValueError("Canonical-side contrast needs both policy and artifact path")
        side_frame, side_receipt = load_canonical_side_table(
            train,
            outer,
            canonical_sides_path,
            config.canonical_sides["sha256"],
            release_sha256=release_binding["manifest_sha256"],
            folds_sha256=file_sha256(
                (args.folds if args.folds.is_dir() else args.folds.parent) / "fold.csv"
            ),
            fold=config.fold,
        )
        side_receipt["enabled"] = config.canonical_sides["enabled"]
        data_side_frame = side_frame if config.canonical_sides["enabled"] else None
    else:
        side_receipt = None
        data_side_frame = None
    if config.diagnostic_study_limit is not None:
        train, outer = (
            train.head(config.diagnostic_study_limit),
            outer.head(config.diagnostic_study_limit),
        )
    series_bytes = args.series.read_bytes()
    if hashlib.sha256(series_bytes).hexdigest() != release_binding["series_sha256"]:
        raise ValueError("Series sampling table differs from the accepted data release")
    series = pd.read_csv(io.BytesIO(series_bytes))
    from rsna.competitive.data import collate_studies, evaluation_slice_sampling

    data_config = {
        name: getattr(config, name)
        for name in ("slices", "side", "max_series", "slice_sampling", "series_sampling")
    }
    if config.cache_contrast is not None:
        data_config["cache_contrast"] = config.cache_contrast
    if config.neighbor_channels is not None:
        data_config["neighbor_channels"] = config.neighbor_channels
    if data_side_frame is not None:
        data_config["sides"] = data_side_frame
    training_config = dict(data_config)
    if (
        config.fitting_supervision is not None
        and config.fitting_supervision["kind"] == "label_source_v1"
    ):
        training_config["soft_targets"] = True
    training_data = dataset_class(
        train, series, args.cache, augment=config.augment, **training_config
    )
    evaluation_config = dict(
        data_config, slice_sampling=evaluation_slice_sampling(config.slice_sampling)
    )
    outer_data = (
        dataset_class(outer, series, args.cache, **evaluation_config) if len(outer) else None
    )
    if training_data.source_revision != release_binding["source_revision"] or (
        outer_data is not None
        and (training_data.cache_side, training_data.source_revision)
        != (outer_data.cache_side, outer_data.source_revision)
    ):
        raise ValueError("Image cache transform/source differs from the accepted release")
    channels = neighbor_input_channels(config.neighbor_channels)
    encoder, embedding_dim, encoder_provenance = create_encoder(
        config.encoder, **({"input_channels": channels} if channels == 3 else {})
    )
    configure_encoder_training(encoder, config.freeze_encoder)
    model = model_class(
        encoder,
        embedding_dim,
        pooling=config.pooling,
        hidden_dim=config.hidden_dim,
        encode_chunk_size=config.encode_chunk_size,
        slice_context=config.slice_context,
        **({"witness": config.witness} if config.witness is not None else {}),
        **({"plane_memory": config.plane_memory} if config.plane_memory is not None else {}),
        **({"patch_summary": config.patch_summary} if config.patch_summary is not None else {}),
        **relation_model_kwargs(config.relation),
        **({"input_channels": channels} if channels == 3 else {}),
    ).to(device)
    delta_guard = (
        FrozenBaseDelta(model, encoder_provenance)
        if config.checkpoint_format == "frozen_base_delta"
        else None
    )
    external_initialization = initialize_private_mrnet(args, model, delta_guard)
    encoder_parameters = {id(parameter) for parameter in encoder.parameters()}
    groups = [
        {
            "params": [
                p for p in model.parameters() if p.requires_grad and id(p) not in encoder_parameters
            ],
            "lr": config.head_lr,
        }
    ]
    if any(p.requires_grad for p in encoder.parameters()):
        groups.append(
            {
                "params": [p for p in encoder.parameters() if p.requires_grad],
                "lr": config.encoder_lr,
            }
        )
    optimizer = torch.optim.AdamW(groups, weight_decay=config.weight_decay)
    epoch_updates = EffectiveBatchSampler(
        len(training_data),
        config.batch_size,
        config.accumulation_steps,
        seed=config.seed,
        epoch=0,
        rank=rank,
        world_size=world,
    ).updates
    total_updates = epoch_updates * config.epochs
    wrapped = (
        DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)
        if world > 1
        else model
    )
    # Bind numerical configuration and content, never machine-specific paths.
    bound_config = asdict(config)
    if config.precision != "fp16":
        bound_config.pop("amp_init_scale")
    if config.relation is None:
        # Preserve old policy serialization; code provenance still changes and
        # exact older checkpoints must use their original pinned source.
        bound_config.pop("relation")
    if config.plane_memory is None:
        bound_config.pop("plane_memory")
    if config.witness is None:
        bound_config.pop("witness")
    if config.fitting_supervision is None:
        bound_config.pop("fitting_supervision")
    if config.neighbor_channels is None:
        bound_config.pop("neighbor_channels")
    if config.canonical_sides is None:
        bound_config.pop("canonical_sides")
    if config.patch_summary is None:
        bound_config.pop("patch_summary")
    if config.cache_contrast is None:
        bound_config.pop("cache_contrast")
    bound_config["encoder"] = encoder_provenance
    package_root = Path(__file__).resolve().parents[1]
    code_hashes = {
        str(path.relative_to(package_root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(package_root.rglob("*.py"))
    }
    binding = {
        "config": bound_config,
        "release": release_binding,
        "folds_sha256": _hash(folds.sort_values("study_uid").to_dict("records")),
        "training_uids_sha256": _uid_hash(train),
        "outer_uids_sha256": _uid_hash(outer),
        "world_size": world,
        "device_type": device.type,
        "torch_version": str(torch.__version__),
        "targets": list(TARGETS),
        "code_sha256": code_hashes,
        "cache_side": training_data.cache_side,
        "cache_version": CACHE_VERSION,
        "source_revision": training_data.source_revision,
    }
    if external_initialization is not None:
        binding.update(
            external_initialization=external_initialization,
            private_research_only=True,
            external_weight_publication=False,
        )
    if supervision_receipt is not None:
        binding["experimental_fitting_supervision"] = supervision_receipt
    if side_receipt is not None:
        binding["experimental_canonical_sides"] = side_receipt
    if world > 1:
        rank_bindings: list = [None] * world
        dist.all_gather_object(rank_bindings, _hash(binding))
        if len(set(rank_bindings)) != 1:
            raise ValueError("Distributed ranks disagree on source, settings, weights or code")
    epoch, next_update, step = 0, 0, 0
    scaler = (
        torch.amp.GradScaler(device.type, init_scale=config.amp_init_scale)
        if config.precision == "fp16"
        else None
    )
    if args.resume:
        checkpoint = load_checkpoint(args.resume, binding)
        restore_checkpoint_model(model, checkpoint, delta_guard)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scaler is not None:
            scaler.load_state_dict(checkpoint["amp_scaler"])
        epoch, next_update, step = (
            checkpoint["epoch"],
            checkpoint["next_update"],
            checkpoint["step"],
        )
        if (
            any(type(value) is not int or value < 0 for value in (epoch, next_update, step))
            or epoch > config.epochs
            or (epoch == config.epochs and next_update != 0)
        ):
            raise ValueError("Checkpoint progress counters are invalid")
        restore_rng(checkpoint["rng_by_rank"][rank], device)
    teacher = None
    if config.masked_cell is not None and config.masked_cell["kind"] == "ema_consistency":
        # Mean-teacher: an EMA shadow of the student, initialized from the
        # (possibly resumed) weights. The teacher is checkpointed with the
        # rest of the training state so a mid-epoch resume continues the
        # exact same EMA trajectory instead of restarting from the restored
        # student.
        teacher = copy.deepcopy(model).requires_grad_(False)
        teacher.eval()
        if args.resume and checkpoint.get("teacher") is not None:
            teacher.load_state_dict(checkpoint["teacher"])
    started = time.monotonic()
    updates_this_invocation = 0
    history = []

    def save(current_epoch: int, current_update: int, completed_step: int) -> None:
        state = capture_rng(device)
        rng_states: list = [None] * world
        if world > 1:
            dist.all_gather_object(rng_states, state)
        else:
            rng_states[0] = state
        if rank == 0:
            save_checkpoint(
                args.output / "checkpoint.pt",
                {
                    **(
                        delta_guard.checkpoint_model(model)
                        if delta_guard is not None
                        else {"schema_version": 1, "model": model.state_dict()}
                    ),
                    "binding": binding,
                    "optimizer_boundary": True,
                    "optimizer": optimizer.state_dict(),
                    **({"amp_scaler": scaler.state_dict()} if scaler is not None else {}),
                    "epoch": current_epoch,
                    "next_update": current_update,
                    "step": completed_step,
                    "rng_by_rank": rng_states,
                    **({"teacher": teacher.state_dict()} if teacher is not None else {}),
                    "scheduled_epoch": scheduled_epoch(current_epoch, current_update, bound_config),
                },
                min_free_gib=config.min_free_gib,
            )
            if scheduled_epoch(current_epoch, current_update, bound_config):
                write_epoch_snapshot(
                    args.output / "checkpoint.pt",
                    args.output / "epoch-checkpoints",
                    binding,
                    current_epoch,
                    completed_step,
                )
        if world > 1:
            dist.barrier()

    if rank == 0:
        atomic_json(
            args.output / "run.json",
            {
                "schema_version": 1,
                "status": "running",
                "binding": binding,
                "diagnostic_run": config.diagnostic_study_limit is not None,
                "training_studies": len(train),
                "outer_studies": len(outer),
                "trainable_parameters": sum(
                    p.numel() for p in model.parameters() if p.requires_grad
                ),
                "encoder_trainable_parameters": sum(
                    p.numel() for p in encoder.parameters() if p.requires_grad
                ),
                "selection_policy": "fixed_epoch_count; no_internal_evaluation"
                if config.fit_scope == "all_development"
                else "fixed_epoch_count; outer_predictions_only_after_final_epoch",
                "held_out_confirmation_and_gold_used": False,
            },
        )
    while epoch < config.epochs:
        sampler = EffectiveBatchSampler(
            len(training_data),
            config.batch_size,
            config.accumulation_steps,
            seed=config.seed,
            epoch=epoch,
            rank=rank,
            world_size=world,
            start_update=next_update,
        )
        loader = DataLoader(
            SeededDataset(training_data),
            batch_sampler=sampler,
            num_workers=config.workers,
            pin_memory=device.type == "cuda",
            generator=torch.Generator().manual_seed(config.seed + epoch + rank),
            collate_fn=collate_studies,
        )
        iterator = iter(loader)
        wrapped.train()
        if not any(p.requires_grad for p in encoder.parameters()):
            encoder.eval()
        for update in range(next_update, sampler.updates):
            batches = [next(iterator) for _ in range(config.accumulation_steps)]
            masks = [batch["label_mask"].to(device) for batch in batches]
            counts = observed_counts(masks)
            masked_enabled = config.masked_cell is not None and config.masked_cell["enabled"]
            masked_counts = (
                observed_counts([(mask == 0).to(mask.dtype) for mask in masks])
                if masked_enabled
                else None
            )
            smooth_enabled = config.slice_smooth is not None and config.slice_smooth["enabled"]
            pair_count = (
                adjacent_slice_pairs([batch["slice_mask"].to(device) for batch in batches])
                if smooth_enabled
                else None
            )
            relation_batches = (
                [
                    relations_for_batch(batch, config.relation, seed=config.seed, epoch=epoch)
                    for batch in batches
                ]
                if config.relation is not None
                else []
            )
            relation_count = (
                relation_study_count(relation_batches, device)
                if config.relation is not None
                else torch.zeros((), device=device)
            )
            optimizer.zero_grad(set_to_none=True)
            loss_total = torch.zeros((), device=device)
            classification_total = torch.zeros((), device=device)
            relation_total = torch.zeros((), device=device)
            masked_total = torch.zeros((), device=device)
            smooth_total = torch.zeros((), device=device)
            if (
                counts.any()
                or (
                    config.relation is not None
                    and config.relation["weight"] > 0
                    and relation_count > 0
                )
                or (masked_counts is not None and masked_counts.any())
                or (pair_count is not None and pair_count > 0)
            ):
                for micro, (batch, mask) in enumerate(zip(batches, masks)):
                    synchronize = micro == len(batches) - 1 or world == 1
                    with contextlib.nullcontext() if synchronize else wrapped.no_sync():
                        with torch.autocast(
                            device.type,
                            dtype=torch.float16 if config.precision == "fp16" else torch.bfloat16,
                            enabled=config.precision != "fp32",
                        ):
                            if config.relation is None:
                                if smooth_enabled:
                                    logits, slice_scores = wrapped(
                                        *_model_inputs(batch, device),
                                        return_slice_scores=True,
                                    )
                                else:
                                    logits = wrapped(*_model_inputs(batch, device))
                                    slice_scores = None
                                auxiliary = torch.zeros((), device=device)
                            else:
                                logits, relation_sum = wrapped(
                                    *_model_inputs(batch, device), relations=relation_batches[micro]
                                )
                                auxiliary = effective_relation_loss(relation_sum, relation_count)
                            classification = effective_batch_loss(
                                logits, batch["labels"].to(device), mask, counts
                            )
                            loss = (
                                classification
                                if config.relation is None
                                else classification + config.relation["weight"] * auxiliary
                            )
                            if masked_enabled:
                                assert masked_counts is not None
                                if config.masked_cell["kind"] == "entropy_max":
                                    # Entropy maximization (ESA arm): subtracting
                                    # the masked-cell entropy rewards uncertain
                                    # predictions where the labels observe nothing.
                                    masked_entropy = masked_cell_entropy(
                                        logits, mask, masked_counts
                                    )
                                    loss = loss - config.masked_cell["weight"] * masked_entropy
                                    masked_total += masked_entropy.detach()
                                else:
                                    # Mean-teacher consistency: pull masked-cell
                                    # logits toward the EMA shadow's prediction,
                                    # ramped from zero over the warmup fraction.
                                    assert teacher is not None
                                    with torch.no_grad():
                                        teacher_logits = teacher(*_model_inputs(batch, device))
                                    consistency = masked_cell_consistency(
                                        logits, teacher_logits, mask, masked_counts
                                    )
                                    ramp_value = min(
                                        1.0,
                                        step
                                        / max(
                                            1,
                                            math.ceil(
                                                config.masked_cell["ramp_fraction"] * total_updates
                                            ),
                                        ),
                                    )
                                    loss = loss + (
                                        ramp_value * config.masked_cell["weight"] * consistency
                                    )
                                    masked_total += consistency.detach()
                            if smooth_enabled:
                                assert slice_scores is not None and pair_count is not None
                                # SmMIL smoothness: physically adjacent slices
                                # should carry similar attention scores — a
                                # lesion occupies a contiguous run, not
                                # scattered singletons.
                                tv = slice_score_tv(
                                    slice_scores,
                                    batch["slice_mask"].to(device),
                                    pair_count,
                                )
                                loss = loss + config.slice_smooth["weight"] * tv
                                smooth_total += tv.detach()
                        if scaler is None:
                            loss.backward()
                        else:
                            scaler.scale(loss).backward()
                    loss_total += loss.detach()
                    classification_total += classification.detach()
                    relation_total += auxiliary.detach()
                if scaler is not None:
                    scaler.unscale_(optimizer)
                norm = nn.utils.clip_grad_norm_(
                    model.parameters(), config.clip_grad_norm, error_if_nonfinite=True
                )
                multiplier = learning_rate_multiplier(
                    step, total_updates, config.warmup_updates, config.lr_schedule
                )
                for group, base_lr in zip(
                    optimizer.param_groups, [config.head_lr, config.encoder_lr]
                ):
                    group["lr"] = base_lr * multiplier
                if scaler is None:
                    optimizer.step()
                else:
                    # Nonfinite unscaled gradients fail at clipping above;
                    # skipped AMP steps never masquerade as optimizer progress.
                    scaler.step(optimizer)
                    scaler.update()
                if teacher is not None:
                    with torch.no_grad():
                        for t_param, s_param in zip(teacher.parameters(), model.parameters()):
                            t_param.mul_(config.masked_cell["alpha"]).add_(
                                s_param.detach(), alpha=1 - config.masked_cell["alpha"]
                            )
                        for t_buf, s_buf in zip(teacher.buffers(), model.buffers()):
                            t_buf.copy_(s_buf)
                step += 1
            else:
                norm = torch.tensor(0.0)
            if world > 1:
                dist.all_reduce(loss_total)
                loss_total /= world
                if config.relation is not None:
                    dist.all_reduce(classification_total)
                    classification_total /= world
                    dist.all_reduce(relation_total)
                    relation_total /= world
                if masked_enabled:
                    dist.all_reduce(masked_total)
                    masked_total /= world
            updates_this_invocation += 1
            epoch_finished = update + 1 == sampler.updates
            checkpoint_epoch, checkpoint_update = (
                (epoch + 1, 0) if epoch_finished else (epoch, update + 1)
            )
            stopping = torch.tensor(
                int(
                    stop_requested[0]
                    or external_stop_requested(args.output)
                    or (
                        args.stop_after_updates is not None
                        and updates_this_invocation >= args.stop_after_updates
                    )
                ),
                device=device,
            )
            if world > 1:
                dist.all_reduce(stopping, op=dist.ReduceOp.MAX)
            if (
                stopping.item()
                or epoch_finished
                or updates_this_invocation % config.checkpoint_every_updates == 0
            ):
                save(checkpoint_epoch, checkpoint_update, step)
            if rank == 0:
                event = {
                    "epoch": epoch,
                    "update": update + 1,
                    "optimizer_steps": step,
                    "loss": float(loss_total),
                    "gradient_norm": float(norm),
                    "observed_cells": float(counts.sum()),
                    "elapsed_seconds": time.monotonic() - started,
                }
                if "adapter" in config.encoder and step == 1:
                    gradients = [p.grad for p in encoder.parameters() if p.requires_grad]
                    if any(gradient is None for gradient in gradients):
                        raise ValueError("An intended adapter parameter has no gradient")
                    encoder_norm = torch.linalg.vector_norm(
                        torch.stack(
                            [
                                gradient.float().norm()
                                for gradient in gradients
                                if gradient is not None
                            ]
                        )
                    )
                    if not torch.isfinite(encoder_norm) or encoder_norm <= 0:
                        raise ValueError("Adapter gradients are zero or nonfinite")
                    event["encoder_gradient_norm"] = float(encoder_norm)
                if config.relation is not None:
                    event.update(
                        classification_loss=float(classification_total),
                        relation_loss=float(relation_total),
                        relation_studies=float(relation_count),
                        relation_weight=config.relation["weight"],
                    )
                if masked_enabled:
                    assert masked_counts is not None
                    event.update(
                        masked_cells=float(masked_counts.sum()),
                        masked_weight=config.masked_cell["weight"],
                    )
                    if config.masked_cell["kind"] == "entropy_max":
                        event["masked_entropy"] = float(masked_total)
                    else:
                        event["masked_consistency"] = float(masked_total)
                if smooth_enabled:
                    assert pair_count is not None
                    event.update(
                        slice_pairs=float(pair_count),
                        slice_smooth=float(smooth_total),
                        slice_smooth_weight=config.slice_smooth["weight"],
                    )
                if device.type == "cuda":
                    event["peak_gpu_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
                history.append(event)
                print(json.dumps(event, allow_nan=False), flush=True)
            if stopping.item():
                if rank == 0:
                    atomic_json(
                        args.output / "invocation.json",
                        {
                            "status": "checkpointed_for_resume",
                            "exit_code": 75,
                            "updates": history,
                            "binding_sha256": _hash(binding),
                            "checkpoint_sha256": file_sha256(args.output / "checkpoint.pt"),
                        },
                    )
                return 75
        epoch += 1
        next_update = 0
    if config.fit_scope == "all_development":
        if rank == 0:
            atomic_json(
                args.output / "result.json",
                {
                    "schema_version": 1,
                    "status": "completed",
                    "binding_sha256": _hash(binding),
                    "checkpoint_sha256": file_sha256(args.output / "checkpoint.pt"),
                    "diagnostic_run": config.diagnostic_study_limit is not None,
                    "fit_scope": config.fit_scope,
                    "outer_fold": None,
                    "seed": config.seed,
                    "optimizer_steps": step,
                    "training_studies": len(train),
                    "outer_studies": 0,
                    "evaluation_source": None,
                    "checkpoint_selection": "fixed_final_epoch_no_internal_selection",
                    "metrics": None,
                    "predictions_sha256": None,
                    "elapsed_seconds": time.monotonic() - started,
                },
            )
            atomic_json(
                args.output / "invocation.json",
                {
                    "status": "completed",
                    "exit_code": 0,
                    "updates": history,
                    "binding_sha256": _hash(binding),
                },
            )
            run_record = json.loads((args.output / "run.json").read_text())
            run_record["status"] = "completed"
            atomic_json(args.output / "run.json", run_record)
        if world > 1:
            dist.barrier()
        return 0
    assert outer_data is not None
    # Equal-length rank schedules allow coordinated cancellation even in the
    # final partial evaluation batch. Padding produces no exported prediction.
    outer_sampler = EffectiveBatchSampler(
        len(outer_data),
        config.batch_size,
        1,
        seed=config.seed,
        epoch=0,
        rank=rank,
        world_size=world,
    )
    outer_loader = DataLoader(
        SeededDataset(outer_data),
        batch_sampler=outer_sampler,
        num_workers=config.workers,
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=collate_studies,
    )
    try:
        local_predictions, _ = predict(
            model, outer_loader, device, config, stop_requested, args.output
        )
    except PredictionInterrupted:
        if rank == 0:
            atomic_json(
                args.output / "invocation.json",
                {
                    "status": "final_inference_interrupted_resume_final_checkpoint",
                    "exit_code": 75,
                    "binding_sha256": _hash(binding),
                    "checkpoint_sha256": file_sha256(args.output / "checkpoint.pt"),
                },
            )
        return 75
    prediction_parts: list = [None] * world
    if world > 1:
        dist.all_gather_object(prediction_parts, local_predictions)
    else:
        prediction_parts[0] = local_predictions
    if rank == 0:
        predictions = (
            pd.concat(prediction_parts, ignore_index=True)
            .sort_values("study_uid")
            .reset_index(drop=True)
        )
        if (
            set(predictions.study_uid) != set(outer.study_uid)
            or predictions.study_uid.duplicated().any()
        ):
            raise ValueError("Outer predictions do not cover the exact declared evaluation pool")
        actual = outer.set_index("study_uid").loc[predictions.study_uid]
        metrics = auc_metrics(
            actual[list(TARGETS)].to_numpy(),
            predictions[list(TARGETS)].to_numpy(),
            actual[[f"{target}__mask" for target in TARGETS]].to_numpy(),
        )
        temporary = args.output / "outer_predictions.csv.partial"
        predictions.to_csv(temporary, index=False)
        os.replace(temporary, args.output / "outer_predictions.csv")
        atomic_json(
            args.output / "result.json",
            {
                "schema_version": 1,
                "status": "completed",
                "binding_sha256": _hash(binding),
                "checkpoint_sha256": file_sha256(args.output / "checkpoint.pt"),
                "diagnostic_run": config.diagnostic_study_limit is not None,
                "outer_fold": config.fold,
                "seed": config.seed,
                "optimizer_steps": step,
                "training_studies": len(train),
                "outer_studies": len(outer),
                "evaluation_source": "accepted_train_outer_fold_report_supervision",
                "checkpoint_selection": "fixed_final_epoch_no_outer_selection",
                "metrics": metrics,
                "elapsed_seconds": time.monotonic() - started,
                "cuda_peak_memory_bytes": torch.cuda.max_memory_allocated(device)
                if device.type == "cuda"
                else None,
                "predictions_sha256": hashlib.sha256(
                    (args.output / "outer_predictions.csv").read_bytes()
                ).hexdigest(),
            },
        )
        atomic_json(
            args.output / "invocation.json",
            {
                "status": "completed",
                "exit_code": 0,
                "updates": history,
                "binding_sha256": _hash(binding),
            },
        )
        run_record = json.loads((args.output / "run.json").read_text())
        run_record["status"] = "completed"
        atomic_json(args.output / "run.json", run_record)
    if world > 1:
        dist.barrier()
    return 0


def main() -> None:
    stop_requested = [False]
    signal.signal(signal.SIGUSR1, lambda *_: stop_requested.__setitem__(0, True))
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("config", "studies", "series", "cache", "folds", "artifact-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument(
        "--fitting-supervision", type=Path, help="Exact private reviewed fitting-overlay artifact"
    )
    parser.add_argument("--canonical-sides", type=Path, help="Exact canonical-side table artifact")
    parser.add_argument(
        "--mrnet-checkpoint", type=Path, help="Private final source adapter initialization"
    )
    parser.add_argument("--mrnet-checkpoint-sha256")
    parser.add_argument("--mrnet-terminal", type=Path)
    parser.add_argument("--mrnet-terminal-sha256")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--stop-after-updates",
        type=int,
        help="Diagnostic interruption at an optimizer boundary; exits 75",
    )
    args = parser.parse_args()
    if args.stop_after_updates is not None and args.stop_after_updates < 1:
        parser.error("--stop-after-updates must be positive")
    raise SystemExit(run(args, stop_requested))


if __name__ == "__main__":
    main()
