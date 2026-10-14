"""Bounded engineering throughput profile of the canonical training update.

Engineering evidence only: no accuracy, production acceptance or checkpoint is
produced. Importing this module touches no CUDA device, environment secret or file.
Only loader/microbatch/chunk settings may differ from the bound recipe; the global
effective batch, loss masks/normalization, sampling, optimizer, LR schedule and
precision stay identical. Batch/chunk changes alter floating-point reduction order
and adapter/head dropout draws, so they are never claimed to be bit-identical.

Admitted recipes: the plain native-cache canonical classifier, the public ordered-
triplet input (``public_input``) with or without the source-slot / window-bag
``classifier_strategy``, and the ``label_source_v1`` fitting supervision (enabled or
control). Every other recipe component is refused.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import math
import os
import random
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from rsna.competitive.adapters import configure_encoder_training
from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.delta_checkpoint import FrozenBaseDelta
from rsna.competitive.loss import effective_batch_loss, observed_counts
from rsna.competitive.public_input import verify_public_cohort
from rsna.competitive.sampling import EffectiveBatchSampler, SeededDataset
from rsna.competitive.source_slot import training_inputs as source_slot_training_inputs
from rsna.competitive.supervision import apply_fitting_supervision
from rsna.competitive.train import (
    TrainConfig,
    _hash,
    _model_inputs,
    _uid_hash,
    learning_rate_multiplier,
    select_development,
    verified_job_release,
)
from rsna.competitive.window_bag import KIND as WINDOW_BAG_KIND
from rsna.competitive.window_bag import TRAINING_VIEWS as WINDOW_BAG_VIEWS
from rsna.competitive.window_bag import training_inputs as window_bag_training_inputs

SCHEMA_VERSION = 1
CANONICAL_EFFECTIVE_STUDIES = 8
THROUGHPUT_FIELDS = ("batch_size", "accumulation_steps", "encode_chunk_size", "workers")
MAX_WORKERS = 2
HARD_MAX_MEASURED_UPDATES = 12
HARD_MAX_WARMUP_UPDATES = 4
HARD_MAX_WALL_SECONDS = 7200.0
PHASES = (
    "data_wait",
    "cpu_transform",
    "host_to_device",
    "forward",
    "encoder_forward",
    "backward",
    "encoder_backward",
    "optimizer",
)

NATIVE_FORMAT = "native_cache_npz"
PUBLIC_FORMAT = "public_ordered_triplet_npz"
MRNET_FIELDS = (
    "mrnet_checkpoint",
    "mrnet_checkpoint_sha256",
    "mrnet_terminal",
    "mrnet_terminal_sha256",
)

BIT_IDENTICAL = "bit_identical_expected"
NOT_BIT_IDENTICAL = "statistically_equivalent_not_bit_identical"

# Candidate overrides; everything else is inherited from the bound recipe config.
CANDIDATE_SETTINGS: dict[str, dict[str, int]] = {
    "mb1-acc8-chunk8-w0": dict(batch_size=1, accumulation_steps=8, encode_chunk_size=8, workers=0),
    "mb1-acc8-chunk8-w1": dict(batch_size=1, accumulation_steps=8, encode_chunk_size=8, workers=1),
    "mb1-acc8-chunk8-w2": dict(batch_size=1, accumulation_steps=8, encode_chunk_size=8, workers=2),
    "mb1-acc8-chunk32-w0": dict(
        batch_size=1, accumulation_steps=8, encode_chunk_size=32, workers=0
    ),
    "mb4-acc2-chunk32-w0": dict(
        batch_size=4, accumulation_steps=2, encode_chunk_size=32, workers=0
    ),
    "mb4-acc2-chunk32-w1": dict(
        batch_size=4, accumulation_steps=2, encode_chunk_size=32, workers=1
    ),
}


def recipe_differences(base: TrainConfig, other: TrainConfig) -> dict[str, list]:
    first, second = dataclasses.asdict(base), dataclasses.asdict(other)
    return {name: [first[name], second[name]] for name in first if first[name] != second[name]}


def apply_setting(base: TrainConfig, overrides: dict[str, int]) -> TrainConfig:
    """Return the recipe with only throughput fields changed and effective batch kept."""
    if not isinstance(overrides, dict) or set(overrides) - set(THROUGHPUT_FIELDS):
        raise ValueError("Only loader, microbatch, accumulation and chunk settings may change")
    if any(type(value) is not int or value < 0 for value in overrides.values()):
        raise ValueError("Throughput settings must be nonnegative integers")
    candidate = dataclasses.replace(
        base,
        batch_size=overrides.get("batch_size", base.batch_size),
        accumulation_steps=overrides.get("accumulation_steps", base.accumulation_steps),
        encode_chunk_size=overrides.get("encode_chunk_size", base.encode_chunk_size),
        workers=overrides.get("workers", base.workers),
    )
    candidate.validate()
    if not 0 <= candidate.workers <= MAX_WORKERS:
        raise ValueError("Minimal CPU prefetch allows at most two loader workers")
    effective = base.batch_size * base.accumulation_steps
    if candidate.batch_size * candidate.accumulation_steps != effective:
        raise ValueError("The global effective batch must stay unchanged")
    if set(recipe_differences(base, candidate)) - set(THROUGHPUT_FIELDS):
        raise ValueError("A non-throughput recipe field changed")
    return candidate


def numerical_audit(base: TrainConfig, candidate: TrainConfig) -> dict[str, Any]:
    """State which exact trajectory relation is expected; never assume bit equality."""
    changed = sorted(recipe_differences(base, candidate))
    if set(changed) - set(THROUGHPUT_FIELDS):
        raise ValueError("Audit only covers throughput settings")
    reasons = []
    if "batch_size" in changed or "accumulation_steps" in changed:
        reasons.append(
            "Per-update study membership/order and observed-count normalization are unchanged, "
            "but microbatch sums, padded batched kernels and FP16 gradient accumulation change "
            "floating-point reduction order."
        )
    if "encode_chunk_size" in changed or "batch_size" in changed:
        dropout = (
            base.encoder.get("adapter", {}).get("dropout", 0)
            if isinstance(base.encoder, dict)
            else 0
        )
        reasons.append(
            "Encoder chunks change GEMM shapes/kernels"
            + (
                f" and adapter dropout ({dropout}) draws different CUDA RNG masks per view"
                if dropout
                else ""
            )
            + "."
        )
    strategy = base.classifier_strategy
    if strategy is not None:
        reasons.append(
            "Source window groups derive from (seed, epoch, update) and augmentation from "
            "per-study CPU generators in the main process, so window/augmentation draws are "
            "unchanged by microbatch, chunk or loader workers."
        )
        if "batch_size" in changed:
            reasons.append(
                f"The slot head's dropout ({strategy['dropout']}) draws its device RNG mask "
                "per microbatch call, so microbatch changes alter head dropout masks."
            )
    if changed == ["workers"] or not changed:
        expectation = BIT_IDENTICAL
        reasons.append(
            "Items are seeded per study by SeededDataset and the loader generator is separate "
            "from the training RNG, so only CPU scheduling changes. Requires actual evidence."
        )
    else:
        expectation = NOT_BIT_IDENTICAL
    return {
        "changed_fields": changed,
        "expectation": expectation,
        "reasons": reasons,
        "effective_studies_per_update": candidate.batch_size * candidate.accumulation_steps,
    }


def validate_caps(measured_updates: int, warmup_updates: int, max_wall_seconds: float) -> dict:
    if type(measured_updates) is not int or not 1 <= measured_updates <= HARD_MAX_MEASURED_UPDATES:
        raise ValueError("Measured updates must be within the hard profiling cap")
    if type(warmup_updates) is not int or not 1 <= warmup_updates <= HARD_MAX_WARMUP_UPDATES:
        raise ValueError("At least one, and a bounded number of, warmup updates is required")
    if (
        isinstance(max_wall_seconds, bool)
        or not isinstance(max_wall_seconds, (int, float))
        or not math.isfinite(max_wall_seconds)
        or not 0 < max_wall_seconds <= HARD_MAX_WALL_SECONDS
    ):
        raise ValueError("Wall-time cap must be positive and bounded")
    return {
        "measured_updates": measured_updates,
        "warmup_updates": warmup_updates,
        "max_wall_seconds": float(max_wall_seconds),
        "max_studies": (measured_updates + warmup_updates) * CANONICAL_EFFECTIVE_STUDIES,
    }


def code_tree_sha256(package_root: Path | None = None) -> dict[str, str]:
    root = package_root or Path(__file__).resolve().parents[1]
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*.py"))
    }


class TimedItems(Dataset):
    """Record each item's whole read/verify/decompress/transform time, in any process."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)  # type: ignore[arg-type]

    def __getitem__(self, index: int) -> dict:
        started = time.perf_counter()
        item = dict(self.dataset[index])
        item["profile_item_seconds"] = torch.tensor(time.perf_counter() - started)
        return item


class PhaseClock:
    """Wall timing with device synchronization exactly at phase boundaries."""

    def __init__(self, device: torch.device):
        self.device = device
        self.totals = dict.fromkeys(PHASES, 0.0)

    def sync(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def measure(self, phase: str, function: Callable[[], Any]) -> Any:
        if phase not in PHASES:
            raise ValueError("Unknown profiling phase")
        self.sync()
        started = time.perf_counter()
        result = function()
        self.sync()
        self.totals[phase] += time.perf_counter() - started
        return result


class EncoderTimer:
    """Encoder forward (per chunk) and backward (first encoder gradient to end) timing.

    CUDA events avoid adding synchronization inside the measured forward/backward.
    Gradient-checkpoint recomputation happens inside the encoder backward interval.
    """

    def __init__(self, encoder: nn.Module, device: torch.device):
        self.device = device
        self.forward_pairs: list[tuple[Any, Any]] = []
        self.first_gradient: Any = None
        self.encoded_images = 0
        self._pending: Any = None
        self.handles = [
            encoder.register_forward_pre_hook(self._before),
            encoder.register_forward_hook(self._after),
        ]
        # Slot classifiers call ``forward_features`` directly (no module hooks fire); an
        # instance-level wrapper times it without changing arguments or outputs.
        self._features_owner: nn.Module | None = None
        features = getattr(encoder, "forward_features", None)
        if callable(features) and "forward_features" not in vars(encoder):

            def timed_features(images, *args, **kwargs):
                self._before(encoder, (images,))
                output = features(images, *args, **kwargs)
                self._after(encoder, (images,), output)
                return output

            object.__setattr__(encoder, "forward_features", timed_features)
            self._features_owner = encoder

    def _mark(self) -> Any:
        if self.device.type == "cuda":
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            return event
        return time.perf_counter()

    @staticmethod
    def _elapsed(start: Any, end: Any) -> float:
        if isinstance(start, float):
            return end - start
        return start.elapsed_time(end) / 1000

    def _before(self, module, inputs) -> None:
        if inputs and isinstance(inputs[0], Tensor):
            self.encoded_images += len(inputs[0])
        self._pending = self._mark()

    def _after(self, module, inputs, output) -> None:
        self.forward_pairs.append((self._pending, self._mark()))
        tensors = output if isinstance(output, tuple) else (output,)
        for tensor in tensors:
            if isinstance(tensor, Tensor) and tensor.requires_grad:
                tensor.register_hook(self._gradient)

    def _gradient(self, gradient: Tensor) -> None:
        if self.first_gradient is None:
            self.first_gradient = self._mark()

    def reset(self) -> None:
        self.forward_pairs, self.first_gradient = [], None

    def collect(self, backward_end: Any) -> tuple[float, float]:
        forward = sum(self._elapsed(start, end) for start, end in self.forward_pairs)
        backward = (
            0.0 if self.first_gradient is None else self._elapsed(self.first_gradient, backward_end)
        )
        self.reset()
        return forward, backward

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        if self._features_owner is not None:
            object.__delattr__(self._features_owner, "forward_features")
            self._features_owner = None


def build_profile_state(
    args: Any,
    config: TrainConfig,
    device: torch.device,
    create_encoder: Callable,
    dataset_class: Callable,
    model_class: Callable,
    load_folds: Callable,
) -> dict[str, Any]:
    """Mirror ``train._run_locked``'s single-rank setup/RNG order for admitted recipes.

    Admitted: the plain native-cache classifier, the public ordered-triplet input with
    or without the source-slot/window-bag classifier, and ``label_source_v1`` fitting
    supervision. The evaluation roster is bound (cohort/overlay checks) but never read.
    """
    supervision = config.fitting_supervision
    if (
        any(
            value is not None
            for value in (
                config.relation,
                config.witness,
                config.plane_memory,
                config.neighbor_channels,
                config.patch_summary,
                config.cache_contrast,
            )
        )
        or (supervision is not None and supervision.get("kind") != "label_source_v1")
        or (config.classifier_strategy is not None and config.public_input is None)
        or config.pooling != "attention"
        or config.fit_scope != "outer_fold"
        or any(getattr(args, name, None) is not None for name in MRNET_FIELDS)
    ):
        raise ValueError(
            "The profiler is bound to the plain, public-triplet or source-slot outer-fold "
            "recipes with optional label_source_v1 fitting supervision"
        )
    torch.set_num_threads(2)
    random.seed(config.seed)
    np.random.seed(config.seed % 2**32)
    torch.manual_seed(config.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    startup: dict[str, float] = {}
    mark = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal mark
        now = time.perf_counter()
        startup[name] = now - mark
        mark = now

    manifest, release_binding = verified_job_release(
        args.studies,
        args.artifact_root,
        0,
        1,
        getattr(args, "release_admission_cache", None),
        getattr(args, "release_admission_fallback", False),
    )
    lap("release_admission")
    folds = load_folds(args.folds, manifest, release_binding)
    lap("folds")
    train, outer = select_development(manifest, folds, config.fold, config.fit_scope)
    supervision_path = getattr(args, "fitting_supervision", None)
    supervision_receipt = None
    if supervision is not None or supervision_path is not None:
        train, supervision_receipt = apply_fitting_supervision(
            train,
            outer,
            supervision_path,
            supervision,
            release_sha256=release_binding["manifest_sha256"],
            folds_sha256=file_sha256(
                (args.folds if args.folds.is_dir() else args.folds.parent) / "fold.csv"
            ),
            fold=config.fold,
        )
    public_receipt = (
        None
        if config.public_input is None
        else verify_public_cohort(
            args.cache, config.public_input, train, outer, release_binding["source_revision"]
        )
    )
    lap("supervision_and_cohort")
    if config.diagnostic_study_limit is not None:
        train = train.head(config.diagnostic_study_limit)
    series_bytes = args.series.read_bytes()
    if hashlib.sha256(series_bytes).hexdigest() != release_binding["series_sha256"]:
        raise ValueError("Series sampling table differs from the accepted data release")
    series = pd.read_csv(io.BytesIO(series_bytes))
    lap("series")
    data_config: dict[str, Any] = {
        name: getattr(config, name)
        for name in ("slices", "side", "max_series", "slice_sampling", "series_sampling")
    }
    if config.public_input is not None:
        # As in the trainer, the bound policy owns its source; native caches never substitute.
        from rsna.competitive.public_input import PublicTripletStudyDataset

        dataset_class = PublicTripletStudyDataset
        data_config.update(
            public_input=config.public_input, source_revision=release_binding["source_revision"]
        )
    if supervision is not None:
        data_config["soft_targets"] = True
    training_data = dataset_class(train, series, args.cache, augment=config.augment, **data_config)
    if training_data.source_revision != release_binding["source_revision"]:
        raise ValueError("Image cache source differs from the accepted release")
    if public_receipt is not None and training_data.index_sha256 != public_receipt["index_sha256"]:
        raise ValueError("Public triplet cache index changed during dataset construction")
    lap("dataset_initialization")
    channels = 3 if config.public_input is not None else 1
    encoder, embedding_dim, encoder_provenance = create_encoder(
        config.encoder,
        **(
            {"input_channels": 3, "input_contract": "public_ordered_triplet"}
            if config.public_input is not None
            else {}
        ),
    )
    lap("encoder_initialization")
    configure_encoder_training(encoder, config.freeze_encoder)
    model: nn.Module
    if config.classifier_strategy is not None:
        from rsna.competitive.source_slot import SourceSlotStudyNet
        from rsna.competitive.window_bag import WindowBagSlotStudyNet

        model = (
            WindowBagSlotStudyNet
            if config.classifier_strategy["kind"] == WINDOW_BAG_KIND
            else SourceSlotStudyNet
        )(
            encoder,
            embedding_dim,
            config.classifier_strategy,
            encode_chunk_size=config.encode_chunk_size,
            input_channels=channels,
        ).to(device)
    else:
        model = model_class(
            encoder,
            embedding_dim,
            pooling=config.pooling,
            hidden_dim=config.hidden_dim,
            encode_chunk_size=config.encode_chunk_size,
            slice_context=config.slice_context,
            **({"input_channels": channels} if channels == 3 else {}),
        ).to(device)
    delta_guard = (
        FrozenBaseDelta(model, encoder_provenance)
        if config.checkpoint_format == "frozen_base_delta"
        else None
    )
    encoder_ids = {id(parameter) for parameter in encoder.parameters()}
    groups = [
        {
            "params": [
                p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids
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
    lap("model_optimizer")
    scaler = (
        torch.amp.GradScaler(device.type, init_scale=config.amp_init_scale)
        if config.precision == "fp16"
        else None
    )
    epoch_updates = EffectiveBatchSampler(
        len(training_data),
        config.batch_size,
        config.accumulation_steps,
        seed=config.seed,
        epoch=0,
    ).updates
    return dict(
        train=train,
        training_data=training_data,
        encoder=encoder,
        encoder_provenance=encoder_provenance,
        model=model,
        optimizer=optimizer,
        scaler=scaler,
        total_updates=epoch_updates * config.epochs,
        release_binding=release_binding,
        startup_seconds=startup,
        delta_guard=delta_guard,
        supervision_receipt=supervision_receipt,
        public_receipt=public_receipt,
    )


def _memory(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {"peak_allocated_bytes": None, "peak_reserved_bytes": None}
    return {
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _training_autocast(config: TrainConfig, device: torch.device) -> torch.autocast:
    return torch.autocast(
        device.type,
        dtype=torch.float16 if config.precision == "fp16" else torch.bfloat16,
        enabled=config.precision != "fp32",
    )


def profile_updates(
    state: dict[str, Any],
    config: TrainConfig,
    device: torch.device,
    caps: dict,
    collate: Callable,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """Run the trainer's exact first-epoch update arithmetic with phase timing.

    Slot-classifier window selection/augmentation runs on CPU (``cpu_transform``) before
    the copy (``host_to_device``), inside the same autocast context the trainer uses:
    CPU-autocast-sensitive ops (e.g. ``affine_grid``) change values in CPU FP16/BF16
    diagnostics, while a CUDA autocast context leaves this CPU work unchanged.
    """
    model, optimizer, scaler = state["model"], state["optimizer"], state["scaler"]
    strategy = config.classifier_strategy
    classifier_inputs = (
        None
        if strategy is None
        else (
            window_bag_training_inputs
            if strategy["kind"] == WINDOW_BAG_KIND
            else source_slot_training_inputs
        )
    )
    cpu = torch.device("cpu")
    encoder = state["encoder"]
    training_data = state["training_data"]
    sampler = EffectiveBatchSampler(
        len(training_data),
        config.batch_size,
        config.accumulation_steps,
        seed=config.seed,
        epoch=0,
    )
    planned = caps["warmup_updates"] + caps["measured_updates"]
    if planned > sampler.updates:
        raise ValueError("Profiling cap exceeds the first epoch's updates")
    loader = DataLoader(
        SeededDataset(TimedItems(training_data)),
        batch_sampler=sampler,
        num_workers=config.workers,
        pin_memory=device.type == "cuda",
        generator=torch.Generator().manual_seed(config.seed),
        collate_fn=collate,
    )
    iterator: Any = iter(loader)
    model.train()
    if not any(p.requires_grad for p in encoder.parameters()):
        encoder.eval()
    timer = EncoderTimer(encoder, device)
    records: list[dict] = []
    step, started, stop_reason = 0, clock(), "measured_updates_completed"
    try:
        for update in range(planned):
            if clock() - started > caps["max_wall_seconds"]:
                stop_reason = "wall_cap"
                break
            phases = PhaseClock(device)
            timer.encoded_images = 0
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            update_started = time.perf_counter()
            batches = [
                phases.measure("data_wait", lambda: next(iterator))
                for _ in range(config.accumulation_steps)
            ]
            masks = phases.measure(
                "host_to_device", lambda: [b["label_mask"].to(device) for b in batches]
            )
            counts = observed_counts(masks)
            optimizer.zero_grad(set_to_none=True)
            loss_total = torch.zeros((), device=device)
            if counts.any():
                for batch, mask in zip(batches, masks):
                    if classifier_inputs is None:
                        inputs = phases.measure(
                            "host_to_device",
                            lambda: (_model_inputs(batch, device), batch["labels"].to(device)),
                        )
                    else:
                        assert strategy is not None

                        def select():
                            with _training_autocast(config, device):
                                return classifier_inputs(
                                    batch, cpu, strategy, seed=config.seed, epoch=0, update=update
                                )

                        selected = phases.measure("cpu_transform", select)
                        inputs = phases.measure(
                            "host_to_device",
                            lambda: (
                                tuple(t.to(device, non_blocking=True) for t in selected),
                                batch["labels"].to(device),
                            ),
                        )

                    def forward():
                        with _training_autocast(config, device):
                            logits = model(*inputs[0])
                            return effective_batch_loss(logits, inputs[1], mask, counts)

                    loss = phases.measure("forward", forward)
                    phases.measure(
                        "backward",
                        lambda: (loss if scaler is None else scaler.scale(loss)).backward(),
                    )
                    encoder_forward, encoder_backward = timer.collect(timer._mark())
                    phases.totals["encoder_forward"] += encoder_forward
                    phases.totals["encoder_backward"] += encoder_backward
                    loss_total += loss.detach()

                def optimize():
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    norm = nn.utils.clip_grad_norm_(
                        model.parameters(), config.clip_grad_norm, error_if_nonfinite=True
                    )
                    multiplier = learning_rate_multiplier(
                        step, state["total_updates"], config.warmup_updates, config.lr_schedule
                    )
                    for group, base_lr in zip(
                        optimizer.param_groups, [config.head_lr, config.encoder_lr]
                    ):
                        group["lr"] = base_lr * multiplier
                    if scaler is None:
                        optimizer.step()
                    else:
                        scaler.step(optimizer)
                        scaler.update()
                    return norm

                norm = phases.measure("optimizer", optimize)
                step += 1
            else:
                norm = torch.tensor(0.0)
            phases.sync()
            wall = time.perf_counter() - update_started
            valid = torch.cat([b["sample_valid"] for b in batches])
            records.append(
                {
                    "update": update + 1,
                    "warmup": update < caps["warmup_updates"],
                    "wall_seconds": wall,
                    "phase_seconds": dict(phases.totals),
                    "item_seconds_sum": float(
                        sum(float(b["profile_item_seconds"].sum()) for b in batches)
                    ),
                    "studies": int(valid.sum()),
                    "views": sum(encoded_views(b, strategy) for b in batches),
                    "encoder_images": timer.encoded_images,
                    "microbatch_studies": [len(b["study_uid"]) for b in batches],
                    "study_uids": [uid for b in batches for uid in b["study_uid"]],
                    "observed_cells": float(counts.sum()),
                    "loss": float(loss_total),
                    "gradient_norm": float(norm),
                    "optimizer_steps": step,
                    **_memory(device),
                }
            )
    finally:
        timer.close()
        iterator = None
    return {"records": records, "stop_reason": stop_reason, "optimizer_steps": step}


def encoded_views(batch: dict, strategy: dict | None) -> int:
    """Encoder images one microbatch contributes: valid slices, or acquired slots x windows."""
    valid = batch["sample_valid"]
    if strategy is None:
        return int(batch["slice_mask"][valid].sum())
    windows = len(WINDOW_BAG_VIEWS) if strategy["kind"] == WINDOW_BAG_KIND else 1
    return int(batch["series_mask"][valid].sum()) * windows


def summarize(records: list[dict]) -> dict[str, Any]:
    measured = [record for record in records if not record["warmup"]]
    if not measured:
        return {"measured_updates": 0}
    wall = sum(record["wall_seconds"] for record in measured)
    studies = sum(record["studies"] for record in measured)
    views = sum(record["views"] for record in measured)
    peaks = [r["peak_allocated_bytes"] for r in measured if r["peak_allocated_bytes"] is not None]
    reserved = [r["peak_reserved_bytes"] for r in measured if r["peak_reserved_bytes"] is not None]
    return {
        "measured_updates": len(measured),
        "wall_seconds": wall,
        "seconds_per_update": wall / len(measured),
        "studies": studies,
        "views": views,
        "studies_per_second": studies / wall,
        "views_per_second": views / wall,
        "phase_seconds": {
            phase: sum(record["phase_seconds"][phase] for record in measured) for phase in PHASES
        },
        "item_seconds_sum": sum(record["item_seconds_sum"] for record in measured),
        "peak_allocated_bytes": max(peaks) if peaks else None,
        "peak_reserved_bytes": max(reserved) if reserved else None,
    }


def input_format(dataset: Any) -> str | None:
    """The cache format a dataset reads; ``None`` for unsupported (e.g. synthetic) data."""
    from rsna.competitive.data import CompetitiveStudyDataset
    from rsna.competitive.public_input import PublicTripletStudyDataset

    if isinstance(dataset, PublicTripletStudyDataset):
        return PUBLIC_FORMAT
    if isinstance(dataset, CompetitiveStudyDataset):
        return NATIVE_FORMAT
    return None


def expected_input_format(config: TrainConfig) -> str:
    return PUBLIC_FORMAT if config.public_input is not None else NATIVE_FORMAT


def _evict(path: Path, evict_page_cache: bool) -> None:
    if evict_page_cache and hasattr(os, "posix_fadvise"):
        with path.open("rb") as handle:
            os.posix_fadvise(handle.fileno(), 0, 0, getattr(os, "POSIX_FADV_DONTNEED"))


def decompose_inputs(
    dataset: Any, indices: list[int], evict_page_cache: bool, expected_format: str
) -> dict[str, Any]:
    """Separate read, checksum, decompression and CPU transform for real studies.

    Native-cache and public-triplet members are decomposed by their own readers; a
    dataset whose format differs from the bound recipe's is refused.
    """
    actual = input_format(dataset)
    if actual is None or actual != expected_format:
        raise ValueError("Dataset input format differs from the bound recipe; refused")
    timings = (
        _decompose_public(dataset, indices, evict_page_cache)
        if actual == PUBLIC_FORMAT
        else _decompose_native(dataset, indices, evict_page_cache)
    )
    return {"format": actual, "studies": len(indices), **timings}


def _decompose_public(dataset: Any, indices: list[int], evict_page_cache: bool) -> dict[str, Any]:
    """Public NPZ member read, SHA-256, decompression and source ten-window construction.

    Training window selection/augmentation is timed per update as ``cpu_transform``.
    """
    from rsna.competitive.public_input import MEMBER_ARRAYS
    from rsna.schema import TARGETS

    totals = dict.fromkeys(("read", "checksum", "decompress", "transform"), 0.0)
    total_bytes = 0
    empty = torch.zeros(len(TARGETS), dtype=torch.float32)
    for index in indices:
        uid = dataset.studies.iloc[index].study_uid
        member = dataset.index["members"][uid]
        path = dataset.root / member["path"]
        _evict(path, evict_page_cache)
        started = time.perf_counter()
        payload = path.read_bytes()
        totals["read"] += time.perf_counter() - started
        total_bytes += len(payload)
        started = time.perf_counter()
        if (
            len(payload) != member["bytes"]
            or hashlib.sha256(payload).hexdigest() != member["sha256"]
        ):
            raise ValueError("Public triplet member checksum/size mismatch during decomposition")
        totals["checksum"] += time.perf_counter() - started
        started = time.perf_counter()
        with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
            if set(archive.files) != set(MEMBER_ARRAYS):
                raise ValueError("Public triplet member arrays differ from the source contract")
            arrays = {name: archive[name] for name in MEMBER_ARRAYS}
        totals["decompress"] += time.perf_counter() - started
        started = time.perf_counter()
        tensors = {name: torch.from_numpy(value) for name, value in arrays.items()}
        dataset.adapter.study_inputs(
            tensors["images"],
            tensors["slot_mask"],
            tensors["features"],
            tensors["positions"],
            empty,
            empty,
            study_uid=uid,
        )
        totals["transform"] += time.perf_counter() - started
    return {**{f"{name}_seconds": value for name, value in totals.items()}, "bytes": total_bytes}


def _decompose_native(dataset: Any, indices: list[int], evict_page_cache: bool) -> dict[str, Any]:
    """Native cache volume read, checksum, decompression and slice/neighbor sampling."""
    from rsna.competitive.data import _indices, _positions
    from rsna.competitive.neighbor_input import sample_neighbor_images
    from rsna.data import select_series

    totals = dict.fromkeys(("read", "checksum", "decompress", "transform"), 0.0)
    total_bytes = 0
    for index in indices:
        uid = dataset.studies.iloc[index].study_uid
        for entry in select_series(dataset.series[uid], dataset.max_series, False):
            series_uid = entry["SeriesInstanceUID"]
            path = dataset.cache / dataset.raw_split / uid / f"{series_uid}.npz"
            meta = dataset._metadata[(uid, series_uid)][1]
            _evict(path, evict_page_cache)
            started = time.perf_counter()
            payload = path.read_bytes()
            totals["read"] += time.perf_counter() - started
            total_bytes += len(payload)
            started = time.perf_counter()
            if len(payload) != meta.get("cache_bytes") or hashlib.sha256(
                payload
            ).hexdigest() != meta.get("cache_sha256"):
                raise ValueError("Cache volume checksum/size mismatch during decomposition")
            totals["checksum"] += time.perf_counter() - started
            started = time.perf_counter()
            with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
                volume = archive["volume"]
            totals["decompress"] += time.perf_counter() - started
            started = time.perf_counter()
            count = meta["n_slices"]
            selected = _indices(count, dataset.slices, False)
            sample_neighbor_images(volume, meta, selected, dataset.side, dataset.neighbor_channels)
            _positions(meta, count)
            totals["transform"] += time.perf_counter() - started
    return {**{f"{name}_seconds": value for name, value in totals.items()}, "bytes": total_bytes}


def build_receipt(
    *,
    setting: str,
    base: TrainConfig,
    candidate: TrainConfig,
    config_sha256: str,
    caps: dict,
    profile: dict,
    bindings: dict,
    release_binding: dict,
    train: pd.DataFrame,
    device: torch.device,
    decomposition: dict | None,
    evidence_blockers: list[str],
    startup_seconds: dict[str, float] | None = None,
    release_admission: str = "full_original",
) -> dict[str, Any]:
    profiled_uids = [uid for record in profile["records"] for uid in record["study_uids"]]
    return {
        "schema_version": SCHEMA_VERSION,
        "kind": "training_throughput_profile_v1",
        "setting": setting,
        "overrides": {name: getattr(candidate, name) for name in THROUGHPUT_FIELDS},
        "numerical_audit": numerical_audit(base, candidate),
        "base_config_sha256": config_sha256,
        "candidate_config_sha256": _hash(dataclasses.asdict(candidate)),
        "caps": caps,
        "bindings": bindings,
        "release": release_binding,
        "training_uids_sha256": _uid_hash(train),
        "profiled_uids_sha256": hashlib.sha256("\n".join(profiled_uids).encode()).hexdigest(),
        "code_sha256": code_tree_sha256(),
        "torch_version": str(torch.__version__),
        "device_type": device.type,
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "device_total_bytes": (
            int(torch.cuda.get_device_properties(device).total_memory)
            if device.type == "cuda"
            else None
        ),
        "pid": os.getpid(),
        "stop_reason": profile["stop_reason"],
        "release_admission": release_admission,
        "startup_seconds": startup_seconds,
        "updates": [
            {key: value for key, value in row.items() if key != "study_uids"}
            | {
                "study_uids_sha256": hashlib.sha256(
                    json.dumps(row["study_uids"], separators=(",", ":")).encode()
                ).hexdigest()
            }
            for row in profile["records"]
        ],
        "timing_scope": "synchronized_instrumented_updates_excludes_setup_and_input_decomposition",
        "encoder_backward_scope": "first_encoder_output_gradient_to_whole_backward_end_includes_interleaving",
        "summary": summarize(profile["records"]),
        "input_decomposition": decomposition,
        "evidence_eligible": not evidence_blockers,
        "evidence_blockers": evidence_blockers,
        "checkpoint_written": False,
        "accuracy_claim": False,
        "production_acceptance": False,
    }


def write_new_receipt(path: Path, receipt: dict) -> None:
    payload = json.dumps(receipt, sort_keys=True, indent=2, allow_nan=False).encode()
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise ValueError("A distinct profiling receipt path is required") from error
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
