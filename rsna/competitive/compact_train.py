"""Fixed-final Native-CoAtNet-Bag12 fitting with complete optimizer-boundary resume.

No per-epoch outer evaluation, best-checkpoint selection, or competition actions.
The same numerical engine serves bounded recovery and the full fixed schedule.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import signal
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from rsna.competitive.checkpoint import (
    atomic_json,
    capture_rng,
    file_sha256,
    restore_rng,
    run_lock,
    save_checkpoint,
)
from rsna.competitive.compact_bag import (
    MODEL_NAME,
    SAMPLER_VERSION,
    TIMM_VERSION,
    WEIGHT_SHA256,
    CompactBagClassifier,
    candidate_windows,
    load_encoder,
    sample_windows,
    window_tensor,
)
from rsna.schema import TARGETS

VERSION = "native_coatnet_bag12_N3_full_v1"
OVERLAY_SHA256 = "c8fd2e860d8a3913dbeab34a9d1078316a8f893360124eac088453dc2ed6cf51"


def dataset_input_policy(data):
    policy = getattr(data, "input_policy", None)
    if policy is None:
        return None
    from rsna.competitive.compact_source_neighbors import POLICY

    if policy != POLICY:
        raise ValueError("Unrecognized compact input policy")
    return dict(policy)


def model_windows(sample, windows, policy):
    if policy is None:
        if "source_window_images" in sample:
            raise ValueError("Source-neighbor tensors require explicit dataset binding")
        return window_tensor(sample["images"], windows)
    from rsna.competitive.compact_source_neighbors import select_window_tensor

    return select_window_tensor(sample, windows)


@dataclass(frozen=True)
class Settings:
    seed: int = 20260922
    epochs: int = 16
    effective_batch: int = 8
    encoder_lr: float = 3e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.02
    pct_start: float = 0.15
    div_factor: float = 25.0
    final_div_factor: float = 10000.0
    clip_norm: float = 3.0

    def validate(self) -> None:
        if any(type(v) is not int or v < 1 for v in (self.epochs, self.effective_batch)):
            raise ValueError("Positive integer schedule required")
        if (
            type(self.seed) is not int
            or not all(
                math.isfinite(v) and v > 0
                for v in (
                    self.encoder_lr,
                    self.head_lr,
                    self.weight_decay,
                    self.pct_start,
                    self.div_factor,
                    self.final_div_factor,
                    self.clip_norm,
                )
            )
            or self.pct_start >= 1
        ):
            raise ValueError("Invalid optimizer/scheduler settings")


def order_for_epoch(count: int, seed: int, epoch: int) -> list[int]:
    return torch.randperm(count, generator=torch.Generator().manual_seed(seed + epoch)).tolist()


def make_optimizer(model: CompactBagClassifier, settings: Settings, steps: int):
    settings.validate()
    encoder = list(model.encoder.parameters())
    head = [p for name, p in model.named_parameters() if not name.startswith("encoder.")]
    if not encoder or not head or not all(p.requires_grad for p in [*encoder, *head]):
        raise ValueError("Both complete encoder and head must train")
    optimizer = torch.optim.AdamW(
        [{"params": encoder, "lr": settings.encoder_lr}, {"params": head, "lr": settings.head_lr}],
        weight_decay=settings.weight_decay,
        betas=(0.9, 0.999),
        eps=1e-8,
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=[settings.encoder_lr, settings.head_lr],
        total_steps=steps,
        pct_start=settings.pct_start,
        anneal_strategy="cos",
        div_factor=settings.div_factor,
        final_div_factor=settings.final_div_factor,
        three_phase=False,
        cycle_momentum=False,
    )
    return optimizer, scheduler


def masked_cell_loss(logits, labels, mask, active_cells: float):
    """One bag's contribution to BCE over all active cells of the effective batch."""
    if logits.shape != (12,) or labels.shape != logits.shape or mask.shape != logits.shape:
        raise ValueError("Expected twelve aligned logits, labels and masks")
    if not torch.isfinite(mask).all() or not ((mask == 0) | (mask == 1)).all():
        raise ValueError("Binary finite label masks required")
    observed = mask.bool()
    if (
        not math.isfinite(active_cells)
        or active_cells <= 0
        or not torch.isfinite(logits).all()
        or not torch.isfinite(labels[observed]).all()
        or ((labels[observed] < 0) | (labels[observed] > 1)).any()
    ):
        raise ValueError("Nonfinite prediction/target or invalid active-cell denominator")
    safe = torch.where(observed, labels, torch.zeros_like(labels))
    return (
        nn.functional.binary_cross_entropy_with_logits(
            logits.float(), safe.float(), reduction="none"
        )
        * mask
    ).sum() / active_cells


def _snapshot(output: Path, model: nn.Module, binding: dict, epoch: int, step: int) -> None:
    path = output / "epoch-checkpoints" / f"epoch-{epoch:03d}.pt"
    metadata = {"schema_version": 1, "binding": binding, "epoch": epoch, "optimizer_steps": step}
    if path.exists():
        old = torch.load(path, map_location="cpu", weights_only=True)
        if {k: v for k, v in old.items() if k != "model"} != metadata or (
            set(old["model"]) != set(model.state_dict())
            or any(
                not torch.equal(old["model"][k], v.detach().cpu())
                for k, v in model.state_dict().items()
            )
        ):
            raise ValueError("Existing immutable epoch differs from deterministic replay")
    else:
        save_checkpoint(path, {**metadata, "model": model.state_dict()})
    receipt = {
        **metadata,
        "checkpoint_bytes": path.stat().st_size,
        "checkpoint_sha256": file_sha256(path),
    }
    atomic_json(path.with_suffix(".json"), receipt)


def fit(
    model: CompactBagClassifier,
    data: Sequence,
    output: Path,
    binding: dict,
    settings: Settings,
    *,
    device: torch.device,
    resume: bool = False,
    stop_after_updates: int | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> dict:
    """Train at one bag per forward; checkpoint only at complete effective batches.

    ``data`` is the admitted fitting dataset, never the outer dataset. A bounded
    diagnostic may inject a small numerical fixture into this unchanged engine.
    """
    settings.validate()
    if not len(data):
        raise ValueError("Empty fitting dataset")
    if stop_after_updates is not None and (
        type(stop_after_updates) is not int or stop_after_updates < 1
    ):
        raise ValueError("Stop boundary must be a positive integer")
    batches = math.ceil(len(data) / settings.effective_batch)
    total = batches * settings.epochs
    policy = dataset_input_policy(data)
    if policy is not None and binding.get("input_policy") != policy:
        raise ValueError("Source-neighbor dataset requires exact input-policy binding")
    runtime = {
        "device": str(device),
        "GPU": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    expected = dict(
        binding,
        engine=VERSION,
        settings=asdict(settings),
        sampler=SAMPLER_VERSION,
        torch_version=str(torch.__version__),
        fitting_count=len(data),
        total_steps=total,
        runtime=runtime,
    )
    if policy is not None:
        from rsna.competitive.compact_source_neighbors import ENGINE

        expected["engine"] = ENGINE
    checkpoint = output / "checkpoint.pt"
    model.to(device)
    optimizer, scheduler = make_optimizer(model, settings, total)
    epoch, cursor, step, history = 0, 0, 0, []
    with run_lock(output):
        if resume:
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if state.get("binding") != expected or not state.get("optimizer_boundary"):
                raise ValueError("Resume binding or optimizer boundary differs")
            epoch, cursor, step = state["epoch"], state["cursor"], state["optimizer_steps"]
            if (
                type(epoch) is not int
                or not 0 <= epoch <= settings.epochs
                or type(cursor) is not int
                or not 0 <= cursor < len(data)
                or cursor % settings.effective_batch
                or step != epoch * batches + cursor // settings.effective_batch
                or (epoch == settings.epochs and cursor != 0)
                or state["order"]
                != (
                    order_for_epoch(len(data), settings.seed, epoch)
                    if epoch < settings.epochs
                    else []
                )
                or len(state["history"]) != step
            ):
                raise ValueError("Resume counters, sampler cursor/order or history differs")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            if scheduler.last_epoch != step:
                raise ValueError("Resume scheduler counter differs")
            history = state["history"]
            restore_rng(state["rng"], device)
        elif checkpoint.exists() or (output / "run.json").exists():
            raise ValueError("Existing fitting run requires explicit resume")
        else:
            atomic_json(
                output / "run.json",
                {"binding": expected, "diagnostic_only": binding.get("diagnostic_only", True)},
            )

        def save() -> None:
            save_checkpoint(
                checkpoint,
                {
                    "schema_version": 1,
                    "binding": expected,
                    "optimizer_boundary": True,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "epoch": epoch,
                    "cursor": cursor,
                    "optimizer_steps": step,
                    "history": history,
                    "order": order_for_epoch(len(data), settings.seed, epoch)
                    if epoch < settings.epochs
                    else [],
                    "rng": capture_rng(device),
                },
            )

        model.train()
        while epoch < settings.epochs:
            order = order_for_epoch(len(data), settings.seed, epoch)
            batch_indices = order[cursor : cursor + settings.effective_batch]
            samples = [data[i] for i in batch_indices]
            masks = [sample["label_mask"] for sample in samples]
            if any(not torch.isfinite(m).all() or not ((m == 0) | (m == 1)).all() for m in masks):
                raise ValueError("Invalid fitting masks")
            active = sum(float(m.double().sum()) for m in masks)
            if active <= 0:
                raise ValueError("Effective batch has no observed cells")
            optimizer.zero_grad(set_to_none=True)
            rate = [group["lr"] for group in optimizer.param_groups]
            loss_value = 0.0
            for sample in samples:
                windows = sample_windows(
                    sample["series_mask"],
                    sample["slice_mask"],
                    run_seed=settings.seed,
                    epoch=epoch + 1,
                    study_uid=sample["study_uid"],
                )
                images = model_windows(sample, windows, policy).to(device)
                logits = model(images)
                loss = masked_cell_loss(
                    logits, sample["labels"].to(device), sample["label_mask"].to(device), active
                )
                loss.backward()
                loss_value += float(loss.detach())
            if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise ValueError("Missing/nonfinite encoder or head gradient")
            norm = nn.utils.clip_grad_norm_(
                model.parameters(), settings.clip_norm, error_if_nonfinite=True
            )
            optimizer.step()
            scheduler.step()
            step += 1
            cursor += len(samples)
            history.append(
                {
                    "step": step,
                    "epoch": epoch + 1,
                    "studies": len(samples),
                    "active_cells": active,
                    "loss": loss_value,
                    "gradient_norm": float(norm),
                    "learning_rates": rate,
                }
            )
            atomic_json(
                output / "progress.json",
                {
                    "optimizer_steps": step,
                    "total_steps": total,
                    "epoch": epoch + 1,
                    "last_update": history[-1],
                },
            )
            finished_epoch = cursor == len(data)
            if finished_epoch:
                _snapshot(output, model, expected, epoch + 1, step)
                epoch += 1
                cursor = 0
            stopped = (stop_after_updates is not None and step >= stop_after_updates) or (
                should_stop is not None and should_stop()
            )
            if finished_epoch or stopped:
                save()
            if stopped and step < total:
                return {
                    "status": "stopped_optimizer_boundary",
                    "optimizer_steps": step,
                    "total_steps": total,
                }
        if step != total or len(history) != total:
            raise ValueError("Fixed-final update count differs")
        return {"status": "fixed_final_fit_complete", "optimizer_steps": step, "total_steps": total}


def predict(model: CompactBagClassifier, data: Sequence, device: torch.device, chunk_size=12):
    """Final-only evaluation of all valid windows; no test-time BN updates."""
    model.to(device).eval()
    policy = dataset_input_policy(data)
    rows = []
    with torch.inference_mode():
        for sample in data:
            windows = candidate_windows(sample["series_mask"], sample["slice_mask"])
            images = model_windows(sample, windows, policy).to(device)
            probability = model(images, chunk_size=chunk_size).sigmoid().float().cpu()
            if not torch.isfinite(probability).all():
                raise ValueError("Nonfinite final predictions")
            rows.append(
                {
                    "study_uid": sample["study_uid"],
                    **dict(zip(TARGETS, probability.tolist(), strict=True)),
                }
            )
    return pd.DataFrame(rows, columns=["study_uid", *TARGETS])


def input_census(data) -> dict:
    """Check every study's exact selected-series window count without decoding pixels."""
    from rsna.data import select_series

    histogram: dict[str, int] = {}
    digest = hashlib.sha256()
    for uid in data.studies.study_uid:
        selected = select_series(data.series[uid], 6, False)
        counts = [
            min(16, data._metadata[(uid, row["SeriesInstanceUID"])][1]["n_slices"])
            for row in selected
        ]
        sm = torch.tensor([True] * len(counts) + [False] * (6 - len(counts)))
        km = torch.arange(16)[None] < torch.tensor(counts + [0] * (6 - len(counts)))[:, None]
        windows = candidate_windows(sm, km)
        key = str(len(windows))
        histogram[key] = histogram.get(key, 0) + 1
        digest.update(json.dumps([uid, counts], separators=(",", ":")).encode())
    return {
        "study_count": len(data),
        "window_count_histogram": histogram,
        "selected_count_binding_sha256": digest.hexdigest(),
    }


def admitted_datasets(args):
    """Reuse exact native MRI/release/fold and fitting-overlay admissions."""
    from rsna.competitive.data import CompetitiveStudyDataset
    from rsna.competitive.folds import load_folds
    from rsna.competitive.label_source import load_fitting_label_source
    from rsna.competitive.train import select_development
    from rsna.release import load_verified_release

    manifest, release = load_verified_release(args.studies, args.artifact_root)
    folds = load_folds(args.folds, manifest, release)
    fitting, outer = select_development(manifest, folds, 0)
    if len(fitting) != 2247 or len(outer) != 1156:
        raise ValueError("Compact N3 requires exact original2247/1156 development rosters")
    folds_path = args.folds / "fold.csv" if args.folds.is_dir() else args.folds
    fitting, supervision = load_fitting_label_source(
        fitting,
        outer,
        args.fitting_supervision,
        OVERLAY_SHA256,
        release_sha256=release["manifest_sha256"],
        folds_sha256=file_sha256(folds_path),
        fold=0,
    )
    if file_sha256(args.series) != release["series_sha256"]:
        raise ValueError("Native series table differs from accepted release")
    series = pd.read_csv(args.series)
    if args.engineering:
        fitting, outer = fitting.head(16), outer.head(16)
    kwargs = dict(
        cache=args.cache,
        slices=16,
        side=384,
        max_series=6,
        augment=False,
        slice_sampling="uniform",
        series_sampling="selected",
    )
    train = CompetitiveStudyDataset(fitting, series, soft_targets=True, **kwargs)
    evaluation = CompetitiveStudyDataset(outer, series, soft_targets=False, **kwargs)
    binding = {
        "release": release,
        "supervision": supervision,
        "fitting_uids_sha256": hashlib.sha256(
            json.dumps(sorted(fitting.study_uid)).encode()
        ).hexdigest(),
        "outer_uids_sha256": hashlib.sha256(
            json.dumps(sorted(outer.study_uid)).encode()
        ).hexdigest(),
        "folds_file_sha256": file_sha256(folds_path),
        "cache": str(args.cache.resolve()),
        "diagnostic_only": args.engineering,
        "initialization_sha256": WEIGHT_SHA256,
        "initialization_revision": "2c31d82376cc244c907c064332941e6c65be81f1",
        "architecture": MODEL_NAME,
        "timm_version": TIMM_VERSION,
        "targets": list(TARGETS),
        "native_input": {
            "side": 384,
            "slices": 16,
            "max_series": 6,
            "mean": [0.5] * 3,
            "std": [0.5] * 3,
        },
    }
    policy = getattr(args, "input_policy", "n3")
    if policy == "matched-source-neighbors":
        from rsna.competitive.compact_source_neighbors import POLICY, MatchedSourceNeighborDataset

        binding["input_policy"] = dict(POLICY)
        return (
            MatchedSourceNeighborDataset(train),
            MatchedSourceNeighborDataset(evaluation),
            binding,
        )
    elif policy != "n3":
        raise ValueError("Unrecognized compact input policy")
    return train, evaluation, binding


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "studies",
        "series",
        "folds",
        "artifact-root",
        "cache",
        "fitting-supervision",
        "weights",
        "output",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--engineering", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-updates", type=int)
    parser.add_argument("--inputs-only", action="store_true")
    parser.add_argument("--input-policy", choices=("n3", "matched-source-neighbors"), default="n3")
    args = parser.parse_args()
    if (args.output / "result.json").exists():
        parser.error("Fixed-final run already finalized; do not repeat fitting/evaluation")
    if args.stop_after_updates is not None and not args.engineering:
        parser.error("Explicit stop boundary is engineering-only; full fitting stops by signal")
    train, outer, binding = admitted_datasets(args)
    source = Path(__file__).resolve().parents[1]
    binding["source_sha256"] = {
        str(p.relative_to(source.parent)): file_sha256(p) for p in sorted(source.rglob("*.py"))
    }
    if args.inputs_only:
        census = {"fitting": input_census(train), "outer": input_census(outer)}
        tensors = hashlib.sha256()
        for data in (train, outer):
            for i in range(min(16, len(data))):
                sample = data[i]
                windows = candidate_windows(sample["series_mask"], sample["slice_mask"])
                values = model_windows(sample, windows, dataset_input_policy(data))
                tensors.update(values.contiguous().numpy().tobytes())
                for name in ("labels", "label_mask", "series_mask", "slice_mask"):
                    tensors.update(sample[name].contiguous().numpy().tobytes())
        args.output.mkdir(parents=True, exist_ok=False)
        atomic_json(
            args.output / "inputs.json",
            {
                "status": "native_input_gate_passed",
                "binding": binding,
                "fitting_count": len(train),
                "outer_count": len(outer),
                "model_loaded": False,
                "all_study_window_census": census,
                "engineering32_tensors_sha256": tensors.hexdigest(),
            },
        )
        return
    device = torch.device(args.device)
    if device.type == "cpu" and not args.engineering:
        parser.error("The full compact recipe uses a BF16-capable GPU")
    if device.type == "cuda" and (
        not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()
    ):
        raise ValueError("Compact N3 training requires allocated BF16-capable CUDA")
    if device.type == "cuda":
        if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8" or torch.cuda.device_count() != 1:
            raise ValueError("Require one assigned GPU and deterministic CUDA workspace")
        torch.use_deterministic_algorithms(True, warn_only=False)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    random.seed(20260922)
    np.random.seed(20260922)
    torch.manual_seed(20260922)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(20260922)
    model = CompactBagClassifier(load_encoder(args.weights))
    head_digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        if not name.startswith("encoder."):
            head_digest.update(name.encode())
            head_digest.update(value.contiguous().view(torch.uint8).numpy().tobytes())
    binding["initial_head_sha256"] = head_digest.hexdigest()
    settings = Settings(epochs=2 if args.engineering else 16)
    stop = [False]

    def request_stop(signum, frame):
        stop[0] = True

    for sig in (signal.SIGTERM, signal.SIGUSR1):
        signal.signal(sig, request_stop)
    started = time.monotonic()
    result = fit(
        model,
        train,
        args.output,
        binding,
        settings,
        device=device,
        resume=args.resume,
        stop_after_updates=args.stop_after_updates,
        should_stop=lambda: stop[0],
    )
    if result["status"] != "fixed_final_fit_complete":
        atomic_json(args.output / "stopped.json", result)
        raise SystemExit(75)
    predictions = predict(model, outer, device)
    predictions.to_csv(args.output / "outer_predictions.csv", index=False)
    from rsna.model import auc_metrics

    aligned = outer.studies.set_index("study_uid").loc[predictions.study_uid]
    metrics = auc_metrics(
        aligned[list(TARGETS)].to_numpy(dtype=float),
        predictions[list(TARGETS)].to_numpy(dtype=float),
        aligned[[t + "__mask" for t in TARGETS]].to_numpy(dtype=float),
    )
    atomic_json(
        args.output / "result.json",
        {
            **result,
            "metrics": metrics,
            "elapsed_seconds": time.monotonic() - started,
            "diagnostic_only": args.engineering,
            "predictions_sha256": file_sha256(args.output / "outer_predictions.csv"),
            "checkpoint_sha256": file_sha256(args.output / "checkpoint.pt"),
        },
    )


if __name__ == "__main__":
    main()
