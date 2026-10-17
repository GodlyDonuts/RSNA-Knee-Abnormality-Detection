"""Full-study/readout counterfactuals on immutable licensed encoder features.

These models share an MRI-trained encoder supplied under its separate licence.
Only its exact norm, finding attention and classifiers learn here. This is not
an independent whole-model OOF estimate or a clinical accuracy claim.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
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
from rsna.competitive.licensed_importance import Recipe as ObservationRecipe
from rsna.competitive.licensed_importance import sample
from rsna.competitive.licensed_parent_train import PARENT_SHA, lr_factor
from rsna.competitive.released_context import anchored_loss
from rsna.competitive.released_context_infer import PARENT_SOURCE

VERSION = "licensed_cached_MIL_label_coverage_factorial_v1"
READOUT_SHA = "8ce6546b4ec7a50fe6a3e9c0a0dd51e813d78a418ad61094427282bbbd72b8c5"


@dataclass(frozen=True)
class Recipe:
    label_role: str
    windows: int
    epochs: int = 3
    effective_batch: int = 8
    seed: int = 20261016
    head_lr: float = 3e-5
    weight_decay: float = 0.02
    warmup_fraction: float = 0.1
    final_lr_fraction: float = 0.01
    source_weight: float = 0.25
    clip: float = 1.0

    def __post_init__(self):
        if self.label_role not in ("original", "released_S6") or self.windows not in (16, 94):
            raise ValueError("One fixed source and either16/full94observations required")
        if any(type(v) is not int or v < 1 for v in (self.epochs, self.effective_batch)):
            raise ValueError("Positive fixed epochs/effective batch required")
        if type(self.seed) is not int or not 0 <= self.seed < 2**63:
            raise ValueError("Fixed valid seed required")
        values = (
            self.head_lr,
            self.weight_decay,
            self.warmup_fraction,
            self.final_lr_fraction,
            self.source_weight,
            self.clip,
        )
        if any(type(x) not in (int, float) or not math.isfinite(x) or x <= 0 for x in values):
            raise ValueError("Finite positive recipe values required")
        if self.warmup_fraction >= 1 or self.source_weight > 1 or self.final_lr_fraction > 1:
            raise ValueError("Invalid fixed schedule/source weight")


class Readout(nn.Module):
    """The admitted eight-tensor readout; encoder is supplied only at inference."""

    def __init__(self, original):
        super().__init__()
        if (
            original.get("format") != "licensed_parent_MIL_readout_v1"
            or original.get("parent_checkpoint_sha256") != PARENT_SHA
            or original.get("source_notebook_sha256") != PARENT_SOURCE
        ):
            raise ValueError("Exact admitted original MRI parent readout required")
        self.norm = nn.LayerNorm(1024, eps=1e-5)
        self.att = nn.Sequential(
            nn.Linear(1024, 256), nn.Tanh(), nn.Dropout(0.0), nn.Linear(256, 12)
        )
        self.clsW = nn.Parameter(torch.empty(12, 1024))
        self.clsb = nn.Parameter(torch.empty(12))
        self.load_state_dict(original["model"], strict=True)
        if not all(
            v.dtype == torch.float32 and torch.isfinite(v).all() for v in self.state_dict().values()
        ):
            raise ValueError("Complete finite FP32 norm/attention/classifier required")

    def forward(self, features):
        if (
            features.ndim != 3
            or features.shape[-1] != 1024
            or not 1 <= features.shape[1] <= 94
            or features.dtype != torch.float32
        ):
            raise ValueError("B,K,1024FP32 original raw encoder features required")
        h = self.norm(features)
        weights = self.att(h).float().softmax(1)
        return (torch.einsum("bkn,bkf->bnf", weights, h) * self.clsW).sum(-1) + self.clsb


def observations(uid, epoch, recipe):
    # Consume the same16random uniforms and view variate in every factorial arm.
    # Full94simply keeps all original windows instead of the sampled indices.
    base = ObservationRecipe(
        epochs=recipe.epochs, effective_batch=recipe.effective_batch, seed=recipe.seed
    )
    selected, mirror, _ = sample(uid, base, epoch, torch.full((94,), 1 / 94, dtype=torch.float64))
    return selected if recipe.windows == 16 else list(range(94)), mirror


def fit(original, data, output, binding, recipe, *, resume=False, stop_after=None):
    """One CPU-only engine; wrapper admits exact features/labels/provenance first.

    data.uids are unique; data.case(index, selected, mirror) returns raw features,
    unchanged view-specific teacher logits, source labels and bool masks.
    All original immutable epochs remain present. Checkpoints contain complete
    learned/AdamW/LR/counter/history/RNG state; no encoder tensor is duplicated.
    """
    if not isinstance(recipe, Recipe) or not len(data) or len(set(data.uids)) != len(data):
        raise ValueError("Nonempty unique admitted fitting population required")
    if (
        binding.get("parent_checkpoint_sha256") != PARENT_SHA
        or binding.get("original_readout_sha256") != READOUT_SHA
    ):
        raise ValueError("Original whole encoder/readout provenance required")
    if binding.get("protected_gold_confirmation_used") is not False:
        raise ValueError("Protected clinical cohorts must remain unused")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    random.seed(recipe.seed)
    np.random.seed(recipe.seed % 2**32)
    torch.manual_seed(recipe.seed)
    model = Readout(original).float().cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=recipe.head_lr,
        weight_decay=recipe.weight_decay,
        foreach=False,
        fused=False,
    )
    batches = math.ceil(len(data) / recipe.effective_batch)
    total = recipe.epochs * batches
    if stop_after is not None and (type(stop_after) is not int or not 1 <= stop_after < total):
        raise ValueError("Strict intermediate optimizer stop boundary required")
    expected = {
        **binding,
        "version": VERSION,
        "recipe": asdict(recipe),
        "population": len(data),
        "population_sha256": hashlib.sha256(
            json.dumps(data.uids, separators=(",", ":")).encode()
        ).hexdigest(),
        "torch_version": str(torch.__version__),
        "precision": "CPU_FP32",
        "encoder": "unchanged_original_MRI_parent_readonly_features",
        "head_dropout": 0.0,
        "total_steps": total,
    }
    output = Path(output)
    device = torch.device("cpu")
    epoch = cursor = completed = 0
    history = []
    with run_lock(output):
        if resume:
            state = torch.load(output / "checkpoint.pt", map_location="cpu", weights_only=True)
            if state.get("binding") != expected or state.get("optimizer_boundary") is not True:
                raise ValueError("Exact complete source/data/recipe/optimizer boundary required")
            epoch, cursor, completed = state["epoch"], state["cursor"], state["completed"]
            if not (
                type(epoch) is int
                and 0 <= epoch <= recipe.epochs
                and type(cursor) is int
                and 0 <= cursor < batches
                and completed == epoch * batches + cursor
                and (epoch < recipe.epochs or cursor == 0)
            ):
                raise ValueError("Exact epoch/cursor/update relation required")
            if len(state["history"]) != completed or state["schedule"] != {
                "completed": completed,
                "total": total,
                "kind": "warmup_cosine",
            }:
                raise ValueError("Complete numerical history/schedule required")
            if stop_after is not None and stop_after <= completed:
                raise ValueError("Future stop boundary required")
            if any(not torch.isfinite(v).all() for v in state["model"].values()) or any(
                row.get("update") != i + 1 for i, row in enumerate(state["history"])
            ):
                raise ValueError("Finite complete model and contiguous numerical history required")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            history = state["history"]
            restore_rng(state["rng"], device)
        else:
            if (output / "checkpoint.pt").exists() or (output / "run.json").exists():
                raise ValueError("Existing output requires explicit original resume")
            atomic_json(output / "run.json", {"binding": expected})

        def snapshot():
            return {
                "binding": expected,
                "optimizer_boundary": True,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "cursor": cursor,
                "completed": completed,
                "history": history,
                "schedule": {"completed": completed, "total": total, "kind": "warmup_cosine"},
                "rng": capture_rng(device),
            }

        while epoch < recipe.epochs:
            generator = torch.Generator().manual_seed(recipe.seed + epoch)
            order = torch.randperm(len(data), generator=generator).tolist()
            while cursor < batches:
                ids = order[cursor * recipe.effective_batch : (cursor + 1) * recipe.effective_batch]
                examples = [data.case(i, *observations(data.uids[i], epoch, recipe)) for i in ids]
                x, teacher, labels, mask = [
                    torch.stack([case[n] for case in examples])
                    for n in ("features", "parent_logits", "labels", "mask")
                ]
                factor = lr_factor(completed, total, recipe)
                for group in optimizer.param_groups:
                    group["lr"] = recipe.head_lr * factor
                model.train()
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = anchored_loss(
                    logits, teacher, labels, mask, source_weight=recipe.source_weight
                )
                if not torch.isfinite(loss):
                    raise ValueError("Nonfinite cached-head loss")
                loss.backward()
                norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), recipe.clip, error_if_nonfinite=True
                )
                optimizer.step()
                completed += 1
                cursor += 1
                history.append(
                    {
                        "update": completed,
                        "epoch": epoch + 1,
                        "batch": cursor,
                        "studies": len(ids),
                        "loss": float(loss.detach()),
                        "gradient_norm": float(norm),
                        "head_lr": recipe.head_lr * factor,
                    }
                )
                if completed % 16 == 0:
                    atomic_json(
                        output / "progress.json",
                        {"completed": completed, "total": total, "last": history[-1]},
                    )
                if cursor == batches:
                    epoch += 1
                    cursor = 0
                    state = snapshot()
                    D = output / "epoch-checkpoints"
                    D.mkdir(exist_ok=True)
                    head = D / f"epoch-{epoch:03d}.pt"
                    full = D / f"epoch-{epoch:03d}-optimizer.pt"
                    if head.exists() or full.exists():
                        raise ValueError("Immutable original epoch already exists")
                    save_checkpoint(
                        head,
                        {
                            "binding": expected,
                            "epoch": epoch,
                            "completed": completed,
                            "model": model.state_dict(),
                        },
                    )
                    save_checkpoint(full, state)
                    atomic_json(
                        D / f"epoch-{epoch:03d}.json",
                        {
                            "epoch": epoch,
                            "completed": completed,
                            "model": {"bytes": head.stat().st_size, "sha256": file_sha256(head)},
                            "optimizer": {
                                "bytes": full.stat().st_size,
                                "sha256": file_sha256(full),
                            },
                        },
                    )
                    save_checkpoint(output / "checkpoint.pt", state)
                    break
                if stop_after is not None and completed == stop_after:
                    save_checkpoint(output / "checkpoint.pt", snapshot())
                    return {
                        "status": "stopped_at_original_optimizer_boundary",
                        "completed": completed,
                    }
            if stop_after is not None and completed == stop_after:
                return {"status": "stopped_at_original_optimizer_boundary", "completed": completed}
        atomic_json(
            output / "progress.json", {"completed": completed, "total": total, "last": history[-1]}
        )
        return {
            "status": "actual_cached_MIL_fixed_final_complete",
            "completed": completed,
            "total": total,
            "epochs": epoch,
            "trainable_parameters": sum(p.numel() for p in model.parameters()),
            "encoder_trained": False,
            "clinical_or_Kaggle_accuracy_claim": False,
        }


@torch.no_grad()
def predict(original, state, examples):
    """Full94two-view predictions; no labels or report input permitted here."""
    model = Readout(original).float().cpu()
    model.load_state_dict(state["model"], strict=True)
    model.eval()
    result = []
    for features in examples:
        if features.shape != (2, 94, 1024) or not torch.isfinite(features).all():
            raise ValueError("Two complete original feature views required")
        result.append(model(features).sigmoid().mean(0))
    return torch.stack(result)
