"""Matched complete-encoder B/C/C0 loop, independent of parent distillation.

Loaders must return two source-masked contexts and a complete source-region
roster. This engine does not authorize a cohort, source, GPU or clinical result.
One optimizer step averages distinct studies, including a smaller final batch.
"""

from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from rsna.competitive.predictive.density import Supports, two_mask_loss
from rsna.competitive.predictive.model import (
    Predictive,
    ReconstructionControl,
    control_two_mask_loss,
)


@dataclass(frozen=True)
class Recipe:
    arm: str
    epochs: int = 6
    effective_batch: int = 8
    encoder_lr: float = 3e-5
    head_lr: float = 3e-4
    weight_decay: float = 0.02
    warmup_fraction: float = 0.05
    end_lr_fraction: float = 0.1
    density_weight: float = 0.1
    head_only_updates: int = 0
    clip: float = 1.0
    seed: int = 20261020

    def __post_init__(self):
        if (
            self.arm not in {"B", "C", "C0"}
            or any(type(n) is not int or n < 1 for n in (self.epochs, self.effective_batch))
            or type(self.head_only_updates) is not int
            or self.head_only_updates < 0
            or type(self.seed) is not int
            or not all(
                math.isfinite(v)
                for v in (
                    self.encoder_lr,
                    self.head_lr,
                    self.weight_decay,
                    self.warmup_fraction,
                    self.end_lr_fraction,
                    self.density_weight,
                    self.clip,
                )
            )
            or min(self.encoder_lr, self.head_lr, self.clip) <= 0
            or min(self.weight_decay, self.density_weight) < 0
            or not 0 <= self.warmup_fraction < 1
            or not 0 < self.end_lr_fraction <= 1
            or (self.arm == "C0" and self.density_weight != 0)
        ):
            raise ValueError("Explicit finite matched numerical recipe required")


def objective(model, case, recipe, device):
    """No old logits, teacher targets or report text accepted at this interface."""
    if set(case) != {"contexts", "labels", "mask"} or len(case["contexts"]) != 2:
        raise ValueError("Only two complete masked contexts and original labels/masks required")
    if isinstance(model, Predictive) != (recipe.arm != "B") or not isinstance(
        model, (Predictive, ReconstructionControl)
    ):
        raise ValueError("The exact B/C/C0 architecture must match its declared arm")
    outputs = []
    observed = []
    for context in case["contexts"]:
        if set(context) != {
            "images",
            "token_metadata",
            "region_metadata",
            "targets",
            "observed",
            "supports",
        }:
            raise ValueError("Bound raw-rendered contexts and complete source supports required")
        support = context["supports"]
        if not isinstance(support, Supports):
            raise ValueError("Complete geometry-only support prior required for both arms")
        support = Supports(
            *(x.to(device) for x in (support.indptr, support.regions, support.log_prior)),
            support.region_count,
        )
        args = [
            context[n].to(device)
            for n in ("images", "token_metadata", "region_metadata", "targets", "observed")
        ]
        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            outputs.append(model(*args, support) if recipe.arm != "B" else model(*args))
        observed.append(args[-1][None])
    labels, mask = (case[n].to(device) for n in ("labels", "mask"))
    if recipe.arm == "B":
        return control_two_mask_loss(outputs, labels, mask, density_weight=recipe.density_weight)
    return two_mask_loss(
        [o["evidence"] for o in outputs],
        torch.stack([o["context_logits"] for o in outputs])[None],
        labels,
        mask,
        observed,
        model.descriptor_width,
        recipe.density_weight,
    )


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Recovery requires the same allocated GPU topology")
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])


def save_immutable(path, state):
    """Exclusive final path; no silent replacement of any prior snapshot."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with path.open("xb") as handle:
        torch.save(state, handle)


def fit(
    model,
    data,
    recipe,
    *,
    binding,
    output,
    resume=None,
    stop_after=None,
    on_update=None,
    epoch_snapshots=True,
):
    """Fresh/full/resume use the identical loop and original loader contract.

    Binding must include immutable source, base, roster and raw-input provenance;
    it is checked verbatim on resume. All private identifiers remain in private
    checkpoints. The wrapper owns resource caps and epoch publication.
    """
    if (
        not isinstance(recipe, Recipe)
        or len(data) < 1
        or not binding
        or type(epoch_snapshots) is not bool
    ):
        raise ValueError("Nonempty original cohort and explicit immutable binding required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    device = next(model.parameters()).device
    steps = math.ceil(len(data) / recipe.effective_batch)
    total = steps * recipe.epochs
    if recipe.head_only_updates >= total or (stop_after is not None and not 0 < stop_after < total):
        raise ValueError("Explicit bounded head warmup and partial-stop cursor required")
    optimizer = torch.optim.AdamW(
        [
            {"params": list(model.encoder.parameters()), "lr": recipe.encoder_lr},
            {
                "params": [p for n, p in model.named_parameters() if not n.startswith("encoder.")],
                "lr": recipe.head_lr,
            },
        ],
        weight_decay=recipe.weight_decay,
    )
    step = 0
    history = []
    if resume is not None:
        old = torch.load(resume, map_location="cpu", weights_only=False)
        if (
            old["recipe"] != asdict(recipe)
            or old["binding"] != binding
            or old["cohort_size"] != len(data)
            or old["epoch_snapshots"] != epoch_snapshots
        ):
            raise ValueError("Recovery source/base/roster/input/recipe binding differs")
        model.load_state_dict(old["model"], strict=True)
        optimizer.load_state_dict(old["optimizer"])
        step, history = old["step"], old["history"]
        if step != len(history) or not 0 < step < total:
            raise ValueError("Original contiguous update cursor required")
        restore_rng(old["rng"])

    def snapshot():
        return {
            "recipe": asdict(recipe),
            "binding": binding,
            "cohort_size": len(data),
            "epoch_snapshots": epoch_snapshots,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "step": step,
            "history": history,
            "rng": rng_state(),
        }

    while step < total:
        epoch, batch = divmod(step, steps)
        order = torch.randperm(
            len(data), generator=torch.Generator().manual_seed(recipe.seed + epoch)
        ).tolist()
        indices = order[batch * recipe.effective_batch : (batch + 1) * recipe.effective_batch]
        model.train()
        optimizer.zero_grad(set_to_none=True)
        fraction = step / max(1, total - 1)
        warm = recipe.warmup_fraction
        factor = (
            min(1.0, (step + 1) / max(1, warm * total))
            if fraction < warm
            else recipe.end_lr_fraction
            + (1 - recipe.end_lr_fraction)
            * (1 + math.cos(math.pi * (fraction - warm) / (1 - warm)))
            / 2
        )
        for group, initial in zip(optimizer.param_groups, (recipe.encoder_lr, recipe.head_lr)):
            group["lr"] = initial * factor
        loss_value = exam_value = density_value = 0.0
        for index in indices:
            loss, measures = objective(model, data.case(index, epoch), recipe, device)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite real training loss; no fallback")
            (loss / len(indices)).backward()
            loss_value += float(loss.detach()) / len(indices)
            exam_value += float(measures["exam_loss"].detach()) / len(indices)
            density_value += float(measures["density_loss"].detach()) / len(indices)
        # One device-to-host transfer preserves each original FP64 reduction and
        # Python's ordered summation while avoiding a CUDA sync per parameter.
        squares = []
        encoder_flags = []
        for name, parameter in model.named_parameters():
            if parameter.grad is not None:
                squares.append(parameter.grad.detach().double().square().sum())
                encoder_flags.append(name.startswith("encoder."))
        values = torch.stack(squares).cpu().tolist() if squares else []
        encoder_norm = sum(v for v, flag in zip(values, encoder_flags) if flag) ** 0.5
        head_norm = sum(v for v, flag in zip(values, encoder_flags) if not flag) ** 0.5
        if not math.isfinite(encoder_norm) or not math.isfinite(head_norm):
            raise FloatingPointError("Nonfinite encoder or head gradient before warmup/clipping")
        if step < recipe.head_only_updates:
            for p in model.encoder.parameters():
                p.grad = None
        grad = torch.nn.utils.clip_grad_norm_(
            model.parameters(), recipe.clip, error_if_nonfinite=True
        )
        optimizer.step()
        step += 1
        record = {
            "step": step,
            "epoch": epoch + 1,
            "studies": len(indices),
            "loss": loss_value,
            "exam_loss": exam_value,
            "density_loss": density_value,
            "encoder_gradient_norm": encoder_norm,
            "head_gradient_norm": head_norm,
            "gradient_norm": float(grad),
            "LR": [g["lr"] for g in optimizer.param_groups],
            "encoder_updated": step > recipe.head_only_updates,
        }
        history.append(record)
        if on_update is not None:
            on_update(record)
        if epoch_snapshots and step % steps == 0:
            save_immutable(output / f"epoch-{step // steps:03d}.pt", snapshot())
        if step == stop_after:
            save_immutable(output / "stopped.pt", snapshot())
            return snapshot()
    save_immutable(output / "final.pt", snapshot())
    return snapshot()
