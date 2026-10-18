"""New fitting-only stochastic full-encoder experiment, without optimizer updates."""

from __future__ import annotations

import hashlib
import random
import time
from dataclasses import replace

import numpy as np
import torch

from rsna.competitive.checkpoint import capture_rng, restore_rng
from rsna.competitive.licensed_importance import Recipe, sample, validate_proposal
from rsna.competitive.licensed_parent_train import train_with_frozen_bn
from rsna.competitive.paired_encoder_noise import GradientReference, PairedGradientError
from rsna.competitive.paired_logit_control import PairedLogitReadout, correction
from rsna.competitive.released_context import anchored_loss


def _seed(uid, view, phase, draw):
    raw = hashlib.sha256(f"paired_encoder_noise_v1/{uid}/{view}/{phase}/{draw}".encode()).digest()
    return int.from_bytes(raw[:8], "little") % 2**63


def _seed_streams(seed, device):
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.random.default_generator.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)


def _gradient(model, images, case, device, seed, *, logq=None, shift=None):
    _seed_streams(seed, device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.perf_counter()
    with torch.autocast(
        device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"
    ):
        if logq is None:
            logits = model(images.unsqueeze(0)).float()
        else:
            logits = model(
                images.unsqueeze(0),
                logq.to(device).unsqueeze(0),
                baseline_shift=shift.to(device).reshape(1, 12),
            ).float()
    loss = anchored_loss(
        logits,
        case["parent_logits"].to(device).reshape(1, 12).float(),
        case["labels"].to(device).reshape(1, 12).float(),
        case["mask"].to(device).reshape(1, 12),
        source_weight=0.25,
    )
    named = dict(model.named_parameters())
    values = torch.autograd.grad(loss * 1024.0, tuple(named.values()), allow_unused=False)
    gradients = {n: g.detach().float() / 1024.0 for n, g in zip(named, values, strict=True)}
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    seconds = time.perf_counter() - start
    return gradients, logits.detach().sigmoid().double(), seconds


@torch.no_grad()
@torch.random.fork_rng(devices=[])
def _teacher(readout, features, logq):
    rng = torch.get_rng_state().clone()
    value = readout(features, logq)
    if (
        value.shape != (features.shape[0], 12)
        or value.dtype != torch.float32
        or value.device.type != "cpu"
        or value.requires_grad
        or not torch.isfinite(value).all()
        or not torch.equal(rng, torch.get_rng_state())
    ):
        raise ValueError("Deterministic frozen CPU parent logits without stochastic RNG required")
    return value


def measure_case(
    model,
    views,
    features,
    teacher_readout,
    proposal,
    uid,
    recipe,
    *,
    device,
    expected_parameters,
    reference_draws=16,
    paired_draws=32,
    progress=None,
):
    """Each loss is SINGLE view; balance views only when aggregating separate draws.

    Full-bag dropout draws use a distinct seed namespace. The two sampled arms
    share exact windows, view and dropout seed. Report whole-parameter preclip
    per-study gradients; optimizer/update/runtime or disease acceptance is absent.
    """
    validate_proposal(proposal)
    if (
        PairedLogitReadout not in type(model).__mro__
        or not isinstance(recipe, Recipe)
        or recipe.role != "influence"
        or recipe.windows != 16
        or recipe.source_weight != 0.25
        or type(reference_draws) is not int
        or not 2 <= reference_draws <= 32
        or type(paired_draws) is not int
        or not 2 <= paired_draws <= 64
        or not isinstance(uid, str)
        or not uid
        or not isinstance(views, (list, tuple))
        or len(views) != 2
        or type(expected_parameters) is not int
        or expected_parameters <= 0
        or not isinstance(features, torch.Tensor)
        or not all(p.requires_grad for p in model.parameters())
        or sum(p.numel() for p in model.parameters()) != expected_parameters
        or not any(n.startswith("backbone.") for n, _ in model.named_parameters())
        or features.ndim != 3
        or features.shape[:2] != (2, 94)
        or features.device.type != "cpu"
        or features.dtype != torch.float32
        or features.requires_grad
        or not torch.isfinite(features).all()
    ):
        raise ValueError(
            "Complete actual encoder, two original views and admitted fitting-only recipe required"
        )
    for case in views:
        if (
            case["images"].ndim != 4
            or case["images"].shape[:2] != (94, 3)
            or case["images"].dtype != torch.float32
            or not torch.isfinite(case["images"]).all()
        ):
            raise ValueError("Full94 original RGB input views required")
    rng = capture_rng(device)
    modes = [(module, module.training) for module in model.modules()]
    bn = {
        n: b.detach().clone()
        for n, b in model.named_buffers()
        if any(x in n for x in ("running_mean", "running_var", "num_batches_tracked"))
    }
    results = []
    try:
        train_with_frozen_bn(model)
        teacher_full = _teacher(teacher_readout, features, None)
        if teacher_full.shape != (2, 12) or teacher_full.dtype != torch.float32:
            raise ValueError("Finite original deterministic two-view parent readout required")
        draw_recipe = replace(recipe, epochs=512)
        for view, case in enumerate(views):
            images = case["images"].to(device)
            reference = GradientReference()
            probabilities = torch.zeros((1, 12), dtype=torch.float64, device=device)
            probability_m2 = torch.zeros_like(probabilities)
            reference_seconds = []
            for draw in range(reference_draws):
                gradients, probs, seconds = _gradient(
                    model, images, case, device, _seed(uid, view, "dense", draw)
                )
                reference.add(gradients)
                delta = probs - probabilities
                probabilities.add_(delta / (draw + 1))
                probability_m2.add_(delta * (probs - probabilities))
                reference_seconds.append(seconds)
                del gradients
                if progress:
                    progress(
                        {
                            "view": view,
                            "phase": "dense",
                            "completed": draw + 1,
                            "total": reference_draws,
                        }
                    )
            screen = PairedGradientError(reference)
            del reference
            forward_errors = {"ordinary": 0.0, "controlled": 0.0}
            timing: dict[str, list[float]] = {"ordinary": [], "controlled": []}
            selected_draws = []
            for virtual_epoch in range(512):
                positions, mirror, logq = sample(uid, draw_recipe, virtual_epoch, proposal)
                if int(mirror) != view:
                    continue
                selected_draws.append((virtual_epoch, positions, logq))
                if len(selected_draws) == paired_draws:
                    break
            if len(selected_draws) != paired_draws:
                raise ValueError("Fixed balanced-view engineering sampling schedule exhausted")
            for draw, (virtual_epoch, positions, logq) in enumerate(selected_draws):
                with torch.no_grad(), torch.random.fork_rng(devices=[]):
                    delta = correction(
                        teacher_full[view : view + 1],
                        _teacher(teacher_readout, features[view : view + 1, positions], logq),
                    )[0]
                sampled = images[positions]
                seed = _seed(uid, view, "sampled", virtual_epoch)
                args = (("ordinary", torch.zeros_like(delta)), ("controlled", delta))
                # Alternate execution order so a warm-cache/timing order cannot always favor one arm.
                if draw % 2:
                    args = args[::-1]
                gradients_by_arm = {}
                for arm, shift in args:
                    gradients, probs, seconds = _gradient(
                        model, sampled, case, device, seed, logq=logq, shift=shift
                    )
                    gradients_by_arm[arm] = gradients
                    timing[arm].append(seconds)
                    forward_errors[arm] += float((probs - probabilities).square().mean())
                screen.add(gradients_by_arm["ordinary"], gradients_by_arm["controlled"])
                del gradients_by_arm
                if progress:
                    progress(
                        {
                            "view": view,
                            "phase": "paired",
                            "completed": draw + 1,
                            "total": paired_draws,
                        }
                    )
            result = screen.result()
            result.update(
                view=view,
                forward_MSE={k: v / paired_draws for k, v in forward_errors.items()},
                dense_forward_mean_squared_standard_error=float(
                    probability_m2.mean() / (reference_draws * (reference_draws - 1))
                ),
                dense_forward_backward_seconds=sum(reference_seconds),
                sampled_forward_backward_seconds={k: sum(v) for k, v in timing.items()},
                actual_optimizer_step_timing_measured=False,
                dropout_arms_seed_matched=True,
                independent_dense_seed_namespace=True,
                loss_uses_one_view=True,
                preclip_per_study_gradients=True,
            )
            results.append(result)
            del screen, images
        if any(
            not torch.equal(value, dict(model.named_buffers())[name]) for name, value in bn.items()
        ):
            raise ValueError("Actual frozen parent BN buffers changed")
        return {
            "views": results,
            "parameters": expected_parameters,
            "fixed_gradient_scale": 1024.0,
            "precision": "FP16_forward_FP32_loss" if device.type == "cuda" else "CPU_FP32",
            "optimizer_or_parameter_updates": 0,
            "no_accuracy_or_production_fit_acceptance": True,
        }
    finally:
        for module, training in modes:
            module.training = training
        restore_rng(rng, device)
