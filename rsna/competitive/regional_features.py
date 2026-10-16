"""Frozen pre-pooling spatial summaries; not localization or diagnostic evidence.

Retain all1024channels. Fixed FP16 storage is explicitly part of the prospective
training/serving representation, while every reduction runs in FP32. The original
backbone/global pooling is observed, never replaced or modified.
"""

from __future__ import annotations

import torch

REGIONAL_VERSION = "licensed_parent_regional_summaries_v1"
REGIONAL_NAMES = (
    "upper_left_minus_global",
    "upper_right_minus_global",
    "lower_left_minus_global",
    "lower_right_minus_global",
    "center_minus_global",
    "spatial_population_SD",
    "spatial_max_minus_global",
)


@torch.no_grad()
def regional_summary(maps):
    if (
        not isinstance(maps, torch.Tensor)
        or maps.ndim != 4
        or maps.shape[0] < 1
        or tuple(maps.shape[1:]) != (1024, 10, 10)
        or maps.dtype not in (torch.float16, torch.bfloat16, torch.float32)
        or not torch.isfinite(maps).all()
    ):
        raise ValueError("Exact finite parent NCHW1024x10x10 prepool maps required")
    with torch.autocast(device_type=maps.device.type, enabled=False):
        x = maps.float()
        mean = x.mean((2, 3))
        regions = [
            x[:, :, h, w].mean((2, 3)) - mean
            for h, w in (
                (slice(0, 5), slice(0, 5)),
                (slice(0, 5), slice(5, 10)),
                (slice(5, 10), slice(0, 5)),
                (slice(5, 10), slice(5, 10)),
                (slice(2, 8), slice(2, 8)),
            )
        ]
        centered = x - mean[:, :, None, None]
        result = torch.stack(
            [*regions, centered.square().mean((2, 3)).sqrt(), x.amax((2, 3)) - mean], dim=1
        )
    if result.shape != (len(x), 7, 1024) or not torch.isfinite(result).all():
        raise ValueError("Finite complete FP32 regional representation required")
    return result


@torch.no_grad()
def regional_storage(summary):
    if (
        summary.ndim != 3
        or summary.shape[0] < 1
        or tuple(summary.shape[1:]) != (7, 1024)
        or summary.dtype != torch.float32
        or not torch.isfinite(summary).all()
    ):
        raise ValueError("Complete finite FP32 summary required before storage")
    stored = summary.to(torch.float16)
    if not torch.isfinite(stored).all():
        raise ValueError("Regional FP16 overflow is not an accepted silent clamp")
    error = ((stored.float() - summary).abs() / summary.abs().clamp_min(1)).max()
    if float(error) > 0.001:
        raise ValueError("Fixed quantization bound exceeded")
    return stored, float(error)


class RegionalCapture:
    """One owned observer per eval backbone, with explicit forward boundaries."""

    def __init__(self, backbone):
        if backbone.training or any(m.training for m in backbone.modules()):
            raise ValueError("Frozen eval backbone required")
        if (
            backbone.num_features != 1024
            or type(backbone.head.global_pool).__name__ != "SelectAdaptivePool2d"
        ):
            raise ValueError("Actual admitted original global-pool graph required")
        self.values = []
        self.active = False
        self.qualified = False
        self.maximum_CPU_GPU_scaled_error = 0.0
        self.maximum_quantization_scaled_error = 0.0
        self.handle = backbone.head.global_pool.register_forward_pre_hook(self.observe)

    def begin(self):
        if self.active or self.values:
            raise ValueError("Prior capture must be completely consumed")
        self.active = True

    def observe(self, module, args):
        if not self.active:
            return None
        if len(args) != 1:
            raise ValueError("One original feature-map argument required")
        summary = regional_summary(args[0])
        if not self.qualified:
            reference = regional_summary(args[0].detach().cpu())
            error = ((summary.cpu() - reference).abs() / reference.abs().clamp_min(1)).max()
            self.maximum_CPU_GPU_scaled_error = float(error)
            if self.maximum_CPU_GPU_scaled_error > 0.0001:
                raise ValueError("Actual first-chunk CPU/GPU regional reduction differs")
            self.qualified = True
        stored, error = regional_storage(summary)
        self.maximum_quantization_scaled_error = max(self.maximum_quantization_scaled_error, error)
        self.values.append(stored)
        return None  # A forward-pre-hook must never replace the original input.

    def end(self, expected_windows=94):
        if not self.active or not self.values:
            raise ValueError("An actual active complete forward is required")
        result = torch.cat(self.values)
        self.values = []
        self.active = False
        if result.shape != (expected_windows, 7, 1024):
            raise ValueError("Original window/chunk membership changed")
        return result
