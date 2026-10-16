"""Full-state compact inference with explicit FP32/FP16 execution for Kaggle T4.

Training remains the frozen BF16 recipe. This separate evaluator restores every
encoder/head/BN tensor; it needs no extra ImageNet checkpoint at deployment.
Precision/runtime qualification still requires actual target-device evidence.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch
from torch import nn

from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.compact_bag import (
    MODEL_NAME,
    SAMPLER_VERSION,
    TIMM_VERSION,
    WEIGHT_SHA256,
    CompactBagClassifier,
    candidate_windows,
    disable_encoder_dropout,
    window_tensor,
)
from rsna.schema import TARGETS


class DeploymentBagClassifier(CompactBagClassifier):
    """Evaluation-only precision policy; attention and pooling stay FP32."""

    def __init__(self, encoder: nn.Module, *, precision: str, dimension: int = 1024):
        if precision not in {"float32", "float16"}:
            raise ValueError("Explicit float32/float16 inference policy required")
        super().__init__(encoder, dimension)
        self.precision = precision
        self.eval()

    def forward(self, windows, *, chunk_size=12):
        if self.training:
            raise ValueError("Deployment precision cannot replace the admitted training recipe")
        if (
            windows.ndim != 4
            or windows.shape[1] != 3
            or not len(windows)
            or windows.dtype != torch.float32
            or type(chunk_size) is not int
            or chunk_size < 1
        ):
            raise ValueError("Finite FP32 window batch and positive inference chunk required")
        if not torch.isfinite(windows).all():
            raise ValueError("Nonfinite MRI windows")
        if self.precision == "float16" and not windows.is_cuda:
            raise ValueError("FP16 policy requires an actual CUDA device")
        features = []
        for chunk in windows.split(chunk_size):
            with torch.autocast(
                windows.device.type,
                dtype=torch.float16,
                enabled=windows.is_cuda and self.precision == "float16",
            ):
                features.append(self.encoder(chunk))
        with torch.autocast(windows.device.type, enabled=False):
            logits = self.pool(torch.cat(features).float())
        if not torch.isfinite(logits).all():
            raise ValueError("Nonfinite compact inference logits")
        return logits


def _blueprint() -> nn.Module:
    import timm

    if timm.__version__ != TIMM_VERSION:
        raise ValueError("Inference must preserve the admitted timm architecture version")
    encoder = timm.create_model(MODEL_NAME, pretrained=False, drop_rate=0.0, drop_path_rate=0.0)
    encoder.reset_classifier(0)
    disabled = disable_encoder_dropout(encoder)
    if len(disabled) != 16 or encoder.num_features != 1024:
        raise ValueError("Inference encoder differs from the actual admitted graph")
    if sum(p.numel() for p in encoder.parameters()) != 72856264:
        raise ValueError("Unexpected compact inference trunk")
    return encoder


def load_model(
    checkpoint: Path,
    expected_sha256: str,
    *,
    precision: str = "float32",
    device: torch.device = torch.device("cpu"),
    allow_diagnostic: bool = False,
) -> tuple[DeploymentBagClassifier, dict]:
    """Restore one complete scheduled epoch; reject diagnostic weights by default."""
    if type(allow_diagnostic) is not bool or file_sha256(checkpoint) != expected_sha256:
        raise ValueError("Exact checkpoint identity and explicit diagnostic policy required")
    if checkpoint.stat().st_size > 512 * 2**20:
        raise ValueError("Deploy the immutable full-model epoch, not an optimizer archive")
    value = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        set(value) != {"schema_version", "binding", "epoch", "optimizer_steps", "model"}
        or value["schema_version"] != 1
    ):
        raise ValueError("Expected an immutable scheduled compact epoch")
    binding = value["binding"]
    diagnostic = binding.get("diagnostic_only")
    if type(diagnostic) is not bool or (diagnostic and not allow_diagnostic):
        raise ValueError("Engineering checkpoint is not competition eligible")
    if (
        binding.get("architecture") != MODEL_NAME
        or binding.get("timm_version") != TIMM_VERSION
        or binding.get("initialization_sha256") != WEIGHT_SHA256
        or binding.get("sampler") != SAMPLER_VERSION
        or binding.get("targets") != list(TARGETS)
        or binding.get("engine") != "native_coatnet_bag12_N3_full_v1"
        or binding.get("native_input")
        != {"side": 384, "slices": 16, "max_series": 6, "mean": [0.5] * 3, "std": [0.5] * 3}
    ):
        raise ValueError("Checkpoint architecture/input/target provenance differs")
    epochs, count, steps_per_epoch = (2, 16, 2) if diagnostic else (16, 2247, 281)
    epoch, steps = value["epoch"], value["optimizer_steps"]
    if (
        type(epoch) is not int
        or not 1 <= epoch <= epochs
        or type(steps) is not int
        or steps != epoch * steps_per_epoch
        or binding.get("fitting_count") != count
        or binding.get("settings", {}).get("epochs") != epochs
        or binding.get("settings", {}).get("effective_batch") != 8
        or binding.get("total_steps") != epochs * steps_per_epoch
    ):
        raise ValueError("Checkpoint schedule or fitted scope differs")
    if any(v.is_floating_point() and v.dtype != torch.float32 for v in value["model"].values()):
        raise ValueError("Original full-state FP32 tensors required before execution qualification")
    model = DeploymentBagClassifier(_blueprint(), precision=precision).float()
    model.train(True)
    model.load_state_dict(value["model"], strict=True)
    model.to(device).eval().requires_grad_(False)
    return model, {
        "checkpoint_sha256": expected_sha256,
        "epoch": epoch,
        "optimizer_steps": steps,
        "diagnostic_only": diagnostic,
        "competition_eligible": not diagnostic,
        "precision": precision,
        "actual_target_runtime_qualified": False,
    }


def prediction_frame(model, data, expected_ids: list[str], *, device, chunk_size=12):
    """Produce the exact test.csv roster in its recorded order, with no label reads."""
    if (
        not expected_ids
        or any(not isinstance(x, str) or not x for x in expected_ids)
        or (len(set(expected_ids)) != len(expected_ids))
    ):
        raise ValueError("Distinct complete expected test roster required")
    seen = {}
    allowed = set(expected_ids)
    model.to(device).eval()
    with torch.inference_mode():
        for sample in data:
            uid = sample["study_uid"]
            if uid not in allowed or uid in seen:
                raise ValueError("Duplicate or unexpected MRI study")
            if sample["images"].dtype != torch.float32:
                raise ValueError("Native FP32 grayscale input required")
            windows = candidate_windows(sample["series_mask"], sample["slice_mask"])
            images = window_tensor(sample["images"], windows).to(device)
            seen[uid] = model(images, chunk_size=chunk_size).sigmoid().float().cpu().tolist()
    if set(seen) != allowed:
        raise ValueError("MRI prediction roster is incomplete")
    return pd.DataFrame(
        [[uid, *seen[uid]] for uid in expected_ids], columns=["StudyInstanceUID", *TARGETS]
    )
