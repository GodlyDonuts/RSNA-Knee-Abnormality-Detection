"""Native-CoAtNet-Bag12 N3: within-series windows and joint disease attention.

This is a separate full-finetuning architecture, not a modification of the DINO
trainer. Inputs come from the existing native grayscale dataset in [0, 1].
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import Tensor, nn

SAMPLER_VERSION = "native_coatnet_bag12_sha256_torch_cpu_v1"
MODEL_NAME = "coatnet_rmlp_2_rw_384.sw_in12k_ft_in1k"
WEIGHT_SHA256 = "91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e"
TIMM_VERSION = "1.0.30"
Window = tuple[int, int]


def candidate_windows(series_mask: Tensor, slice_mask: Tensor) -> list[Window]:
    """Enumerate only complete triples of valid selected samples in one series."""
    if (
        series_mask.dtype != torch.bool
        or slice_mask.dtype != torch.bool
        or series_mask.ndim != 1
        or slice_mask.ndim != 2
        or series_mask.shape[0] != slice_mask.shape[0]
        or not 1 <= len(series_mask) <= 6
        or not 1 <= slice_mask.shape[1] <= 16
    ):
        raise ValueError("Expected at most six series and sixteen boolean slice slots")
    series_mask, slice_mask = series_mask.cpu(), slice_mask.cpu()
    result: list[Window] = []
    for s in range(len(series_mask)):
        valid = slice_mask[s]
        n = int(valid.sum())
        if not series_mask[s] and n:
            raise ValueError("Inactive series contains valid slices")
        if not torch.equal(valid, torch.arange(len(valid)) < n):
            raise ValueError("Native valid samples must precede padding")
        if series_mask[s]:
            result.extend((s, j) for j in range(max(0, n - 2)))
    if not result:
        raise ValueError("Study has no valid three-sample window")
    return result


def bag_seed(run_seed: int, epoch: int, study_uid: str) -> int:
    """Canonical JSON UTF-8 SHA256, first eight bytes big endian, unsigned63."""
    if (
        type(run_seed) is not int
        or type(epoch) is not int
        or epoch < 0
        or not isinstance(study_uid, str)
        or not study_uid
    ):
        raise ValueError("Explicit integer seed/epoch and nonempty study identity required")
    raw = json.dumps(
        [run_seed, epoch, study_uid, SAMPLER_VERSION], separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") & ((1 << 63) - 1)


def sample_windows(
    series_mask: Tensor, slice_mask: Tensor, *, run_seed: int, epoch: int, study_uid: str
) -> list[Window]:
    """Two per available series, then fill without replacement to twelve.

    The local CPU torch.Generator is independent of global torch/CUDA RNG. Pin
    torch's version in the run binding. Returning canonical order also makes
    full-bag behavior identical when the study contains twelve or fewer windows.
    """
    candidates = candidate_windows(series_mask, slice_mask)
    seed = bag_seed(run_seed, epoch, study_uid)
    if len(candidates) <= 12:
        return candidates
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chosen: set[Window] = set()
    for s in range(len(series_mask)):
        group = [w for w in candidates if w[0] == s]
        order = torch.randperm(len(group), generator=generator).tolist()
        chosen.update(group[i] for i in order[:2])
    remaining = [w for w in candidates if w not in chosen]
    order = torch.randperm(len(remaining), generator=generator).tolist()
    chosen.update(remaining[i] for i in order[: 12 - len(chosen)])
    return [w for w in candidates if w in chosen]


def window_tensor(images: Tensor, windows: list[Window]) -> Tensor:
    """Stack grayscale [S,K,1,H,W] triples and apply exactly (x-.5)/.5."""
    if (
        images.ndim != 5
        or images.shape[2] != 1
        or not images.is_floating_point()
        or not torch.isfinite(images).all()
        or images.min() < 0
        or images.max() > 1
        or not windows
        or len(windows) != len(set(windows))
    ):
        raise ValueError("Unique windows and finite native grayscale values in [0,1] required")
    if any(s < 0 or s >= images.shape[0] or j < 0 or j + 3 > images.shape[1] for s, j in windows):
        raise ValueError("Window crosses selected-series bounds")
    return torch.stack([images[s, j : j + 3, 0] for s, j in windows]).mul(2).sub(1)


def disable_encoder_dropout(encoder: nn.Module) -> list[str]:
    """Keep BN training semantics; disable hidden relative-position dropout too."""
    changed = []
    for name, module in encoder.named_modules():
        if isinstance(module, nn.Dropout):
            if module.p:
                changed.append(name)
            module.p = 0.0
        if getattr(module, "drop_prob", 0) != 0:
            raise ValueError("Unexpected active encoder stochastic depth")
    return changed


def load_encoder(weights: Path) -> nn.Module:
    """Strict original ImageNet classifier load, followed by classifier removal."""
    import timm
    from safetensors.torch import load_file

    if timm.__version__ != TIMM_VERSION:
        raise ValueError("Compact architecture requires pinned timm1.0.30")
    digest = hashlib.sha256()
    with weights.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            digest.update(block)
    if digest.hexdigest() != WEIGHT_SHA256:
        raise ValueError("Compact ImageNet initialization differs from pinned original")
    encoder = timm.create_model(MODEL_NAME, pretrained=False, drop_rate=0.0, drop_path_rate=0.0)
    encoder.load_state_dict(load_file(str(weights)), strict=True)
    encoder.reset_classifier(0)
    paths = disable_encoder_dropout(encoder)
    expected = {
        f"stages.{stage}.blocks.{i}.attn.rel_pos.mlp.drop1"
        for stage, count in ((2, 14), (3, 2))
        for i in range(count)
    }
    if set(paths) != expected or encoder.num_features != 1024:
        raise ValueError("Instantiated encoder differs from the accepted module inventory")
    if sum(p.numel() for p in encoder.parameters()) != 72_856_264:
        raise ValueError("Unexpected compact encoder parameter count")
    if not all(p.requires_grad for p in encoder.parameters()):
        raise ValueError("Compact recipe fully fine-tunes its encoder")
    return encoder


class CompactBagClassifier(nn.Module):
    """One shared encoder; per-finding attention over full-dimensional features."""

    def __init__(self, encoder: nn.Module, dimension: int = 1024):
        super().__init__()
        self.encoder = encoder
        disable_encoder_dropout(encoder)
        self.norm = nn.LayerNorm(dimension)
        self.attention = nn.Sequential(nn.Linear(dimension, 256), nn.Tanh(), nn.Linear(256, 12))
        self.dropout = nn.Dropout(0.2)
        self.classifiers = nn.ModuleList(nn.Linear(dimension, 1) for _ in range(12))

    def pool(self, features: Tensor) -> Tensor:
        if (
            features.ndim != 2
            or not len(features)
            or features.shape[1] != self.norm.normalized_shape[0]
        ):
            raise ValueError("Expected a nonempty matrix of full-dimensional embeddings")
        h = self.norm(features.float())
        weights = self.attention(h).softmax(dim=0)
        pooled = self.dropout(weights.T @ h)
        return torch.cat([head(pooled[i]) for i, head in enumerate(self.classifiers)])

    def forward(self, windows: Tensor, *, chunk_size: int | None = None) -> Tensor:
        if windows.ndim != 4 or windows.shape[1] != 3 or not len(windows):
            raise ValueError("Expected a nonempty window batch [M,3,H,W]")
        if chunk_size is not None and (
            self.training or type(chunk_size) is not int or chunk_size < 1
        ):
            raise ValueError("Window chunking is evaluation-only with saved BN statistics")
        chunks = windows.split(chunk_size) if chunk_size is not None else (windows,)
        embeddings = []
        for chunk in chunks:
            with torch.autocast(windows.device.type, dtype=torch.bfloat16, enabled=windows.is_cuda):
                embeddings.append(self.encoder(chunk))
        # Explicitly defeat an enclosing autocast: attention/pooling/loss are FP32.
        with torch.autocast(windows.device.type, enabled=False):
            return self.pool(torch.cat(embeddings))
