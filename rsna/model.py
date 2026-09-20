from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score
from torch import Tensor, nn
from torch.nn import functional as F

from rsna.schema import TARGETS


class StudyNet(nn.Module):
    def __init__(self, width: int = 32):
        super().__init__()
        if width < 8 or width % 8:
            raise ValueError("Width must be a positive multiple of 8")
        blocks = []
        previous = 1
        for channels in (width, width * 2, width * 4):
            blocks.extend(
                [
                    nn.Conv2d(previous, channels, 3, stride=2, padding=1, bias=False),
                    nn.GroupNorm(8, channels),
                    nn.SiLU(),
                    nn.Conv2d(channels, channels, 3, padding=1, bias=False),
                    nn.GroupNorm(8, channels),
                    nn.SiLU(),
                ]
            )
            previous = channels
        self.encoder = nn.Sequential(*blocks, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.slice_attention = nn.Linear(previous, 1)
        self.sequence_embedding = nn.Linear(5, previous)
        self.series_attention = nn.Linear(previous, len(TARGETS))
        self.classifier = nn.Linear(previous, len(TARGETS))

    def forward(self, images: Tensor, features: Tensor, series_mask: Tensor) -> Tensor:
        if not series_mask.any(dim=1).all():
            raise ValueError("Every study needs at least one valid series")
        batch, series, slices, channels, height, width = images.shape
        encoded = self.encoder(images.reshape(-1, channels, height, width))
        encoded = encoded.reshape(batch, series, slices, -1)
        slice_weights = self.slice_attention(encoded).softmax(dim=2)
        pooled = (encoded * slice_weights).sum(dim=2) + self.sequence_embedding(features)
        attention = self.series_attention(pooled).masked_fill(
            ~series_mask[..., None], float("-inf")
        )
        weights = attention.softmax(dim=1)
        return (weights * self.classifier(pooled)).sum(dim=1)


def masked_bce(logits: Tensor, labels: Tensor, mask: Tensor) -> Tensor:
    if not (mask >= 0).all() or not ((labels >= 0) & (labels <= 1)).all():
        raise ValueError("Invalid label or supervision weight")
    observed = mask.sum(dim=0)
    losses = (F.binary_cross_entropy_with_logits(logits, labels, reduction="none") * mask).sum(
        dim=0
    )
    return (
        (losses / observed.clamp_min(1))[observed > 0].mean()
        if (observed > 0).any()
        else logits.sum() * 0
    )


def auc_metrics(labels: np.ndarray, scores: np.ndarray, mask: np.ndarray) -> dict:
    if (
        labels.shape != scores.shape
        or labels.shape != mask.shape
        or labels.shape[1] != len(TARGETS)
    ):
        raise ValueError("Expected aligned study-by-target arrays")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("Invalid probabilities")
    per_target: dict[str, float | None] = {}
    counts = {}
    for i, target in enumerate(TARGETS):
        keep = mask[:, i] > 0
        actual = labels[keep, i]
        counts[target] = int(keep.sum())
        per_target[target] = (
            float(roc_auc_score(actual, scores[keep, i])) if len(np.unique(actual)) == 2 else None
        )
    valid = [score for score in per_target.values() if score is not None]
    return {
        "per_target_auc": per_target,
        "observed_studies": counts,
        "macro_auc": float(np.mean(valid)) if len(valid) == len(TARGETS) else None,
        "macro_auc_available_targets": float(np.mean(valid)) if valid else None,
        "n_scored_targets": len(valid),
    }
