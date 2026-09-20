#!/usr/bin/env python3
"""CPU-only structural tests for the DINO + physical-ALiBi hierarchy."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

try:
    from .constants import CACHE_SCHEMA_VERSION
    from .data import FeatureStudyDataset, collate_studies
    from .model import KneeAlibiModel, KneeModelConfig, PhysicalAlibiAttention
except ImportError:
    from constants import CACHE_SCHEMA_VERSION
    from data import FeatureStudyDataset, collate_studies
    from model import KneeAlibiModel, KneeModelConfig, PhysicalAlibiAttention


def synthetic_batch() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(41)
    batch, series, slices, feature_dim = 4, 3, 6, 8
    features = torch.randn(batch, series, slices, feature_dim, generator=generator)
    base = torch.tensor([0.0, 1.2, 2.4, 3.6, 4.8, 6.0])
    positions = base[None, None, :].expand(batch, series, -1).clone()
    positions[:, 1] *= 1.3
    slice_mask = torch.ones(batch, series, slices, dtype=torch.bool)
    slice_mask[0, 2, 4:] = False
    series_mask = slice_mask.any(dim=-1)
    return {
        "features": features,
        "positions_mm": positions,
        "slice_mask": slice_mask,
        "series_mask": series_mask,
        "plane": torch.tensor([[1, 2, 3]]).expand(batch, -1).clone(),
        "fluid": torch.tensor([[2, 2, 1]]).expand(batch, -1).clone(),
        "fatsat": torch.tensor([[2, 1, 1]]).expand(batch, -1).clone(),
    }


def assert_close(a: torch.Tensor, b: torch.Tensor, tolerance: float, message: str) -> float:
    error = float((a - b).abs().max())
    if error > tolerance:
        raise AssertionError(f"{message}: max error {error:.3e} > {tolerance:.3e}")
    return error


def main() -> None:
    torch.manual_seed(7)
    config = KneeModelConfig(
        feature_dim=8,
        hidden_dim=16,
        n_heads=4,
        series_depth=1,
        study_depth=1,
        dropout=0.0,
        aggregator="physical_alibi",
        num_targets=3,
    )
    model = KneeAlibiModel(config).eval()
    batch = synthetic_batch()
    with torch.no_grad():
        reference = model(**batch)

    permutation = torch.tensor([4, 0, 5, 2, 1, 3])
    permuted = dict(batch)
    for key in ("features", "positions_mm", "slice_mask"):
        permuted[key] = batch[key].index_select(2, permutation)
    with torch.no_grad():
        permuted_output = model(**permuted)
    permutation_error = assert_close(
        reference, permuted_output, 2e-5, "physical attention changed under joint slice permutation"
    )

    padded = dict(batch)
    padded["features"] = F.pad(batch["features"], (0, 0, 0, 3))
    padded["positions_mm"] = F.pad(batch["positions_mm"], (0, 3), value=999.0)
    padded["slice_mask"] = F.pad(batch["slice_mask"], (0, 3), value=False)
    with torch.no_grad():
        padded_output = model(**padded)
    padding_error = assert_close(
        reference, padded_output, 2e-5, "masked padding changed model output"
    )

    attention = PhysicalAlibiAttention(16, 4, distance_mode="physical_alibi")
    positions_regular = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    positions_gap = torch.tensor([[0.0, 1.0, 5.0, 6.0]])
    mask = torch.ones_like(positions_regular, dtype=torch.bool)
    regular_bias = attention.attention_bias(positions_regular, mask, has_cls=False)
    gap_bias = attention.attention_bias(positions_gap, mask, has_cls=False)
    geometry_delta = float((regular_bias - gap_bias).abs().max())
    if geometry_delta < 0.1:
        raise AssertionError("physical ALiBi did not respond to an irregular physical gap")

    # Confirm the tensor-only cache schema and dynamic collator agree.
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        rows = []
        for index in range(2):
            item = {
                "schema_version": torch.tensor(CACHE_SCHEMA_VERSION),
                "features": batch["features"][index, : 2 + index].half(),
                "positions_mm": batch["positions_mm"][index, : 2 + index],
                "slice_mask": batch["slice_mask"][index, : 2 + index],
                "series_mask": batch["series_mask"][index, : 2 + index],
                "plane": batch["plane"][index, : 2 + index],
                "fluid": batch["fluid"][index, : 2 + index],
                "fatsat": batch["fatsat"][index, : 2 + index],
            }
            path = root / f"study{index}.pt"
            torch.save(item, path)
            rows.append(
                {
                    "StudyInstanceUID": f"study{index}",
                    "cache_file": str(path),
                    "ACL": index,
                    "MCL": 1 - index,
                    "Medial Meniscus": index,
                }
            )
        dataset = FeatureStudyDataset(pd.DataFrame(rows), targets=["ACL", "MCL", "Medial Meniscus"])
        collated = collate_studies([dataset[0], dataset[1]])
        if tuple(collated["features"].shape) != (2, 3, 6, 8):
            raise AssertionError(f"unexpected collated shape {tuple(collated['features'].shape)}")

    # Tiny deterministic overfit: validates gradients across both hierarchy levels.
    train_model = KneeAlibiModel(config).train()
    labels = torch.tensor(
        [[0.0, 1.0, 0.0], [1.0, 0.0, 1.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    )
    optimizer = torch.optim.Adam(train_model.parameters(), lr=0.02)
    losses = []
    for _ in range(30):
        optimizer.zero_grad(set_to_none=True)
        logits = train_model(**batch)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))
    if not losses[-1] < 0.55 * losses[0]:
        raise AssertionError(f"tiny overfit did not converge: {losses[0]:.4f} -> {losses[-1]:.4f}")

    print(
        json.dumps(
            {
                "status": "pass",
                "permutation_max_error": permutation_error,
                "padding_max_error": padding_error,
                "geometry_bias_delta": geometry_delta,
                "overfit_loss_initial": losses[0],
                "overfit_loss_final": losses[-1],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
