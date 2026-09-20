#!/usr/bin/env python3
"""CPU smoke test for packed-study target-wise MIL."""

from __future__ import annotations

import torch
from torch import nn

try:
    from .raw_mil import RawStudyMILModel, collate_raw_studies
except ImportError:
    from raw_mil import RawStudyMILModel, collate_raw_studies


class TinyBackbone(nn.Module):
    output_dim = 8

    def forward(self, pixels: torch.Tensor) -> torch.Tensor:
        value = pixels.mean(dim=(2, 3))
        return torch.cat([value, value[:, :2], value], dim=1)[:, : self.output_dim]


def item(uid: str, slices: int) -> dict[str, object]:
    return {
        "uid": uid,
        "pixels": torch.rand(slices, 3, 16, 16),
        "plane": torch.arange(slices) % 3,
        "fluid": torch.arange(slices) % 3,
        "fatsat": (torch.arange(slices) + 1) % 3,
        "labels": torch.zeros(12),
        "label_mask": torch.ones(12, dtype=torch.bool),
        "confidence": torch.ones(12),
        "gold_mask": torch.ones(12, dtype=torch.bool),
    }


def main() -> None:
    batch = collate_raw_studies([item("a", 4), item("b", 7)])
    model = RawStudyMILModel(
        TinyBackbone(),
        12,
        hidden_dim=16,
        encoder_batch_size=3,
        pool="topk",
        topk=2,
        report_dim=10,
    )
    output = model(
        batch["pixels"],
        batch["plane"],
        batch["fluid"],
        batch["fatsat"],
        batch["study_index"],
        batch["num_studies"],
        return_aux=True,
    )
    assert output["logits"].shape == (2, 12)
    assert output["report_embedding"].shape == (2, 10)
    (output["logits"].sum() + output["report_embedding"].sum()).backward()
    assert all(parameter.grad is not None for parameter in model.head.parameters())
    print("raw MIL smoke test passed")


if __name__ == "__main__":
    main()
