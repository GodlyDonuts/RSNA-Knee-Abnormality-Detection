"""Standalone compute context preserves source inputs, masks, roster and sigmoid semantics."""

from argparse import Namespace

import numpy as np
import pytest
import torch
from torch import nn

from rsna.schema import TARGETS
from scripts.kaggle.fp16_resident_probe import FP16ResidentProbe, predict_scores
from scripts.kaggle.resident_worker import parse_command


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(22)
        self.linear = nn.Linear(4, len(TARGETS))
        self.inputs = []

    def forward(self, images, features, series_mask, slice_mask, positions):
        self.inputs.append(
            (
                images.clone(),
                features.clone(),
                series_mask.clone(),
                slice_mask.clone(),
                positions.clone(),
            )
        )
        return self.linear(images)


def batches():
    return [
        {
            "images": torch.arange(12).reshape(3, 4).float() / 12,
            "features": torch.zeros(3, 5),
            "series_mask": torch.ones(3, 1, dtype=torch.bool),
            "slice_mask": torch.ones(3, 1, 1, dtype=torch.bool),
            "positions": torch.zeros(3, 1, 1, 2),
            "study_uid": ["a", "b", "padding"],
            "sample_valid": torch.tensor([True, True, False]),
        }
    ]


def test_fp16_arithmetic_uses_same_inputs_and_fp32_sigmoid_with_original_roster():
    model = Model()
    before = {k: v.clone() for k, v in model.state_dict().items()}
    reference, _ = predict_scores(
        model, batches(), torch.device("cpu"), precision="fp32", seed=4, fold=0
    )
    candidate, _ = predict_scores(
        model, batches(), torch.device("cpu"), precision="fp16", seed=4, fold=0
    )
    assert candidate.study_uid.tolist() == reference.study_uid.tolist() == ["a", "b"]
    assert list(candidate.columns) == ["study_uid", "fold", "seed", *TARGETS]
    np.testing.assert_allclose(
        candidate[list(TARGETS)], reference[list(TARGETS)], rtol=0, atol=2e-4
    )
    assert all(torch.equal(a, b) for a, b in zip(model.inputs[0], model.inputs[1]))
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())


@pytest.mark.parametrize("bad", ["nan", "duplicate", "mask"])
def test_bad_predictions_are_rejected_without_fallback_or_partial_output(bad):
    model = Model()
    rows = batches()
    if bad == "nan":
        rows[0]["images"][0, 0] = float("nan")
    if bad == "duplicate":
        rows[0]["study_uid"][1] = "a"
    if bad == "mask":
        rows[0]["sample_valid"] = torch.ones(3)
    with pytest.raises(ValueError):
        predict_scores(model, rows, torch.device("cpu"), precision="fp16", seed=4, fold=0)


@pytest.mark.parametrize("device,admit", [("cpu", True), ("cuda", False)])
def test_cuda_probe_cannot_be_implicitly_admitted(device, admit):
    args = Namespace(precision="fp16", device=device)
    with pytest.raises(ValueError, match="Explicit CUDA-only"):
        FP16ResidentProbe(args, admit_performance_probe=admit)


def test_existing_worker_parser_keeps_fp32_contract_unless_explicit_probe():
    command = [
        "python",
        "--checkpoint",
        "c.pt",
        "--series",
        "s.csv",
        "--template",
        "t.csv",
        "--cache",
        "cache",
        "--output",
        "out.csv",
        "--encoder-spec",
        "e.json",
        "--device",
        "cuda",
        "--precision",
        "fp16",
    ]
    with pytest.raises(SystemExit):
        parse_command(command)
    assert parse_command(command, allow_fp16_probe=True).precision == "fp16"
