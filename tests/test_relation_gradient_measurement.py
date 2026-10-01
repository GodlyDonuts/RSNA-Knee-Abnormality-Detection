import hashlib
import io
import json
import tarfile
from dataclasses import replace
from pathlib import Path

import pandas as pd
import pytest
import torch
from torch import nn

from rsna.competitive.adapters import ResidualLinear
from rsna.competitive.checkpoint import capture_rng, restore_rng
from rsna.competitive.data import collate_studies
from rsna.competitive.loss import effective_batch_loss, observed_counts
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.relations import relation_model_kwargs
from rsna.competitive.train import TrainConfig, _model_inputs
from scripts.newton.audit_relation_geometry import signature
from scripts.newton.measure_relation_gradients import (
    gradient_summary,
    matched_configs,
    measure,
    validate_cache,
)

POLICY = dict(weight=0.05, max_pairs=6, target_mode="physical", hidden_dim=8, head_seed=11)


class Encoder(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.linear = ResidualLinear(nn.Linear(1, 8), rank=2, alpha=2, dropout=dropout)

    def forward(self, images):
        return self.linear(images.mean((2, 3)))


def fixture(dropout=0):
    torch.manual_seed(81)
    model = CompetitiveStudyNet(Encoder(dropout), 8, hidden_dim=8, **relation_model_kwargs(POLICY))
    rows = []
    for index in range(3):
        rows.append(
            dict(
                images=torch.rand(3, 4, 1, 16, 16),
                positions=torch.tensor([[-0.13, 1], [-0.05, 1], [0.02, 1], [0.19, 1]])
                .repeat(3, 1, 1)
                .mul(0 if index == 2 else 1),
                features=torch.eye(3, 5),
                series_mask=torch.ones(3, dtype=torch.bool),
                slice_mask=torch.ones(3, 4, dtype=torch.bool),
                sample_valid=torch.tensor(index != 1),
                study_uid=f"training-{index}",
                labels=torch.randint(2, (12,)).float(),
                label_mask=torch.tensor([index != 1] * 7 + [False] * 5).float(),
            )
        )
    config = TrainConfig(precision="fp32", relation=POLICY)
    return model, rows, config


def test_accumulated_encoder_scale_matches_direct_classification_backward():
    model, rows, config = fixture()
    measured = measure(model, [collate_studies([row]) for row in rows], config, torch.device("cpu"))
    batch = collate_studies(rows)
    model.train()
    logits = model(*_model_inputs(batch, torch.device("cpu")))
    loss = effective_batch_loss(
        logits, batch["labels"], batch["label_mask"], observed_counts([batch["label_mask"]])
    )
    parameters = [p for p in model.encoder.parameters() if p.requires_grad]
    expected = torch.autograd.grad(loss, parameters)
    expected_norm = sum(grad.double().square().sum() for grad in expected).sqrt().item()
    assert measured["classification_loss"] == pytest.approx(loss.item(), rel=1e-6)
    assert measured["encoder_gradients"]["classification_norm"] == pytest.approx(
        expected_norm, rel=1e-5
    )
    assert measured["observed_target_counts"] == [2.0] * 7 + [0.0] * 5
    assert measured["represented_studies"] == 1  # Invalid tail and unknown geometry excluded.
    assert measured["relation_pairs"] == 18
    assert all(p.grad is None for p in model.parameters())


def test_controls_replay_dropout_without_mutating_models_or_consuming_pair_rng():
    model, rows, config = fixture(dropout=0.2)
    device = torch.device("cpu")
    initial = {name: value.clone() for name, value in model.state_dict().items()}
    rng = capture_rng(device)
    results = {}
    for name in ("zero", "physical", "shuffled"):
        restore_rng(rng, device)
        policy = {
            **POLICY,
            "weight": 0 if name == "zero" else 0.05,
            "target_mode": "shuffled" if name == "shuffled" else "physical",
        }
        results[name] = measure(
            model,
            [collate_studies([row]) for row in rows],
            replace(config, relation=policy),
            device,
        )
    for value in results.values():
        for key in (
            "classification_gradient_sha256",
            "classification_logits_sha256",
            "pair_roster_sha256",
        ):
            assert value[key] == results["zero"][key]
    assert (
        results["physical"]["target_roster_sha256"] != results["shuffled"]["target_roster_sha256"]
    )
    assert results["zero"]["encoder_gradients"]["weighted_relation_norm"] == 0
    assert all(torch.equal(initial[name], value) for name, value in model.state_dict().items())
    assert all(p.grad is None for p in model.parameters())


def test_opposing_objectives_report_cancellation_instead_of_nan_or_a_clinical_claim():
    first = [torch.tensor([3.0, 4.0])]
    opposite = [torch.tensor([-6.0, -8.0])]
    result = gradient_summary(first, opposite, 0.5, 1)
    assert result["classification_norm"] == 5
    assert result["unweighted_relation_norm"] == 10
    assert result["unweighted_gradient_cosine"] == -1
    assert result["weighted_relation_to_classification_ratio"] == 1
    assert result["combined_norm"] == 0
    assert result["encoder_only_clip_multiplier"] == 1
    empty = gradient_summary([torch.zeros(2)], [torch.ones(2)], 0, 1)
    assert empty["weighted_relation_to_classification_ratio"] is None
    assert empty["unweighted_gradient_cosine"] is None
    json.dumps(empty, allow_nan=False)


@pytest.mark.parametrize("fault", ["nonfinite", "shape", "weight", "clip", "empty"])
def test_gradient_statistics_refuse_invalid_measurements(fault):
    first, second, weight, clip = [torch.ones(2)], [torch.ones(2)], 0.05, 1
    if fault == "nonfinite":
        second[0][0] = float("nan")
    elif fault == "shape":
        second = [torch.ones(3)]
    elif fault == "weight":
        weight = float("inf")
    elif fault == "clip":
        clip = 0
    else:
        first = []
    with pytest.raises(ValueError):
        gradient_summary(first, second, weight, clip)


@pytest.mark.parametrize(
    "fault", [None, "pixels", "replacement_envelope", "archive", "duplicate", "foreign"]
)
def test_local_images_and_geometry_are_bound_to_original_whole_byte_acceptance(tmp_path, fault):
    cache = tmp_path / "cache"
    path = cache / "train_series/training/series.json"
    path.parent.mkdir(parents=True)
    original = b"original bytes already accepted by whole-archive verification"
    metadata = {
        "cache_path": "train_series/training/series.npz",
        "cache_bytes": len(original),
        "cache_sha256": hashlib.sha256(original).hexdigest(),
    }
    raw = json.dumps(metadata).encode()
    path.write_bytes(raw)
    path.with_suffix(".npz").write_bytes(original)
    archive = tmp_path / "original.tar"
    with tarfile.open(archive, "w:") as handle:
        member = tarfile.TarInfo("cache/train_series/training/series.json")
        member.size = len(raw)
        handle.addfile(member, io.BytesIO(raw))
    proof = {
        "assets": {
            str(archive): {
                "stat": signature(archive),
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            }
        }
    }
    table = pd.DataFrame([dict(StudyInstanceUID="training", SeriesInstanceUID="series")])
    if fault == "pixels":
        path.with_suffix(".npz").write_bytes(b"x" * len(original))
    elif fault == "replacement_envelope":
        replacement = b"different source with a self-consistent envelope"
        path.with_suffix(".npz").write_bytes(replacement)
        path.write_text(
            json.dumps(
                {
                    **metadata,
                    "cache_bytes": len(replacement),
                    "cache_sha256": hashlib.sha256(replacement).hexdigest(),
                }
            )
        )
    elif fault == "archive":
        archive.write_bytes(archive.read_bytes() + b"changed")
    elif fault == "duplicate":
        table = pd.concat([table, table])
    elif fault == "foreign":
        table["StudyInstanceUID"] = "outer"
    if fault is None:
        result = validate_cache(cache, archive, proof, ["training"], table)
        assert result["training_only_studies"] == 1 and len(result["members"]) == 2
        assert result["members"][metadata["cache_path"]]["sha256"] == metadata["cache_sha256"]
    else:
        with pytest.raises(ValueError):
            validate_cache(cache, archive, proof, ["training"], table)


@pytest.mark.parametrize(
    "fault", [None, "side", "sampling", "encoder", "missing_head", "unequal_head"]
)
def test_full_input_config_gate_keeps_matched_scientific_controls(tmp_path, fault):
    source = (
        Path(__file__).resolve().parents[1]
        / "initiative/experiments/competitive-v1/slice-relations-v1"
    )
    for name in ("zero", "physical", "shuffled"):
        value = json.loads((source / f"{name}-pilot-config.json").read_text())
        if fault == "side":
            value["side"] = 32
        elif fault == "sampling":
            value["slice_sampling"] = "stratified"
        elif fault == "encoder":
            value["encoder"]["architecture"] = "dinov2_b14_reg"
        elif fault == "missing_head":
            value["relation"] = None
        elif fault == "unequal_head" and name == "shuffled":
            value["relation"]["head_seed"] += 1
        (tmp_path / f"{name}-pilot-config.json").write_text(json.dumps(value))
    if fault is None:
        assert set(matched_configs(tmp_path)) == {"zero", "physical", "shuffled"}
    else:
        with pytest.raises(ValueError):
            matched_configs(tmp_path)
