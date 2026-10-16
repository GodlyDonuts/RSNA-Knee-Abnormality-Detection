import copy

import pytest
import torch
from torch import nn

from rsna.competitive import compact_infer as inference
from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.compact_bag import (
    MODEL_NAME,
    SAMPLER_VERSION,
    TIMM_VERSION,
    WEIGHT_SHA256,
    CompactBagClassifier,
)
from rsna.schema import TARGETS


class Encoder(nn.Module):
    def __init__(self, dimension=4):
        super().__init__()
        self.projection = nn.Conv2d(3, dimension, 1)
        self.bn = nn.BatchNorm2d(dimension)

    def forward(self, x):
        return self.bn(self.projection(x)).mean((2, 3))


def test_evaluator_preserves_original_fp32_and_saved_bn():
    torch.manual_seed(2)
    training = CompactBagClassifier(Encoder(), dimension=4).eval()
    deploy = inference.DeploymentBagClassifier(
        copy.deepcopy(training.encoder), precision="float32", dimension=4
    )
    deploy.load_state_dict(training.state_dict())
    x = torch.randn(17, 3, 4, 4)
    before = deploy.encoder.bn.running_mean.clone()
    with torch.inference_mode():
        assert torch.allclose(training(x), deploy(x), atol=1e-6)
        assert torch.allclose(training(x), deploy(x, chunk_size=3), atol=1e-6)
    assert torch.equal(before, deploy.encoder.bn.running_mean)
    deploy.train()
    with pytest.raises(ValueError, match="training recipe"):
        deploy(x)


def test_t4_does_not_silently_use_bfloat16_or_fake_cpu_fp16():
    with pytest.raises(ValueError):
        inference.DeploymentBagClassifier(Encoder(), precision="bfloat16", dimension=4)
    model = inference.DeploymentBagClassifier(Encoder(), precision="float16", dimension=4)
    with pytest.raises(ValueError, match="actual CUDA"):
        model(torch.zeros(2, 3, 4, 4))


def checkpoint(tmp_path, diagnostic):
    model = inference.DeploymentBagClassifier(Encoder(1024), precision="float32")
    b = {
        "architecture": MODEL_NAME,
        "timm_version": TIMM_VERSION,
        "initialization_sha256": WEIGHT_SHA256,
        "sampler": SAMPLER_VERSION,
        "targets": list(TARGETS),
        "engine": "native_coatnet_bag12_N3_full_v1",
        "native_input": {
            "side": 384,
            "slices": 16,
            "max_series": 6,
            "mean": [0.5] * 3,
            "std": [0.5] * 3,
        },
        "diagnostic_only": diagnostic,
        "fitting_count": 16 if diagnostic else 2247,
        "settings": {"epochs": 2 if diagnostic else 16, "effective_batch": 8},
        "total_steps": 4 if diagnostic else 4496,
    }
    value = {
        "schema_version": 1,
        "binding": b,
        "epoch": 2 if diagnostic else 16,
        "optimizer_steps": 4 if diagnostic else 4496,
        "model": model.state_dict(),
    }
    p = tmp_path / "synthetic-full-epoch.pt"
    torch.save(value, p)
    return p, value


def test_complete_full_state_restoration_needs_no_imagenet_file(tmp_path, monkeypatch):
    p, value = checkpoint(tmp_path, False)
    monkeypatch.setattr(inference, "_blueprint", lambda: Encoder(1024))
    model, metadata = inference.load_model(p, file_sha256(p))
    assert metadata["competition_eligible"] and not metadata["actual_target_runtime_qualified"]
    assert all(not q.requires_grad for q in model.parameters())
    assert all(torch.equal(v, model.state_dict()[k]) for k, v in value["model"].items())
    with pytest.raises(ValueError, match="identity"):
        inference.load_model(p, "0" * 64)


def test_diagnostic_model_rejected_by_default_and_explicit_probe_stays_ineligible(
    tmp_path, monkeypatch
):
    p, _ = checkpoint(tmp_path, True)
    monkeypatch.setattr(inference, "_blueprint", lambda: Encoder(1024))
    with pytest.raises(ValueError, match="not competition eligible"):
        inference.load_model(p, file_sha256(p))
    _, metadata = inference.load_model(p, file_sha256(p), allow_diagnostic=True)
    assert not metadata["competition_eligible"]


@pytest.mark.parametrize("field", ["epoch", "optimizer_steps", "targets", "model"])
def test_scope_head_and_target_drift_fail_closed(tmp_path, monkeypatch, field):
    p, value = checkpoint(tmp_path, False)
    monkeypatch.setattr(inference, "_blueprint", lambda: Encoder(1024))
    if field == "epoch":
        value["epoch"] = 15
    elif field == "optimizer_steps":
        value["optimizer_steps"] = 4
    elif field == "targets":
        value["binding"]["targets"] = list(reversed(TARGETS))
    else:
        value["model"].pop("classifiers.0.bias")
    torch.save(value, p)
    with pytest.raises((ValueError, RuntimeError)):
        inference.load_model(p, file_sha256(p))


def sample(uid):
    return {
        "study_uid": uid,
        "images": torch.zeros(1, 4, 1, 4, 4),
        "series_mask": torch.ones(1, dtype=torch.bool),
        "slice_mask": torch.ones(1, 4, dtype=torch.bool),
    }


def test_test_csv_roster_order_is_exact_and_labels_are_unneeded():
    model = inference.DeploymentBagClassifier(Encoder(), precision="float32", dimension=4)
    rows = inference.prediction_frame(
        model, [sample("b"), sample("a")], ["a", "b"], device=torch.device("cpu")
    )
    assert list(rows) == ["StudyInstanceUID", *TARGETS]
    assert rows.StudyInstanceUID.tolist() == ["a", "b"]
    assert rows[list(TARGETS)].ge(0).all().all() and rows[list(TARGETS)].le(1).all().all()
    for data in [[sample("a")], [sample("a"), sample("a")], [sample("x"), sample("a")]]:
        with pytest.raises(ValueError):
            inference.prediction_frame(model, data, ["a", "b"], device=torch.device("cpu"))
