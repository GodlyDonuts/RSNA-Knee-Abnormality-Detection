import copy
import hashlib
import json

import pytest
import torch
from torch import nn

from rsna.competitive import compact_average_infer as module
from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.compact_average import BASE_ENGINE, ENGINE, POLICY
from rsna.competitive.compact_bag import MODEL_NAME, SAMPLER_VERSION, TIMM_VERSION, WEIGHT_SHA256
from rsna.competitive.compact_infer import DeploymentBagClassifier
from rsna.competitive.compact_infer import load_model as original_loader
from rsna.schema import TARGETS


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 1024, 1)
        self.bn = nn.BatchNorm2d(1024)

    def forward(self, x):
        return self.bn(self.conv(x)).mean((2, 3))


def fixture():
    model = DeploymentBagClassifier(Encoder(), precision="float32")
    source = ["a" * 64, "b" * 64, "c" * 64]
    binding = {
        "engine": ENGINE,
        "architecture": MODEL_NAME,
        "timm_version": TIMM_VERSION,
        "initialization_sha256": WEIGHT_SHA256,
        "sampler": SAMPLER_VERSION,
        "targets": list(TARGETS),
        "fitting_count": 2247,
        "total_steps": 4496,
        "settings": {"epochs": 16, "effective_batch": 8},
        "diagnostic_only": False,
        "native_input": {
            "side": 384,
            "slices": 16,
            "max_series": 6,
            "mean": [0.5] * 3,
            "std": [0.5] * 3,
        },
        "weight_average": {
            "policy": copy.deepcopy(POLICY),
            "source_engine": BASE_ENGINE,
            "source_checkpoint_sha256": source,
            "derived_not_original_scheduled_epoch": True,
            "learned_parameter_names_sha256": hashlib.sha256(
                json.dumps(sorted(dict(model.named_parameters()))).encode()
            ).hexdigest(),
        },
    }
    return {
        "schema_version": 1,
        "binding": binding,
        "epoch": 16,
        "optimizer_steps": 4496,
        "model": model.state_dict(),
    }, source


def test_distinct_loader_restores_full_state_and_preserves_original_rejection(
    tmp_path, monkeypatch
):
    value, source = fixture()
    path = tmp_path / "derived.pt"
    torch.save(value, path)
    monkeypatch.setattr(module, "_blueprint", Encoder)
    model, scope = module.load_model(path, file_sha256(path), source)
    assert (
        not model.training
        and scope["competition_eligible"]
        and not scope["actual_target_runtime_qualified"]
    )
    assert all(torch.equal(t, model.state_dict()[n]) for n, t in value["model"].items())
    with pytest.raises(ValueError):
        original_loader(path, file_sha256(path))


@pytest.mark.parametrize(
    "change", ["policy", "source_order", "diagnostic", "loss", "engine", "count", "steps", "nan"]
)
def test_invalid_derivation_or_scope_refused(change):
    value, source = fixture()
    if change == "policy":
        value["binding"]["weight_average"]["policy"]["all_buffers"] = "average"
    elif change == "source_order":
        source = source[::-1]
    elif change == "diagnostic":
        value["binding"]["diagnostic_only"] = True
    elif change == "loss":
        value["binding"]["loss_policy"] = {"weighted": True}
    elif change == "engine":
        value["binding"]["engine"] = BASE_ENGINE
    elif change == "count":
        value["binding"]["fitting_count"] = 2592
    elif change == "steps":
        value["optimizer_steps"] = 4000
    else:
        value["model"]["encoder.conv.weight"].fill_(float("nan"))
    with pytest.raises(ValueError):
        module.validate_derived(value, source)
