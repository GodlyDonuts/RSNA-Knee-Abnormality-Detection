import copy
import json
import random

import numpy as np
import pytest
import torch
from torch import nn

from rsna.competitive.adapters import ResidualLinear
from rsna.competitive.mrnet import MRNetAdapterNet
from rsna.competitive.mrnet_training import MRNetRecipe, fit_source, predict_source
from scripts.newton.probe_mrnet_adapter_recovery import same


class Exams:
    def __init__(self, split="train", n=9):
        self.split = split
        self.training_permitted = split == "train"
        self.uids = tuple(f"{i:04d}" for i in range(n))

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, index):
        return {
            "source_split": self.split,
            "images": torch.full((3, 2, 1, 16, 16), (index + 1) / (len(self) + 1)),
            "slice_mask": torch.ones(3, 2, dtype=torch.bool),
            "ordinal": torch.tensor([0.0, 1.0]).expand(3, 2),
            "labels": torch.tensor([index % 2, (index + 1) % 2, 1.0]),
        }


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = ResidualLinear(nn.Linear(1, 8), rank=2, alpha=2, dropout=0.1)

    def forward(self, images):
        return self.linear(images.mean((2, 3)))


PROVENANCE = {"pretrained": True, "config_sha256": "a" * 64, "weights_sha256": "b" * 64}
BINDING = {"private_research_only": True, "external_weight_publication": False, "source": "toy"}


def model():
    random.seed(83)
    np.random.seed(83)
    torch.manual_seed(83)
    return MRNetAdapterNet(Encoder(), 8, hidden_dim=12, chunk=3)


def test_all_exams_once_each_epoch_and_small_final_batch_not_dropped(tmp_path):
    net = model()
    base = net.encoder.linear.weight.detach().clone()
    result = fit_source(
        net,
        Exams(),
        PROVENANCE,
        BINDING,
        MRNetRecipe(epochs=2),
        tmp_path / "run",
        torch.device("cpu"),
    )
    assert result["step"] == 4
    assert [row["exams"] for row in result["updates"]] == [8, 1, 8, 1]
    for epoch in (1, 2):
        cases = [
            uid for row in result["updates"] if row["epoch"] == epoch for uid in row["source_cases"]
        ]
        assert sorted(cases) == list(Exams().uids)
        assert (tmp_path / "run" / f"epoch-{epoch:03d}.pt").is_file()
        receipt = json.loads((tmp_path / "run" / f"epoch-{epoch:03d}.json").read_text())
        assert receipt["binding"]["source_target_order"] == ["acl", "meniscus", "abnormal"]
    assert torch.equal(net.encoder.linear.weight, base)
    assert net.encoder.linear.lora_B.abs().sum() > 0


def test_interrupt_mid_epoch_restores_complete_state_order_dropout_and_predictions(tmp_path):
    recipe = MRNetRecipe(epochs=2)
    full_net, full = model(), tmp_path / "full"
    fit_source(full_net, Exams(), PROVENANCE, BINDING, recipe, full, torch.device("cpu"))
    expected = torch.load(full / "last.pt", weights_only=True)
    expected_prediction = predict_source(full_net, Exams("valid", 3), torch.device("cpu"))
    partial = tmp_path / "partial"
    stopped = fit_source(
        model(),
        Exams(),
        PROVENANCE,
        BINDING,
        recipe,
        partial,
        torch.device("cpu"),
        stop_after=1,
    )
    assert stopped["status"] == "checkpointed_for_resume"
    replay = model()  # Construct a fresh frozen base, adapter, head and optimizer.
    fit_source(
        replay, Exams(), PROVENANCE, BINDING, recipe, partial, torch.device("cpu"), resume=True
    )
    assert same(expected, torch.load(partial / "last.pt", weights_only=True))
    assert expected_prediction == predict_source(replay, Exams("valid", 3), torch.device("cpu"))
    with pytest.raises(FileExistsError):
        fit_source(model(), Exams(), PROVENANCE, BINDING, recipe, partial, torch.device("cpu"))


@pytest.mark.parametrize("kind", ["roster", "recipe", "scope"])
def test_recovery_rejects_changed_membership_recipe_or_publication_scope(tmp_path, kind):
    recipe, output = MRNetRecipe(epochs=2), tmp_path / "run"
    fit_source(
        model(), Exams(), PROVENANCE, BINDING, recipe, output, torch.device("cpu"), stop_after=1
    )
    exams, binding = Exams(), copy.deepcopy(BINDING)
    if kind == "roster":
        exams.uids = tuple(reversed(exams.uids))
    elif kind == "recipe":
        recipe = MRNetRecipe(epochs=2, encoder_lr=0.001)
    else:
        binding["external_weight_publication"] = True
    with pytest.raises(ValueError):
        fit_source(
            model(), exams, PROVENANCE, binding, recipe, output, torch.device("cpu"), resume=True
        )


def test_source_validation_never_enters_optimizer(tmp_path):
    with pytest.raises(ValueError, match="source training"):
        fit_source(
            model(),
            Exams("valid"),
            PROVENANCE,
            BINDING,
            MRNetRecipe(epochs=2),
            tmp_path / "run",
            torch.device("cpu"),
        )
    assert not (tmp_path / "run").exists()
