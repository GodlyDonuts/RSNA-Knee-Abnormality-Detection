"""Whole-window-bag arms on the canonical trainer: full / stop1 / resume, engineering only."""

import io
import json

import numpy as np
import pandas as pd
import pytest
import torch
from test_public_triplet_training import POLICY
from test_source_matched_slot import (
    CPU,
    ENGINEERING,
    OFFICIAL,
    assert_same,
    cohort,  # noqa: F401  (module fixture)
    fake_create_encoder,
    numerical,
    run_trainer,
)
from test_window_bag import RECIPE

from rsna.competitive import public_input as pi
from rsna.competitive import window_bag as wb
from rsna.competitive.adapters import configure_encoder_training
from rsna.competitive.data import collate_studies
from rsna.competitive.delta_checkpoint import FrozenBaseDelta
from rsna.competitive.train import TrainConfig
from rsna.schema import TARGETS


def engineering(arm):
    values = dict(json.loads((RECIPE / f"{arm}-r32-config.json").read_text()), **ENGINEERING)
    values["fitting_supervision"] = None
    return values


def trainer(cohort, values, output, record=None, **kwargs):  # noqa: F811
    from rsna.competitive import train

    patch = pytest.MonkeyPatch()
    if record is not None:
        original = train.window_bag_training_inputs

        def recording(batch, device, strategy, *, seed, epoch, update):
            inputs = original(batch, device, strategy, seed=seed, epoch=epoch, update=update)
            record.append((epoch, update, inputs[0].shape, inputs[3].shape))
            return inputs

        patch.setattr(train, "window_bag_training_inputs", recording)
    try:
        return run_trainer(cohort, values, output, **kwargs)
    finally:
        patch.undo()


@pytest.fixture(scope="module", params=["mean", "query"])
def recovered(request, cohort):  # noqa: F811
    arm = request.param
    root = cohort["base"] / f"bag-{arm}"
    values = engineering(arm)
    record: list = []
    assert trainer(cohort, values, root / "full", record=record) == 0
    assert trainer(cohort, values, root / "replay", stop=1) == 75
    partial = torch.load(root / "replay/checkpoint.pt", weights_only=True)
    first = json.loads((root / "replay/invocation.json").read_text())
    assert trainer(cohort, values, root / "replay", resume=root / "replay/checkpoint.pt") == 0
    return dict(arm=arm, root=root, values=values, record=record, partial=partial, first=first)


def test_full_and_stop1_resume_have_identical_complete_state_and_predictions(recovered):
    root = recovered["root"]
    full = torch.load(root / "full/checkpoint.pt", weights_only=True)
    replay = torch.load(root / "replay/checkpoint.pt", weights_only=True)
    assert (full["epoch"], full["next_update"], full["step"]) == (2, 0, 4)
    assert (recovered["partial"]["next_update"], recovered["partial"]["step"]) == (1, 1)
    assert full["model_storage"]["format"] == "frozen_base_delta_v1"
    for key in (
        "model",
        "model_storage",
        "optimizer",
        "amp_scaler",
        "rng_by_rank",
        "binding",
        "epoch",
        "next_update",
        "step",
    ):
        assert_same(full[key], replay[key], key)
    assert [g["lr"] for g in full["optimizer"]["param_groups"]] == [
        g["lr"] for g in replay["optimizer"]["param_groups"]
    ]
    assert set(full["rng_by_rank"][0]) == {"torch", "cuda", "python", "numpy"}
    assert (root / "full/outer_predictions.csv").read_bytes() == (
        root / "replay/outer_predictions.csv"
    ).read_bytes()
    full_history = json.loads((root / "full/invocation.json").read_text())["updates"]
    resumed = json.loads((root / "replay/invocation.json").read_text())["updates"]
    assert numerical(full_history) == numerical(recovered["first"]["updates"] + resumed)
    assert [event["optimizer_steps"] for event in full_history] == [1, 2, 3, 4]
    assert full_history[0]["encoder_gradient_norm"] > 0
    full_result = json.loads((root / "full/result.json").read_text())
    replay_result = json.loads((root / "replay/result.json").read_text())
    assert full_result["metrics"] == replay_result["metrics"]
    assert full_result["predictions_sha256"] == replay_result["predictions_sha256"]


def test_every_update_trains_on_the_whole_four_group_bag(recovered):
    record = recovered["record"]
    assert len(record) == 4 * 8
    assert {(epoch, update) for epoch, update, _, _ in record} == {(0, 0), (0, 1), (1, 0), (1, 1)}
    assert all(shape[2] == 4 and tuple(mask) == (1, 6, 4) for _, _, shape, mask in record)


def test_binding_declares_the_policy_and_only_adds_the_window_query(recovered):
    root = recovered["root"]
    run = json.loads((root / "full/run.json").read_text())
    assert (
        run["binding"]["config"]["classifier_strategy"]
        == recovered["values"]["classifier_strategy"]
    )
    assert run["encoder_trainable_parameters"] == 32 * (20 * 4 * 2 * 32 + 12 * 2 * 96)
    checkpoint = torch.load(root / "full/checkpoint.pt", weights_only=True)
    head = {k for k in checkpoint["model"] if not k.startswith("encoder.")}
    assert head == {
        "head.proj.0.weight",
        "head.proj.0.bias",
        "head.proj.1.weight",
        "head.proj.1.bias",
        "head.slot_emb",
        "head.query",
        "head.out.weight",
        "head.out.bias",
        "window_query",
    }
    query = checkpoint["model"]["window_query"]
    if recovered["arm"] == "mean":
        assert torch.equal(query, torch.zeros_like(query))
    else:
        assert query.count_nonzero() > 0


def test_both_arms_share_initial_state_rng_inputs_and_first_loss(cohort, recovered):  # noqa: F811
    other = "query" if recovered["arm"] == "mean" else "mean"
    root = cohort["base"] / f"bag-{other}"
    if not (root / "full/invocation.json").exists():
        pytest.skip("paired arm not yet recovered")
    mine = json.loads((recovered["root"] / "full/invocation.json").read_text())["updates"][0]
    theirs = json.loads((root / "full/invocation.json").read_text())["updates"][0]
    assert mine["loss"] == theirs["loss"]
    a = torch.load(recovered["root"] / "full/checkpoint.pt", weights_only=True)
    b = torch.load(root / "full/checkpoint.pt", weights_only=True)
    assert_same(a["rng_by_rank"], b["rng_by_rank"], "rng")


def test_restored_student_reproduces_trainer_predictions_mri_only(recovered, cohort):  # noqa: F811
    from rsna.competitive import train

    root = recovered["root"]
    checkpoint = torch.load(root / "full/checkpoint.pt", weights_only=True)
    config = TrainConfig(**recovered["values"])
    torch.manual_seed(0)
    encoder = fake_create_encoder(config.encoder)[0]
    configure_encoder_training(encoder, True)
    model = wb.WindowBagSlotStudyNet(encoder, 32, config.classifier_strategy, encode_chunk_size=8)
    FrozenBaseDelta(model, checkpoint["binding"]["config"]["encoder"]).restore(model, checkpoint)
    backbone = model.encoder.backbone.state_dict()
    assert all(torch.equal(backbone[k], v) for k, v in OFFICIAL.items())
    expected = pd.read_csv(root / "full/outer_predictions.csv")
    manifest = cohort["manifest"]
    outer = manifest[manifest.study_uid.isin(expected.study_uid)].reset_index(drop=True)
    data = pi.PublicTripletStudyDataset(
        outer,
        pd.DataFrame(),
        cohort["base"] / "cache",
        public_input=POLICY,
        source_revision="synthetic",
        slices=10,
        side=336,
        max_series=6,
    )
    loader = torch.utils.data.DataLoader(data, batch_size=1, collate_fn=collate_studies)
    predictions, _ = train.predict(model, loader, CPU, config)
    roundtrip = pd.read_csv(io.StringIO(predictions.to_csv(index=False)))
    pd.testing.assert_frame_equal(roundtrip, expected, check_exact=True)
    for uid, values in zip(predictions.study_uid, predictions[list(TARGETS)].to_numpy()):
        source = cohort["rows"][uid]
        deployed = wb.deployment_probabilities(
            model,
            torch.from_numpy(source["images"]),
            torch.from_numpy(source["slot_mask"]),
            CPU,
            config.precision,
        )
        assert np.array_equal(deployed.numpy(), values)
