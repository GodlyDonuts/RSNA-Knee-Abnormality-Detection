"""Source-matched v15 slot classifier on one DINOv3 student: engineering only, never accuracy."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from test_public_triplet_training import PIXEL_GROUP, POLICY, row
from torch import nn
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from rsna.competitive import public_input as pi
from rsna.competitive import source_slot as ss
from rsna.competitive import suffix_adapters as sa
from rsna.competitive.adapters import adapter_provenance, install_adapters
from rsna.competitive.backbones import _GrayEncoder
from rsna.competitive.data import collate_studies
from rsna.competitive.delta_checkpoint import FrozenBaseDelta
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.train import TrainConfig
from rsna.schema import TARGETS
from tools.research.public_dinov2_input_reference import reference
from tools.research.public_teacher_evidence import teacher
from tools.research.source_matched_slot import reference as training_reference

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "initiative/experiments/competitive-v1/source-matched-slot-v1"
LEGACY = ROOT / "initiative/experiments/competitive-v1/public-input-recipe-v1"
CAPACITY = ROOT / "initiative/experiments/competitive-v1/dense-suffix-capacity-v1"
PINNED_CONFIG = json.loads(
    (ROOT / "configs/competitive/weights/dinov3-large.config.json").read_text()
)
CPU = torch.device("cpu")
SOURCE = reference.SOURCE_PATH.read_bytes()
TINY = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=20,
    num_attention_heads=4,
    patch_size=16,
    num_register_tokens=4,
)
FAKE_SHA = {"config_sha256": "1" * 64, "weights_sha256": "2" * 64}


def configs():
    return {
        "r32": json.loads((RECIPE / "r32-config.json").read_text()),
        "dense": json.loads((RECIPE / "dense-prefix18-config.json").read_text()),
    }


STRATEGY = configs()["r32"]["classifier_strategy"]


def official_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        return copy.deepcopy(DINOv3ViTModel(DINOv3ViTConfig(**TINY)).state_dict())


OFFICIAL = official_state()


def tiny_backbone():
    model = DINOv3ViTModel(DINOv3ViTConfig(**TINY))
    model.load_state_dict(OFFICIAL, strict=True)
    return model


def fake_create_encoder(spec, *, input_channels=3, input_contract="public_ordered_triplet"):
    """Pinned-L stand-in with the real strict-load/install/wrap order of ``create_encoder``."""
    assert input_channels == 3 and input_contract == "public_ordered_triplet"
    model = tiny_backbone()
    provenance = dict(
        FAKE_SHA,
        architecture="dinov3_l16",
        pretrained=True,
        embedding_dim=32,
        input_contract="public_ordered_triplet_0_1_imagenet_rgb_v1",
    )
    if "suffix_adapter" in spec:
        provenance["suffix_adapter"] = sa.install_suffix_adapter(model, spec["suffix_adapter"])
    if "adapter" in spec:
        policy = {k: spec["adapter"][k] for k in sa_policy_keys()}
        provenance["adapter"] = install_adapters(model, policy)
    return _GrayEncoder(model, 16, 3), 32, provenance


def sa_policy_keys():
    return ("rank", "alpha", "dropout", "ffn_last_layers", "gradient_checkpointing")


@pytest.fixture(scope="module")
def namespace():
    ns, _ = teacher.load_namespace(SOURCE, PIXEL_GROUP)
    return ns


@pytest.fixture(scope="module")
def training_namespace():
    return training_reference.load_namespace(SOURCE)


class ReferenceBackbone(nn.Module):
    """DINOv2-shaped stand-in: ``last_hidden_state = [CLS, patches]``, no registers."""

    def __init__(self, dim=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim)
        self.patch = nn.Conv2d(3, dim, 48, 48)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, pixel_values):
        patches = self.patch(pixel_values).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls.expand(len(patches), -1, -1), patches], 1)
        return SimpleNamespace(last_hidden_state=tokens)


class ReferenceEncoder(nn.Module):
    """The canonical encoder contract over the reference backbone: [0,1] in, ImageNet once."""

    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        self.calls: list[int] = []

    def forward_features(self, images):
        self.calls.append(len(images))
        out = self.backbone(pixel_values=(images - self.mean) / self.std).last_hidden_state
        side = int((out.shape[1] - 1) ** 0.5)
        return out[:, 0], out[:, 1:].reshape(len(images), side, side, out.shape[-1])


def matched_models(namespace, seed=3):
    torch.manual_seed(seed)
    backbone = ReferenceBackbone()
    original = namespace["Model"](backbone, 8, pool="cls_mean", prior=False).eval()
    ours = ss.SourceSlotStudyNet(ReferenceEncoder(backbone), 8, STRATEGY).eval()
    ours.head.load_state_dict(original.head.state_dict(), strict=True)
    return original, ours


def study(seed, absent):
    source = row(seed, absent=absent)
    images = torch.from_numpy(source["images"])
    mask = torch.from_numpy(source["slot_mask"])
    return source, images, mask


def batch_inputs(windows, mask):
    views = windows.shape[1]
    return (
        windows[None],
        torch.zeros(1, 6, 5),
        mask[None],
        mask[None, :, None].expand(1, 6, views),
        torch.zeros(1, 6, views, 2),
    )


# --- Original source equality -------------------------------------------------------


def test_slot_head_has_original_parameters_initialization_and_outputs(namespace):
    torch.manual_seed(7)
    original = namespace["SlotHead"](64, 6, 12, prior=False)
    after_original = torch.get_rng_state()
    torch.manual_seed(7)
    ours = ss.SourceSlotHead(64, 6, 12)
    assert torch.equal(torch.get_rng_state(), after_original)
    state, expected = ours.state_dict(), original.state_dict()
    assert list(state) == list(expected) and "slot_prior" not in state
    assert all(torch.equal(state[k], expected[k]) for k in state)
    x = torch.randn(3, 6, 64)
    mask = torch.tensor([[1, 1, 1, 1, 1, 1], [1, 0, 1, 0, 0, 1], [0, 0, 0, 0, 0, 1.0]])
    for train in (False, True):
        original.train(train)
        ours.train(train)
        torch.manual_seed(1)
        a = original(x, mask)
        torch.manual_seed(1)
        assert torch.equal(ours(x, mask), a)


@pytest.mark.parametrize("absent", [(), (2, 5), (1, 2, 3, 4, 5)])
def test_ten_window_probability_aggregation_equals_original_predict_member(namespace, absent):
    original, ours = matched_models(namespace)
    _, images, mask = study(5, absent)
    expected = namespace["predict_member"](
        original, images.numpy()[None], mask.numpy().astype(np.float32)[None], [0], CPU, 336
    )
    assert list(namespace["window_starts"](12, 3)) == list(ss.EVALUATION_VIEWS)
    windows = ss.source_row_windows(images)
    with torch.no_grad():
        actual = ours.window_probabilities(*batch_inputs(windows, mask))
    assert np.array_equal(actual.numpy(), expected)
    with torch.no_grad():
        assert torch.equal(ss.deployment_probabilities(ours, images, mask, CPU, "fp32"), actual[0])


@pytest.mark.parametrize("group", range(4))
def test_training_group_window_and_logits_equal_original_take_group(
    namespace, training_namespace, group
):
    original, ours = matched_models(namespace)
    _, images, mask = study(9, (3,))
    rows = training_namespace["take_group"](images[None], group)
    windows = ss.source_row_windows(images)
    view = ss.TRAINING_VIEWS[group]
    assert view == 3 * group and torch.equal(windows[:, view], rows[0].float() / 255)
    with torch.no_grad():
        expected = original(rows, mask[None].float())
        actual = ours(*batch_inputs(windows[:, view : view + 1], mask))
    assert torch.equal(actual, expected)


def test_augmentation_equals_original_draws_and_uint8_arithmetic(training_namespace):
    _, images, mask = study(2, (4,))
    window = images[:, 3:6]
    torch.manual_seed(41)
    expected = training_namespace["augment"](window[None].clone())[0]
    assert expected.dtype == torch.uint8
    actual = ss.source_augment(window.float() / 255, torch.Generator().manual_seed(41))
    assert torch.equal((actual * 255).round().to(torch.uint8), expected)
    assert torch.equal(actual, expected.float() / 255)
    assert not torch.equal(expected, window)


def test_reviewed_training_nodes_are_hash_bound():
    review = training_reference.load_review()
    assert review["node_text_sha256"] == ss.TRAINING_NODE_TEXT_SHA256
    assert teacher.load_teacher_review()["node_text_sha256"] == ss.REFERENCE_NODE_TEXT_SHA256
    assert STRATEGY["source_sha256"] == pi.SOURCE_SHA256 == review["source_sha256"]


# --- DINOv3 bridge, masks, RNG and normalization ---------------------------------------


def test_dinov3_cls_mean_uses_normalized_cls_and_excludes_four_registers():
    encoder = _GrayEncoder(tiny_backbone(), 16, 3).eval()
    model = ss.SourceSlotStudyNet(encoder, 32, STRATEGY).eval()
    window = torch.rand(1, 6, 3, 48, 48)
    mask = torch.tensor([[True, False, True, False, False, False]])
    with torch.no_grad():
        features = model.slot_features(window, mask)
        tokens = encoder.backbone(
            pixel_values=encoder._normalized_rgb(window[mask])
        ).last_hidden_state
    assert tokens.shape[1] == 1 + 4 + 9
    expected = torch.cat([tokens[:, 0], tokens[:, 5:].mean(1)], dim=1)
    with_registers = torch.cat([tokens[:, 0], tokens[:, 1:].mean(1)], dim=1)
    assert torch.equal(features[mask], expected)
    assert not torch.allclose(features[mask], with_registers)
    assert features.shape == (1, 6, 64) and features[~mask].count_nonzero() == 0


def test_absent_slots_never_reach_the_encoder_and_inputs_are_checked(namespace):
    _, ours = matched_models(namespace)
    _, images, mask = study(4, (1, 3, 5))
    windows = ss.source_row_windows(images)
    with torch.no_grad():
        ours.window_probabilities(*batch_inputs(windows, mask))
    assert ours.encoder.calls == [3] * 10
    single = batch_inputs(windows[:, :1], mask)
    cases = [
        (batch_inputs(windows[:, :2], mask), "views"),
        (batch_inputs(windows[:, :1] * 255, mask), r"\[0,1\]"),
        (batch_inputs(windows[:, :1], torch.zeros(6, dtype=torch.bool)), "acquired"),
        ((single[0].clone().fill_(0.5), *single[1:]), "zero"),
        ((*single[:3], ~single[3], single[4]), "slot mask"),
    ]
    for inputs, match in cases:
        with pytest.raises(ValueError, match=match if match != "views" else None):
            ours(*inputs)


def test_window_selection_is_deterministic_uniform_and_leaves_global_rng_untouched():
    groups = [ss.training_group(20260922, epoch, update) for epoch in range(2) for update in range(281)]
    assert set(groups) == set(range(4)) and all(groups.count(g) > 100 for g in range(4))
    before = torch.get_rng_state()
    torch.manual_seed(999)
    assert [ss.training_group(20260922, e, u) for e in range(2) for u in range(281)] == groups
    _, images, mask = study(6, (5,))
    batch = collate_studies(
        [
            {
                "images": ss.source_row_windows(images),
                "features": torch.zeros(6, 5),
                "series_mask": mask,
                "slice_mask": mask[:, None].expand(6, 10).clone(),
                "positions": torch.zeros(6, 10, 2),
                "labels": torch.zeros(12),
                "label_mask": torch.ones(12),
                "study_uid": "s0",
            }
        ]
    )
    state = torch.get_rng_state()
    plain = ss.training_inputs(
        batch, CPU, dict(STRATEGY, augmentation="none"), seed=1, epoch=0, update=7
    )
    view = ss.TRAINING_VIEWS[ss.training_group(1, 0, 7)]
    assert torch.equal(plain[0], batch["images"][:, :, view : view + 1])
    first = ss.training_inputs(batch, CPU, STRATEGY, seed=1, epoch=0, update=7)
    again = ss.training_inputs(batch, CPU, STRATEGY, seed=1, epoch=0, update=7)
    other = ss.training_inputs(batch, CPU, STRATEGY, seed=1, epoch=1, update=7)
    assert torch.equal(torch.get_rng_state(), state)
    assert all(torch.equal(a, b) for a, b in zip(first, again))
    assert not torch.equal(first[0], other[0]) or ss.training_group(1, 1, 7) != view // 3
    assert first[0][0, ~mask].count_nonzero() == 0
    levels = first[0] * 255
    assert torch.equal(levels.round(), levels) and first[0].max() <= 1
    torch.set_rng_state(before)


def test_rows_are_divided_once_and_imagenet_normalized_once(tmp_path):
    rows = {"s0": row(0, absent=(2, 5)), "s1": row(1)}
    pi.write_public_triplet_cache(
        tmp_path,
        rows,
        pixel_group=copy.deepcopy(PIXEL_GROUP),
        source_revision="synthetic",
        fitting_uids=["s1"],
        evaluation_uids=["s0"],
    )
    data = pi.PublicTripletStudyDataset(
        pd.DataFrame({"study_uid": ["s0"]}),
        pd.DataFrame(),
        tmp_path,
        public_input=POLICY,
        source_revision="synthetic",
        slices=10,
        side=336,
        max_series=6,
    )
    item = data[0]
    windows = ss.source_row_windows(torch.from_numpy(rows["s0"]["images"]))
    assert torch.equal(item["images"], windows)
    encoder = _GrayEncoder(tiny_backbone(), 16, 3)
    once = (windows[0, :1] - encoder.mean) / encoder.std
    assert torch.equal(encoder._normalized_rgb(windows[0, :1]), once)
    with pytest.raises(ValueError, match=r"\[0,1\]"):
        encoder._normalized_rgb(once)


# --- Configs, legacy behaviour --------------------------------------------------------


def test_recipes_are_separately_versioned_and_bind_the_declared_graphs():
    legacy = json.loads((LEGACY / "public-triplet-config.json").read_text())
    values = configs()
    assert values["r32"] == dict(legacy, classifier_strategy=STRATEGY)
    dense = values["dense"]
    expected = dict(legacy, classifier_strategy=STRATEGY, encoder_lr=1e-5)
    expected["encoder"] = {k: v for k, v in legacy["encoder"].items() if k != "adapter"}
    expected["encoder"]["suffix_adapter"] = dict(
        method=sa.DENSE, frozen_prefix_layers=18, dropout=0.05, gradient_checkpointing=True
    )
    assert dense == expected
    for config in values.values():
        TrainConfig(**config).validate()
    assert ss.validate_strategy(STRATEGY)["augmentation"] == ss.AUGMENT
    provenance = sa.suffix_adapter_provenance(dense["encoder"]["suffix_adapter"], PINNED_CONFIG)
    assert provenance["trainable_parameters"] == 75_497_472
    assert {int(name.split(".")[2]) for name in provenance["target_modules"]} == set(range(18, 24))
    r32 = adapter_provenance(
        {k: legacy["encoder"]["adapter"][k] for k in sa_policy_keys()}, PINNED_CONFIG
    )
    assert r32["method"] == "rs_lora_v1" and r32["rank"] == 32
    manifest = json.loads((ROOT / "initiative/manifest.json").read_text())
    for path in sorted(CAPACITY.glob("*.json")) + sorted(RECIPE.iterdir()):
        key = str(path.relative_to(ROOT / "initiative"))
        assert manifest[key]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    for name in ("dense-config.json", "suffix-r32-config.json"):
        prefix12 = json.loads((CAPACITY / name).read_text())
        assert prefix12["encoder"]["suffix_adapter"]["frozen_prefix_layers"] == 12
        assert "classifier_strategy" not in prefix12


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda v: v.update(public_input=None), "public rows"),
        (lambda v: v.update(pooling="max"), "public rows"),
        (lambda v: v["classifier_strategy"].update(prior=True), "Exact"),
        (lambda v: v["classifier_strategy"].update(hidden=128), "Exact"),
        (lambda v: v["classifier_strategy"].update(dropout=0), "Exact"),
        (lambda v: v["classifier_strategy"].update(augmentation="flip"), "Exact"),
        (lambda v: v["classifier_strategy"].update(pool="cls_mean"), "Exact"),
        (lambda v: v["classifier_strategy"].update(extra=1), "Exact"),
        (lambda v: v["classifier_strategy"].update(source_sha256="0" * 64), "Exact"),
    ],
)
def test_strategy_refusals(change, match):
    values = copy.deepcopy(configs()["r32"])
    change(values)
    with pytest.raises(ValueError, match=match):
        TrainConfig(**values).validate()


def test_defaults_and_existing_recipes_are_unchanged():
    assert TrainConfig().classifier_strategy is None
    for path in [*LEGACY.glob("*config.json"), *CAPACITY.glob("*config.json")]:
        values = json.loads(path.read_text())
        assert "classifier_strategy" not in values
        TrainConfig(**values).validate()


# --- Canonical trainer: full / stop1 / resume ----------------------------------------

ENGINEERING = dict(
    epochs=2, diagnostic_study_limit=16, warmup_updates=2, submission_every_epochs=0, min_free_gib=0
)


def trainer_cohort(count=32):
    rows = []
    for i in range(count):
        values = dict(study_uid=f"s{i:02d}", leakage_group=f"g{i}", split="train")
        for t in TARGETS:
            values[t], values[t + "__mask"] = (i + len(t)) % 2, float((i + len(t)) % 3 != 0)
            values[t + "__provenance"] = "synthetic_weak"
        rows.append(values)
    manifest = pd.DataFrame(rows)
    folds = manifest[["study_uid", "leakage_group"]].copy()
    folds["fold"] = [0] * (count // 2) + [1] * (count - count // 2)
    return manifest, folds


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    base = tmp_path_factory.mktemp("source-slot")
    manifest, folds = trainer_cohort()
    rows = {
        uid: row(i, absent=(1, 3, 4, 5) if i % 2 else (1, 2, 3, 4, 5))
        for i, uid in enumerate(manifest.study_uid)
    }
    pi.write_public_triplet_cache(
        base / "cache",
        rows,
        pixel_group=copy.deepcopy(PIXEL_GROUP),
        source_revision="synthetic",
        fitting_uids=folds.study_uid[folds.fold != 0],
        evaluation_uids=folds.study_uid[folds.fold == 0],
    )
    series = base / "series.csv"
    series.write_text("SeriesInstanceUID\na\n")
    release = {
        "series_sha256": hashlib.sha256(series.read_bytes()).hexdigest(),
        "source_revision": "synthetic",
    }
    return dict(base=base, manifest=manifest, folds=folds, rows=rows, series=series, release=release)


def run_trainer(cohort, values, output, resume=None, stop=None, record=None):
    from rsna.competitive import train

    args = argparse.Namespace(
        studies=cohort["base"] / "unused",
        artifact_root=cohort["base"],
        folds=cohort["base"] / "unused",
        series=cohort["series"],
        cache=cohort["base"] / "cache",
        output=output,
        resume=resume,
        stop_after_updates=stop,
    )
    patch = pytest.MonkeyPatch()
    patch.setattr(
        train, "load_verified_release", lambda *_: (cohort["manifest"].copy(), cohort["release"])
    )
    if record is not None:
        original = train.source_slot_training_inputs

        def recording(batch, device, strategy, *, seed, epoch, update):
            inputs = original(batch, device, strategy, seed=seed, epoch=epoch, update=update)
            record.append((epoch, update, ss.training_group(seed, epoch, update), inputs[0]))
            return inputs

        patch.setattr(train, "source_slot_training_inputs", recording)
    output.mkdir(parents=True, exist_ok=True)
    try:
        return train._run_locked(
            args,
            TrainConfig(**values),
            [False],
            0,
            1,
            CPU,
            fake_create_encoder,
            None,
            CompetitiveStudyNet,
            lambda *_: cohort["folds"].copy(),
        )
    finally:
        patch.undo()


def engineering(arm):
    values = dict(configs()[arm], **ENGINEERING)
    values["fitting_supervision"] = None
    return values


@pytest.fixture(scope="module", params=["r32", "dense"])
def recovered(request, cohort):
    arm = request.param
    root = cohort["base"] / arm
    values = engineering(arm)
    record: list = []
    assert run_trainer(cohort, values, root / "full", record=record) == 0
    assert run_trainer(cohort, values, root / "replay", stop=1) == 75
    partial = torch.load(root / "replay/checkpoint.pt", weights_only=True)
    first = json.loads((root / "replay/invocation.json").read_text())
    assert run_trainer(cohort, values, root / "replay", resume=root / "replay/checkpoint.pt") == 0
    return dict(arm=arm, root=root, values=values, record=record, partial=partial, first=first)


def assert_same(a, b, path="root"):
    if isinstance(a, torch.Tensor):
        assert isinstance(b, torch.Tensor) and a.dtype == b.dtype and torch.equal(a, b), path
    elif isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b), path
        for key in a:
            assert_same(a[key], b[key], f"{path}.{key}")
    elif isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        for index, (x, y) in enumerate(zip(a, b)):
            assert_same(x, y, f"{path}[{index}]")
    else:
        assert a == b, path


def numerical(history):
    return [{k: v for k, v in event.items() if k != "elapsed_seconds"} for event in history]


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


def test_binding_declares_strategy_graph_and_frozen_official_base(recovered):
    root = recovered["root"]
    run = json.loads((root / "full/run.json").read_text())
    config = run["binding"]["config"]
    assert config["classifier_strategy"] == STRATEGY
    expected = {"r32": 32 * (20 * 4 * 2 * 32 + 12 * 2 * 96), "dense": 2 * (4 * 32**2 + 2 * 2048)}
    assert run["encoder_trainable_parameters"] == expected[recovered["arm"]]
    checkpoint = torch.load(root / "full/checkpoint.pt", weights_only=True)
    learned = set(checkpoint["model"])
    assert {k for k in learned if not k.startswith("encoder.")} == {
        "head.proj.0.weight",
        "head.proj.0.bias",
        "head.proj.1.weight",
        "head.proj.1.bias",
        "head.slot_emb",
        "head.query",
        "head.out.weight",
        "head.out.bias",
    }
    suffix = (".dense_delta",) if recovered["arm"] == "dense" else (".lora_A", ".lora_B")
    assert all(k.endswith(suffix) for k in learned if k.startswith("encoder."))
    if recovered["arm"] == "dense":
        assert all(".layer.18." in k or ".layer.19." in k for k in learned if "encoder." in k)


def test_each_update_reads_one_shared_source_group_for_all_eight_studies(recovered):
    record = recovered["record"]
    assert len(record) == 4 * 8
    by_update: dict = {}
    for epoch, update, group, window in record:
        by_update.setdefault((epoch, update), set()).add(group)
        assert window.shape[2] == 1
    assert list(by_update) == [(0, 0), (0, 1), (1, 0), (1, 1)]
    assert all(len(groups) == 1 for groups in by_update.values())


def test_restored_student_reproduces_trainer_predictions_mri_only(recovered, cohort):
    from rsna.competitive import train

    root = recovered["root"]
    checkpoint = torch.load(root / "full/checkpoint.pt", weights_only=True)
    config = TrainConfig(**recovered["values"])
    torch.manual_seed(0)
    encoder = fake_create_encoder(config.encoder)[0]
    from rsna.competitive.adapters import configure_encoder_training

    configure_encoder_training(encoder, True)
    model = ss.SourceSlotStudyNet(encoder, 32, STRATEGY, encode_chunk_size=8)
    guard = FrozenBaseDelta(model, checkpoint["binding"]["config"]["encoder"])
    guard.restore(model, checkpoint)
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
    roundtrip = pd.read_csv(__import__("io").StringIO(predictions.to_csv(index=False)))
    pd.testing.assert_frame_equal(roundtrip, expected, check_exact=True)
    for uid, values in zip(predictions.study_uid, predictions[list(TARGETS)].to_numpy()):
        source = cohort["rows"][uid]
        deployed = ss.deployment_probabilities(
            model,
            torch.from_numpy(source["images"]),
            torch.from_numpy(source["slot_mask"]),
            CPU,
            config.precision,
        )
        assert np.array_equal(deployed.numpy(), values)


def test_legacy_public_triplet_trainer_path_and_binding_unchanged(cohort, tmp_path):
    values = engineering("r32")
    values.pop("classifier_strategy")
    assert run_trainer(cohort, values, tmp_path / "legacy", stop=1) == 75
    run = json.loads((tmp_path / "legacy/run.json").read_text())
    assert "classifier_strategy" not in run["binding"]["config"]
    checkpoint = torch.load(tmp_path / "legacy/checkpoint.pt", weights_only=True)
    assert "output_weight" in checkpoint["model"] and not any(
        k.startswith("head.") for k in checkpoint["model"]
    )


def test_student_has_no_teacher_or_reference_dependency():
    source = (ROOT / "rsna/competitive/source_slot.py").read_text()
    assert "tools." not in source and "import tools" not in source
