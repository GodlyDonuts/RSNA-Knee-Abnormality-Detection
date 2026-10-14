"""Identity-preserving patch focus for the source-slot classifier (issue #62).

Engineering only: exact identity, gradients, masking, precision and trainer recovery on
synthetic inputs. Nothing here is GPU acceptance or accuracy evidence.
"""

import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from test_public_triplet_training import row
from torch import nn

from rsna.competitive import patch_focus as pf
from rsna.competitive import source_slot as ss
from rsna.competitive import window_bag as wb
from rsna.competitive.train import TrainConfig
from rsna.schema import TARGETS

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "initiative/experiments/competitive-v1/source-patch-focus-v1"
SLOT_RECIPE = ROOT / "initiative/experiments/competitive-v1/source-matched-slot-v1"
CPU = torch.device("cpu")
T, D = len(TARGETS), 8
ARMS = ("control", "candidate")
# Reviewed live sources reused unchanged.
SOURCE_SLOT_SHA256 = "3e4e1bb95881a6521ff4c2fc30e4175e5b23e2e5dcaab5a225c2be552b4066e6"


def configs():
    return {arm: json.loads((RECIPE / f"{arm}-r32-config.json").read_text()) for arm in ARMS}


STRATEGIES = {arm: values["classifier_strategy"] for arm, values in configs().items()}
SLOT_STRATEGY = json.loads((SLOT_RECIPE / "r32-config.json").read_text())["classifier_strategy"]


class Backbone(nn.Module):
    def __init__(self, dim=D):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, 16, 16)
        self.cls = nn.Linear(3, dim)
        self.calls: list[int] = []

    def forward_features(self, images):
        self.calls.append(len(images))
        return self.cls(images.mean((2, 3))), self.patch(images).permute(0, 2, 3, 1)


def model(arm, seed=3):
    torch.manual_seed(seed)
    return pf.PatchFocusSlotStudyNet(Backbone(), D, STRATEGIES[arm])


def old(seed=3):
    torch.manual_seed(seed)
    return ss.SourceSlotStudyNet(Backbone(), D, SLOT_STRATEGY)


def window(views=1, absent=(1, 4), batch=3, side=32, seed=0):
    generator = torch.Generator().manual_seed(seed)
    mask = torch.ones(batch, 6, dtype=torch.bool)
    mask[:, list(absent)] = False
    images = torch.randint(0, 256, (batch, 6, views, 3, side, side), generator=generator) / 255
    images = images * mask[:, :, None, None, None, None]
    return (
        images,
        torch.zeros(batch, 6, 5),
        mask,
        mask[:, :, None].expand(batch, 6, views),
        torch.zeros(batch, 6, views, 2),
    )


def randomize_query(net, seed=5, scale=1.0):
    with torch.no_grad():
        net.patch_query.copy_(torch.randn(T, D, generator=torch.Generator().manual_seed(seed)))
        net.patch_query.mul_(scale)


# --- Policy, sources and configs ---------------------------------------------------------


def test_reviewed_sources_are_byte_unchanged_and_dispatch_is_by_exact_kind():
    source = (ROOT / "rsna/competitive/source_slot.py").read_bytes()
    assert hashlib.sha256(source).hexdigest() == SOURCE_SLOT_SHA256
    assert pf.classifier_module(STRATEGIES["candidate"]) is pf
    assert pf.classifier_module(SLOT_STRATEGY) is ss
    assert pf.classifier_module(wb.STRATEGY_FIXED) is wb
    assert pf.classifier_module(None) is ss
    assert TrainConfig().classifier_strategy is None


def test_arms_differ_only_in_patch_focus_and_reuse_the_slot_policy():
    control, candidate = (pf.validate_strategy(STRATEGIES[arm]) for arm in ARMS)
    assert {k for k in control if control[k] != candidate[k]} == {"patch_focus"}
    assert (control["patch_focus"], candidate["patch_focus"]) == ("disabled", "query_residual")
    assert pf.slot_strategy(control) == ss.validate_strategy(SLOT_STRATEGY)
    with pytest.raises(ValueError, match="Exact source-matched"):
        ss.validate_strategy(candidate)


@pytest.mark.parametrize(
    "change",
    [
        lambda v: v.update(patch_focus="learned"),
        lambda v: v.update(patch_focus=None),
        lambda v: v.update(patch_pool="mean"),
        lambda v: v.update(kind=ss.KIND),
        lambda v: v.update(training_windows=wb.TRAINING_WINDOWS),
        lambda v: v.update(evaluation_windows=wb.EVALUATION_WINDOWS),
        lambda v: v.update(hidden=128),
        lambda v: v.update(dropout=0),
        lambda v: v.update(prior=True),
        lambda v: v.update(augmentation="flip"),
        lambda v: v.update(teacher="patch_v1"),
        lambda v: v.pop("patch_focus"),
    ],
)
def test_strategy_tampering_is_refused(change):
    for arm in ARMS:
        values = copy.deepcopy(configs()[arm])
        change(values["classifier_strategy"])
        with pytest.raises(ValueError, match="Exact"):
            TrainConfig(**values).validate()
        with pytest.raises(ValueError, match="Exact"):
            pf.PatchFocusSlotStudyNet(Backbone(), D, values["classifier_strategy"])


def test_configs_change_only_the_classifier_strategy_and_are_manifest_bound():
    slot = json.loads((SLOT_RECIPE / "r32-config.json").read_text())
    manifest = json.loads((ROOT / "initiative/manifest.json").read_text())
    for values in configs().values():
        assert {k: v for k, v in values.items() if k != "classifier_strategy"} == {
            k: v for k, v in slot.items() if k != "classifier_strategy"
        }
        TrainConfig(**values).validate()
        with pytest.raises(ValueError, match="public rows"):
            TrainConfig(**dict(copy.deepcopy(values), public_input=None)).validate()
    for path in sorted(RECIPE.iterdir()):
        key = str(path.relative_to(ROOT / "initiative"))
        assert manifest[key]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()


def test_stateful_heads_are_refused():
    class Stateful(ss.SourceSlotHead):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.register_buffer("seen", torch.zeros(()))

    patch = pytest.MonkeyPatch()
    patch.setattr(ss, "SourceSlotHead", Stateful)
    try:
        with pytest.raises(ValueError, match="stateless"):
            model("candidate")
    finally:
        patch.undo()


# --- Exact identity at zero queries --------------------------------------------------------


def test_parameters_and_factory_rng_equal_the_old_model_plus_a_zero_query():
    reference = old()
    after_old = torch.get_rng_state()
    for arm in ARMS:
        net = model(arm)
        assert torch.equal(torch.get_rng_state(), after_old)
        a, b = reference.state_dict(), net.state_dict()
        assert list(b) == ["patch_query", *a] and all(torch.equal(a[k], b[k]) for k in a)
        assert torch.equal(b["patch_query"], torch.zeros(T, D))
        assert not list(net.head.buffers())


@pytest.mark.parametrize("arm", ARMS)
@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("precision", ["fp32", "bf16"])
def test_zero_query_logits_probabilities_and_rng_are_bitwise_the_old_model(
    arm, training, precision
):
    reference, net = old().train(training), model(arm).train(training)
    autocast = dict(device_type="cpu", dtype=torch.bfloat16, enabled=precision == "bf16")
    outputs = []
    for candidate in (reference, net):
        torch.manual_seed(11)
        with torch.autocast(**autocast):
            logits = candidate(*window())
            probabilities = candidate.window_probabilities(*window(views=10, seed=1))
        outputs.append((logits, probabilities, torch.get_rng_state()))
    for a, b in zip(*outputs):
        assert a.dtype == b.dtype and torch.equal(a, b)
    if training:
        torch.manual_seed(11)
        dropped = reference(*window())
        torch.manual_seed(12)
        assert not torch.equal(dropped, reference(*window()))


def test_slot_tokens_match_the_reviewed_features_from_one_encoder_pass():
    net, reference = model("candidate"), old()
    images, _, mask, _, _ = window()
    base, patches = net.slot_tokens(images[:, :, 0], mask)
    assert torch.equal(base, reference.slot_features(images[:, :, 0], mask))
    assert patches.shape == (int(mask.sum()), 4, D)
    net.encoder.calls.clear()
    net(*window())
    assert sum(net.encoder.calls) == int(mask.sum())


def test_both_residual_terms_share_dropout_draws_and_cancel_only_at_zero():
    net = model("candidate").train()
    images, _, mask, _, _ = window()
    base, patches = net.slot_tokens(images[:, :, 0], mask)
    torch.manual_seed(4)
    zero = net.focused_logits(base, patches, mask)
    torch.manual_seed(4)
    assert torch.equal(zero, net.head(base, mask.float()))
    focus = pf.focus_features(base, patches, mask, torch.zeros(T, D))
    assert torch.equal(focus, base[:, None].expand_as(focus))
    randomize_query(net)
    torch.manual_seed(4)
    moved = net.focused_logits(base, patches, mask)
    assert not torch.equal(moved, zero) and torch.isfinite(moved).all()
    torch.manual_seed(4)
    net.head(base, mask.float())
    after_head = torch.get_rng_state()
    torch.manual_seed(4)
    net.focused_logits(base, patches, mask)
    assert torch.equal(torch.get_rng_state(), after_head)


def test_target_conditioned_head_reads_each_target_row_and_matches_the_head():
    net = model("candidate").eval()
    x, mask = torch.randn(5, 6, 2 * D), (torch.rand(5, 6) > 0.3).float()
    mask[:, 0] = 1
    tiled = pf.target_conditioned_head(net.head, x[:, None].expand(5, T, 6, 2 * D), mask)
    torch.testing.assert_close(tiled, net.head(x, mask), rtol=0, atol=1e-6)
    changed = x[:, None].expand(5, T, 6, 2 * D).clone()
    changed[:, 3] += 1
    moved = pf.target_conditioned_head(net.head, changed, mask)
    differs = (moved != tiled).any(0)
    assert differs[3] and not differs[[t for t in range(T) if t != 3]].any()


# --- Gradients ------------------------------------------------------------------------------


def masked_loss(logits, seed=0):
    generator = torch.Generator().manual_seed(seed)
    labels = torch.randint(0, 2, logits.shape, generator=generator).float()
    mask = torch.ones_like(logits)
    mask[:, ::3] = 0
    loss = nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (loss * mask).sum() / mask.sum(), mask


def gradients(net):
    torch.manual_seed(2)
    loss, mask = masked_loss(net(*window()))
    loss.backward()
    return {n: p.grad for n, p in net.named_parameters() if p.grad is not None}, mask


def test_zero_query_gets_informative_gradients_only_for_observed_targets():
    net = model("candidate").train()
    grads, mask = gradients(net)
    query = grads["patch_query"]
    assert torch.isfinite(query).all()
    observed = mask.any(0)
    assert (query[observed].abs().sum(1) > 0).all()
    assert query[~observed].count_nonzero() == 0


def test_control_gradients_are_bitwise_the_old_model_and_its_query_gradient_is_zero():
    reference, _ = gradients(old().train())
    control, _ = gradients(model("control").train())
    assert control["patch_query"].count_nonzero() == 0
    assert set(control) - set(reference) == {"patch_query"}
    assert all(torch.equal(control[k], reference[k]) for k in reference)
    candidate, _ = gradients(model("candidate").train())
    for k in reference:
        # Mathematically equal; floating accumulation order may differ by rounding.
        torch.testing.assert_close(candidate[k], reference[k], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("scale", [1e2, 1e4])
def test_large_queries_keep_finite_values_and_gradients_under_bf16(scale):
    net = model("candidate").train()
    randomize_query(net, scale=scale)
    torch.manual_seed(2)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = net(*window(absent=(1, 2, 3, 4, 5)))
    loss, _ = masked_loss(logits.float())
    loss.backward()
    assert torch.isfinite(logits).all()
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)


def test_absent_slots_never_reach_the_encoder_and_never_contribute():
    net = model("candidate").eval()
    randomize_query(net)
    images, features, mask, slices, positions = window(absent=(1, 2, 4))
    with torch.no_grad():
        logits = net(images, features, mask, slices, positions)
    base, patches = net.slot_tokens(images[:, :, 0], mask)
    focus = pf.focus_features(base, patches, mask, net.patch_query)
    assert (focus[~mask[:, None].expand(-1, T, -1)] == 0).all()
    poisoned = images.clone()
    poisoned[:, 1] = 0.5
    with pytest.raises(ValueError, match="Absent"):
        net(poisoned, features, mask, slices, positions)
    with pytest.raises(ValueError, match="finite"):
        pf.focus_features(base, patches.clone().fill_(float("nan")), mask, net.patch_query)
    assert torch.isfinite(logits).all()


# --- Expressivity only ----------------------------------------------------------------------


def test_equal_cls_and_patch_means_with_different_cooccurrence_are_separable():
    """The mean-pooled head cannot separate these; the focused residual can. Not accuracy."""
    e = torch.eye(D)
    patches = torch.stack(
        [
            torch.stack([e[0] + e[1], -(e[0] + e[1]), e[2], -e[2]]),
            torch.stack([e[0] - e[1], -(e[0] - e[1]), e[2], -e[2]]),
        ]
    )
    cls = torch.ones(2, D)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[:, 0] = True
    base = torch.zeros(2, 6, 2 * D)
    base[:, 0] = torch.cat([cls, patches.mean(1)], dim=1)
    assert torch.equal(base[0], base[1])
    net = model("candidate").eval()
    reference = net.head(base, mask.float())
    assert torch.equal(reference[0], reference[1])
    assert torch.equal(net.focused_logits(base, patches, mask), reference)
    with torch.no_grad():
        net.patch_query.zero_()
        net.patch_query[:, 0] = 8.0
    focused = net.focused_logits(base, patches, mask)
    assert (focused[0] - focused[1]).abs().min() > 1e-4


# --- Inference and inputs -------------------------------------------------------------------


class _Strided(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, D, 48, 48)
        self.cls = nn.Linear(3, D)

    def forward_features(self, images):
        return self.cls(images.mean((2, 3))), self.patch(images).permute(0, 2, 3, 1)


def test_mri_only_deployment_matches_ten_window_evaluation():
    source = row(4, absent=(1, 3))
    images, mask = torch.from_numpy(source["images"]), torch.from_numpy(source["slot_mask"])
    torch.manual_seed(2)
    net = pf.PatchFocusSlotStudyNet(_Strided(), D, STRATEGIES["candidate"]).eval()
    randomize_query(net)
    deployed = pf.deployment_probabilities(net, images, mask, CPU, "fp32")
    inputs = (
        ss.source_row_windows(images)[None],
        torch.zeros(1, 6, 5),
        mask[None],
        mask[None, :, None].expand(1, 6, 10),
        torch.zeros(1, 6, 10, 2),
    )
    with torch.no_grad():
        assert torch.equal(deployed, net.window_probabilities(*inputs)[0])
    assert deployed.shape == (T,) and ((deployed > 0) & (deployed < 1)).all()
    with pytest.raises(ValueError, match="patch-focus student"):
        pf.deployment_probabilities(
            ss.SourceSlotStudyNet(_Strided(), D, SLOT_STRATEGY), images, mask, CPU
        )


def test_training_inputs_are_the_source_slot_inputs():
    source = [row(seed, absent=(2, 5)) for seed in range(2)]
    images = torch.stack([torch.from_numpy(r["images"])[None].float() / 255 for r in source])
    windows = torch.stack([images[:, 0, :, s : s + 3] for s in ss.EVALUATION_VIEWS], 2)
    mask = torch.stack([torch.from_numpy(r["slot_mask"]) for r in source])
    batch = dict(
        images=windows[:, :, :, :, :64, :64].contiguous(),
        features=torch.zeros(2, 6, 5),
        series_mask=mask,
        slice_mask=mask[:, :, None].expand(2, 6, 10),
        positions=torch.zeros(2, 6, 10, 2),
        study_uid=["a", "b"],
    )
    for augmentation in (ss.AUGMENT, "none"):
        for update in range(4):
            focus = pf.training_inputs(
                batch,
                CPU,
                dict(STRATEGIES["candidate"], augmentation=augmentation),
                seed=1,
                epoch=2,
                update=update,
            )
            slot = ss.training_inputs(
                batch,
                CPU,
                dict(SLOT_STRATEGY, augmentation=augmentation),
                seed=1,
                epoch=2,
                update=update,
            )
            assert all(torch.equal(a, b) for a, b in zip(focus, slot))


def test_student_has_no_teacher_reference_or_coordinate_dependency():
    source = (ROOT / "rsna/competitive/patch_focus.py").read_text()
    assert "tools." not in source and "teacher" not in source and "positions[" not in source


# --- Canonical trainer on the engineering encoder ------------------------------------------


def _diagnostic_run(tmp_path, strategy, output, stop=None, resume=None):
    import argparse

    from test_public_triplet_training import POLICY, build, cohort

    from rsna.competitive import train
    from rsna.competitive.backbones import create_encoder
    from rsna.competitive.model import CompetitiveStudyNet

    manifest, folds = cohort()
    if not (tmp_path / "cache").exists():
        build(tmp_path / "cache", manifest, folds)
        (tmp_path / "series.csv").write_text("SeriesInstanceUID\na\n")
    series = tmp_path / "series.csv"
    release = {
        "series_sha256": hashlib.sha256(series.read_bytes()).hexdigest(),
        "source_revision": "synthetic",
    }
    config = TrainConfig(
        epochs=2,
        batch_size=1,
        accumulation_steps=2,
        slices=10,
        side=336,
        max_series=6,
        freeze_encoder=False,
        min_free_gib=0,
        diagnostic_study_limit=16,
        precision="fp16",
        public_input=POLICY,
        classifier_strategy=copy.deepcopy(strategy),
    )
    args = argparse.Namespace(
        studies=tmp_path / "unused",
        artifact_root=tmp_path,
        folds=tmp_path / "unused",
        series=series,
        cache=tmp_path / "cache",
        output=output,
        resume=resume,
        stop_after_updates=stop,
    )
    output.mkdir(parents=True, exist_ok=True)
    patch = pytest.MonkeyPatch()
    patch.setattr(train, "load_verified_release", lambda *_: (manifest.copy(), release))
    try:
        return train._run_locked(
            args,
            config,
            [False],
            0,
            1,
            CPU,
            create_encoder,
            None,
            CompetitiveStudyNet,
            lambda *_: folds.copy(),
        )
    finally:
        patch.undo()


@pytest.fixture(scope="module")
def diagnostic(tmp_path_factory):
    root = tmp_path_factory.mktemp("patch-focus-diagnostic")
    runs = {}
    for arm in ARMS:
        base = root / arm
        assert _diagnostic_run(root, STRATEGIES[arm], base / "full") == 0
        assert _diagnostic_run(root, STRATEGIES[arm], base / "replay", stop=1) == 75
        first = json.loads((base / "replay/invocation.json").read_text())["updates"]
        resume = base / "replay/checkpoint.pt"
        assert _diagnostic_run(root, STRATEGIES[arm], base / "replay", resume=resume) == 0
        runs[arm] = dict(base=base, first=first)
    assert _diagnostic_run(root, SLOT_STRATEGY, root / "slot/full") == 0
    runs["slot"] = dict(base=root / "slot")
    return runs


def _events(path):
    return [
        {k: v for k, v in e.items() if k != "elapsed_seconds"}
        for e in json.loads(path.read_text())["updates"]
    ]


@pytest.mark.parametrize("arm", ARMS)
def test_trainer_full_and_stop1_resume_have_identical_complete_state_and_csv(diagnostic, arm):
    from test_competitive_training import _assert_tree_equal

    base = diagnostic[arm]["base"]
    full = torch.load(base / "full/checkpoint.pt", weights_only=True)
    replay = torch.load(base / "replay/checkpoint.pt", weights_only=True)
    _assert_tree_equal(full, replay)
    assert "amp_scaler" in full and set(full["rng_by_rank"][0]) >= {"torch", "python", "numpy"}
    assert full["binding"]["config"]["classifier_strategy"] == STRATEGIES[arm]
    query = full["model"]["patch_query"]
    assert (query.count_nonzero() == 0) == (arm == "control")
    assert (base / "full/outer_predictions.csv").read_bytes() == (
        base / "replay/outer_predictions.csv"
    ).read_bytes()
    first = [
        {k: v for k, v in e.items() if k != "elapsed_seconds"} for e in diagnostic[arm]["first"]
    ]
    assert _events(base / "full/invocation.json") == first + _events(
        base / "replay/invocation.json"
    )


def test_trainer_arms_share_first_update_loss_rng_and_the_old_recipe(diagnostic):
    from test_competitive_training import _assert_tree_equal

    control, candidate, slot = (
        _events(diagnostic[arm]["base"] / "full/invocation.json")
        for arm in ("control", "candidate", "slot")
    )
    assert control[0]["loss"] == candidate[0]["loss"] == slot[0]["loss"]
    a, b = (
        torch.load(diagnostic[arm]["base"] / "full/checkpoint.pt", weights_only=True)
        for arm in ARMS
    )
    assert set(a["model"]) == set(b["model"])
    _assert_tree_equal(a["rng_by_rank"], b["rng_by_rank"])


# --- Routing, serialization and key rejection ----------------------------------------------


def test_one_model_routing_serves_trainer_and_checkpoint_rebuilds():
    assert pf.classifier_model(STRATEGIES["candidate"]) is pf.PatchFocusSlotStudyNet
    assert pf.classifier_model(STRATEGIES["control"]) is pf.PatchFocusSlotStudyNet
    assert pf.classifier_model(SLOT_STRATEGY) is ss.SourceSlotStudyNet
    window = json.loads(
        (
            ROOT
            / "initiative/experiments/competitive-v1/source-window-bag-v1/query-r32-config.json"
        ).read_text()
    )
    assert pf.classifier_model(window["classifier_strategy"]) is wb.WindowBagSlotStudyNet
    for broken in (
        dict(STRATEGIES["candidate"], patch_focus="other"),
        dict(SLOT_STRATEGY, extra=1),
    ):
        with pytest.raises(ValueError, match="Exact"):
            pf.classifier_model(broken)
    train_source = (ROOT / "rsna/competitive/train.py").read_text()
    assert "classifier_model(config.classifier_strategy)(" in train_source
    assert "SourceSlotStudyNet" not in train_source and "WindowBagSlotStudyNet" not in train_source


def test_strict_state_dicts_reject_missing_extra_or_reshaped_queries():
    focus, slot = model("candidate"), old()
    randomize_query(focus)
    with pytest.raises(RuntimeError, match="Unexpected key.*patch_query"):
        slot.load_state_dict(focus.state_dict(), strict=True)
    with pytest.raises(RuntimeError, match="Missing key.*patch_query"):
        model("control").load_state_dict(slot.state_dict(), strict=True)
    wrong = dict(focus.state_dict(), patch_query=torch.zeros(T, D + 1))
    with pytest.raises(RuntimeError, match="size mismatch"):
        model("control").load_state_dict(wrong, strict=True)
    rebuilt = model("control")
    rebuilt.load_state_dict(focus.state_dict(), strict=True)
    assert torch.equal(rebuilt.patch_query, focus.patch_query)


def _checkpoint(diagnostic, arm, run="full"):
    return torch.load(diagnostic[arm]["base"] / run / "checkpoint.pt", weights_only=True)


def test_trainer_checkpoints_rebuild_only_through_their_bound_strategy(diagnostic):
    from rsna.competitive.backbones import create_encoder
    from rsna.competitive.delta_checkpoint import restore_checkpoint_model

    for arm in ARMS:
        checkpoint = _checkpoint(diagnostic, arm)
        binding = checkpoint["binding"]
        assert (
            binding["code_sha256"]["competitive/patch_focus.py"]
            == hashlib.sha256((ROOT / "rsna/competitive/patch_focus.py").read_bytes()).hexdigest()
        )
        config = TrainConfig(**dict(binding["config"], encoder={"architecture": "diagnostic_tiny"}))
        three = dict(input_channels=3, input_contract="public_ordered_triplet")
        encoder, dim, provenance = create_encoder(config.encoder, **three)
        assert provenance == binding["config"]["encoder"]
        net = pf.classifier_model(config.classifier_strategy)(
            encoder, dim, config.classifier_strategy, encode_chunk_size=config.encode_chunk_size
        )
        restore_checkpoint_model(net, checkpoint, None)
        assert all(torch.equal(v, checkpoint["model"][k]) for k, v in net.state_dict().items())
        slot = ss.SourceSlotStudyNet(create_encoder(config.encoder, **three)[0], dim, SLOT_STRATEGY)
        with pytest.raises(RuntimeError, match="patch_query"):
            restore_checkpoint_model(slot, checkpoint, None)
    slot_checkpoint = torch.load(
        diagnostic["slot"]["base"] / "full/checkpoint.pt", weights_only=True
    )
    with pytest.raises(RuntimeError, match="patch_query"):
        restore_checkpoint_model(net, slot_checkpoint, None)


def test_checkpoints_refuse_cross_arm_resume_and_competition_inference(diagnostic, tmp_path):
    from rsna.competitive.checkpoint import load_checkpoint
    from rsna.competitive.infer import validate_inference_checkpoint

    control, candidate = (_checkpoint(diagnostic, arm) for arm in ARMS)
    assert control["binding"] != candidate["binding"]
    path = diagnostic["control"]["base"] / "full/checkpoint.pt"
    with pytest.raises(ValueError, match="configuration"):
        load_checkpoint(path, candidate["binding"])
    assert load_checkpoint(path, control["binding"])["binding"] == control["binding"]
    root = diagnostic["control"]["base"].parent
    with pytest.raises(ValueError, match="configuration"):
        _diagnostic_run(
            root,
            STRATEGIES["candidate"],
            tmp_path / "cross",
            resume=diagnostic["control"]["base"] / "replay/checkpoint.pt",
        )
    for checkpoint in (control, candidate):
        with pytest.raises(ValueError, match="no admitted deployment input"):
            validate_inference_checkpoint(checkpoint, allow_diagnostic=True)


# --- Encoder gradient flow under reduced precision --------------------------------------


def _encoder_gradients(net, precision, autocast_device="cpu"):
    net.train()
    torch.manual_seed(2)
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    with torch.autocast(autocast_device, dtype=dtype):
        logits = net(*window(side=64))
    loss, _ = masked_loss(logits.float())
    loss.backward()
    return {
        name: p.grad.detach().clone()
        for name, p in net.named_parameters()
        if p.requires_grad and name.startswith("encoder.")
    }


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_diagnostic_encoder_receives_gradients_through_cls_mean_and_focus(precision):
    from rsna.competitive.backbones import create_encoder

    three = dict(input_channels=3, input_contract="public_ordered_triplet")
    grads = {}
    for arm, scale in (("control", 0.0), ("candidate", 0.0), ("candidate", 1.0)):
        torch.manual_seed(3)
        encoder, dim, _ = create_encoder({"architecture": "diagnostic_tiny"}, **three)
        net = pf.classifier_model(STRATEGIES[arm])(encoder, dim, STRATEGIES[arm])
        with torch.no_grad():
            net.patch_query.copy_(torch.randn(T, dim, generator=torch.Generator().manual_seed(5)))
            net.patch_query.mul_(scale)
        grads[arm, scale] = _encoder_gradients(net, precision)
    for values in grads.values():
        assert values and all(
            torch.isfinite(g).all() and g.count_nonzero() for g in values.values()
        )
    control, zero, moved = grads["control", 0.0], grads["candidate", 0.0], grads["candidate", 1.0]
    for name in control:
        torch.testing.assert_close(zero[name], control[name], rtol=1e-3, atol=1e-6)
    assert any(not torch.allclose(moved[k], control[k]) for k in control)


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_tiny_dinov3_r32_prefix_gradients_are_finite_and_reach_every_adapted_layer(precision):
    pytest.importorskip("transformers")
    from test_source_matched_slot import configs as slot_configs
    from test_source_matched_slot import fake_create_encoder

    from rsna.competitive.adapters import configure_encoder_training

    spec = slot_configs()["r32"]["encoder"]
    grads = {}
    for arm, scale in (("control", 0.0), ("candidate", 1.0)):
        torch.manual_seed(3)
        encoder, dim, provenance = fake_create_encoder(spec)
        configure_encoder_training(encoder, True)
        net = pf.classifier_model(STRATEGIES[arm])(encoder, dim, STRATEGIES[arm])
        with torch.no_grad():
            net.patch_query.copy_(torch.randn(T, dim, generator=torch.Generator().manual_seed(5)))
            net.patch_query.mul_(scale)
        grads[arm] = _encoder_gradients(net, precision)
        trainable = {n for n, p in net.named_parameters() if p.requires_grad}
        assert set(grads[arm]) == {n for n in trainable if n.startswith("encoder.")}
        assert all(torch.isfinite(g).all() for g in grads[arm].values())
        layers = {n.split(".layer.")[1].split(".")[0] for n in grads[arm] if ".layer." in n}
        nonzero = {
            n.split(".layer.")[1].split(".")[0]
            for n, g in grads[arm].items()
            if ".layer." in n and g.count_nonzero()
        }
        assert layers and nonzero == layers
    assert any(
        not torch.equal(grads["candidate"][k], grads["control"][k]) for k in grads["control"]
    )


PROVENANCE = {"pretrained": True, "weights_sha256": "a" * 64, "config_sha256": "b" * 64}


def frozen(net):
    net.encoder.requires_grad_(False)
    return net


def test_delta_checkpoint_restores_the_learned_query_exactly():
    from rsna.competitive.delta_checkpoint import FrozenBaseDelta

    source = frozen(model("candidate"))
    randomize_query(source)
    stored = FrozenBaseDelta(source, PROVENANCE).checkpoint_model(source)
    assert "patch_query" in stored["model"]
    assert not any(name.startswith("encoder.") for name in stored["model"])
    target = frozen(model("candidate"))
    assert target.patch_query.count_nonzero() == 0
    FrozenBaseDelta(target, PROVENANCE).restore(target, stored)
    assert torch.equal(target.patch_query, source.patch_query)
    inputs = window(views=10)
    with torch.no_grad():
        assert torch.equal(
            source.eval().window_probabilities(*inputs),
            target.eval().window_probabilities(*inputs),
        )


def test_delta_checkpoints_are_refused_across_old_and_patch_focus_students():
    from rsna.competitive.delta_checkpoint import FrozenBaseDelta, restore_checkpoint_model

    focus, previous = frozen(model("candidate")), frozen(old())
    focus_state = FrozenBaseDelta(focus, PROVENANCE).checkpoint_model(focus)
    old_state = FrozenBaseDelta(previous, PROVENANCE).checkpoint_model(previous)
    with pytest.raises(ValueError, match="tensor schema"):
        restore_checkpoint_model(
            frozen(model("candidate")),
            old_state,
            FrozenBaseDelta(frozen(model("candidate")), PROVENANCE),
        )
    target = frozen(old())
    with pytest.raises(ValueError, match="tensor schema"):
        restore_checkpoint_model(target, focus_state, FrozenBaseDelta(target, PROVENANCE))
    # A delta checkpoint never loads without its frozen-base guard.
    with pytest.raises(ValueError, match="verified frozen base"):
        restore_checkpoint_model(frozen(model("candidate")), focus_state, None)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda m: m.pop("patch_query"),
        lambda m: m.update(patch_query=torch.zeros(T, D + 1)),
        lambda m: m.update(patch_query=torch.zeros(T, D, dtype=torch.float16)),
        lambda m: m.update(patch_query=torch.full((T, D), float("nan"))),
        lambda m: m.update(extra_query=torch.zeros(T, D)),
    ],
    ids=["missing", "shape", "dtype", "nonfinite", "extra"],
)
def test_tampered_patch_query_is_refused(tamper):
    from rsna.competitive.delta_checkpoint import FrozenBaseDelta

    source = frozen(model("candidate"))
    stored = FrozenBaseDelta(source, PROVENANCE).checkpoint_model(source)
    stored["model"] = dict(stored["model"])
    tamper(stored["model"])
    target = frozen(model("candidate"))
    with pytest.raises(ValueError, match="Delta learned tensor|Invalid delta learned"):
        FrozenBaseDelta(target, PROVENANCE).restore(target, stored)
    assert target.patch_query.count_nonzero() == 0


@pytest.mark.parametrize("precision", ["fp16", "bf16"])
def test_tiny_dinov3_r32_control_gradients_are_bitwise_the_source_slot_student(precision):
    pytest.importorskip("transformers")
    from test_source_matched_slot import configs as slot_configs
    from test_source_matched_slot import fake_create_encoder

    from rsna.competitive.adapters import configure_encoder_training

    spec = slot_configs()["r32"]["encoder"]
    grads = {}
    for name, strategy in (("slot", SLOT_STRATEGY), ("control", STRATEGIES["control"])):
        torch.manual_seed(3)
        encoder, dim, _ = fake_create_encoder(spec)
        configure_encoder_training(encoder, True)
        net = pf.classifier_model(strategy)(encoder, dim, strategy)
        grads[name] = _encoder_gradients(net, precision)
    assert set(grads["control"]) == set(grads["slot"])
    assert all(torch.equal(grads["control"][k], grads["slot"][k]) for k in grads["slot"])
