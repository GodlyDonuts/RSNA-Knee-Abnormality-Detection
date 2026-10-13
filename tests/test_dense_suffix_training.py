"""Canonical frozen-prefix dense/R32 suffix training: engineering only, never accuracy."""

import argparse
import copy
import hashlib
import io
import json
from pathlib import Path

import pandas as pd
import pytest
import torch
from test_public_triplet_training import PIXEL_GROUP, POLICY, row
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from rsna.competitive import prefix_cache as pc
from rsna.competitive import public_input as pi
from rsna.competitive import suffix_adapters as sa
from rsna.competitive.adapters import DenseDeltaLinear, ResidualLinear, configure_encoder_training
from rsna.competitive.backbones import _encoder_provenance, _GrayEncoder
from rsna.competitive.delta_checkpoint import FrozenBaseDelta
from rsna.competitive.model import CompetitiveStudyNet
from rsna.competitive.train import TrainConfig
from rsna.schema import TARGETS
from scripts.newton import run_dense_suffix_recovery as recovery
from tools.research.dense_suffix_delta import dense_delta as prototype

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / recovery.RECIPE
LEGACY = ROOT / "initiative/experiments/competitive-v1/public-input-recipe-v1"
PINNED_CONFIG = json.loads(
    (ROOT / "configs/competitive/weights/dinov3-large.config.json").read_text()
)
COMMON = dict(frozen_prefix_layers=12, dropout=0.05, gradient_checkpointing=True)
DENSE = dict(method=sa.DENSE, **COMMON)
LORA = dict(method=sa.LORA, rank=4, alpha=8.0, **COMMON)
TINY = dict(
    hidden_size=32,
    intermediate_size=64,
    num_hidden_layers=14,
    num_attention_heads=4,
    patch_size=16,
    num_register_tokens=4,
)
FAKE_SHA = {"config_sha256": "1" * 64, "weights_sha256": "2" * 64}


def official_state():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        return copy.deepcopy(DINOv3ViTModel(DINOv3ViTConfig(**TINY)).state_dict())


OFFICIAL = official_state()


def tiny_backbone():
    """Mimic the strict official loader: construct (consuming RNG), then strict-load."""
    model = DINOv3ViTModel(DINOv3ViTConfig(**TINY))
    model.load_state_dict(OFFICIAL, strict=True)
    return model


def fake_create_encoder(spec, *, input_channels=3, input_contract="public_ordered_triplet"):
    """Pinned-L stand-in: same strict-load/install/wrap order as ``create_encoder``."""
    assert input_channels == 3 and input_contract == "public_ordered_triplet"
    model = tiny_backbone()
    installed = sa.install_suffix_adapter(model, spec["suffix_adapter"])
    provenance = dict(
        FAKE_SHA,
        architecture="dinov3_l16",
        pretrained=True,
        embedding_dim=32,
        input_contract="public_ordered_triplet_0_1_imagenet_rgb_v1",
        suffix_adapter=installed,
    )
    return _GrayEncoder(model, 16, 3), 32, provenance


def encoder(policy):
    return fake_create_encoder({"suffix_adapter": policy})[0]


def windows(count=3, seed=0):
    generator = torch.Generator().manual_seed(seed)
    return torch.rand(count, 3, 32, 32, generator=generator)


@pytest.mark.parametrize("policy", [DENSE, LORA])
def test_zero_residual_suffix_path_equals_official_eval_forward(policy):
    official = tiny_backbone().eval()
    images = windows()
    model = encoder(policy).eval()
    with torch.no_grad():
        expected = official(pixel_values=model._normalized_rgb(images))
        pooled = model(images)
        sequence, features_pooled = model._backbone_output(images)
    assert torch.equal(pooled, expected.pooler_output)
    assert torch.equal(sequence, expected.last_hidden_state)
    assert torch.equal(features_pooled, pooled)


def test_canonical_graph_matches_reviewed_prototype_and_pinned_counts():
    config = DINOv3ViTConfig(**{k: PINNED_CONFIG[k] for k in TINY} | {"num_hidden_layers": 24})
    with torch.device("meta"):
        dense, lora, reviewed = (DINOv3ViTModel(config) for _ in range(3))
    record = sa.install_suffix_adapter(dense, dict(DENSE, gradient_checkpointing=False))
    control = sa.install_suffix_adapter(lora, dict(LORA, rank=32, gradient_checkpointing=False))
    old = prototype.install_dense_suffix_deltas(
        reviewed, dict(frozen_prefix_layers=12, dropout=0.05)
    )
    assert record["target_modules"] == control["target_modules"] == old["target_modules"]
    assert record["trainable_parameters"] == old["trainable_parameters"] == 150_994_944
    assert control["trainable_parameters"] == 7_077_888
    assert sum(isinstance(m, DenseDeltaLinear) for m in dense.modules()) == 72
    assert sum(isinstance(m, ResidualLinear) for m in lora.modules()) == 72
    torch.manual_seed(3)
    base = torch.nn.Linear(8, 6)
    canonical, reviewed_linear = DenseDeltaLinear(base, 0.0), prototype.DenseDeltaLinear(base, 0.0)
    with torch.no_grad():
        canonical.dense_delta.normal_()
        reviewed_linear.dense_delta.copy_(canonical.dense_delta)
    inputs = torch.randn(4, 8)
    assert torch.equal(canonical(inputs), reviewed_linear(inputs))


@pytest.mark.parametrize("policy", [DENSE, LORA])
def test_requires_grad_and_training_configuration_never_unfreeze_official_tensors(policy):
    model = encoder(policy)
    learned = {".dense_delta"} if policy["method"] == sa.DENSE else {".lora_A", ".lora_B"}
    for _ in range(2):
        model.requires_grad_(True)
        configure_encoder_training(model, True)
        trainable = {n for n, p in model.named_parameters() if p.requires_grad}
        assert trainable and all(n.endswith(tuple(learned)) for n in trainable)
        assert ".layer.0." not in "".join(trainable) and ".layer.11." not in "".join(trainable)
    model.requires_grad_(False)
    assert not any(p.requires_grad for p in model.parameters())
    with pytest.raises(ValueError, match="frozen"):
        configure_encoder_training(model, False)
    model.backbone.get_submodule("model.layer.12.attention.q_proj").weight.requires_grad_(True)
    with pytest.raises(ValueError, match="official DINOv3 tensor"):
        sa.check_trainable_partition(model.backbone)


@pytest.mark.parametrize("policy", [DENSE, LORA])
def test_train_mode_keeps_prefix_and_rope_eval_and_only_suffix_trains(policy):
    model = encoder(dict(policy, dropout=0.0))
    head = torch.nn.Linear(32, 2)
    configure_encoder_training(model, True)
    model.train()
    backbone = model.backbone
    prefix = sa.frozen_prefix_modules(backbone, 12)
    assert not any(m.training for module in prefix for m in module.modules())
    assert backbone.get_submodule("model.layer.12").training and backbone.norm.training
    fingerprint = sa.official_state_fingerprint(backbone)
    images = windows(2, seed=4)
    with torch.no_grad():
        evaluation = copy.deepcopy(model).eval()(images)
    assert torch.equal(model(images), evaluation)  # dropout 0, eval RoPE in both modes
    learned = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(learned + list(head.parameters()), lr=1e-2, weight_decay=0.1)
    for _ in range(2):
        optimizer.zero_grad()
        head(model(images)).square().mean().backward()
        assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)
        assert all(p.grad is not None for p in learned)
        optimizer.step()
    assert sa.official_state_fingerprint(backbone) == fingerprint
    state = backbone.state_dict()
    assert all(torch.equal(state[k], v) for k, v in OFFICIAL.items())


def test_prefix_is_fp32_without_autocast_or_gradients():
    model = encoder(DENSE).eval()
    pixels = model._normalized_rgb(windows(2))
    reference, _ = sa.run_prefix(model.backbone, pixels, 12)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        hidden, rope = sa.run_prefix(model.backbone, pixels, 12)
    assert hidden.dtype == torch.float32 and not hidden.requires_grad
    assert torch.equal(hidden, reference)
    with pytest.raises(ValueError, match="dtype"):
        sa.run_prefix(model.backbone, pixels.half(), 12)
    model.backbone.rope_embeddings.train()
    with pytest.raises(ValueError, match="evaluation mode"):
        sa.run_prefix(model.backbone, pixels, 12)


def test_matched_arms_share_head_initialization_and_rng_stream():
    states = []
    for policy in (DENSE, LORA):
        torch.manual_seed(20260922)
        net = CompetitiveStudyNet(encoder(policy), 32, hidden_dim=12, input_channels=3)
        head = {k: v for k, v in net.state_dict().items() if not k.startswith("encoder.")}
        states.append((head, torch.get_rng_state()))
    assert states[0][0].keys() == states[1][0].keys()
    assert all(torch.equal(states[0][0][k], states[1][0][k]) for k in states[0][0])
    assert torch.equal(states[0][1], states[1][1])


@pytest.mark.parametrize(
    "policy, match",
    [
        (dict(DENSE, frozen_prefix_layers=0), "nonempty"),
        (dict(DENSE, frozen_prefix_layers=14), "nonempty"),
        (dict(DENSE, dropout=1.0), "dropout"),
        (dict(DENSE, rank=4), "exact"),
        (dict(LORA, alpha=float("nan")), "alpha"),
        (dict(DENSE, method="dense_v0"), "explicit"),
        (dict(DENSE, gradient_checkpointing=1), "boolean"),
    ],
)
def test_invalid_suffix_policy_refused_before_installation(policy, match):
    model = tiny_backbone()
    with pytest.raises(ValueError, match=match):
        sa.install_suffix_adapter(model, policy)
    assert not any(isinstance(m, (DenseDeltaLinear, ResidualLinear)) for m in model.modules())


def test_already_adapted_or_stochastic_backbones_refused():
    model = tiny_backbone()
    sa.install_suffix_adapter(model, DENSE)
    with pytest.raises(ValueError, match="already"):
        sa.install_suffix_adapter(model, LORA)
    stochastic = DINOv3ViTModel(DINOv3ViTConfig(**TINY, drop_path_rate=0.1))
    with pytest.raises(ValueError, match="drop_path_rate"):
        sa.install_suffix_adapter(stochastic, DENSE)


def matched():
    return {arm: json.loads((RECIPE / name).read_text()) for arm, name in recovery.ARMS.items()}


def test_recipe_is_separately_versioned_and_differs_only_in_policy_and_lr():
    configs = recovery.load_recipe(RECIPE)
    legacy = json.loads((LEGACY / "public-triplet-config.json").read_text())
    for arm, values in configs.items():
        expected = dict(legacy, encoder_lr=recovery.PINNED[arm]["lr"])
        expected["encoder"] = {k: v for k, v in legacy["encoder"].items() if k != "adapter"}
        expected["encoder"]["suffix_adapter"] = values["encoder"]["suffix_adapter"]
        assert values == expected
        assert values["encoder"]["suffix_adapter"]["frozen_prefix_layers"] == 12
    assert configs["dense"]["encoder"]["suffix_adapter"] == dict(DENSE)
    assert configs["suffix_r32"]["encoder"]["suffix_adapter"] == dict(LORA, rank=32)
    spec = {k: v for k, v in legacy["encoder"].items() if k != "adapter"}
    provenance = {
        arm: sa.suffix_adapter_provenance(v["encoder"]["suffix_adapter"], PINNED_CONFIG)
        for arm, v in configs.items()
    }
    assert {a: p["trainable_parameters"] for a, p in provenance.items()} == {
        "suffix_r32": 7_077_888,
        "dense": 150_994_944,
    }
    old = _encoder_provenance(legacy["encoder"], PINNED_CONFIG, "ConvNextForImageClassification")
    plain = _encoder_provenance(spec, PINNED_CONFIG, "ConvNextForImageClassification")
    assert "suffix_adapter" not in old and "suffix_adapter" not in plain
    assert old["adapter"]["method"] == "rs_lora_v1"


@pytest.mark.parametrize(
    "change, match",
    [
        (lambda v: v["encoder"].update(adapter=dict(rank=32)), "adapter"),
        (lambda v: v.update(freeze_encoder=False), "frozen"),
        (lambda v: v.update(checkpoint_format="full"), "Suffix adapters"),
        (lambda v: v["encoder"].update(architecture="dinov3_b16"), "."),
        (lambda v: v["encoder"]["suffix_adapter"].update(rank=32), "exact"),
        (lambda v: v["encoder"]["suffix_adapter"].update(frozen_prefix_layers=24), "nonempty"),
    ],
)
def test_trainer_config_refuses_mixed_or_unfrozen_suffix_recipes(change, match):
    values = copy.deepcopy(matched()["dense"])
    change(values)
    with pytest.raises(ValueError, match=match):
        TrainConfig(**values).validate()


def test_old_recipes_defaults_and_bindings_are_unchanged(tmp_path):
    assert "suffix_adapter" not in TrainConfig().encoder
    for name in ("public-triplet-config.json", "control-config.json"):
        values = json.loads((LEGACY / name).read_text())
        assert "suffix_adapter" not in values["encoder"]
        TrainConfig(**values).validate()
    with pytest.raises(ValueError, match="Matched arms"):
        changed = matched()
        changed["dense"]["seed"] += 1
        for arm, name in recovery.ARMS.items():
            (tmp_path / name).write_text(json.dumps(changed[arm]))
        recovery.load_recipe(tmp_path)


def test_engineering_config_binds_only_mounted_paths_and_fixed_scope():
    recipe = matched()["dense"]
    spec = {k: v for k, v in recipe["encoder"].items() if k != "suffix_adapter"}
    spec.update(weights_path="/mnt/w.safetensors", config_path="/mnt/c.json")
    values = recovery.engineering_config(recipe, spec)
    assert values["encoder"]["suffix_adapter"] == recipe["encoder"]["suffix_adapter"]
    assert {k for k in values if values[k] != recipe.get(k)} == {"encoder", *recovery.ENGINEERING}
    for bad in (dict(spec, weights_sha256="0" * 64), dict(spec, adapter={})):
        with pytest.raises(ValueError, match="Mounted"):
            recovery.engineering_config(recipe, bad)


# --- Exact prefix records -------------------------------------------------------------


def public_cache(tmp_path, count, absent=(1, 2, 3, 4, 5)):
    rows = {f"s{i:02d}": row(i, absent=absent) for i in range(count)}
    uids = sorted(rows)
    evaluation = uids[: count // 2]
    pi.write_public_triplet_cache(
        tmp_path,
        rows,
        pixel_group=copy.deepcopy(PIXEL_GROUP),
        source_revision="synthetic",
        fitting_uids=uids[count // 2 :],
        evaluation_uids=evaluation,
    )
    return uids, evaluation


def cache_item(tmp_path, uid="s00"):
    data = pi.PublicTripletStudyDataset(
        pd.DataFrame({"study_uid": [uid]}),
        pd.DataFrame(),
        tmp_path,
        public_input=POLICY,
        source_revision="synthetic",
        slices=10,
        side=336,
        max_series=6,
    )
    member = pc.member_identity(data.index, uid)
    source = pc.source_identity(POLICY, data.index_sha256, "synthetic")
    return data[0], member, source


def test_prefix_record_reproduces_online_suffix_and_binds_full_identity(tmp_path):
    public_cache(tmp_path, 2, absent=(1, 2, 3, 4))
    item, member, source = cache_item(tmp_path)
    model = encoder(DENSE).eval()
    with torch.no_grad():
        model.backbone.get_submodule("model.layer.13.mlp.up_proj").dense_delta.normal_()
    record = pc.build_record(model, item, member, source)
    assert record["hidden"].dtype == torch.float32
    assert record["hidden"].shape == (20, 446, 32)
    assert record["window_slots"].tolist() == [[s, v] for s in (0, 5) for v in range(10)]
    with torch.no_grad():
        online = model(item["images"][item["slice_mask"]])
        cached = pc.run_cached_suffix(record, model, member, source)
    assert torch.equal(cached, online)
    pc.verify_by_recompute(record, model, item, member, source)
    identity = record["identity"]
    assert identity["member"]["sha256"] == member["sha256"]
    assert identity["source"]["public_input_sha256"] == pi.canonical_sha256(POLICY)
    assert identity["prefix_execution"] == sa.PREFIX_EXECUTION


def _redigest(record):
    record["digests"] = {k: pc.tensor_sha256(record[k]) for k in pc.TENSOR_KEYS}


def test_prefix_record_tampering_and_lossy_casts_refused(tmp_path):
    public_cache(tmp_path, 2, absent=(1, 2, 3, 4, 5))
    item, member, source = cache_item(tmp_path)
    model = encoder(LORA).eval()
    record = pc.build_record(model, item, member, source)
    cases = []
    corrupt = dict(record, hidden=record["hidden"].clone())
    corrupt["hidden"][0, 0, 0] += 1
    cases.append((corrupt, member, source, "hidden identity"))
    half = dict(record, hidden=record["hidden"].half())
    _redigest(half)
    cases.append((half, member, source, "dtype"))
    slots = dict(record, window_slots=record["window_slots"].flip(0))
    _redigest(slots)
    cases.append((slots, member, source, "slot"))
    cases.append((record, dict(member, sha256="0" * 64), source, "identity mismatch"))
    cases.append((record, member, dict(source, index_sha256="0" * 64), "identity mismatch"))
    missing = dict(record, digests={k: v for k, v in record["digests"].items() if k != "rope_cos"})
    cases.append((missing, member, source, "digest set"))
    for candidate, bound_member, bound_source, match in cases:
        with pytest.raises(ValueError, match=match):
            pc.validate_record(candidate, model, bound_member, bound_source)
    forged = dict(record, hidden=record["hidden"].clone())
    forged["hidden"].mul_(1.5)
    _redigest(forged)
    pc.validate_record(forged, model, member, source)  # self-consistent forgery...
    with pytest.raises(ValueError, match="recomputed"):
        pc.verify_by_recompute(forged, model, item, member, source)  # ...needs recompute
    with torch.no_grad():
        model.backbone.get_submodule("model.layer.0.attention.q_proj").weight.add_(1e-3)
    with pytest.raises(ValueError, match="identity mismatch"):
        pc.validate_record(record, model, member, source)
    augmented = dict(source, augment=True)
    with pytest.raises(ValueError, match="non-augmented"):
        pc.build_record(model, item, member, augmented)


def test_storage_measurement_reads_actual_members_and_rosters(tmp_path):
    uids, evaluation = public_cache(tmp_path, 4, absent=(2, 3, 4, 5))
    measured = pc.measure_storage(
        tmp_path, POLICY, "synthetic", PINNED_CONFIG, {"evaluation": evaluation}
    )
    assert measured["bytes_per_window_fp32"] == 446 * 1024 * 4 == 1_826_816
    assert measured["all"]["acquired_windows"] == 4 * 2 * 10
    assert measured["all"]["hidden_bytes_fp32"] == 80 * 1_826_816
    assert measured["all"]["dense_6x10_bound_bytes_fp32"] == 4 * 60 * 1_826_816
    assert measured["evaluation"]["members"] == 2 and measured["cache_written"] is False
    with pytest.raises(ValueError, match="roster"):
        pc.measure_storage(tmp_path, POLICY, "synthetic", PINNED_CONFIG, {"evaluation": uids})
    member = tmp_path / pi.CACHE_DIRECTORY / "members" / f"{uids[0]}.npz"
    member.write_bytes(member.read_bytes()[:-1] + b"\0")
    with pytest.raises(ValueError, match="bytes differ"):
        pc.measure_storage(tmp_path, POLICY, "synthetic", PINNED_CONFIG)


# --- Canonical trainer through the immutable recovery wrapper --------------------------

TINY_EXPECTED = {
    "suffix_r32": dict(
        method=sa.LORA, trainable_parameters=2 * 32 * (4 * 2 * 32 + 2 * 96), lr=1e-4
    ),
    "dense": dict(method=sa.DENSE, trainable_parameters=2 * (4 * 32 * 32 + 2 * 32 * 64), lr=1e-5),
}


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
def recovered(tmp_path_factory):
    from rsna.competitive import train

    base = tmp_path_factory.mktemp("dense-suffix-recovery")
    manifest, folds = trainer_cohort()
    rows = {uid: row(i, absent=(1, 2, 3, 4, 5)) for i, uid in enumerate(manifest.study_uid)}
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
    spec = {k: v for k, v in matched()["dense"]["encoder"].items() if k != "suffix_adapter"}
    (base / "spec.json").write_text(json.dumps(spec))
    source_manifest = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "rsna").rglob("*.py"))
    }
    (base / "source-manifest.json").write_text(json.dumps(source_manifest))
    calls = []

    def executor(root, phase, config_path, extra):
        """In-process canonical ``_run_locked``; CPU stand-ins replace only the release,
        official weights and the private fitting overlay (unavailable synthetically)."""
        options = dict(zip(extra[::2], extra[1::2]))
        values = json.loads(config_path.read_text())
        values["fitting_supervision"] = None
        calls.append((root.name, phase))
        args = argparse.Namespace(
            studies=base / "unused",
            artifact_root=base,
            folds=base / "unused",
            series=series,
            cache=base / "cache",
            output=Path(options["--output"]),
            resume=Path(options["--resume"]) if "--resume" in options else None,
            stop_after_updates=int(options.get("--stop-after-updates", 0)) or None,
        )
        return train._run_locked(
            args,
            TrainConfig(**values),
            [False],
            0,
            1,
            torch.device("cpu"),
            fake_create_encoder,
            None,
            CompetitiveStudyNet,
            lambda *_: folds.copy(),
        )

    patch = pytest.MonkeyPatch()
    patch.setattr(train, "load_verified_release", lambda *_: (manifest.copy(), release))
    args = argparse.Namespace(
        source=ROOT,
        source_manifest=base / "source-manifest.json",
        encoder_spec=base / "spec.json",
        artifact_root=base,
        cache=base / "cache",
        overlay=base / "unused",
        output=base / "recovery",
        device="cpu",
        max_output_bytes=2**30,
        remaining_seconds=600.0,
    )
    try:
        result = recovery.run(args, executor, TINY_EXPECTED)
    finally:
        patch.undo()
    return dict(
        result=result, output=args.output, calls=calls, manifest=manifest, base=base, args=args
    )


def test_wrapper_accepts_both_arms_with_complete_exact_recovery(recovered):
    result = recovered["result"]
    assert recovered["calls"] == [
        (arm, phase) for arm in ("suffix_r32", "dense") for phase in ("full", "stop", "resume")
    ]
    assert result["status"] == "actual_matched_dense_suffix_full_stop1_resume_recovery_accepted"
    assert result["diagnostic_only"] and not result["competition_submission"]
    for arm, summary in result["arms"].items():
        assert summary["trainable_encoder_parameters"] == TINY_EXPECTED[arm]["trainable_parameters"]
        assert [r["optimizer_steps"] for r in summary["finite_updates"]] == [1, 2, 3, 4]
    accepted = json.loads((recovered["output"] / "accepted.json").read_text())
    assert accepted == result
    with pytest.raises(FileExistsError):
        recovery.run(recovered["args"], lambda *_: 0, TINY_EXPECTED)


@pytest.mark.parametrize("arm", ["suffix_r32", "dense"])
def test_frozen_base_is_unchanged_through_optimizer_resume_and_inference(recovered, arm):
    from rsna.competitive import train

    root = recovered["output"] / arm
    checkpoint = torch.load(root / "full/checkpoint.pt", weights_only=True)
    config = TrainConfig(**json.loads((root / "config.json").read_text()))
    torch.manual_seed(0)
    model = CompetitiveStudyNet(
        fake_create_encoder(config.encoder)[0], 32, hidden_dim=config.hidden_dim, input_channels=3
    )
    configure_encoder_training(model.encoder, True)
    guard = FrozenBaseDelta(model, checkpoint["binding"]["config"]["encoder"])
    assert guard.metadata()["frozen_tensors"] == checkpoint["model_storage"]["frozen_tensors"]
    guard.restore(model, checkpoint)
    backbone_state = model.encoder.backbone.state_dict()
    assert all(torch.equal(backbone_state[k], v) for k, v in OFFICIAL.items())
    expected = pd.read_csv(root / "full/outer_predictions.csv")
    manifest = recovered["manifest"]
    outer = manifest[manifest.study_uid.isin(expected.study_uid)].reset_index(drop=True)
    data = pi.PublicTripletStudyDataset(
        outer,
        pd.DataFrame(),
        recovered["base"] / "cache",
        public_input=POLICY,
        source_revision="synthetic",
        slices=10,
        side=336,
        max_series=6,
    )
    from rsna.competitive.data import collate_studies

    loader = torch.utils.data.DataLoader(data, batch_size=1, collate_fn=collate_studies)
    predictions, _ = train.predict(model, loader, torch.device("cpu"), config)
    inferred = pd.read_csv(io.StringIO(predictions.to_csv(index=False)))
    pd.testing.assert_frame_equal(inferred, expected, check_exact=True)


def _tampered(recovered, arm, edit):
    root = recovered["output"] / arm
    copy_root = recovered["base"] / f"tamper-{arm}-{len(list(recovered['base'].glob('tamper-*')))}"
    for name in ("full", "replay"):
        (copy_root / name).mkdir(parents=True)
        for path in (root / name).iterdir():
            if path.is_file():
                (copy_root / name / path.name).write_bytes(path.read_bytes())
    resumed = torch.load(copy_root / "replay/checkpoint.pt", weights_only=True)
    first = {"updates": json.loads((root / "full/invocation.json").read_text())["updates"][:1]}
    partial = copy.deepcopy(resumed)
    partial.update(epoch=0, next_update=1, step=1)
    edit(copy_root, resumed, partial, first)
    torch.save(resumed, copy_root / "replay/checkpoint.pt")
    return recovery.compare_recovery(copy_root, partial, first, TINY_EXPECTED[arm])


def _first_state(resumed, key):
    return next(iter(resumed["optimizer"]["state"].values()))[key]


@pytest.mark.parametrize(
    "edit",
    [
        lambda root, c, p, f: _first_state(c, "exp_avg").view(-1)[0].add_(1e-6),
        lambda root, c, p, f: c["optimizer"]["param_groups"][1].update(lr=1.0),
        lambda root, c, p, f: c["rng_by_rank"][0].update(python=(3, (1,) * 625, None)),
        lambda root, c, p, f: c["rng_by_rank"][0]["torch"].view(-1)[0].add_(1),
        lambda root, c, p, f: c["amp_scaler"].update(scale=1.0),
        lambda root, c, p, f: next(iter(c["model"].values())).view(-1)[0].add_(1e-6),
        lambda root, c, p, f: p["model_storage"]["frozen_tensors"][
            next(iter(p["model_storage"]["frozen_tensors"]))
        ].update(sha256="0" * 64),
        lambda root, c, p, f: f["updates"][0].update(loss=0.0),
        lambda root, c, p, f: (root / "replay/outer_predictions.csv").write_bytes(
            (root / "replay/outer_predictions.csv").read_bytes() + b"\n"
        ),
        lambda root, c, p, f: p.update(step=2),
    ],
)
def test_wrapper_refuses_any_state_history_base_or_prediction_difference(recovered, edit):
    _tampered(recovered, "dense", lambda *a: None)  # untampered copy is accepted
    with pytest.raises(ValueError):
        _tampered(recovered, "dense", edit)


def test_wrapper_refuses_wrong_declared_graph_and_unmatched_bindings(recovered):
    with pytest.raises(ValueError, match="Trainable graph"):
        _tampered_expected = dict(TINY_EXPECTED["dense"], trainable_parameters=1)
        root = recovered["output"] / "dense"
        resumed = torch.load(root / "replay/checkpoint.pt", weights_only=True)
        partial = copy.deepcopy(resumed)
        partial.update(epoch=0, next_update=1, step=1)
        first = {"updates": json.loads((root / "full/invocation.json").read_text())["updates"][:1]}
        recovery.compare_recovery(root, partial, first, _tampered_expected)
    bindings = {
        arm: torch.load(recovered["output"] / arm / "full/checkpoint.pt", weights_only=True)[
            "binding"
        ]
        for arm in recovery.ARMS
    }
    recovery.compare_matched(bindings)
    bindings["dense"]["training_uids_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="Matched arms"):
        recovery.compare_matched(bindings)


def test_wrapper_refuses_changed_source_before_any_output(tmp_path):
    manifest = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted((ROOT / "rsna").rglob("*.py"))
    }
    name = "rsna/competitive/suffix_adapters.py"
    for bad in ({**manifest, name: "0" * 64}, {k: v for k, v in manifest.items() if k != name}):
        (tmp_path / "manifest.json").write_text(json.dumps(bad))
        args = recovery.parser().parse_args(
            [
                *("--source", str(ROOT), "--source-manifest", str(tmp_path / "manifest.json")),
                *("--encoder-spec", "x", "--artifact-root", "x", "--cache", "x"),
                *("--overlay", "x", "--output", str(tmp_path / "out")),
                *("--device", "cpu", "--remaining-seconds", "10"),
            ]
        )
        with pytest.raises(ValueError, match="manifest"):
            recovery.run(args, lambda *_: 0)
        assert not (tmp_path / "out").exists()


def test_uint8_member_is_divided_once_and_imagenet_normalized_once(tmp_path):
    """v2 input contract: uint8 -> dataset /255 -> encoder ImageNet exactly once."""
    public_cache(tmp_path, 2, absent=(1, 2, 3, 4, 5))
    item, _, _ = cache_item(tmp_path)
    member = row(0, absent=(1, 2, 3, 4, 5))["images"]
    windows_u8 = torch.stack(
        [torch.from_numpy(member[:, start : start + 3]) for start in range(10)], dim=1
    )
    images = item["images"][item["slice_mask"]]
    assert torch.equal(images, windows_u8[item["slice_mask"]].float().div(255))
    model = encoder(DENSE).eval()
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    once = (images - mean) / std
    assert torch.equal(model._normalized_rgb(images), once)
    official = tiny_backbone().eval()
    with torch.no_grad():
        assert torch.equal(model(images), official(pixel_values=once).pooler_output)
    undivided = windows_u8[item["slice_mask"]].float()
    for wrong in (undivided, once):  # /255 omitted, or ImageNet applied twice
        with pytest.raises(ValueError, match=r"\[0,1\]"):
            model(wrong)
