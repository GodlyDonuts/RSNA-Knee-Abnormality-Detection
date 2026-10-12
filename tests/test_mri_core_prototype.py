"""Synthetic CPU contracts for the isolated MRI-CORE prototype; not accuracy evidence."""

import argparse
import copy
import hashlib
import math
from pathlib import Path

import pandas as pd
import pytest
import torch
from test_competitive_training import SyntheticStudies, _assert_tree_equal
from torch.nn import functional as F

from rsna.competitive.adapters import ResidualLinear, configure_encoder_training
from rsna.competitive.train import TrainConfig
from rsna.schema import TARGETS
from scripts.research import mri_core_prototype as mri

TINY = {
    "embed_dim": 32,
    "depth": 2,
    "num_heads": 4,
    "mlp_hidden_dim": 64,
    "patch_size": 16,
    "native_grid": 2,
    "in_chans": 3,
    "layerscale": True,
    "layer_norm_eps": 1e-6,
}
POLICY = {
    "rank": 4,
    "alpha": 8.0,
    "dropout": 0.0,
    "ffn_last_layers": 1,
    "gradient_checkpointing": False,
}


def teacher_state(seed=5):
    generator = torch.Generator().manual_seed(seed)
    backbone = {
        "backbone." + name: torch.randn(shape, generator=generator) * 0.2
        for name, shape in mri.expected_checkpoint_shapes(TINY).items()
    }
    head = {"dino_head.mlp.0.weight": torch.randn(8, 32, generator=generator)}
    return {"teacher": {**backbone, **head}}


def write(path, raw):
    torch.save(raw, path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def tiny(monkeypatch, tmp_path):
    monkeypatch.setattr(mri, "PRODUCTION_CONFIG", dict(TINY))
    monkeypatch.setattr(mri, "WEIGHTS_SHA256_PREFIX", "")
    path = tmp_path / mri.CHECKPOINT_FILENAME
    digest = write(path, teacher_state())

    def spec(side=32, **changes):
        value = {
            "architecture": mri.ARCHITECTURE,
            "weights_path": str(path),
            "weights_sha256": digest,
            "source_revision": mri.PINNED_SOURCE_REVISION,
            "architecture_config": dict(TINY),
            "image_side": side,
            "input_normalization": "imagenet_mean_std",
        }
        value.update(changes)
        return value

    return spec


def reference_forward(raw, images, target_grid):
    """Fused-QKV DINOv2 CLS equations, written independently of the module.

    Shares the prototype's assumptions (CLS kept, bicubic, eps, normalization):
    regression coverage only, not parity with the upstream SAM image encoder.
    """
    s = {k[9:]: v for k, v in raw["teacher"].items() if k.startswith("backbone.")}
    mean = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
    x = (images.expand(-1, 3, -1, -1) - mean) / std
    x = F.conv2d(x, s["patch_embed.proj.weight"], s["patch_embed.proj.bias"], stride=16)
    x = x.flatten(2).transpose(1, 2)
    pos = s["pos_embed"]
    grid = pos[:, 1:].reshape(1, 2, 2, -1).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(target_grid,) * 2, mode="bicubic", align_corners=False)
    pos = torch.cat([pos[:, :1], grid.permute(0, 2, 3, 1).reshape(1, -1, 32)], 1)
    x = torch.cat([s["cls_token"].expand(len(x), -1, -1), x], 1) + pos
    for i in range(TINY["depth"]):
        p = f"blocks.{i}."
        h = F.layer_norm(x, (32,), s[p + "norm1.weight"], s[p + "norm1.bias"], 1e-6)
        qkv = F.linear(h, s[p + "attn.qkv.weight"], s[p + "attn.qkv.bias"])
        q, k, v = qkv.reshape(len(x), x.shape[1], 3, 4, 8).permute(2, 0, 3, 1, 4)
        attn = (q * 8**-0.5 @ k.transpose(-2, -1)).softmax(-1)
        h = (attn @ v).transpose(1, 2).reshape(x.shape)
        x = x + s[p + "ls1.gamma"] * F.linear(h, s[p + "attn.proj.weight"], s[p + "attn.proj.bias"])
        h = F.layer_norm(x, (32,), s[p + "norm2.weight"], s[p + "norm2.bias"], 1e-6)
        h = F.linear(
            F.gelu(F.linear(h, s[p + "mlp.fc1.weight"], s[p + "mlp.fc1.bias"])),
            s[p + "mlp.fc2.weight"],
            s[p + "mlp.fc2.bias"],
        )
        x = x + s[p + "ls2.gamma"] * h
    return F.layer_norm(x, (32,), s["norm.weight"], s["norm.bias"], 1e-6)[:, 0]


@pytest.mark.parametrize("side", [32, 16, 48])
def test_key_mapping_and_interpolation_match_fused_reference(tiny, side):
    encoder, width, provenance = mri.create_mri_core_encoder(tiny(side))
    images = torch.rand(3, 1, side, side, generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        actual = encoder(images)
        expected = reference_forward(teacher_state(), images, side // 16)
    assert width == 32 and actual.shape == (3, 32)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)
    assert provenance["position_interpolation"] == {
        "method": mri.POSITION_INTERPOLATION,
        "native_grid": 2,
        "target_grid": side // 16,
    }
    assert provenance["model_family"] == "dinov2_vit" and "dinov3" not in provenance["architecture"]
    assert provenance["discarded_checkpoint_keys"] == [
        "dino_head.mlp.0.weight",
        "backbone.mask_token",
    ]
    assert provenance["input_normalization_source_verified"] is False
    assert provenance["input_normalization_upstream_contract"] == "unknown"
    assert provenance["source_revision"] == mri.PINNED_SOURCE_REVISION
    assert provenance["forward_variant"] == mri.FORWARD_VARIANT
    assert provenance["upstream_forward_equivalence"] == "not_established"
    assert provenance["upstream_reference_parity_accepted"] is False
    assert provenance["layout_validation_scope"] == mri.LAYOUT_VALIDATION_SCOPE
    assert provenance["knee_coverage"] == provenance["competition_data_overlap"] == "unknown"
    assert provenance["config_sha256"] == mri.canonical_sha256(TINY)
    assert "mask_token" not in dict(encoder.backbone.named_parameters())


def test_position_interpolation_preserves_cls_and_native_grid():
    pos = torch.randn(1, 5, 6)
    assert torch.equal(mri.interpolate_position_embedding(pos, 2, 2), pos)
    resized = mri.interpolate_position_embedding(pos, 2, 3)
    assert resized.shape == (1, 10, 6) and torch.equal(resized[:, 0], pos[:, 0])
    constant = torch.ones(1, 5, 6)
    torch.testing.assert_close(
        mri.interpolate_position_embedding(constant, 2, 4), torch.ones(1, 17, 6)
    )
    with pytest.raises(ValueError, match="native grid"):
        mri.interpolate_position_embedding(torch.randn(1, 197, 6), 2, 3)


def _bilinear_positions(pos_embed, native_grid, target_grid):
    width = pos_embed.shape[-1]
    grid = pos_embed[:, 1:].reshape(1, native_grid, native_grid, width).permute(0, 3, 1, 2)
    grid = F.interpolate(grid, size=(target_grid,) * 2, mode="bilinear", align_corners=False)
    return torch.cat([pos_embed[:, :1], grid.permute(0, 2, 3, 1).reshape(1, -1, width)], 1)


def test_layout_checks_do_not_detect_behavioural_assumptions(tiny, monkeypatch):
    """Same keys/shapes, different behaviour: loads silently, outputs change."""
    images = torch.rand(2, 1, 48, 48, generator=torch.Generator().manual_seed(4))
    with torch.no_grad():
        baseline = mri.create_mri_core_encoder(tiny(48))[0](images)
        with monkeypatch.context() as patched:
            patched.setattr(mri, "interpolate_position_embedding", _bilinear_positions)
            bilinear = mri.create_mri_core_encoder(tiny(48))[0](images)
        other_eps = {**TINY, "layer_norm_eps": 1e-2}
        monkeypatch.setattr(mri, "PRODUCTION_CONFIG", other_eps)
        eps = mri.create_mri_core_encoder(tiny(48, architecture_config=other_eps))[0](images)
    for changed in (bilinear, eps):
        assert changed.shape == baseline.shape and torch.isfinite(changed).all()
        assert not torch.equal(changed, baseline)


def _mutations():
    def drop(raw):
        del raw["teacher"]["backbone.blocks.1.mlp.fc2.bias"]

    def extra(raw):
        raw["teacher"]["backbone.register_tokens"] = torch.zeros(1, 4, 32)

    def chunked(raw):
        raw["teacher"]["backbone.blocks.0.0.norm1.weight"] = raw["teacher"].pop(
            "backbone.blocks.0.norm1.weight"
        )

    def stray(raw):
        raw["teacher"]["student.backbone.norm.weight"] = torch.zeros(32)

    def shape(raw):
        raw["teacher"]["backbone.pos_embed"] = torch.zeros(1, 197, 32)

    def nonfinite(raw):
        raw["teacher"]["backbone.norm.bias"][0] = math.nan

    def integer(raw):
        raw["teacher"]["backbone.norm.bias"] = torch.zeros(32, dtype=torch.long)

    def top(raw):
        raw["student"] = raw["teacher"]

    return [drop, extra, chunked, stray, shape, nonfinite, integer, top]


@pytest.mark.parametrize("mutate", _mutations(), ids=lambda f: f.__name__)
def test_checkpoint_layout_fails_closed(tiny, tmp_path, mutate):
    raw = teacher_state()
    mutate(raw)
    path = tmp_path / "mutated.pth"
    digest = write(path, raw)
    with pytest.raises(ValueError, match="MRI-CORE"):
        mri.create_mri_core_encoder(tiny(weights_path=str(path), weights_sha256=digest))


@pytest.mark.parametrize(
    "changes",
    [
        {"architecture": "dinov3_l16"},
        {"architecture": "dinov2_b14_reg"},
        {"weights_sha256": "0" * 64},
        {"source_revision": "main"},
        {"source_revision": "c" * 40},
        {"input_normalization": "unit_interval"},
        {"image_side": 24},
        {"architecture_config": {**TINY, "embed_dim": 64}},
        {"architecture_config": {**TINY, "num_register_tokens": 4}},
        {"pretrained": True},
        {"adapter": {**POLICY, "rank": 0}},
    ],
)
def test_spec_fails_closed(tiny, changes):
    with pytest.raises(ValueError):
        mri.create_mri_core_encoder(tiny(**changes))


def test_production_identity_is_bound(tiny, monkeypatch):
    spec = tiny()
    with monkeypatch.context() as patched:
        patched.setattr(mri, "WEIGHTS_SHA256_PREFIX", "0ca90aeba09568f2")
        with pytest.raises(ValueError, match="recorded checkpoint identity"):
            mri.create_mri_core_encoder(spec)
    with pytest.raises(ValueError, match="single-channel"):
        mri.create_mri_core_encoder(spec, input_channels=3)
    encoder, _, _ = mri.create_mri_core_encoder(spec)
    for bad in (torch.rand(1, 3, 32, 32), torch.rand(1, 1, 32, 32) + 1, torch.rand(1, 1, 16, 16)):
        with pytest.raises(ValueError):
            encoder(bad)


def test_reference_vitb_contract_and_costs():
    shapes = mri.expected_checkpoint_shapes(mri.PRODUCTION_CONFIG)
    assert shapes["pos_embed"] == (1, 197, 768)
    assert shapes["patch_embed.proj.weight"] == (768, 3, 16, 16)
    assert sum(math.prod(s) for s in shapes.values()) == 85_817_856
    policy = {"rank": 32, "ffn_last_layers": 6}
    assert mri.adapter_trainable_parameters(mri.PRODUCTION_CONFIG, policy) == 3_833_856
    assert mri.adapter_trainable_parameters(mri.PRODUCTION_CONFIG, {**policy, "rank": 85}) == (
        10_183_680
    )
    vitb = mri.vit_forward_flops(768, 12, 3072, 576, 1)
    vitl = mri.vit_forward_flops(1024, 24, 4096, 576, 5)
    assert round(vitb / 1e9, 1) == 111.0 and round(vitl / 1e9, 1) == 385.0


def test_zero_residual_identity_adapter_updates_and_frozen_base(tiny):
    torch.manual_seed(3)
    plain, _, _ = mri.create_mri_core_encoder(tiny())
    adapted, _, provenance = mri.create_mri_core_encoder(tiny(adapter=dict(POLICY)))
    configure_encoder_training(adapted, freeze_base=True)
    images = torch.rand(4, 1, 32, 32, generator=torch.Generator().manual_seed(2))
    with torch.no_grad():
        assert torch.equal(plain(images), adapted(images))
    attention = [f"blocks.{i}.attn.{p}" for i in range(2) for p in ("q", "k", "v", "proj")]
    expected = [*attention, "blocks.1.mlp.fc1", "blocks.1.mlp.fc2"]
    assert provenance["adapter"]["target_modules"] == expected
    assert provenance["adapter"]["trainable_parameters"] == 4 * (2 * 4 * 2 * 32 + 1 * 2 * 96)
    trainable = sorted(n for n, p in adapted.named_parameters() if p.requires_grad)
    assert trainable == sorted(f"backbone.{t}.{x}" for t in expected for x in ("lora_A", "lora_B"))
    assert sum(adapted.state_dict()[n].numel() for n in trainable) == 2816
    before = copy.deepcopy(adapted.state_dict())
    optimizer = torch.optim.AdamW([p for p in adapted.parameters() if p.requires_grad], lr=0.01)
    for _ in range(2):
        optimizer.zero_grad()
        adapted(images).square().mean().backward()
        optimizer.step()
    after = adapted.state_dict()
    assert all(
        torch.equal(before[n], after[n])
        for n in before
        if not n.endswith("lora_A") and not n.endswith("lora_B")
    )
    assert all(not torch.equal(before[n], after[n]) for n in before if n.endswith("lora_B"))
    with torch.no_grad():
        assert not torch.equal(plain(images), adapted(images))
    with pytest.raises(ValueError, match="frozen"):
        configure_encoder_training(adapted, freeze_base=False)
    with pytest.raises(ValueError, match="already installed"):
        mri.install_mri_core_adapters(adapted.backbone, POLICY)


def test_gradient_checkpointing_matches_plain_gradients(tiny):
    grads = []
    for enabled in (False, True):
        torch.manual_seed(11)
        encoder, _, _ = mri.create_mri_core_encoder(
            tiny(adapter={**POLICY, "gradient_checkpointing": enabled})
        )
        configure_encoder_training(encoder, freeze_base=True)
        for module in encoder.modules():
            if isinstance(module, ResidualLinear):
                torch.nn.init.constant_(module.lora_B, 0.01)
        encoder(torch.full((2, 1, 32, 32), 0.5)).square().sum().backward()
        grads.append({n: p.grad for n, p in encoder.named_parameters() if p.requires_grad})
    assert grads[0].keys() == grads[1].keys()
    for name in grads[0]:
        torch.testing.assert_close(grads[0][name], grads[1][name], rtol=1e-6, atol=1e-7)


def test_default_paths_do_not_admit_mri_core(tiny):
    from rsna.competitive.backbones import create_encoder

    with pytest.raises(ValueError):
        create_encoder(tiny())
    with pytest.raises(ValueError, match="DINOv3-L adapters"):
        TrainConfig(
            encoder=tiny(adapter=dict(POLICY)), diagnostic_study_limit=2, side=32
        ).validate()
    package = Path(__file__).resolve().parents[1] / "rsna"
    assert not any("mri_core" in p.read_text() for p in package.rglob("*.py"))


def test_trainer_delta_full_stop_resume_is_exact(tiny, tmp_path, monkeypatch):
    from rsna.competitive import train
    from rsna.competitive.model import CompetitiveStudyNet

    rows = []
    for i in range(10):
        row = {"study_uid": f"study{i:02}", "leakage_group": f"group{i}", "split": "train"}
        for j, target in enumerate(TARGETS):
            row[target], row[target + "__mask"] = i % 2, int(j < 6 or i % 3 == 0)
            row[target + "__provenance"] = "synthetic_weak"
        rows.append(row)
    manifest = pd.DataFrame(rows)
    overlay = manifest[["study_uid", "leakage_group"]].copy()
    overlay["fold"] = [0] * 4 + [1] * 6
    series_path = tmp_path / "series.csv"
    series_path.write_text("SeriesInstanceUID\na\n")
    binding = {
        "series_sha256": hashlib.sha256(series_path.read_bytes()).hexdigest(),
        "source_revision": "synthetic",
    }
    monkeypatch.setattr(train, "load_verified_release", lambda *args: (manifest.copy(), binding))
    config = TrainConfig(
        encoder=tiny(side=16, adapter={**POLICY, "dropout": 0.1}),
        checkpoint_format="frozen_base_delta",
        freeze_encoder=True,
        diagnostic_study_limit=20,
        epochs=2,
        batch_size=2,
        accumulation_steps=2,
        slices=2,
        side=16,
        max_series=3,
        hidden_dim=16,
        encoder_lr=0.001,
        min_free_gib=0,
    )
    args = argparse.Namespace(
        studies=tmp_path / "unused",
        artifact_root=tmp_path,
        folds=tmp_path / "unused",
        series=series_path,
        cache=tmp_path,
        output=tmp_path / "full",
        resume=None,
        stop_after_updates=None,
    )

    def execute():
        return train._run_locked(
            args,
            config,
            [False],
            0,
            1,
            torch.device("cpu"),
            mri.create_mri_core_encoder,
            SyntheticStudies,
            CompetitiveStudyNet,
            lambda *args: overlay.copy(),
        )

    assert execute() == 0
    full = torch.load(args.output / "checkpoint.pt", weights_only=True)
    predictions = (args.output / "outer_predictions.csv").read_bytes()
    encoder_binding = full["binding"]["config"]["encoder"]
    assert encoder_binding["architecture"] == mri.ARCHITECTURE
    assert encoder_binding["adapter"]["target_modules"][-1] == "blocks.1.mlp.fc2"
    assert "encoder.backbone.blocks.0.attn.q.weight" not in str(full["model"].keys())
    args.output, args.stop_after_updates = tmp_path / "resumed", 1
    assert execute() == 75
    args.resume, args.stop_after_updates = args.output / "checkpoint.pt", None
    assert execute() == 0
    _assert_tree_equal(full, torch.load(args.output / "checkpoint.pt", weights_only=True))
    assert (args.output / "outer_predictions.csv").read_bytes() == predictions
