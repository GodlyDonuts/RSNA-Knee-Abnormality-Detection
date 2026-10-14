import copy
import json
import math
import shutil

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from rsna.competitive import weight_average as wa
from rsna.competitive.adapters import DenseDeltaLinear, ResidualLinear, adapter_provenance
from rsna.competitive.checkpoint import (
    atomic_json,
    file_sha256,
    load_checkpoint,
    save_checkpoint,
)
from rsna.competitive.delta_checkpoint import FrozenBaseDelta, restore_checkpoint_model
from rsna.competitive.snapshots import write_epoch_snapshot

WIDTH, INNER, DEPTH, RANK, ALPHA = 24, 48, 2, 2, 3.0
DIMS = {"num_hidden_layers": DEPTH, "hidden_size": WIDTH, "intermediate_size": INNER}
STEPS_PER_EPOCH = 7


class Layer(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(WIDTH)
        self.attention = nn.Module()
        for part in "qkvo":
            setattr(self.attention, f"{part}_proj", nn.Linear(WIDTH, WIDTH))
        self.mlp = nn.Module()
        self.mlp.up_proj = nn.Linear(WIDTH, INNER)
        self.mlp.down_proj = nn.Linear(INNER, WIDTH)

    def forward(self, x):
        h, a = self.norm(x), self.attention
        x = x + a.o_proj(a.q_proj(h) * torch.sigmoid(a.k_proj(h)) + a.v_proj(h))
        return x + self.mlp.down_proj(F.gelu(self.mlp.up_proj(h)))


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_tokens = nn.Parameter(torch.randn(1, 4, WIDTH))
        self.model = nn.Module()
        self.model.layer = nn.ModuleList(Layer() for _ in range(DEPTH))
        self.calls = 0

    def forward(self, x):
        self.calls += 1
        x = torch.cat([self.register_tokens.expand(len(x), -1, -1), x], 1)
        for layer in self.model.layer:
            x = layer(x)
        return x[:, 0]


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = Backbone()
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None])
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225])[None, :, None])

    def forward(self, x):
        return self.backbone((x - self.mean[:, :1]) / self.std[:, :1])


class Net(nn.Module):
    def __init__(self, method, extra_buffer=False):
        super().__init__()
        torch.manual_seed(5)
        self.encoder = Encoder()
        self.norm = nn.LayerNorm(WIDTH)
        self.head = nn.Linear(WIDTH, 3)
        if extra_buffer:
            self.register_buffer("head_running_mean", torch.zeros(3))
        self.encoder.requires_grad_(False)
        modules = dict(self.encoder.backbone.named_modules())
        for name in provenance(method)[method_key(method)]["target_modules"]:
            parent, attribute = name.rsplit(".", 1)
            base = modules[name]
            residual = (
                DenseDeltaLinear(base, 0.0)
                if method == wa.DENSE_METHOD
                else ResidualLinear(base, RANK, ALPHA, 0.0)
            )
            setattr(modules[parent], attribute, residual)

    def forward(self, x):
        return self.head(self.norm(self.encoder(x)))


def method_key(method):
    return "adapter" if method == "rs_lora_v1" else "suffix_adapter"


def provenance(method, weights="a"):
    base = {"pretrained": True, "weights_sha256": weights * 64, "config_sha256": "b" * 64}
    if method == "rs_lora_v1":
        policy = {
            "rank": RANK,
            "alpha": ALPHA,
            "dropout": 0.0,
            "ffn_last_layers": 1,
            "gradient_checkpointing": False,
        }
        return {**base, "adapter": adapter_provenance(policy, DIMS)}
    names = [
        f"model.layer.1.{group}.{part}_proj"
        for group, parts in (("attention", "qkvo"), ("mlp", ("up", "down")))
        for part in parts
    ]
    return {**base, "suffix_adapter": {"method": method, "scaling": 1.0, "target_modules": names}}


def binding(method, fold=0, seed=1, weights="a", fitting=None):
    return {
        "config": {
            "epochs": 10,
            "checkpoint_format": "frozen_base_delta",
            "freeze_encoder": True,
            "submission_every_epochs": 1,
            "diagnostic_study_limit": None,
            "fold": fold,
            "seed": seed,
            "fitting_supervision": fitting,
            "encoder": provenance(method, weights),
        },
        "release": {"binding_sha256": "c" * 64},
        "code_sha256": {"competitive/train.py": "d" * 64},
        "world_size": 1,
    }


def make_run(root, method="rs_lora_v1", status="completed", **kwargs):
    bound = binding(method, **kwargs)
    model = Net(method)
    guard = FrozenBaseDelta(model, bound["config"]["encoder"])
    generator = torch.Generator().manual_seed(100 + kwargs.get("seed", 1))
    for epoch in range(1, 11):
        with torch.no_grad():
            for _, parameter in model.named_parameters():
                if parameter.requires_grad:
                    parameter.copy_(torch.randn(parameter.shape, generator=generator) * 0.3)
        state = {
            **guard.checkpoint_model(model),
            "binding": bound,
            "optimizer_boundary": True,
            "optimizer": {"state": {}, "param_groups": []},
            "epoch": epoch,
            "next_update": 0,
            "step": epoch * STEPS_PER_EPOCH,
            "rng_by_rank": [{}],
            "scheduled_epoch": True,
        }
        save_checkpoint(root / "checkpoint.pt", state, min_free_gib=0.0)
        write_epoch_snapshot(
            root / "checkpoint.pt", root / "epoch-checkpoints", bound, epoch, state["step"]
        )
    atomic_json(
        root / "run.json",
        {
            "status": status,
            "binding": bound,
            "diagnostic_run": False,
            "held_out_confirmation_and_gold_used": False,
        },
    )
    return bound


def snapshot(root, epoch):
    return torch.load(root / "epoch-checkpoints" / f"epoch-{epoch:03d}.pt", weights_only=True)


def effective(module):
    if isinstance(module, DenseDeltaLinear):
        return module.weight.double() + module.dense_delta.double()
    return module.weight.double() + module.scaling * (
        module.lora_B.double() @ module.lora_A.double()
    )


@pytest.mark.parametrize("method", ["rs_lora_v1", wa.DENSE_METHOD])
def test_effective_projection_is_exact_mean_of_five_adapted_weights(tmp_path, method):
    make_run(tmp_path, method)
    sources = [snapshot(tmp_path, epoch)["model"] for epoch in wa.AVERAGED_EPOCHS]
    fresh = Net(method)
    base = copy.deepcopy(fresh.state_dict())
    candidate = wa.average_snapshots(tmp_path)
    wa.apply_candidate(fresh, candidate)
    modules = dict(fresh.named_modules())
    targets = candidate["representation"]["modules"]
    assert len(targets) == (10 if method == "rs_lora_v1" else 6)
    scaling = ALPHA / math.sqrt(RANK)
    for name in targets:
        module = modules[name]
        w0 = base[name + ".weight"].double()
        if method == wa.DENSE_METHOD:
            adapted = [w0 + s[name + ".dense_delta"].double() for s in sources]
        else:
            assert module.lora_A.shape == (5 * RANK, module.in_features)
            assert module.scaling == pytest.approx(scaling / 5)
            adapted = [
                w0 + scaling * (s[name + ".lora_B"].double() @ s[name + ".lora_A"].double())
                for s in sources
            ]
            factor_average = w0 + scaling * (
                torch.stack([s[name + ".lora_B"].double() for s in sources]).mean(0)
                @ torch.stack([s[name + ".lora_A"].double() for s in sources]).mean(0)
            )
            assert not torch.allclose(factor_average, torch.stack(adapted).mean(0), atol=1e-3)
        expected = torch.stack(adapted).mean(0)
        torch.testing.assert_close(effective(module), expected, rtol=0, atol=1e-6)
        x = torch.randn(5, module.in_features)
        torch.testing.assert_close(
            module(x).double(),
            F.linear(x.double(), expected, base[name + ".bias"].double()),
            rtol=0,
            atol=1e-5,
        )
        assert torch.equal(module.bias, base[name + ".bias"])
    after = fresh.state_dict()
    for name, value in base.items():
        if name.startswith("encoder.") and not name.endswith((".lora_A", ".lora_B", "_delta")):
            assert torch.equal(after[name], value), name
    assert torch.equal(after["encoder.mean"], base["encoder.mean"])
    assert torch.equal(
        after["encoder.backbone.register_tokens"], base["encoder.backbone.register_tokens"]
    )
    for name in ("norm.weight", "norm.bias", "head.weight", "head.bias"):
        torch.testing.assert_close(
            after[name], torch.stack([s[name] for s in sources]).mean(0), rtol=0, atol=1e-7
        )
    assert not any(p.requires_grad for p in fresh.parameters()) and not fresh.training


def test_epoch10_identity_and_single_backbone_forward(tmp_path):
    make_run(tmp_path)
    epoch10 = Net("rs_lora_v1")
    source = snapshot(tmp_path, 10)
    FrozenBaseDelta(epoch10, source["binding"]["config"]["encoder"]).restore(epoch10, source)
    identity = Net("rs_lora_v1")
    wa.apply_candidate(identity, wa.average_snapshots(tmp_path, wa.IDENTITY_CONTROL_EPOCHS))
    averaged = Net("rs_lora_v1")
    wa.apply_candidate(averaged, wa.average_snapshots(tmp_path))
    x = torch.randn(3, 6, WIDTH)
    epoch10.eval()
    with torch.no_grad():
        assert torch.equal(identity(x), epoch10(x))
        logits = averaged(x)
    assert averaged.encoder.backbone.calls == 1 and torch.isfinite(logits).all()
    assert not torch.equal(logits, epoch10(x))
    assert getattr(averaged, wa._ATTRIBUTE)["role"] == "uniform_epoch6_10_average"


def test_storage_bound_written_once_and_sources_intact(tmp_path):
    make_run(tmp_path)
    directory = tmp_path / "epoch-checkpoints"
    before = {path.name: file_sha256(path) for path in sorted(directory.iterdir())}
    candidate = wa.average_snapshots(tmp_path)
    receipt = wa.write_candidate(candidate, tmp_path / "average" / "candidate.pt")
    wa.verify_sources_intact(tmp_path, candidate)
    assert before == {path.name: file_sha256(path) for path in sorted(directory.iterdir())}
    storage = receipt["storage"]
    assert storage["candidate_residual_parameters"] == 5 * storage["source_residual_parameters"]
    assert storage["candidate_residual_parameters"] < storage["dense_equivalent_parameters"]
    assert receipt["candidate_bytes"] < 2 * storage["candidate_tensor_bytes"] + 2**20
    with pytest.raises(ValueError, match="replace"):
        wa.write_candidate(candidate, tmp_path / "average" / "candidate.pt")
    loaded = wa.load_candidate(tmp_path / "average" / "candidate.pt", receipt["candidate_sha256"])
    assert loaded["model_sha256"] == candidate["model_sha256"]


def test_canonical_dinov3_large_rank32_storage_bound():
    dims = {"num_hidden_layers": 24, "hidden_size": 1024, "intermediate_size": 4096}
    policy = {
        "rank": 32,
        "alpha": 8,
        "dropout": 0.0,
        "ffn_last_layers": 12,
        "gradient_checkpointing": False,
    }
    source = adapter_provenance(policy, dims)["trainable_parameters"]
    dense = 24 * 4 * 1024 * 1024 + 12 * 2 * 1024 * 4096
    assert (source, 5 * source, dense) == (10_223_616, 51_118_080, 201_326_592)
    assert 5 * source * 4 == 204_472_320 and 5 * source < dense // 3


def test_inference_only_candidate_refuses_optimizer_resume(tmp_path):
    bound = make_run(tmp_path)
    candidate = wa.average_snapshots(tmp_path)
    path = tmp_path / "candidate.pt"
    wa.write_candidate(candidate, path)
    assert not {"optimizer", "rng_by_rank", "amp_scaler", "binding"} & set(candidate)
    with pytest.raises(ValueError):
        load_checkpoint(path, bound)
    model = Net("rs_lora_v1")
    guard = FrozenBaseDelta(model, bound["config"]["encoder"])
    with pytest.raises(ValueError):
        restore_checkpoint_model(model, candidate, guard)
    wa.apply_candidate(model, candidate)
    with pytest.raises(ValueError, match="immutable encoder base"):
        FrozenBaseDelta(model, bound["config"]["encoder"])
    with pytest.raises(ValueError):
        wa.apply_candidate(model, candidate)
    tampered = dict(candidate)
    tampered["optimizer"] = {}
    torch.save(tampered, tmp_path / "tampered.pt")
    with pytest.raises(ValueError, match="inference-only"):
        wa.load_candidate(tmp_path / "tampered.pt", file_sha256(tmp_path / "tampered.pt"))
    with pytest.raises(ValueError, match="bytes"):
        wa.load_candidate(path, "0" * 64)


def _swap(target, donor, epoch=8):
    for suffix in (".pt", ".json"):
        name = f"epoch-{epoch:03d}{suffix}"
        (target / "epoch-checkpoints" / name).unlink()
        shutil.copy(donor / "epoch-checkpoints" / name, target / "epoch-checkpoints" / name)


@pytest.mark.parametrize(
    "donor",
    [{"seed": 2}, {"fold": 1}, {"weights": "e"}, {"fitting": {"source": "public_v2"}}],
    ids=["mixed-run", "mixed-fold", "mixed-base", "mixed-label-policy"],
)
def test_mixed_snapshots_fail_closed(tmp_path, donor):
    make_run(tmp_path / "a")
    make_run(tmp_path / "b", **donor)
    _swap(tmp_path / "a", tmp_path / "b")
    with pytest.raises(ValueError, match="differ"):
        wa.average_snapshots(tmp_path / "a")


def test_missing_incomplete_tampered_and_unselected_epochs_fail_closed(tmp_path):
    make_run(tmp_path)
    snapshots = tmp_path / "epoch-checkpoints"
    with pytest.raises(ValueError, match="predeclared"):
        wa.average_snapshots(tmp_path, (8, 9, 10))
    original = (snapshots / "epoch-009.pt").read_bytes()
    (snapshots / "epoch-009.pt").write_bytes(original + b"0")
    with pytest.raises(ValueError, match="tampered"):
        wa.average_snapshots(tmp_path)
    (snapshots / "epoch-009.pt").write_bytes(original)
    wa.average_snapshots(tmp_path)
    (snapshots / "epoch-007.json").unlink()
    with pytest.raises(ValueError, match="Missing"):
        wa.average_snapshots(tmp_path)


def test_incomplete_run_and_unsupported_schedule_fail_closed(tmp_path):
    make_run(tmp_path / "running", status="running")
    with pytest.raises(ValueError, match="completed"):
        wa.average_snapshots(tmp_path / "running")
    make_run(tmp_path / "gapped")
    (tmp_path / "gapped" / "epoch-checkpoints" / "epoch-006.pt").unlink()
    with pytest.raises(ValueError, match="Missing"):
        wa.average_snapshots(tmp_path / "gapped")
    record = json.loads((tmp_path / "running" / "run.json").read_text())
    record["status"] = "completed"
    record["binding"]["config"]["submission_every_epochs"] = 2
    (tmp_path / "running" / "run.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="fixed-ten-epoch"):
        wa.average_snapshots(tmp_path / "running")


def test_unsupported_buffers_and_graphs_are_refused(tmp_path):
    make_run(tmp_path)
    candidate = wa.average_snapshots(tmp_path)
    with pytest.raises(ValueError):
        wa.apply_candidate(Net("rs_lora_v1", extra_buffer=True), candidate)
    with pytest.raises(ValueError, match="buffer"):
        wa._learned_layout(
            {
                **candidate["source_model_storage"],
                "learned_tensors": {
                    **candidate["source_model_storage"]["learned_tensors"],
                    "head.running_mean": {"shape": [3], "dtype": "torch.float32"},
                },
            },
            wa.encoder_policy(candidate["source_binding"]["config"]["encoder"]),
        )
    with pytest.raises(ValueError, match="Unsupported residual"):
        wa.encoder_policy(
            {"pretrained": True, "adapter": {"method": "dora_v1", "scaling": "alpha/sqrt(rank)"}}
        )
    with pytest.raises(ValueError, match="exactly one"):
        wa.encoder_policy({"pretrained": True})
    with pytest.raises(ValueError):
        wa.apply_candidate(Net(wa.DENSE_METHOD), candidate)


def test_source_code_must_match_training_except_this_module():
    bound = {"code_sha256": {"competitive/train.py": "d" * 64}}
    wa.verify_source_code(bound, {"competitive/train.py": "d" * 64, wa.SELF: "f" * 64})
    with pytest.raises(ValueError):
        wa.verify_source_code(bound, {"competitive/train.py": "e" * 64})
    with pytest.raises(ValueError):
        wa.verify_source_code(bound, {"competitive/train.py": "d" * 64, "other.py": "f" * 64})
