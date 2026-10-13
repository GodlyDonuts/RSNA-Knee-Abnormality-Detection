import copy

import pytest
import torch
from transformers import DINOv3ViTConfig, DINOv3ViTModel

from rsna.competitive.adapters import ResidualLinear
from tools.research.dense_suffix_delta import dense_delta as dd
from tools.research.dense_suffix_delta import prefix_cache as pc

DENSE = dict(frozen_prefix_layers=2, dropout=0.1)
LORA = dict(frozen_prefix_layers=2, rank=4, alpha=8.0, dropout=0.1)
SOURCE = dict(renderer="synthetic", shape=[2, 4, 32, 32], windows=3, scale="1/255", augment=False)


def small_model(**changes):
    torch.manual_seed(7)
    options = dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=4,
        patch_size=16,
        num_register_tokens=4,
    )
    return DINOv3ViTModel(DINOv3ViTConfig(**(options | changes))).eval()


def test_dense_linear_zero_delta_is_exact_and_aliases_anchor():
    torch.manual_seed(1)
    base = torch.nn.Linear(16, 24)
    inputs = torch.randn(5, 16)
    expected = base(inputs)
    module = dd.DenseDeltaLinear(base, 0.5).train()
    assert module.weight is base.weight and module.bias is base.bias
    assert module.dense_delta.data_ptr() != base.weight.data_ptr()
    assert torch.equal(module(inputs), expected)
    assert set(module.state_dict()) == {"weight", "bias", "dense_delta"}


def test_zero_delta_model_identity_and_exact_prefix_suffix_split():
    model = small_model()
    images = torch.randn(3, 3, 32, 32)
    with torch.no_grad():
        expected = model(pixel_values=images)
    fingerprint = dd.official_state_fingerprint(model)
    record = dd.install_dense_suffix_deltas(model, DENSE)
    assert dd.official_state_fingerprint(model) == fingerprint
    with torch.no_grad():
        actual = model(pixel_values=images)
        hidden, rope = dd.run_prefix(model, images, 2)
        sequence, pooled = dd.run_suffix(model, hidden, rope, 2)
    assert torch.equal(actual.pooler_output, expected.pooler_output)
    assert torch.equal(sequence, expected.last_hidden_state)
    assert torch.equal(pooled, expected.pooler_output)
    assert record["target_modules"][0] == "model.layer.2.attention.q_proj"


def test_only_suffix_deltas_learn_and_official_anchor_is_unchanged():
    model = small_model()
    official = copy.deepcopy(model.state_dict())
    fingerprint = dd.official_state_fingerprint(model)
    record = dd.install_dense_suffix_deltas(model, DENSE)
    head = torch.nn.Linear(32, 2)
    learned = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(learned + list(head.parameters()), lr=0.01, weight_decay=0.1)
    images = torch.randn(2, 3, 32, 32)
    model.train()
    for _ in range(3):
        optimizer.zero_grad()
        head(model(pixel_values=images).pooler_output).square().mean().backward()
        names = {n for n, p in model.named_parameters() if p.grad is not None}
        assert names == {f"{m}.dense_delta" for m in record["target_modules"]}
        optimizer.step()
    assert dd.official_state_fingerprint(model) == fingerprint
    state = model.state_dict()
    assert all(torch.equal(state[name], tensor) for name, tensor in official.items())
    assert all(p.abs().sum() > 0 for p in learned)
    prefix_grads = [p for n, p in model.named_parameters() if ".layer.0." in n or ".layer.1." in n]
    assert prefix_grads and all(p.grad is None for p in prefix_grads)


def test_learned_state_restores_exactly_without_breaking_aliases():
    model = small_model()
    dd.install_dense_suffix_deltas(model, DENSE)
    with torch.no_grad():
        for parameter in (p for p in model.parameters() if p.requires_grad):
            parameter.normal_()
    state = dd.learned_state(model)
    assert all(name.endswith(".dense_delta") for name in state)
    restored = small_model()
    dd.install_dense_suffix_deltas(restored, DENSE)
    module = restored.get_submodule("model.layer.3.mlp.down_proj")
    pointer = module.dense_delta.data_ptr()
    dd.restore_learned_state(restored, state)
    assert module.dense_delta.data_ptr() == pointer
    assert (
        module.weight is restored.state_dict(keep_vars=True)["model.layer.3.mlp.down_proj.weight"]
    )
    images = torch.randn(2, 3, 32, 32)
    with torch.no_grad():
        assert torch.equal(
            model(pixel_values=images).pooler_output, restored(pixel_values=images).pooler_output
        )
    with pytest.raises(ValueError, match="keys"):
        dd.restore_learned_state(restored, dict(list(state.items())[1:]))
    bad = dict(state)
    name = next(iter(bad))
    bad[name] = bad[name].half()
    with pytest.raises(ValueError, match="Invalid"):
        dd.restore_learned_state(restored, bad)


def test_pinned_large_graph_counts_and_matched_lora_control():
    config = DINOv3ViTConfig(
        hidden_size=1024,
        intermediate_size=4096,
        num_hidden_layers=24,
        num_attention_heads=16,
        num_register_tokens=4,
    )
    policy = dict(frozen_prefix_layers=12, dropout=0.0)
    with torch.device("meta"):
        dense = DINOv3ViTModel(config)
        lora = DINOv3ViTModel(config)
    record = dd.install_dense_suffix_deltas(dense, policy)
    learned = {n: p for n, p in dense.named_parameters() if p.requires_grad}
    assert len(record["target_modules"]) == len(learned) == 72
    assert sum(p.numel() for p in learned.values()) == record["trainable_parameters"] == 150_994_944
    assert {tuple(p.shape) for p in learned.values()} == {(1024, 1024), (4096, 1024), (1024, 4096)}
    control = dd.install_suffix_lora(lora, dict(policy, rank=32, alpha=8.0))
    assert control["target_modules"] == record["target_modules"]
    assert control["trainable_parameters"] == 7_077_888
    assert sum(isinstance(m, ResidualLinear) for m in lora.modules()) == 72
    with pytest.raises(ValueError, match="already adapted"):
        dd.install_dense_suffix_deltas(dense, policy)


@pytest.mark.parametrize(
    "policy",
    [
        dict(frozen_prefix_layers=0, dropout=0.0),
        dict(frozen_prefix_layers=4, dropout=0.0),
        dict(frozen_prefix_layers=2, dropout=1.0),
        dict(frozen_prefix_layers=2, dropout=0.0, rank=4),
    ],
)
def test_invalid_dense_policy_is_rejected(policy):
    with pytest.raises(ValueError):
        dd.install_dense_suffix_deltas(small_model(), policy)


def test_stochastic_prefix_configs_are_rejected():
    with pytest.raises(ValueError, match="drop_path_rate"):
        dd.install_dense_suffix_deltas(small_model(drop_path_rate=0.1), DENSE)


def _cache_inputs():
    torch.manual_seed(3)
    windows = torch.randn(2, 3, 3, 32, 32)
    mask = torch.tensor([[True, True, True], [False, False, False]])
    return windows, mask


def test_cached_suffix_matches_full_forward_on_acquired_images_only():
    model = small_model()
    windows, mask = _cache_inputs()
    record = pc.build_record(model, windows, mask, 2, SOURCE)
    assert record["hidden"].shape == (3, 5 + 4, 32) and record["hidden"].dtype == torch.float32
    dd.install_dense_suffix_deltas(model, DENSE)
    with torch.no_grad():
        expected = model(pixel_values=windows[mask]).pooler_output
        assert torch.equal(pc.run_cached_suffix(record, model, 2, SOURCE), expected)


def _tamper(record, model, change):
    record = dict(record, identity=dict(record["identity"]), digests=dict(record["digests"]))
    source = SOURCE
    if change == "feature":
        record["hidden"] = record["hidden"] + 1e-6
    elif change == "feature_digest":
        record["hidden"] = record["hidden"].clone()
        record["hidden"][0, 0, 0] += 1
        record["digests"]["hidden"] = pc.tensor_sha256(record["hidden"])
    elif change == "half":
        record["hidden"] = record["hidden"].half()
        record["digests"]["hidden"] = pc.tensor_sha256(record["hidden"])
    elif change == "rope":
        record["rope_sin"] = record["rope_sin"].flip(0)
        record["digests"]["rope_sin"] = pc.tensor_sha256(record["rope_sin"])
    elif change == "mask":
        record["slice_mask"] = torch.ones_like(record["slice_mask"])
        record["digests"]["slice_mask"] = pc.tensor_sha256(record["slice_mask"])
    elif change == "config":
        model.config.layer_norm_eps = 1e-6
    elif change == "official":
        with torch.no_grad():
            model.get_submodule("model.layer.0.attention.q_proj").weight.add_(1e-6)
    elif change == "source":
        source = dict(SOURCE, renderer="other")
    return record, source


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ("feature", "hidden identity"),
        ("feature_digest", "recomputed"),
        ("half", "dtype"),
        ("rope", "RoPE"),
        ("mask", "mask"),
        ("config", "identity mismatch"),
        ("official", "identity mismatch"),
        ("source", "identity mismatch"),
    ],
)
def test_cache_rejects_wrong_identity(change, match):
    model = small_model()
    windows, mask = _cache_inputs()
    record = pc.build_record(model, windows, mask, 2, SOURCE)
    if change == "feature_digest":
        tampered, source = _tamper(record, model, change)
        pc.validate_record(tampered, model, 2, source)
        pc.verify_by_recompute(record, model, windows, 2, SOURCE)
        with pytest.raises(ValueError, match="recomputed"):
            pc.verify_by_recompute(tampered, model, windows, 2, source)
        return
    tampered, source = _tamper(record, model, change)
    with pytest.raises(ValueError, match=match):
        pc.validate_record(tampered, model, 2, source)


def test_cache_rejects_wrong_prefix_depth_and_augmented_contract():
    model = small_model()
    windows, mask = _cache_inputs()
    record = pc.build_record(model, windows, mask, 2, SOURCE)
    with pytest.raises(ValueError, match="identity mismatch"):
        pc.validate_record(record, model, 1, SOURCE)
    with pytest.raises(ValueError, match="non-augmented"):
        pc.build_record(model, windows, mask, 2, dict(SOURCE, augment=True))
    with pytest.raises(ValueError, match="eval mode"):
        pc.build_record(model.train(), windows, mask, 2, SOURCE)


def test_pinned_storage_arithmetic():
    config = dict(num_register_tokens=4, patch_size=16, hidden_size=1024)
    per_image = pc.storage_bytes(config, 336, 336, 1)
    assert per_image == 446 * 1024 * 4 == 1_826_816
    assert pc.storage_bytes(config, 336, 336, 60) == 109_608_960


def test_exact_cached_training_after_multiple_suffix_updates():
    model = small_model()
    windows, mask = _cache_inputs()
    record = pc.build_record(model, windows, mask, 2, SOURCE)
    dd.install_dense_suffix_deltas(model, dict(DENSE, dropout=0.0))
    model.eval()
    head = torch.nn.Linear(32, 2)
    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad] + list(head.parameters()), lr=0.002
    )
    anchor = dd.official_state_fingerprint(model)
    for _ in range(3):
        opt.zero_grad()
        cached = pc.run_cached_suffix(record, model, 2, SOURCE)
        full = model(pixel_values=windows[mask]).pooler_output
        assert torch.equal(cached, full)
        head(cached).square().mean().backward()
        opt.step()
        assert dd.official_state_fingerprint(model) == anchor
    assert any(p.abs().sum() > 0 for p in model.parameters() if p.requires_grad)


def test_missing_cached_digest_refused():
    model = small_model()
    windows, mask = _cache_inputs()
    record = pc.build_record(model, windows, mask, 2, SOURCE)
    del record["digests"]["hidden"]
    with pytest.raises(ValueError, match="digest set"):
        pc.validate_record(record, model, 2, SOURCE)
