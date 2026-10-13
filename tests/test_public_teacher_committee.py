import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from rsna.competitive import committee_evidence as ce
from rsna.competitive import public_input as pi
from rsna.competitive import teacher_evidence as te
from rsna.schema import TARGETS
from tools.research.public_dinov2_input_reference import reference
from tools.research.public_teacher_evidence import committee, teacher

ROOT = Path(__file__).resolve().parents[1]
PIXEL_GROUP = json.loads(
    (
        ROOT
        / "initiative/audits/public-dinov2-real-inputs-20261013-v2/actual-input-comparison.json"
    ).read_text()
)["provenance"]["member_pixel_group"]
CPU = torch.device("cpu")
DIM = 8


class Config:
    hidden_size = DIM

    def to_dict(self):
        return {"model_type": "dinov2", "hidden_size": DIM, "num_hidden_layers": 12}


class Embeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch = nn.Conv2d(3, DIM, 48, 48)
        self.cls = nn.Parameter(torch.randn(1, 1, DIM))
        self.position = nn.Parameter(torch.randn(1, 50, DIM) * 0.1)

    def forward(self, pixel_values, bool_masked_pos=None):
        assert bool_masked_pos is None
        patches = self.patch(pixel_values).flatten(2).transpose(1, 2)
        return torch.cat([self.cls.expand(len(patches), -1, -1), patches], 1) + self.position


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm = nn.LayerNorm(DIM)
        self.mix = nn.Linear(DIM, DIM)
        self.register_buffer("scale", torch.ones(DIM))

    def forward(self, hidden):
        return (hidden + self.scale * torch.tanh(self.mix(self.norm(hidden))),)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.ModuleList(Block() for _ in range(12))


class StubDinov2(nn.Module):
    """Dinov2Model-shaped stand-in: embeddings, encoder.layer[12], layernorm."""

    def __init__(self):
        super().__init__()
        self.config = Config()
        self.embeddings = Embeddings()
        self.encoder = Encoder()
        self.layernorm = nn.LayerNorm(DIM)

    def forward(self, pixel_values):
        hidden = self.embeddings(pixel_values)
        for layer in self.encoder.layer:
            hidden = layer(hidden)[0]
        return SimpleNamespace(last_hidden_state=self.layernorm(hidden))


STUB_PREFIX = (4 + 6 * 5, 3 * 48 * 48 * DIM + DIM + DIM + 50 * DIM + 6 * (4 * DIM + DIM * DIM))


@pytest.fixture(scope="module")
def namespace():
    ns, _ = teacher.load_namespace(reference.SOURCE_PATH.read_bytes(), PIXEL_GROUP)
    return ns


def members(namespace, n=committee.MEMBERS, pool="cls_mean_focal"):
    torch.manual_seed(0)
    lead = namespace["Model"](StubDinov2(), DIM, pool=pool, prior=True)
    prefix = {k: v for k, v in lead.state_dict().items() if k in committee.prefix_names(lead)}
    out = []
    for seed in range(n):
        torch.manual_seed(100 + seed)
        model = namespace["Model"](StubDinov2(), DIM, pool=pool, prior=bool(seed % 2))
        model.load_state_dict(prefix, strict=False)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        out.append(model)
    return out


def ids(n=committee.MEMBERS):
    return [f"m{i:02d}" for i in range(n)]


def study(seed=0, occupied=(0, 1, 2, 4)):
    g = torch.Generator().manual_seed(seed)
    images = torch.randint(0, 256, (6, 12, 336, 336), generator=g, dtype=torch.uint8)
    mask = torch.zeros(6, dtype=torch.bool)
    mask[list(occupied)] = True
    return images, mask


@pytest.fixture(scope="module")
def built(namespace):
    return committee.Committee(namespace, members(namespace), ids(), STUB_PREFIX)


def test_prefix_names_are_embeddings_and_first_six_blocks(namespace, built):
    names = committee.prefix_names(built.full[0])
    assert all(n.startswith("backbone.embeddings.") or ".layer." in n for n in names)
    assert {n.split(".")[3] for n in names if ".layer." in n} == {str(i) for i in range(6)}
    assert built.prefix_summary["tensors"] == STUB_PREFIX[0]
    assert built.prefix_summary["elements"] == STUB_PREFIX[1]
    assert committee.PREFIX_TENSORS == 113 and committee.PREFIX_ELEMENTS == 11_404_416


def test_shared_forward_is_bitwise_full_graph_and_computes_prefix_once(namespace, built):
    images, mask = study()
    prefix = built.prefix
    prefix.computed = prefix.reused = 0
    shared = built.collect_study(images, mask, CPU, check_full=False)
    assert (prefix.computed, prefix.reused) == (te.WINDOWS, te.WINDOWS * (committee.MEMBERS - 1))
    full = built.collect_study(images, mask, CPU, mode="full")
    for key in ("member_window_logits", "member_probabilities", "slot_mask", "input_sha256"):
        assert np.array_equal(shared[key], full[key])
    assert shared["member_window_logits"].shape == (20, te.WINDOWS, len(TARGETS))
    for index, model in enumerate(built.full):
        cache = images.numpy()[None]
        original = namespace["predict_member"](
            model, cache, mask.numpy().astype(np.float32)[None], [0], CPU, namespace["IMG"]
        )
        assert np.array_equal(shared["member_probabilities"][index], original[0])
    assert len({shared["member_probabilities"][i].tobytes() for i in range(20)}) == 20
    built.collect_study(images, mask, CPU)  # checked mode passes too


def test_original_fingerprints_shared_equal_full(namespace, built):
    references = [torch.from_numpy(namespace["fingerprint"](m, CPU, 336)) for m in built.full]
    assert built.verify_fingerprints(CPU, references) == [0.0] * committee.MEMBERS
    shifted = [r + 1.0 for r in references]
    with pytest.raises(namespace["WeightsError"]):
        built.verify_fingerprints(CPU, shifted)


def test_changed_prefix_tensor_buffer_config_or_member_count_refuses(namespace):
    group = members(namespace)
    with pytest.raises(teacher.TeacherMismatch, match="exactly"):
        committee.Committee(namespace, group[:19], ids(19), STUB_PREFIX)
    with pytest.raises(teacher.TeacherMismatch, match="duplicated"):
        committee.Committee(namespace, group, ["m00"] * 20, STUB_PREFIX)
    with pytest.raises(teacher.TeacherMismatch, match="size"):
        committee.Committee(namespace, group, ids())
    group[7].backbone.encoder.layer[5].mix.weight.data[0, 0] += 1e-7
    with pytest.raises(teacher.TeacherMismatch, match="layer.5.mix.weight"):
        committee.Committee(namespace, group, ids(), STUB_PREFIX)
    group = members(namespace)
    group[3].backbone.encoder.layer[2].scale.data[1] = -0.0 + 1.0 + 1e-6
    with pytest.raises(teacher.TeacherMismatch, match="scale"):
        committee.Committee(namespace, group, ids(), STUB_PREFIX)
    group = members(namespace)
    group[4].mean.data[0, 0] = 0.4851
    with pytest.raises(teacher.TeacherMismatch, match="mean"):
        committee.Committee(namespace, group, ids(), STUB_PREFIX)
    group = members(namespace)
    group[5].backbone.config = type("Other", (Config,), {"to_dict": lambda s: {"x": 1}})()
    with pytest.raises(teacher.TeacherMismatch, match="config"):
        committee.Committee(namespace, group, ids(), STUB_PREFIX)
    group = members(namespace)
    group[6].backbone.embeddings.position.data = group[6].backbone.embeddings.position.data.double()
    with pytest.raises(teacher.TeacherMismatch, match="position"):
        committee.Committee(namespace, group, ids(), STUB_PREFIX)
    group = members(namespace)
    group[8].train()
    with pytest.raises(teacher.TeacherMismatch, match="evaluation"):
        committee.Committee(namespace, group, ids(), STUB_PREFIX)


def test_suffix_differences_are_allowed_and_kept_per_member(namespace):
    group = members(namespace)
    group[9].backbone.encoder.layer[6].mix.weight.data += 1.0
    group[9].head.out.bias.data += 1.0
    built = committee.Committee(namespace, group, ids(), STUB_PREFIX)
    images, mask = study(seed=3)
    built.collect_study(images, mask, CPU)


def test_prefix_mutation_after_verification_refuses(namespace):
    built = committee.Committee(namespace, members(namespace), ids(), STUB_PREFIX)
    images, mask = study()
    built.full[12].backbone.embeddings.cls.data.add_(1e-3)  # bypasses version counters
    with pytest.raises(teacher.TeacherMismatch, match="changed after verification"):
        built.collect_study(images, mask, CPU, check_full=False)
    built = committee.Committee(namespace, members(namespace), ids(), STUB_PREFIX)
    with torch.no_grad():
        built.full[3].backbone.encoder.layer[4].mix.bias.add_(0.0)  # same bytes, new version
    with pytest.raises(teacher.TeacherMismatch, match="changed after verification"):
        built.collect_study(images, mask, CPU, check_full=False)
    assert built.prefix._entries == []


def test_cache_never_reused_across_inputs_dtype_grad_or_studies(namespace, built):
    prefix = built.prefix
    prefix.clear()
    prefix.computed = prefix.reused = 0
    x = torch.randn(2, 3, 336, 336)
    with torch.no_grad():
        a = prefix(x)
        assert prefix(x.clone()) is a and prefix.reused == 1
        changed = x.clone()
        changed[1, 2, 335, 335] = torch.nextafter(changed[1, 2, 335, 335], torch.tensor(9.0))
        assert prefix(changed) is not a
        negzero = torch.zeros(2, 3, 336, 336)
        z = prefix(negzero)
        assert prefix(-negzero) is not z  # -0.0 differs bitwise from 0.0
        assert prefix(x[:1].clone()) is not a  # changed batch/window rows
        assert prefix.computed == 5
        with pytest.raises(teacher.TeacherMismatch, match="exceeded"):
            for i in range(20):
                prefix(torch.full((1, 3, 336, 336), float(i)))
        prefix.clear()
        a = prefix(x)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            assert prefix(x) is not a  # precision boundary
        assert prefix(x) is a
        with torch.inference_mode():
            assert prefix(x) is not a
    with pytest.raises(teacher.TeacherMismatch, match="evaluation-only"):
        prefix(x)
    prefix.clear()
    images, mask = study(seed=1)
    other, _ = study(seed=2)
    first = built.collect_study(images, mask, CPU, check_full=False)
    second = built.collect_study(other, mask, CPU, check_full=False)
    assert first["input_sha256"] != second["input_sha256"]
    assert not np.array_equal(first["member_window_logits"], second["member_window_logits"])
    assert built.prefix._entries == []


def test_missing_slots_masks_and_invalid_inputs(namespace, built):
    images, mask = study(occupied=(5,))
    sparse = built.collect_study(images, mask, CPU)
    dense = built.collect_study(images, torch.ones(6, dtype=torch.bool), CPU)
    assert sparse["slot_mask"].tolist() == [False] * 5 + [True]
    assert not np.array_equal(sparse["member_probabilities"], dense["member_probabilities"])
    with pytest.raises(teacher.TeacherMismatch):
        built.collect_study(images, torch.zeros(6, dtype=torch.bool), CPU)
    with pytest.raises(teacher.TeacherMismatch):
        built.collect_study(images.float(), mask, CPU)
    with pytest.raises(teacher.TeacherMismatch, match="mode"):
        built.collect_study(images, mask, CPU, mode="cached")
    with pytest.raises(teacher.TeacherMismatch, match="pixel_values"):
        built.views[0].backbone(torch.zeros(1, 3, 336, 336), output_attentions=True)


def test_full_graph_is_untouched_by_views(namespace, built):
    full = built.full[0]
    assert full.backbone.__class__ is StubDinov2
    assert built.views[0].backbone.__class__ is committee.SuffixBackbone
    assert type(built.views[0]) is type(full)
    assert built.views[0].head is full.head
    hook = built.views[0].register_forward_hook(lambda *a: None)
    assert not full._forward_hooks
    hook.remove()


def test_inconsistent_shared_output_is_refused(namespace, built, monkeypatch):
    images, mask = study()
    original = teacher.collect_study

    def drift(ns, model, *args):
        row = original(ns, model, *args)
        if model.backbone.__class__ is committee.SuffixBackbone:
            row["window_logits"] = np.nextafter(row["window_logits"], np.float32(9))
        return row

    monkeypatch.setattr(teacher, "collect_study", drift)
    with pytest.raises(teacher.TeacherMismatch, match="window_logits differs"):
        built.collect_study(images, mask, CPU)
    assert built.prefix._entries == []


def test_real_hf_dinov2_shared_prefix_is_bitwise_full_graph(namespace):
    transformers = pytest.importorskip("transformers")
    config = transformers.Dinov2Config(
        hidden_size=12,
        num_hidden_layers=12,
        num_attention_heads=2,
        intermediate_size=24,
        patch_size=48,
        image_size=336,
    )
    torch.manual_seed(0)
    lead = namespace["Model"](transformers.Dinov2Model(config), 12, pool="cls_mean")
    prefix = {k: v for k, v in lead.state_dict().items() if k in committee.prefix_names(lead)}
    group = []
    for seed in range(committee.MEMBERS):
        torch.manual_seed(seed + 1)
        model = namespace["Model"](transformers.Dinov2Model(config), 12, pool="cls_mean")
        model.load_state_dict(prefix, strict=False)
        model.eval().requires_grad_(False)
        group.append(model)
    size = committee.prefix_digest(group[0])
    built = committee.Committee(namespace, group, ids(), (size["tensors"], size["elements"]))
    images, mask = study()
    built.collect_study(images, mask, CPU)
    references = [torch.from_numpy(namespace["fingerprint"](m, CPU, 336)) for m in group]
    assert built.verify_fingerprints(CPU, references) == [0.0] * committee.MEMBERS


def committee_binding(uids, **change):
    members_ = [
        {
            "id": f"m{i:02d}",
            "file": f"m{i:02d}.pt",
            "checkpoint_sha256": f"{i:064x}",
            "config": {"variant": "small", "fold": i % 5},
            "config_sha256": pi.canonical_sha256({"variant": "small", "fold": i % 5}),
        }
        for i in range(20)
    ]
    binding = {
        "kind": ce.KIND,
        "source": ce.SOURCE,
        "members": members_,
        "committee_receipt_sha256": "a" * 64,
        "member_manifest_sha256": "b" * 64,
        "prefix": {"layers": 6, "tensors": 113, "elements": 11_404_416},
        "forward": "shared_prefix_bitwise_checked",
        "source_sha256": "c" * 64,
        "renderer_node_text_sha256": "d" * 64,
        "teacher_node_text_sha256": "e" * 64,
        "teacher_review_sha256": "f" * 64,
        "pixel_group_sha256": "1" * 64,
        "backbone_config_sha256": "2" * 64,
        "runtime": {"device": "cpu"},
        "fitting_roster_sha256": te.FITTING_ROSTER_SHA256,
        "requested_roster_sha256": pi.roster_sha256(uids),
        "requested_studies": list(uids),
        "input": {"slots": 6},
        "targets": list(TARGETS),
        "windows": te.WINDOWS,
        "storage": {"cap_bytes": ce.MAX_EVIDENCE_BYTES, "funding_receipt_sha256": "3" * 64},
    }
    binding.update(change)
    return binding


def committee_row(seed, probabilities=None):
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(20, te.WINDOWS, len(TARGETS))).astype(np.float32)
    if probabilities is None:
        probabilities = rng.uniform(size=(20, len(TARGETS))).astype(np.float32)
    return {
        "member_window_logits": logits,
        "member_probabilities": np.asarray(probabilities, np.float32),
        "slot_mask": np.array([1, 1, 0, 1, 0, 0], bool),
        "input_sha256": np.asarray(f"{seed:064x}"),
    }


def test_committee_storage_mean_disagreement_and_immutable_reconcile(tmp_path):
    uids = ["u1", "u2"]
    binding = committee_binding(uids)
    agree = np.full((20, len(TARGETS)), 0.9, np.float32)
    split = np.concatenate([np.full((10, len(TARGETS)), 0.05), np.full((10, len(TARGETS)), 0.95)])
    receipt = ce.write_evidence(
        tmp_path / "c", binding, [committee_row(1, agree), committee_row(2, split)]
    )
    assert receipt["claims"]["teacher_training_roster"] == "unknown"
    assert receipt["claims"]["clinical_truth"] is False
    loaded, arrays = ce.load_evidence(tmp_path / "c", binding)
    assert np.allclose(arrays["committee_probabilities"], [[0.9] * 12, [0.5] * 12])
    assert np.allclose(arrays["between_model_std"], [[0.0] * 12, [0.45] * 12], atol=1e-6)
    soft, weight = ce.committee_targets(loaded, arrays, ["u2", "absent", "u1"])
    assert torch.allclose(weight[2], torch.full((12,), 0.8), atol=1e-6)
    assert (weight[0] == 0).all() and (weight[1] == 0).all() and (soft[1] == 0).all()
    assert ce.reconcile(tmp_path / "c", binding)["binding_sha256"] == receipt["binding_sha256"]
    assert ce.reconcile(tmp_path / "absent", binding) is None
    with pytest.raises(te.EvidenceError):
        ce.reconcile(tmp_path / "c", committee_binding(uids, forward="full_graph"))
    with pytest.raises(FileExistsError):
        ce.write_evidence(tmp_path / "c", binding, [committee_row(1), committee_row(2)])


def test_committee_storage_refuses_unfunded_bad_members_and_tampering(tmp_path):
    uids = ["u1"]
    unfunded = committee_binding(uids, storage={"cap_bytes": ce.MAX_EVIDENCE_BYTES})
    with pytest.raises(ce.CommitteeEvidenceError, match="not funded"):
        ce.write_evidence(tmp_path / "a", unfunded, [committee_row(1)])
    assert not (tmp_path / "a").exists()
    eight = committee_binding(uids, storage={"cap_bytes": te.MAX_EVIDENCE_BYTES})
    with pytest.raises(ce.CommitteeEvidenceError):
        ce.validate_binding(eight)
    short = committee_binding(uids)
    short["members"] = short["members"][:19]
    with pytest.raises(ce.CommitteeEvidenceError):
        ce.validate_binding(short)
    dup = committee_binding(uids)
    dup["members"][1] = dict(dup["members"][0])
    with pytest.raises(ce.CommitteeEvidenceError, match="duplicated"):
        ce.validate_binding(dup)
    stale = committee_binding(uids)
    stale["members"][4]["config"] = {"variant": "small", "fold": 9}
    with pytest.raises(ce.CommitteeEvidenceError, match="incomplete"):
        ce.validate_binding(stale)
    with pytest.raises(ce.CommitteeEvidenceError, match="Invalid"):
        ce.validate_binding(committee_binding(uids, targets=list(reversed(TARGETS))))
    bad = committee_row(1)
    bad["member_window_logits"][3, 2, 1] = np.nan
    with pytest.raises(ce.CommitteeEvidenceError, match="Non-finite"):
        ce.write_evidence(tmp_path / "b", committee_binding(uids), [bad])
    missing = committee_row(1)
    missing["slot_mask"][:] = False
    with pytest.raises(ce.CommitteeEvidenceError):
        ce.write_evidence(tmp_path / "b", committee_binding(uids), [missing])
    assert not (tmp_path / "b").exists()
    arrays = ce._arrays(uids, [committee_row(1)])
    arrays["committee_probabilities"] = arrays["committee_probabilities"] + 1e-3
    with pytest.raises(ce.CommitteeEvidenceError, match="mean"):
        ce._check_arrays(arrays, uids)


def test_full_fitting_roster_fits_proposed_cap_not_single_teacher_cap():
    uids = [f"1.2.840.{i:012d}.{i:06d}" for i in range(te.FITTING_STUDIES)]
    rows = [committee_row(i) for i in range(te.FITTING_STUDIES)]
    size = len(ce._serialize(ce._arrays(uids, rows)))
    assert te.MAX_EVIDENCE_BYTES < size < ce.MAX_EVIDENCE_BYTES


def test_root_inspection_digest_binds_actual_tensor_bytes(namespace):
    group = members(namespace)
    original = committee.inspected_prefix_digest(group[0])
    assert all(committee.inspected_prefix_digest(m) == original for m in group)
    group[0].backbone.embeddings.cls.data.add_(0.001)
    assert committee.inspected_prefix_digest(group[0]) != original
