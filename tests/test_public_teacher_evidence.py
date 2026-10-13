import copy
import io
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

from rsna.competitive import public_input as pi
from rsna.competitive import teacher_evidence as te
from rsna.schema import TARGETS
from tools.research.public_dinov2_input_reference import provenance, reference
from tools.research.public_teacher_evidence import teacher

ROOT = Path(__file__).resolve().parents[1]
PIXEL_GROUP = json.loads(
    (ROOT / "initiative/audits/public-dinov2-real-inputs-20261013-v2/actual-input-comparison.json")
    .read_text()
)["provenance"]["member_pixel_group"]
CPU = torch.device("cpu")
MARKS: list[int] = []


def _mark():
    MARKS.append(1)
    return {}


class Payload:
    def __reduce__(self):
        return (_mark, ())


class StubBackbone(nn.Module):
    """Architecture stand-in exposing only what the original Model.forward reads."""

    def __init__(self, dim=8):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=dim)
        self.patch = nn.Conv2d(3, dim, 48, 48)
        self.cls = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, pixel_values):
        patches = self.patch(pixel_values).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls.expand(len(patches), -1, -1), patches], 1)
        return SimpleNamespace(last_hidden_state=tokens)


@pytest.fixture(scope="module")
def namespace():
    ns, receipt = teacher.load_namespace(reference.SOURCE_PATH.read_bytes(), PIXEL_GROUP)
    ns["_receipt"] = receipt
    return ns


def checkpoint_bytes(namespace, seed=0, mutate=None):
    torch.manual_seed(seed)
    model = namespace["Model"](StubBackbone(), 8, pool="cls_mean_focal", prior=True)
    fingerprint = torch.from_numpy(namespace["fingerprint"](model, CPU, 336))
    state = model.state_dict()
    if mutate:
        mutate(state)
    buffer = io.BytesIO()
    torch.save({"model": state, "fingerprint": fingerprint}, buffer)
    return buffer.getvalue()


CONFIG = {"variant": "small", "pool": "cls_mean_focal", "prior": True, "unfreeze_last": 4}


def build(namespace, raw, seed=1):
    torch.manual_seed(seed)
    ck = teacher.load_checkpoint(raw, teacher.sha256(raw))
    return teacher.build_teacher(namespace, CONFIG, ck, StubBackbone(), CPU)


def study(seed=0, occupied=(0, 1, 2, 4)):
    g = torch.Generator().manual_seed(seed)
    images = torch.randint(0, 256, (6, 12, 336, 336), generator=g, dtype=torch.uint8)
    mask = torch.zeros(6, dtype=torch.bool)
    mask[list(occupied)] = True
    images[~mask] = 0
    return images, mask


def binding(uids, **change):
    out = {
        "kind": te.KIND,
        "teacher": te.TEACHER,
        "source_sha256": provenance.SOURCE_SHA256,
        "renderer_node_text_sha256": pi.NODE_TEXT_SHA256,
        "teacher_node_text_sha256": "a" * 64,
        "teacher_review_sha256": "b" * 64,
        "checkpoint_sha256": provenance.CHECKPOINT_SHA256,
        "member_manifest_sha256": "c" * 64,
        "member_config": CONFIG,
        "pixel_group_sha256": pi.PIXEL_GROUP_SHA256,
        "backbone_config_sha256": "d" * 64,
        "runtime": {"torch": torch.__version__, "device": "cpu"},
        "fitting_roster_sha256": te.FITTING_ROSTER_SHA256,
        "requested_roster_sha256": pi.roster_sha256(uids),
        "requested_studies": list(uids),
        "input": {"mode": "render"},
        "targets": list(TARGETS),
        "windows": te.WINDOWS,
    }
    out.update(change)
    return out


def row(seed, logits=None):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(10, 12, generator=g) if logits is None else logits
    return {
        "window_logits": z.numpy().astype(np.float32),
        "probabilities": torch.sigmoid(z).mean(0).numpy().astype(np.float32),
        "slot_mask": np.array([1, 1, 0, 0, 0, 0], bool),
        "input_sha256": np.asarray(f"{seed:064x}"),
    }


def test_teacher_namespace_binds_reviewed_nodes_and_exact_target_order(namespace):
    assert list(namespace["TARGETS"]) == list(TARGETS)
    assert namespace["_receipt"]["teacher_node_text_sha256"] == (
        teacher.load_teacher_review()["node_text_sha256"]
    )
    assert namespace["_receipt"]["notebook_top_level_executed"] is False
    for unsafe in ("build_model", "find_weights", "find_dinov2", "infer_from_package"):
        assert unsafe not in namespace
    for name in teacher.TEACHER_NODES:
        assert name in namespace


def test_source_review_and_pixel_group_mismatch_refuse(namespace):
    raw = bytearray(reference.SOURCE_PATH.read_bytes())
    raw[-3] ^= 1
    with pytest.raises(ValueError):
        teacher.load_namespace(bytes(raw), PIXEL_GROUP)
    review = copy.deepcopy(teacher.load_teacher_review())
    review["node_text_sha256"] = "0" * 64
    with pytest.raises(teacher.TeacherMismatch):
        teacher.load_namespace(reference.SOURCE_PATH.read_bytes(), PIXEL_GROUP, review)
    with pytest.raises(teacher.TeacherMismatch):
        teacher.load_namespace(
            reference.SOURCE_PATH.read_bytes(), {**PIXEL_GROUP, "crop_mm": 161.25}
        )


def test_checkpoint_hash_weights_only_and_strict_state(namespace):
    raw = checkpoint_bytes(namespace)
    with pytest.raises(teacher.TeacherMismatch):
        teacher.load_checkpoint(raw)  # synthetic bytes are not the audited member
    buffer = io.BytesIO()
    torch.save({"model": {"x": torch.zeros(1)}, "fingerprint": Payload()}, buffer)
    unsafe = buffer.getvalue()
    with pytest.raises(Exception):
        teacher.load_checkpoint(unsafe, teacher.sha256(unsafe))
    assert MARKS == []
    missing = checkpoint_bytes(namespace, mutate=lambda s: s.pop("head.query"))
    extra = checkpoint_bytes(namespace, mutate=lambda s: s.update(extra=torch.zeros(1)))
    for bad in (missing, extra):
        with pytest.raises(teacher.TeacherMismatch):
            build(namespace, bad)
    shifted = checkpoint_bytes(namespace, mutate=lambda s: s["head.out.bias"].add_(1.0))
    with pytest.raises(namespace["WeightsError"]):
        build(namespace, shifted)
    with pytest.raises(teacher.TeacherMismatch):
        ck = teacher.load_checkpoint(raw, teacher.sha256(raw))
        teacher.build_teacher(namespace, {**CONFIG, "variant": "base"}, ck, StubBackbone(), CPU)


def test_backbone_config_is_hash_and_architecture_bound():
    raw = json.dumps({**teacher.BACKBONE_FIELDS, "image_size": 518}).encode()
    assert teacher.backbone_config(raw, teacher.sha256(raw))["hidden_size"] == 384
    with pytest.raises(teacher.TeacherMismatch):
        teacher.backbone_config(raw, "0" * 64)
    wrong = json.dumps({**teacher.BACKBONE_FIELDS, "hidden_size": 768}).encode()
    with pytest.raises(teacher.TeacherMismatch):
        teacher.backbone_config(wrong, teacher.sha256(wrong))


def test_no_grad_collection_reproduces_original_predict_member(namespace):
    model = build(namespace, checkpoint_bytes(namespace))
    assert not any(p.requires_grad for p in model.parameters()) and not model.training
    images, mask = study()
    out = teacher.collect_study(namespace, model, images, mask, CPU)
    assert set(out) == {"window_logits", "probabilities", "slot_mask", "input_sha256"}
    assert out["window_logits"].shape == (10, 12) and np.isfinite(out["window_logits"]).all()
    original = namespace["predict_member"](
        model, images.numpy()[None], mask.numpy().astype(np.float32)[None], [0], CPU, 336
    )
    np.testing.assert_array_equal(out["probabilities"], original[0])
    assert all(p.grad is None for p in model.parameters())
    again = teacher.collect_study(namespace, model, images, mask, CPU)
    np.testing.assert_array_equal(again["window_logits"], out["window_logits"])
    assert not hasattr(model, "_forward_hooks") or not model._forward_hooks


def test_collection_refuses_invalid_inputs_and_non_finite_teacher(namespace):
    model = build(namespace, checkpoint_bytes(namespace))
    images, mask = study()
    with pytest.raises(teacher.TeacherMismatch):
        teacher.collect_study(namespace, model, images.float(), mask, CPU)
    with pytest.raises(teacher.TeacherMismatch):
        teacher.collect_study(namespace, model, images, torch.zeros(6, dtype=torch.bool), CPU)
    nan = build(
        namespace,
        checkpoint_bytes(namespace, mutate=lambda s: s["head.out.bias"].fill_(float("nan"))),
    )
    with pytest.raises(teacher.TeacherMismatch):
        teacher.collect_study(namespace, nan, images, mask, CPU)


def test_no_fabricated_attention_or_slot_evidence(tmp_path):
    review = teacher.load_teacher_review()
    assert set(review["not_exposed"]) == {"slot_attention", "patch_attention", "per_slot_logits"}
    assert set(review["exposed_evidence"]) == {"window_logits", "probabilities"}
    receipt = te.write_evidence(tmp_path / "e", binding(["u1"]), [row(1)])
    _, arrays = te.load_evidence(tmp_path / "e")
    assert set(arrays) == set(te.ARRAYS)
    assert not any("att" in name or "label" in name for name in arrays)
    assert receipt["claims"]["attention_or_slot_evidence"] == "not_exposed_by_original_source"
    assert receipt["claims"]["teacher_training_roster"] == "unknown"
    assert receipt["claims"]["clean_heldout_teacher_comparison"] is False


def test_roster_refuses_leakage_duplicates_caps_and_unpinned_rosters():
    fitting = [f"f{i}" for i in range(20)]
    sha = pi.roster_sha256(fitting)
    kwargs = {"fitting_sha256": sha, "fitting_studies": 20}
    assert te.fitting_request(fitting, ["f3", "f1"], max_studies=8, **kwargs) == ["f3", "f1"]
    for requested, cap in ((["f1", "outer7"], 8), (["f1", "f1"], 8), (fitting[:9], 8), ([], 8)):
        with pytest.raises(te.EvidenceError):
            te.fitting_request(fitting, requested, max_studies=cap, **kwargs)
    with pytest.raises(te.EvidenceError):
        te.fitting_request(fitting, ["f1"], max_studies=8)
    with pytest.raises(te.EvidenceError):
        te.fitting_request(fitting[:-1] + ["outer"], ["f1"], max_studies=8, **kwargs)


def test_storage_cap_and_non_finite_evidence_refused_before_creation(tmp_path, monkeypatch):
    with pytest.raises(te.EvidenceError):
        te.write_evidence(tmp_path / "nan", binding(["u1"]), [row(1, torch.full((10, 12), np.nan))])
    assert not (tmp_path / "nan").exists()
    monkeypatch.setattr(te, "MAX_EVIDENCE_BYTES", 64)
    with pytest.raises(te.EvidenceError):
        te.write_evidence(tmp_path / "cap", binding(["u1"]), [row(1)])
    assert not (tmp_path / "cap").exists()
    with pytest.raises(te.EvidenceError):
        te.write_evidence(tmp_path / "bad", binding(["u1"], targets=list(TARGETS)[::-1]), [row(1)])


def test_immutable_evidence_reconciles_identically_and_refuses_changes(tmp_path):
    out, uids = tmp_path / "e", ["u1", "u2"]
    assert te.reconcile(out, binding(uids)) is None
    receipt = te.write_evidence(out, binding(uids), [row(1), row(2)])
    assert te.reconcile(out, binding(uids)) == receipt
    with pytest.raises(FileExistsError):
        te.write_evidence(out, binding(uids), [row(1), row(2)])
    with pytest.raises(te.EvidenceError):
        te.reconcile(out, binding(uids, checkpoint_sha256="e" * 64))
    payload = bytearray((out / te.EVIDENCE_FILE).read_bytes())
    payload[-5] ^= 1
    (out / te.EVIDENCE_FILE).write_bytes(bytes(payload))
    with pytest.raises(te.EvidenceError):
        te.reconcile(out, binding(uids))
    (tmp_path / "partial").mkdir()
    with pytest.raises(te.EvidenceError):
        te.reconcile(tmp_path / "partial", binding(uids))


def test_teacher_targets_are_separate_soft_evidence_not_labels(tmp_path):
    uncertain = torch.zeros(10, 12)
    receipt = te.write_evidence(tmp_path / "e", binding(["u1", "u2"]), [row(1), row(2, uncertain)])
    receipt, arrays = te.load_evidence(tmp_path / "e")
    soft, weight = te.teacher_targets(receipt, arrays, ["u2", "absent", "u1"])
    assert (weight[0] == 0).all() and (soft[1] == 0).all() and (weight[1] == 0).all()
    assert ((weight >= 0) & (weight <= 1)).all()
    labels = torch.tensor([[1.0] * 12, [0.0] * 12, [0.0] * 12])
    label_mask = torch.tensor([[1.0] * 12, [0.0] * 12, [1.0] * 12])
    before = labels.clone(), label_mask.clone()
    counts = te.disagreement(soft, weight, labels, label_mask)
    assert torch.equal(labels, before[0]) and torch.equal(label_mask, before[1])
    assert all(c["uncertain_observed"] == 1 and c["unlabeled"] == 1 for c in counts.values())
    assert receipt["summary"]["uncertain_cells_confidence_below_0.5"]["ACL"] >= 1


def test_zero_auxiliary_weight_is_exactly_the_ordinary_baseline():
    def step(weight):
        torch.manual_seed(0)
        model = nn.Linear(4, 12)
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        x = torch.randn(3, 4)
        labels, mask = torch.ones(3, 12), torch.ones(3, 12)
        logits = model(x)
        base = te.effective_batch_loss(logits, labels, mask, mask.sum(0).double())
        soft, conf = torch.full((3, 12), 0.2), torch.full((3, 12), 0.5)

        def auxiliary():
            if weight == 0:
                raise AssertionError("teacher term evaluated at zero weight")
            return te.teacher_auxiliary_loss(logits, soft, conf, conf.sum(0).double())

        loss = te.combined_loss(base, auxiliary, weight)
        assert (loss is base) == (weight == 0)
        loss.backward()
        opt.step()
        return [p.detach().clone() for p in model.parameters()]

    def plain():
        torch.manual_seed(0)
        model = nn.Linear(4, 12)
        opt = torch.optim.AdamW(model.parameters(), lr=0.1)
        x = torch.randn(3, 4)
        mask = torch.ones(3, 12)
        te.effective_batch_loss(model(x), torch.ones(3, 12), mask, mask.sum(0).double()).backward()
        opt.step()
        return [p.detach().clone() for p in model.parameters()]

    assert all(torch.equal(a, b) for a, b in zip(step(0.0), plain()))
    assert not all(torch.equal(a, b) for a, b in zip(step(0.5), plain()))
    for bad in (float("nan"), 1.5, -0.1, True, "0.5"):
        with pytest.raises(te.EvidenceError):
            te.combined_loss(torch.zeros(()), lambda: torch.zeros(()), bad)


def test_student_inference_has_no_teacher_dependency():
    student = ("rsna/competitive/model.py", "rsna/competitive/train.py", "rsna/infer.py")
    for path in (*student, "rsna/model.py"):
        file = ROOT / path
        if file.exists():
            text = file.read_text()
            assert "teacher_evidence" not in text and "public_teacher_evidence" not in text


def test_acceptance_cli_is_bounded_and_mode_checked():
    from scripts.newton import collect_public_teacher_evidence as cli

    common = ["--source", "s", "--member-manifest", "m", "--checkpoint", "c"]
    common += ["--backbone-config", "b", "--backbone-config-sha256", "0" * 64]
    common += ["--fitting-roster", "r", "--out", "o"]
    with pytest.raises(SystemExit):
        cli.parse(["--mode", "render", *common])
    with pytest.raises(SystemExit):
        cli.parse(["--mode", "cache", *common, "--cache", "x"])
    args = cli.parse(["--mode", "render", *common, "--dicom-root", "d", "--series", "s.csv"])
    assert args.roster_size == te.ACCEPTANCE_STUDIES == 8 and args.device == "cpu"
