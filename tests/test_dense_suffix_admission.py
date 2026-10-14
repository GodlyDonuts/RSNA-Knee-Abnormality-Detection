import copy
import hashlib
import json
from pathlib import Path

import pytest
import torch
from test_dense_suffix_training import DENSE, LORA, TINY_EXPECTED, fake_create_encoder

from rsna.competitive.model import CompetitiveStudyNet
from scripts.newton import admit_dense_suffix_inputs as admit
from scripts.newton import probe_public_input_recipe_inputs as gate
from scripts.newton import run_dense_suffix_recovery as recovery

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = json.loads((ROOT / admit.ORIGINAL_RECEIPT).read_text())
R32 = dict(LORA, rank=32)
GIB = 2**30


def current_code():
    return {str(p.relative_to(ROOT / "rsna")): gate.sha(p) for p in (ROOT / "rsna").rglob("*.py")}


def admitted_code():
    """The actual frozen GPU run's source, not the subsequently evolving main tree."""
    wrapper = json.loads(
        (
            ROOT
            / "initiative/audits/dense-suffix-admission-root-review-20261014-v1"
            / "actual-CPU-accepted.json"
        ).read_text()
    )
    receipt = wrapper["receipt"]
    original_bytes = (json.dumps(receipt, sort_keys=True, indent=2) + "\n").encode()
    assert (
        hashlib.sha256(original_bytes).hexdigest()
        == ("158c242faa37c46e3b8a2e632d68b3ae690174bb3ec3b85abed2a46ffc53c834")
        == wrapper["receipt_sha256"]
    )
    assert receipt["source_git"] == "0ecece12407e5b802eec6c33302213a3b5352353"
    return dict(receipt["code_sha256"])


def funding(**changes):
    values: dict = dict(
        funded_by="root",
        original_input_receipt_sha256=admit.ORIGINAL_RECEIPT_SHA256,
        merged_cache_receipt_sha256=admit.MERGED_RECEIPT_SHA256,
        new_CPU_source_and_receipt_bytes=admit.CONTROL_BYTES,
        new_GPU_or_checkpoint_bytes=0,
        new_shared_bytes_files=0,
        prefix_cache_bytes=0,
        existing_liabilities_bytes=50 * GIB,
        reserve_bytes=20 * GIB,
    )
    values |= changes
    values.setdefault(
        "required_free_bytes",
        values["new_CPU_source_and_receipt_bytes"]
        + values["existing_liabilities_bytes"]
        + values["reserve_bytes"],
    )
    return values


def test_original_receipt_and_recipes_are_the_tracked_immutable_bytes():
    assert gate.sha(ROOT / admit.ORIGINAL_RECEIPT) == admit.ORIGINAL_RECEIPT_SHA256
    for arm, name in recovery.ARMS.items():
        assert gate.sha(ROOT / recovery.RECIPE / name) == admit.RECIPE_SHA256[arm]
    assert ORIGINAL["merged_cache_receipt_sha256"] == admit.MERGED_RECEIPT_SHA256
    inputs = admit.validate_original(ORIGINAL)
    assert inputs["overlay"] == str(admit.ORIGINAL_OVERLAY)
    assert (ORIGINAL["full_fitting_studies"], ORIGINAL["full_outer_studies"]) == (2247, 1156)
    assert ORIGINAL["public_cohort"]["members"] == 3403


@pytest.mark.parametrize(
    "edit",
    [
        lambda r: r.update(CPU_job="899733"),
        lambda r: r.update(full_outer_studies=1155),
        lambda r: r["public_cohort"].update(members=3402),
        lambda r: r["caches"].update(public="/tmp/other"),
        lambda r: r.update(protected_labels_used_for_recipe_or_metrics=True),
        lambda r: r.update(GPU_or_optimizer_or_clinical_inference_executed=True),
        lambda r: r["trainer_inputs"].update(series="/tmp/series.csv"),
        lambda r: r["supervision"].update(overlay_sha256="0" * 64),
    ],
)
def test_original_receipt_drift_refused(edit):
    original = copy.deepcopy(ORIGINAL)
    edit(original)
    with pytest.raises(ValueError):
        admit.validate_original(original)


def test_both_arms_equal_admitted_public_recipe_except_policy_and_lr():
    configs = admit.load_arms(ROOT, ORIGINAL["configs"]["public"])
    assert {arm: configs[arm]["encoder_lr"] for arm in admit.ARMS} == {
        "suffix_r32": 1e-4,
        "dense": 1e-5,
    }
    for values in configs.values():
        assert (values["epochs"], values["batch_size"] * values["accumulation_steps"]) == (10, 8)
        assert (values["side"], values["slices"], values["fold"]) == (336, 10, 0)
        assert values["fitting_supervision"]["kind"] == "label_source_v1"
        assert values["encoder"]["suffix_adapter"]["frozen_prefix_layers"] == 12


@pytest.mark.parametrize(
    "edit",
    [
        lambda c: c.update(epochs=2),
        lambda c: c.update(hidden_dim=c["hidden_dim"] + 1),
        lambda c: c["encoder"].update(weights_sha256="0" * 64),
        lambda c: c["fitting_supervision"].update(sha256="0" * 64),
        lambda c: c.update(fold=1),
    ],
)
def test_admitted_public_recipe_drift_refused(edit):
    public = copy.deepcopy(ORIGINAL["configs"]["public"])
    edit(public)
    with pytest.raises(ValueError):
        admit.load_arms(ROOT, public)


def test_encoder_spec_must_be_the_pinned_mounted_official_spec():
    configs = admit.load_arms(ROOT, ORIGINAL["configs"]["public"])
    spec = {k: v for k, v in configs["dense"]["encoder"].items() if k != "suffix_adapter"}
    assert admit.encoder_spec(configs, spec)["encoder_spec_sha256"] == gate.canonical(spec)
    for change in ({"weights_path": "/tmp/other.safetensors"}, {"architecture": "dinov3_b16"}):
        with pytest.raises(ValueError):
            admit.encoder_spec(configs, spec | change)


def test_code_delta_is_exact_and_flags_changed_loader_code():
    delta = admit.code_delta(ORIGINAL["code_sha256"], admitted_code())
    assert delta["input_label_code_changed"] == ["competitive/train.py", "release_cache.py"]
    assert (delta["changed_files"], delta["added_files"], delta["removed_files"]) == (2, 3, 0)


@pytest.mark.parametrize(
    "edit",
    [
        lambda c: c.update({"competitive/public_input.py": "0" * 64}),
        lambda c: c.update({"competitive/supervision.py": "0" * 64}),
        lambda c: c.update({"competitive/train.py": "0" * 64}),
        lambda c: c.update({"competitive/new_loader.py": "0" * 64}),
        lambda c: c.pop("labels.py"),
    ],
)
def test_old_receipt_cannot_admit_any_other_loader_or_label_change(edit):
    code = admitted_code()
    edit(code)
    with pytest.raises(ValueError, match="reviewed delta"):
        admit.code_delta(ORIGINAL["code_sha256"], code)


def test_evolving_main_is_not_silently_admitted_into_the_frozen_dense_run():
    assert current_code() != admitted_code()
    with pytest.raises(ValueError, match="reviewed delta"):
        admit.code_delta(ORIGINAL["code_sha256"], current_code())


def test_changed_loader_semantics_must_replay_the_admitted_tensors():
    proofs = {
        role: dict(ORIGINAL["tensor_sha256"][f"public/{role}"])
        for role in ("fitting", "development")
    }
    uids = {"fitting": ORIGINAL["training_16_sha256"], "development": ORIGINAL["outer_16_sha256"]}
    admit.check_replay(proofs, uids, ORIGINAL)
    for role in proofs:
        for key in ("images", "labels", "other"):
            changed = copy.deepcopy(proofs)
            changed[role][key] = "0" * 64
            with pytest.raises(ValueError, match="tensors"):
                admit.check_replay(changed, uids, ORIGINAL)
        with pytest.raises(ValueError, match="membership"):
            admit.check_replay(proofs, dict(uids, **{role: "0" * 64}), ORIGINAL)


def test_funding_requires_retained_liabilities_reserve_and_no_new_bytes():
    assert admit.validate_funding(funding())["required_free_bytes"] == (
        admit.CONTROL_BYTES + 70 * GIB
    )
    for change in (
        dict(reserve_bytes=20 * GIB - 1),
        dict(existing_liabilities_bytes=0),
        dict(prefix_cache_bytes=347 * GIB),
        dict(new_GPU_or_checkpoint_bytes=1),
        dict(new_shared_bytes_files=1),
        dict(funded_by="child"),
        dict(merged_cache_receipt_sha256="0" * 64),
        dict(required_free_bytes=1),
        dict(existing_liabilities_bytes=1.5),
    ):
        with pytest.raises(ValueError):
            admit.validate_funding(funding(**change))
    with pytest.raises(ValueError):
        admit.validate_funding({**funding(), "extra": 0})


def test_fresh_storage_proof_is_required(tmp_path, monkeypatch):
    values = funding()
    monkeypatch.setattr(admit.shutil, "disk_usage", lambda _: type("U", (), {"free": 10})())
    with pytest.raises(ValueError, match="free space"):
        admit.check_storage(tmp_path, values)
    free = values["required_free_bytes"]
    monkeypatch.setattr(admit.shutil, "disk_usage", lambda _: type("U", (), {"free": free})())
    assert admit.check_storage(tmp_path, values) == free


def staged(tmp_path):
    code = tmp_path / "code"
    names = [*admit.REQUIRED_SOURCE]
    names += [str(p.relative_to(ROOT)) for p in (ROOT / "rsna").rglob("*.py")]
    files = {}
    for name in names:
        target = code / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        files[name] = gate.sha(target)
    return code, {"source_git": "a" * 40, "files": files}


def test_source_binds_wrapper_recipes_receipt_and_every_rsna_module(tmp_path):
    code, manifest = staged(tmp_path)
    assert admit.verify_source(code, manifest) == recovery.verify_source(code, manifest["files"])
    missing = copy.deepcopy(manifest)
    missing["files"].pop(admit.WRAPPER)
    with pytest.raises(ValueError, match="binding"):
        admit.verify_source(code, missing)
    (code / admit.WRAPPER).write_text("# changed\n")
    with pytest.raises(ValueError, match="differs"):
        admit.verify_source(code, manifest)


def fresh_arm(arm, seed=20260922):
    torch.manual_seed(seed)
    policy = {"suffix_r32": R32, "dense": DENSE}[arm]
    encoder, dim, provenance = fake_create_encoder({"suffix_adapter": policy})
    model = CompetitiveStudyNet(encoder, dim, hidden_dim=12, input_channels=3)
    return model, provenance


def states(**seeds):
    result, provenance = {}, {}
    for arm in admit.ARMS:
        model, provenance[arm] = fresh_arm(arm, seeds.get(arm, 20260922))
        result[arm] = admit.arm_state(model, provenance[arm], arm, TINY_EXPECTED)
    return result, provenance


def tiny_original(provenance, frozen):
    public = {k: v for k, v in provenance.items() if k != "suffix_adapter"}
    return {
        "initial_learned_state": {"public": {"frozen_base_sha256": frozen}},
        "encoder_provenance": {"public": public | {"adapter": {"method": "rs_lora_v1"}}},
    }


def test_both_graphs_frozen_official_base_and_identical_head_rng_admitted():
    values, provenance = states()
    assert values["suffix_r32"]["encoder_learned_parameters"] == 2 * 32 * (4 * 64 + 2 * 96)
    assert values["dense"]["encoder_learned_parameters"] == 2 * (4 * 32 * 32 + 2 * 32 * 64)
    original = tiny_original(provenance["dense"], values["dense"]["frozen_base_sha256"])
    admit.check_matched(values, provenance, original)


def test_pinned_production_graph_counts():
    assert recovery.PINNED["suffix_r32"]["trainable_parameters"] == 7_077_888
    assert recovery.PINNED["dense"]["trainable_parameters"] == 150_994_944
    assert 12 * 32 * (4 * 2 * 1024 + 2 * (1024 + 4096)) == 7_077_888
    assert 12 * (4 * 1024 * 1024 + 2 * 1024 * 4096) == 150_994_944


def test_head_or_factory_rng_difference_refused():
    values, provenance = states(dense=1)
    original = tiny_original(provenance["dense"], values["dense"]["frozen_base_sha256"])
    with pytest.raises(ValueError, match="head/factory RNG"):
        admit.check_matched(values, provenance, original)


def test_frozen_base_or_provenance_difference_from_original_refused():
    values, provenance = states()
    with pytest.raises(ValueError, match="original admitted base"):
        admit.check_matched(values, provenance, tiny_original(provenance["dense"], "0" * 64))
    original = tiny_original(provenance["dense"], values["dense"]["frozen_base_sha256"])
    original["encoder_provenance"]["public"]["weights_sha256"] = "3" * 64
    with pytest.raises(ValueError, match="provenance"):
        admit.check_matched(values, provenance, original)


def test_unfrozen_official_tensor_or_wrong_graph_refused():
    model, provenance = fresh_arm("dense")
    model.encoder.backbone.embeddings.patch_embeddings.weight.requires_grad_(True)
    with pytest.raises(ValueError, match="official"):
        admit.arm_state(model, provenance, "dense", TINY_EXPECTED)
    model, provenance = fresh_arm("dense")
    with pytest.raises(ValueError, match="pinned suffix arm"):
        admit.arm_state(model, provenance, "dense", recovery.PINNED)
    model, provenance = fresh_arm("suffix_r32")
    with pytest.raises(ValueError, match="pinned suffix arm"):
        admit.arm_state(model, provenance, "dense", TINY_EXPECTED)


def test_nonzero_initial_residual_refused():
    model, provenance = fresh_arm("dense")
    with torch.no_grad():
        next(p for n, p in model.named_parameters() if n.endswith(".dense_delta")).add_(1)
    with pytest.raises(ValueError, match="zero-initialized"):
        admit.arm_state(model, provenance, "dense", TINY_EXPECTED)


def test_future_obligations_are_explicit_unfunded_and_never_cache_prefixes():
    configs = admit.load_arms(ROOT, ORIGINAL["configs"]["public"])
    value = admit.future_obligations(configs)
    assert value["prefix_cache_bytes"] == 0 and value["prefix_cache_materialized"] is False
    assert value["streams_acquired_triplets_online"] is True
    assert value["GPU_recovery"]["phases"] == ["full", "stop1", "resume"]
    assert value["GPU_recovery"]["funded"] is value["production_fixed10"]["funded"] is False
    assert value["production_fixed10"]["epochs"] == {"suffix_r32": 10, "dense": 10}
    bounds = value["production_fixed10"]["checkpoint_upper_bound_bytes"]
    assert bounds["dense"] == 11 * 12 * 150_994_944 + GIB


def test_receipt_refuses_case_level_fields(tmp_path):
    private = ["1.2.840.9999", "patient-7"]
    for value in (
        {"members": ["1.2.840.9999"]},
        {"nested": {"patient-7": 1}},
        {"row": [{"uid": "1.2.840.9999"}]},
    ):
        with pytest.raises(ValueError, match="case-level"):
            admit.write_once(tmp_path / "x.json", value, private)
    assert not (tmp_path / "x.json").exists()
    admit.write_once(tmp_path / "x.json", {"studies": 3403, "sha": "a" * 64}, private)
    with pytest.raises(FileExistsError):
        admit.write_once(tmp_path / "x.json", {"studies": 3403}, private)


def test_cpu_admission_refuses_gpu_or_wrong_placement(monkeypatch, tmp_path):
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    with pytest.raises(ValueError, match="CPU-only"):
        admit.run(tmp_path)
