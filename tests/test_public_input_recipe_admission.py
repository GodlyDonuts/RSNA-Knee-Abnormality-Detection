"""Public-input recipe CPU admission and one-H100 recovery wrapper; engineering only."""

import copy
import hashlib
import json
import math
import shutil
from pathlib import Path

import pytest
import torch
from test_public_triplet_training import Capture, build, cohort, dataset

from rsna.competitive.backbones import _GrayEncoder
from rsna.competitive.model import CompetitiveStudyNet
from scripts.newton import probe_public_input_recipe_inputs as gate
from scripts.newton import probe_public_input_recipe_recovery as recovery
from scripts.newton.probe_patch_witness_trainer_recovery import compare_recovery

ROOT = Path(__file__).resolve().parents[1]


def funding(**changes):
    values = {
        "funded_by": "root",
        "merged_cache_receipt_sha256": "a" * 64,
        "new_CPU_receipt_bytes": gate.CPU_RECEIPT_CAP_BYTES,
        "new_recovery_checkpoint_bytes": gate.CHECKPOINT_CAP_BYTES,
        "new_shared_bytes_files": 0,
        "existing_liabilities_bytes": 50 * gate.GIB,
        "reserve_bytes": gate.RESERVE_BYTES,
    }
    values.update(changes)
    values.setdefault(
        "required_free_bytes",
        values["new_CPU_receipt_bytes"]
        + values["new_recovery_checkpoint_bytes"]
        + values["existing_liabilities_bytes"]
        + values["reserve_bytes"],
    )
    return values


def test_explicit_funding_envelope_is_accepted():
    assert gate.validate_funding(funding())["new_shared_bytes_files"] == 0


@pytest.mark.parametrize(
    "changes",
    [
        {"new_shared_bytes_files": 1},
        {"new_recovery_checkpoint_bytes": gate.CHECKPOINT_CAP_BYTES + 1},
        {"new_CPU_receipt_bytes": gate.CPU_RECEIPT_CAP_BYTES * 2},
        {"reserve_bytes": gate.RESERVE_BYTES - 1},
        {"required_free_bytes": 1},
        {"funded_by": "issue"},
        {"merged_cache_receipt_sha256": "unverified"},
        {"existing_liabilities_bytes": -1},
        {"existing_liabilities_bytes": 1.5},
        {"extra": 0},
    ],
)
def test_unfunded_or_uncapped_envelope_is_rejected(changes):
    with pytest.raises(ValueError):
        gate.validate_funding(funding(**changes))


def stage(tmp_path):
    code = tmp_path / "code"
    for name in (*gate.REQUIRED_SOURCE, "rsna/__init__.py"):
        (code / name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, code / name)
    files = {
        str(path.relative_to(code)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in code.rglob("*")
        if path.is_file()
    }
    return code, {"source_git": "b" * 40, "files": files}


def test_source_manifest_binds_every_staged_trainer_byte(tmp_path):
    code, manifest = stage(tmp_path)
    hashes = gate.verify_source(code, manifest)
    assert set(hashes) == {"__init__.py", "competitive/train.py", "competitive/public_input.py"}
    assert gate.load_configs(code).keys() == {"native", "public"}


@pytest.mark.parametrize(
    "tamper", ["byte", "unlisted", "missing", "symlink", "escape", "config", "cap", "git"]
)
def test_source_tampering_or_cap_is_rejected(tmp_path, monkeypatch, tamper):
    code, manifest = stage(tmp_path)
    if tamper == "byte":
        with (code / "rsna/competitive/train.py").open("a") as stream:
            stream.write("\n")
    elif tamper == "unlisted":
        (code / "rsna/extra.py").write_text("")
    elif tamper == "missing":
        del manifest["files"]["scripts/newton/enforce_deterministic_runtime.py"]
    elif tamper == "symlink":
        target = code / "rsna/__init__.py"
        target.unlink()
        target.symlink_to(ROOT / "rsna/__init__.py")
    elif tamper == "escape":
        manifest["files"]["../outside.py"] = "c" * 64
    elif tamper == "config":
        path = code / gate.CONFIG_PATHS["public"]
        values = json.loads(path.read_text())
        values["slices"] = 12
        path.write_text(json.dumps(values))
        manifest["files"][gate.CONFIG_PATHS["public"]] = hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
    elif tamper == "cap":
        monkeypatch.setattr(gate, "SOURCE_CAP_BYTES", 1024)
    else:
        manifest["source_git"] = "short"
    with pytest.raises(ValueError):
        gate.verify_source(code, manifest)


def test_pinned_arms_differ_only_in_public_input_and_engineering_schedule():
    configs = gate.load_configs(ROOT)
    engineering = {arm: recovery.engineering_config(configs[arm], arm) for arm in gate.ARMS}
    native, public = engineering["native"], engineering["public"]
    assert {k for k in set(native) | set(public) if native.get(k) != public.get(k)} == {
        "public_input",
        "side",
        "slices",
    }
    for values in engineering.values():
        assert (values["epochs"], values["diagnostic_study_limit"], values["warmup_updates"]) == (
            2,
            16,
            2,
        )
        assert values["batch_size"] * values["accumulation_steps"] == 8
        assert values["encoder"]["adapter"]["rank"] == 32
        assert values["seed"] == configs["native"]["seed"]


@pytest.mark.parametrize(
    "arm,field,value",
    [
        ("native", "precision", "bf16"),
        ("public", "amp_init_scale", 65536),
        ("public", "accumulation_steps", 4),
        ("native", "encode_chunk_size", 4),
        ("native", "side", 336),
        ("native", "public_input", "public"),
        ("public", "rank", 64),
        ("other", None, None),
    ],
)
def test_engineering_recipe_drift_is_rejected(arm, field, value):
    configs = gate.load_configs(ROOT)
    base = copy.deepcopy(configs["public" if arm == "other" else arm])
    if field == "rank":
        base["encoder"]["adapter"]["rank"] = value
    elif field == "public_input":
        base["public_input"] = configs["public"]["public_input"]
    elif field is not None:
        base[field] = value
    with pytest.raises(ValueError):
        recovery.engineering_config(base, arm)


def test_complete_cache_receipt_binds_root_verified_index(tmp_path):
    cohort_receipt = {"index_sha256": "1" * 64, "cohort_sha256": "2" * 64, "members": 3403}
    receipt = {
        "cache": str(tmp_path),
        "studies": 3403,
        "fitting_studies": 2247,
        "development_studies": 1156,
        "complete_shards": 20,
        "index_sha256": "1" * 64,
        "cohort_manifest_sha256": gate.SOURCE_COHORT_MANIFEST_SHA256,
        "status": "actual_complete_public_source_cache_restored_finalized_and_runtime_replayed",
        "all_adapter_tensor_replays_equal_original_Kaggle": True,
        "labels_in_cache": False,
        "GPU_model_optimizer_or_accuracy_executed": False,
    }
    gate.validate_merged_receipt(receipt, cohort_receipt, tmp_path)
    original_schema = {key: value for key, value in receipt.items() if key != "cache"}
    gate.validate_merged_receipt(original_schema, cohort_receipt, tmp_path)
    for key, value in [
        ("cache", "/elsewhere"),
        ("studies", 3402),
        ("index_sha256", "3" * 64),
        ("cohort_manifest_sha256", "3" * 64),
        ("all_adapter_tensor_replays_equal_original_Kaggle", False),
        ("status", "pending"),
    ]:
        with pytest.raises(ValueError):
            gate.validate_merged_receipt(dict(receipt, **{key: value}), cohort_receipt, tmp_path)
    with pytest.raises(ValueError):
        gate.validate_merged_receipt(receipt, dict(cohort_receipt, members=16), tmp_path)


def test_public_engineering_tensors_are_source_uint8_divided_once(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ENGINEERING_STUDIES", 4)
    manifest, folds = cohort()
    rows = build(tmp_path, manifest, folds)
    studies = dataset(tmp_path, manifest)
    proofs, window = gate.tensor_proofs(manifest, studies, "public")
    assert set(proofs) == {"images", "other", "labels"}
    valid = torch.from_numpy(rows["s0"]["slot_mask"])
    assert torch.equal(window, torch.from_numpy(rows["s0"]["images"][:, :3]).float()[valid] / 255)
    again, _ = gate.tensor_proofs(manifest, dataset(tmp_path, manifest), "public")
    assert again == proofs


@pytest.mark.parametrize("tamper", ["twice", "absent", "label", "mask", "uid", "shape"])
def test_public_engineering_tensor_or_label_tampering_is_rejected(tmp_path, tamper):
    manifest, folds = cohort()
    build(tmp_path, manifest, folds)
    studies = dataset(tmp_path, manifest)
    item = studies[0]
    if tamper == "twice":
        item["images"] = item["images"] / 255
    elif tamper == "absent":
        item["images"][~item["series_mask"].bool()] = 0.5
    elif tamper == "label":
        item["labels"] = item["labels"] + item["label_mask"]
    elif tamper == "mask":
        item["label_mask"] = 1 - item["label_mask"]
    elif tamper == "uid":
        item["study_uid"] = "s1"
    else:
        item["images"] = item["images"][:, :9]
    with pytest.raises(ValueError):
        gate.check_public_item(studies, item)
        gate.check_label_rows(manifest, item, 0)


@pytest.mark.parametrize("channels", [1, 3])
def test_imagenet_normalization_is_applied_exactly_once(channels):
    encoder = _GrayEncoder(Capture(), 16, input_channels=channels)
    window = torch.rand(2, channels, 32, 32)
    assert len(gate.check_imagenet_once(encoder, window)) == 64
    with pytest.raises(ValueError):
        gate.check_imagenet_once(encoder, (window - 0.485) / 0.229)
    encoder.mean.add_(0.01)
    with pytest.raises(ValueError):
        gate.check_imagenet_once(encoder, window)


def factory(provenance):
    def create(spec, input_channels=1, input_contract="neighbor_delta"):
        contract = "gray_0_1_imagenet_rgb_v1"
        if input_channels == 3:
            assert input_contract == "public_ordered_triplet"
            contract = "public_ordered_triplet_0_1_imagenet_rgb_v1"
        torch.rand(3)  # adapter construction consumes RNG identically in both arms
        return (
            _GrayEncoder(Capture(), 16, input_channels=input_channels),
            3,
            dict(provenance, input_contract=contract),
        )

    return create


def initialized(configs, create=None):
    create = create or factory(
        {"weights_sha256": "a" * 64, "config_sha256": "c" * 64, "pretrained": True}
    )
    states, provenance = {}, {}
    for arm in gate.ARMS:
        config = dict(configs[arm], hidden_dim=12)
        _, provenance[arm], states[arm] = gate.initialize(config, arm, create, CompetitiveStudyNet)
    return states, provenance


def test_native_and_public_wrappers_share_learned_initialization_and_all_rng():
    configs = gate.load_configs(ROOT)
    states, provenance = initialized(configs)
    gate.common_initialization(states, provenance)
    assert states["native"]["learned_parameters"] > 0
    assert len(states["public"]["frozen_base_sha256"]) == 64


@pytest.mark.parametrize("tamper", ["seed", "hidden", "provenance", "contract"])
def test_initialization_or_provenance_mismatch_is_rejected(tamper):
    configs = copy.deepcopy(gate.load_configs(ROOT))
    states, provenance = initialized(configs)
    if tamper == "seed":
        configs["public"]["seed"] += 1
        states, provenance = initialized(configs)
    elif tamper == "hidden":
        states["public"]["learned_parameters"] += 1
    elif tamper == "provenance":
        provenance["public"]["weights_sha256"] = "x" * 64
    else:
        provenance["public"]["input_contract"] = "neighbor_delta_0_1_imagenet_rgb_v1"
    with pytest.raises(ValueError):
        gate.common_initialization(states, provenance)


def test_cpu_receipt_is_written_once_under_its_cap(tmp_path, monkeypatch):
    gate.write_once(tmp_path / "CPU-accepted.json", {"status": gate.STATUS})
    with pytest.raises(FileExistsError):
        gate.write_once(tmp_path / "CPU-accepted.json", {"status": gate.STATUS})
    monkeypatch.setattr(gate, "CPU_RECEIPT_CAP_BYTES", 8)
    with pytest.raises(ValueError):
        gate.write_once(tmp_path / "other.json", {"status": gate.STATUS})
    assert not (tmp_path / "other.json").exists()


def test_cpu_and_gpu_wrappers_refuse_unowned_placement(monkeypatch):
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    with pytest.raises(ValueError):
        gate.check_placement()
    with pytest.raises(ValueError):
        recovery.check_placement()


def cpu_receipt():
    configs = gate.load_configs(ROOT)
    tensors = {}
    for role in ("fitting", "development"):
        tensors[f"native/{role}"] = {
            kind: gate.NATIVE_TENSOR_SHA256[kind][role] for kind in ("images", "other", "labels")
        }
        tensors[f"public/{role}"] = dict(tensors[f"native/{role}"], images="p" * 64)
    state = {
        "learned_graph_values_sha256": "g" * 64,
        "post_factory_RNG_sha256": "r" * 64,
        "frozen_base_sha256": gate.canonical({"encoder.base": "b" * 64}),
    }
    return {
        "status": gate.STATUS,
        "source_manifest_sha256": "m" * 64,
        "funding_sha256": "f" * 64,
        "code_sha256": {"competitive/train.py": "t" * 64},
        "config_sha256": dict(gate.CONFIG_SHA256),
        "configs": configs,
        "caches": {"native": gate.NATIVE_CACHE, "public": gate.PUBLIC_CACHE},
        "initial_learned_state": {"native": dict(state), "public": dict(state)},
        "GPU_or_optimizer_or_clinical_inference_executed": False,
        "merge_render_download_or_transfer_executed": False,
        "protected_labels_used_for_recipe_or_metrics": False,
        "original_large_base_proof_and_stats_reused": True,
        "public_cohort": {"members": 3403, "index_sha256": "i" * 64},
        "full_fitting_studies": 2247,
        "full_outer_studies": 1156,
        "tensor_sha256": tensors,
        "training_16_sha256": "u" * 64,
        "outer_16_sha256": "o" * 64,
        "supervision": {"sha256": gate.OVERLAY_SHA256},
        "encoder_provenance": {
            "native": {"input_contract": "gray_0_1_imagenet_rgb_v1"},
            "public": {"input_contract": "public_ordered_triplet_0_1_imagenet_rgb_v1"},
        },
        "trainer_inputs": {
            "overlay": "/o",
            "studies": "/s",
            "artifact-root": "/a",
            "folds": "/f",
            "series": "/r",
            "cache": gate.NATIVE_CACHE,
        },
    }


def validate(cpu):
    return recovery.validate_cpu_receipt(
        cpu, "m" * 64, "f" * 64, {"competitive/train.py": "t" * 64}, gate.load_configs(ROOT)
    )


def test_independent_cpu_receipt_is_required_by_recovery():
    assert validate(cpu_receipt())["status"] == gate.STATUS


@pytest.mark.parametrize(
    "tamper",
    [
        "status",
        "manifest",
        "funding",
        "code",
        "config",
        "cache",
        "init",
        "gpu",
        "merge",
        "gold",
        "members",
        "native_tensor",
        "labels",
    ],
)
def test_changed_cpu_receipt_is_rejected(tamper):
    cpu = cpu_receipt()
    if tamper == "status":
        cpu["status"] = "tiny diagnostic accuracy"
    elif tamper == "manifest":
        cpu["source_manifest_sha256"] = "x" * 64
    elif tamper == "funding":
        cpu["funding_sha256"] = "x" * 64
    elif tamper == "code":
        cpu["code_sha256"] = {}
    elif tamper == "config":
        cpu["configs"]["public"]["slices"] = 12
    elif tamper == "cache":
        cpu["caches"]["public"] = gate.NATIVE_CACHE
    elif tamper == "init":
        cpu["initial_learned_state"]["public"]["post_factory_RNG_sha256"] = "x" * 64
    elif tamper == "gpu":
        cpu["GPU_or_optimizer_or_clinical_inference_executed"] = True
    elif tamper == "merge":
        cpu["merge_render_download_or_transfer_executed"] = True
    elif tamper == "gold":
        cpu["protected_labels_used_for_recipe_or_metrics"] = True
    elif tamper == "members":
        cpu["public_cohort"]["members"] = 16
    elif tamper == "native_tensor":
        cpu["tensor_sha256"]["native/fitting"]["images"] = "x" * 64
    else:
        cpu["tensor_sha256"]["public/development"]["labels"] = "x" * 64
    with pytest.raises(ValueError):
        validate(cpu)


def recovered(arm, cpu):
    values = recovery.engineering_config(cpu["configs"][arm], arm)
    bound = {key: value for key, value in values.items() if key != "encoder"}
    bound["encoder"] = cpu["encoder_provenance"][arm]
    binding = {
        "config": bound,
        "training_uids_sha256": cpu["training_16_sha256"],
        "outer_uids_sha256": cpu["outer_16_sha256"],
        "code_sha256": cpu["code_sha256"],
        "device_type": "cuda",
        "world_size": 1,
        "experimental_fitting_supervision": cpu["supervision"],
    }
    if arm == "public":
        binding.update(public_input_cache=cpu["public_cohort"], private_research_only=True)
    rng = {
        "torch": torch.tensor([1], dtype=torch.uint8),
        "cuda": torch.tensor([2], dtype=torch.uint8),
        "python": (3, (1, 2), None),
        "numpy": ["MT19937", [1], 0, 0, 0.0],
    }
    full = dict(
        schema_version=2,
        optimizer_boundary=True,
        epoch=2,
        next_update=0,
        step=4,
        binding=binding,
        model={"encoder.lora_A": torch.tensor([0.25]), "head.weight": torch.tensor([0.5])},
        model_storage={
            "format": "frozen_base_delta_v1",
            "encoder_provenance": cpu["encoder_provenance"][arm],
            "frozen_tensors": {"encoder.base": "b" * 64},
        },
        optimizer={
            "state": {0: {"step": torch.tensor(4.0), "exp_avg": torch.tensor([0.5])}},
            "param_groups": [{"lr": 1e-4, "params": [0]}, {"lr": 3e-4, "params": []}],
        },
        amp_scaler={"scale": 128.0, "_growth_tracker": 4},
        rng_by_rank=[rng],
    )
    partial = copy.deepcopy(full)
    partial.update(epoch=0, next_update=1, step=1)
    rows = [
        dict(epoch=e, update=u, optimizer_steps=s, loss=0.5, gradient_norm=1.0, elapsed_seconds=s)
        for e, u, s in [(0, 1, 1), (0, 2, 2), (1, 1, 3), (1, 2, 4)]
    ]
    return values, full, partial, rows


@pytest.mark.parametrize("arm", gate.ARMS)
def test_decoded_complete_recovery_state_is_bound_to_admission(arm):
    cpu = cpu_receipt()
    values, full, partial, rows = recovered(arm, cpu)
    history = ({"updates": rows}, {"updates": rows[:1]}, {"updates": rows[1:]})
    state = compare_recovery(full, copy.deepcopy(full), partial, *history, b"csv", b"csv")
    assert len(state) == 64
    recovery.validate_state(full, values, arm, cpu, cpu["code_sha256"])


@pytest.mark.parametrize(
    "tamper",
    [
        "roster",
        "outer",
        "frozen",
        "code",
        "device",
        "config",
        "base",
        "supervision",
        "public_cache",
        "scaler",
        "scale",
        "lr",
        "counter",
        "rng",
        "cuda_rng",
    ],
)
def test_recovered_state_tampering_is_rejected(tamper):
    cpu = cpu_receipt()
    arm = "native" if tamper == "public_cache" else "public"
    values, full, _, _ = recovered(arm, cpu)
    binding = full["binding"]
    if tamper == "roster":
        binding["training_uids_sha256"] = "x" * 64
    elif tamper == "outer":
        binding["outer_uids_sha256"] = "x" * 64
    elif tamper == "frozen":
        full["model_storage"]["frozen_tensors"] = {"encoder.base": "x" * 64}
    elif tamper == "code":
        binding["code_sha256"] = {}
    elif tamper == "device":
        binding["device_type"] = "cpu"
    elif tamper == "config":
        binding["config"]["epochs"] = 10
    elif tamper == "base":
        full["model_storage"]["encoder_provenance"] = {"input_contract": "other"}
    elif tamper == "supervision":
        binding["experimental_fitting_supervision"] = {}
    elif tamper == "public_cache":
        binding["public_input_cache"] = cpu["public_cohort"]
    elif tamper == "scaler":
        full["amp_scaler"]["_growth_tracker"] = 3
    elif tamper == "scale":
        full["amp_scaler"]["scale"] = math.inf
    elif tamper == "lr":
        full["optimizer"]["param_groups"][0]["lr"] = math.nan
    elif tamper == "counter":
        full["optimizer"]["state"][0]["step"] = torch.tensor(3.0)
    elif tamper == "rng":
        del full["rng_by_rank"][0]["numpy"]
    else:
        full["rng_by_rank"][0]["cuda"] = None
    with pytest.raises(ValueError):
        recovery.validate_state(full, values, arm, cpu, cpu["code_sha256"])


def predictions(tmp_path, rows=16, value=0.5, columns=None):
    from rsna.schema import TARGETS

    header = columns or ["study_uid", "fold", "seed", *TARGETS]
    lines = [",".join(header)]
    lines += [",".join([f"o{i}", "0", "1", *[str(value)] * 12]) for i in range(rows)]
    path = tmp_path / "outer_predictions.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


def test_exact_development_16x12_prediction_bytes_are_required(tmp_path):
    from scripts.newton.probe_patch_witness_trainer_recovery import digest

    outer = digest(sorted(f"o{i}" for i in range(16)))
    path = predictions(tmp_path)
    assert (
        recovery.validate_predictions(path, outer) == hashlib.sha256(path.read_bytes()).hexdigest()
    )
    for changes in ({"rows": 15}, {"value": 1.5}, {"value": "nan"}):
        with pytest.raises(ValueError):
            recovery.validate_predictions(predictions(tmp_path, **changes), outer)
    with pytest.raises(ValueError):
        recovery.validate_predictions(predictions(tmp_path), "x" * 64)


def test_trainer_is_the_unchanged_canonical_module_under_deterministic_runtime(tmp_path):
    cpu = cpu_receipt()
    command = recovery.trainer_command(tmp_path, tmp_path / "config.json", cpu, "public")
    assert command[2] == str(tmp_path / "scripts/newton/enforce_deterministic_runtime.py")
    assert command[3:5] == ["--module", "rsna.competitive.train"]
    assert command[command.index("--cache") + 1] == gate.PUBLIC_CACHE
    native = recovery.trainer_command(tmp_path, tmp_path / "config.json", cpu, "native")
    assert native[native.index("--cache") + 1] == gate.NATIVE_CACHE
    assert native[native.index("--fitting-supervision") + 1] == "/o"


def test_recovery_output_cap_is_enforced(tmp_path, monkeypatch):
    (tmp_path / "checkpoint.pt").write_bytes(b"x" * 32)
    recovery.check_cap(tmp_path)
    monkeypatch.setattr(recovery, "CHECKPOINT_CAP_BYTES", 16)
    with pytest.raises(ValueError):
        recovery.check_cap(tmp_path)
