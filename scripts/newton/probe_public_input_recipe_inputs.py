"""Owning-node CPU admission for the matched native-versus-public-triplet input recipe.

Rebinds the original release, folds, fitting overlay, native cache and the root-verified
complete public triplet cache, then proves engineering tensors, labels/masks, the uint8
``/255`` then ImageNet-once contract and common official learned initialization/RNG.
No GPU, optimizer, merge, rendering, transfer or clinical inference is executed.
"""

import argparse
import gc
import hashlib
import importlib
import json
import os
import random
from pathlib import Path, PurePosixPath

import numpy as np
import torch

RECIPE = "initiative/experiments/competitive-v1/public-input-recipe-v1"
ARMS = ("native", "public")
CONFIG_PATHS = {
    "native": f"{RECIPE}/control-config.json",
    "public": f"{RECIPE}/public-triplet-config.json",
}
CONFIG_SHA256 = {
    "native": "ec987a8668881dc4674d80fcc712b9912aef51804803f288836c721f555c38bd",
    "public": "5de4b6543fed5f2da715422b6806eedd3e8a8aedc7d055a64d6ee27179fd22c5",
}
ARM_FIELDS = {"public_input", "side", "slices"}
REQUIRED_SOURCE = (
    *CONFIG_PATHS.values(),
    "scripts/newton/probe_public_input_recipe_inputs.py",
    "scripts/newton/probe_public_input_recipe_recovery.py",
    "scripts/newton/probe_patch_witness_trainer_recovery.py",
    "scripts/newton/enforce_deterministic_runtime.py",
    "tools/research/public_dinov2_input_reference/dino_adapter.py",
    "rsna/competitive/train.py",
    "rsna/competitive/public_input.py",
)
NODE = "evc31"
FITTING_STUDIES, OUTER_STUDIES = 2247, 1156
ENGINEERING_STUDIES = 16
ORIGINAL_ROOT = Path("/tmp/rsna-public-label-repeat-inputs-20261012-v1")
ORIGINAL_STATUS = (
    "actual_public_label_source_original_MRI_control_outer_and_official_initialization_CPU_accepted"
)
ORIGINAL_CPU_JOB = "899733"
NATIVE_CACHE = "/tmp/rsna-native384-cohort-866458-lp6rcfdw/restored/cache"
PUBLIC_CACHE = "/tmp/rsna-kaggle-public-source-shards-20261013-v1/merged-cache"
SOURCE_COHORT_MANIFEST_SHA256 = "3f265ddfb5b35857e75745b3476cf7761d6d41a93f3651839272c763ed5e7776"
MERGED_RECEIPT = "/tmp/rsna-kaggle-public-source-shards-20261013-v1/merged-cache-verified.json"
OVERLAY_SHA256 = "c8fd2e860d8a3913dbeab34a9d1078316a8f893360124eac088453dc2ed6cf51"
# Original CPU899733 native tensors under public fitting supervision; no MRI is duplicated.
NATIVE_TENSOR_SHA256 = {
    "images": {
        "fitting": "19acbfacba11125b7ac97b92eef14d28604e45936be53951be77ecebb416b6ba",
        "development": "2bca11d52080164621ad9cd91d33dd41b8f4f927c164c9d15ba85ae434ab3530",
    },
    "other": {
        "fitting": "5572607e3e66d6242aa1d7995e263311a7b297d7185546b0914b3f11fad28f07",
        "development": "fd596b7059974807382ee7a2788b6e613dac2aa94143025a2d4b0a185199c8d1",
    },
    "labels": {
        "fitting": "9144fa0a08baa8c4bf0eb1f43434d93b2a27a07129e7a2967aaa7281ced258d5",
        "development": "4f5a10a106b0dc560923c0c5077529a8c7f37e69df1ebd38bbc449733e9181cf",
    },
}
BASE_PROOF = "/home/sa305415/rsna-knee/production-launch-native384-a32-c9-v1/cpu-preflight.json"
BASE_PROOF_SHA256 = "fa9e8a52ce05e81c08c2c3ae3a5eec28999109be74af2d4c19b8ddef0e76e9b5"
IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
OTHER_KEYS = ("features", "positions", "slice_mask", "series_mask", "labels", "label_mask")
GIB = 2**30
SOURCE_CAP_BYTES = 8 * 2**20
CPU_RECEIPT_CAP_BYTES = 2**20
CHECKPOINT_CAP_BYTES = 6 * GIB
RESERVE_BYTES = 20 * GIB
FUNDING_KEYS = {
    "funded_by",
    "merged_cache_receipt_sha256",
    "new_CPU_receipt_bytes",
    "new_recovery_checkpoint_bytes",
    "new_shared_bytes_files",
    "existing_liabilities_bytes",
    "reserve_bytes",
    "required_free_bytes",
}
STATUS = (
    "public_input_recipe_native_and_public_triplet_inputs_and_official_initialization_CPU_accepted"
)


def sha(path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def signature(path) -> list[int]:
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]


def _is_sha(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= set("0123456789abcdef")


def validate_funding(funding: dict) -> dict:
    """Require explicit root funding for exactly the capped new bytes; never shared growth."""
    if not isinstance(funding, dict) or set(funding) != FUNDING_KEYS:
        raise ValueError("Explicit public-input recipe funding fields differ")
    sizes = [funding[key] for key in FUNDING_KEYS - {"funded_by", "merged_cache_receipt_sha256"}]
    if any(type(value) is not int or value < 0 for value in sizes):
        raise ValueError("Funding sizes must be non-negative integer bytes")
    if (
        funding["funded_by"] != "root"
        or not _is_sha(funding["merged_cache_receipt_sha256"])
        or funding["new_CPU_receipt_bytes"] != CPU_RECEIPT_CAP_BYTES
        or funding["new_recovery_checkpoint_bytes"] != CHECKPOINT_CAP_BYTES
        or funding["new_shared_bytes_files"] != 0
        or funding["reserve_bytes"] < RESERVE_BYTES
        or funding["required_free_bytes"]
        != funding["new_CPU_receipt_bytes"]
        + funding["new_recovery_checkpoint_bytes"]
        + funding["existing_liabilities_bytes"]
        + funding["reserve_bytes"]
    ):
        raise ValueError("Public-input recipe funding/caps differ from the explicit envelope")
    return funding


def verify_source(code: Path, manifest: dict) -> dict:
    """Bind every staged source byte to the reviewed manifest under the source cap."""
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"source_git", "files"}
        or not isinstance(manifest["source_git"], str)
        or len(manifest["source_git"]) != 40
        or not isinstance(manifest["files"], dict)
        or not set(REQUIRED_SOURCE) <= set(manifest["files"])
    ):
        raise ValueError("Immutable public-input recipe source manifest differs")
    code = code.resolve()
    total = 0
    for name, digest in manifest["files"].items():
        relative = PurePosixPath(name)
        path = code / name
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or not _is_sha(digest)
            or path.is_symlink()
            or not path.is_file()
            or path.resolve().parent != (code / relative.parent).resolve()
        ):
            raise ValueError("Unsafe or missing staged source member")
        total += path.stat().st_size
        if total > SOURCE_CAP_BYTES:
            raise ValueError("Staged source exceeds its explicit cap")
        if sha(path) != digest:
            raise ValueError("Staged numerical source differs from its manifest")
    staged = {
        str(path.relative_to(code))
        for path in (code / "rsna").rglob("*.py")
        if "__pycache__" not in path.parts
    }
    if staged != {name for name in manifest["files"] if name.startswith("rsna/")}:
        raise ValueError("Staged trainer package differs from its manifest")
    for arm in ARMS:
        if manifest["files"][CONFIG_PATHS[arm]] != CONFIG_SHA256[arm]:
            raise ValueError("Actual-source-bound recipe configuration differs")
    return {
        name.removeprefix("rsna/"): digest
        for name, digest in manifest["files"].items()
        if name.startswith("rsna/")
    }


def load_configs(code: Path) -> dict[str, dict]:
    """Load both pinned arms and require that only the input fields differ."""
    from rsna.competitive.train import TrainConfig

    configs = {}
    for arm in ARMS:
        path = code / CONFIG_PATHS[arm]
        if sha(path) != CONFIG_SHA256[arm]:
            raise ValueError("Actual-source-bound recipe configuration differs")
        configs[arm] = json.loads(path.read_text())
        TrainConfig(**configs[arm]).validate()
    native, public = configs["native"], configs["public"]
    if (
        "public_input" in native
        or public.get("public_input") is None
        or (native["side"], native["slices"], public["side"], public["slices"])
        != (384, 16, 336, 10)
        or {key for key in set(native) | set(public) if native.get(key) != public.get(key)}
        != ARM_FIELDS
        or native["encoder"]["adapter"]["rank"] != 32
        or (native["precision"], native["amp_init_scale"], native["encode_chunk_size"])
        != ("fp16", 128, 8)
        or native["batch_size"] * native["accumulation_steps"] != 8
        or native["fit_scope"] != "outer_fold"
        or native["fitting_supervision"]
        != {"enabled": True, "kind": "label_source_v1", "sha256": OVERLAY_SHA256}
    ):
        raise ValueError("Matched recipe arms must differ only in public input/side/slices")
    return configs


def validate_original_receipt(original: dict, overlay: Path) -> dict:
    """Reuse the original label-source CPU admission without repeating its collection."""
    inputs = original.get("trainer_inputs", {})
    if (
        original.get("status") != ORIGINAL_STATUS
        or original.get("CPU_job") != ORIGINAL_CPU_JOB
        or set(inputs) != {"artifact-root", "cache", "folds", "series", "studies"}
        or inputs["cache"] != NATIVE_CACHE
        or (original["full_fitting_studies"], original["full_outer_studies"])
        != (FITTING_STUDIES, OUTER_STUDIES)
        or original["image_tensor_sha256"]["public"] != NATIVE_TENSOR_SHA256["images"]["fitting"]
        or original["image_tensor_sha256"]["outer_engineering"]
        != NATIVE_TENSOR_SHA256["images"]["development"]
        or original["other_tensor_sha256"]["public"] != NATIVE_TENSOR_SHA256["other"]["fitting"]
        or original["other_tensor_sha256"]["outer_engineering"]
        != NATIVE_TENSOR_SHA256["other"]["development"]
        or original["protected_labels_used_for_recipe_or_metrics"] is not False
        or original["GPU_or_optimizer_or_clinical_inference_executed"] is not False
        or sha(overlay) != OVERLAY_SHA256
    ):
        raise ValueError("Original label-source CPU admission/overlay binding differs")
    return inputs


def validate_merged_receipt(receipt: dict, cohort: dict, cache: Path) -> None:
    """Accept only root's terminal complete-cache verification of this exact index."""
    if (
        not isinstance(receipt, dict)
        or ("cache" in receipt and receipt["cache"] != str(cache))
        or receipt.get("status")
        != "actual_complete_public_source_cache_restored_finalized_and_runtime_replayed"
        or receipt.get("fitting_studies") != FITTING_STUDIES
        or receipt.get("development_studies") != OUTER_STUDIES
        or receipt.get("complete_shards") != 20
        or receipt.get("all_adapter_tensor_replays_equal_original_Kaggle") is not True
        or receipt.get("labels_in_cache") is not False
        or receipt.get("GPU_model_optimizer_or_accuracy_executed") is not False
        or receipt.get("studies") != FITTING_STUDIES + OUTER_STUDIES
        or receipt.get("index_sha256") != cohort["index_sha256"]
        or receipt.get("cohort_manifest_sha256") != SOURCE_COHORT_MANIFEST_SHA256
        or cohort["members"] != FITTING_STUDIES + OUTER_STUDIES
    ):
        raise ValueError("Root complete-cache verification differs from the bound public cache")


def check_label_rows(part, item, index: int) -> None:
    from rsna.schema import TARGETS

    row = part.iloc[index]
    mask = np.asarray([bool(row[target + "__mask"]) for target in TARGETS])
    labels = np.asarray(
        [row[target] if valid else 0 for target, valid in zip(TARGETS, mask, strict=True)],
        dtype=np.float32,
    )
    if (
        item["study_uid"] != row.study_uid
        or not np.array_equal(labels, item["labels"].numpy())
        or not np.array_equal(
            mask.astype(np.float32), item["label_mask"].numpy().astype(np.float32)
        )
    ):
        raise ValueError("Engineering label/mask tensor differs from the admitted source frame")


def check_public_item(dataset, item) -> torch.Tensor:
    """Require exact overlapping uint8 windows divided by 255 once; return slot-valid views."""
    raw = dataset._member(item["study_uid"])["images"]
    images, valid = item["images"], item["series_mask"].bool()
    if (
        raw.dtype != torch.uint8
        or images.dtype != torch.float32
        or images.shape != (6, 10, 3, 336, 336)
        or not torch.isfinite(images).all()
        or not valid.any()
    ):
        raise ValueError("Public triplet engineering tensor contract differs")
    for view in range(10):
        if not torch.equal(images[:, view], raw[:, view : view + 3].float() / 255):
            raise ValueError("Public triplet window is not the exact source uint8 divided once")
    if images.amax() > 1 or images.amin() < 0 or images[~valid].any():
        raise ValueError("Public triplet values/absent slots differ from the 0..1 contract")
    if not torch.equal(images[:, 0], raw[:, :3].float() / 255):
        raise ValueError("Public triplet first window differs from source slices 0..2")
    return images[valid, 0]


def tensor_proofs(part, dataset, arm: str) -> tuple[dict, torch.Tensor]:
    hashes = {name: hashlib.sha256() for name in ("images", "other", "labels")}
    first = None
    for index in range(ENGINEERING_STUDIES):
        item = dataset[index]
        check_label_rows(part, item, index)
        if arm == "public":
            window = check_public_item(dataset, item)
        else:
            images = item["images"]
            if images.dtype != torch.float32 or images.shape[1:] != (16, 1, 384, 384):
                raise ValueError("Native engineering tensor contract differs")
            if not torch.isfinite(images).all() or images.amin() < 0 or images.amax() > 1:
                raise ValueError("Native engineering values differ from the 0..1 contract")
            window = images[item["series_mask"].bool(), 0]
        if first is None:
            first = window.clone()
        hashes["images"].update(item["images"].numpy().tobytes())
        for key in OTHER_KEYS:
            hashes["other"].update(item[key].numpy().tobytes())
        for key in ("labels", "label_mask"):
            hashes["labels"].update(item[key].numpy().tobytes())
    if first is None:
        raise ValueError("No engineering tensors")
    return {name: value.hexdigest() for name, value in hashes.items()}, first


def check_imagenet_once(encoder, window: torch.Tensor) -> str:
    """The cache/adapter stay in [0,1]; the encoder normalizes exactly once."""
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    if not (
        torch.equal(encoder.mean.cpu().float(), mean)
        and torch.equal(encoder.std.cpu().float(), std)
    ):
        raise ValueError("Encoder ImageNet statistics differ")
    rgb = window.expand(-1, 3, -1, -1) if window.shape[1] == 1 else window
    expected = (rgb - mean) / std
    observed = encoder._normalized_rgb(window)
    if not torch.equal(observed, expected):
        raise ValueError("ImageNet normalization is not applied exactly once")
    return hashlib.sha256(observed.contiguous().numpy().tobytes()).hexdigest()


def learned_state(model) -> dict:
    from rsna.semantic_supervision import digest

    records = {}
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            array = parameter.detach().contiguous().numpy()
            records[name] = {
                "shape": list(array.shape),
                "dtype": str(array.dtype),
                "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
            }
    return {
        "learned_graph_values_sha256": digest(records),
        "post_factory_RNG_sha256": rng_digest(),
        "learned_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def rng_digest() -> str:
    python = random.getstate()
    numpy_state = np.random.get_state()
    return canonical(
        {
            "torch": hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            "python": [python[0], list(python[1]), python[2]],
            "numpy": [
                numpy_state[0],
                hashlib.sha256(numpy_state[1].tobytes()).hexdigest(),
                *[float(value) for value in numpy_state[2:]],
            ],
        }
    )


def initialize(config: dict, arm: str, create_encoder, model_class) -> tuple:
    """Replay the canonical trainer seed/factory order for one input arm."""
    from rsna.competitive.adapters import configure_encoder_training
    from rsna.competitive.delta_checkpoint import FrozenBaseDelta
    from rsna.competitive.relations import relation_model_kwargs

    # A cold Hugging Face modeling import consumes Python RNG once. The two
    # arms must see the same import state before their independent seed reset.
    importlib.import_module("transformers.models.dinov3_vit.modeling_dinov3_vit")
    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed % 2**32)
    torch.manual_seed(seed)
    three = {"input_channels": 3, "input_contract": "public_ordered_triplet"}
    encoder, dim, provenance = create_encoder(
        config["encoder"], **(three if arm == "public" else {})
    )
    configure_encoder_training(encoder, config["freeze_encoder"])
    model = model_class(
        encoder,
        dim,
        pooling=config["pooling"],
        hidden_dim=config["hidden_dim"],
        encode_chunk_size=config["encode_chunk_size"],
        slice_context=config["slice_context"],
        **relation_model_kwargs(config.get("relation")),
        **({"input_channels": 3} if arm == "public" else {}),
    )
    state = learned_state(model)
    frozen = FrozenBaseDelta(model, provenance).metadata()["frozen_tensors"]
    return model, provenance, dict(state, frozen_base_sha256=canonical(frozen))


def common_initialization(states: dict, provenance: dict) -> None:
    if states["native"] != states["public"]:
        raise ValueError("Official common learned graph/values/RNG differ between input arms")
    contracts = {
        "native": "gray_0_1_imagenet_rgb_v1",
        "public": "public_ordered_triplet_0_1_imagenet_rgb_v1",
    }
    for arm in ARMS:
        if provenance[arm].get("input_contract") != contracts[arm]:
            raise ValueError("Encoder input contract differs from the input arm")
    strip = {
        arm: {
            k: v
            for k, v in provenance[arm].items()
            if k not in {"input_contract", "input_channels"}
        }
        for arm in ARMS
    }
    if strip["native"] != strip["public"]:
        raise ValueError("Official encoder provenance differs between input arms")


def write_once(path: Path, value: dict) -> None:
    payload = json.dumps(value, sort_keys=True, indent=2).encode() + b"\n"
    if len(payload) > CPU_RECEIPT_CAP_BYTES:
        raise ValueError("CPU receipt exceeds its explicit cap")
    with path.open("xb") as stream:
        stream.write(payload)


def check_placement() -> None:
    if (
        os.uname().nodename != NODE
        or not os.environ.get("SLURM_JOB_ID")
        or any(os.environ.get(name) for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS"))
        or os.environ.get("CUDA_VISIBLE_DEVICES") not in {None, ""}
    ):
        raise ValueError("Original owning-node CPU-only allocation required")


def run(root: Path) -> dict:
    import shutil

    from rsna.competitive import backbones
    from rsna.competitive.data import CompetitiveStudyDataset
    from rsna.competitive.folds import load_folds
    from rsna.competitive.model import CompetitiveStudyNet
    from rsna.competitive.public_input import PublicTripletStudyDataset, verify_public_cohort
    from rsna.competitive.supervision import apply_fitting_supervision
    from rsna.competitive.train import _uid_hash, select_development
    from rsna.release import load_verified_release

    check_placement()
    torch.set_num_threads(1)
    code = root / "code"
    manifest = json.loads((root / "source.json").read_text())
    code_sha256 = verify_source(code, manifest)
    funding_path = root / "funding.json"
    funding = validate_funding(json.loads(funding_path.read_text()))
    if shutil.disk_usage(root).free < funding["required_free_bytes"]:
        raise ValueError("Funded node-local free space is unavailable")
    configs = load_configs(code)
    overlay = ORIGINAL_ROOT / "overlay.json"
    original_path = ORIGINAL_ROOT / "CPU-accepted.json"
    inputs = validate_original_receipt(json.loads(original_path.read_text()), overlay)
    frame, binding = load_verified_release(Path(inputs["studies"]), Path(inputs["artifact-root"]))
    folds_path = Path(inputs["folds"])
    folds = load_folds(folds_path, frame, binding)
    fitting, outer = select_development(frame, folds, 0, "outer_fold")
    if (len(fitting), len(outer)) != (FITTING_STUDIES, OUTER_STUDIES) or set(
        fitting.leakage_group
    ) & set(outer.leakage_group):
        raise ValueError("Original fitting/evaluation population or leakage boundary changed")
    fitting, supervision = apply_fitting_supervision(
        fitting,
        outer,
        overlay,
        configs["public"]["fitting_supervision"],
        release_sha256=binding["manifest_sha256"],
        folds_sha256=sha(folds_path / "fold.csv"),
        fold=0,
    )
    public_cache = Path(PUBLIC_CACHE)
    policy = configs["public"]["public_input"]
    cohort = verify_public_cohort(public_cache, policy, fitting, outer, binding["source_revision"])
    receipt_path = Path(MERGED_RECEIPT)
    if sha(receipt_path) != funding["merged_cache_receipt_sha256"]:
        raise ValueError("Root complete-cache verification receipt changed")
    validate_merged_receipt(json.loads(receipt_path.read_text()), cohort, public_cache)
    series_bytes = Path(inputs["series"]).read_bytes()
    if hashlib.sha256(series_bytes).hexdigest() != binding["series_sha256"]:
        raise ValueError("Series sampling table differs from the accepted release")
    import io

    import pandas as pd

    series = pd.read_csv(io.BytesIO(series_bytes))
    parts = {
        "fitting": fitting.head(ENGINEERING_STUDIES),
        "development": outer.head(ENGINEERING_STUDIES),
    }
    if set(parts["fitting"].leakage_group) & set(parts["development"].leakage_group):
        raise ValueError("Engineering fitting/development tensors are not disjoint")
    proofs, windows = {}, {}
    for arm in ARMS:
        config = configs[arm]
        common = {
            "slices": config["slices"],
            "side": config["side"],
            "max_series": config["max_series"],
            "slice_sampling": config["slice_sampling"],
            "series_sampling": config["series_sampling"],
        }
        for role, part in parts.items():
            soft = {"soft_targets": True} if role == "fitting" else {}
            if arm == "public":
                dataset = PublicTripletStudyDataset(
                    part,
                    series,
                    public_cache,
                    public_input=policy,
                    source_revision=binding["source_revision"],
                    **common,
                    **soft,
                )
            else:
                dataset = CompetitiveStudyDataset(
                    part, series, Path(NATIVE_CACHE), **common, **soft
                )
            proofs[arm, role], window = tensor_proofs(part, dataset, arm)
            windows.setdefault(arm, window)
            del dataset
            gc.collect()
    for role in parts:
        if (
            proofs["native", role]["images"] != NATIVE_TENSOR_SHA256["images"][role]
            or proofs["native", role]["other"] != NATIVE_TENSOR_SHA256["other"][role]
        ):
            raise ValueError("Native engineering tensors differ from the original admission")
        if proofs["native", role]["labels"] != proofs["public", role]["labels"]:
            raise ValueError("Input arms disagree on engineering labels/masks")
    proof_path = Path(BASE_PROOF)
    if sha(proof_path) != BASE_PROOF_SHA256:
        raise ValueError("Original whole-byte scientific asset proof changed")
    asset_proof = json.loads(proof_path.read_text())
    checked = backbones._checked_file
    stats = {}

    def original_file(spec, key):
        if key == "weights":
            path = Path(spec["weights_path"])
            record = asset_proof["scientific_assets"][str(path)]
            stats[str(path)] = signature(path)
            if (
                stats[str(path)] != record["signature"]
                or spec["weights_sha256"] != record["sha256"]
            ):
                raise ValueError("Original whole-byte-verified official base stat binding changed")
            return path
        return checked(spec, key)

    states, provenance, normalized = {}, {}, {}
    backbones._checked_file = original_file
    try:
        for arm in ARMS:
            model, provenance[arm], states[arm] = initialize(
                configs[arm], arm, backbones.create_encoder, CompetitiveStudyNet
            )
            normalized[arm] = check_imagenet_once(model.encoder, windows[arm])
            del model
            gc.collect()
    finally:
        backbones._checked_file = checked
    common_initialization(states, provenance)
    value = {
        "status": STATUS,
        "CPU_job": os.environ["SLURM_JOB_ID"],
        "source_git": manifest["source_git"],
        "source_manifest_sha256": sha(root / "source.json"),
        "code_sha256": code_sha256,
        "funding_sha256": sha(funding_path),
        "config_sha256": CONFIG_SHA256,
        "configs": configs,
        "trainer_inputs": {**inputs, "overlay": str(overlay)},
        "caches": {"native": NATIVE_CACHE, "public": PUBLIC_CACHE},
        "original_CPU_receipt_sha256": sha(original_path),
        "merged_cache_receipt_sha256": funding["merged_cache_receipt_sha256"],
        "public_cohort": cohort,
        "supervision": supervision,
        "full_fitting_studies": FITTING_STUDIES,
        "full_outer_studies": OUTER_STUDIES,
        "training_16_sha256": _uid_hash(parts["fitting"]),
        "outer_16_sha256": _uid_hash(parts["development"]),
        "tensor_sha256": {f"{arm}/{role}": proof for (arm, role), proof in proofs.items()},
        "first_window_imagenet_sha256": normalized,
        "initial_learned_state": states,
        "encoder_provenance": provenance,
        "official_base_fresh_stats": stats,
        "original_large_base_proof_and_stats_reused": True,
        "GPU_or_optimizer_or_clinical_inference_executed": False,
        "merge_render_download_or_transfer_executed": False,
        "protected_labels_used_for_recipe_or_metrics": False,
        "diagnostic_only": True,
    }
    write_once(root / "CPU-accepted.json", value)
    print(json.dumps({"status": STATUS, "CPU_job": value["CPU_job"]}), flush=True)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
