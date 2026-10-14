"""Owning-node CPU admission for the integrated dense-suffix capacity experiment.

One CPU-only, zero-GPU step on the original owning node. It reuses the original
public-input CPU receipt, root's complete-cache receipt and the whole-byte official
weight proof (fresh file-stat binding only), replays the 16 fitting / 16 development
public tensors through the exact staged loader code, and constructs both predeclared
arms (suffix R32 and dense delta) in the canonical trainer seed/factory order.

Nothing here trains, reserves a GPU, materializes the optional prefix cache,
downloads, renders, transfers or rehashes the cohort. One sanitized receipt
(hashes/counts only, no case-level field) is written exclusively. It does not
accept GPU recovery; root must still run the unchanged full/stop1/resume wrapper.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import shutil
from collections.abc import Iterable, Mapping
from pathlib import Path

import torch

from scripts.newton import probe_public_input_recipe_inputs as gate
from scripts.newton import run_dense_suffix_recovery as recovery

STATUS = "dense_suffix_capacity_owning_node_inputs_and_matched_initialization_CPU_accepted"
ARMS = tuple(recovery.ARMS)
WRAPPER = "scripts/newton/run_dense_suffix_recovery.py"
SELF = "scripts/newton/admit_dense_suffix_inputs.py"
ORIGINAL_RECEIPT = (
    "initiative/audits/public-input-corrected-CPU-dispatch-20261013-v2/CPU-accepted.json"
)
ORIGINAL_RECEIPT_SHA256 = "adb931d591c3cbe4bac79ad8ad1050ece5c9cd2c37dc98e818716e731f050ce2"
ORIGINAL_CPU_JOB = "899766"
MERGED_RECEIPT_SHA256 = "d2c0ad74370f7613b1a9983b9437b7e8e407d02242bb3328be36bf7a57d81ee0"
ORIGINAL_OVERLAY = Path("/tmp/rsna-public-label-repeat-inputs-20261012-v1/overlay.json")
RECIPE_SHA256 = {
    "suffix_r32": "92b6116f9d2ee7311dd74890b37fff7ea12474d6cd21a5df71922d375e78d2aa",
    "dense": "19332d2aa5c80d8134d76978e9f8e54fe56407b744002826acaafef2fd0984d6",
}
REQUIRED_SOURCE = (
    SELF,
    WRAPPER,
    "scripts/newton/probe_public_input_recipe_inputs.py",
    "scripts/newton/probe_patch_witness_trainer_recovery.py",
    "scripts/newton/enforce_deterministic_runtime.py",
    "tools/research/public_dinov2_input_reference/dino_adapter.py",
    ORIGINAL_RECEIPT,
    *(f"{recovery.RECIPE}/{name}" for name in recovery.ARMS.values()),
)
# Exact rsna/ delta since the original receipt's code. Any other difference,
# including any further input/label-loader edit, is refused before data access.
CODE_DELTA = {
    "changed": {
        "competitive/throughput.py": (
            "d35ae9c617d9446b48bcbc00e7defd9a0ba3e9daf5f2a64bcfb20125a60d4e5d"
        ),
        "competitive/train.py": "65ba6b1129ded73bf7445af326acbb4967fc4b5d15dfe406b33955146bd874f9",
    },
    "added": {
        "competitive/weight_average.py": (
            "c50d0427e6ff1369aa4f60a91dd3c8472150d516dad241a83ad9ffe1e9c15253"
        ),
        "competitive/window_bag.py": (
            "d99ae8e86d0d008c016a27e73cbee186141db230f056113a2246e0e26c22aedb"
        ),
        "release_cache.py": "83660e78e3b6ce745bbf89275157474f97290a43bb0c00a4a3da687352f38b6a",
    },
    "removed": [],
}
# Modules that decide pixels, labels, masks, rosters, fold or release admission.
# A delta touching any of them is only admitted by the fresh tensor/cohort replay.
INPUT_LABEL_CODE = frozenset(
    {
        "schema.py",
        "labels.py",
        "release.py",
        "release_cache.py",
        "semantic_supervision.py",
        "series_metadata.py",
        "competitive/data.py",
        "competitive/folds.py",
        "competitive/public_input.py",
        "competitive/supervision.py",
        "competitive/train.py",
    }
)
SUFFIX_LEARNED = {"suffix_r32": (".lora_A", ".lora_B"), "dense": (".dense_delta",)}
FROZEN_PREFIX_LAYERS = 12
GIB = 2**30
SOURCE_CAP_BYTES = 8 * 2**20
CPU_RECEIPT_CAP_BYTES = 2**20
CONTROL_BYTES = 16 * 2**20
RESERVE_BYTES = 20 * GIB
RECOVERY_OUTPUT_CAP_BYTES = 12 * GIB
FUNDING_KEYS = {
    "funded_by",
    "original_input_receipt_sha256",
    "merged_cache_receipt_sha256",
    "new_CPU_source_and_receipt_bytes",
    "new_GPU_or_checkpoint_bytes",
    "new_shared_bytes_files",
    "prefix_cache_bytes",
    "existing_liabilities_bytes",
    "reserve_bytes",
    "required_free_bytes",
}


def validate_funding(funding: Mapping) -> dict:
    """CPU control bytes only; prior obligations and the 20 GiB floor stay retained."""
    if not isinstance(funding, Mapping) or set(funding) != FUNDING_KEYS:
        raise ValueError("Funding declaration fields differ")
    sizes = (
        "new_CPU_source_and_receipt_bytes",
        "new_GPU_or_checkpoint_bytes",
        "new_shared_bytes_files",
        "prefix_cache_bytes",
        "existing_liabilities_bytes",
        "reserve_bytes",
        "required_free_bytes",
    )
    if any(type(funding[key]) is not int or funding[key] < 0 for key in sizes):
        raise ValueError("Funding sizes must be non-negative integers")
    if (
        funding["funded_by"] != "root"
        or funding["original_input_receipt_sha256"] != ORIGINAL_RECEIPT_SHA256
        or funding["merged_cache_receipt_sha256"] != MERGED_RECEIPT_SHA256
        or funding["new_CPU_source_and_receipt_bytes"] != CONTROL_BYTES
        or funding["new_GPU_or_checkpoint_bytes"] != 0
        or funding["new_shared_bytes_files"] != 0
        or funding["prefix_cache_bytes"] != 0
        or funding["existing_liabilities_bytes"] <= 0
        or funding["reserve_bytes"] < RESERVE_BYTES
        or funding["required_free_bytes"]
        != CONTROL_BYTES + funding["existing_liabilities_bytes"] + funding["reserve_bytes"]
    ):
        raise ValueError("CPU control funding, retained liabilities or 20 GiB reserve differ")
    return dict(funding)


def check_storage(path: Path, funding: Mapping) -> int:
    free = shutil.disk_usage(path).free
    if free < funding["required_free_bytes"]:
        raise ValueError("Fresh node-local free space does not cover the funded envelope")
    return free


def verify_source(code: Path, manifest: Mapping) -> dict[str, str]:
    """Exact staged files; every rsna module bound; no links, escapes or extras."""
    files = manifest.get("files") if isinstance(manifest, Mapping) else None
    if (
        set(manifest) != {"source_git", "files"}
        or not isinstance(manifest["source_git"], str)
        or len(manifest["source_git"]) != 40
        or not isinstance(files, Mapping)
        or not set(REQUIRED_SOURCE) <= set(files)
    ):
        raise ValueError("Source manifest lacks the exact admission/wrapper/recipe binding")
    package = {str(p.relative_to(code)) for p in (code / "rsna").rglob("*.py") if p.is_file()}
    if package != {name for name in files if name.startswith("rsna/")}:
        raise ValueError("Source manifest must bind exactly every canonical rsna module")
    total = 0
    for name, value in files.items():
        path = code / name
        if (
            not isinstance(name, str)
            or name.startswith("/")
            or ".." in Path(name).parts
            or path.is_symlink()
            or not path.is_file()
            or not gate._is_sha(value)
        ):
            raise ValueError("Unsafe or missing staged source file")
        total += path.stat().st_size
        if gate.sha(path) != value:
            raise ValueError("Staged source differs from its manifest")
    if total > SOURCE_CAP_BYTES:
        raise ValueError("Staged source exceeds its explicit cap")
    for arm, name in recovery.ARMS.items():
        if files[f"{recovery.RECIPE}/{name}"] != RECIPE_SHA256[arm]:
            raise ValueError("Predeclared dense-suffix recipe changed")
    if files[ORIGINAL_RECEIPT] != ORIGINAL_RECEIPT_SHA256:
        raise ValueError("Original public-input CPU receipt changed")
    return {n.removeprefix("rsna/"): h for n, h in files.items() if n.startswith("rsna/")}


def code_delta(original: Mapping[str, str], current: Mapping[str, str]) -> dict:
    """The exact rsna/ difference from the code that produced the original receipt."""
    delta = {
        "changed": {
            k: current[k] for k in sorted(current) if k in original and original[k] != current[k]
        },
        "added": {k: current[k] for k in sorted(current) if k not in original},
        "removed": sorted(k for k in original if k not in current),
    }
    if delta != CODE_DELTA:
        raise ValueError("Staged code differs from the original receipt beyond the reviewed delta")
    touched = sorted(INPUT_LABEL_CODE & (set(delta["changed"]) | set(delta["added"])))
    return {
        "code_delta_sha256": gate.canonical(delta),
        "changed_files": len(delta["changed"]),
        "added_files": len(delta["added"]),
        "removed_files": len(delta["removed"]),
        "input_label_code_changed": touched,
        "input_label_code_unchanged_sha256": gate.canonical(
            {k: current[k] for k in sorted(INPUT_LABEL_CODE & set(current)) if k not in touched}
        ),
    }


def validate_original(original: Mapping) -> dict:
    """Reuse the original receipt; it alone cannot admit changed loader semantics."""
    inputs = original.get("trainer_inputs", {})
    if (
        original.get("status") != gate.STATUS
        or original.get("CPU_job") != ORIGINAL_CPU_JOB
        or (original.get("full_fitting_studies"), original.get("full_outer_studies"))
        != (gate.FITTING_STUDIES, gate.OUTER_STUDIES)
        or original.get("public_cohort", {}).get("members")
        != gate.FITTING_STUDIES + gate.OUTER_STUDIES
        or original.get("caches", {}).get("public") != gate.PUBLIC_CACHE
        or original.get("merged_cache_receipt_sha256") != MERGED_RECEIPT_SHA256
        or inputs.get("overlay") != str(ORIGINAL_OVERLAY)
        or original.get("supervision", {}).get("overlay_sha256") != gate.OVERLAY_SHA256
        or original.get("GPU_or_optimizer_or_clinical_inference_executed") is not False
        or original.get("protected_labels_used_for_recipe_or_metrics") is not False
        or original.get("original_large_base_proof_and_stats_reused") is not True
    ):
        raise ValueError("Original public-input CPU receipt binding differs")
    artifact = Path(inputs["artifact-root"])
    expected = {
        "studies": artifact / "initiative/releases/report-supervision-v1/studies.parquet",
        "folds": artifact / "initiative/experiments/competitive-v1/folds",
        "series": artifact
        / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv",
    }
    if any(inputs.get(key) != str(path) for key, path in expected.items()):
        raise ValueError("Original trainer inputs differ from the GPU wrapper's derivation")
    return dict(inputs)


def load_arms(code: Path, original_public: Mapping) -> dict[str, dict]:
    """Both arms equal the admitted public recipe except suffix policy and encoder LR."""
    configs = recovery.load_recipe(code / recovery.RECIPE)
    base = json.loads(json.dumps(original_public))
    base["encoder"].pop("adapter")
    base.pop("encoder_lr")
    for arm, values in configs.items():
        stripped = json.loads(json.dumps(values))
        policy = stripped["encoder"].pop("suffix_adapter")
        stripped.pop("encoder_lr")
        if (
            stripped != base
            or policy.get("frozen_prefix_layers") != FROZEN_PREFIX_LAYERS
            or values["epochs"] != 10
            or values.get("diagnostic_study_limit") is not None
        ):
            raise ValueError("Arm differs from the admitted public recipe or fixed budget")
    return configs


def encoder_spec(configs: Mapping[str, dict], spec: Mapping) -> dict:
    """The exact mounted spec the wrapper will receive, checked for both arms."""
    for arm in ARMS:
        recovery.engineering_config(configs[arm], spec)
        encoder = {k: v for k, v in configs[arm]["encoder"].items() if k != "suffix_adapter"}
        if dict(spec) != encoder:
            raise ValueError("Encoder spec differs from the pinned recipe paths")
    return {"encoder_spec_sha256": gate.canonical(dict(spec))}


def check_replay(proofs: Mapping, uid_hashes: Mapping, original: Mapping) -> None:
    for role, key in (("fitting", "training_16_sha256"), ("development", "outer_16_sha256")):
        if uid_hashes[role] != original[key]:
            raise ValueError("Engineering 16-study membership differs from the original receipt")
        if proofs[role] != original["tensor_sha256"][f"public/{role}"]:
            raise ValueError("Changed loader code no longer reproduces the admitted tensors")


def arm_state(
    model, provenance: Mapping, arm: str, expected: Mapping[str, Mapping] = recovery.PINNED
) -> dict:
    """Sanitized learned-graph/head/RNG/frozen-official summary of one fresh arm."""
    from rsna.competitive import suffix_adapters as sa
    from rsna.competitive.delta_checkpoint import FrozenBaseDelta

    backbone = model.encoder.backbone
    sa.check_trainable_partition(backbone)
    policy = provenance["suffix_adapter"]
    learned = {n: p for n, p in model.named_parameters() if p.requires_grad}
    encoder = {n: p for n, p in learned.items() if n.startswith("encoder.")}
    head = {n: p for n, p in learned.items() if not n.startswith("encoder.")}
    layers = {int(n.split(".layer.", 1)[1].split(".", 1)[0]) for n in encoder}
    if (
        policy["method"] != expected[arm]["method"]
        or policy["trainable_parameters"] != expected[arm]["trainable_parameters"]
        or sum(p.numel() for p in encoder.values()) != expected[arm]["trainable_parameters"]
        or not all(n.endswith(SUFFIX_LEARNED[arm]) for n in encoder)
        or min(layers, default=-1) != FROZEN_PREFIX_LAYERS
        or not head
    ):
        raise ValueError("Encoder trainable graph differs from the pinned suffix arm")
    residual = [p for n, p in encoder.items() if not n.endswith(".lora_A")]
    if any(bool(p.detach().any()) for p in residual):
        raise ValueError("Suffix residual is not zero-initialized")
    if any(
        p.requires_grad
        for n, p in model.named_parameters()
        if n.startswith("encoder.") and n not in encoder
    ):
        raise ValueError("An official encoder parameter requires gradients")
    records = {}
    for name, parameter in head.items():
        array = parameter.detach().contiguous().numpy()
        records[name] = [
            list(array.shape),
            str(array.dtype),
            hashlib.sha256(array.tobytes()).hexdigest(),
        ]
    frozen = FrozenBaseDelta(model, dict(provenance)).metadata()["frozen_tensors"]
    return {
        "encoder_learned_parameters": sum(p.numel() for p in encoder.values()),
        "encoder_learned_tensors": len(encoder),
        "head_parameters": sum(p.numel() for p in head.values()),
        "head_values_sha256": gate.canonical(records),
        "post_factory_RNG_sha256": gate.rng_digest(),
        "frozen_base_sha256": gate.canonical(frozen),
        "official_state_sha256": sa.official_state_fingerprint(backbone),
    }


def check_matched(states: Mapping[str, dict], provenance: Mapping[str, dict], original) -> None:
    """Identical official base/head/RNG across arms; only the suffix policy differs."""
    common = ("head_parameters", "head_values_sha256", "post_factory_RNG_sha256")
    official = ("frozen_base_sha256", "official_state_sha256")
    if any(states["suffix_r32"][key] != states["dense"][key] for key in common + official):
        raise ValueError("Fresh head/factory RNG or frozen official base differ between arms")
    if (
        states["dense"]["frozen_base_sha256"]
        != original["initial_learned_state"]["public"]["frozen_base_sha256"]
    ):
        raise ValueError("Frozen official tensors differ from the original admitted base")
    strip = {
        arm: {k: v for k, v in provenance[arm].items() if k != "suffix_adapter"} for arm in ARMS
    }
    admitted = {k: v for k, v in original["encoder_provenance"]["public"].items() if k != "adapter"}
    if strip["suffix_r32"] != strip["dense"] or strip["dense"] != admitted:
        raise ValueError("Official encoder provenance differs from the original admission")


def future_obligations(configs: Mapping[str, dict]) -> dict:
    """Explicit later GPU obligations; nothing here is funded or reserved."""
    production = {}
    for arm, values in configs.items():
        learned = recovery.PINNED[arm]["trainable_parameters"]
        snapshots = values["epochs"] // values["submission_every_epochs"]
        # Weights + two AdamW moments per learned entry, per snapshot and the resume file.
        production[arm] = (snapshots + 1) * 3 * 4 * learned + GIB
    return {
        "GPU_recovery": {
            "wrapper": WRAPPER,
            "phases": ["full", "stop1", "resume"],
            "arms_sequential_on_one_assigned_GPU": True,
            "output_cap_bytes": RECOVERY_OUTPUT_CAP_BYTES,
            "reserve_bytes": RESERVE_BYTES,
            "funded": False,
        },
        "production_fixed10": {
            "epochs": {arm: configs[arm]["epochs"] for arm in ARMS},
            "effective_batch": {
                arm: configs[arm]["batch_size"] * configs[arm]["accumulation_steps"] for arm in ARMS
            },
            "checkpoint_upper_bound_bytes": production,
            "requires_accepted_GPU_recovery_first": True,
            "funded": False,
        },
        "streams_acquired_triplets_online": True,
        "prefix_cache_bytes": 0,
        "prefix_cache_materialized": False,
    }


def assert_private(value, private: Iterable[str]) -> None:
    """Refuse any receipt key or string value equal to a case-level identifier."""
    hidden = {str(item) for item in private if str(item)}
    stack = [value]
    while stack:
        item = stack.pop()
        if isinstance(item, Mapping):
            if any(str(key) in hidden for key in item):
                raise ValueError("Receipt would expose a case-level field")
            stack.extend(item.values())
        elif isinstance(item, (list, tuple)):
            stack.extend(item)
        elif isinstance(item, str) and item in hidden:
            raise ValueError("Receipt would expose a case-level field")


def write_once(path: Path, value: Mapping, private: Iterable[str]) -> None:
    assert_private(value, private)
    payload = json.dumps(value, sort_keys=True, indent=2) + "\n"
    if len(payload.encode()) > CPU_RECEIPT_CAP_BYTES:
        raise ValueError("CPU receipt exceeds its explicit cap")
    with path.open("x") as stream:
        stream.write(payload)


def run(root: Path) -> dict:
    from rsna.competitive import backbones
    from rsna.competitive.folds import load_folds
    from rsna.competitive.model import CompetitiveStudyNet
    from rsna.competitive.public_input import PublicTripletStudyDataset, verify_public_cohort
    from rsna.competitive.supervision import apply_fitting_supervision
    from rsna.competitive.train import _uid_hash, select_development, verified_job_release

    gate.check_placement()
    if torch.cuda.is_available():
        raise ValueError("CPU admission must not see a GPU")
    torch.set_num_threads(1)
    code = root / "code"
    manifest_path = root / "source.json"
    manifest = json.loads(manifest_path.read_text())
    code_sha256 = verify_source(code, manifest)
    funding_path = root / "funding.json"
    funding = validate_funding(json.loads(funding_path.read_text()))
    observed_free = check_storage(root, funding)
    recovery_manifest_path = root / "recovery-source-manifest.json"
    if json.loads(recovery_manifest_path.read_text()) != manifest["files"]:
        raise ValueError("GPU wrapper source manifest differs from the admitted source")
    if recovery.verify_source(code, manifest["files"]) != code_sha256:
        raise ValueError("GPU wrapper would bind different numerical source")
    original = json.loads((code / ORIGINAL_RECEIPT).read_text())
    inputs = validate_original(original)
    delta = code_delta(original["code_sha256"], code_sha256)
    configs = load_arms(code, original["configs"]["public"])
    spec_path = root / "encoder-spec.json"
    spec = encoder_spec(configs, json.loads(spec_path.read_text()))

    config = configs["dense"]
    overlay = Path(inputs["overlay"])
    frame, binding = verified_job_release(
        Path(inputs["studies"]), Path(inputs["artifact-root"]), 0, 1
    )
    folds_path = Path(inputs["folds"])
    folds = load_folds(folds_path, frame, binding)
    fitting, outer = select_development(frame, folds, config["fold"], config["fit_scope"])
    if (len(fitting), len(outer)) != (gate.FITTING_STUDIES, gate.OUTER_STUDIES) or set(
        fitting.leakage_group
    ) & set(outer.leakage_group):
        raise ValueError("Complete fitting/evaluation population or leakage boundary changed")
    fitting, supervision = apply_fitting_supervision(
        fitting,
        outer,
        overlay,
        config["fitting_supervision"],
        release_sha256=binding["manifest_sha256"],
        folds_sha256=gate.sha(folds_path / "fold.csv"),
        fold=config["fold"],
    )
    if supervision != original["supervision"]:
        raise ValueError("Fitting labels/masks/fold supervision differ from the original receipt")
    cache = Path(gate.PUBLIC_CACHE)
    policy = config["public_input"]
    cohort = verify_public_cohort(cache, policy, fitting, outer, binding["source_revision"])
    if cohort != original["public_cohort"]:
        raise ValueError("Complete public cohort membership differs from the original receipt")
    receipt_path = Path(gate.MERGED_RECEIPT)
    if gate.sha(receipt_path) != MERGED_RECEIPT_SHA256:
        raise ValueError("Root complete-cache verification receipt changed")
    gate.validate_merged_receipt(json.loads(receipt_path.read_text()), cohort, cache)
    series_bytes = Path(inputs["series"]).read_bytes()
    if hashlib.sha256(series_bytes).hexdigest() != binding["series_sha256"]:
        raise ValueError("Series sampling table differs from the accepted release")
    import pandas as pd

    series = pd.read_csv(io.BytesIO(series_bytes))
    limit = gate.ENGINEERING_STUDIES
    parts = {"fitting": fitting.head(limit), "development": outer.head(limit)}
    common = {
        name: config[name]
        for name in ("slices", "side", "max_series", "slice_sampling", "series_sampling")
    }
    proofs: dict[str, dict] = {}
    windows: list[torch.Tensor] = []
    for role, part in parts.items():
        soft: dict = {"soft_targets": True} if role == "fitting" else {}
        dataset = PublicTripletStudyDataset(
            part,
            series,
            cache,
            public_input=policy,
            source_revision=binding["source_revision"],
            **common,
            **soft,
        )
        proofs[role], first = gate.tensor_proofs(part, dataset, "public")
        windows.append(first)
        del dataset
        gc.collect()
    check_replay(proofs, {role: _uid_hash(part) for role, part in parts.items()}, original)

    proof_path = Path(gate.BASE_PROOF)
    if gate.sha(proof_path) != gate.BASE_PROOF_SHA256:
        raise ValueError("Original whole-byte scientific asset proof changed")
    asset_proof = json.loads(proof_path.read_text())
    checked = backbones._checked_file
    stats: dict[str, list[int]] = {}

    def original_file(values, key):
        if key == "weights":
            path = Path(values["weights_path"])
            record = asset_proof["scientific_assets"][str(path)]
            stats[str(path)] = gate.signature(path)
            if (
                stats[str(path)] != record["signature"]
                or stats[str(path)] != original["official_base_fresh_stats"].get(str(path))
                or values["weights_sha256"] != record["sha256"]
            ):
                raise ValueError("Whole-byte-verified official base stat binding changed")
            return path
        return checked(values, key)

    states, provenance, normalized = {}, {}, {}
    backbones._checked_file = original_file
    try:
        for arm in ARMS:
            model, provenance[arm], _ = gate.initialize(
                configs[arm], "public", backbones.create_encoder, CompetitiveStudyNet
            )
            states[arm] = arm_state(model, provenance[arm], arm)
            normalized[arm] = gate.check_imagenet_once(model.encoder, windows[0])
            del model
            gc.collect()
    finally:
        backbones._checked_file = checked
    check_matched(states, provenance, original)
    if set(normalized.values()) != {original["first_window_imagenet_sha256"]["public"]}:
        raise ValueError("Public 336 RGB triplets no longer normalize exactly as admitted")

    value = {
        "status": STATUS,
        "CPU_job": os.environ["SLURM_JOB_ID"],
        "source_git": manifest["source_git"],
        "source_manifest_sha256": gate.sha(manifest_path),
        "recovery_source_manifest_sha256": gate.sha(recovery_manifest_path),
        "wrapper_sha256": manifest["files"][WRAPPER],
        "admission_sha256": manifest["files"][SELF],
        "recipe_sha256": dict(RECIPE_SHA256),
        "code_sha256": code_sha256,
        "code_delta_from_original": delta,
        "encoder_spec": spec | {"path_sha256": gate.sha(spec_path)},
        "funding_sha256": gate.sha(funding_path),
        "fresh_storage": {
            "observed_free_bytes": observed_free,
            "required_free_bytes": funding["required_free_bytes"],
            "existing_liabilities_bytes": funding["existing_liabilities_bytes"],
            "reserve_bytes": funding["reserve_bytes"],
        },
        "original_CPU_receipt_sha256": ORIGINAL_RECEIPT_SHA256,
        "merged_cache_receipt_sha256": MERGED_RECEIPT_SHA256,
        "whole_byte_base_proof_sha256": gate.BASE_PROOF_SHA256,
        "official_base_fresh_stats_equal_original": True,
        "release_sha256": binding["manifest_sha256"],
        "folds_sha256": supervision["folds_sha256"],
        "overlay_sha256": supervision["overlay_sha256"],
        "public_cohort": cohort,
        "supervision_sha256": gate.canonical(supervision),
        "full_fitting_studies": len(fitting),
        "full_outer_studies": len(outer),
        "engineering_studies": {role: len(part) for role, part in parts.items()},
        "training_16_sha256": original["training_16_sha256"],
        "outer_16_sha256": original["outer_16_sha256"],
        "public_tensor_sha256": proofs,
        "first_window_imagenet_sha256": original["first_window_imagenet_sha256"]["public"],
        "arms": {
            arm: states[arm]
            | {
                "method": recovery.PINNED[arm]["method"],
                "encoder_lr": configs[arm]["encoder_lr"],
                "suffix_provenance_sha256": gate.canonical(provenance[arm]["suffix_adapter"]),
            }
            for arm in ARMS
        },
        "future_obligations": future_obligations(configs),
        "GPU_wrapper_arguments": {
            "source": str(code),
            "source-manifest": str(recovery_manifest_path),
            "encoder-spec": str(spec_path),
            "artifact-root": inputs["artifact-root"],
            "cache": gate.PUBLIC_CACHE,
            "overlay": inputs["overlay"],
            "max-output-bytes": RECOVERY_OUTPUT_CAP_BYTES,
        },
        "GPU_or_optimizer_or_clinical_inference_executed": False,
        "merge_render_download_or_transfer_executed": False,
        "prefix_cache_materialized": False,
        "protected_labels_used_for_recipe_or_metrics": False,
        "new_labels_created": False,
        "accuracy_claimed": False,
        "GPU_acceptance_inferred": False,
        "diagnostic_only": True,
    }
    private = [str(uid) for frame_part in (fitting, outer) for uid in frame_part.study_uid]
    private += [str(group) for frame_part in (fitting, outer) for group in frame_part.leakage_group]
    write_once(root / "CPU-accepted.json", value, private)
    print(json.dumps({"status": STATUS, "CPU_job": value["CPU_job"]}), flush=True)
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
