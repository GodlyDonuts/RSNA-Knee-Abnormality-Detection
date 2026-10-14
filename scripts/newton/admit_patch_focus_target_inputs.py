"""Separately versioned evc40 target CPU admission for the patch-focus arms (issue #84).

The original owning-node route (``admit_patch_focus_inputs.py``) is unchanged and still
refuses evc40. This route is one standalone one-CPU, zero-GPU allocation on evc40 that
reuses, never regenerates or re-transfers:

* the overlay root restored ONCE by CPU916757 (exact original SHA/bytes/0400/one link,
  unchanged since its private restoration proof) and root's same-16-MiB funding proof;
* the whole-byte verified private relay of the complete public triplet cache (exact
  original index/manifest/cohort, merged-receipt d2c0ad74 retained in relay lineage,
  every member single-link/0400/size and unchanged since the relay receipt; no rehash);
* the original public-input CPU receipt, release/fold byte bindings (bounded: exact
  studies/release-receipt/fold bytes instead of the timed-out full archive observer)
  and the whole-byte official base proof (stat binding, or one byte rehash when only the
  shared mount's device number differs on this node).

It then runs the unchanged patch-focus evidence: complete 2247/1156 fold/supervision/
cohort boundary, exact 16 fitting + 16 outer public tensor replays, bitwise matched
initial models/RNG and the backward-only routing gradients. The exact actual runtime
must equal root's declaration. Only one sanitized hashes/counts receipt is created
exclusively inside the funded root. No optimizer step, GPU, transfer or overlay write.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import platform
import shutil
import stat
import sys
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path, PurePosixPath

import torch

from scripts.newton import admit_dense_suffix_inputs as dense
from scripts.newton import admit_patch_focus_inputs as admit
from scripts.newton import probe_public_input_recipe_inputs as gate
from scripts.newton import triplet_cache_relay as triplet

VERSION = "source_patch_focus_target_evc40_CPU_admission_v1"
STATUS = "source_patch_focus_target_evc40_actual_inputs_and_matched_initialization_CPU_accepted"
DECLARATION_STATUS = "root_declared_patch_focus_target_evc40_CPU_admission"
NODE = triplet.TARGET_NODE
SELF = "scripts/newton/admit_patch_focus_target_inputs.py"
ROOT = Path("/tmp/rsna-patch-focus-target-admission-20261014-v2")
RUNTIME = Path("/tmp/rsna-environment-907632-7sr9a7i6/environment")
RUNTIME_PACKAGES = ("torch", "numpy", "pandas", "transformers", "safetensors")
FUNDING, RESTORED, DECLARATION = "funding-overlay.json", "overlay-restored.json", "target.json"
OUTPUT = "target-CPU-accepted.json"
OVERLAY = dense.ORIGINAL_OVERLAY
OVERLAY_SHA256, OVERLAY_BYTES, OVERLAY_JOB = gate.OVERLAY_SHA256, 1057379, "916757"
RESTORATION_AUDIT = (
    "initiative/audits/patch-focus-target-overlay-restoration-20261014-v1/verified.json"
)
RELAY_AUDIT = (
    "initiative/audits/dense-triplet-cache-relay-20261014-v1/actual-transfer-verified.json"
)
RELAY_RECEIPT = triplet.CONTROL / triplet.VERIFIED
RELAY_FUNDING_SHA256 = "9756e953870cef737fe0baa6c403a38121186495a3f9d984585533cb7764f934"
MANIFEST_SHA256 = "09ea7bc6aaf7c52bda2bb94eff51db37aac8b066bcc795c63d137bfe837699dc"
CACHE_FILES, CACHE_BYTES = triplet.STUDIES + 1, 27668082285
# Parent/pipeline, original-source and once-only restoration jobs never admit this route.
FORBIDDEN_JOBS = frozenset({"908454", dense.ORIGINAL_CPU_JOB, gate.ORIGINAL_CPU_JOB, OVERLAY_JOB})
CAP_BYTES = dense.CONTROL_BYTES  # The SAME 16 MiB root already funded (overlay charged).
LIABILITIES_BYTES = 338320442039
RESERVE_BYTES = dense.RESERVE_BYTES
REQUIRED_SOURCE = (
    *admit.REQUIRED_SOURCE,
    SELF,
    "scripts/newton/triplet_cache_relay.py",
    "scripts/newton/native_cache_relay.py",
    RESTORATION_AUDIT,
    RELAY_AUDIT,
)
DECLARATION_KEYS = {
    "status",
    "version",
    "funded_by",
    "issue",
    "node",
    "experiment",
    "owning_root",
    "funding_overlay_sha256",
    "overlay_restored_sha256",
    "relay_receipt_sha256",
    "original_input_receipt_sha256",
    "merged_cache_receipt_sha256",
    "base_proof_sha256",
    "runtime",
    "cap_bytes",
    "existing_liabilities_bytes",
    "reserve_bytes",
    "new_GPU_or_checkpoint_bytes",
    "new_shared_bytes_files",
    "GPU_funded",
}


def check_placement(nodename: str, env: Mapping[str, str]) -> str:
    """A fresh standalone one-CPU zero-GPU allocation on evc40; never a parent/old job."""
    job = env.get("SLURM_JOB_ID", "")
    if (
        nodename != NODE
        or not job.isdigit()
        or job in FORBIDDEN_JOBS
        or env.get("SLURM_CPUS_PER_TASK") != "1"
        or env.get("SLURM_JOB_NODELIST", NODE) != NODE
        or any(env.get(k) for k in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS"))
        or env.get("SLURM_GPUS_ON_NODE", "0") != "0"
        or env.get("CUDA_VISIBLE_DEVICES") not in (None, "", "NoDevFiles")
    ):
        raise ValueError("Fresh standalone evc40 one-CPU zero-GPU allocation required")
    return job


def runtime_identity() -> dict:
    return {
        "prefix": sys.prefix,
        "python": platform.python_version(),
        **{name: _version(name) for name in RUNTIME_PACKAGES},
    }


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def check_runtime(actual: Mapping, declared: Mapping) -> dict:
    if dict(actual) != dict(declared) or actual.get("prefix") != str(RUNTIME):
        raise ValueError("Actual runtime differs from the declared prepared environment")
    return dict(actual)


def _no_links(path: Path) -> None:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("Escaped or relative control path")
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Linked control path or ancestor")


def _regular(path: Path) -> os.stat_result:
    _no_links(path)
    s = os.lstat(path)
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
        raise ValueError("Linked or non-regular control file")
    return s


def load_exact(path: Path, digest: str) -> tuple[dict, os.stat_result]:
    s = _regular(path)
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != digest:
        raise ValueError("Control file differs from its pinned SHA")
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("Control file is not an object")
    return value, s


def validate_declaration(value: Mapping) -> dict:
    """Root's effective target declaration: same 16 MiB cap, no GPU, liabilities retained."""
    if not isinstance(value, Mapping) or set(value) != DECLARATION_KEYS:
        raise ValueError("Target declaration fields differ")
    sizes = ("cap_bytes", "existing_liabilities_bytes", "reserve_bytes")
    if any(type(value[k]) is not int for k in (*sizes, "new_GPU_or_checkpoint_bytes")):
        raise ValueError("Declaration sizes must be integers")
    if any(
        not gate._is_sha(value[k])
        for k in ("funding_overlay_sha256", "overlay_restored_sha256", "relay_receipt_sha256")
    ):
        raise ValueError("Declaration lineage hashes are malformed")
    if (
        value["status"] != DECLARATION_STATUS
        or value["version"] != VERSION
        or value["funded_by"] != "root"
        or value["issue"] != 84
        or value["node"] != NODE
        or value["experiment"] != admit.EXPERIMENT
        or value["owning_root"] != str(ROOT)
        or value["original_input_receipt_sha256"] != dense.ORIGINAL_RECEIPT_SHA256
        or value["merged_cache_receipt_sha256"] != dense.MERGED_RECEIPT_SHA256
        or value["base_proof_sha256"] != gate.BASE_PROOF_SHA256
        or not isinstance(value["runtime"], Mapping)
        or value["cap_bytes"] != CAP_BYTES
        or value["existing_liabilities_bytes"] != LIABILITIES_BYTES
        or value["reserve_bytes"] < RESERVE_BYTES
        or value["new_GPU_or_checkpoint_bytes"] != 0
        or value["new_shared_bytes_files"] != 0
        or value["GPU_funded"] is not False
    ):
        raise ValueError("Target declaration lineage, cap, reserve or GPU funding differs")
    return dict(value)


def validate_restoration(funding: Mapping, restored: Mapping, audit: Mapping) -> None:
    """Private proofs must agree with the reviewed sanitized restoration receipt."""
    if any(funding.get(k) != v for k, v in audit["funding"].items()):
        raise ValueError("Overlay funding proof differs from the reviewed restoration")
    if (
        restored.get("overlay_sha256") != OVERLAY_SHA256
        or restored.get("bytes") != OVERLAY_BYTES
        or any(restored.get(k) != v for k, v in audit.items() if k != "funding")
    ):
        raise ValueError("Overlay restoration proof differs from the reviewed restoration")
    if (
        audit["node"] != NODE
        or audit["CPU_job"] != OVERLAY_JOB
        or audit["overlay_sha256"] != OVERLAY_SHA256
        or audit["bytes"] != OVERLAY_BYTES
        or audit["funding"]["new_control_and_overlay_cap_bytes"] != CAP_BYTES
        or audit["funding"]["private_output"] != str(OVERLAY)
        or audit["funding"]["GPU_checkpoint_funding"] is not False
    ):
        raise ValueError("Reviewed restoration receipt lineage differs")


def check_overlay(path: Path, proof: os.stat_result) -> dict:
    """Inspect the once-restored overlay; it is never written, copied or regenerated."""
    if path != OVERLAY:
        raise ValueError("Overlay must be the original restored path")
    s = _regular(path)
    if (
        stat.S_IMODE(s.st_mode) != 0o400
        or s.st_size != OVERLAY_BYTES
        or s.st_ctime_ns > proof.st_mtime_ns
    ):
        raise ValueError("Restored overlay mode/size or inode changed after its proof")
    before = gate.signature(path)
    if gate.sha(path) != OVERLAY_SHA256 or gate.signature(path) != before:
        raise ValueError("Restored overlay differs from the exact original bytes")
    return {"sha256": OVERLAY_SHA256, "bytes": OVERLAY_BYTES, "mode": "0400", "links": 1}


def validate_relay(receipt: Mapping, audit: Mapping) -> None:
    if (
        dict(receipt) != dict(audit)
        or receipt.get("status") != "original_complete_public_triplet_cache_private_relay_verified"
        or receipt.get("relay_version") != triplet.VERSION
        or receipt.get("node") != NODE
        or receipt.get("cache") != gate.PUBLIC_CACHE
        or receipt.get("index_sha256") != triplet.INDEX_SHA256
        or receipt.get("cohort_sha256") != triplet.COHORT_SHA256
        or receipt.get("manifest_sha256") != MANIFEST_SHA256
        or receipt.get("merged_receipt_sha256") != dense.MERGED_RECEIPT_SHA256
        or receipt.get("CPU_receipt_sha256") != triplet.CPU_RECEIPT_SHA256
        or receipt.get("funding_sha256") != RELAY_FUNDING_SHA256
        or (receipt.get("studies"), receipt.get("files")) != (triplet.STUDIES, CACHE_FILES)
        or receipt.get("bytes") != CACHE_BYTES
        or receipt.get("every_original_byte_size_SHA_count_EOF_layout_verified") is not True
        or receipt.get("GPU_used") is not False
    ):
        raise ValueError("Verified target relay receipt or its lineage differs")


def check_cache(cache: Path, manifest: Mapping, receipt: os.stat_result) -> dict:
    """Exact relayed layout, unchanged since the relay proof; no bulk byte rehash."""
    _no_links(cache)
    if any(
        not isinstance(n, str) or n.startswith("/") or ".." in PurePosixPath(n).parts
        for n in manifest
    ):
        raise ValueError("Escaped cache manifest member")
    layout = sorted(str(p.relative_to(cache)) for p in cache.rglob("*") if not p.is_dir())
    if layout != sorted(manifest):
        raise ValueError("Target cache layout differs from the exact relayed manifest")
    signatures = triplet.source_layout(cache, dict(manifest))
    for name in manifest:
        s = os.lstat(cache / name)
        if (
            stat.S_IMODE(s.st_mode) != 0o400
            or s.st_dev != receipt.st_dev
            or s.st_ctime_ns > receipt.st_mtime_ns
        ):
            raise ValueError("Relayed cache member changed after its verified receipt")
    return {
        "files": len(manifest),
        "bytes": triplet.total_bytes(dict(manifest)),
        "signatures_sha256": gate.canonical(signatures),
    }


def root_usage(root: Path) -> int:
    """All bytes already in the funded root; links or special files are refused."""
    _no_links(root)
    total = 0
    for directory, names, files in os.walk(root):
        for name in (*names, *files):
            path = Path(directory) / name
            s = os.lstat(path)
            if stat.S_ISDIR(s.st_mode):
                continue
            if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
                raise ValueError("Linked or special file inside the funded root")
            total += s.st_size
    return total


def check_budget(root_bytes: int, free_bytes: int, declaration: Mapping) -> dict:
    """Same 16 MiB cap (overlay already charged) plus one receipt; liabilities retained."""
    used = OVERLAY_BYTES + root_bytes
    remaining = CAP_BYTES - used
    required = declaration["existing_liabilities_bytes"] + declaration["reserve_bytes"] + remaining
    if remaining < dense.CPU_RECEIPT_CAP_BYTES or free_bytes < required:
        raise ValueError("Same 16 MiB cap or retained liabilities/20 GiB reserve exceeded")
    return {
        "cap_bytes": CAP_BYTES,
        "charged_bytes": used,
        "remaining_bytes": remaining,
        "observed_free_bytes": free_bytes,
        "required_free_bytes": required,
        "existing_liabilities_bytes": declaration["existing_liabilities_bytes"],
        "reserve_bytes": declaration["reserve_bytes"],
    }


def bounded_release(inputs: Mapping, original: Mapping):
    """Exact shared release/receipt/fold bytes of the original receipt; no archive replay."""
    import pandas as pd

    from rsna.release import validate_checkpoint_binding

    studies = Path(inputs["studies"])
    binding = json.loads((Path(inputs["folds"]) / "folds.json").read_text())["release_binding"]
    validate_checkpoint_binding(binding)
    payload = studies.read_bytes()
    if (
        hashlib.sha256(payload).hexdigest() != binding["manifest_sha256"]
        or binding["manifest_sha256"] != original["supervision"]["release_sha256"]
        or gate.sha(studies.parent / "verified_release.json") != binding["release_receipt_sha256"]
    ):
        raise ValueError("Shared release bytes differ from the original accepted binding")
    return pd.read_parquet(io.BytesIO(payload)), binding


def base_binding(path: Path, record: Mapping, original_stats: list[int]) -> str:
    """Original stat binding; if only this node's mount device differs, rehash once."""
    current = gate.signature(path)
    if record["signature"] != original_stats:
        raise ValueError("Whole-byte-verified official base stat binding changed")
    if current == original_stats:
        return "original_stat_binding"
    if current[1:] == original_stats[1:]:
        if gate.sha(path) != record["sha256"] or gate.signature(path) != current:
            raise ValueError("Official base bytes differ from the whole-byte proof")
        return "target_mount_whole_byte_rehash"
    raise ValueError("Whole-byte-verified official base stat binding changed")


def run(root: Path) -> dict:
    from rsna.competitive import backbones
    from rsna.competitive.data import collate_studies
    from rsna.competitive.folds import load_folds
    from rsna.competitive.patch_focus import classifier_model
    from rsna.competitive.public_input import PublicTripletStudyDataset, verify_public_cohort
    from rsna.competitive.source_slot import training_inputs
    from rsna.competitive.supervision import apply_fitting_supervision
    from rsna.competitive.train import _uid_hash, select_development
    from rsna.schema import TARGETS

    job = check_placement(os.uname().nodename, os.environ)
    if torch.cuda.is_available():
        raise ValueError("Genuine zero-GPU admission required")
    torch.set_num_threads(1)
    if root != ROOT:
        raise ValueError("Admission must run inside the funded owning root")
    _no_links(root)
    output = root / OUTPUT
    if output.exists() or output.is_symlink():
        raise ValueError("Existing target admission destination forbids repetition")
    code = root / "code"
    manifest_path = root / "source.json"
    manifest = json.loads(manifest_path.read_text())
    if not set(REQUIRED_SOURCE) <= set(manifest.get("files", {})):
        raise ValueError("Source manifest lacks the target admission/proof binding")
    code_sha256 = admit.verify_source(code, manifest)
    declaration_path = root / DECLARATION
    declaration = validate_declaration(json.loads(_read_regular(declaration_path)))
    runtime = check_runtime(runtime_identity(), declaration["runtime"])
    funding, _ = load_exact(root / FUNDING, declaration["funding_overlay_sha256"])
    restored, proof_stat = load_exact(root / RESTORED, declaration["overlay_restored_sha256"])
    audit = json.loads((code / RESTORATION_AUDIT).read_text())
    validate_restoration(funding, restored, audit)
    overlay = check_overlay(OVERLAY, proof_stat)
    budget = check_budget(root_usage(root), shutil.disk_usage(root).free, declaration)

    relay, relay_stat = load_exact(RELAY_RECEIPT, declaration["relay_receipt_sha256"])
    validate_relay(relay, json.loads((code / RELAY_AUDIT).read_text()))
    cache = Path(gate.PUBLIC_CACHE)
    index = cache / triplet.INDEX
    _regular(index)
    cache_manifest = triplet.manifest_from_index(index.read_bytes())
    if triplet.canonical_sha256(cache_manifest) != MANIFEST_SHA256:
        raise ValueError("Relayed index manifest differs")
    cache_proof = check_cache(cache, cache_manifest, relay_stat)
    if (cache_proof["files"], cache_proof["bytes"]) != (CACHE_FILES, CACHE_BYTES):
        raise ValueError("Relayed cache count/bytes differ")
    # The producer's merged-cache-verified.json was never relayed; never forge or require it.
    merged_present = Path(gate.MERGED_RECEIPT).exists() or Path(gate.MERGED_RECEIPT).is_symlink()
    if merged_present:
        raise ValueError("Unrelayed producer merged receipt unexpectedly present on target")

    recovery_manifest_path = root / "recovery-source-manifest.json"
    if json.loads(recovery_manifest_path.read_text()) != manifest["files"]:
        raise ValueError("GPU wrapper source manifest differs from the admitted source")
    if admit.recovery.verify_source(code, manifest["files"]) != code_sha256:
        raise ValueError("GPU wrapper would bind different numerical source")
    original = json.loads((code / dense.ORIGINAL_RECEIPT).read_text())
    inputs = dense.validate_original(original)
    if Path(inputs["overlay"]) != OVERLAY:
        raise ValueError("Original overlay lineage differs")
    delta = admit.code_delta(original["code_sha256"], code_sha256)
    configs = admit.load_arms(code, original["configs"]["public"])
    spec_path = root / "encoder-spec.json"
    spec = admit.encoder_spec(configs, json.loads(spec_path.read_text()))

    config = configs["reference"]
    frame, binding = bounded_release(inputs, original)
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
        OVERLAY,
        config["fitting_supervision"],
        release_sha256=binding["manifest_sha256"],
        folds_sha256=gate.sha(folds_path / "fold.csv"),
        fold=config["fold"],
    )
    if supervision != original["supervision"]:
        raise ValueError("Fitting labels/masks/fold supervision differ from the original receipt")
    policy = config["public_input"]
    cohort = verify_public_cohort(cache, policy, fitting, outer, binding["source_revision"])
    if cohort != original["public_cohort"] or cohort["index_sha256"] != relay["index_sha256"]:
        raise ValueError("Complete public cohort membership differs from the original receipt")
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

    def dataset(part, soft: bool):
        return PublicTripletStudyDataset(
            part,
            series,
            cache,
            public_input=policy,
            source_revision=binding["source_revision"],
            **common,
            soft_targets=soft,
        )

    proofs: dict[str, dict] = {}
    windows: list[torch.Tensor] = []
    for role, part in parts.items():
        data = dataset(part, role == "fitting")
        proofs[role], first = gate.tensor_proofs(part, data, "public")
        windows.append(first)
        del data
        gc.collect()
    dense.check_replay(proofs, {role: _uid_hash(part) for role, part in parts.items()}, original)

    # Real fitting-only MRI study; development rows are never read for gradients.
    position, routing_mask, routing_fixture = admit.routing_probe(fitting)
    item = dataset(fitting.iloc[position : position + 1], True)[0]
    batch = collate_studies([item])
    batch["study_uid"] = list(fitting.study_uid.iloc[position : position + 1])
    seed = config["seed"]
    probe_inputs = training_inputs(
        batch, torch.device("cpu"), config["classifier_strategy"], seed=seed, epoch=0, update=0
    )
    labels, original_mask = batch["labels"].float(), batch["label_mask"].float()
    fitting_mask = torch.tensor(
        fitting.iloc[position : position + 1][[f"{t}__mask" for t in TARGETS]].to_numpy(),
        dtype=torch.float32,
    )
    if not torch.equal(original_mask, fitting_mask):
        raise ValueError("Dataset mask differs from the admitted fitting row")
    probe_study = {
        "fitting_position": position,
        "images_sha256": hashlib.sha256(item["images"].numpy().tobytes()).hexdigest(),
        "labels_and_mask_sha256": hashlib.sha256(
            labels.numpy().tobytes() + original_mask.numpy().tobytes()
        ).hexdigest(),
        "routing_fixture": routing_fixture,
        "routing_mask_sha256": hashlib.sha256(routing_mask.numpy().tobytes()).hexdigest(),
        "acquired_slots": int(batch["series_mask"].sum()),
    }

    proof_path = Path(gate.BASE_PROOF)
    if gate.sha(proof_path) != gate.BASE_PROOF_SHA256:
        raise ValueError("Original whole-byte scientific asset proof changed")
    asset_proof = json.loads(proof_path.read_text())
    checked = backbones._checked_file
    methods: dict[str, str] = {}

    def original_file(values, key):
        if key == "weights":
            path = Path(values["weights_path"])
            record = asset_proof["scientific_assets"][str(path)]
            if values["weights_sha256"] != record["sha256"]:
                raise ValueError("Official base SHA differs from the whole-byte proof")
            stats = original["official_base_fresh_stats"].get(str(path))
            methods[str(path)] = methods.get(str(path)) or base_binding(path, record, stats)
            return path
        return checked(values, key)

    normalized: list[str] = []

    def build(name: str) -> tuple:
        values = configs[name]
        strategy = values["classifier_strategy"]

        def factory(encoder, dim, **kwargs):
            return classifier_model(strategy)(
                encoder,
                dim,
                strategy,
                encode_chunk_size=values["encode_chunk_size"],
                input_channels=3,
            )

        model, provenance, state = gate.initialize(
            values, "public", backbones.create_encoder, factory
        )
        normalized.append(gate.check_imagenet_once(model.encoder, windows[0]))
        return model, provenance, state

    backbones._checked_file = original_file
    try:
        evidence, states = admit.matched_initialization(
            build, probe_inputs, labels, routing_mask, seed, original
        )
    finally:
        backbones._checked_file = checked
    if set(normalized) != {original["first_window_imagenet_sha256"]["public"]}:
        raise ValueError("Public 336 RGB triplets no longer normalize exactly as admitted")
    if gate.sha(OVERLAY) != OVERLAY_SHA256 or output.exists():
        raise ValueError("Overlay changed or destination appeared during admission")

    value = {
        "status": STATUS,
        "version": VERSION,
        "experiment": admit.EXPERIMENT,
        "node": NODE,
        "CPU_job": job,
        "standalone_one_CPU_zero_GPU": True,
        "runtime": runtime,
        "source_git": manifest["source_git"],
        "source_manifest_sha256": gate.sha(manifest_path),
        "recovery_source_manifest_sha256": gate.sha(recovery_manifest_path),
        "admission_sha256": manifest["files"][SELF],
        "original_admission_sha256": manifest["files"][admit.SELF],
        "wrapper_sha256": manifest["files"][admit.WRAPPER],
        "recipe_sha256": dict(admit.RECIPE_SHA256),
        "code_sha256": code_sha256,
        "code_delta_from_original": delta,
        "encoder_spec": spec | {"path_sha256": gate.sha(spec_path)},
        "declaration_sha256": gate.sha(declaration_path),
        "funding_overlay_sha256": declaration["funding_overlay_sha256"],
        "overlay_restored_sha256": declaration["overlay_restored_sha256"],
        "restoration_CPU_job": OVERLAY_JOB,
        "restored_overlay": overlay,
        "same_16MiB_cap": budget,
        "relay_receipt_sha256": declaration["relay_receipt_sha256"],
        "relay_cache": cache_proof,
        "relay_manifest_sha256": MANIFEST_SHA256,
        "original_CPU_receipt_sha256": dense.ORIGINAL_RECEIPT_SHA256,
        "merged_cache_receipt_sha256": dense.MERGED_RECEIPT_SHA256,
        "merged_cache_receipt_admitted_via": "verified_relay_lineage_unrelayed_file_absent",
        "whole_byte_base_proof_sha256": gate.BASE_PROOF_SHA256,
        "official_base_binding": sorted(set(methods.values())),
        "official_encoder_provenance_equal_original": True,
        "release_binding": "bounded_exact_shared_release_receipt_and_fold_bytes",
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
        "gradient_probe_study": probe_study,
        "matched_initialization_and_gradients": evidence,
        "future_obligations": admit.future_obligations(
            configs, {name: states[name]["learned_parameters"] for name in admit.NAMES}
        )
        | {"GPU_funded_or_reserved_by_this_gate": False},
        "optimizer_step_executed": False,
        "GPU_or_clinical_inference_executed": False,
        "overlay_written_or_regenerated": False,
        "merge_render_download_rehash_or_transfer_executed": False,
        "protected_labels_used_for_recipe_or_metrics": False,
        "new_labels_created": False,
        "accuracy_claimed": False,
        "GPU_acceptance_inferred": False,
        "diagnostic_only": True,
    }
    private = [str(uid) for frame_part in (fitting, outer) for uid in frame_part.study_uid]
    private += [str(group) for frame_part in (fitting, outer) for group in frame_part.leakage_group]
    dense.write_once(output, value, private)
    print(json.dumps({"status": STATUS, "CPU_job": job}), flush=True)
    return value


def _read_regular(path: Path) -> bytes:
    _regular(path)
    return path.read_bytes()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    run(parser.parse_args().root)
