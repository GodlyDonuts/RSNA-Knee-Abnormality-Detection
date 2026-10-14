"""Separately versioned evc40 target admission for the matched source patch-focus arms.

PREPARED ONLY; Devin dispatches, transfers and acquires nothing. Root alone stages and runs
two once-only steps inside one NEW standalone one-CPU / zero-GPU evc40 allocation (never the
healthy 908454 parent, never the cancelled dense 899766 state):

``overlay``
    Regenerates, in RAM, the ORIGINAL fold-0 fitting overlay (SHA c8fd2e86..., 1057379
    bytes) from the existing evc40 final overlay (SHA 54b5a652...), the exact shared
    study/fold bytes and the original 2247/1156 UID roster hashes. The only provenance
    change is restoring the original source name ``llm_labels_v4_blend``; the source CSV and
    revision are unchanged. Bytes are written exclusively (0400) only after the exact
    original SHA/size match, inside the separately funded 16 MiB control envelope.

``admit``
    The patch-focus CPU admission on the target. It accepts the actual verified private
    cache relay receipt (target-verified.json, byte-equal to the Git audit copy) plus fresh
    original-publication member signatures (regular, single-link, 0400, unchanged since the
    relay verification, exact membership/sizes, no bulk rehash) instead of the original
    producer's ``merged-cache-verified.json``, which was never transferred and must not
    exist on the target. Everything else is the original admission: original input receipt,
    source/fold/base lineage, exact 16 fitting + 16 outer tensor replays, official base stat
    binding, real initial model/base/RNG equality and the target-routed gradient probe.

The original evc31 admission (``admit_patch_focus_inputs.py``) is unchanged and still
refuses evc40. No GPU, optimizer, download, merge, render or transfer; no MRI, state or
case-level identifier leaves the node; receipts carry aggregate counts/hashes only.
Engineering evidence only, never an accuracy claim.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import os
import stat
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import torch

from scripts.newton import admit_dense_suffix_inputs as dense
from scripts.newton import admit_patch_focus_inputs as admit
from scripts.newton import probe_public_input_recipe_inputs as gate
from scripts.newton import run_patch_focus_recovery as recovery
from scripts.newton import triplet_cache_relay as relay

VERSION = "source_patch_focus_evc40_target_admission_v1"
STATUS = "source_patch_focus_evc40_target_actual_inputs_and_matched_initialization_CPU_accepted"
OVERLAY_STATUS = "original_fitting_overlay_reconstructed_exactly_on_evc40_target"
SELF = "scripts/newton/admit_patch_focus_inputs_evc40.py"
NODE = "evc40"
# Healthy all-data fit and the cancelled dense parent: never reused, never touched.
FORBIDDEN_JOBS = frozenset({"908454", "899766"})
CONTROL_ROOT = Path("/tmp/rsna-patch-focus-evc40-admission-20261014-v1")
OVERLAY_NAME, OVERLAY_RECEIPT, CPU_RECEIPT = (
    "overlay.json",
    "overlay-reconstructed.json",
    "CPU-accepted.json",
)

FINAL_OVERLAY = Path("/tmp/rsna-final-public-source-fit-evc40-20261013-v1/overlay.json")
FINAL_OVERLAY_SHA256 = "54b5a652949120764a596b4d495f11950ff103a909b7efc5fe59da9cd6f2fe85"
ORIGINAL_OVERLAY_SHA256 = gate.OVERLAY_SHA256
ORIGINAL_OVERLAY_BYTES = 1_057_379
FINAL_SOURCE_NAME = "Shio-Koube/knee3"
ORIGINAL_SOURCE = {
    "name": "llm_labels_v4_blend",
    "csv_sha256": "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4",
    "revision": "373f4e1daf36984fa48bf3fc8fd9883fe17cb584",
}
PINS = {
    "release_sha256": "2bf49fd2a8ae0b7c5af7e8c105fc3f2337845a92654d136fe67bc41a1e52fae7",
    "folds_sha256": "fe590e5072a0d82abeb377598aeef0021bbe77fe4529a19b9826a7d40d322050",
    "fitting_uid_sha256": relay.FITTING_UIDS_SHA256,
    "outer_uid_sha256": relay.EVALUATION_UIDS_SHA256,
    "development_uid_sha256": relay.COHORT_SHA256,
    "fitting_studies": gate.FITTING_STUDIES,
    "outer_studies": gate.OUTER_STUDIES,
    "fold": 0,
    "overlay_sha256": ORIGINAL_OVERLAY_SHA256,
    "overlay_bytes": ORIGINAL_OVERLAY_BYTES,
}

RELAY_RECEIPT = relay.CONTROL / relay.VERIFIED
RELAY_AUDIT = (
    "initiative/audits/dense-triplet-cache-relay-20261014-v1/actual-transfer-verified.json"
)
RELAY_RECEIPT_SHA256 = "7104498b4c82a94a4bdd60da055dd301882b4f9b8bfe07ba29ec379075f09071"
RELAY_FILES, RELAY_BYTES = relay.STUDIES + 1, 27_668_082_285
RUNTIME_AUDIT = (
    "initiative/audits/final-public-source-evc40-staging-20261013-v1/runtime-restored.json"
)
RUNTIME_AUDIT_SHA256 = "34b2df693162c0be5907df3f928bddc24c0500aec3dd623e70107337082600f2"
RUNTIME_PYTHON = "/tmp/rsna-environment-907632-7sr9a7i6/environment/bin/python"
RUNTIME_ARCHIVE_SHA256 = "89597ed4d974e5874c7bb2ffb42545e6c0bb6a9ac796a5839a4b1902917ef338"
FEASIBILITY_AUDIT = "initiative/audits/patch-focus-target-feasibility-20261014-v1/v3-read-only.json"
FEASIBILITY_AUDIT_SHA256 = "aed4fcec930f4985c0cac4edeb987c03039e7dd8ce78d73cb4ccf6b06cb0596f"
PINNED_SOURCE = {
    RELAY_AUDIT: RELAY_RECEIPT_SHA256,
    RUNTIME_AUDIT: RUNTIME_AUDIT_SHA256,
    FEASIBILITY_AUDIT: FEASIBILITY_AUDIT_SHA256,
}
REQUIRED_SOURCE = (*admit.REQUIRED_SOURCE, SELF, "scripts/newton/triplet_cache_relay.py")

CONTROL_BYTES = 16 * 2**20
RESERVE_BYTES = 20 * 2**30
EXISTING_LIABILITIES_FLOOR = 338_320_442_039
FUNDING_STATUS = "root_admitted_patch_focus_evc40_target_CPU_control_funding"
FUNDING_KEYS = {
    "status",
    "admission_version",
    "experiment",
    "node",
    "funded_by",
    "original_input_receipt_sha256",
    "merged_cache_receipt_sha256",
    "cache_relay_receipt_sha256",
    "original_overlay_sha256",
    "new_CPU_source_and_receipt_bytes",
    "new_GPU_or_checkpoint_bytes",
    "new_shared_bytes_files",
    "prefix_cache_bytes",
    "GPUs_funded",
    "GPU_reserved",
    "existing_liabilities_bytes",
    "reserve_bytes",
    "required_free_bytes",
}


def _sha_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def roster_sha256(uids: Sequence[str]) -> str:
    """Same roster hash as the label-source builder/loader (sorted compact JSON)."""
    return _sha_bytes(json.dumps(sorted(uids), separators=(",", ":")).encode())


# --- Placement, runtime and funding -------------------------------------------------------


def check_target_placement(env: Mapping[str, str] | None = None, node: str | None = None) -> str:
    """A NEW standalone one-CPU zero-GPU evc40 allocation; never an old/foreign parent."""
    env = os.environ if env is None else env
    node = os.uname().nodename if node is None else node
    job = env.get("SLURM_JOB_ID")
    if (
        node != NODE
        or not job
        or job in FORBIDDEN_JOBS
        or env.get("SLURM_CPUS_PER_TASK") != "1"
        or any(env.get(name) for name in ("SLURM_JOB_GPUS", "SLURM_STEP_GPUS", "SLURM_GPUS"))
        or env.get("CUDA_VISIBLE_DEVICES") not in (None, "", "NoDevFiles")
        or torch.cuda.is_available()
    ):
        raise ValueError("New standalone evc40 one-CPU zero-GPU allocation required")
    return job


def check_runtime(
    runtime_audit: Mapping, executable: str, transformers_version: str, expected_version: str
) -> dict:
    """Actual interpreter is the once-restored original runtime; versions equal the spec."""
    runtime = runtime_audit.get("runtime", {}) if isinstance(runtime_audit, Mapping) else {}
    if (
        runtime_audit.get("status") != "distinct_evc40_original_runtime_restored_once"
        or runtime.get("python") != RUNTIME_PYTHON
        or runtime.get("archive_sha256") != RUNTIME_ARCHIVE_SHA256
        or executable != RUNTIME_PYTHON
    ):
        raise ValueError("Admission must run under the actual restored original runtime")
    if transformers_version != expected_version:
        raise ValueError("Installed transformers differs from the pinned encoder spec")
    return {
        "python": executable,
        "runtime_archive_sha256": RUNTIME_ARCHIVE_SHA256,
        "python_version": sys.version.split()[0],
        "torch_version": torch.__version__,
        "transformers_version": transformers_version,
    }


def validate_funding(funding: Mapping) -> dict:
    """Fresh 16 MiB control cap only; every existing liability plus the 20 GiB floor kept."""
    if not isinstance(funding, Mapping) or set(funding) != FUNDING_KEYS:
        raise ValueError("evc40 target funding fields differ")
    sizes = (
        "new_CPU_source_and_receipt_bytes",
        "new_GPU_or_checkpoint_bytes",
        "new_shared_bytes_files",
        "prefix_cache_bytes",
        "GPUs_funded",
        "existing_liabilities_bytes",
        "reserve_bytes",
        "required_free_bytes",
    )
    if any(type(funding[key]) is not int or funding[key] < 0 for key in sizes):
        raise ValueError("Funding sizes must be non-negative integers")
    if (
        funding["status"] != FUNDING_STATUS
        or funding["admission_version"] != VERSION
        or funding["experiment"] != admit.EXPERIMENT
        or funding["node"] != NODE
        or funding["funded_by"] != "root"
        or funding["original_input_receipt_sha256"] != dense.ORIGINAL_RECEIPT_SHA256
        or funding["merged_cache_receipt_sha256"] != dense.MERGED_RECEIPT_SHA256
        or funding["cache_relay_receipt_sha256"] != RELAY_RECEIPT_SHA256
        or funding["original_overlay_sha256"] != ORIGINAL_OVERLAY_SHA256
        or funding["new_CPU_source_and_receipt_bytes"] != CONTROL_BYTES
        or funding["new_GPU_or_checkpoint_bytes"] != 0
        or funding["new_shared_bytes_files"] != 0
        or funding["prefix_cache_bytes"] != 0
        or funding["GPUs_funded"] != 0
        or funding["GPU_reserved"] is not False
        or funding["existing_liabilities_bytes"] < EXISTING_LIABILITIES_FLOOR
        or funding["reserve_bytes"] < RESERVE_BYTES
        or funding["required_free_bytes"]
        != CONTROL_BYTES + funding["existing_liabilities_bytes"] + funding["reserve_bytes"]
    ):
        raise ValueError("evc40 control funding, retained liabilities or 20 GiB floor differ")
    return dict(funding)


# --- Exact original overlay reconstruction ---------------------------------------------


def _real_dir(path: Path) -> None:
    if path.is_symlink() or not path.is_dir() or any(p.is_symlink() for p in path.parents):
        raise ValueError("Linked, escaped or absent directory")


def _regular(path: Path) -> os.stat_result:
    s = os.lstat(path)
    if not stat.S_ISREG(s.st_mode) or s.st_nlink != 1:
        raise ValueError("Linked or non-regular file")
    return s


def read_final_overlay(path: Path = FINAL_OVERLAY, expected: str = FINAL_OVERLAY_SHA256) -> bytes:
    """The existing evc40 final overlay, exact pinned bytes only."""
    if path != FINAL_OVERLAY and expected == FINAL_OVERLAY_SHA256:
        raise ValueError("Final overlay must be the exact pinned evc40 path")
    _real_dir(path.parent)
    if _regular(path).st_size > 64 * 2**20:
        raise ValueError("Final overlay exceeds its loader cap")
    raw = path.read_bytes()
    if _sha_bytes(raw) != expected:
        raise ValueError("Final overlay differs from the admitted evc40 source")
    return raw


def original_rosters(studies: Path, folds_csv: Path, pins: Mapping = PINS) -> tuple[list, list]:
    """Exact shared study/fold bytes, then the original builder's fold-0 selection."""
    import pandas as pd

    from rsna.competitive.train import select_development

    if gate.sha(studies) != pins["release_sha256"] or gate.sha(folds_csv) != pins["folds_sha256"]:
        raise ValueError("Shared study/fold bytes differ from the original overlay binding")
    train, outer = select_development(
        pd.read_parquet(studies), pd.read_csv(folds_csv), pins["fold"], "outer_fold"
    )
    return train.study_uid.tolist(), outer.study_uid.tolist()


ORIGINAL_KEYS = (
    "schema_version",
    "status",
    "source",
    "release_sha256",
    "folds_sha256",
    "fit_scope",
    "fold",
    "fitting_uid_sha256",
    "outer_uid_sha256",
    "study_count",
    "studies",
)


def reconstruct(
    final: Mapping, fitting: Sequence[str], outer: Sequence[str], pins: Mapping = PINS
) -> bytes:
    """Original label_source_v1 bytes from the final all-development overlay.

    Membership, order, scope, release/fold and source CSV/revision must equal the original
    binding; studies are copied unchanged. Only the source name is restored.
    """
    from rsna.schema import TARGETS

    fitting, outer = list(fitting), list(outer)
    if (
        len(fitting) != pins["fitting_studies"]
        or len(outer) != pins["outer_studies"]
        or len(set(fitting)) != len(fitting)
        or len(set(outer)) != len(outer)
        or set(fitting) & set(outer)
        or roster_sha256(fitting) != pins["fitting_uid_sha256"]
        or roster_sha256(outer) != pins["outer_uid_sha256"]
        or roster_sha256(fitting + outer) != pins["development_uid_sha256"]
    ):
        raise ValueError("Original fitting/outer roster or roster hashes differ")
    if not isinstance(final, Mapping) or list(final) != list(ORIGINAL_KEYS):
        raise ValueError("Final overlay does not have the builder's exact key order")
    source = final["source"]
    if (
        final["schema_version"] != 1
        or final["status"] != "label_source_final_fit_only"
        or final["fit_scope"] != "all_development"
        or final["fold"] != pins["fold"]
        or final["release_sha256"] != pins["release_sha256"]
        or final["folds_sha256"] != pins["folds_sha256"]
        or final["fitting_uid_sha256"] != pins["development_uid_sha256"]
        or final["outer_uid_sha256"] != roster_sha256([])
        or not isinstance(source, Mapping)
        or list(source) != list(ORIGINAL_SOURCE)
        or source["name"] != FINAL_SOURCE_NAME
        or {k: v for k, v in source.items() if k != "name"}
        != {k: v for k, v in ORIGINAL_SOURCE.items() if k != "name"}
    ):
        raise ValueError("Final overlay scope, release/folds, roster or pinned source differ")
    studies = final["studies"]
    if not isinstance(studies, list) or final["study_count"] != len(studies):
        raise ValueError("Final overlay study table is inconsistent")
    uids = [s.get("study_uid") if isinstance(s, Mapping) else None for s in studies]
    if len(set(uids)) != len(uids) or set(uids) != set(fitting) | set(outer):
        raise ValueError("Final overlay membership differs from the development roster")
    for study in studies:
        if (
            list(study) != ["study_uid", "labels", "mask"]
            or not isinstance(study["labels"], dict)
            or not isinstance(study["mask"], dict)
            or list(study["labels"]) != list(TARGETS)
            or list(study["mask"]) != list(TARGETS)
            or any(type(v) is not int or v not in (0, 1) for v in study["mask"].values())
        ):
            raise ValueError("Final overlay study lacks exact binary target masks")
    member = set(fitting)
    kept = [study for study in studies if study["study_uid"] in member]
    if [study["study_uid"] for study in kept] != fitting:
        raise ValueError("Final overlay fitting order differs from the original selection")
    artifact = {
        "schema_version": 1,
        "status": "label_source_experiment_only",
        "source": dict(ORIGINAL_SOURCE),
        "release_sha256": pins["release_sha256"],
        "folds_sha256": pins["folds_sha256"],
        "fit_scope": "outer_fold",
        "fold": pins["fold"],
        "fitting_uid_sha256": pins["fitting_uid_sha256"],
        "outer_uid_sha256": pins["outer_uid_sha256"],
        "study_count": len(kept),
        "studies": kept,
    }
    return (json.dumps(artifact, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def control_usage(root: Path) -> int:
    """Bytes already inside the funded control directory; any link is refused."""
    total = 0
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Linked entry inside the funded control directory")
        if path.is_file():
            total += _regular(path).st_size
    return total


def write_overlay(root: Path, raw: bytes, funding: Mapping, pins: Mapping = PINS) -> Path:
    """Exclusive 0400 write of exactly the original bytes inside the 16 MiB control cap."""
    if _sha_bytes(raw) != pins["overlay_sha256"] or len(raw) != pins["overlay_bytes"]:
        raise ValueError("Reconstruction is not the exact original overlay; nothing written")
    _real_dir(root)
    destination = root / OVERLAY_NAME
    if destination.exists() or destination.is_symlink() or (root / OVERLAY_RECEIPT).exists():
        raise ValueError("Existing overlay destination forbids repetition")
    if control_usage(root) + len(raw) + gate.CPU_RECEIPT_CAP_BYTES > CONTROL_BYTES:
        raise ValueError("Overlay would exceed the funded control cap")
    dense.check_storage(root, funding)
    fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600)
    with os.fdopen(fd, "wb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    os.chmod(destination, 0o400)
    if gate.sha(destination) != pins["overlay_sha256"]:
        raise ValueError("Written overlay differs from the exact original")
    return destination


def check_overlay(root: Path, pins: Mapping = PINS) -> Path:
    """The previously reconstructed exact original overlay, unchanged since its write."""
    path = root / OVERLAY_NAME
    s = _regular(path)
    if (
        stat.S_IMODE(s.st_mode) != 0o400
        or s.st_size != pins["overlay_bytes"]
        or gate.sha(path) != pins["overlay_sha256"]
    ):
        raise ValueError("Reconstructed original overlay is absent or changed")
    receipt = json.loads((root / OVERLAY_RECEIPT).read_text())
    if (
        receipt.get("status") != OVERLAY_STATUS
        or receipt.get("overlay_sha256") != (pins["overlay_sha256"])
    ):
        raise ValueError("Overlay reconstruction receipt is absent or differs")
    return path


# --- Verified relay proof instead of the untransferred producer receipt ----------------


def validate_relay_receipt(raw: bytes, audit: bytes) -> dict:
    """Actual target-verified relay receipt, byte-equal to its Git audit copy."""
    if raw != audit or _sha_bytes(raw) != RELAY_RECEIPT_SHA256:
        raise ValueError("Relay receipt differs from the verified actual transfer")
    receipt = json.loads(raw)
    if (
        receipt.get("status") != "original_complete_public_triplet_cache_private_relay_verified"
        or receipt.get("relay_version") != relay.VERSION
        or receipt.get("node") != NODE
        or receipt.get("cache") != gate.PUBLIC_CACHE
        or receipt.get("index_sha256") != relay.INDEX_SHA256
        or receipt.get("cohort_sha256") != relay.COHORT_SHA256
        or receipt.get("merged_receipt_sha256") != dense.MERGED_RECEIPT_SHA256
        or receipt.get("CPU_receipt_sha256") != relay.CPU_RECEIPT_SHA256
        or (receipt.get("files"), receipt.get("bytes")) != (RELAY_FILES, RELAY_BYTES)
        or receipt.get("studies") != relay.STUDIES
        or receipt.get("every_original_byte_size_SHA_count_EOF_layout_verified") is not True
        or receipt.get("GPU_used") is not False
    ):
        raise ValueError("Relay receipt does not bind the original cache lineage")
    return receipt


def require_no_forged_merged_receipt(path: Path = Path(gate.MERGED_RECEIPT)) -> None:
    """The original producer receipt was never relayed; a target copy cannot be original."""
    if path.exists() or path.is_symlink():
        raise ValueError("Unexpected target merged-cache receipt; original was not transferred")


def check_publication(cache: Path, manifest: Mapping[str, Mapping], receipt: Path) -> dict:
    """Fresh stat proof that the relayed bytes are the published originals; no rehash."""
    for directory in (cache, cache / relay.DIRECTORY, cache / relay.MEMBERS_DIR):
        _real_dir(directory)
    verified = _regular(receipt).st_mtime_ns
    layout = sorted(str(p.relative_to(cache)) for p in cache.rglob("*") if not p.is_dir())
    if layout != sorted(manifest):
        raise ValueError("Target cache membership differs from the original index")
    signatures = []
    for name in sorted(manifest):
        s = _regular(cache / name)
        if (
            stat.S_IMODE(s.st_mode) != 0o400
            or s.st_size != manifest[name]["bytes"]
            or s.st_ctime_ns > verified
            or s.st_mtime_ns > verified
        ):
            raise ValueError("Target member changed since its verified original publication")
        signatures.append([name, s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns])
    return {
        "files": len(signatures),
        "bytes": sum(row[3] for row in signatures),
        "signatures_sha256": gate.canonical(signatures),
        "regular_single_link_0400_unchanged_since_relay_verification": True,
        "member_bytes_rehashed": False,
    }


def target_cache_proof(code: Path, cohort: Mapping) -> dict:
    cache = Path(gate.PUBLIC_CACHE)
    require_no_forged_merged_receipt()
    receipt = validate_relay_receipt(RELAY_RECEIPT.read_bytes(), (code / RELAY_AUDIT).read_bytes())
    index = (cache / relay.INDEX).read_bytes()
    if _sha_bytes(index) != relay.INDEX_SHA256 or cohort["index_sha256"] != relay.INDEX_SHA256:
        raise ValueError("Target cache index differs from the original pinned index")
    manifest = relay.manifest_from_index(index)
    publication = check_publication(cache, manifest, RELAY_RECEIPT)
    if (publication["files"], publication["bytes"]) != (receipt["files"], receipt["bytes"]):
        raise ValueError("Target publication differs from the verified relay aggregate")
    return {
        "relay_receipt_sha256": RELAY_RECEIPT_SHA256,
        "relay_manifest_sha256": receipt["manifest_sha256"],
        "index_sha256": relay.INDEX_SHA256,
        "merged_receipt_sha256_lineage": dense.MERGED_RECEIPT_SHA256,
        "original_merged_receipt_file_on_target": False,
        "publication": publication,
    }


# --- Steps ------------------------------------------------------------------------------


def verify_source(code: Path, manifest: Mapping) -> dict[str, str]:
    """Original admission source binding plus this wrapper, relay code and pinned audits."""
    code_sha256 = admit.verify_source(code, manifest)
    files = manifest["files"]
    if not set(REQUIRED_SOURCE) | set(PINNED_SOURCE) <= set(files) or any(
        files[name] != value for name, value in PINNED_SOURCE.items()
    ):
        raise ValueError("Staged source lacks the evc40 wrapper or pinned target audits")
    return code_sha256


def _original(code: Path) -> tuple[dict, dict]:
    path = code / dense.ORIGINAL_RECEIPT
    if gate.sha(path) != dense.ORIGINAL_RECEIPT_SHA256:
        raise ValueError("Original public-input CPU receipt changed")
    original = json.loads(path.read_text())
    return original, dense.validate_original(original)


def _write_once(path: Path, value: Mapping, private: Sequence[str]) -> None:
    if control_usage(path.parent) + gate.CPU_RECEIPT_CAP_BYTES > CONTROL_BYTES:
        raise ValueError("Receipt would exceed the funded control cap")
    dense.write_once(path, value, private)


def overlay_step(root: Path) -> dict:
    job = check_target_placement()
    torch.set_num_threads(1)
    _real_dir(root)
    funding_path = root / "funding.json"
    funding = validate_funding(json.loads(funding_path.read_text()))
    code = root / "code"
    _, inputs = _original(code)
    fitting, outer = original_rosters(Path(inputs["studies"]), Path(inputs["folds"]) / "fold.csv")
    raw = reconstruct(json.loads(read_final_overlay()), fitting, outer)
    path = write_overlay(root, raw, funding)
    value = {
        "status": OVERLAY_STATUS,
        "admission_version": VERSION,
        "CPU_job": job,
        "node": NODE,
        "final_overlay_sha256": FINAL_OVERLAY_SHA256,
        "overlay_sha256": gate.sha(path),
        "overlay_bytes": path.stat().st_size,
        "source": dict(ORIGINAL_SOURCE),
        "restored_provenance_fields": ["source.name"],
        "fitting_studies": len(fitting),
        "outer_studies": len(outer),
        "fitting_uid_sha256": PINS["fitting_uid_sha256"],
        "outer_uid_sha256": PINS["outer_uid_sha256"],
        "funding_sha256": gate.sha(funding_path),
        "labels_or_masks_changed": False,
        "source_acquired_or_downloaded": False,
        "evc31_accessed": False,
        "GPU_used": False,
    }
    _write_once(root / OVERLAY_RECEIPT, value, [*fitting, *outer])
    print(json.dumps({"status": OVERLAY_STATUS, "CPU_job": job}), flush=True)
    return value


def admit_step(root: Path) -> dict:
    """The original patch-focus admission with only the target-specific substitutions."""
    import transformers

    from rsna.competitive import backbones
    from rsna.competitive.data import collate_studies
    from rsna.competitive.folds import load_folds
    from rsna.competitive.patch_focus import classifier_model
    from rsna.competitive.public_input import PublicTripletStudyDataset, verify_public_cohort
    from rsna.competitive.source_slot import training_inputs
    from rsna.competitive.supervision import apply_fitting_supervision
    from rsna.competitive.train import _uid_hash, select_development, verified_job_release
    from rsna.schema import TARGETS

    job = check_target_placement()
    torch.set_num_threads(1)
    _real_dir(root)
    code = root / "code"
    manifest_path = root / "source.json"
    manifest = json.loads(manifest_path.read_text())
    code_sha256 = verify_source(code, manifest)
    funding_path = root / "funding.json"
    funding = validate_funding(json.loads(funding_path.read_text()))
    observed_free = dense.check_storage(root, funding)
    recovery_manifest_path = root / "recovery-source-manifest.json"
    if json.loads(recovery_manifest_path.read_text()) != manifest["files"]:
        raise ValueError("GPU wrapper source manifest differs from the admitted source")
    if recovery.verify_source(code, manifest["files"]) != code_sha256:
        raise ValueError("GPU wrapper would bind different numerical source")
    original, inputs = _original(code)
    runtime = check_runtime(
        json.loads((code / RUNTIME_AUDIT).read_text()),
        sys.executable,
        transformers.__version__,
        original["configs"]["public"]["encoder"]["transformers_version"],
    )
    delta = admit.code_delta(original["code_sha256"], code_sha256)
    configs = admit.load_arms(code, original["configs"]["public"])
    spec_path = root / "encoder-spec.json"
    spec = admit.encoder_spec(configs, json.loads(spec_path.read_text()))
    overlay = check_overlay(root)

    config = configs["reference"]
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
    cache_proof = target_cache_proof(code, cohort)
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

    value = {
        "status": STATUS,
        "admission_version": VERSION,
        "experiment": admit.EXPERIMENT,
        "node": NODE,
        "CPU_job": job,
        "source_git": manifest["source_git"],
        "source_manifest_sha256": gate.sha(manifest_path),
        "recovery_source_manifest_sha256": gate.sha(recovery_manifest_path),
        "wrapper_sha256": manifest["files"][admit.WRAPPER],
        "admission_sha256": manifest["files"][SELF],
        "original_admission_sha256": manifest["files"][admit.SELF],
        "recipe_sha256": dict(admit.RECIPE_SHA256),
        "code_sha256": code_sha256,
        "code_delta_from_original": delta,
        "runtime": runtime,
        "encoder_spec": spec | {"path_sha256": gate.sha(spec_path)},
        "funding_sha256": gate.sha(funding_path),
        "fresh_storage": {
            "observed_free_bytes": observed_free,
            "required_free_bytes": funding["required_free_bytes"],
            "existing_liabilities_bytes": funding["existing_liabilities_bytes"],
            "reserve_bytes": funding["reserve_bytes"],
        },
        "original_CPU_receipt_sha256": dense.ORIGINAL_RECEIPT_SHA256,
        "target_cache": cache_proof,
        "whole_byte_base_proof_sha256": gate.BASE_PROOF_SHA256,
        "official_base_fresh_stats_equal_original": True,
        "official_encoder_provenance_equal_original": True,
        "release_sha256": binding["manifest_sha256"],
        "folds_sha256": supervision["folds_sha256"],
        "overlay_sha256": supervision["overlay_sha256"],
        "overlay_reconstructed_from_final_sha256": FINAL_OVERLAY_SHA256,
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
        ),
        "GPU_wrapper_arguments": {
            "source": str(code),
            "source-manifest": str(recovery_manifest_path),
            "encoder-spec": str(spec_path),
            "artifact-root": inputs["artifact-root"],
            "cache": gate.PUBLIC_CACHE,
            "overlay": str(overlay),
            "max-output-bytes": admit.RECOVERY_OUTPUT_CAP_BYTES,
        },
        "optimizer_step_executed": False,
        "GPU_or_clinical_inference_executed": False,
        "merge_render_download_or_transfer_executed": False,
        "protected_labels_used_for_recipe_or_metrics": False,
        "new_labels_created": False,
        "accuracy_claimed": False,
        "GPU_acceptance_inferred": False,
        "diagnostic_only": True,
    }
    private = [str(uid) for part in (fitting, outer) for uid in part.study_uid]
    private += [str(group) for part in (fitting, outer) for group in part.leakage_group]
    _write_once(root / CPU_RECEIPT, value, private)
    print(json.dumps({"status": STATUS, "CPU_job": job}), flush=True)
    return value


STEPS: dict[str, Callable[[Path], dict]] = {"overlay": overlay_step, "admit": admit_step}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=sorted(STEPS))
    parser.add_argument("--root", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.root != CONTROL_ROOT:
        raise SystemExit("Only the separately funded evc40 control root is admitted")
    STEPS[arguments.step](arguments.root)
