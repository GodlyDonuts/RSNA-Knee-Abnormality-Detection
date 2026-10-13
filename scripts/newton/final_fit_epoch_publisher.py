"""Preserve one completed all-development RSNA-only final-fit epoch once, privately.

Distinct from every fold/experiment publisher: one final arm, 3,403 fitting studies,
zero evaluation studies, label_source_final_v1 supervision, fixed10 x426 updates.
The token arrives only as JSON on stdin; argv carries only the completed epoch.
Any existing journal or remote path means read-only reconciliation, never retry.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path

ROOT = Path("/tmp/rsna-final-public-source-fit-evc40-20261013-v1")
RUN = "production"
NODE = "evc40"
REPO = "GodlyDonuts/RSNA-Data"
PREFIX = (
    "experiments/dinov3-v1/training-epochs/"
    "public-label-source-final-all-development-1bcd6c0-fixed10-seed20260922"
)
OLD_PREFIXES = tuple(
    "experiments/dinov3-v1/training-epochs/" + name
    for name in (
        "adapter-a32-all-8911f90-seed20260922",
        "adapter-a32-fold0-6f1ee6d-seed20260922",
        "adapter-a32-native384-all-development-c9fcc7e-seed20260922",
        "adapter-a32-native384-fold0-c9fcc7e-seed20260922",
        "adapter-a32-sampling-fold0-7ac7a97-seed20260922",
        "adapter-a64-sampled-fold0-c9fcc7e-seed20260922",
        "cache-contrast-fold0-8b627d25-fixed10-seed20260922",
        "cls-patch-summary-fold0-0183ef3-fixed10-seed20260922",
        "effusion-reviewed31-fold0-665c0d5-screen3-v1",
        "effusion-reviewed31-fold0-665c0d5-seed20261004-screen3-v1",
        "fold0-seed20260922",
        "fold1-seed20260922",
        "fold2-seed20260922",
        "neighbor-channels-fold0-d4c3c65-fixed3-seed20260922",
        "patchattention-fold0-203f419-seed20260922",
        "patchattention-fold1-203f419-seed20260922",
        "patchattention-fold2-203f419-seed20260922",
        "plane-memory-fold0-b2bb3a9-fixed3-seed20260922",
        "public-label-coverage-fold0-a9902f3-fixed10-seed20260922",
        "public-label-fixedmask-fold0-a9902f3-fixed10-seed20260922",
        "public-label-source-fold0-a9902f3-fixed10-seed20260922",
        "public-label-source-fold0-a9902f3-fixed10-seed20261012",
        "smooth-witness-fold0-3b7e437-fixed3-seed20260922",
    )
)
SOURCE_GIT = "1bcd6c0e2e2a47d6894d5c545d6950e0c6f0d5b9"
CPU_JOB = "905196"
CPU_STATUS = "actual_final_public_source_CPU_roster_and_16MRI_inputs_verified"
CPU_RECEIPT_SHA = "107a15da1a2beafa294bdcba1f62652d7b94a105ee5640207d74b8656dce52a4"
T4_KERNEL = (138516408, "godlydonuts/rsna-final-publicsource-recovery-20261013-v2", 1)
T4_RECEIPT_SHA = "ea6a25536b2c95d959d604d6c8b1e7c2db7126567e1caccaa3fac8f9fec1d635"
T4_CPU_LINK_SHA = "88014aeeff1eeb60b4593d84ddf9e0b09e252afc5c3745046424197acdacbff3"
CONFIG_SHA = "5b3268f5cc33698d8bbe05a9fbdf7d966212b6f16f2c6be66c8e82cc8b1ca8a6"
OVERLAY_SHA = "54b5a652949120764a596b4d495f11950ff103a909b7efc5fe59da9cd6f2fe85"
RELEASE_SHA = "2bf49fd2a8ae0b7c5af7e8c105fc3f2337845a92654d136fe67bc41a1e52fae7"
FOLDS_FILE_SHA = "fe590e5072a0d82abeb377598aeef0021bbe77fe4529a19b9826a7d40d322050"
SOURCE_CSV_SHA = "eb4c2b29e7e84adbb4b5ccee825ca20ade6a43dbe295f13be442d549e4f5dca4"
FITTING_UID_SHA = "6d00d9aa90799aca0f32b5e0918f9716f00f7a3e516db3a4625aaed46b373844"
EMPTY_ROSTER_SHA = hashlib.sha256(b"[]").hexdigest()
ENCODER = {
    "architecture": "dinov3_l16",
    "source_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "source_revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
    "weights_sha256": "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
    "config_sha256": "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5",
}
LICENSE = Path(
    "/home/sa305415/rsna-knee/encoder-stage/configs/competitive/weights/dinov3-large.LICENSE.md"
)
LICENSE_SHA = "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e"
LICENSE_SOURCE = (
    "https://github.com/facebookresearch/dinov3/blob/"
    "ffb4bb89c6558ca3244655c25a3955d01788b732/LICENSE.md"
)
LOCK_RUNNER = Path("/home/sa305415/rsna-knee/native-full-cache-two-lanes-v1/common/runner.py")
LOCK_SHA = "3575a18ecd7a13a2becc9348c48a9ebda95f8934069e67dd1912737e6194118c"
TORCH_VERSION = "2.14.0+cu130"
STUDIES, EPOCHS, UPDATES, SEED = 3403, 10, 426, 20260922
CHECKPOINT_CAP = 128 * 2**20
LOCAL_CAP = 6 * 2**30
SMALL_OBJECT_BOUND = 2**20
FLOOR = 20 * 2**30
MEMBERS = (
    "checkpoint.pt",
    "snapshot.json",
    "source-and-input-provenance.json",
    "DINOv3-LICENSE.md",
    "manifest.json",
)
ADMISSION_STATUS = "actual_final_public_source_production_admitted"
FORBIDDEN_BINDING = (
    "external_initialization",
    "private_research_only",
    "external_weight_publication",
    "public_input_cache",
)
CONTAMINATION = re.compile(r"(prediction|metric|evaluation|outer|best|selected)", re.IGNORECASE)
RAW_KEYS = {
    "study_uid",
    "series_uid",
    "study_uids",
    "series_uids",
    "uids",
    "studies",
    "labels",
    "mask",
    "masks",
    "report",
    "reports",
    "patient",
    "patient_id",
    "observed_shared_quota",
}
DICOM_UID = re.compile(r"\d+(?:\.\d+){4,}")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b""):
            h.update(block)
    return h.hexdigest()


def digest(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def save(path: Path, value: object) -> None:
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def regular(path: Path, root: Path) -> Path:
    relative = path.relative_to(root)
    if (
        path.is_symlink()
        or root.is_symlink()
        or any((root / parent).is_symlink() for parent in relative.parents if parent != Path("."))
        or not path.is_file()
    ):
        raise ValueError("Exact regular non-linked control file required")
    return path


def load(path: Path, root: Path) -> dict:
    value = json.loads(regular(path, root).read_text())
    if not isinstance(value, dict):
        raise ValueError("Invalid control record")
    return value


def assert_sanitized(value: object) -> None:
    """Published receipts carry hashes/counts only; no case IDs or clinical values."""
    if isinstance(value, dict):
        for key, item in value.items():
            if key in RAW_KEYS:
                raise ValueError("Raw data/control field forbidden in published receipt")
            assert_sanitized(item)
    elif isinstance(value, list):
        for item in value:
            assert_sanitized(item)
    elif isinstance(value, str) and DICOM_UID.search(value):
        raise ValueError("Raw case identifier forbidden in published receipt")


def admitted_inputs(root: Path) -> tuple[dict, dict, dict, dict]:
    admission = load(root / "production-admission.json", root)
    cpu = load(root / "CPU-accepted.json", root)
    t4 = load(root / "T4-recovery-accepted.json", root)
    if (
        admission.get("status") != ADMISSION_STATUS
        or admission.get("node") != NODE
        or admission.get("source_git") != SOURCE_GIT
        or admission.get("private_prefix") != PREFIX
        or admission.get("shared_future_bytes") != 0
        or admission.get("CPU_receipt_sha256") != CPU_RECEIPT_SHA
        or admission.get("T4_recovery_receipt_sha256") != T4_RECEIPT_SHA
        or admission.get("final_config_sha256") != CONFIG_SHA
        or admission.get("overlay_sha256") != OVERLAY_SHA
        or not re.fullmatch(r"[0-9a-f]{64}", str(admission.get("initial_run_binding_sha256")))
        or sha(regular(root / "publish.py", root)) != admission.get("publisher_sha256")
    ):
        raise ValueError("Actual root-owned final production admission required")
    if any(PREFIX == old or PREFIX.startswith(old + "/") for old in OLD_PREFIXES) or any(
        old.startswith(PREFIX) for old in OLD_PREFIXES
    ):
        raise ValueError("Fresh distinct private prefix required")
    if (
        sha(root / "CPU-accepted.json") != CPU_RECEIPT_SHA
        or cpu.get("status") != CPU_STATUS
        or cpu.get("CPU_job") != CPU_JOB
        or cpu.get("source_git") != SOURCE_GIT
        or (cpu.get("training_studies"), cpu.get("outer_studies")) != (STUDIES, 0)
        or cpu.get("masked_out_source_studies") != 0
        or cpu.get("fitting_uid_sha256") != FITTING_UID_SHA
        or cpu.get("release_sha256") != RELEASE_SHA
        or cpu.get("folds_sha256") != FOLDS_FILE_SHA
        or cpu.get("overlay_sha256") != OVERLAY_SHA
        or cpu.get("config_sha256") != CONFIG_SHA
        or cpu.get("source_csv_sha256") != SOURCE_CSV_SHA
        or (cpu.get("optimizer_updates_per_epoch"), cpu.get("fixed_final_optimizer_updates"))
        != (UPDATES, UPDATES * EPOCHS)
    ):
        raise ValueError("Original accepted CPU905196 final-scope receipt required")
    recovery = t4.get("actual_recovery", {})
    updates = recovery.get("finite_updates")
    if (
        sha(root / "T4-recovery-accepted.json") != T4_RECEIPT_SHA
        or t4.get("status") != "COMPLETE"
        or t4.get("actual_remote", {}).get("status") != "COMPLETE"
        or (t4.get("kernel_id", T4_KERNEL[0]), t4.get("kernel_slug"), t4.get("version"))
        != T4_KERNEL
        or t4.get("exact_source_private_offline_T4_verified") is not True
        or recovery.get("status") != "actual_single_T4_final_public_source_recovery_accepted"
        or recovery.get("science_git") != SOURCE_GIT
        or recovery.get("CPU_receipt_sha256") != T4_CPU_LINK_SHA
        or recovery.get("exact_complete_state_scaler_RNG_recovery") is not True
        or recovery.get("amp_scaler", {}).get("_growth_tracker") != 4
        or not isinstance(updates, list)
        or len(updates) != 4
    ):
        raise ValueError("Exact private T4 final-scope recovery receipt required")
    raw = load(root / "final-config.json", root)
    if sha(root / "final-config.json") != CONFIG_SHA or sha(
        regular(root / "overlay.json", root)
    ) != (OVERLAY_SHA):
        raise ValueError("Exact raw final config/overlay required")
    expected = {
        name.removeprefix("code/rsna/"): value
        for name, value in cpu.get("source_modules", {}).items()
        if name.startswith("code/rsna/")
    }
    code = root / "code" / "rsna"
    actual = (
        {
            str(p.relative_to(code)): p
            for p in sorted(code.rglob("*.py"))
            if "__pycache__" not in p.parts
        }
        if code.is_dir()
        else {}
    )
    if not expected or set(actual) != set(expected):
        raise ValueError("Exact original numerical source file set required")
    for name, path in actual.items():
        if sha(regular(path, root)) != expected[name]:
            raise ValueError("Frozen scientific source changed")
    return admission, cpu, t4, {"raw_config": raw, "code_sha256": expected}


def validate_binding(binding: dict, raw: dict, code_sha256: dict) -> None:
    config = binding.get("config", {})
    for key, value in raw.items():
        if key != "encoder" and config.get(key) != value:
            raise ValueError("Bound recipe differs from exact raw final config")
    if (
        config.get("fit_scope") != "all_development"
        or config.get("fitting_supervision")
        != {"kind": "label_source_final_v1", "sha256": OVERLAY_SHA, "enabled": True}
        or (config.get("epochs"), config.get("warmup_updates"), config.get("seed"))
        != (
            EPOCHS,
            UPDATES,
            SEED,
        )
        or config.get("submission_every_epochs") != 1
        or config.get("diagnostic_study_limit") is not None
        or config.get("fold") != 0
        or config.get("checkpoint_format") != "frozen_base_delta"
    ):
        raise ValueError("Final scope/source/recipe differs")
    encoder = config.get("encoder", {})
    raw_encoder = raw.get("encoder", {})
    if any(encoder.get(k) != v or raw_encoder.get(k) != v for k, v in ENCODER.items()):
        raise ValueError("Official DINOv3-L base provenance differs")
    if any(k in binding for k in FORBIDDEN_BINDING):
        raise ValueError("Only official-base RSNA-only weights are admitted")
    receipt = binding.get("experimental_fitting_supervision", {})
    if (
        receipt.get("overlay_sha256") != OVERLAY_SHA
        or receipt.get("release_sha256") != RELEASE_SHA
        or receipt.get("folds_sha256") != FOLDS_FILE_SHA
        or receipt.get("fitting_uid_sha256") != FITTING_UID_SHA
        or receipt.get("outer_uid_sha256") != EMPTY_ROSTER_SHA
        or receipt.get("fitting_studies") != STUDIES
        or receipt.get("fit_scope") != "all_development"
        or receipt.get("final_fit_only") is not True
        or receipt.get("internal_validation_available") is not False
        or receipt.get("evaluation_masks_changed") is not False
        or receipt.get("enabled") is not True
        or receipt.get("source", {}).get("csv_sha256") != SOURCE_CSV_SHA
    ):
        raise ValueError("label_source_final_v1 all-development supervision receipt differs")
    if (
        binding.get("training_uids_sha256") != FITTING_UID_SHA
        or binding.get("outer_uids_sha256") != EMPTY_ROSTER_SHA
        or binding.get("release", {}).get("manifest_sha256") != RELEASE_SHA
        or binding.get("code_sha256") != code_sha256
        or (binding.get("world_size"), binding.get("device_type")) != (1, "cuda")
        or binding.get("torch_version") != TORCH_VERSION
    ):
        raise ValueError("Roster/release/source/runtime binding differs")


def validate_run(run: Path, root: Path) -> dict:
    record = load(run / "run.json", root)
    if (
        record.get("diagnostic_run") is not False
        or (record.get("training_studies"), record.get("outer_studies")) != (STUDIES, 0)
        or record.get("selection_policy") != "fixed_epoch_count; no_internal_evaluation"
        or record.get("held_out_confirmation_and_gold_used") is not False
    ):
        raise ValueError("Full non-diagnostic all-development run required")
    for path in run.rglob("*"):
        if path.is_symlink() or CONTAMINATION.search(path.name):
            raise ValueError("Held-out/diagnostic/selection artifact contaminates final run")
    if (run / "result.json").exists():
        result = load(run / "result.json", root)
        if (
            result.get("fit_scope") != "all_development"
            or result.get("diagnostic_run") is not False
            or result.get("outer_studies") != 0
            or result.get("evaluation_source") is not None
            or result.get("metrics") is not None
            or result.get("predictions_sha256") is not None
            or result.get("checkpoint_selection") != "fixed_final_epoch_no_internal_selection"
        ):
            raise ValueError("Final fit must not carry held-out metrics/predictions")
    return record


def expected_snapshot(binding: dict, epoch: int, checkpoint: Path) -> dict:
    return {
        "schema_version": 1,
        "status": "scheduled_epoch_snapshot",
        "epoch": epoch,
        "optimizer_steps": UPDATES * epoch,
        "checkpoint_file": f"epoch-{epoch:03d}.pt",
        "checkpoint_sha256": sha(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "binding_sha256": digest(binding),
        "release_binding_sha256": binding["release"]["binding_sha256"],
        "fold": 0,
        "seed": SEED,
        "submission_every_epochs": 1,
        "diagnostic_only": False,
        "submission_eligible": True,
        "submitted": False,
    }


def validate_checkpoint_state(checkpoint: Path, binding: dict, epoch: int) -> None:
    import torch

    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if (
        state.get("schema_version") != 2
        or state.get("binding") != binding
        or (state.get("epoch"), state.get("next_update"), state.get("step"))
        != (epoch, 0, UPDATES * epoch)
        or state.get("optimizer_boundary") is not True
        or state.get("scheduled_epoch") is not True
        or "amp_scaler" not in state
        or len(state.get("rng_by_rank", [])) != 1
    ):
        raise ValueError("Checkpoint is not the exact completed scheduled epoch state")


def local_obligations(run: Path, current: int) -> int:
    epochs = run / "epoch-checkpoints"
    inodes: dict[tuple[int, int], int] = {}
    for path in [*epochs.glob("epoch-*.pt"), run / "checkpoint.pt"]:
        if path.is_file():
            s = path.stat()
            inodes[(s.st_dev, s.st_ino)] = s.st_size
    completed = len(list(epochs.glob("epoch-*.pt")))
    # Remaining scheduled epochs, one atomic successor, one verification download
    # of every member, and the small journal objects for every epoch.
    return (
        sum(inodes.values())
        + (EPOCHS - completed + 2) * CHECKPOINT_CAP
        + EPOCHS * len(MEMBERS) * SMALL_OBJECT_BOUND
    )


def source_files(epoch: int, root: Path = ROOT) -> tuple[dict, dict, dict]:
    if type(epoch) is not int or epoch not in range(1, EPOCHS + 1):
        raise ValueError("One exact completed final-fit epoch required")
    admission, cpu, t4, frozen = admitted_inputs(root)
    run = root / RUN
    record = validate_run(run, root)
    binding = record["binding"]
    if digest(binding) != admission["initial_run_binding_sha256"]:
        raise ValueError("Root-owned initial run binding differs")
    validate_binding(binding, frozen["raw_config"], frozen["code_sha256"])
    metadata = run / "epoch-checkpoints" / f"epoch-{epoch:03d}.json"
    checkpoint = metadata.with_suffix(".pt")
    regular(metadata, root)
    regular(checkpoint, root)
    if checkpoint.stat().st_size >= CHECKPOINT_CAP:
        raise ValueError("Completed epoch exceeds funded checkpoint cap")
    if json.loads(metadata.read_text()) != expected_snapshot(binding, epoch, checkpoint):
        raise ValueError("Actual immutable completed snapshot differs")
    validate_checkpoint_state(checkpoint, binding, epoch)
    if local_obligations(run, epoch) > LOCAL_CAP:
        raise ValueError("Local checkpoint/publication obligations exceed original bound")
    if sha(regular(LICENSE, LICENSE.parent)) != LICENSE_SHA:
        raise ValueError("Pinned official license differs")
    provenance = {
        "schema_version": 1,
        "status": "final_public_source_all_development_input_provenance",
        "source_git": SOURCE_GIT,
        "source_sha256": frozen["code_sha256"],
        "CPU_job": CPU_JOB,
        "CPU_receipt_sha256": CPU_RECEIPT_SHA,
        "inputs": {
            k: cpu[k]
            for k in (
                "release_sha256",
                "folds_sha256",
                "overlay_sha256",
                "config_sha256",
                "source_csv_sha256",
                "fitting_uid_sha256",
                "training_studies",
                "outer_studies",
                "masked_out_source_studies",
                "optimizer_updates_per_epoch",
                "fixed_final_optimizer_updates",
            )
        },
        "outer_uid_sha256": EMPTY_ROSTER_SHA,
        "T4_recovery": {
            "kernel_id": T4_KERNEL[0],
            "kernel_slug": T4_KERNEL[1],
            "version": T4_KERNEL[2],
            "receipt_sha256": T4_RECEIPT_SHA,
            "CPU_link_sha256": T4_CPU_LINK_SHA,
            "complete_state_sha256": t4["actual_recovery"].get("complete_state_sha256"),
            "finite_updates": 4,
            "exact_complete_state_scaler_RNG_recovery": True,
        },
        "admission_sha256": sha(root / "production-admission.json"),
        "initial_run_binding_sha256": admission["initial_run_binding_sha256"],
        "encoder": ENCODER,
        "license_source": LICENSE_SOURCE,
        "license_sha256": LICENSE_SHA,
        "fitting_supervision": "label_source_final_v1",
        "fit_scope": "all_development",
        "held_out_metrics_or_predictions": False,
    }
    manifest = {
        "schema_version": 1,
        "status": "completed_final_public_source_all_development_fit_epoch",
        "repo_id": REPO,
        "prefix": f"{PREFIX}/epoch-{epoch:03d}",
        "scientific_source_git": SOURCE_GIT,
        "epoch": epoch,
        "optimizer_updates": UPDATES * epoch,
        "checkpoint_sha256": sha(checkpoint),
        "binding_sha256": digest(binding),
        "release_binding_sha256": binding["release"]["binding_sha256"],
        "CPU_receipt_sha256": CPU_RECEIPT_SHA,
        "T4_recovery_receipt_sha256": T4_RECEIPT_SHA,
        "training_studies": STUDIES,
        "outer_studies": 0,
        "fixed_final_epochs": EPOCHS,
        "checkpoint_selection": "fixed_scheduled_epoch_only; no_epoch_selection",
        "fit_scope": "all_development",
        "fitting_supervision": "label_source_final_v1",
        "official_base_and_license_required": True,
        "encoder": ENCODER,
        "clinical_or_Kaggle_gain_claimed": False,
        "protected_gold_or_confirmation_used": False,
        "held_out_metrics_or_predictions": False,
    }
    members = {
        "checkpoint.pt": checkpoint,
        "snapshot.json": metadata,
        "DINOv3-LICENSE.md": LICENSE,
    }
    for value in (provenance, manifest, json.loads(metadata.read_text())):
        assert_sanitized(value)
    return manifest, provenance, members


def remote_paths(api, paths: list[str], revision: str) -> set[str]:
    return {
        item.path
        for item in api.get_paths_info(REPO, paths, repo_type="dataset", revision=revision)
    }


def verify_private_revision(api, prefix: str, expected: dict, revision: str, root: Path) -> None:
    paths = [f"{prefix}/{name}" for name in MEMBERS]
    if remote_paths(api, paths, revision) != set(paths):
        raise ValueError("Returned private revision incomplete")
    if api.repo_info(REPO, repo_type="dataset", revision=revision).private is not True:
        raise ValueError("Returned revision is not private")
    with tempfile.TemporaryDirectory(prefix="verify-", dir=root) as tmp:
        for name, row in expected.items():
            path = Path(
                api.hf_hub_download(
                    REPO,
                    filename=f"{prefix}/{name}",
                    repo_type="dataset",
                    revision=revision,
                    local_dir=tmp,
                    force_download=True,
                )
            )
            if path.stat().st_size != row["bytes"] or sha(path) != row["sha256"]:
                raise ValueError("Private re-download bytes differ")


def canonical_lock():
    if sha(LOCK_RUNNER) != LOCK_SHA:
        raise ValueError("Canonical publication lock helper changed")
    spec = importlib.util.spec_from_file_location("original_lock", LOCK_RUNNER)
    assert spec is not None and spec.loader is not None
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    return helper.publication_lock(LOCK_RUNNER.parent.parent / "publication.lock")


def read_token(stream) -> str:
    value = json.load(stream)
    if (
        not isinstance(value, dict)
        or set(value) != {"token"}
        or not isinstance(value["token"], str)
        or not value["token"].startswith("hf_")
    ):
        raise ValueError("Exactly one private credential on stdin required")
    return value["token"]


def require_placement() -> None:
    if (
        os.uname().nodename != NODE
        or not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
        or os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_STEP_GPUS")
    ):
        raise ValueError("Owning-node one-CPU zero-GPU publication required")


def publish(
    epoch: int, root: Path = ROOT, *, stdin=None, api_factory=None, operation=None, lock=None
) -> dict:
    require_placement()
    token = read_token(sys.stdin if stdin is None else stdin)
    if api_factory is None or operation is None:
        from huggingface_hub import CommitOperationAdd, HfApi

        api_factory, operation = api_factory or HfApi, operation or CommitOperationAdd

    with canonical_lock() if lock is None else lock:
        prefix = f"{PREFIX}/epoch-{epoch:03d}"
        journal = root / "epoch-publication" / f"epoch-{epoch:03d}"
        if journal.exists():
            raise ValueError("Existing publication intent requires read-only reconciliation")
        manifest, provenance, members = source_files(epoch, root)
        download = sum(p.stat().st_size for p in members.values()) + 2 * SMALL_OBJECT_BOUND
        if shutil.disk_usage(root).free < FLOOR + download:
            raise ValueError("Publication scratch/reserve funding exhausted")
        api = api_factory(token=token)
        info = api.repo_info(REPO, repo_type="dataset")
        if info.private is not True:
            raise ValueError("Final checkpoints require the private dataset")
        paths = [f"{prefix}/{name}" for name in MEMBERS]
        if api.get_paths_info(REPO, [*paths, prefix], repo_type="dataset", revision=info.sha):
            raise ValueError("Existing remote path requires read-only reconciliation; never repeat")
        journal.mkdir(mode=0o700, parents=True)
        save(journal / "source-and-input-provenance.json", provenance)
        save(journal / "manifest.json", manifest)
        members = {
            "checkpoint.pt": members["checkpoint.pt"],
            "snapshot.json": members["snapshot.json"],
            "source-and-input-provenance.json": journal / "source-and-input-provenance.json",
            "DINOv3-LICENSE.md": members["DINOv3-LICENSE.md"],
            "manifest.json": journal / "manifest.json",
        }
        expected = {n: {"bytes": p.stat().st_size, "sha256": sha(p)} for n, p in members.items()}
        if source_files(epoch, root)[0] != manifest:
            raise ValueError("Completed snapshot changed during publication preparation")
        save(
            journal / "intent.json",
            {
                "epoch": epoch,
                "prefix": prefix,
                "parent_commit": info.sha,
                "files": expected,
                "automatic_retry": False,
                "checkpoint_sha256": manifest["checkpoint_sha256"],
            },
        )
        try:
            commit = api.create_commit(
                REPO,
                repo_type="dataset",
                parent_commit=info.sha,
                commit_message=f"Preserve final public-source all-development epoch {epoch}",
                operations=[
                    operation(path_in_repo=f"{prefix}/{n}", path_or_fileobj=str(p))
                    for n, p in members.items()
                ],
                num_threads=1,
            )
        except BaseException as error:
            save(
                journal / "uncertain.json",
                {"error_type": type(error).__name__, "automatic_retry": False},
            )
            raise RuntimeError("Publication mutation uncertain; reconcile read-only") from None
        save(journal / "commit.json", {"revision": commit.oid, "prefix": prefix})
        verify_private_revision(api, prefix, expected, commit.oid, root)
        receipt = {
            "status": "private_final_public_source_epoch_full_download_size_SHA_verified",
            "job_id": os.environ["SLURM_JOB_ID"],
            "epoch": epoch,
            "prefix": prefix,
            "revision": commit.oid,
            "files": expected,
            "file_count": len(expected),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
        }
        save(journal / "verified.json", receipt)
        return receipt


def reconcile(epoch: int, root: Path = ROOT, *, stdin=None, api_factory=None) -> dict:
    """Read-only: never writes the journal, never mutates the Hub, never retries."""
    token = read_token(sys.stdin if stdin is None else stdin)
    if api_factory is None:
        from huggingface_hub import HfApi

        api_factory = HfApi
    journal = root / "epoch-publication" / f"epoch-{epoch:03d}"
    intent = load(journal / "intent.json", root)
    prefix = f"{PREFIX}/epoch-{epoch:03d}"
    if intent.get("prefix") != prefix or intent.get("automatic_retry") is not False:
        raise ValueError("Original intent differs")
    api = api_factory(token=token)
    if not (journal / "commit.json").exists():
        head = api.repo_info(REPO, repo_type="dataset").sha
        present = sorted(remote_paths(api, [f"{prefix}/{n}" for n in MEMBERS], head))
        return {"status": "uncertain_requires_owner_review", "revision": head, "present": present}
    commit = load(journal / "commit.json", root)
    verify_private_revision(api, prefix, intent["files"], commit["revision"], root)
    return {
        "status": "existing_publication_read_only_verified",
        "revision": commit["revision"],
        "file_count": len(intent["files"]),
    }


def main() -> None:
    os.umask(0o077)
    if len(sys.argv) not in (2, 3) or (len(sys.argv) == 3 and sys.argv[1] != "reconcile"):
        raise SystemExit("usage: publish.py [reconcile] EPOCH")
    try:
        if len(sys.argv) == 3:
            result = reconcile(int(sys.argv[2]))
        else:
            result = publish(int(sys.argv[1]))
    except Exception as error:
        message = str(error) if isinstance(error, (ValueError, RuntimeError)) else ""
        print(
            json.dumps(
                {
                    "status": "blocked",
                    "error_type": type(error).__name__,
                    "error": re.sub(r"hf_[A-Za-z0-9]+", "[redacted]", message),
                }
            ),
            file=sys.stderr,
        )
        raise SystemExit(1) from None
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
