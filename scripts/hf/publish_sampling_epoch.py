"""Publish one immutable completed sampling rank-32 epoch to the private HF dataset.

The access token arrives only as JSON on stdin. Upload intents and receipts have
no credentials; an uncertain upload is reconciled read-only before any retry.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "GodlyDonuts/RSNA-Data"
TRAINING_GIT = "7ac7a9779d3c8bfd756a44eceb6414e9bb763b85"
PREFIX = "experiments/dinov3-v1/training-epochs/adapter-a32-sampling-fold0-7ac7a97-seed20260922"
RELEASE_SHA = "e2ecb283915a5b1810bfc9a2f6e1dda1f1d0d762f91cf2487140af3d05cb6c35"
LICENSE_SHA = "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e"
LICENSE_SOURCE = (
    "https://github.com/facebookresearch/dinov3/blob/"
    "ffb4bb89c6558ca3244655c25a3955d01788b732/LICENSE.md"
)


EXPECTED_CONFIG = {
    "accumulation_steps": 8,
    "augment": False,
    "batch_size": 1,
    "checkpoint_every_updates": 25,
    "checkpoint_format": "frozen_base_delta",
    "clip_grad_norm": 1.0,
    "diagnostic_study_limit": None,
    "encode_chunk_size": 16,
    "encoder_lr": 0.0001,
    "epochs": 10,
    "fit_scope": "outer_fold",
    "fold": 0,
    "freeze_encoder": True,
    "head_lr": 0.0003,
    "hidden_dim": 128,
    "lr_schedule": "cosine",
    "max_series": 6,
    "min_free_gib": 20.0,
    "pooling": "attention",
    "precision": "bf16",
    "seed": 20260922,
    "series_sampling": "selected",
    "side": 224,
    "slice_context": "none",
    "slice_sampling": "uniform",
    "slices": 16,
    "submission_every_epochs": 1,
    "warmup_updates": 141,
    "weight_decay": 0.0001,
    "workers": 0,
}
EXPECTED_ENCODER = {
    "architecture": "dinov3_l16",
    "config_sha256": "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5",
    "embedding_dim": 1024,
    "input_contract": "gray_0_1_imagenet_rgb_v1",
    "license": "other",
    "model_class": "dinov3_vit",
    "pooling": "normalized_cls",
    "pretrained": True,
    "pretrained_state_key_mapping": "dinov3_official_layer_to_model_layer_transformers_5_6_1_v1",
    "source_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "source_revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
    "state_dict_format": "safetensors",
    "transformers_version": "5.6.1",
    "weights_sha256": "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
}


EXPECTED_BINDING_FIELDS = {
    "cache_side": 256,
    "cache_version": "series-u16-v3",
    "device_type": "cuda",
    "folds_sha256": "8699248177042dc2e95b1e1947e25cbe5bc39e42c43ee4919398afc219665e07",
    "outer_uids_sha256": "2f365bb0ce8e052f22dcab78bb2bbb7e03c0919ab28ba94ab9d3d8d6e08f0fdb",
    "source_revision": "0b5e410288232f570c69ba4f44723356e6ae6ab7",
    "targets": [
        "ACL",
        "MCL",
        "Medial Meniscus",
        "Lateral Meniscus",
        "Medial OA",
        "Lateral OA",
        "PF OA",
        "Effusion",
        "Synovitis",
        "Baker's",
        "Contusion",
        "Fracture",
    ],
    "torch_version": "2.14.0+cu130",
    "training_uids_sha256": "e9d38ee21a114973177b89f4602129c7dc047927b8499991d3be30fd89737f3a",
    "world_size": 1,
}


def canonical(value: dict) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def expected_training_config(policy="uniform") -> dict:
    names = []
    for layer in range(24):
        names.extend(f"model.layer.{layer}.attention.{part}_proj" for part in ("q", "k", "v", "o"))
        if layer >= 12:
            names.extend(f"model.layer.{layer}.mlp.{part}_proj" for part in ("up", "down"))
    adapter = {
        "method": "rs_lora_v1",
        "rank": 32,
        "alpha": 8.0,
        "dropout": 0.05,
        "ffn_last_layers": 12,
        "gradient_checkpointing": True,
        "scaling": "alpha/sqrt(rank)",
        "base_frozen": True,
        "zero_initial_residual": True,
        "target_modules": names,
        "trainable_parameters": 10_223_616,
    }
    return {
        **EXPECTED_CONFIG,
        "slice_sampling": policy,
        "encoder": {**EXPECTED_ENCODER, "adapter": adapter},
    }


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for part in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(part)
    return h.hexdigest()


def atomic_new(path: Path, record: dict) -> None:
    with path.open("x") as stream:
        json.dump(record, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


LOCK_RUNNER = Path("/home/sa305415/rsna-knee/native-full-cache-two-lanes-v1/common/runner.py")
LOCK_RUNNER_SHA = "3575a18ecd7a13a2becc9348c48a9ebda95f8934069e67dd1912737e6194118c"
PUBLICATION_LOCK = LOCK_RUNNER.parent.parent / "publication.lock"
SOURCE_MANIFEST_SHA = "6ede8415384bbe37d4e32e8458fb92586b2c7abd66dfe9b0e9bcd12e12bb81be"


def load_pinned(path, expected_sha, name):
    if path.is_symlink() or sha(path) != expected_sha:
        raise ValueError("Pinned sampling publication helper changed")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


CPU_PROOF_SHA = "6b51abdfca83906149867d7aa2b7041180a8d11e5c855f78fa64afd60723f3b4"
PILOT_PROOF_SHA = "6c8594df5320fde080a6a2a030b92342cf1d717c1fef1842b98e6e73a3e733f9"
INITIAL_STATE_SHA = "c5e808b1ab410224b662b84d72964418d4691c0bc2447877392a49bfca9f92e5"
INITIAL_LOGITS_SHA = "d91b89309fbbac489a1f6d8c54611a69386dc051b23b4f27bcd28db2248e7476"


def sampling_provenance(run, control):
    path = run / "sampling-input-provenance.json"
    source = control / "source.json"
    if (
        path.is_symlink()
        or not path.is_file()
        or source.is_symlink()
        or sha(source) != SOURCE_MANIFEST_SHA
    ):
        raise ValueError("Pinned full sampling source and run provenance are required")
    value = json.loads(path.read_text())
    expected_source = json.loads(source.read_text())
    policy = value.get("policy")
    if (
        value.get("status") != "full_sampling_inputs_and_actual_initialization_accepted"
        or value.get("source_git") != TRAINING_GIT
        or expected_source.get("source_git") != TRAINING_GIT
        or value.get("source_files") != expected_source.get("files")
        or policy not in ("uniform", "stratified")
        or value.get("source_manifest_sha256") != SOURCE_MANIFEST_SHA
        or value.get("accepted_data_masks_or_folds_changed") is not False
        or value.get("base_weights_reused_not_diagnostic_checkpoint") is not True
        or value.get("strict_determinism") is not True
        or value.get("cache_side") != 256
        or value.get("cache_members") != 48772
        or value.get("training_studies") != 2247
        or value.get("outer_studies") != 1156
        or value.get("checkpoint_cap_bytes") != 128 * 1024**2
        or value.get("initialization", {}).get("status")
        != "same_initial_learned_tensors_and_zero_adapter_logits_verified"
    ):
        raise ValueError("Sampling production provenance differs")
    if (
        value.get("CPU_proof_sha256") != CPU_PROOF_SHA
        or value.get("pilot_proof_sha256") != PILOT_PROOF_SHA
        or hashlib.sha256(canonical(value["initialization"]["initial_state"]).encode()).hexdigest()
        != INITIAL_STATE_SHA
        or value["initialization"].get("initial_logits_sha256") != INITIAL_LOGITS_SHA
        or value.get("config_sha256") != sha(control / (policy + "-config.json"))
        or value.get("payload_manifest_sha256") != sha(control / "files.json")
    ):
        raise ValueError("Original pilot, input acceptance, initialization or launch differs")
    payload = json.loads((control / "files.json").read_text())
    if payload["runs"][policy] != str(run):
        raise ValueError("Publication role does not match the unique launch output")
    for name, row in payload["files"].items():
        file = control / name
        if file.is_symlink() or file.stat().st_size != row["bytes"] or sha(file) != row["sha256"]:
            raise ValueError("Pinned launch payload changed before publication")
    return value, path


def shared_publication_lock():
    module = load_pinned(LOCK_RUNNER, LOCK_RUNNER_SHA, "sampling_epoch_lock")
    return module.publication_lock(PUBLICATION_LOCK)


def source_files(
    snapshot: Path, run: Path, license_file: Path, code_root: Path
) -> tuple[dict, dict]:
    fold = 0
    meta = json.loads(snapshot.read_text())
    epoch = meta.get("epoch")
    if type(epoch) is not int or epoch not in range(1, 11):
        raise ValueError("Exactly one completed primary epoch is required")
    status = meta.get("status")
    expected = "scheduled_epoch_snapshot"
    if status != expected:
        raise ValueError("Epoch archive producer/status differs")
    if (
        snapshot.parent.resolve() != (run / "epoch-checkpoints").resolve()
        or meta["checkpoint_file"] != f"epoch-{epoch:03d}.pt"
    ):
        raise ValueError("Epoch checkpoint path is not a file name")
    checkpoint = snapshot.parent / meta["checkpoint_file"]
    if (
        checkpoint.is_symlink()
        or not checkpoint.is_file()
        or checkpoint.stat().st_size != meta.get("checkpoint_bytes")
        or sha(checkpoint) != meta.get("checkpoint_sha256")
        or meta.get("optimizer_steps") != epoch * 281
        or meta.get("fold") != fold
        or meta.get("seed") != 20260922
        or meta.get("diagnostic_only") is not False
        or meta.get("submission_eligible") is not True
        or meta.get("submission_every_epochs") != 1
    ):
        raise ValueError("Epoch checkpoint or provenance differs")
    run_record = json.loads((run / "run.json").read_text())
    binding = run_record["binding"]
    provenance, provenance_path = sampling_provenance(run, code_root)
    policy = provenance["policy"]
    current_code = {
        name.removeprefix("rsna/"): row["sha256"]
        for name, row in provenance["source_files"].items()
        if name.startswith("rsna/") and name.endswith(".py")
    }
    binding_sha = hashlib.sha256(
        json.dumps(binding, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()
    if (
        binding_sha != meta.get("binding_sha256")
        or binding["config"].get("submission_every_epochs") != 1
        or binding["config"].get("pooling") != "attention"
        or binding["config"].get("diagnostic_study_limit") is not None
        or binding["config"].get("seed") != 20260922
        or binding["config"].get("fold") != fold
        or binding["world_size"] != 1
        or binding.get("code_sha256") != current_code
        or run_record.get("training_studies") != 2247
        or binding["config"].get("fit_scope") != "outer_fold"
        or run_record.get("outer_studies") != 1156
        or binding["config"].get("checkpoint_format") != "frozen_base_delta"
        or binding["config"].get("slice_sampling") != policy
        or binding["config"].get("series_sampling") != "selected"
        or binding["config"].get("slice_context") != "none"
        or canonical(binding["config"]) != canonical(expected_training_config(policy))
        or canonical({key: binding.get(key) for key in EXPECTED_BINDING_FIELDS})
        != canonical(EXPECTED_BINDING_FIELDS)
        or binding.get("cache_side") != 256
        or binding.get("cache_version") != "series-u16-v3"
        or binding["release"]["binding_sha256"] != RELEASE_SHA
        or checkpoint.stat().st_size >= 128 * 1024**2
        or binding["release"]["binding_sha256"] != meta.get("release_binding_sha256")
        or sha(license_file) != LICENSE_SHA
    ):
        raise ValueError("Epoch run binding or official license differs")
    config = binding["config"]
    adapter = config["encoder"].get("adapter", {})
    if (
        adapter.get("method") != "rs_lora_v1"
        or adapter.get("rank") != 32
        or adapter.get("trainable_parameters") != 10_223_616
        or adapter.get("base_frozen") is not True
        or len(adapter.get("target_modules", [])) != 120
        or config.get("freeze_encoder") is not True
        or config.get("precision") != "bf16"
        or config.get("lr_schedule") != "cosine"
        or config.get("epochs") != 10
        or config.get("encoder_lr") != 0.0001
        or config.get("head_lr") != 0.0003
        or run_record.get("encoder_trainable_parameters") != 10_223_616
        or config["encoder"].get("weights_sha256")
        != "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179"
        or config["encoder"].get("config_sha256")
        != "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5"
    ):
        raise ValueError("Adapter recipe or actual trainable count differs")
    files = {
        "sampling-input-provenance.json": provenance_path,
        "checkpoint.pt": checkpoint,
        "snapshot.json": snapshot,
        "DINOv3-LICENSE.md": license_file,
    }
    manifest = {
        "schema_version": 1,
        "status": "completed_training_epoch",
        "repo_id": REPO,
        "training_git_commit": TRAINING_GIT,
        "epoch": epoch,
        "fold": fold,
        "seed": 20260922,
        "optimizer_steps": meta["optimizer_steps"],
        "checkpoint_sha256": meta["checkpoint_sha256"],
        "checkpoint_bytes": meta["checkpoint_bytes"],
        "snapshot_status": status,
        "pooling": "attention",
        "adapter": adapter,
        "binding_sha256": binding_sha,
        "release_binding_sha256": binding["release"]["binding_sha256"],
        "encoder": {
            k: binding["config"]["encoder"][k]
            for k in ("architecture", "source_id", "source_revision", "weights_sha256")
        },
        "license_source": LICENSE_SOURCE,
        "license_sha256": LICENSE_SHA,
        "files": {
            name: {"sha256": sha(path), "bytes": path.stat().st_size}
            for name, path in files.items()
        },
        "competition_score_claimed": False,
        "sampling_input_provenance": provenance,
        "sampling_input_provenance_sha256": sha(provenance_path),
        "fit_scope": "outer_fold",
        "experiment": "matched_sampling_rank32",
        "checkpoint_schema_version": 2,
        "checkpoint_format": "frozen_base_delta",
        "requires_pinned_original_weights": True,
        "slice_sampling": policy,
        "series_sampling": "selected",
        "slice_context": "none",
        "internal_evaluation_policy": "fixed_final_only; no_internal_epoch_selection",
    }
    return manifest, files


def check_remote(api: HfApi, paths: list[str], revision: str | None = None) -> list[str]:
    return [
        item.path
        for item in api.get_paths_info(REPO, paths=paths, revision=revision, repo_type="dataset")
    ]


def verify_downloads(api: HfApi, source: dict[str, Path], prefix: str, revision: str) -> None:
    if shutil.disk_usage("/tmp").free < 20 * 1024**3 + sum(
        p.stat().st_size for p in source.values()
    ):
        raise ValueError("Node-local space is below the download verification floor")
    with tempfile.TemporaryDirectory(prefix="rsna-hf-epoch-", dir="/tmp") as temp:
        for name, original in source.items():
            actual = Path(
                api.hf_hub_download(
                    REPO,
                    f"{prefix}/{name}",
                    repo_type="dataset",
                    revision=revision,
                    local_dir=temp,
                )
            )
            if actual.stat().st_size != original.stat().st_size or sha(actual) != sha(original):
                raise ValueError("Private Hub download differs from source bytes")


def _run_unlocked(args: argparse.Namespace) -> dict:
    if (
        not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
    ):
        raise ValueError("Private checkpoint upload requires a CPU-only Slurm allocation")
    value = json.load(sys.stdin)
    if (
        set(value) != {"token"}
        or not isinstance(value["token"], str)
        or not value["token"].startswith("hf_")
    ):
        raise ValueError("Exactly one Hugging Face token is required on stdin")
    api = HfApi(token=value["token"])
    info = api.repo_info(REPO, repo_type="dataset")
    if info.private is not True:
        raise ValueError("Training checkpoints may only enter the private dataset")
    manifest, files = source_files(args.snapshot, args.run, args.license_file, args.code_root)
    epoch = manifest["epoch"]
    prefix = f"{PREFIX}/{manifest['slice_sampling']}/epoch-{epoch:03d}"
    journal = args.journal / f"epoch-{epoch:03d}"
    journal.mkdir(mode=0o700, parents=True, exist_ok=True)
    manifest_path = journal / "manifest.json"
    expected = {**files, "manifest.json": manifest_path}
    paths = [f"{prefix}/{name}" for name in expected]
    intent_path = journal / "intent.json"
    commit_path = journal / "commit-result.json"
    receipt_path = journal / "verified.json"
    if args.mode == "publish":
        if any(path.exists() for path in (manifest_path, intent_path, commit_path, receipt_path)):
            raise ValueError("Epoch publication was already attempted; reconcile read-only")
        if check_remote(api, paths):
            raise ValueError("Dedicated private epoch path already has files")
        atomic_new(manifest_path, manifest)
        atomic_new(
            intent_path,
            {
                "schema_version": 1,
                "status": "upload_intent",
                "repo_id": REPO,
                "prefix": prefix,
                "parent_commit": info.sha,
                "manifest_sha256": sha(manifest_path),
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "files": {
                    name: {"sha256": sha(path), "bytes": path.stat().st_size}
                    for name, path in expected.items()
                },
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        commit = api.create_commit(
            REPO,
            repo_type="dataset",
            parent_commit=info.sha,
            commit_message=f"Preserve verified DINOv3-L sampling adapter rank-32 {manifest['slice_sampling']} epoch {epoch}",
            operations=[
                CommitOperationAdd(path_in_repo=f"{prefix}/{name}", path_or_fileobj=path)
                for name, path in expected.items()
            ],
        )
        atomic_new(
            commit_path,
            {
                "schema_version": 1,
                "status": "commit_returned",
                "revision": commit.oid,
                "prefix": prefix,
            },
        )
    elif args.mode == "reconcile":
        if not intent_path.is_file() or not manifest_path.is_file():
            raise ValueError("No original epoch publication intent exists")
        original = json.loads(intent_path.read_text())
        if (
            original.get("manifest_sha256") != sha(manifest_path)
            or original.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
            or original.get("prefix") != prefix
        ):
            raise ValueError("Original upload intent differs from current source")
        if not commit_path.is_file():
            raise ValueError("Upload result uncertain; inspect remote path before any retry")
    else:
        raise ValueError("Unsupported publication mode")
    commit = json.loads(commit_path.read_text())
    revision = commit["revision"]
    if set(check_remote(api, paths, revision)) != set(paths):
        raise ValueError("Private commit is missing an epoch payload member")
    if receipt_path.exists():
        prior = json.loads(receipt_path.read_text())
        if (
            prior.get("status") != "private_epoch_fully_verified"
            or prior.get("revision") != revision
            or prior.get("checkpoint_sha256") != manifest["checkpoint_sha256"]
            or prior.get("manifest_sha256") != sha(manifest_path)
        ):
            raise ValueError("Previously verified epoch receipt differs")
        return prior
    verify_downloads(api, expected, prefix, revision)
    receipt = {
        "schema_version": 1,
        "status": "private_epoch_fully_verified",
        "repo_id": REPO,
        "revision": revision,
        "prefix": prefix,
        "epoch": epoch,
        "checkpoint_sha256": manifest["checkpoint_sha256"],
        "manifest_sha256": sha(manifest_path),
        "files_verified_by_download": len(expected),
        "verified_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    atomic_new(receipt_path, receipt)
    return receipt


def run(args: argparse.Namespace) -> dict:
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("Sampling private publication requires one CPU and zero GPUs")
    with shared_publication_lock():
        return _run_unlocked(args)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("publish", "reconcile"))
    for name in ("snapshot", "run", "license-file", "journal", "code-root"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        print(json.dumps(run(args), sort_keys=True), flush=True)
    except Exception as exc:
        print(json.dumps({"status": "blocked", "error_type": type(exc).__name__}), file=sys.stderr)
        raise SystemExit(1) from None


if __name__ == "__main__":
    main()
