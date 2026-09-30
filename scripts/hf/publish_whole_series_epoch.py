"""Publish one immutable completed whole-series adapter-A32 epoch to the private HF dataset.

The access token arrives only as JSON on stdin. Upload intents and receipts have
no credentials; an uncertain upload is reconciled read-only before any retry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import CommitOperationAdd, HfApi

REPO = "GodlyDonuts/RSNA-Data"
TRAINING_GIT = "c9fcc7eefb09103942be89bf4fa87410eb9ec5ab"
VARIANTS = {
    "control": {"context": "none"},
    "context": {"context": "neighbor3"},
}
LICENSE_SHA = "25d122eb8f5b880fd23c736fb6ea8018ee45c12237e00b8a86d14c653904999e"
LICENSE_SOURCE = (
    "https://github.com/facebookresearch/dinov3/blob/"
    "ffb4bb89c6558ca3244655c25a3955d01788b732/LICENSE.md"
)


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


def source_files(
    snapshot: Path, run: Path, license_file: Path, code_root: Path, variant: str = "control"
) -> tuple[dict, dict]:
    spec = VARIANTS[variant]
    fold = 0
    meta = json.loads(snapshot.read_text())
    epoch = meta.get("epoch")
    if type(epoch) is not int or epoch not in range(1, 11):
        raise ValueError("Exactly one completed primary epoch is required")
    status = meta.get("status")
    expected = "scheduled_epoch_snapshot"
    if status != expected:
        raise ValueError("Epoch archive producer/status differs")
    if Path(meta["checkpoint_file"]).name != meta["checkpoint_file"]:
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
        or meta.get("diagnostic_only", False) is not False
    ):
        raise ValueError("Epoch checkpoint or provenance differs")
    run_record = json.loads((run / "run.json").read_text())
    binding = run_record["binding"]
    current_code = {
        str(p.relative_to(code_root / "rsna")): sha(p)
        for p in sorted((code_root / "rsna").rglob("*.py"))
    }
    git_head = subprocess.check_output(["git", "-C", str(code_root), "rev-parse", "HEAD"])
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
        or git_head.decode().strip() != TRAINING_GIT
        or binding.get("code_sha256") != current_code
        or run_record.get("training_studies") != 2247
        or binding["config"].get("fit_scope") != "outer_fold"
        or run_record.get("outer_studies") != 1156
        or binding["config"].get("checkpoint_format") != "frozen_base_delta"
        or binding["config"].get("slice_sampling") != "all"
        or binding["config"].get("series_sampling") != "all"
        or binding["config"].get("slice_context") != spec["context"]
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
        "fit_scope": "outer_fold",
        "variant": variant,
        "checkpoint_schema_version": 2,
        "checkpoint_format": "frozen_base_delta",
        "requires_pinned_original_weights": True,
        "slice_sampling": "all",
        "series_sampling": "all",
        "slice_context": spec["context"],
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


def run(args: argparse.Namespace) -> dict:
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURM_JOB_GPUS"):
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
    variant = args.variant
    manifest, files = source_files(
        args.snapshot, args.run, args.license_file, args.code_root, variant
    )
    epoch = manifest["epoch"]
    prefix = f"experiments/dinov3-v1/training-epochs/whole-series-{variant}-a32-fold0-c9fcc7e-seed20260922/epoch-{epoch:03d}"
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
            commit_message=f"Preserve verified DINOv3-L adapter-A32 whole-series {variant} epoch {epoch}",
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("publish", "reconcile"))
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
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
