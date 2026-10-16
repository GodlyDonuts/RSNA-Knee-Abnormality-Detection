"""Preserve one original compact N3 epoch privately, with full download verification.

CPU only; token on stdin. Existing intents require read-only reconciliation.
Uses the unchanged, hash-pinned canonical lock and Hub verification machinery.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = "GodlyDonuts/RSNA-Data"
PREFIX = "experiments/compact-native-n3/fold0-dcfff319-seed20260922"
SOURCE_GIT = "dcfff319cb7bd7bbae679a20da2b1b9268e8d680"
CORE = Path(
    "/home/sa305415/rsna-knee/production-launch-native384-a32-c9-v1/publish_native384_epoch.py"
)
CORE_SHA = "4e327e271816e4a4055b4f3ad28a8dab59fc650a1367fd13bbd7fe06d9a5efde"
ASSETS = Path("/tmp/rsna-compact-native-n3-assets-20261016-v1")
ROOT = Path("/tmp/rsna-compact-native-n3-production-evc33-20261016-v1")
INPUT = Path("/tmp/rsna-compact-target-admission-evc33-20261016-v3/accepted.json")
INPUT_SHA = "0482c3e43def44388c371348e2602650d251f9b4ba6f0d903c87520f3a3b4c88"
RECOVERY = Path("/tmp/rsna-compact-real-recovery-evc33-20261016-v1/accepted.json")
RECOVERY_SHA = "b6d0a90d8b50ee95f8ce13497b7103d2ca1f88be9a939f71303d5b0deb67c2d5"
LICENSE_SHA = "71b111620fa32c17a80ccd6db2da759d31a591e497d8b5f382174f59d1b59d49"
CARD_SHA = "e13c01cc31843c8345937ccc528504033bf83c0ce6aae92c2b26662f428bae2f"


def sha(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def core():
    if CORE.is_symlink() or sha(CORE) != CORE_SHA:
        raise ValueError("Canonical publication dependency changed")
    spec = importlib.util.spec_from_file_location("compact_hf_core", CORE)
    if spec is None or spec.loader is None:
        raise ValueError("Canonical publication module cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def validate_snapshot(snapshot: dict, run: dict, epoch: int) -> None:
    b = snapshot.get("binding", {})
    if (
        type(epoch) is not int
        or epoch not in range(1, 17)
        or snapshot.get("epoch") != epoch
        or snapshot.get("optimizer_steps") != 281 * epoch
        or snapshot.get("schema_version") != 1
        or b != run.get("binding")
        or run.get("diagnostic_only") is not False
        or b.get("diagnostic_only") is not False
        or b.get("engine") != "native_coatnet_bag12_N3_full_v1"
        or b.get("fitting_count") != 2247
        or b.get("total_steps") != 4496
        or b.get("settings", {}).get("epochs") != 16
        or b.get("settings", {}).get("effective_batch") != 8
        or b.get("initialization_sha256")
        != "91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e"
        or b.get("initialization_revision") != "2c31d82376cc244c907c064332941e6c65be81f1"
        or b.get("supervision", {}).get("overlay_sha256")
        != "c8fd2e860d8a3913dbeab34a9d1078316a8f893360124eac088453dc2ed6cf51"
    ):
        raise ValueError("Original non-diagnostic fixed16 epoch/run binding required")
    if not 250 * 2**20 <= snapshot.get("checkpoint_bytes", 0) <= 512 * 2**20:
        raise ValueError("Expected immutable complete compact model, not optimizer/delta")


def source_files(epoch: int) -> tuple[dict, dict[str, Path]]:
    if os.uname().nodename != "evc33":
        raise ValueError("Original owning node required")
    run_path = ROOT / "production/run.json"
    snapshot_path = ROOT / f"production/epoch-checkpoints/epoch-{epoch:03d}.json"
    checkpoint = snapshot_path.with_suffix(".pt")
    snapshot = json.loads(snapshot_path.read_text())
    validate_snapshot(snapshot, json.loads(run_path.read_text()), epoch)
    if sha(INPUT) != INPUT_SHA or sha(RECOVERY) != RECOVERY_SHA:
        raise ValueError("Original actual input/recovery receipt changed")
    admitted = json.loads(INPUT.read_text())
    b = snapshot["binding"]
    original = json.loads((INPUT.parent / "inputs/inputs.json").read_text())["binding"]
    if any(b.get(k) != v for k, v in original.items()):
        raise ValueError("Epoch differs from original CPU source/fold/roster/input admission")
    source = Path(admitted["source"])
    if admitted["source_git"] != SOURCE_GIT or not b.get("source_sha256"):
        raise ValueError("Original scientific source required")
    for name, digest in b["source_sha256"].items():
        path = (source / name).resolve()
        if not path.is_relative_to(source.resolve()) or sha(path) != digest:
            raise ValueError("Original scientific source member changed")
    if checkpoint.is_symlink() or checkpoint.stat().st_size != snapshot["checkpoint_bytes"]:
        raise ValueError("Original complete checkpoint bytes required")
    if sha(checkpoint) != snapshot["checkpoint_sha256"]:
        raise ValueError("Checkpoint differs from completed epoch sidecar")
    license_file = ASSETS / "runtime-overlay/timm-1.0.30.dist-info/licenses/LICENSE"
    card = ASSETS / "initialization/README.md"
    if sha(license_file) != LICENSE_SHA or sha(card) != CARD_SHA:
        raise ValueError("Original Apache license/model card changed")
    files = {
        "checkpoint.pt": checkpoint,
        "snapshot.json": snapshot_path,
        "run.json": run_path,
        "input-admission.json": INPUT,
        "actual-GPU-recovery.json": RECOVERY,
        "Apache-2.0-LICENSE": license_file,
        "initialization-model-card.md": card,
    }
    manifest = {
        "epoch": epoch,
        "optimizer_steps": snapshot["optimizer_steps"],
        "training_source_git": SOURCE_GIT,
        "publisher_sha256": sha(Path(__file__)),
        "checkpoint_sha256": snapshot["checkpoint_sha256"],
        "license": "apache-2.0",
        "initialization_revision": b["initialization_revision"],
        "files": {name: {"bytes": p.stat().st_size, "sha256": sha(p)} for name, p in files.items()},
        "accuracy_claim": False,
    }
    return manifest, files


def publish(epoch: int, mode: str) -> dict:
    from huggingface_hub import CommitOperationAdd, HfApi

    if os.environ.get("SLURM_CPUS_PER_TASK") != "1" or os.environ.get("SLURM_JOB_GPUS"):
        raise ValueError("Exactly one CPU and zero GPU required")
    common = core()
    with common.shared_publication_lock():
        manifest, files = source_files(epoch)
        credential = json.load(sys.stdin)
        if set(credential) != {"token"} or not credential["token"].startswith("hf_"):
            raise ValueError("One credential on stdin required")
        api = HfApi(token=credential["token"])
        info = api.repo_info(REPO, repo_type="dataset")
        if info.private is not True:
            raise ValueError("Private repository required")
        prefix = f"{PREFIX}/epoch-{epoch:03d}"
        journal = ROOT / f"publication/epoch-{epoch:03d}"
        journal.mkdir(parents=True, mode=0o700, exist_ok=True)
        payload = {**files, "manifest.json": journal / "manifest.json"}
        paths = [f"{prefix}/{name}" for name in payload]
        intent, commit, verified = (
            journal / name for name in ("intent.json", "commit.json", "verified.json")
        )
        if mode == "publish":
            if any(p.exists() for p in [intent, commit, verified, payload["manifest.json"]]):
                raise ValueError("Existing attempt: reconcile read-only")
            if common.check_remote(api, paths):
                raise ValueError("Private epoch path already exists")
            common.atomic_new(payload["manifest.json"], manifest)
            common.atomic_new(
                intent,
                {
                    "prefix": prefix,
                    "manifest_sha256": sha(payload["manifest.json"]),
                    "parent": info.sha,
                },
            )
            result = api.create_commit(
                REPO,
                repo_type="dataset",
                parent_commit=info.sha,
                commit_message=f"Preserve compact native N3 full-model epoch {epoch}",
                operations=[
                    CommitOperationAdd(path_in_repo=f"{prefix}/{name}", path_or_fileobj=p)
                    for name, p in payload.items()
                ],
            )
            common.atomic_new(commit, {"revision": result.oid})
        elif mode != "reconcile":
            raise ValueError("Unknown publication mode")
        original = json.loads(intent.read_text())
        if original["prefix"] != prefix or original["manifest_sha256"] != sha(
            payload["manifest.json"]
        ):
            raise ValueError("Original intent changed")
        if json.loads(payload["manifest.json"].read_text()) != manifest:
            raise ValueError("Original manifest changed")
        revision = json.loads(commit.read_text())["revision"]
        if set(common.check_remote(api, paths, revision)) != set(paths):
            raise ValueError("Missing private payload objects")
        receipt = {
            "status": "private_epoch_fully_verified",
            "epoch": epoch,
            "revision": revision,
            "prefix": prefix,
            "checkpoint_sha256": manifest["checkpoint_sha256"],
            "manifest_sha256": sha(payload["manifest.json"]),
            "files_verified_by_download": len(payload),
        }
        if verified.exists():
            if json.loads(verified.read_text()) != receipt:
                raise ValueError("Prior verified receipt differs")
            return receipt
        common.verify_downloads(api, payload, prefix, revision)
        common.atomic_new(verified, receipt)
        return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("publish", "reconcile", "inspect"))
    parser.add_argument("epoch", type=int)
    args = parser.parse_args()
    if args.mode == "inspect":
        manifest, _ = source_files(args.epoch)
        print(json.dumps(manifest), flush=True)
    else:
        print(json.dumps(publish(args.epoch, args.mode)), flush=True)


if __name__ == "__main__":
    main()
