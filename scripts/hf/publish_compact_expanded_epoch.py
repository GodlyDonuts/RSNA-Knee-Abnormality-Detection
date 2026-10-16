"""Preserve one original expanded2592 compact epoch privately, with full download verification.

CPU only; token on stdin. Existing intents require read-only reconciliation.
Uses the unchanged, hash-pinned canonical lock and Hub verification machinery.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

REPO = "GodlyDonuts/RSNA-Data"
PREFIX = "experiments/compact-expanded2592/fold0-a1ca2385-seed20260922"
SOURCE_GIT = "a1ca238526dad75b2a5cea46edba44fd95c71938"
CORE = Path(
    "/home/sa305415/rsna-knee/production-launch-native384-a32-c9-v1/publish_native384_epoch.py"
)
CORE_SHA = "4e327e271816e4a4055b4f3ad28a8dab59fc650a1367fd13bbd7fe06d9a5efde"
ASSETS = Path("/tmp/rsna-compact-native-n3-assets-20261016-v1")
ROOT = Path("/tmp/rsna-compact-expanded2592-production-20261016-v1")
INPUT = Path("/tmp/rsna-compact-expanded2592-inputs-20261016-v1/inputs/accepted.json")
INPUT_SHA = "235538195accbac10369779324f10c3fbd7eea42981c95a7164144a17591c3de"
RECOVERY = Path("/tmp/rsna-compact-expanded2592-engine-20261016-v1/recovery-v1/accepted.json")
FUNDING = Path(
    "/tmp/rsna-compact-expanded2592-engine-20261016-v1/production-control-v1/funding.json"
)
SOURCE = Path("/tmp/rsna-compact-expanded2592-engine-20261016-v1/code")
POPULATION_SHA = "90f4229b0617a18a32e754b80f483cac30744046a789376062a67199021e8c7c"
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
        or snapshot.get("optimizer_steps") != 324 * epoch
        or snapshot.get("schema_version") != 1
        or b != run.get("binding")
        or run.get("diagnostic_only") is not False
        or b.get("diagnostic_only") is not False
        or b.get("engine") != "native_coatnet_bag12_N3_full_v1"
        or b.get("input_policy") is not None
        or b.get("loss_policy") is not None
        or b.get("fitting_population", {}).get("version") != "original_N3_plus345_fitting_only_v1"
        or hashlib.sha256(
            json.dumps(
                b.get("fitting_population", {}).get("population"),
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        != POPULATION_SHA
        or b.get("fitting_population", {}).get("CPU_input_sha256") != INPUT_SHA
        or b.get("fitting_population", {}).get("metadata_union_sha256")
        != "1887a3e3e0c1f3422a61e66d2767b06ebe1af6756a8ee047f2890412dc164288"
        or b.get("fitting_population", {}).get("sixteen_passes_updates") != 5184
        or b.get("fitting_population", {}).get("compute_matched_to_original") is not False
        or b.get("fitting_count") != 2592
        or b.get("total_steps") != 5184
        or b.get("settings", {}).get("epochs") != 16
        or b.get("settings", {}).get("effective_batch") != 8
        or b.get("initialization_sha256")
        != "91570c7af10fea0af03c1b0a49be3e4775596239890eb4193c75129b3045623e"
        or b.get("initialization_revision") != "2c31d82376cc244c907c064332941e6c65be81f1"
        or b.get("supervision", {}).get("overlay_sha256")
        != "4ca82ceb772e0cc59e657b6281a3c6ee49e2f4803d77148828054607ac51b8c7"
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
    funded = json.loads(FUNDING.read_text())
    recovery_sha = funded.get("actual_recovery_sha256", "")
    if len(recovery_sha) != 64 or sha(INPUT) != INPUT_SHA or sha(RECOVERY) != recovery_sha:
        raise ValueError("Original actual expanded input/recovery bytes changed")
    admitted = json.loads(INPUT.read_text())
    recovered = json.loads(RECOVERY.read_text())
    if (
        funded.get("source_git") != SOURCE_GIT
        or recovered.get("status") != "actual_expanded2592_H100_recovery_accepted"
        or recovered.get("job_id") != "933382"
        or recovered.get("source_git") != SOURCE_GIT
        or recovered.get("input_admission_sha256") != INPUT_SHA
        or recovered.get("full_optimizer_updates") != 5184
        or recovered.get("initial_head_and_official_encoder_match_original_N3") is not True
        or recovered.get("comparison", {}).get(
            "complete_model_BN_AdamW_OneCycle_counters_sampler_allRNG_history_equal"
        )
        is not True
        or recovered.get("comparison", {}).get("exact_final16x12_CSV_equal") is not True
    ):
        raise ValueError("Original actual expanded recovery and matched initialization required")
    active = json.loads((ROOT / "active-job.json").read_text())["job_id"]
    header = json.loads((ROOT / "invocations" / active / "start.json").read_text())
    if header["actual_recovery_sha256"] != recovery_sha or header["funding_sha256"] != sha(FUNDING):
        raise ValueError("Epoch production invocation differs from its original funding/recovery")
    b = snapshot["binding"]
    original = json.loads(
        Path(
            "/tmp/rsna-compact-native-n3-production-evc33-20261016-v1/production/run.json"
        ).read_text()
    )["binding"]
    unchanged = (
        "release",
        "outer_uids_sha256",
        "folds_file_sha256",
        "native_input",
        "targets",
        "architecture",
        "timm_version",
        "initialization_sha256",
        "initialization_revision",
        "initial_head_sha256",
        "settings",
    )
    if any(b.get(k) != original.get(k) for k in unchanged):
        raise ValueError("Expanded epoch changed original endpoint/input/model/recipe")
    if b["fitting_population"]["population"] != admitted["population"] or not b.get(
        "source_sha256"
    ):
        raise ValueError("Original admitted expanded source membership required")
    for name, digest in b["source_sha256"].items():
        path = (SOURCE / name).resolve()
        if not path.is_relative_to(SOURCE.resolve()) or sha(path) != digest:
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
        "production-funding.json": FUNDING,
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
                commit_message=f"Preserve compact expanded2592 fixed16 full-model epoch {epoch}",
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
