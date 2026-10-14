"""Write-once private preservation of one completed dense-suffix comparison epoch.

PREPARED ONLY; root stages it as ``ROOT/publish.py``. Every launch precondition
(actual CPU and GPU recovery receipts, explicit production funding, exact source,
configs, folds and official base) is re-admitted through the staged launcher. The
pinned original publication transport supplies the canonical lock, stdin token,
private-only destination and complete forced re-download/size/SHA verification.
Any existing journal or remote path, or an uncertain mutation, is never retried;
``reconcile`` is read-only. Fixed scheduled epochs only: no selection or metrics.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
import shutil
import sys
from pathlib import Path

ROOT = Path("/tmp/rsna-dense-suffix-fit-20261014-v1")
ADMISSION = Path("/tmp/rsna-dense-suffix-admission-20261014-v1")
TRANSPORT_SHA = "0f929dc1131c61f471c2e28c6a99367ecdc9fe31c45b52a413710643a054c7da"
PREFIX = (
    "experiments/dinov3-v1/training-epochs/"
    "dense-suffix-capacity-fold0-0ecece12-fixed10-seed20260922"
)
SIBLING_PREFIXES = (
    "experiments/dinov3-v1/training-epochs/"
    "source-slot-publictriplet-fold0-4fedbec3-fixed10-seed20260922",
)
SMALL_OBJECT_BOUND = 2**20


def _load(path: Path, name: str, expected: str | None = None):
    path = Path(path)
    if expected is not None and hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Pinned staged {name} changed")
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def transport(root: Path):
    return _load(root / "publication-transport.py", "original_publication_transport", TRANSPORT_SHA)


def launcher(root: Path):
    # launch.py bytes are bound by root funding; admitted() rechecks them.
    return _load(root / "launch.py", "dense_suffix_production_launcher")


def prefix_for(arm: str, epoch: int) -> str:
    return f"{PREFIX}/{arm}/epoch-{epoch:03d}"


def epoch_cap(launch, pins, arm: str) -> int:
    return pins["checkpoint_upper_bound_bytes"][arm] // launch.EPOCHS


def expected_snapshot(launch, binding: dict, epoch: int, checkpoint: Path) -> dict:
    return {
        "schema_version": 1,
        "status": "scheduled_epoch_snapshot",
        "epoch": epoch,
        "optimizer_steps": launch.UPDATES * epoch,
        "checkpoint_file": f"epoch-{epoch:03d}.pt",
        "checkpoint_sha256": launch.sha(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "binding_sha256": launch.canonical(binding),
        "release_binding_sha256": binding["release"]["binding_sha256"],
        "fold": 0,
        "seed": launch.SEED,
        "submission_every_epochs": 1,
        "diagnostic_only": False,
        "submission_eligible": True,
        "submitted": False,
    }


def validate_state(launch, state: dict, binding: dict, arm: str, epoch: int) -> None:
    import torch

    storage = state.get("model_storage", {})
    model = state.get("model", {})
    encoder = [n for n in model if n.startswith("encoder.")]
    rng = state.get("rng_by_rank", [])
    scale = state.get("amp_scaler", {}).get("scale", 0)
    if (
        state.get("schema_version") != 2
        or state.get("binding") != binding
        or (state.get("epoch"), state.get("next_update"), state.get("step"))
        != (epoch, 0, launch.UPDATES * epoch)
        or state.get("scheduled_epoch") is not True
        or state.get("optimizer_boundary") is not True
        or storage.get("format") != "frozen_base_delta_v1"
        or set(model) != set(storage.get("learned_tensors", {}))
        or set(model) & set(storage.get("frozen_tensors", {}))
        or sum(model[n].numel() for n in encoder) != launch.PINNED[arm]["trainable_parameters"]
        or any(not torch.isfinite(t).all() for t in model.values())
        or not state.get("optimizer", {}).get("state")
        or len(rng) != 1
        or set(rng[0]) != {"torch", "cuda", "python", "numpy"}
        or rng[0]["cuda"] is None
        or not isinstance(scale, (int, float))
        or not math.isfinite(scale)
        or scale <= 0
    ):
        raise ValueError("Exact completed scheduled learned/base/optimizer/RNG/scaler required")


def source_files(arm: str, epoch: int, root: Path = ROOT, admission: Path = ADMISSION, pins=None):
    import torch

    launch = launcher(root)
    pins = launch.PINS if pins is None else pins
    if arm not in launch.ARMS or type(epoch) is not int or epoch not in range(1, launch.EPOCHS + 1):
        raise ValueError("One exact completed arm epoch required")
    core = transport(root)
    if any(
        p == PREFIX or p.startswith(PREFIX) or PREFIX.startswith(p)
        for p in (*core.OLD_PREFIXES, *SIBLING_PREFIXES)
    ):
        raise ValueError("Fresh distinct private prefix required")
    state = launch.admitted(root, admission, pins)
    admission_record = launch.load(root / "production-admission.json")
    if admission_record != launch.static_admission(root, admission, state):
        raise ValueError("Original production admission differs")
    out = root / "production" / arm
    terminals = sorted((root / "journal" / arm).glob("invocation-*-terminal.json"))
    if not terminals:
        raise ValueError("No original trainer invocation for this arm")
    record = launch.load(out / "run.json")
    binding = record["binding"]
    values = state["configs"][arm]
    launch.validate_binding(binding, arm, values, state["cpu"])
    bound = {launch.load(t).get("binding_sha256") for t in terminals} - {None}
    if (
        bound - {launch.canonical(binding)}
        or record.get("diagnostic_run") is not False
        or (
            record.get("training_studies"),
            record.get("outer_studies"),
        )
        != (launch.FITTING, launch.OUTER)
    ):
        raise ValueError("Run binding differs from the original journaled binding")
    metadata = out / "epoch-checkpoints" / f"epoch-{epoch:03d}.json"
    checkpoint = metadata.with_suffix(".pt")
    launch.regular(metadata)
    launch.regular(checkpoint)
    if checkpoint.stat().st_size > epoch_cap(launch, pins, arm):
        raise ValueError("Completed epoch exceeds funded checkpoint bound")
    if json.loads(metadata.read_text()) != expected_snapshot(launch, binding, epoch, checkpoint):
        raise ValueError("Actual immutable snapshot counters/bytes/hash/binding differ")
    validate_state(
        launch, torch.load(checkpoint, map_location="cpu", weights_only=True), binding, arm, epoch
    )
    if core.sha(core.LICENSE) != core.LICENSE_SHA:
        raise ValueError("Pinned official license changed")
    cpu, funding = state["cpu"], state["funding"]
    provenance = {
        "schema_version": 1,
        "status": "dense_suffix_capacity_epoch_source_and_input_provenance",
        "source_git": cpu["source_git"],
        "source_sha256": state["code"],
        "CPU_job": cpu["CPU_job"],
        "CPU_receipt_sha256": pins["CPU_receipt_sha256"],
        "GPU_recovery_receipt_sha256": funding["GPU_recovery_receipt_sha256"],
        "production_admission_sha256": launch.sha(root / "production-admission.json"),
        "funding_sha256": launch.sha(root / "production-funding.json"),
        "config_sha256": funding["config_sha256"][arm],
        "recipe_sha256": pins["recipe_sha256"][arm],
        "folds_sha256": cpu["folds_sha256"],
        "release_sha256": cpu["release_sha256"],
        "supervision_sha256": cpu["supervision_sha256"],
        "overlay_sha256": cpu["overlay_sha256"],
        "encoder": pins["official_encoder"],
        "license_sha256": core.LICENSE_SHA,
        "license_source": core.LICENSE_SOURCE,
        "protected_gold_or_confirmation_used": False,
        "external_initialization": False,
    }
    manifest = {
        "schema_version": 1,
        "status": "completed_dense_suffix_capacity_scheduled_epoch_private_research_only",
        "prefix": prefix_for(arm, epoch),
        "arm": arm,
        "suffix_method": launch.PINNED[arm]["method"],
        "encoder_learned_parameters": launch.PINNED[arm]["trainable_parameters"],
        "encoder_lr": launch.PINNED[arm]["lr"],
        "source_git": cpu["source_git"],
        "epoch": epoch,
        "optimizer_updates": launch.UPDATES * epoch,
        "checkpoint_sha256": launch.sha(checkpoint),
        "binding_sha256": launch.canonical(binding),
        "training_studies": launch.FITTING,
        "outer_studies": launch.OUTER,
        "fixed_final_epochs": launch.EPOCHS,
        "checkpoint_selection": "fixed_scheduled_epoch_only; no_epoch_selection",
        "clinical_or_Kaggle_gain_claimed": False,
        "competition_deployment_accepted": False,
    }
    for value in (manifest, provenance, json.loads(metadata.read_text())):
        core.assert_sanitized(value)
    members = {
        "checkpoint.pt": checkpoint,
        "snapshot.json": metadata,
        "DINOv3-LICENSE.md": core.LICENSE,
    }
    return core, launch, funding, manifest, provenance, members


def require_placement(node: str) -> None:
    if (
        os.uname().nodename != node
        or not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
        or os.environ.get("SLURM_JOB_GPUS")
        or os.environ.get("SLURM_STEP_GPUS")
    ):
        raise ValueError("Owning-node one-CPU zero-GPU publication required")


def publish(
    arm: str,
    epoch: int,
    root: Path = ROOT,
    admission: Path = ADMISSION,
    *,
    stdin=None,
    api_factory=None,
    operation=None,
    lock=None,
    pins=None,
    placement=require_placement,
) -> dict:
    core, launch, funding, manifest, provenance, members = source_files(
        arm, epoch, root, admission, pins
    )
    placement(funding["node"])
    token = core.read_token(sys.stdin if stdin is None else stdin)
    if api_factory is None or operation is None:
        from huggingface_hub import CommitOperationAdd, HfApi

        api_factory, operation = api_factory or HfApi, operation or CommitOperationAdd
    with core.canonical_lock() if lock is None else lock:
        prefix = manifest["prefix"]
        journal = root / "epoch-publication" / arm / f"epoch-{epoch:03d}"
        if journal.exists():
            raise ValueError("Existing publication journal requires read-only reconciliation")
        download = sum(p.stat().st_size for p in members.values()) + 2 * SMALL_OBJECT_BOUND
        if download > funding["publication_bytes"] or shutil.disk_usage(root).free < (
            launch.RESERVE + download
        ):
            raise ValueError("Funded publication scratch and reserve unavailable")
        api = api_factory(token=token)
        info = api.repo_info(core.REPO, repo_type="dataset")
        if info.private is not True:
            raise ValueError("Epoch checkpoints require the private dataset")
        paths = [f"{prefix}/{n}" for n in core.MEMBERS]
        if api.get_paths_info(core.REPO, [*paths, prefix], repo_type="dataset", revision=info.sha):
            raise ValueError("Existing remote path requires read-only reconciliation; never repeat")
        journal.mkdir(mode=0o700, parents=True)
        core.save(journal / "source-and-input-provenance.json", provenance)
        core.save(journal / "manifest.json", manifest)
        members = {
            "checkpoint.pt": members["checkpoint.pt"],
            "snapshot.json": members["snapshot.json"],
            "source-and-input-provenance.json": journal / "source-and-input-provenance.json",
            "DINOv3-LICENSE.md": members["DINOv3-LICENSE.md"],
            "manifest.json": journal / "manifest.json",
        }
        expected = {
            n: {"bytes": p.stat().st_size, "sha256": core.sha(p)} for n, p in members.items()
        }
        if source_files(arm, epoch, root, admission, pins)[3] != manifest:
            raise ValueError("Completed snapshot changed during publication preparation")
        core.save(
            journal / "intent.json",
            {
                "arm": arm,
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
                core.REPO,
                repo_type="dataset",
                parent_commit=info.sha,
                commit_message=f"Preserve private dense-suffix {arm} fold0 epoch {epoch}",
                operations=[
                    operation(path_in_repo=f"{prefix}/{n}", path_or_fileobj=str(p))
                    for n, p in members.items()
                ],
                num_threads=1,
            )
        except BaseException as error:
            core.save(
                journal / "uncertain.json",
                {"error_type": type(error).__name__, "automatic_retry": False},
            )
            raise RuntimeError("Publication mutation uncertain; reconcile read-only") from None
        core.save(journal / "commit.json", {"revision": commit.oid, "prefix": prefix})
        core.verify_private_revision(api, prefix, expected, commit.oid, root)
        receipt = {
            "status": "private_dense_suffix_epoch_full_download_size_SHA_verified",
            "arm": arm,
            "epoch": epoch,
            "prefix": prefix,
            "revision": commit.oid,
            "files": expected,
            "file_count": len(expected),
            "checkpoint_sha256": manifest["checkpoint_sha256"],
        }
        core.save(journal / "verified.json", receipt)
        return receipt


def reconcile(arm: str, epoch: int, root: Path = ROOT, *, stdin=None, api_factory=None) -> dict:
    """Read-only: never writes the journal, never mutates the Hub, never retries."""
    core = transport(root)
    token = core.read_token(sys.stdin if stdin is None else stdin)
    if api_factory is None:
        from huggingface_hub import HfApi

        api_factory = HfApi
    journal = root / "epoch-publication" / arm / f"epoch-{epoch:03d}"
    intent = core.load(journal / "intent.json", root)
    prefix = prefix_for(arm, epoch)
    if intent.get("prefix") != prefix or intent.get("automatic_retry") is not False:
        raise ValueError("Original intent differs")
    api = api_factory(token=token)
    if not (journal / "commit.json").exists():
        head = api.repo_info(core.REPO, repo_type="dataset").sha
        present = sorted(core.remote_paths(api, [f"{prefix}/{n}" for n in core.MEMBERS], head))
        return {"status": "uncertain_requires_owner_review", "revision": head, "present": present}
    commit = core.load(journal / "commit.json", root)
    core.verify_private_revision(api, prefix, intent["files"], commit["revision"], root)
    return {
        "status": "existing_publication_read_only_verified",
        "revision": commit["revision"],
        "file_count": len(intent["files"]),
    }


def main() -> None:
    os.umask(0o077)
    sys.path.insert(0, str(ADMISSION / "code"))
    args = sys.argv[1:]
    try:
        if len(args) == 3 and args[0] == "reconcile":
            result = reconcile(args[1], int(args[2]))
        elif len(args) == 2:
            result = publish(args[0], int(args[1]))
        else:
            raise SystemExit("usage: publish.py [reconcile] ARM EPOCH")
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
