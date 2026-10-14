"""Private write-once preservation of a completed source-slot fitting epoch.

Uses the pinned existing publication transport/lock/re-download verifier. Source,
rosters and classifier acceptance are specific to this run, never inherited from
the all-development model which supplied the transport implementation.
"""

import importlib.util
import json
import math
import os
import shutil
import sys
from pathlib import Path

ROOT = Path("/tmp/rsna-source-slot-fit-20261013-v1")
ADMISSION = Path("/tmp/rsna-source-slot-recovery-20261013-v1")
PREFIX = "experiments/dinov3-v1/training-epochs/source-slot-publictriplet-fold0-4fedbec3-fixed10-seed20260922"
TRANSPORT_SHA = "0f929dc1131c61f471c2e28c6a99367ecdc9fe31c45b52a413710643a054c7da"
CAP = 256 * 2**20


def transport(root: Path):
    import hashlib

    path = root / "publication-transport.py"
    if hashlib.sha256(path.read_bytes()).hexdigest() != TRANSPORT_SHA:
        raise ValueError("Pinned original publication transport changed")
    spec = importlib.util.spec_from_file_location("original_publication_transport", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.UPDATES = 281
    return module


def validate_epoch(state, record, epoch, original, own, values):
    import torch

    from rsna.schema import TARGETS
    from scripts.newton.probe_public_input_recipe_inputs import canonical

    binding = record["binding"]
    config = binding["config"]
    if (
        type(epoch) is not int
        or epoch not in range(1, 11)
        or record.get("diagnostic_run") is not False
        or (record.get("training_studies"), record.get("outer_studies")) != (2247, 1156)
        or any(config.get(k) != v for k, v in values.items() if k != "encoder")
        or config["encoder"] != original["encoder_provenance"]["public"]
        or binding["code_sha256"] != own["code_sha256"]
        or binding["training_uids_sha256"] != original["supervision"]["fitting_uid_sha256"]
        or binding["outer_uids_sha256"] != original["supervision"]["outer_uid_sha256"]
        or binding.get("experimental_fitting_supervision") != original["supervision"]
        or binding.get("public_input_cache") != original["public_cohort"]
        or binding.get("private_research_only") is not True
        or "external_initialization" in binding
        or (binding["device_type"], binding["world_size"]) != ("cuda", 1)
        or binding.get("targets") != list(TARGETS)
        or state.get("binding") != binding
        or (state.get("epoch"), state.get("next_update"), state.get("step"))
        != (epoch, 0, 281 * epoch)
        or state.get("scheduled_epoch") is not True
        or state.get("optimizer_boundary") is not True
        or state.get("schema_version") != 2
    ):
        raise ValueError("Exact completed non-diagnostic source-slot epoch required")
    storage = state["model_storage"]
    if (
        storage["format"] != "frozen_base_delta_v1"
        or storage["encoder_provenance"] != config["encoder"]
        or canonical(storage["frozen_tensors"]) != own["initialization"]["frozen_base_sha256"]
        or set(state["model"]) != set(storage["learned_tensors"])
        or sum(t.numel() for t in state["model"].values())
        != own["initialization"]["learned_parameters"]
        or any(
            list(t.shape) != storage["learned_tensors"][n]["shape"]
            or str(t.dtype) != storage["learned_tensors"][n]["dtype"]
            for n, t in state["model"].items()
        )
        or any(not torch.isfinite(t).all() for t in state["model"].values())
        or not state.get("optimizer", {}).get("state")
        or len(state.get("rng_by_rank", [])) != 1
        or set(state["rng_by_rank"][0]) != {"torch", "cuda", "python", "numpy"}
        or state["rng_by_rank"][0]["cuda"] is None
        or not math.isfinite(state.get("amp_scaler", {}).get("scale", 0))
        or state.get("amp_scaler", {}).get("scale", 0) <= 0
    ):
        raise ValueError("Complete learned/base/optimizer/RNG/scaler envelope required")
    return binding


def source_files(epoch: int, root: Path = ROOT):
    import torch

    sys.path.insert(0, str(ADMISSION / "code"))
    from scripts.newton.probe_source_slot_recovery import CPU_STATUS, prerequisites

    core = transport(root)
    _, hashes, original, values = prerequisites(ADMISSION)
    own = json.loads((ADMISSION / "CPU-accepted.json").read_text())
    recovery = json.loads((ADMISSION / "recovery-accepted.json").read_text())
    admission = core.load(root / "production-admission.json", root)
    if (
        own["status"] != CPU_STATUS
        or own["code_sha256"] != hashes
        or own["config"] != values
        or recovery["status"] != "source_slot_r32_actual_canonical_GPU_recovery_accepted"
        or recovery["CPU_receipt_sha256"] != core.sha(ADMISSION / "CPU-accepted.json")
        or admission["recovery_receipt_sha256"] != core.sha(ADMISSION / "recovery-accepted.json")
        or admission["own_CPU_sha256"] != core.sha(ADMISSION / "CPU-accepted.json")
        or admission["source_git"] != own["source_git"]
        or admission["config_sha256"] != core.sha(root / "config.json")
        or json.loads((root / "config.json").read_text()) != values
    ):
        raise ValueError("This exact classifier's actual CPU/GPU/production admission required")
    run = root / "production"
    record = core.load(run / "run.json", root)
    snapshot = run / "epoch-checkpoints" / f"epoch-{epoch:03d}.json"
    checkpoint = snapshot.with_suffix(".pt")
    core.regular(snapshot, root)
    core.regular(checkpoint, root)
    if checkpoint.stat().st_size > CAP:
        raise ValueError("Immutable delta checkpoint exceeds funded cap")
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    binding = validate_epoch(state, record, epoch, original, own, values)
    metadata = json.loads(snapshot.read_text())
    if metadata != core.expected_snapshot(binding, epoch, checkpoint):
        raise ValueError("Actual snapshot counters/bytes/hash/binding differ")
    if core.sha(core.LICENSE) != core.LICENSE_SHA:
        raise ValueError("Pinned official license changed")
    provenance = {
        "source_git": own["source_git"],
        "code_sha256": hashes,
        "config_sha256": core.sha(root / "config.json"),
        "own_CPU_sha256": core.sha(ADMISSION / "CPU-accepted.json"),
        "own_GPU_recovery_sha256": core.sha(ADMISSION / "recovery-accepted.json"),
        "production_admission_sha256": core.sha(root / "production-admission.json"),
        "public_input_cache": original["public_cohort"],
        "encoder": original["encoder_provenance"]["public"],
        "license_sha256": core.LICENSE_SHA,
        "license_source": core.LICENSE_SOURCE,
        "fitting_supervision": original["supervision"],
        "protected_gold_or_confirmation_used": False,
        "competition_deployment_accepted": False,
    }
    manifest = {
        "status": "completed_source_slot_scheduled_epoch_private_research_only",
        "source_git": own["source_git"],
        "epoch": epoch,
        "optimizer_updates": 281 * epoch,
        "prefix": f"{PREFIX}/epoch-{epoch:03d}",
        "checkpoint_sha256": metadata["checkpoint_sha256"],
        "binding_sha256": metadata["binding_sha256"],
        "training_studies": 2247,
        "outer_studies": 1156,
        "fixed_final_epochs": 10,
        "clinical_or_Kaggle_gain_claimed": False,
        "competition_deployment_accepted": False,
    }
    for value in (manifest, provenance, metadata):
        core.assert_sanitized(value)
    return (
        manifest,
        provenance,
        {"checkpoint.pt": checkpoint, "snapshot.json": snapshot, "DINOv3-LICENSE.md": core.LICENSE},
    )


def publish(epoch: int, root: Path = ROOT):
    import torch
    from huggingface_hub import CommitOperationAdd, HfApi

    if (
        os.uname().nodename != "evc31"
        or not os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
        or os.environ.get("SLURM_STEP_GPUS")
        or torch.cuda.device_count()
    ):
        raise ValueError("Actual owning-node one-CPU zero-GPU step required")
    core = transport(root)
    token = core.read_token(sys.stdin)
    with core.canonical_lock():
        journal = root / "epoch-publication" / f"epoch-{epoch:03d}"
        if journal.exists():
            raise ValueError("Existing journal requires read-only reconciliation")
        manifest, provenance, members = source_files(epoch, root)
        prefix = manifest["prefix"]
        funding = json.loads((root / "funding.json").read_text())
        if shutil.disk_usage(root).free < funding["required_free_bytes"] + 2 * CAP:
            raise ValueError("Publication scratch and retained reserve unavailable")
        api = HfApi(token=token)
        info = api.repo_info(core.REPO, repo_type="dataset")
        if info.private is not True or api.get_paths_info(
            core.REPO,
            [prefix, *[f"{prefix}/{n}" for n in core.MEMBERS]],
            repo_type="dataset",
            revision=info.sha,
        ):
            raise ValueError("Private destination and original empty prefix required")
        journal.mkdir(mode=0o700, parents=True)
        core.save(journal / "source-and-input-provenance.json", provenance)
        core.save(journal / "manifest.json", manifest)
        members.update(
            {n: journal / n for n in ("source-and-input-provenance.json", "manifest.json")}
        )
        expected = {
            n: {
                "bytes": p.stat().st_size,
                "sha256": manifest["checkpoint_sha256"] if n == "checkpoint.pt" else core.sha(p),
            }
            for n, p in members.items()
        }
        core.save(
            journal / "intent.json",
            {
                "parent_commit": info.sha,
                "prefix": prefix,
                "files": expected,
                "automatic_retry": False,
            },
        )
        try:
            commit = api.create_commit(
                core.REPO,
                repo_type="dataset",
                parent_commit=info.sha,
                operations=[
                    CommitOperationAdd(path_in_repo=f"{prefix}/{n}", path_or_fileobj=str(p))
                    for n, p in members.items()
                ],
                commit_message=f"Preserve private source-slot fold0 epoch {epoch}",
                num_threads=1,
            )
        except BaseException as error:
            core.save(
                journal / "uncertain.json",
                {"error_type": type(error).__name__, "automatic_retry": False},
            )
            raise RuntimeError("Uncertain mutation: reconcile read-only, never retry") from None
        core.save(journal / "commit.json", {"revision": commit.oid, "prefix": prefix})
        core.verify_private_revision(api, prefix, expected, commit.oid, root)
        receipt = {
            "status": "private_source_slot_epoch_full_download_size_SHA_verified",
            "epoch": epoch,
            "revision": commit.oid,
            "prefix": prefix,
            "files": expected,
            "file_count": 5,
            "checkpoint_sha256": manifest["checkpoint_sha256"],
        }
        core.save(journal / "verified.json", receipt)
        return receipt


if __name__ == "__main__":
    os.umask(0o077)
    print(json.dumps(publish(int(sys.argv[1]))), flush=True)
