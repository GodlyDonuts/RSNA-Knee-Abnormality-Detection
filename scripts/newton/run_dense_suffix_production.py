"""One-GPU sequential fixed-ten matched dense-suffix versus suffix-R32 production.

PREPARED ONLY. Root stages this file as ``ROOT/launch.py`` and the private epoch
publisher as ``ROOT/publish.py`` only after it admits the ACTUAL original GPU
recovery receipt plus fresh explicit node-local checkpoint/runtime/publication
funding and assigned-device/source/config/fold bindings in
``ROOT/production-funding.json``. The engineering 12 GiB envelope never funds this.

The unchanged canonical ``rsna.competitive.train`` runs from the exactly admitted
source ``0ecece12`` tree: ``suffix_r32`` to completion, then ``dense``. Each arm fits
2,247 studies for 10 x 281 updates and predicts the same 1,156 development studies
once at fixed epoch 10. No best-epoch selection, protected gold, confirmation or
external initialization. A trainer exit 75 (deadline SIGUSR1/preemption) is resumed
only by a later launch through ``--resume`` of the bound checkpoint. Every trainer
invocation has an exclusive intent and terminal record; an intent without terminal
is uncertain replay and is refused. Nothing here claims accuracy.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import signal
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path

ROOT = Path("/tmp/rsna-dense-suffix-fit-20261014-v1")
ADMISSION = Path("/tmp/rsna-dense-suffix-admission-20261014-v1")
RECIPE = "initiative/experiments/competitive-v1/dense-suffix-capacity-v1"
ARMS = {"suffix_r32": "suffix-r32-config.json", "dense": "dense-config.json"}
PINNED = {
    "suffix_r32": {"method": "suffix_rs_lora_v1", "trainable_parameters": 7_077_888, "lr": 1e-4},
    "dense": {"method": "dense_suffix_delta_v1", "trainable_parameters": 150_994_944, "lr": 1e-5},
}
CPU_STATUS = "dense_suffix_capacity_owning_node_inputs_and_matched_initialization_CPU_accepted"
RECOVERY_STATUS = "actual_matched_dense_suffix_additional_deadline_recovery_accepted"
RECOVERY_RECEIPT = "accepted-after-deadline.json"
RECOVERY_VERIFIER = "verifier.py"
DENSE_RETAINED_BOUNDARY = "checkpoint-before-deadline-resume-step002.pt"
ADMISSION_STATUS = "actual_dense_suffix_fixed10_production_admitted"
TERMINAL_STATUS = "matched_dense_suffix_fixed10_both_arms_completed_requires_fixed_final_replay"
# Actual CPU899766 admission (initiative/audits/dense-suffix-admission-root-review-20261014-v1).
PINS = {
    "node": "evc31",
    "source_git": "0ecece12407e5b802eec6c33302213a3b5352353",
    "CPU_receipt_sha256": "158c242faa37c46e3b8a2e632d68b3ae690174bb3ec3b85abed2a46ffc53c834",
    "CPU_job": "899766",
    "recovery_source_manifest_sha256": (
        "89f7ce1706866898de805a116f563c2b9e37440964d683f44b7240d6fc8d0f96"
    ),
    "recovery_wrapper_sha256": "582fa5872d34a6ff9192207a7c81d0e9bf0f85d35a5c5133732f4f1494221360",
    "encoder_spec_path_sha256": "39b14c28485e7a6c426e71536704a7537017097c39949fe4b13bbb55ac75f84a",
    "recipe_sha256": {
        "suffix_r32": "92b6116f9d2ee7311dd74890b37fff7ea12474d6cd21a5df71922d375e78d2aa",
        "dense": "19332d2aa5c80d8134d76978e9f8e54fe56407b744002826acaafef2fd0984d6",
    },
    "official_encoder": {
        "source_id": "facebook/dinov3-vitl16-pretrain-lvd1689m",
        "source_revision": "ea8dc2863c51be0a264bab82070e3e8836b02d51",
        "weights_sha256": "dcb2e45127cccbf1601e5f42fef165eea275c8e5213197e8dcf3f48822718179",
        "config_sha256": "135ecd23e34a70b6fbed8b083fdecb319b7e3a54e3d849258bbe4ddcf1783bb5",
    },
    "checkpoint_upper_bound_bytes": {"suffix_r32": 2_008_023_040, "dense": 21_005_074_432},
    # Actual root GPU recovery: the original wrapper timed out in dense resume (its
    # accepted.json never existed); one continuation from the retained step-2 boundary
    # was independently CPU-verified (initiative/audits/dense-suffix-deadline-
    # continuation-20261014-v1). These limits are bound exactly, never relaxed.
    "recovery_output": "/tmp/rsna-dense-suffix-GPU-recovery-20261014-v1",
    "recovery_receipt_sha256": "cb8602821133e77e8c904c6b4c05a802600904e378ba58828d6c4e32edfb3de4",
    "recovery_verifier_sha256": "774a7b3f167ad3e8c6d37acd51413c9e2cdbed452fdb10223ec1d82e01b4329d",
    "recovery_limits": {
        "additional_interruption": {
            "original_timeout_preserved": True,
            "dense_resume_saved_boundary": [1, 0, 2],
            "dense_history_sources": [
                "stop.stdout step1",
                "resume.stdout committed step2",
                "replay/invocation.json continued steps3/4",
            ],
            "unpromoted_step4_partial_preserved": True,
        },
        "suffix_r32_stop1_partial_checked_by_original_GPU_wrapper_before_dense": True,
        "suffix_r32_stop1_partial_independently_retained_for_CPU_recheck": False,
        "complete_model_base_AdamW_LR_counters_all_RNG_scaler_and_CSV_equal": True,
        "diagnostic_only": True,
        "clinical_accuracy_evidence": False,
        "competition_submission": False,
    },
}
EPOCHS, UPDATES, FITTING, OUTER, SEED = 10, 281, 2247, 1156, 20260922
GIB = 2**30
RESERVE = 20 * GIB
ENGINEERING_ENVELOPE = 12 * GIB
MAX_INVOCATIONS = 6
MIN_START_SECONDS = 3600
STOP_MARGIN_SECONDS = 1200
STOP_GRACE_SECONDS = 900
RESUMABLE = {"checkpointed_for_resume", "final_inference_interrupted_resume_final_checkpoint"}
FUNDING_KEYS = {
    "funded_by",
    "production_funded",
    "node",
    "parent_job",
    "assigned_GPU",
    "CPUs",
    "source_git",
    "CPU_receipt_sha256",
    "GPU_recovery_output",
    "GPU_recovery_receipt_path",
    "GPU_recovery_receipt_sha256",
    "GPU_recovery_verifier_path",
    "GPU_recovery_verifier_sha256",
    "launcher_sha256",
    "publisher_sha256",
    "config_sha256",
    "folds_sha256",
    "release_sha256",
    "supervision_sha256",
    "overlay_sha256",
    "checkpoint_bytes",
    "runtime_bytes",
    "publication_bytes",
    "existing_liabilities_bytes",
    "reserve_bytes",
    "required_free_bytes",
    "shared_future_bytes",
}
Executor = Callable[[Path, Path, list[str], Path, float], int]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def save(path: Path, value: object) -> None:
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def regular(path: Path) -> Path:
    path = Path(path)
    if path.is_symlink() or any(p.is_symlink() for p in path.parents) or not path.is_file():
        raise ValueError(f"Exact regular non-linked file required: {path.name}")
    return path


def load(path: Path) -> dict:
    value = json.loads(regular(path).read_text())
    if not isinstance(value, dict):
        raise ValueError("Invalid control record")
    return value


def tree_bytes(path: Path) -> int:
    if not path.exists():
        return 0
    seen: dict[tuple[int, int], int] = {}
    for item in path.rglob("*"):
        if item.is_file() and not item.is_symlink():
            s = item.stat()
            seen[(s.st_dev, s.st_ino)] = s.st_size
    return sum(seen.values())


def production_config(recipe: Mapping, spec: Mapping) -> dict:
    """The predeclared fixed-ten recipe on the mounted official base; nothing else changes."""
    from rsna.competitive.train import TrainConfig

    encoder = dict(recipe["encoder"])
    policy = encoder.pop("suffix_adapter")
    if set(spec) != set(encoder) or any(
        spec[k] != encoder[k] for k in encoder if k not in {"weights_path", "config_path"}
    ):
        raise ValueError("Mounted encoder spec differs from the pinned official base")
    values = dict(recipe, encoder=dict(spec, suffix_adapter=policy))
    TrainConfig(**values).validate()
    fixed = {
        "epochs": EPOCHS,
        "diagnostic_study_limit": None,
        "fit_scope": "outer_fold",
        "fold": 0,
        "seed": SEED,
        "warmup_updates": UPDATES,
        "submission_every_epochs": 1,
        "checkpoint_format": "frozen_base_delta",
        "batch_size": 1,
        "accumulation_steps": 8,
        "precision": "fp16",
    }
    if any(values.get(k) != v for k, v in fixed.items()):
        raise ValueError("Predeclared fixed-ten fold0 effective-batch-8 recipe required")
    return values


def verify_source(source: Path, manifest_path: Path, cpu: Mapping, pins: Mapping) -> dict:
    """The exact admitted tree: every rsna module, wrapper and both recipe files."""
    if sha(regular(manifest_path)) != pins["recovery_source_manifest_sha256"]:
        raise ValueError("Admitted recovery source manifest changed")
    manifest = json.loads(manifest_path.read_text())
    package = {str(p.relative_to(source)) for p in (source / "rsna").rglob("*.py") if p.is_file()}
    if not package or package != {n for n in manifest if n.startswith("rsna/")}:
        raise ValueError("Source manifest must bind exactly every canonical rsna module")
    for name, value in manifest.items():
        if sha(regular(source / name)) != value:
            raise ValueError("Admitted numerical source differs from its manifest")
    if manifest.get("scripts/newton/run_dense_suffix_recovery.py") != pins[
        "recovery_wrapper_sha256"
    ] or any(manifest.get(f"{RECIPE}/{n}") != pins["recipe_sha256"][a] for a, n in ARMS.items()):
        raise ValueError("Admitted wrapper/recipe source differs")
    code = {n.removeprefix("rsna/"): h for n, h in manifest.items() if n.startswith("rsna/")}
    if code != cpu["code_sha256"]:
        raise ValueError("Source differs from the CPU-admitted numerical code")
    return code


def admitted_cpu(admission: Path, pins: Mapping) -> dict:
    path = admission / "CPU-accepted.json"
    if sha(regular(path)) != pins["CPU_receipt_sha256"]:
        raise ValueError("Actual CPU input/initial-state admission receipt changed")
    cpu = load(path)
    if (
        cpu.get("status") != CPU_STATUS
        or cpu.get("CPU_job") != pins["CPU_job"]
        or cpu.get("source_git") != pins["source_git"]
        or cpu.get("GPU_acceptance_inferred") is not False
        or cpu.get("recipe_sha256") != pins["recipe_sha256"]
        or cpu.get("recovery_source_manifest_sha256") != pins["recovery_source_manifest_sha256"]
        or cpu.get("wrapper_sha256") != pins["recovery_wrapper_sha256"]
        or (cpu.get("full_fitting_studies"), cpu.get("full_outer_studies")) != (FITTING, OUTER)
        or cpu.get("protected_labels_used_for_recipe_or_metrics") is not False
        or any(
            cpu["arms"][arm]["encoder_learned_parameters"] != PINNED[arm]["trainable_parameters"]
            or cpu["arms"][arm]["encoder_lr"] != PINNED[arm]["lr"]
            for arm in ARMS
        )
        or cpu["future_obligations"]["production_fixed10"]["checkpoint_upper_bound_bytes"]
        != pins["checkpoint_upper_bound_bytes"]
    ):
        raise ValueError("Exact actual dense-suffix CPU admission required")
    return cpu


def admitted_recovery(cpu: Mapping, funding: Mapping, pins: Mapping) -> dict:
    """The ACTUAL after-deadline GPU recovery: exact receipt, verifier and runtime evidence."""
    output = Path(pins["recovery_output"])
    receipt_path = output / RECOVERY_RECEIPT
    if (
        funding["GPU_recovery_output"] != str(output)
        or funding["GPU_recovery_receipt_path"] != str(receipt_path)
        or funding["GPU_recovery_receipt_sha256"] != pins["recovery_receipt_sha256"]
        or funding["GPU_recovery_verifier_sha256"] != pins["recovery_verifier_sha256"]
    ):
        raise ValueError("Root has not admitted this exact actual GPU recovery receipt")
    if not receipt_path.exists():
        raise ValueError("Actual GPU recovery acceptance is absent; production refused")
    if (output / "accepted.json").exists():
        raise ValueError("Recorded original wrapper timeout contradicted; reconcile read-only")
    if sha(regular(receipt_path)) != pins["recovery_receipt_sha256"]:
        raise ValueError("Actual GPU recovery receipt bytes changed")
    if (
        sha(regular(Path(funding["GPU_recovery_verifier_path"])))
        != (pins["recovery_verifier_sha256"])
    ):
        raise ValueError("Actual independent recovery verifier source changed")
    receipt = load(receipt_path)
    if (
        receipt.get("status") != RECOVERY_STATUS
        or receipt.get("CPU_receipt_sha256") != pins["CPU_receipt_sha256"]
        or receipt.get("device") != "cuda"
        or receipt.get("GPUs_used_concurrently") != 1
        or receipt.get("source_manifest_sha256") != pins["recovery_source_manifest_sha256"]
        or receipt.get("recipe_sha256") != pins["recipe_sha256"]
        or any(receipt.get(k) != v for k, v in pins["recovery_limits"].items())
        or set(receipt.get("arms", {})) != set(ARMS)
    ):
        raise ValueError("Exact actual one-GPU matched recovery acceptance required")
    for arm in ARMS:
        row, full = receipt["arms"][arm], output / arm / "full"
        updates = row.get("finite_updates")
        if (
            row.get("exact_complete_state_history_and_prediction_recovery") is not True
            or row.get("trainable_encoder_parameters") != PINNED[arm]["trainable_parameters"]
            or not isinstance(updates, list)
            or [u.get("optimizer_steps") for u in updates] != [1, 2, 3, 4]
            or not all(math.isfinite(u.get("loss", math.nan)) for u in updates)
            or sha(regular(full / "checkpoint.pt")) != row.get("checkpoint_sha256")
            or sha(regular(full / "outer_predictions.csv")) != row.get("prediction_sha256")
        ):
            raise ValueError("Actual recovery runtime evidence differs from its receipt")
        binding = load(full / "run.json")["binding"]
        if binding.get("device_type") != "cuda" or binding.get("code_sha256") != cpu["code_sha256"]:
            raise ValueError("Recovery ran a different numerical source or device")
    intent = load(output / "deadline-continuation-intent.json")
    if sha(regular(output / "dense" / DENSE_RETAINED_BOUNDARY)) != intent.get("checkpoint_sha256"):
        raise ValueError("Retained dense step-2 recovery boundary changed")
    return receipt


def validate_funding(funding: Mapping, cpu: Mapping, root: Path, pins: Mapping) -> None:
    if set(funding) != FUNDING_KEYS:
        raise ValueError("Complete explicit root production funding/binding required")
    upper = sum(pins["checkpoint_upper_bound_bytes"].values())
    publication = 2 * (pins["checkpoint_upper_bound_bytes"]["dense"] // EPOCHS) + 64 * 2**20
    new = funding["checkpoint_bytes"] + funding["runtime_bytes"] + funding["publication_bytes"]
    if (
        funding["funded_by"] != "root"
        or funding["production_funded"] is not True
        or funding["node"] != pins["node"]
        or funding["source_git"] != pins["source_git"]
        or funding["CPU_receipt_sha256"] != pins["CPU_receipt_sha256"]
        or funding["shared_future_bytes"] != 0
        or funding["folds_sha256"] != cpu["folds_sha256"]
        or funding["release_sha256"] != cpu["release_sha256"]
        or funding["supervision_sha256"] != cpu["supervision_sha256"]
        or funding["overlay_sha256"] != cpu["overlay_sha256"]
        or set(funding["config_sha256"]) != set(ARMS)
        or not str(funding["parent_job"]).isdigit()
        or not funding["assigned_GPU"]
        or type(funding["CPUs"]) is not int
        or funding["CPUs"] < 1
    ):
        raise ValueError("Production funding binds a different source/roster/fold/device")
    if new <= ENGINEERING_ENVELOPE:
        raise ValueError("The 12 GiB engineering envelope does not fund fixed-ten production")
    if (
        funding["checkpoint_bytes"] < upper
        or funding["publication_bytes"] < publication
        or funding["runtime_bytes"] <= 0
        or funding["reserve_bytes"] != RESERVE
        or funding["existing_liabilities_bytes"] <= 0
        or funding["required_free_bytes"]
        != funding["existing_liabilities_bytes"] + new + funding["reserve_bytes"]
    ):
        raise ValueError("Explicit node-local checkpoint/runtime/publication funding required")
    if (
        sha(regular(root / "launch.py")) != funding["launcher_sha256"]
        or sha(regular(root / "publish.py")) != (funding["publisher_sha256"])
    ):
        raise ValueError("Staged launcher/publisher differ from root's funded bytes")


def admitted(root: Path = ROOT, admission: Path = ADMISSION, pins: Mapping = PINS) -> dict:
    """Every precondition, rechecked on every launch and by every publication."""
    cpu = admitted_cpu(admission, pins)
    funding = load(root / "production-funding.json")
    validate_funding(funding, cpu, root, pins)
    source = admission / "code"
    code = verify_source(source, admission / "recovery-source-manifest.json", cpu, pins)
    spec_path = admission / "encoder-spec.json"
    if sha(regular(spec_path)) != pins["encoder_spec_path_sha256"]:
        raise ValueError("Mounted official base spec changed")
    spec = json.loads(spec_path.read_text())
    if any(spec.get(k) != v for k, v in pins["official_encoder"].items()):
        raise ValueError("Official DINOv3-L base provenance differs")
    inputs = cpu["GPU_wrapper_arguments"]
    artifact = Path(inputs["artifact-root"])
    if (
        sha(regular(Path(inputs["overlay"]))) != cpu["overlay_sha256"]
        or sha(regular(artifact / "initiative/experiments/competitive-v1/folds/fold.csv"))
        != cpu["folds_sha256"]
    ):
        raise ValueError("Admitted fitting overlay or fold assignment changed")
    recovery = admitted_recovery(cpu, funding, pins)
    configs = {}
    for arm, name in ARMS.items():
        recipe = json.loads((source / RECIPE / name).read_text())
        values = production_config(recipe, spec)
        path = root / f"{arm}-config.json"
        if (
            sha(regular(path)) != funding["config_sha256"][arm]
            or json.loads(path.read_text()) != values
        ):
            raise ValueError("Staged production config differs from the admitted recipe")
        configs[arm] = values
    return {
        "cpu": cpu,
        "funding": funding,
        "recovery": recovery,
        "code": code,
        "configs": configs,
        "source": source,
        "inputs": inputs,
    }


def static_admission(root: Path, admission: Path, state: Mapping) -> dict:
    return {
        "status": ADMISSION_STATUS,
        "source_git": state["cpu"]["source_git"],
        "CPU_receipt_sha256": sha(admission / "CPU-accepted.json"),
        "GPU_recovery_receipt_sha256": state["funding"]["GPU_recovery_receipt_sha256"],
        "funding_sha256": sha(root / "production-funding.json"),
        "config_sha256": state["funding"]["config_sha256"],
        "launcher_sha256": state["funding"]["launcher_sha256"],
        "publisher_sha256": state["funding"]["publisher_sha256"],
        "arms_sequential_on_one_assigned_GPU": list(ARMS),
        "epochs": EPOCHS,
        "updates_per_epoch": UPDATES,
        "training_studies": FITTING,
        "outer_studies": OUTER,
        "fixed_final_epoch_only": True,
        "protected_gold_or_confirmation_used": False,
        "external_initialization": False,
        "accuracy_claimed": False,
    }


def validate_binding(binding: Mapping, arm: str, values: Mapping, cpu: Mapping) -> None:
    config = binding.get("config", {})
    encoder = config.get("encoder", {})
    suffix = encoder.get("suffix_adapter", {})
    if (
        {k: config.get(k) for k in values if k != "encoder"}
        != {k: v for k, v in values.items() if k != "encoder"}
        or suffix.get("method") != PINNED[arm]["method"]
        or suffix.get("trainable_parameters") != PINNED[arm]["trainable_parameters"]
        or any(encoder.get(k) != values["encoder"][k] for k in PINS["official_encoder"])
        or binding.get("code_sha256") != cpu["code_sha256"]
        or (binding.get("device_type"), binding.get("world_size")) != ("cuda", 1)
        or canonical(binding.get("experimental_fitting_supervision")) != cpu["supervision_sha256"]
        or binding.get("public_input_cache") != cpu["public_cohort"]
        or binding.get("release", {}).get("manifest_sha256") != cpu["release_sha256"]
        or "external_initialization" in binding
    ):
        raise ValueError("Trainer binding differs from the admitted source/config/roster/base")


def validate_completed(out: Path, arm: str, values: Mapping, cpu: Mapping) -> dict:
    record, result = load(out / "run.json"), load(out / "result.json")
    validate_binding(record["binding"], arm, values, cpu)
    if (
        record.get("diagnostic_run") is not False
        or (record.get("training_studies"), record.get("outer_studies")) != (FITTING, OUTER)
        or record.get("encoder_trainable_parameters") != PINNED[arm]["trainable_parameters"]
        or record.get("held_out_confirmation_and_gold_used") is not False
        or result.get("status") != "completed"
        or result.get("diagnostic_run") is not False
        or result.get("optimizer_steps") != EPOCHS * UPDATES
        or (result.get("training_studies"), result.get("outer_studies")) != (FITTING, OUTER)
        or result.get("checkpoint_selection") != "fixed_final_epoch_no_outer_selection"
        or result.get("binding_sha256") != canonical(record["binding"])
        or result.get("checkpoint_sha256") != sha(regular(out / "checkpoint.pt"))
        or result.get("predictions_sha256") != sha(regular(out / "outer_predictions.csv"))
        or any(
            not (out / "epoch-checkpoints" / f"epoch-{e:03d}.json").is_file()
            for e in range(1, EPOCHS + 1)
        )
    ):
        raise ValueError("Exact fixed-final 2810-update 2247/1156 completion required")
    return record


def journal_state(journal: Path, out: Path) -> tuple[str, int, dict | None]:
    """('fresh'|'resume'|'completed', next index, last terminal). Uncertain replay refused."""
    intents = sorted(journal.glob("invocation-*-intent.json"))
    if not intents:
        if out.exists():
            raise ValueError("Output without an original intent; reconcile read-only")
        return "fresh", 1, None
    numbers = [int(p.name.split("-")[1]) for p in intents]
    if numbers != list(range(1, len(numbers) + 1)):
        raise ValueError("Original intent sequence is not contiguous")
    for n in numbers:
        if not (journal / f"invocation-{n:02d}-terminal.json").exists():
            raise ValueError("Intent without terminal record is uncertain replay; never retry")
    last = load(journal / f"invocation-{numbers[-1]:02d}-terminal.json")
    if last.get("returncode") == 0:
        return "completed", numbers[-1] + 1, last
    if last.get("returncode") != 75 or last.get("resumable") is not True:
        raise ValueError("Terminal trainer failure preserved; no automatic retry")
    if numbers[-1] >= MAX_INVOCATIONS:
        raise ValueError("Bounded exact-resume count exhausted")
    return "resume", numbers[-1] + 1, last


def verify_resumable(out: Path, terminal: Mapping) -> None:
    invocation = load(out / "invocation.json")
    record = load(out / "run.json")
    if (
        invocation.get("exit_code") != 75
        or invocation.get("status") not in RESUMABLE
        or invocation.get("checkpoint_sha256") != sha(regular(out / "checkpoint.pt"))
        or invocation.get("checkpoint_sha256") != terminal.get("checkpoint_sha256")
        or invocation.get("binding_sha256") != canonical(record["binding"])
        or invocation.get("binding_sha256") != terminal.get("binding_sha256")
    ):
        raise ValueError("Bound exit-75 checkpoint/binding changed; exact resume refused")


def trainer_command(state: Mapping, config: Path) -> list[str]:
    source, inputs = state["source"], state["inputs"]
    artifact = Path(inputs["artifact-root"])
    return [
        sys.executable,
        "-B",
        str(source / "scripts/newton/enforce_deterministic_runtime.py"),
        "--module",
        "rsna.competitive.train",
        "--config",
        str(config),
        "--device",
        "cuda",
        "--fitting-supervision",
        str(inputs["overlay"]),
        "--studies",
        str(artifact / "initiative/releases/report-supervision-v1/studies.parquet"),
        "--artifact-root",
        str(artifact),
        "--folds",
        str(artifact / "initiative/experiments/competitive-v1/folds"),
        "--series",
        str(
            artifact
            / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv"
        ),
        "--cache",
        str(inputs["cache"]),
    ]


def subprocess_executor(state: Mapping) -> Executor:
    threads = str(state["funding"]["CPUs"])
    env = os.environ.copy()
    env.update(
        PYTHONPATH=str(state["source"]),
        PYTHONNOUSERSITE="1",
        PYTHONDONTWRITEBYTECODE="1",
        OMP_NUM_THREADS=threads,
        MKL_NUM_THREADS=threads,
        OPENBLAS_NUM_THREADS=threads,
        WORLD_SIZE="1",
        RANK="0",
        LOCAL_RANK="0",
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
    )

    def execute(config: Path, out: Path, extra: list[str], logs: Path, seconds: float) -> int:
        command = [*trainer_command(state, config), "--output", str(out), *extra]
        with (
            (logs.with_suffix(".stdout")).open("x") as so,
            (logs.with_suffix(".stderr")).open("x") as se,
        ):
            child = subprocess.Popen(command, env=env, cwd=state["source"], stdout=so, stderr=se)
            forward = lambda *_: child.send_signal(signal.SIGUSR1)  # noqa: E731
            previous = signal.signal(signal.SIGTERM, forward)
            try:
                try:
                    return child.wait(timeout=max(seconds - STOP_MARGIN_SECONDS, 1))
                except subprocess.TimeoutExpired:
                    child.send_signal(signal.SIGUSR1)
                try:
                    return child.wait(timeout=STOP_GRACE_SECONDS)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
                    return -9
            finally:
                signal.signal(signal.SIGTERM, previous)

    return execute


def require_placement(funding: Mapping, pins: Mapping = PINS) -> str:
    import torch

    if (
        os.uname().nodename != pins["node"]
        or os.environ.get("SLURM_JOB_ID") != str(funding["parent_job"])
        or os.environ.get("SLURM_CPUS_PER_TASK") != str(funding["CPUs"])
        or (os.environ.get("SLURM_STEP_GPUS") or os.environ.get("SLURM_JOB_GPUS"))
        != str(funding["assigned_GPU"])
        or torch.cuda.device_count() != 1
        or "H100" not in torch.cuda.get_device_name(0)
    ):
        raise ValueError("Funded owning-node parent job and exactly one assigned H100 required")
    return torch.cuda.get_device_name(0)


def parent_remaining_seconds(job: str) -> float:
    record = json.loads(
        subprocess.check_output(["scontrol", "show", "job", job, "--json"], text=True)
    )["jobs"][0]
    end = record["end_time"]
    end = end.get("number") if isinstance(end, dict) else end
    return float(end) - time.time()


def run(
    root: Path = ROOT,
    admission: Path = ADMISSION,
    *,
    execute: Executor | None = None,
    placement: Callable[[Mapping], str] = require_placement,
    remaining: Callable[[str], float] = parent_remaining_seconds,
    pins: Mapping = PINS,
) -> int:
    """Run/continue both arms sequentially. Returns 0 when both are complete, 75 to continue."""
    state = admitted(root, admission, pins)
    funding, cpu = state["funding"], state["cpu"]
    device = placement(funding)
    static = static_admission(root, admission, state)
    if (root / "production-admission.json").exists():
        if load(root / "production-admission.json") != static:
            raise ValueError("Original production admission differs from this launch")
    else:
        save(root / "production-admission.json", static)
    execute = execute or subprocess_executor(state)
    job = str(funding["parent_job"])
    deadline = time.monotonic() + remaining(job)
    production = root / "production"
    production.mkdir(mode=0o700, exist_ok=True)
    records = {}
    for arm in ARMS:
        values, config = state["configs"][arm], root / f"{arm}-config.json"
        out, journal = production / arm, root / "journal" / arm
        journal.mkdir(mode=0o700, parents=True, exist_ok=True)
        mode, n, last = journal_state(journal, out)
        if mode == "completed":
            records[arm] = validate_completed(out, arm, values, cpu)
            continue
        if records.keys() != set(list(ARMS)[: list(ARMS).index(arm)]):
            raise ValueError("Arms run sequentially; previous arm must be complete")
        seconds = deadline - time.monotonic()
        if seconds < MIN_START_SECONDS:
            return 75
        used = tree_bytes(production)
        if (
            used > funding["checkpoint_bytes"] + funding["runtime_bytes"]
            or shutil.disk_usage(root).free + used < funding["required_free_bytes"]
        ):
            raise ValueError("Funded node-local output envelope or reserve unavailable")
        extra: list[str] = []
        if mode == "resume":
            assert last is not None
            verify_resumable(out, last)
            extra = ["--resume", str(out / "checkpoint.pt")]
        save(
            journal / f"invocation-{n:02d}-intent.json",
            {
                "arm": arm,
                "invocation": n,
                "mode": mode,
                "job_id": job,
                "device": device,
                "source_git": cpu["source_git"],
                "config_sha256": funding["config_sha256"][arm],
                "resume_checkpoint_sha256": None if last is None else last["checkpoint_sha256"],
                "automatic_retry": False,
            },
        )
        code = execute(config, out, extra, journal / f"invocation-{n:02d}", seconds)
        terminal: dict = {"returncode": code, "job_id": job, "resumable": False}
        if code == 75:
            try:
                invocation = load(out / "invocation.json")
                record = load(out / "run.json")
                validate_binding(record["binding"], arm, values, cpu)
                if (
                    invocation.get("exit_code") == 75
                    and invocation.get("status") in RESUMABLE
                    and invocation.get("checkpoint_sha256") == sha(out / "checkpoint.pt")
                    and invocation.get("binding_sha256") == canonical(record["binding"])
                ):
                    terminal.update(
                        resumable=True,
                        checkpoint_sha256=invocation["checkpoint_sha256"],
                        binding_sha256=invocation["binding_sha256"],
                    )
            except (OSError, ValueError, KeyError):
                pass
        save(journal / f"invocation-{n:02d}-terminal.json", terminal)
        if code == 75 and terminal["resumable"]:
            return 75
        if code != 0:
            raise ValueError("Trainer failed or stopped uncertainly; originals preserved")
        records[arm] = validate_completed(out, arm, values, cpu)
        if tree_bytes(production) > funding["checkpoint_bytes"] + funding["runtime_bytes"]:
            raise ValueError("Funded node-local output envelope exceeded")
    from scripts.newton.run_dense_suffix_recovery import compare_matched

    compare_matched({arm: records[arm]["binding"] for arm in ARMS})
    if not (root / "terminal.json").exists():
        save(
            root / "terminal.json",
            {
                "status": TERMINAL_STATUS,
                "job_id": job,
                "arms": {
                    arm: {
                        "binding_sha256": canonical(records[arm]["binding"]),
                        "checkpoint_sha256": sha(production / arm / "checkpoint.pt"),
                        "prediction_sha256": sha(production / arm / "outer_predictions.csv"),
                    }
                    for arm in ARMS
                },
                "fixed_epochs": EPOCHS,
                "optimizer_updates_per_arm": EPOCHS * UPDATES,
                "endpoint_selection": False,
                "accuracy_claimed": False,
                "paired_fixed_final_analysis_completed": False,
            },
        )
    return 0


def prepare(root: Path = ROOT, admission: Path = ADMISSION, pins: Mapping = PINS) -> dict:
    """CPU-only: materialize the two exact configs for root to hash into its funding."""
    cpu = admitted_cpu(admission, pins)
    source = admission / "code"
    verify_source(source, admission / "recovery-source-manifest.json", cpu, pins)
    if sha(regular(admission / "encoder-spec.json")) != pins["encoder_spec_path_sha256"]:
        raise ValueError("Mounted official base spec changed")
    spec = json.loads((admission / "encoder-spec.json").read_text())
    hashes = {}
    for arm, name in ARMS.items():
        values = production_config(json.loads((source / RECIPE / name).read_text()), spec)
        save(root / f"{arm}-config.json", values)
        hashes[arm] = sha(root / f"{arm}-config.json")
    return {"config_sha256": hashes, "production_funded": False}


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("prepare", "fit"))
    args = parser.parse_args()
    # Import numerical code only from the exactly admitted tree, never the checkout.
    sys.path.insert(0, str(ADMISSION / "code"))
    if args.phase == "prepare":
        if os.environ.get("SLURM_STEP_GPUS") or os.environ.get("SLURM_JOB_GPUS"):
            raise SystemExit("prepare is CPU-only")
        print(json.dumps(prepare(), sort_keys=True), flush=True)
        return
    code = run()
    print(json.dumps({"status": "complete" if code == 0 else "continue_exact_resume"}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
