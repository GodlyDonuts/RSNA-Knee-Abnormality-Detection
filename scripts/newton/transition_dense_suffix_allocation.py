"""Explicit checkpoint-continuous allocation transition for the dense-suffix production.

PREPARED ONLY; root stages this file as ``ROOT/allocation-transitions-v1/coordinator.py``
and alone dispatches, funds and transfers. The original ``ROOT/launch.py`` (5ff1ed77),
its funding (0f5a2d17), ``production-admission.json`` and journals stay byte-identical:
they bind node evc31/parent 899766 and refuse any other placement by design. This
separately versioned record continues the SAME run, never a new one:

``record``  one-CPU, zero-GPU, on the original node inside the original parent, only
            after an ACTUAL original exit-75 terminal. Re-admits everything through the
            pinned original launcher, inspects the complete checkpoint state (model,
            AdamW, LR, counters, RNG, scaler) and writes ``001/record.json`` once with an
            exact member manifest (path/bytes/SHA) of the original root, admission tree,
            public cache, overlay, runtime and shared inputs at their ORIGINAL paths.
``admit``   on the replacement node inside the funded replacement parent, after root's
            separate ``replacement-funding.json`` (and, for relocation, root's
            ``transport-receipt.json``). Every member must exist at the identical path
            with identical bytes/SHA; nothing is rebound or copied here.
``resume``  GPU slice in the replacement allocation. Rechecks scheduler allocation, node
            state (no MAINTENANCE/RESERVED/DRAIN bypass), at most two GPU nodes, the
            assigned CUDA device and byte/file reserve before any intent; then resumes
            only the exact bound checkpoint with the unchanged canonical trainer through
            the original launcher's executor. Journals are additive under ``001/``.
``verify``  read-only source-bound output and immutable epoch-lineage check.

No accuracy claim; funds no GPU; never deletes original state.
"""

from __future__ import annotations

import argparse
import getpass
import importlib.util
import json
import math
import os
import socket
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from pathlib import Path

ROOT = Path("/tmp/rsna-dense-suffix-fit-20261014-v1")
ADMISSION = Path("/tmp/rsna-dense-suffix-admission-20261014-v1")
SCHEMA = "dense-suffix-allocation-transition-v1"
TRANSITIONS = "allocation-transitions-v1"
INDEX = 1
RECORD_STATUS = "original_exit75_checkpoint_transition_recorded"
ADMIT_STATUS = "replacement_destination_members_exactly_admitted"
TERMINAL_STATUS = "matched_dense_suffix_fixed10_completed_across_explicit_allocation_transition"
# Actual original placement/proof (initiative/audits/dense-suffix-full-production-start-20261014-v1).
ORIGINAL = {
    "node": "evc31",
    "parent_job": "899766",
    "funding_sha256": "0f5a2d17487ff59256891594b214d063728a8b16124d0daa097dc0b149dc217d",
    "launcher_sha256": "5ff1ed777c9452e68e7b464ea981040e5448f7a6340eee62bf6ac2716efcc96a",
    "publisher_sha256": "5a5878e07dc689ddc1cc112f0d5beb950bbc2f08587805820d13a0335841f5ab",
    "runtime_root": "/tmp/rsna-environment-870947-4lemnh2d",
    "transitioned_arm": "dense",
}
FORBIDDEN_NODE_STATES = {
    "MAINT",
    "MAINTENANCE",
    "RESERVED",
    "DRAIN",
    "DRAINED",
    "DRAINING",
    "DOWN",
    "FAIL",
    "FAILING",
    "NOT_RESPONDING",
    "POWERED_DOWN",
    "POWERING_DOWN",
    "REBOOT_REQUESTED",
}
MAX_GPU_NODES = 2
# Node-local private members that a relocation must transport exactly; shared /home inputs
# (artifact folds/studies/series and the official base) are verified in place, never copied.
NODE_LOCAL = ("original_root", "admission", "public_cache", "overlay", "runtime", "recovery")
MAX_INVOCATIONS = 12
FUNDING_KEYS = {
    "funded_by",
    "replacement_funded",
    "schema",
    "transition_record_sha256",
    "coordinator_sha256",
    "original_funding_sha256",
    "original_production_admission_sha256",
    "CPU_receipt_sha256",
    "predecessor_job",
    "node",
    "parent_job",
    "assigned_GPU",
    "CPUs",
    "relocation",
    "transport_receipt_sha256",
    "runtime_python",
    "max_slice_seconds",
    "max_invocations_per_arm",
    "destination_envelope_bytes",
    "existing_liabilities_bytes",
    "reserve_bytes",
    "required_free_bytes",
    "reserve_files",
    "shared_future_bytes",
}
TRANSPORT_KEYS = {
    "executed_by",
    "transition_record_sha256",
    "manifest_sha256",
    "source_node",
    "destination_node",
    "members",
    "bytes",
    "original_state_deleted",
    "automatic_retry",
}
Scheduler = Callable[[str, str], dict]


def _load(path: Path, name: str, expected: str):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Pinned {name} must be a regular file")
    import hashlib

    if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise ValueError(f"Pinned original {name} bytes changed")
    sys.dont_write_bytecode = True
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def original_launcher(root: Path, original: Mapping):
    return _load(root / "launch.py", "original_dense_suffix_launcher", original["launcher_sha256"])


def tdir(root: Path, index: int = INDEX) -> Path:
    return root / TRANSITIONS / f"{index:03d}"


# ---------------------------------------------------------------- manifest ----


def _walk(base: Path) -> list[Path]:
    if base.is_symlink() or not base.exists():
        raise ValueError(f"Original member root missing or linked: {base}")
    if base.is_file():
        return [base]
    items = []
    for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
        here = Path(dirpath)
        for name in sorted(dirnames):
            if (here / name).is_symlink():
                items.append(here / name)
        items.extend(here / name for name in sorted(filenames))
    return sorted(items)


def member(launch, path: Path, cls: str) -> dict:
    if path.is_symlink():
        return {"class": cls, "path": str(path), "kind": "symlink", "target": os.readlink(path)}
    if not path.is_file():
        raise ValueError(f"Unsupported original member type: {path}")
    return {
        "class": cls,
        "path": str(path),
        "kind": "file",
        "bytes": path.stat().st_size,
        "sha256": launch.sha(path),
        "executable": os.access(path, os.X_OK),
    }


def members(launch, roots: Mapping[str, Path], exclude: Path) -> list[dict]:
    out = []
    for cls, base in roots.items():
        for path in _walk(base):
            if path == exclude or exclude in path.parents:
                continue
            out.append(member(launch, path, cls))
    paths = [m["path"] for m in out]
    if len(paths) != len(set(paths)):
        raise ValueError("Manifest classes overlap")
    return out


def member_roots(state: Mapping, root: Path, admission: Path, runtime: Path) -> dict:
    inputs = state["inputs"]
    artifact = Path(inputs["artifact-root"])
    spec = json.loads((admission / "encoder-spec.json").read_text())
    return {
        "original_root": root,
        "admission": admission,
        "public_cache": Path(inputs["cache"]),
        "overlay": Path(inputs["overlay"]),
        "runtime": runtime,
        "recovery": Path(state["funding"]["GPU_recovery_output"]),
        "shared_studies": artifact / "initiative/releases/report-supervision-v1/studies.parquet",
        "shared_folds": artifact / "initiative/experiments/competitive-v1/folds",
        "shared_series": artifact
        / "initiative/audits/reviewed-series-preferences-v1/export-001/series_sampling.csv",
        "shared_base_weights": Path(spec["weights_path"]),
        "shared_base_config": Path(spec["config_path"]),
    }


def verify_members(
    launch,
    manifest: list[dict],
    classes: set[str] | None = None,
    roots: Mapping[str, str] | None = None,
    exclude: Path | None = None,
) -> None:
    """Exact bytes/SHA at the identical original path; with roots, no extra member."""
    wanted = [m for m in manifest if classes is None or m["class"] in classes]
    if roots is not None:
        for cls in {m["class"] for m in wanted}:
            have = {
                str(p)
                for p in _walk(Path(roots[cls]))
                if exclude is None or not (p == exclude or exclude in p.parents)
            }
            if have != {m["path"] for m in wanted if m["class"] == cls}:
                raise ValueError(f"Destination {cls} members differ from the original set")
    for m in wanted:
        path = Path(m["path"])
        if m["kind"] == "symlink":
            if not path.is_symlink() or os.readlink(path) != m["target"]:
                raise ValueError(f"Destination link differs from original: {path}")
            continue
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size != m["bytes"]
            or launch.sha(path) != m["sha256"]
            or os.access(path, os.X_OK) != m["executable"]
        ):
            raise ValueError(f"Destination member differs from original bytes: {path}")


# ---------------------------------------------------------------- journals ----


def arm_invocations(root: Path, arm: str, index: int = INDEX) -> list[tuple[int, Path, Path]]:
    """(n, intent, terminal) over the original journal then the transition journal."""
    rows = []
    for journal in (root / "journal" / arm, tdir(root, index) / "journal" / arm):
        for intent in sorted(journal.glob("invocation-*-intent.json")):
            n = int(intent.name.split("-")[1])
            rows.append((n, intent, journal / f"invocation-{n:02d}-terminal.json"))
    if [n for n, _, _ in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("Invocation sequence across original/transition journals not contiguous")
    for _, _, terminal in rows:
        if not terminal.exists():
            raise ValueError("Intent without terminal record is uncertain replay; never retry")
    return rows


def original_exit75(launch, root: Path, original: Mapping) -> dict:
    """ONLY an actual original resumable exit-75 terminal of the transitioned arm."""
    arm = original["transitioned_arm"]
    journal = root / "journal" / arm
    intents = sorted(journal.glob("invocation-*-intent.json"))
    if not intents:
        raise ValueError("No original invocation; an absent terminal is never transitioned")
    rows = arm_invocations(root, arm)
    n, intent_path, terminal_path = rows[-1]
    if intent_path.parent != journal:
        raise ValueError("Transition journal already continues this arm; duplicate transition")
    terminal, intent = launch.load(terminal_path), launch.load(intent_path)
    if (
        terminal.get("returncode") != 75
        or terminal.get("resumable") is not True
        or str(terminal.get("job_id")) != original["parent_job"]
        or str(intent.get("job_id")) != original["parent_job"]
        or intent.get("arm") != arm
    ):
        raise ValueError("Actual original resumable exit-75 terminal required; none admitted")
    for later in list(launch.ARMS)[list(launch.ARMS).index(arm) + 1 :]:
        if (
            any((root / "journal" / later).glob("invocation-*"))
            or (root / "production" / later).exists()
        ):
            raise ValueError("Later arm already started; transition is not sequential")
    return {"arm": arm, "invocation": n, "intent": intent, "terminal": terminal}


def checkpoint_state(path: Path) -> dict:
    """Complete exact-resume state summary of the canonical frozen-base-delta checkpoint."""
    import hashlib

    import torch

    c = torch.load(path, map_location="cpu", weights_only=True)
    optimizer = c.get("optimizer") or {}
    groups = optimizer.get("param_groups") or []
    return {
        "schema_version": c.get("schema_version"),
        "binding_sha256": hashlib.sha256(
            json.dumps(c.get("binding"), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "optimizer_boundary": c.get("optimizer_boundary"),
        "optimizer_state_entries": len(optimizer.get("state") or {}),
        "learning_rates": [g.get("lr") for g in groups],
        "amp_scaler": "amp_scaler" in c,
        "rng_streams": len(c.get("rng_by_rank") or []),
        "epoch": c.get("epoch"),
        "next_update": c.get("next_update"),
        "step": c.get("step"),
    }


def checkpoint_schema(binding: Mapping) -> int:
    fmt = binding.get("config", {}).get("checkpoint_format")
    return 2 if fmt == "frozen_base_delta" else 1


def validate_checkpoint_state(launch, state: Mapping, binding_sha: str, schema: int) -> None:
    counters = (state.get("epoch"), state.get("next_update"), state.get("step"))
    if any(type(v) is not int or v < 0 for v in counters):
        raise ValueError("Checkpoint counters invalid; exact resume refused")
    epoch, update, step = counters
    if step == 0:
        raise ValueError("Initial-state checkpoint; restart from step 0 is never a transition")
    rates = state.get("learning_rates") or []
    if (
        state.get("schema_version") != schema
        or state.get("binding_sha256") != binding_sha
        or state.get("optimizer_boundary") is not True
        or not state.get("optimizer_state_entries")
        or not rates
        or not all(isinstance(r, float) and math.isfinite(r) and r >= 0 for r in rates)
        or state.get("amp_scaler") is not True
        or state.get("rng_streams") != 1
        or step != epoch * launch.UPDATES + update
        or not 0 < step < launch.EPOCHS * launch.UPDATES
    ):
        raise ValueError("Incomplete AdamW/LR/counter/RNG/scaler checkpoint; resume refused")


# ------------------------------------------------------------------ record ----


def cpu_placement(node: str, job: str) -> None:
    if (
        socket.gethostname().split(".")[0] != node
        or os.environ.get("SLURM_JOB_ID") != str(job)
        or os.environ.get("SLURM_STEP_GPUS")
    ):
        raise ValueError("Genuine zero-GPU step on the required node/parent required")


def record(
    root: Path = ROOT,
    admission: Path = ADMISSION,
    *,
    original: Mapping = ORIGINAL,
    pins: Mapping | None = None,
    placement: Callable[[str, str], None] = cpu_placement,
    inspect: Callable[[Path], dict] = checkpoint_state,
    coordinator: Path | None = None,
) -> dict:
    launch = original_launcher(root, original)
    pins = launch.PINS if pins is None else pins
    if tdir(root, INDEX + 1).exists():
        raise ValueError("Only transition 001 is versioned here; a further one needs review")
    target = tdir(root) / "record.json"
    if target.exists():
        raise ValueError("Transition already recorded; duplicate transition refused")
    placement(original["node"], original["parent_job"])
    actual = original_exit75(launch, root, original)
    if launch.sha(root / "production-funding.json") != original["funding_sha256"]:
        raise ValueError("Original production funding bytes changed")
    state = launch.admitted(root, admission, pins)
    if (
        state["funding"]["launcher_sha256"] != original["launcher_sha256"]
        or state["funding"]["publisher_sha256"] != original["publisher_sha256"]
        or str(state["funding"]["parent_job"]) != original["parent_job"]
        or state["funding"]["node"] != original["node"]
        or launch.load(root / "production-admission.json")
        != launch.static_admission(root, admission, state)
    ):
        raise ValueError("Original production admission/funding/placement proof differs")
    arm, terminal = actual["arm"], actual["terminal"]
    out = root / "production" / arm
    launch.verify_resumable(out, terminal)
    run_record = launch.load(out / "run.json")
    launch.validate_binding(run_record["binding"], arm, state["configs"][arm], state["cpu"])
    complete = inspect(out / "checkpoint.pt")
    schema = checkpoint_schema(run_record["binding"])
    validate_checkpoint_state(launch, complete, terminal["binding_sha256"], schema)
    runtime = Path(original["runtime_root"])
    if runtime not in Path(sys.executable).parents:
        raise ValueError("Record must run under the original pinned runtime")
    roots = member_roots(state, root, admission, runtime)
    manifest = members(launch, roots, root / TRANSITIONS)
    coordinator = Path(__file__) if coordinator is None else coordinator
    value = {
        "schema": SCHEMA,
        "status": RECORD_STATUS,
        "index": INDEX,
        "original": {
            "node": original["node"],
            "parent_job": original["parent_job"],
            "funding_sha256": original["funding_sha256"],
            "production_admission_sha256": launch.sha(root / "production-admission.json"),
            "CPU_receipt_sha256": state["funding"]["CPU_receipt_sha256"],
            "launcher_sha256": original["launcher_sha256"],
            "publisher_sha256": original["publisher_sha256"],
            "source_git": state["cpu"]["source_git"],
            "config_sha256": state["funding"]["config_sha256"],
        },
        "arm": arm,
        "terminal_invocation": actual["invocation"],
        "terminal_sha256": launch.sha(
            root / "journal" / arm / f"invocation-{actual['invocation']:02d}-terminal.json"
        ),
        "checkpoint_sha256": terminal["checkpoint_sha256"],
        "checkpoint_bytes": (out / "checkpoint.pt").stat().st_size,
        "binding_sha256": terminal["binding_sha256"],
        "run_json_sha256": launch.sha(out / "run.json"),
        "invocation_json_sha256": launch.sha(out / "invocation.json"),
        "checkpoint_schema": schema,
        "complete_state": complete,
        "member_roots": {k: str(v) for k, v in roots.items()},
        "manifest": manifest,
        "manifest_sha256": launch.canonical(manifest),
        "members": len(manifest),
        "member_bytes": sum(m.get("bytes", 0) for m in manifest),
        "transport_members": sum(m["class"] in NODE_LOCAL for m in manifest),
        "transport_bytes": sum(m.get("bytes", 0) for m in manifest if m["class"] in NODE_LOCAL),
        "transport_classes": list(NODE_LOCAL),
        "coordinator_sha256": launch.sha(coordinator),
        "path_aliases_retained": True,
        "original_state_deleted": False,
        "GPU_intent": False,
        "accuracy_claimed": False,
    }
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    launch.save(target, value)
    return {k: v for k, v in value.items() if k != "manifest"}


# --------------------------------------------------------- funding & admit ----


def recorded(root: Path, original: Mapping) -> tuple:
    launch = original_launcher(root, original)
    path = tdir(root) / "record.json"
    value = launch.load(path)
    if (
        value.get("schema") != SCHEMA
        or value.get("status") != RECORD_STATUS
        or value.get("index") != INDEX
        or value.get("original", {}).get("funding_sha256") != original["funding_sha256"]
        or value.get("original", {}).get("parent_job") != original["parent_job"]
        or launch.canonical(value.get("manifest")) != value.get("manifest_sha256")
    ):
        raise ValueError("Exact transition record required")
    return launch, value, launch.sha(path)


def replacement_funding(launch, root: Path, value: Mapping, record_sha: str, coordinator) -> dict:
    funding = launch.load(tdir(root) / "replacement-funding.json")
    if set(funding) != FUNDING_KEYS:
        raise ValueError("Complete explicit root replacement funding required")
    prior = value["original"]
    if (
        funding["funded_by"] != "root"
        or funding["replacement_funded"] is not True
        or funding["schema"] != SCHEMA
        or funding["transition_record_sha256"] != record_sha
        or funding["coordinator_sha256"] != value["coordinator_sha256"]
        or launch.sha(coordinator) != funding["coordinator_sha256"]
        or funding["original_funding_sha256"] != prior["funding_sha256"]
        or funding["original_production_admission_sha256"] != prior["production_admission_sha256"]
        or funding["CPU_receipt_sha256"] != prior["CPU_receipt_sha256"]
        or funding["predecessor_job"] != prior["parent_job"]
    ):
        raise ValueError("Replacement funding binds a stale record/admission/funding")
    if (
        not str(funding["parent_job"]).isdigit()
        or str(funding["parent_job"]) == prior["parent_job"]
        or not funding["node"]
        or not funding["assigned_GPU"]
        or type(funding["CPUs"]) is not int
        or funding["CPUs"] < 1
        or type(funding["relocation"]) is not bool
        or funding["relocation"] != (funding["node"] != prior["node"])
        or (funding["transport_receipt_sha256"] is None) == funding["relocation"]
        or Path(value["member_roots"]["runtime"]) not in Path(funding["runtime_python"]).parents
    ):
        raise ValueError("Stale or silently rebound replacement placement/runtime refused")
    if (
        type(funding["max_slice_seconds"]) is not int
        or funding["max_slice_seconds"] < launch.MIN_START_SECONDS
        or type(funding["max_invocations_per_arm"]) is not int
        or not value["terminal_invocation"] < funding["max_invocations_per_arm"] <= MAX_INVOCATIONS
        or funding["reserve_bytes"] != launch.RESERVE
        or funding["shared_future_bytes"] != 0
        or type(funding["reserve_files"]) is not int
        or funding["reserve_files"] < 1
        or funding["existing_liabilities_bytes"] <= 0
        or funding["destination_envelope_bytes"] <= 0
        or funding["required_free_bytes"]
        != funding["existing_liabilities_bytes"]
        + funding["destination_envelope_bytes"]
        + funding["reserve_bytes"]
    ):
        raise ValueError("Explicit replacement slice/byte/file reserve funding required")
    return funding


def transport_receipt(launch, root: Path, value: Mapping, record_sha: str, funding) -> None:
    path = tdir(root) / "transport-receipt.json"
    if not funding["relocation"]:
        if path.exists():
            raise ValueError("Owning-node replacement reuses inputs; no transport expected")
        return
    if launch.sha(path) != funding["transport_receipt_sha256"]:
        raise ValueError("Root transport receipt bytes differ from replacement funding")
    receipt = launch.load(path)
    if set(receipt) != TRANSPORT_KEYS or receipt != {
        "executed_by": "root",
        "transition_record_sha256": record_sha,
        "manifest_sha256": value["manifest_sha256"],
        "source_node": value["original"]["node"],
        "destination_node": funding["node"],
        "members": value["transport_members"],
        "bytes": value["transport_bytes"],
        "original_state_deleted": False,
        "automatic_retry": False,
    }:
        raise ValueError("Explicit exact transport of every original member required")


def owned_bytes(launch, value: Mapping, relocation: bool) -> int:
    roots = value["member_roots"]
    names = list(NODE_LOCAL) if relocation else ["original_root"]
    return sum(launch.tree_bytes(Path(roots[n])) for n in names)


def reserve_ok(launch, root: Path, value: Mapping, funding: Mapping) -> None:
    import shutil

    used = owned_bytes(launch, value, funding["relocation"])
    stats = os.statvfs(root)
    if (
        used > funding["destination_envelope_bytes"]
        or shutil.disk_usage(root).free + used < funding["required_free_bytes"]
        or stats.f_favail < funding["reserve_files"]
    ):
        raise ValueError("Funded destination byte/file reserve unavailable")


def admit(
    root: Path = ROOT,
    *,
    original: Mapping = ORIGINAL,
    placement: Callable[[str, str], None] = cpu_placement,
    coordinator: Path | None = None,
) -> dict:
    coordinator = Path(__file__) if coordinator is None else coordinator
    launch, value, record_sha = recorded(root, original)
    funding = replacement_funding(launch, root, value, record_sha, coordinator)
    target = tdir(root) / "destination-admission.json"
    if target.exists():
        raise ValueError("Destination already admitted; never repeated")
    placement(funding["node"], str(funding["parent_job"]))
    transport_receipt(launch, root, value, record_sha, funding)
    verify_members(launch, value["manifest"], None, value["member_roots"], root / TRANSITIONS)
    reserve_ok(launch, root, value, funding)
    result = {
        "status": ADMIT_STATUS,
        "transition_record_sha256": record_sha,
        "replacement_funding_sha256": launch.sha(tdir(root) / "replacement-funding.json"),
        "manifest_sha256": value["manifest_sha256"],
        "node": funding["node"],
        "parent_job": str(funding["parent_job"]),
        "relocation": funding["relocation"],
        "members_verified": value["members"],
        "bytes_verified": value["member_bytes"],
        "GPU_intent": False,
    }
    launch.save(target, result)
    return result


# ------------------------------------------------------------------ resume ----


def _states(raw) -> set[str]:
    if isinstance(raw, str):
        raw = raw.split("+")
    return {str(s).upper() for s in raw or []}


def scheduler_facts(job: str, node: str) -> dict:
    def show(kind: str, name: str) -> dict:
        text = subprocess.check_output(["scontrol", "show", kind, name, "--json"], text=True)
        return json.loads(text)[f"{kind}s"][0]

    record, host = show("job", job), show("node", node)
    end = record["end_time"]
    rows = subprocess.check_output(
        ["squeue", "-h", "-u", getpass.getuser(), "-t", "RUNNING", "-o", "%i|%b|%N"], text=True
    ).splitlines()
    gpu_nodes = set()
    for row in rows:
        _, tres, nodes = row.split("|")
        if "gpu" in tres.lower():
            if "[" in nodes or "," in nodes:
                raise ValueError("Unparsed multi-node GPU allocation; refuse")
            gpu_nodes.add(nodes)
    return {
        "job_state": _states(record["job_state"]),
        "job_nodes": record["nodes"],
        "end_time": float(end["number"] if isinstance(end, dict) else end),
        "node_state": _states(host["state"]),
        "gpu_nodes": gpu_nodes,
    }


def check_allocation(launch, funding: Mapping, facts: Mapping, now: float) -> float:
    node = funding["node"]
    if (
        facts["job_state"] != {"RUNNING"}
        or facts["job_nodes"] != node
        or facts["end_time"] - now < launch.MIN_START_SECONDS
    ):
        raise ValueError("Genuine RUNNING replacement scheduler allocation required")
    if facts["node_state"] & FORBIDDEN_NODE_STATES:
        raise ValueError("Node MAINTENANCE/RESERVED/DRAIN state is never bypassed")
    if len(set(facts["gpu_nodes"]) | {node}) > MAX_GPU_NODES:
        raise ValueError("No more than two concurrent GPU nodes")
    return facts["end_time"] - now


def gpu_placement(funding: Mapping) -> str:
    import torch

    if (
        socket.gethostname().split(".")[0] != funding["node"]
        or os.environ.get("SLURM_JOB_ID") != str(funding["parent_job"])
        or not os.environ.get("SLURM_STEP_ID")
        or os.environ.get("SLURM_CPUS_PER_TASK") != str(funding["CPUs"])
        or (os.environ.get("SLURM_STEP_GPUS") or os.environ.get("SLURM_JOB_GPUS"))
        != str(funding["assigned_GPU"])
        or not os.environ.get("CUDA_VISIBLE_DEVICES")
        or sys.executable != funding["runtime_python"]
        or torch.cuda.device_count() != 1
        or "H100" not in torch.cuda.get_device_name(0)
    ):
        raise ValueError("Funded replacement step with exactly one assigned H100 required")
    return torch.cuda.get_device_name(0)


def verify_original_lineage(launch, root: Path, value: Mapping) -> None:
    """Original journals, controls and immutable epoch snapshots stay byte-identical."""
    out = f"{root}/production/"
    fixed = [
        m
        for m in value["manifest"]
        if m["class"] == "original_root"
        and (not m["path"].startswith(out) or "/epoch-checkpoints/" in m["path"])
    ]
    verify_members(launch, fixed)
    have = {str(p) for p in (root / "journal").rglob("*") if p.is_file()}
    want = {m["path"] for m in fixed if m["path"].startswith(f"{root}/journal/")}
    if have != want:
        raise ValueError("Original journal changed after the transition record")


def run(
    root: Path = ROOT,
    *,
    original: Mapping = ORIGINAL,
    slice_seconds: float,
    execute=None,
    placement: Callable[[Mapping], str] = gpu_placement,
    scheduler: Scheduler = scheduler_facts,
    inspect: Callable[[Path], dict] = checkpoint_state,
    coordinator: Path | None = None,
) -> int:
    coordinator = Path(__file__) if coordinator is None else coordinator
    launch, value, record_sha = recorded(root, original)
    funding = replacement_funding(launch, root, value, record_sha, coordinator)
    admitted = launch.load(tdir(root) / "destination-admission.json")
    if (
        admitted.get("status") != ADMIT_STATUS
        or admitted.get("transition_record_sha256") != record_sha
        or admitted.get("replacement_funding_sha256")
        != launch.sha(tdir(root) / "replacement-funding.json")
    ):
        raise ValueError("Exact destination admission for this funding required")
    if not (math.isfinite(slice_seconds) and launch.MIN_START_SECONDS <= slice_seconds) or (
        slice_seconds > funding["max_slice_seconds"]
    ):
        raise ValueError("Finite slice within root's funded replacement bound required")
    transport_receipt(launch, root, value, record_sha, funding)
    verify_original_lineage(launch, root, value)
    verify_members(launch, value["manifest"], {"admission", "overlay"}, value["member_roots"])
    facts = scheduler(str(funding["parent_job"]), funding["node"])
    remaining = check_allocation(launch, funding, facts, time.time())
    device = placement(funding)
    reserve_ok(launch, root, value, funding)
    cpu = launch.load(Path(value["member_roots"]["admission"]) / "CPU-accepted.json")
    source = Path(value["member_roots"]["admission"]) / "code"
    inputs = cpu["GPU_wrapper_arguments"]
    state = {"funding": {"CPUs": funding["CPUs"]}, "source": source, "inputs": inputs}
    execute = execute or launch.subprocess_executor(state)
    job = str(funding["parent_job"])
    deadline = time.monotonic() + min(remaining, slice_seconds)
    tj = tdir(root) / "journal"
    records = {}
    for arm in launch.ARMS:
        config = root / f"{arm}-config.json"
        values = json.loads(config.read_text())
        out = root / "production" / arm
        rows = arm_invocations(root, arm)
        last = launch.load(rows[-1][2]) if rows else None
        if last is not None and last.get("returncode") == 0:
            records[arm] = launch.validate_completed(out, arm, values, cpu)
            continue
        if records.keys() != set(list(launch.ARMS)[: list(launch.ARMS).index(arm)]):
            raise ValueError("Arms run sequentially; a later arm requires its predecessor")
        n = len(rows) + 1
        extra: list[str] = []
        if arm == value["arm"]:
            if last is None or last.get("returncode") != 75 or last.get("resumable") is not True:
                raise ValueError("Terminal trainer failure preserved; no automatic retry")
            if n == value["terminal_invocation"] + 1:
                if (
                    launch.sha(rows[-1][2]) != value["terminal_sha256"]
                    or launch.sha(out / "run.json") != value["run_json_sha256"]
                    or launch.sha(out / "invocation.json") != value["invocation_json_sha256"]
                    or last.get("checkpoint_sha256") != value["checkpoint_sha256"]
                ):
                    raise ValueError("Predecessor exit-75 bytes changed after the record")
            launch.verify_resumable(out, last)
            validate_checkpoint_state(
                launch,
                inspect(out / "checkpoint.pt"),
                value["binding_sha256"],
                value["checkpoint_schema"],
            )
            extra = ["--resume", str(out / "checkpoint.pt")]
        elif rows or out.exists():
            if last is None or last.get("returncode") != 75 or last.get("resumable") is not True:
                raise ValueError("Terminal trainer failure preserved; no automatic retry")
            launch.verify_resumable(out, last)
            extra = ["--resume", str(out / "checkpoint.pt")]
        if n > funding["max_invocations_per_arm"]:
            raise ValueError("Bounded exact-resume count exhausted")
        seconds = deadline - time.monotonic()
        if seconds < launch.MIN_START_SECONDS:
            return 75
        if extra and arm == value["arm"] and n == value["terminal_invocation"] + 1:
            preserved = tdir(root) / f"exit75-checkpoint-{value['terminal_invocation']:02d}.pt"
            if not preserved.exists():
                os.link(out / "checkpoint.pt", preserved)
            if launch.sha(preserved) != value["checkpoint_sha256"]:
                raise ValueError("Preserved original exit-75 checkpoint differs")
        (tj / arm).mkdir(mode=0o700, parents=True, exist_ok=True)
        launch.save(
            tj / arm / f"invocation-{n:02d}-intent.json",
            {
                "arm": arm,
                "invocation": n,
                "mode": "resume" if extra else "fresh",
                "job_id": job,
                "predecessor_job": funding["predecessor_job"],
                "transition_record_sha256": record_sha,
                "device": device,
                "source_git": value["original"]["source_git"],
                "config_sha256": value["original"]["config_sha256"][arm],
                "resume_checkpoint_sha256": last["checkpoint_sha256"] if extra and last else None,
                "slice_seconds": seconds,
                "automatic_retry": False,
            },
        )
        code = execute(config, out, extra, tj / arm / f"invocation-{n:02d}", seconds)
        terminal: dict = {"returncode": code, "job_id": job, "resumable": False}
        if code == 75:
            try:
                invocation, run_record = (
                    launch.load(out / "invocation.json"),
                    launch.load(out / "run.json"),
                )
                launch.validate_binding(run_record["binding"], arm, values, cpu)
                if (
                    invocation.get("exit_code") == 75
                    and invocation.get("status") in launch.RESUMABLE
                    and invocation.get("checkpoint_sha256") == launch.sha(out / "checkpoint.pt")
                    and invocation.get("binding_sha256") == launch.canonical(run_record["binding"])
                ):
                    terminal.update(
                        resumable=True,
                        checkpoint_sha256=invocation["checkpoint_sha256"],
                        binding_sha256=invocation["binding_sha256"],
                    )
            except (OSError, ValueError, KeyError):
                pass
        launch.save(tj / arm / f"invocation-{n:02d}-terminal.json", terminal)
        if code == 75 and terminal["resumable"]:
            return 75
        if code != 0:
            raise ValueError("Trainer failed or stopped uncertainly; originals preserved")
        records[arm] = launch.validate_completed(out, arm, values, cpu)
        reserve_ok(launch, root, value, funding)
    verify(root, original=original)
    from scripts.newton.run_dense_suffix_recovery import compare_matched

    compare_matched({arm: records[arm]["binding"] for arm in launch.ARMS})
    final = tdir(root) / "terminal.json"
    if not final.exists():
        launch.save(
            final,
            {
                "status": TERMINAL_STATUS,
                "original_job": value["original"]["parent_job"],
                "replacement_job": job,
                "transition_record_sha256": record_sha,
                "arms": {
                    arm: {
                        "binding_sha256": launch.canonical(records[arm]["binding"]),
                        "checkpoint_sha256": launch.sha(
                            root / "production" / arm / "checkpoint.pt"
                        ),
                        "prediction_sha256": launch.sha(
                            root / "production" / arm / "outer_predictions.csv"
                        ),
                    }
                    for arm in launch.ARMS
                },
                "fixed_epochs": launch.EPOCHS,
                "optimizer_updates_per_arm": launch.EPOCHS * launch.UPDATES,
                "endpoint_selection": False,
                "accuracy_claimed": False,
                "paired_fixed_final_analysis_completed": False,
            },
        )
    return 0


def verify(root: Path = ROOT, *, original: Mapping = ORIGINAL) -> dict:
    """Read-only: source-bound output and immutable epoch publication lineage."""
    launch, value, record_sha = recorded(root, original)
    verify_original_lineage(launch, root, value)
    lineage = {}
    for arm in launch.ARMS:
        out = root / "production" / arm
        if not (out / "run.json").exists():
            continue
        bound = launch.canonical(launch.load(out / "run.json")["binding"])
        if arm == value["arm"] and bound != value["binding_sha256"]:
            raise ValueError("Transitioned run binding differs from the original source-bound run")
        epochs = {}
        for meta in sorted((out / "epoch-checkpoints").glob("epoch-*.json")):
            snap = launch.load(meta)
            if (
                type(snap.get("epoch")) is not int
                or snap.get("binding_sha256") != bound
                or snap.get("optimizer_steps") != launch.UPDATES * snap["epoch"]
                or snap.get("checkpoint_sha256")
                != launch.sha(launch.regular(meta.with_suffix(".pt")))
            ):
                raise ValueError("Epoch snapshot is not bound to the original source/run")
            epochs[snap["epoch"]] = snap["checkpoint_sha256"]
        if list(epochs) != list(range(1, len(epochs) + 1)):
            raise ValueError("Immutable epoch lineage is not contiguous")
        lineage[arm] = epochs
    return {"transition_record_sha256": record_sha, "epochs": lineage}


def replacement_cpu(funding: Mapping) -> Callable[[str], None]:
    def placement(node: str) -> None:
        if (
            node != ORIGINAL["node"]
            or socket.gethostname().split(".")[0] != funding["node"]
            or os.environ.get("SLURM_JOB_ID") != str(funding["parent_job"])
            or os.environ.get("SLURM_CPUS_PER_TASK") != "1"
            or os.environ.get("SLURM_JOB_GPUS")
            or os.environ.get("SLURM_STEP_GPUS")
        ):
            raise ValueError("Replacement-node one-CPU zero-GPU publication required")

    return placement


def publish(
    arm: str,
    epoch: int,
    root: Path = ROOT,
    admission: Path = ADMISSION,
    *,
    original: Mapping = ORIGINAL,
    coordinator: Path | None = None,
    placement: Callable[[Mapping], Callable[[str], None]] = replacement_cpu,
    **kwargs,
) -> dict:
    """Root-run: the unchanged pinned publisher, with only its owning-node placement bound to
    the admitted replacement node. Its journal, prefix, admission and checks are original."""
    coordinator = Path(__file__) if coordinator is None else coordinator
    launch, value, record_sha = recorded(root, original)
    funding = replacement_funding(launch, root, value, record_sha, coordinator)
    admitted = launch.load(tdir(root) / "destination-admission.json")
    if admitted.get("transition_record_sha256") != record_sha:
        raise ValueError("Exact destination admission required before publication")
    lineage = verify(root, original=original)["epochs"]
    if epoch not in lineage.get(arm, {}):
        raise ValueError("Only an immutable source-bound completed epoch is published")
    pub = _load(
        root / "publish.py", "original_dense_suffix_publisher", original["publisher_sha256"]
    )
    return pub.publish(arm, epoch, root, admission, placement=placement(funding), **kwargs)


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("record", "admit", "resume", "verify", "publish"))
    parser.add_argument("--slice-seconds", type=float)
    parser.add_argument("--arm")
    parser.add_argument("--epoch", type=int)
    args = parser.parse_args()
    sys.path.insert(0, str(ADMISSION / "code"))
    if args.phase in {"record", "admit", "verify"}:
        if os.environ.get("SLURM_STEP_GPUS"):
            raise SystemExit(f"{args.phase} is zero-GPU")
        phases: dict[str, Callable[[], dict]] = {"record": record, "admit": admit, "verify": verify}
        result = phases[args.phase]()
        print(json.dumps(result, sort_keys=True), flush=True)
        return
    if args.phase == "publish":
        print(json.dumps(publish(args.arm, args.epoch), sort_keys=True), flush=True)
        return
    if args.slice_seconds is None:
        raise SystemExit("resume requires an explicit --slice-seconds")
    code = run(slice_seconds=args.slice_seconds)
    print(json.dumps({"status": "complete" if code == 0 else "continue_exact_resume"}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
