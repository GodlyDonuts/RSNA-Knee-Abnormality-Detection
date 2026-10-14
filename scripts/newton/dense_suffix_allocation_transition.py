"""Checkpoint-continuous allocation transition for the live dense-suffix production.

PREPARED ONLY; dispatches nothing. Root stages this file as
``ROOT/allocation-transitions/coordinator.py`` and writes, per transition ``k``,
``ROOT/allocation-transitions/transition-kk/replacement-funding.json`` for one
separately funded replacement Slurm parent. The original launcher
(``run_dense_suffix_production.py``), its funding, static production admission,
journals, configs, checkpoints and source bytes stay exactly as written: they are
re-admitted unchanged and never rewritten, deleted or rebound.

``admit`` (never touches CUDA) writes one exclusive ``transition.json`` that binds ONLY an actual
predecessor exit-75 resumable terminal: its exact terminal/journal bytes, checkpoint,
run/invocation records, binding, run/source/config/roster/base/runtime admission and
the prior funding/admission hashes, to the replacement placement. ``fit`` runs inside
the replacement parent: before any resumed GPU intent it rechecks the genuine
scheduler allocation, node state (MAINTENANCE/RESERVED/DRAIN never bypassed), the
two-GPU-node bound, the assigned H100 and the byte/file reserve, then resumes only that
exact checkpoint through the unchanged canonical trainer (``--resume``: AdamW/LR/
counters/RNG are restored by the trainer itself). New invocations are journaled under
the transition only. An owning-node replacement reuses private inputs in place;
relocation requires root's explicit exact transport manifest plus destination
admission at the same absolute paths. No restart from step 0, no uncertain terminal,
no automatic retry, no accuracy claim and no additional GPU funding.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

ROOT = Path("/tmp/rsna-dense-suffix-fit-20261014-v1")
ADMISSION = Path("/tmp/rsna-dense-suffix-admission-20261014-v1")
VERSION = "dense_suffix_allocation_transition_v1"
TRANSITIONS = "allocation-transitions"
COORDINATOR = "coordinator.py"
FUNDING = "replacement-funding.json"
RECORD = "transition.json"
TRANSPORT = "transport.json"
DESTINATION = "destination-admission.json"
ADMITTED_STATUS = "dense_suffix_exit75_checkpoint_allocation_transition_admitted"
DESTINATION_STATUS = "root_destination_exact_transport_admitted"
SEGMENT_TERMINAL_STATUS = "matched_dense_suffix_fixed10_completed_across_allocation_transitions"
# Actual original production (initiative/audits/dense-suffix-full-production-start-20261014-v1).
ORIGINAL = {
    "parent_job": "899766",
    "node": "evc31",
    "funding_sha256": "0f5a2d17487ff59256891594b214d063728a8b16124d0daa097dc0b149dc217d",
    "launcher_sha256": "5ff1ed777c9452e68e7b464ea981040e5448f7a6340eee62bf6ac2716efcc96a",
    "publisher_sha256": "5a5878e07dc689ddc1cc112f0d5beb950bbc2f08587805820d13a0335841f5ab",
}
MAX_TRANSITIONS = 4
MAX_GPU_NODES = 2
SMALL_FILE_BYTES = 64 * 2**20
BLOCKED_NODE_STATES = {
    "MAINTENANCE",
    "RESERVED",
    "DRAIN",
    "DRAINING",
    "DRAINED",
    "DOWN",
    "FAIL",
    "FAILING",
    "NOT_RESPONDING",
    "NO_RESPOND",
    "POWERED_DOWN",
    "POWERING_DOWN",
    "REBOOT_REQUESTED",
    "REBOOT_ISSUED",
    "UNKNOWN",
}
LIVE_JOB_STATES = {"RUNNING", "PENDING", "SUSPENDED", "CONFIGURING", "COMPLETING", "REQUEUED"}
PLACEMENTS = {"owning_node", "relocation"}
FUNDING_KEYS = {
    "funded_by",
    "transition_version",
    "transition",
    "prior_funding_sha256",
    "prior_admission_sha256",
    "prior_transition_sha256",
    "supersedes_unused_transition",
    "coordinator_sha256",
    "placement",
    "parent_job",
    "node",
    "assigned_GPU",
    "CPUs",
    "max_slice_seconds",
    "node_local_envelope_bytes",
    "reserve_bytes",
    "required_free_bytes",
    "required_free_files",
    "shared_future_bytes",
    "additional_GPU_funded_by_issue",
}


@dataclass(frozen=True)
class Segment:
    """One allocation's journal: the original (index 0) or an admitted transition."""

    index: int
    journal: Path
    parent_job: str
    node: str
    record: dict | None


class Slurm:
    """Read-only scheduler queries; never submits, extends, cancels or updates."""

    def job(self, job_id: str) -> dict | None:
        result = subprocess.run(
            ["scontrol", "show", "job", job_id, "--json"], capture_output=True, text=True
        )
        if result.returncode != 0:
            return None
        jobs = json.loads(result.stdout).get("jobs", [])
        return jobs[0] if jobs else None

    def node(self, name: str) -> dict:
        record = json.loads(
            subprocess.check_output(["scontrol", "show", "node", name, "--json"], text=True)
        )
        return record["nodes"][0]

    def running_gpu_nodes(self) -> set[str]:
        """Distinct nodes of this user's RUNNING jobs holding any GPU gres."""
        user = os.environ["USER"]
        rows = subprocess.check_output(
            ["squeue", "-h", "-u", user, "-t", "RUNNING", "-o", "%i|%b|%N"], text=True
        ).splitlines()
        nodes = set()
        for row in rows:
            _, gres, node = row.split("|")
            if "gpu" in gres:
                if "[" in node or "," in node:
                    raise ValueError("Multi-node GPU allocation present; count reconciled by root")
                nodes.add(node.strip())
        return nodes


def _states(value: object) -> set[str]:
    if isinstance(value, str):
        return {s for s in value.upper().replace("+", " ").split() if s}
    if isinstance(value, list):
        return {str(s).upper() for s in value}
    raise ValueError("Unrecognized scheduler state")


def _number(value: object) -> float:
    value = value.get("number") if isinstance(value, dict) else value
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Unrecognized scheduler time")
    return float(value)


def staged_launcher(root: Path) -> ModuleType:
    """The exact original launcher bytes root staged and funded, never the checkout."""
    spec = importlib.util.spec_from_file_location("dense_suffix_staged_launch", root / "launch.py")
    if spec is None or spec.loader is None:
        raise ValueError("Staged original launcher absent")
    module = importlib.util.module_from_spec(spec)
    previous, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = previous
    return module


def transitions(root: Path) -> Path:
    return root / TRANSITIONS


def tdir(root: Path, k: int) -> Path:
    return transitions(root) / f"transition-{k:02d}"


def journal_hashes(launch: ModuleType, journal: Path) -> dict[str, str]:
    """Every original journal byte (intents, terminals, trainer logs) of one segment."""
    if not journal.exists():
        return {}
    return {
        str(p.relative_to(journal)): launch.sha(launch.regular(p))
        for p in sorted(journal.rglob("*"))
        if p.is_file() or p.is_symlink()
    }


def verify_original(launch: ModuleType, root: Path, admission: Path, state: Mapping, original):
    """Original funding/admission/launcher/publisher bytes remain exactly those admitted."""
    funding = state["funding"]
    if (
        launch.sha(launch.regular(root / "production-funding.json")) != original["funding_sha256"]
        or funding["launcher_sha256"] != original["launcher_sha256"]
        or funding["publisher_sha256"] != original["publisher_sha256"]
        or str(funding["parent_job"]) != original["parent_job"]
        or funding["node"] != original["node"]
    ):
        raise ValueError("Original production funding/placement proof changed")
    static = launch.static_admission(root, admission, state)
    if launch.load(root / "production-admission.json") != static:
        raise ValueError("Original static production admission differs")
    return launch.sha(root / "production-admission.json")


def segments(launch: ModuleType, root: Path, original: Mapping) -> list[Segment]:
    """Original plus every admitted transition, each with unchanged predecessor bytes."""
    chain = [Segment(0, root / "journal", original["parent_job"], original["node"], None)]
    base = transitions(root)
    names = sorted(p.name for p in base.glob("transition-*")) if base.exists() else []
    admitted = [n for n in names if (base / n / RECORD).exists()]
    if admitted != [f"transition-{k:02d}" for k in range(1, len(admitted) + 1)]:
        raise ValueError("Admitted transitions are not one contiguous sequence")
    for k in range(1, len(admitted) + 1):
        record = launch.load(tdir(root, k) / RECORD)
        funding_path = tdir(root, k) / FUNDING
        prior = record["predecessor"]
        source = chain[prior["segment"]]
        if (
            record.get("version") != VERSION
            or record.get("status") != ADMITTED_STATUS
            or record.get("transition") != k
            or launch.sha(launch.regular(funding_path)) != record["replacement_funding_sha256"]
            or journal_hashes(launch, source.journal) != prior["journal_sha256"]
        ):
            raise ValueError("Transition record or predecessor journal bytes changed")
        funding = launch.load(funding_path)
        chain.append(
            Segment(
                k, tdir(root, k) / "journal", str(funding["parent_job"]), funding["node"], record
            )
        )
    return chain


def arm_status(launch: ModuleType, arm: str, chain: list[Segment], out: Path):
    """(mode, segment, last terminal, next invocation) across all allocations."""
    used = [s for s in chain if any((s.journal / arm).glob("invocation-*-intent.json"))]
    if not used:
        if out.exists():
            raise ValueError("Output without an original intent; reconcile read-only")
        return "fresh", None, None, 1
    for seg in used[:-1]:
        mode, _, _ = launch.journal_state(seg.journal / arm, out)
        if mode != "resume":
            raise ValueError("Only an exit-75 resumable terminal may precede a later allocation")
    mode, n, last = launch.journal_state(used[-1].journal / arm, out)
    return mode, used[-1], last, n


def _invocation_steps(invocation: Mapping) -> int:
    if invocation.get("status") == "final_inference_interrupted_resume_final_checkpoint":
        return -1
    updates = invocation.get("updates")
    if not isinstance(updates, list) or not updates:
        raise ValueError("Exit-75 record without finite optimizer updates; step-0 restart refused")
    steps = [u.get("optimizer_steps") for u in updates]
    if (
        any(type(s) is not int for s in steps)
        or steps != list(range(steps[0], steps[0] + len(steps)))
        or steps[-1] < 1
        or not all(math.isfinite(u.get("loss", math.nan)) for u in updates)
    ):
        raise ValueError("Exit-75 optimizer history is not contiguous/finite")
    return steps[-1]


def complete_state(launch: ModuleType, out: Path, terminal: Mapping) -> dict:
    """Exact checkpoint/run/invocation bytes of the resumable boundary."""
    launch.verify_resumable(out, terminal)
    invocation = launch.load(out / "invocation.json")
    return {
        "status": invocation["status"],
        "optimizer_steps": _invocation_steps(invocation),
        "checkpoint_sha256": terminal["checkpoint_sha256"],
        "checkpoint_bytes": launch.regular(out / "checkpoint.pt").stat().st_size,
        "binding_sha256": terminal["binding_sha256"],
        "invocation_sha256": launch.sha(out / "invocation.json"),
        "run_sha256": launch.sha(out / "run.json"),
    }


def required_trees(root: Path, admission: Path, state: Mapping) -> list[Path]:
    inputs = state["inputs"]
    return [
        root,
        admission,
        Path(state["funding"]["GPU_recovery_output"]),
        Path(state["funding"]["GPU_recovery_verifier_path"]),
        Path(inputs["overlay"]),
        Path(inputs["cache"]),
        Path(inputs["artifact-root"]),
    ]


def _listing(launch: ModuleType, path: Path, skip: Path, full: bool) -> dict[str, dict]:
    files = [path] if path.is_file() else sorted(p for p in path.rglob("*") if p.is_file())
    listing = {}
    for item in files:
        if skip in item.parents:
            continue
        launch.regular(item)
        size = item.stat().st_size
        row: dict = {"bytes": size}
        if full or size <= SMALL_FILE_BYTES:
            row["sha256"] = launch.sha(item)
        listing[str(item)] = row
    return listing


def verify_transport(
    launch: ModuleType, root: Path, admission: Path, state: Mapping, k: int, funding, full: bool
) -> tuple[str, str]:
    """Relocation: destination bytes at the SAME absolute paths equal root's transport."""
    base = tdir(root, k)
    transport, destination = base / TRANSPORT, base / DESTINATION
    record, admitted = launch.load(transport), launch.load(destination)
    transport_sha = launch.sha(transport)
    if (
        admitted.get("status") != DESTINATION_STATUS
        or admitted.get("funded_by") != "root"
        or admitted.get("transition") != k
        or admitted.get("node") != funding["node"]
        or str(admitted.get("parent_job")) != str(funding["parent_job"])
        or admitted.get("transport_sha256") != transport_sha
        or admitted.get("same_absolute_paths") is not True
        or admitted.get("path_rebinding") is not False
        or record.get("transition") != k
        or record.get("destination_node") != funding["node"]
    ):
        raise ValueError("Explicit root destination admission of this exact transport required")
    files = record.get("files")
    if not isinstance(files, dict):
        raise ValueError("Exact transport file manifest required")
    actual: dict[str, dict] = {}
    for tree in required_trees(root, admission, state):
        if not tree.exists():
            raise ValueError("Transported input/runtime/checkpoint path absent; no rebinding")
        actual.update(_listing(launch, tree, base, full))
    if set(actual) != set(files) or any(
        actual[p]["bytes"] != files[p].get("bytes")
        or ("sha256" in actual[p] and actual[p]["sha256"] != files[p].get("sha256"))
        for p in actual
    ):
        raise ValueError("Destination bytes differ from the exact transport manifest")
    return transport_sha, launch.sha(destination)


def validate_replacement(
    launch: ModuleType, funding: Mapping, k: int, chain: list[Segment], root: Path, original
) -> None:
    parents = {s.parent_job for s in chain}
    supersedes = funding.get("supersedes_unused_transition")
    if set(funding) != FUNDING_KEYS:
        raise ValueError("Complete explicit replacement funding/placement required")
    if (
        funding["funded_by"] != "root"
        or funding["transition_version"] != VERSION
        or funding["transition"] != k
        or not 1 <= k <= MAX_TRANSITIONS
        or funding["additional_GPU_funded_by_issue"] is not False
        or funding["prior_funding_sha256"] != original["funding_sha256"]
        or funding["prior_admission_sha256"] != launch.sha(root / "production-admission.json")
        or funding["prior_transition_sha256"]
        != (None if k == 1 else launch.sha(tdir(root, k - 1) / RECORD))
        or supersedes not in (None, k - 1)
        or (supersedes is not None and k == 1)
        or funding["coordinator_sha256"]
        != launch.sha(launch.regular(transitions(root) / COORDINATOR))
    ):
        raise ValueError("Replacement funding binds a different prior admission/funding")
    if (
        funding["placement"] not in PLACEMENTS
        or not str(funding["parent_job"]).isdigit()
        or str(funding["parent_job"]) in parents
        or not funding["node"]
        or not funding["assigned_GPU"]
        or type(funding["CPUs"]) is not int
        or funding["CPUs"] < 1
        or type(funding["max_slice_seconds"]) is not int
        or funding["max_slice_seconds"] < launch.MIN_START_SECONDS
        or funding["reserve_bytes"] != launch.RESERVE
        or funding["required_free_bytes"] < funding["reserve_bytes"]
        or type(funding["required_free_files"]) is not int
        or funding["required_free_files"] < 1
        or funding["shared_future_bytes"] != 0
        or funding["node_local_envelope_bytes"] < launch.tree_bytes(root)
    ):
        raise ValueError("Stale or incomplete replacement placement/funding")
    owner = owner_node(chain)
    if (funding["placement"] == "owning_node") != (funding["node"] == owner):
        raise ValueError("Owning-node reuse requires the owning node; relocation another node")


def _ran(seg: Segment) -> bool:
    return any(seg.journal.glob("*/invocation-*-intent.json"))


def canonical_journal(launch: ModuleType, journal: Path) -> None:
    """Every intent/terminal is byte-identical to the launcher's own canonical save()."""
    for path in sorted(journal.glob("*/invocation-*-*.json")):
        text = launch.regular(path).read_text()
        value = json.loads(text)
        if text != json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n":
            raise ValueError("Journal record bytes differ from the launcher's own save")


def owner_node(chain: list[Segment]) -> str:
    """Node holding the live state: the latest allocation that actually ran a trainer."""
    ran = [s for s in chain if _ran(s)]
    return (ran[-1] if ran else chain[0]).node


def _active(launch: ModuleType, root: Path, state: Mapping, chain: list[Segment]):
    production = root / "production"
    completed = {}
    for arm in launch.ARMS:
        out = production / arm
        mode, seg, last, n = arm_status(launch, arm, chain, out)
        if mode == "completed":
            record = launch.validate_completed(out, arm, state["configs"][arm], state["cpu"])
            completed[arm] = {
                "binding_sha256": launch.canonical(record["binding"]),
                "checkpoint_sha256": launch.sha(out / "checkpoint.pt"),
                "prediction_sha256": launch.sha(out / "outer_predictions.csv"),
            }
            continue
        return arm, mode, seg, last, n, completed
    raise ValueError("Both arms already completed; no transition required")


def admit(
    root: Path = ROOT,
    admission: Path = ADMISSION,
    *,
    pins: Mapping | None = None,
    original: Mapping = ORIGINAL,
    hostname: Callable[[], str] = lambda: os.uname().nodename,
    scheduler: Slurm | None = None,
) -> dict:
    """Bind one actual exit-75 terminal to one funded replacement placement.

    Reads and hashes only; never touches CUDA, so root may run it as a CPU step of the
    replacement parent before its first ``fit`` step.
    """
    launch = staged_launcher(root)
    pins = launch.PINS if pins is None else pins
    scheduler = scheduler or Slurm()
    state = launch.admitted(root, admission, pins)
    admission_sha = verify_original(launch, root, admission, state, original)
    chain = segments(launch, root, original)
    k = len(chain)
    base = tdir(root, k)
    if (base / RECORD).exists():
        raise ValueError("Duplicate transition refused")
    if k > MAX_TRANSITIONS:
        raise ValueError("Bounded transition count exhausted")
    if k > 1 and not _ran(chain[-1]) and not (base / FUNDING).exists():
        raise ValueError("Duplicate transition refused: admitted transition still unused")
    funding = launch.load(base / FUNDING)
    validate_replacement(launch, funding, k, chain, root, original)
    for seg in chain:
        canonical_journal(launch, seg.journal)
    arm, mode, seg, last, _, completed = _active(launch, root, state, chain)
    if mode != "resume" or seg is None or last is None:
        raise ValueError("Only an actual exit-75 resumable terminal is admitted; no step-0 start")
    latest = max((s for s in chain if any(s.journal.glob("*/invocation-*-intent.json"))),
                 key=lambda s: s.index)  # fmt: skip
    unused = [s for s in chain if s.index > seg.index]
    if latest.index != seg.index:
        raise ValueError("A later allocation already ran; terminal is not the latest boundary")
    if unused and funding["supersedes_unused_transition"] != unused[-1].index:
        raise ValueError("Duplicate transition for the same terminal refused")
    if not unused and funding["supersedes_unused_transition"] is not None:
        raise ValueError("No unused transition to supersede")
    if any(_live(scheduler, s.parent_job) for s in [seg, *unused]):
        raise ValueError("Predecessor allocation still live; never two trainers on one lineage")
    if str(last.get("job_id")) != seg.parent_job:
        raise ValueError("Terminal was not recorded by its predecessor allocation")
    out = root / "production" / arm
    terminals = sorted((seg.journal / arm).glob("invocation-*-terminal.json"))
    state_record = complete_state(launch, out, last)
    if funding["placement"] == "owning_node":
        if hostname() != seg.node:
            raise ValueError("Admission runs on the owning node holding the original state")
        transport = None
    else:
        if hostname() != funding["node"]:
            raise ValueError("Relocated admission runs on the admitted destination")
        transport_sha, destination_sha = verify_transport(
            launch, root, admission, state, k, funding, full=True
        )
        files = launch.load(base / TRANSPORT)["files"]
        if files.get(str(out / "checkpoint.pt"), {}).get("sha256") != last["checkpoint_sha256"]:
            raise ValueError("Transported checkpoint differs from the exact exit-75 terminal")
        transport = {"transport_sha256": transport_sha, "destination_sha256": destination_sha}
    cpu, prior = state["cpu"], state["funding"]
    record = {
        "version": VERSION,
        "status": ADMITTED_STATUS,
        "transition": k,
        "arm": arm,
        "predecessor": {
            "segment": seg.index,
            "parent_job": seg.parent_job,
            "node": seg.node,
            "terminal_name": terminals[-1].name,
            "terminal_sha256": launch.sha(terminals[-1]),
            "terminal": last,
            "journal_sha256": journal_hashes(launch, seg.journal),
            "superseded_unused_transitions": [s.index for s in unused],
        },
        "complete_state": state_record,
        "completed_arms": completed,
        "original": {
            "parent_job": original["parent_job"],
            "node": original["node"],
            "journal_sha256": journal_hashes(launch, root / "journal"),
            "funding_sha256": original["funding_sha256"],
            "production_admission_sha256": admission_sha,
            "launcher_sha256": prior["launcher_sha256"],
            "publisher_sha256": prior["publisher_sha256"],
            "CPU_receipt_sha256": pins["CPU_receipt_sha256"],
            "GPU_recovery_receipt_sha256": prior["GPU_recovery_receipt_sha256"],
            "source_git": cpu["source_git"],
            "source_manifest_sha256": pins["recovery_source_manifest_sha256"],
            "source_code_sha256": state["code"],
            "config_sha256": prior["config_sha256"],
            "folds_sha256": cpu["folds_sha256"],
            "release_sha256": cpu["release_sha256"],
            "supervision_sha256": cpu["supervision_sha256"],
            "overlay_sha256": cpu["overlay_sha256"],
            "official_encoder": pins["official_encoder"],
        },
        "replacement_funding_sha256": launch.sha(base / FUNDING),
        "replacement": {
            "parent_job": str(funding["parent_job"]),
            "node": funding["node"],
            "placement": funding["placement"],
            "assigned_GPU": funding["assigned_GPU"],
            "CPUs": funding["CPUs"],
        },
        "relocation": transport,
        "coordinator_sha256": funding["coordinator_sha256"],
        "canonical_trainer_changed": False,
        "path_rebinding": False,
        "automatic_retry": False,
        "additional_GPU_funded_by_issue": False,
        "accuracy_claimed": False,
    }
    launch.save(base / RECORD, record)
    return record


def _live(scheduler: Slurm, job: str) -> bool:
    record = scheduler.job(job)
    return record is not None and bool(_states(record.get("job_state")) & LIVE_JOB_STATES)


def require_allocation(
    launch: ModuleType, funding: Mapping, scheduler: Slurm, root: Path, chain: list[Segment]
) -> float:
    """Genuine replacement allocation, node state, GPU-node bound and reserve; seconds left."""
    job, node = str(funding["parent_job"]), funding["node"]
    record = scheduler.job(job)
    tres = set(str((record or {}).get("tres_alloc_str", "")).split(","))
    if (
        record is None
        or str(record.get("job_id")) != job
        or _states(record.get("job_state")) != {"RUNNING"}
        or record.get("nodes") != node
        or record.get("reservation")
        or "gres/gpu=1" not in tres
        or f"cpu={funding['CPUs']}" not in tres
    ):
        raise ValueError("Genuine running one-GPU replacement allocation required")
    if _states(scheduler.node(node).get("state")) & BLOCKED_NODE_STATES:
        raise ValueError("Node MAINTENANCE/RESERVED/unavailable state is never bypassed")
    if any(_live(scheduler, s.parent_job) for s in chain[:-1]):
        raise ValueError("Predecessor allocation still live; never two trainers on one lineage")
    nodes = scheduler.running_gpu_nodes()
    if node not in nodes or len(nodes) > MAX_GPU_NODES:
        raise ValueError("At most two concurrent GPU nodes")
    usage, files = shutil.disk_usage(root), os.statvfs(root).f_favail
    if (
        usage.free < funding["required_free_bytes"]
        or files < funding["required_free_files"]
        or launch.tree_bytes(root) > funding["node_local_envelope_bytes"]
    ):
        raise ValueError("Funded byte/file reserve unavailable; original state never deleted")
    return _number(record.get("end_time")) - time.time()


def require_device(funding: Mapping) -> str:
    import torch

    if (
        os.uname().nodename != funding["node"]
        or os.environ.get("SLURM_JOB_ID") != str(funding["parent_job"])
        or os.environ.get("SLURM_CPUS_PER_TASK") != str(funding["CPUs"])
        or (os.environ.get("SLURM_STEP_GPUS") or os.environ.get("SLURM_JOB_GPUS"))
        != str(funding["assigned_GPU"])
        or torch.cuda.device_count() != 1
        or "H100" not in torch.cuda.get_device_name(0)
    ):
        raise ValueError("Replacement parent job and exactly one assigned H100 required")
    return torch.cuda.get_device_name(0)


def _exit75_terminal(launch, out: Path, arm: str, values, cpu, code: int, job: str) -> dict:
    terminal: dict = {"returncode": code, "job_id": job, "resumable": False}
    if code == 75:
        try:
            invocation = launch.load(out / "invocation.json")
            record = launch.load(out / "run.json")
            launch.validate_binding(record["binding"], arm, values, cpu)
            if (
                invocation.get("exit_code") == 75
                and invocation.get("status") in launch.RESUMABLE
                and invocation.get("checkpoint_sha256") == launch.sha(out / "checkpoint.pt")
                and invocation.get("binding_sha256") == launch.canonical(record["binding"])
            ):
                terminal.update(
                    resumable=True,
                    checkpoint_sha256=invocation["checkpoint_sha256"],
                    binding_sha256=invocation["binding_sha256"],
                )
        except (OSError, ValueError, KeyError):
            pass
    return terminal


def run(
    root: Path = ROOT,
    admission: Path = ADMISSION,
    *,
    execute=None,
    device: Callable[[Mapping], str] = require_device,
    scheduler: Slurm | None = None,
    pins: Mapping | None = None,
    original: Mapping = ORIGINAL,
    slice_seconds: float,
) -> int:
    """One bounded slice in the latest admitted replacement allocation (0 done, 75 continue)."""
    launch = staged_launcher(root)
    pins = launch.PINS if pins is None else pins
    scheduler = scheduler or Slurm()
    state = launch.admitted(root, admission, pins)
    verify_original(launch, root, admission, state, original)
    chain = segments(launch, root, original)
    current = chain[-1]
    if current.record is None:
        raise ValueError("No admitted transition; the original launcher owns the original parent")
    k, record = current.index, current.record
    base = tdir(root, k)
    funding = launch.load(base / FUNDING)
    if launch.sha(base / FUNDING) != record["replacement_funding_sha256"]:
        raise ValueError("Replacement funding changed after admission")
    validate_replacement(launch, funding, k, chain[:-1], root, original)
    if (
        record["original"]["journal_sha256"] != journal_hashes(launch, root / "journal")
        or record["original"]["production_admission_sha256"]
        != launch.sha(root / "production-admission.json")
        or record["original"]["config_sha256"] != state["funding"]["config_sha256"]
        or record["original"]["source_code_sha256"] != state["code"]
    ):
        raise ValueError("Original journals/admission/source changed after transition")
    if not (
        math.isfinite(slice_seconds)
        and launch.MIN_START_SECONDS <= slice_seconds <= funding["max_slice_seconds"]
    ):
        raise ValueError("Finite slice within root's funded replacement bound required")
    name = device(funding)
    remaining = require_allocation(launch, funding, scheduler, root, chain)
    deadline = time.monotonic() + min(remaining, slice_seconds)
    runtime = dict(state, funding=dict(state["funding"], CPUs=funding["CPUs"]))
    execute = execute or launch.subprocess_executor(runtime)
    job, production, cpu = current.parent_job, root / "production", state["cpu"]
    records = {}
    arms = list(launch.ARMS)
    admitted_index = arms.index(record["arm"])
    for index, arm in enumerate(arms):
        values, config = state["configs"][arm], root / f"{arm}-config.json"
        out, journal = production / arm, current.journal / arm
        mode, seg, last, n = arm_status(launch, arm, chain, out)
        if mode == "completed":
            records[arm] = launch.validate_completed(out, arm, values, cpu)
            continue
        if records.keys() != set(arms[:index]) or any(
            (production / other).exists() for other in arms[index + 1 :]
        ):
            raise ValueError("Arms run sequentially; a later arm requires its predecessor")
        if seg is None:
            if index <= admitted_index:
                raise ValueError("Initial-state restart of an admitted arm refused")
        elif seg.index != k:
            if (
                arm != record["arm"]
                or seg.index != record["predecessor"]["segment"]
                or last != record["predecessor"]["terminal"]
                or complete_state(launch, out, last) != record["complete_state"]
            ):
                raise ValueError("Only the exact admitted exit-75 checkpoint may be resumed")
            if record["relocation"] is not None:
                verify_transport(launch, root, admission, state, k, funding, full=False)
            n = 1
        if mode == "resume":
            assert last is not None
            launch.verify_resumable(out, last)
        seconds = deadline - time.monotonic()
        if seconds < launch.MIN_START_SECONDS:
            return 75
        if tree_bytes_exceeded(launch, production, root, state["funding"], funding):
            raise ValueError("Funded node-local output envelope unavailable")
        require_allocation(launch, funding, scheduler, root, chain)
        extra = [] if mode == "fresh" else ["--resume", str(out / "checkpoint.pt")]
        journal.mkdir(mode=0o700, parents=True, exist_ok=True)
        launch.save(
            journal / f"invocation-{n:02d}-intent.json",
            {
                "arm": arm,
                "invocation": n,
                "mode": mode,
                "transition": k,
                "transition_sha256": launch.sha(base / RECORD),
                "job_id": job,
                "node": current.node,
                "device": name,
                "source_git": cpu["source_git"],
                "config_sha256": state["funding"]["config_sha256"][arm],
                "resume_checkpoint_sha256": None if last is None else last["checkpoint_sha256"],
                "slice_seconds": seconds,
                "automatic_retry": False,
            },
        )
        code = execute(config, out, extra, journal / f"invocation-{n:02d}", seconds)
        terminal = _exit75_terminal(launch, out, arm, values, cpu, code, job)
        launch.save(journal / f"invocation-{n:02d}-terminal.json", terminal)
        if code == 75 and terminal["resumable"]:
            return 75
        if code != 0:
            raise ValueError("Trainer failed or stopped uncertainly; originals preserved")
        records[arm] = launch.validate_completed(out, arm, values, cpu)
        if tree_bytes_exceeded(launch, production, root, state["funding"], funding):
            raise ValueError("Funded node-local output envelope exceeded")
    from scripts.newton.run_dense_suffix_recovery import compare_matched

    compare_matched({arm: records[arm]["binding"] for arm in arms})
    if not (base / "terminal.json").exists():
        launch.save(
            base / "terminal.json",
            {
                "status": SEGMENT_TERMINAL_STATUS,
                "transition": k,
                "job_id": job,
                "arms": {
                    arm: {
                        "binding_sha256": launch.canonical(records[arm]["binding"]),
                        "checkpoint_sha256": launch.sha(production / arm / "checkpoint.pt"),
                        "prediction_sha256": launch.sha(production / arm / "outer_predictions.csv"),
                    }
                    for arm in arms
                },
                "fixed_epochs": launch.EPOCHS,
                "optimizer_updates_per_arm": launch.EPOCHS * launch.UPDATES,
                "endpoint_selection": False,
                "accuracy_claimed": False,
                "paired_fixed_final_analysis_completed": False,
            },
        )
    return 0


def tree_bytes_exceeded(launch, production: Path, root: Path, prior: Mapping, funding) -> bool:
    return (
        launch.tree_bytes(production) > prior["checkpoint_bytes"] + prior["runtime_bytes"]
        or launch.tree_bytes(root) > funding["node_local_envelope_bytes"]
    )


def main() -> None:
    os.umask(0o077)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("admit", "fit"))
    parser.add_argument("--slice-seconds", type=float, help="bounded replacement step budget")
    args = parser.parse_args()
    sys.path.insert(0, str(ADMISSION / "code"))
    if args.phase == "admit":
        print(json.dumps(admit(), sort_keys=True), flush=True)
        return
    if args.slice_seconds is None:
        raise SystemExit("fit requires an explicit --slice-seconds")
    code = run(slice_seconds=args.slice_seconds)
    print(json.dumps({"status": "complete" if code == 0 else "continue_exact_resume"}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
