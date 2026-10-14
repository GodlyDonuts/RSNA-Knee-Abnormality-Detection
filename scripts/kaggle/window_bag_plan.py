"""Quota, runtime, storage and session-boundary plan for the whole-window bag on Kaggle T4.

Pure functions; no Kaggle call. Every input that depends on the platform (quota, measured
seconds, disk, kernel-source limits) must come from a fresh actual observation recorded by
root. Nothing here assumes a platform limit or a T4 speed.
"""

from __future__ import annotations

import math

FITTING_STUDIES, OUTER_STUDIES = 2247, 1156
EFFECTIVE_BATCH = 8
UPDATES_PER_EPOCH = math.ceil(FITTING_STUDIES / EFFECTIVE_BATCH)  # 281
EPOCHS = 10
TOTAL_UPDATES = UPDATES_PER_EPOCH * EPOCHS  # 2810
ENGINEERING_TIMEOUT_SECONDS = 7200
SESSION_LIMIT_SECONDS = 9 * 3600
# Each production session must finish with margin below the nine-hour ceiling.
SESSION_MARGIN = 0.80
# Kaggle's "T4 x2" accelerator provisions two devices. The sequential path uses only device
# 0; the optional concurrent mode uses both. Either way both are charged exactly once,
# conservatively, for the full requested session timeout.
PROVISIONED_T4 = 2
SHARD_KERNEL_SOURCES = 20
ENGINEERING_DATASET_SOURCES = ("model_bundle", "wheelhouse", "render_capsule")
EXACT_MEMBER_BYTES = 27667165884
GIB = 2**30
# Sidecars, logs, history and the final 1156-row prediction CSV (never MRI/cache/reports).
PUBLICATION_SIDECAR_BYTES = 256 * 2**20
# Actual private offline CPU kernel 138609817/v1 (audit kaggle-input-shape-CPU-20261014-v1)
# COMPLETED with exactly 21 kernel inputs (the 20 original shards plus one other kernel) and
# these 3 Datasets. That is a supported shape, not a platform maximum.
INPUT_SHAPE_PROOF_KERNEL = {
    "slug": "godlydonuts/rsna-input-shape-cpu-21x3-20261014-v1",
    "version": 1,
    "kernel_id": 138609817,
    "source_sha256": "e06b3842e1c23022821063e9bda1f51cf525d399c097c0d70b6da3b6c06e086a",
}
SUPPORTED_KERNEL_SOURCES, SUPPORTED_DATASET_SOURCES = 21, 3
SUPPORTED_DATASET_IDS = (12333543, 12139697, 12567458)  # model bundle, wheelhouse, capsule
# Kaggle documents 20 GB of saved /kaggle/working output (kaggle.com/docs/notebooks); every
# segment output is held to an explicit lower cap, leaving >2.8 GB of margin.
DOCUMENTED_OUTPUT_BYTES = 20 * 10**9
OUTPUT_CAP_CEILING_BYTES = 16 * GIB


def admitted_platform_limits() -> dict:
    """The only accepted platform proof: the actual 21/3 input-shape kernel, as a bound."""
    return {
        "observed_at": "2026-10-14 kernel 138609817/v1 COMPLETE",
        "evidence": "actual_private_CPU_21kernel_3dataset_mount_shape_accepted:"
        + INPUT_SHAPE_PROOF_KERNEL["slug"]
        + "/v1:"
        + INPUT_SHAPE_PROOF_KERNEL["source_sha256"]
        + ";output:kaggle.com/docs/notebooks 20GB /kaggle/working, explicit 16GiB cap",
        "max_kernel_sources": SUPPORTED_KERNEL_SOURCES,
        "max_dataset_sources": SUPPORTED_DATASET_SOURCES,
        "max_output_bytes": OUTPUT_CAP_CEILING_BYTES,
    }


def seconds(value) -> float:
    if not isinstance(value, str) or not value.endswith("s"):
        raise ValueError("Kaggle duration string required")
    return float(value[:-1])


def conservative_available(gpu_quota: dict) -> float:
    """min(total, minimum) - used - reserved, from a fresh GetAcceleratorQuotaStatistics."""
    allowed = min(seconds(gpu_quota["totalTimeAllowed"]), seconds(gpu_quota["minimumTimeAllowed"]))
    return allowed - seconds(gpu_quota["timeUsed"]) - seconds(gpu_quota["timeReserved"])


def session_charge(timeout_seconds: float, provisioned: int = PROVISIONED_T4) -> float:
    if timeout_seconds <= 0 or provisioned < 1:
        raise ValueError("Positive timeout and provisioned GPU count required")
    return timeout_seconds * provisioned


def admit_engineering(gpu_quota: dict, timeout_seconds: float = ENGINEERING_TIMEOUT_SECONDS):
    """One active T4 engineering session, charged for every provisioned GPU."""
    if timeout_seconds > ENGINEERING_TIMEOUT_SECONDS:
        raise ValueError("Engineering timeout exceeds two hours")
    if seconds(gpu_quota["timeReserved"]) != 0:
        raise ValueError("Another reservation exists; reconcile before a new session")
    available, charge = conservative_available(gpu_quota), session_charge(timeout_seconds)
    if available < charge:
        raise ValueError("Fresh conservative GPU allowance cannot fund the session")
    return {
        "available_seconds": available,
        "charge_seconds": charge,
        "remaining_after_seconds": available - charge,
    }


def require_platform_limits(
    proof: dict, kernel_sources: int, dataset_sources: int, output_bytes: int = 0
) -> dict:
    """Refuse unless root recorded the actual API constraint (never an assumed default).

    Counts at or below the actually completed 21/3 shape are admitted; nothing above it.
    """
    keys = {
        "observed_at",
        "evidence",
        "max_kernel_sources",
        "max_dataset_sources",
        "max_output_bytes",
    }
    if not isinstance(proof, dict) or set(proof) != keys:
        raise ValueError("Actual platform source-limit observation required")
    if not isinstance(proof["evidence"], str) or not proof["evidence"].strip():
        raise ValueError("Source-limit evidence must name the actual API constraint")
    for key, need in (
        ("max_kernel_sources", kernel_sources),
        ("max_dataset_sources", dataset_sources),
        ("max_output_bytes", output_bytes),
    ):
        if type(proof[key]) is not int or proof[key] < need:
            raise ValueError("Requested inputs exceed the verified platform limit")
    if (
        proof["max_kernel_sources"] > SUPPORTED_KERNEL_SOURCES
        or proof["max_dataset_sources"] > SUPPORTED_DATASET_SOURCES
        or proof["max_output_bytes"] > OUTPUT_CAP_CEILING_BYTES
    ):
        raise ValueError("Claimed limit exceeds the actual supported shape or output cap")
    return proof


def segment_boundaries(updates_per_session: int, total: int = TOTAL_UPDATES):
    """Contiguous optimizer-boundary segments covering exactly the fixed schedule."""
    if type(updates_per_session) is not int or updates_per_session < 1:
        raise ValueError("Positive integer updates per session required")
    edges = list(range(0, total, updates_per_session)) + [total]
    return [(a, b) for a, b in zip(edges, edges[1:])]


def production_plan(
    measured: dict, gpu_quota: dict, free_scratch_bytes: int, concurrent: bool = False
) -> dict:
    """Fixed 2810-update plan per arm from actual T4 measurements; never changes the recipe.

    ``measured`` requires ``seconds_per_update`` (four-window bag, effective 8),
    ``startup_seconds`` (install, mount verify, full restore, model load),
    ``checkpoint_seconds`` and ``final_evaluation_seconds`` (1156 ten-window studies),
    all from actual T4 sessions, plus ``checkpoint_bytes``.

    ``concurrent`` projects the optional two-T4 mode: both arms share one session (so each
    session is charged once for both provisioned GPUs) and one restored cache, with two
    arms' checkpoints and outputs. Its timings must come from an actual concurrent
    engineering session; the paired segment runner itself is not yet admitted.
    """
    keys = {
        "seconds_per_update",
        "startup_seconds",
        "checkpoint_seconds",
        "final_evaluation_seconds",
        "checkpoint_bytes",
    }
    if set(measured) != keys or any(
        not isinstance(measured[k], (int, float)) or measured[k] <= 0 for k in keys
    ):
        raise ValueError("Actual measured T4 timings/bytes required")
    usable = SESSION_LIMIT_SECONDS * SESSION_MARGIN
    budget = usable - measured["startup_seconds"] - measured["final_evaluation_seconds"]
    # The trainer rewrites its complete checkpoint every 25 updates.
    per_update = measured["seconds_per_update"] + measured["checkpoint_seconds"] / 25
    updates = int(budget // per_update)
    if updates < 1:
        raise ValueError("Measured T4 performance cannot fit one update per session")
    segments = segment_boundaries(min(updates, TOTAL_UPDATES))
    timeout = min(SESSION_LIMIT_SECONDS, math.ceil(usable))
    arms_per_session = 2 if concurrent else 1
    sessions = 2 * len(segments) // arms_per_session
    charge = sessions * session_charge(timeout)
    available = conservative_available(gpu_quota)
    storage = EXACT_MEMBER_BYTES + arms_per_session * 3 * measured["checkpoint_bytes"] + 3 * GIB
    caps = [arms_per_session * publication_bytes(measured, s) for s in segments]
    return {
        "updates_per_epoch": UPDATES_PER_EPOCH,
        "total_updates_per_arm": TOTAL_UPDATES,
        "arms": ["mean", "query"],
        "segments_per_arm": segments,
        "sessions": sessions,
        "session_timeout_seconds": timeout,
        "conservative_charge_seconds": charge,
        "available_seconds": available,
        "quota_fits_now": charge <= available,
        "scratch_bytes_required": storage,
        "output_cap_bytes_per_segment": caps,
        "output_caps_fit": all(cap <= OUTPUT_CAP_CEILING_BYTES for cap in caps),
        "concurrent_arms": concurrent,
        "provisioned_GPUs_charged_per_session": PROVISIONED_T4,
        "scratch_fits": free_scratch_bytes >= storage,
        "only_final_epoch10_evaluated": True,
        "scientific_hyperparameters_changed": False,
        "intermediate_epoch_selection": False,
    }


def publication_bytes(measured: dict, segment: tuple[int, int]) -> int:
    """Cap for one segment's private output: continuation state plus every epoch it completes."""
    start, end = segment
    epochs = end // UPDATES_PER_EPOCH - start // UPDATES_PER_EPOCH
    return (1 + epochs) * measured["checkpoint_bytes"] + PUBLICATION_SIDECAR_BYTES


def next_segment(
    previous_receipt: dict | None,
    arm: str,
    segments: list,
    *,
    engineering_receipt_sha256: str,
    production_config_sha256: str,
    measured: dict,
    previous_kernel: dict | None = None,
) -> dict:
    """Bind the next session to the exact prior kernel output; refuses drift or skipped steps.

    ``previous_kernel`` is root's read-only observation of the completed previous segment:
    ``{"kernel_slug", "version", "source_sha256"}``. The checkpoint SHA alone never suffices.
    """
    from scripts.kaggle.window_bag_t4 import canonical

    common = {
        "engineering_receipt_sha256": engineering_receipt_sha256,
        "production_config_sha256": production_config_sha256,
    }
    if previous_receipt is None:
        if previous_kernel is not None:
            raise ValueError("The first segment has no previous kernel")
        start, end = segments[0]
        return {
            "arm": arm,
            "segment": 0,
            "start_step": start,
            "end_step": end,
            "previous_output": None,
            **common,
            "output_cap_bytes": publication_bytes(measured, (start, end)),
        }
    contract = previous_receipt["contract"]
    index = contract["segment"] + 1
    if (
        previous_receipt.get("status") != "source_window_bag_fixed10_single_T4_segment_completed"
        or contract["arm"] != arm
        or index >= len(segments)
        or segments[contract["segment"]] != (contract["start_step"], contract["end_step"])
        or previous_receipt["final_epoch10_evaluated"] is not False
        or previous_receipt.get("intermediate_selection") is not False
        or {k: contract[k] for k in common} != common
        or previous_receipt["publication"]["checkpoint_sha256"]
        != previous_receipt["checkpoint_sha256"]
    ):
        raise ValueError("Previous segment receipt differs from the schedule")
    if (
        not isinstance(previous_kernel, dict)
        or set(previous_kernel) != {"kernel_slug", "version", "source_sha256"}
        or previous_kernel["version"] != 1
    ):
        raise ValueError("Exact previous kernel slug/version/source identity required")
    start, end = segments[index]
    if start != contract["end_step"]:
        raise ValueError("Segments must be contiguous")
    return {
        "arm": arm,
        "segment": index,
        "start_step": start,
        "end_step": end,
        "previous_output": {
            **previous_kernel,
            "receipt_sha256": canonical(previous_receipt),
            "publication_sha256": previous_receipt["publication"]["manifest_sha256"],
            "checkpoint_sha256": previous_receipt["checkpoint_sha256"],
        },
        **common,
        "output_cap_bytes": publication_bytes(measured, (start, end)),
    }
