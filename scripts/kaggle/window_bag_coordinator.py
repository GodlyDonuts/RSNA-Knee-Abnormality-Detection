"""Conditional single-coordinator phases for the private whole-window T4 notebook.

PREPARED ONLY. Root runs ``run_phase`` inside the canonical Newton coordinator claim
(``scripts/kaggle/coordinator.py:single_coordinator``) with the canonical account ledger,
exactly like the original T4 recoveries. ``preflight`` and ``status`` are read-only.
``push`` performs at most one SaveKernel after a fresh quota, absence, platform-limit and
input-identity check. Any uncertain SaveKernel outcome is reconciled read-only; there is
no automatic retry. Production segments additionally require an accepted engineering
receipt, measured performance, fresh allowance and explicit checkpoint/storage funding.
"""

from __future__ import annotations

from scripts.kaggle import window_bag_plan as plan
from scripts.kaggle import window_bag_t4 as t4

MACHINE_SHAPE = "NvidiaTeslaT4"
CONTRACT_KEYS = {
    "kernel_slug", "source_sha256", "mode", "shard_kernels", "model_dataset", "wheelhouse",
    "capsule_dataset", "runtime_image", "session_timeout_seconds", "platform_limits",
    "production_funding",
}


def validate_contract(contract: dict) -> dict:
    if set(contract) != CONTRACT_KEYS or contract["mode"] not in {"engineering", "segment"}:
        raise ValueError("Exact whole-window Kaggle contract required")
    shards = contract["shard_kernels"]
    if (
        len(shards) != plan.SHARD_KERNEL_SOURCES
        or [s["shard_index"] for s in shards] != list(range(plan.SHARD_KERNEL_SOURCES))
        or any(s["version"] != 1 for s in shards)
        or [s["shard_manifest_sha256"] for s in shards] != list(t4.SHARD_MANIFEST_SHA256)
        or len({s["slug"] for s in shards}) != plan.SHARD_KERNEL_SOURCES
    ):
        raise ValueError("Exactly the 20 original completed shard kernels, version 1")
    limit = (
        plan.ENGINEERING_TIMEOUT_SECONDS if contract["mode"] == "engineering"
        else plan.SESSION_LIMIT_SECONDS * plan.SESSION_MARGIN
    )
    if not 0 < contract["session_timeout_seconds"] <= limit:
        raise ValueError("Session timeout exceeds its bound")
    if contract["mode"] == "segment" and not contract["production_funding"]:
        raise ValueError("Production segments require explicit checkpoint/output/storage funding")
    if contract["mode"] == "engineering" and contract["production_funding"] is not None:
        raise ValueError("Engineering contract must not carry production funding")
    return contract


def kernel_request(contract: dict, source: str, previous_segment_slug: str | None = None):
    kernels = [s["slug"] for s in contract["shard_kernels"]]
    if previous_segment_slug is not None:
        kernels.append(previous_segment_slug)
    return {
        "slug": contract["kernel_slug"],
        "newTitle": contract["kernel_slug"].split("/")[1].replace("-", " "),
        "text": source,
        "language": "python",
        "kernelType": "script",
        "isPrivate": True,
        "enableGpu": True,
        "machineShape": MACHINE_SHAPE,
        "enableInternet": False,
        "datasetDataSources": [
            contract["model_dataset"]["slug"],
            contract["wheelhouse"]["dataset_slug"],
            contract["capsule_dataset"]["slug"],
        ],
        "kernelDataSources": kernels,
        "competitionDataSources": [],
        "sessionTimeoutSeconds": contract["session_timeout_seconds"],
        "dockerImage": contract["runtime_image"],
        "dockerImagePinningType": "original",
    }


def verify_inputs(remote, publication, workflow, contract: dict) -> dict:
    """Read-only: every mounted input is the exact original private immutable object."""
    for shard in contract["shard_kernels"]:
        meta = remote.kernel("GetKernel", shard["slug"], 1)["metadata"]
        if (
            meta.get("isPrivate") is not True
            or meta.get("enableGpu", False) is not False
            or meta.get("enableInternet") is not False
            or remote.kernel("GetKernelSessionStatus", shard["slug"], 1).get("status")
            != "COMPLETE"
        ):
            raise ValueError("Original shard kernel is not the private offline completed CPU v1")
    for key in ("model_dataset", "capsule_dataset"):
        record = contract[key]
        if publication.metadata(remote, record["slug"]) != {
            "id": record["id"], "version": 1, "slug": record["slug"], "private": True
        }:
            raise ValueError("Original private Dataset identity differs")
    workflow.verify_shared_wheelhouse(remote, contract["wheelhouse"])
    return {"shards_complete": plan.SHARD_KERNEL_SOURCES, "datasets": 3}


def run_phase(phase, contract, source, remote, publication, workflow, journal, digest):
    """``journal`` is the root-owned exclusive-JSON writer inside the coordinator claim."""
    validate_contract(contract)
    if digest(source) != contract["source_sha256"]:
        raise ValueError("Prepared notebook source differs from its contract")
    request = kernel_request(contract, source)
    plan.require_platform_limits(
        contract["platform_limits"],
        len(request["kernelDataSources"]),
        len(request["datasetDataSources"]),
    )
    if phase == "status":
        original = remote.kernel("GetKernel", contract["kernel_slug"], 1)
        meta = original["metadata"]
        if digest(original.get("blob", {}).get("source", "")) != contract["source_sha256"]:
            raise ValueError("Pushed source differs from the prepared source")
        for name in ("isPrivate", "enableInternet", "machineShape", "dockerImage"):
            if meta.get(name) != request[name]:
                raise ValueError("Pushed kernel settings differ")
        for name in ("datasetDataSources", "kernelDataSources", "competitionDataSources"):
            if sorted(meta.get(name, [])) != sorted(request[name]):
                raise ValueError("Pushed kernel inputs differ; inspect, never retry")
        return {"status": remote.kernel("GetKernelSessionStatus", contract["kernel_slug"], 1)}
    if phase not in {"preflight", "push"}:
        raise ValueError("Unknown phase")
    inputs = verify_inputs(remote, publication, workflow, contract)
    quota = remote.call("kernels.KernelsApiService", "GetAcceleratorQuotaStatistics", {})
    gpu = quota["gpuQuota"]
    funding = plan.admit_engineering(gpu, min(contract["session_timeout_seconds"], 7200)) if (
        contract["mode"] == "engineering"
    ) else {"available_seconds": plan.conservative_available(gpu),
            "charge_seconds": plan.session_charge(contract["session_timeout_seconds"])}
    if funding["available_seconds"] < funding["charge_seconds"]:
        raise ValueError("Fresh conservative allowance cannot fund this session")
    absence = remote.require_absent_kernel(contract["kernel_slug"])
    proof = {"phase": phase, "inputs": inputs, "gpu_quota": gpu, "funding": funding,
             "absence": absence, "request_sha256": digest(repr(sorted(request.items())))}
    if phase == "preflight":
        return proof
    journal("kernel-create-intent.json", {**proof, "single_SaveKernel": True,
                                          "automatic_retry": False})
    try:
        result = remote.call("kernels.KernelsApiService", "SaveKernel", request)
    except Exception as exc:
        journal("kernel-create-uncertain.json", {"error_type": type(exc).__name__,
                                                 "reconcile_read_only": True})
        raise
    journal("kernel-create-response.json", result)
    if (
        result.get("versionNumber") != 1
        or result.get("ref") != "/code/" + contract["kernel_slug"]
        or result.get("error")
    ):
        raise ValueError("SaveKernel outcome uncertain; reconcile read-only, no retry")
    return {**proof, "kernel_id": result.get("kernelId"), "version": 1}
