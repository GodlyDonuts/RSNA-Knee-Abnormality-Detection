"""Conditional single-coordinator phases for the private whole-window T4 notebook.

PREPARED ONLY. Root runs ``run_phase`` inside the canonical Newton coordinator claim
(``scripts/kaggle/coordinator.py:single_coordinator``) with the canonical account ledger,
exactly like the original T4 recoveries. ``preflight`` and ``status`` are read-only.
``push`` performs at most one SaveKernel after a fresh quota, absence, platform-limit and
input-identity check. Any uncertain SaveKernel outcome is reconciled read-only; there is
no automatic retry. Production segments additionally require the actual accepted engineering
kernel receipt (read back from its completed output and hash-bound), funding equal to its
measured performance and plan, the exact completed previous-segment kernel output, a fresh
allowance and an explicit output cap; all are verified before any segment intent.
"""

from __future__ import annotations

from scripts.kaggle import window_bag_plan as plan
from scripts.kaggle import window_bag_t4 as t4

MACHINE_SHAPE = "NvidiaTeslaT4"
CONTRACT_KEYS = {
    "kernel_slug", "source_sha256", "package_manifest_sha256", "mode", "segment",
    "shard_kernels", "model_dataset", "wheelhouse", "capsule_dataset", "runtime_image",
    "session_timeout_seconds", "platform_limits", "production_funding",
}
FUNDING_KEYS = {"engineering", "measured", "segments_per_arm", "gpu_seconds_authorized",
                "authorized_at"}
KERNEL_IDENTITY_KEYS = {"kernel_slug", "version", "source_sha256", "receipt_sha256"}


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
    if contract["mode"] == "engineering":
        if contract["production_funding"] is not None or contract["segment"] is not None:
            raise ValueError("Engineering contract must not carry production funding/segment")
        return contract
    t4.segment_contract(contract["segment"])
    funding = contract["production_funding"]
    if (
        not isinstance(funding, dict)
        or set(funding) != FUNDING_KEYS
        or not isinstance(funding["engineering"], dict)
        or set(funding["engineering"]) != KERNEL_IDENTITY_KEYS
        or funding["engineering"]["version"] != 1
        or funding["engineering"]["receipt_sha256"]
        != contract["segment"]["engineering_receipt_sha256"]
        or not isinstance(funding["authorized_at"], str)
        or not funding["authorized_at"].strip()
    ):
        raise ValueError("Production segments require exact engineering-bound funding")
    return contract


def bound_in_source(contract: dict, source: str) -> None:
    """The contract's segment and package identity are the ones embedded in the source."""
    if (
        f"WINDOW_BAG_SEGMENT = {contract['segment']!r}\n" not in source
        or f"WINDOW_BAG_MODE = {contract['mode']!r}\n" not in source
        or f"WINDOW_BAG_MANIFEST_SHA256 = {contract['package_manifest_sha256']!r}\n" not in source
    ):
        raise ValueError("Notebook source does not embed the contract's segment/package")


def previous_slug(contract: dict) -> str | None:
    previous = (contract["segment"] or {}).get("previous_output")
    return None if previous is None else previous["kernel_slug"]


def kernel_request(contract: dict, source: str):
    kernels = [s["slug"] for s in contract["shard_kernels"]]
    previous = previous_slug(contract)
    if previous is not None:
        if previous in kernels or previous == contract["kernel_slug"]:
            raise ValueError("Previous segment output must be a distinct kernel source")
        kernels.append(previous)
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


def completed_gpu_kernel(remote, contract, identity, digest, receipt_name, extra_kernels):
    """Read-only: an actual completed private offline T4 v1 of the exact source and inputs."""
    original = remote.kernel("GetKernel", identity["kernel_slug"], 1)
    meta = original["metadata"]
    shards = [s["slug"] for s in contract["shard_kernels"]]
    datasets = kernel_request(contract, "")["datasetDataSources"]
    if (
        digest(original.get("blob", {}).get("source", "")) != identity["source_sha256"]
        or meta.get("isPrivate") is not True
        or meta.get("enableGpu") is not True
        or meta.get("enableInternet") is not False
        or meta.get("machineShape") != MACHINE_SHAPE
        or meta.get("dockerImage") != contract["runtime_image"]
        or sorted(meta.get("datasetDataSources", [])) != sorted(datasets)
        or sorted(meta.get("kernelDataSources", [])) != sorted(shards + extra_kernels)
        or meta.get("competitionDataSources", []) != []
    ):
        raise ValueError("Bound kernel source/settings/inputs differ")
    if remote.kernel("GetKernelSessionStatus", identity["kernel_slug"], 1).get("status") != (
        "COMPLETE"
    ):
        raise ValueError("Bound kernel version is not COMPLETE")
    receipt = remote.output_json(identity["kernel_slug"], 1, receipt_name)
    if t4.canonical(receipt) != identity["receipt_sha256"]:
        raise ValueError("Actual kernel output receipt differs from its bound hash")
    return receipt


def verify_engineering(remote, contract, digest, gpu) -> dict:
    """Bind the actual T4 engineering acceptance and its measured performance."""
    funding, segment = contract["production_funding"], contract["segment"]
    receipt = completed_gpu_kernel(remote, contract, funding["engineering"], digest,
                                   t4.ENGINEERING_RECEIPT, [])
    arms = receipt.get("arms", {})
    if (
        receipt.get("status") != t4.ENGINEERING_STATUS
        or receipt.get("science_git") != t4.SCIENCE_GIT
        or receipt.get("package_manifest_sha256") != contract["package_manifest_sha256"]
        or (receipt.get("fixture_sha256"), receipt.get("bag_CPU_sha256"),
            receipt.get("index_sha256")) != (t4.FIXTURE_SHA256, t4.BAG_CPU_SHA256, t4.INDEX_SHA256)
        or receipt.get("shard_manifest_sha256") != list(t4.SHARD_MANIFEST_SHA256)
        or set(arms) != set(t4.ARMS)
        or any(
            a.get("complete_actual_partial_final_RNG_optimizer_scaler_and_CSV_recovery")
            is not True for a in arms.values()
        )
        or receipt.get("GPUs_used_concurrently") != 1
        or receipt.get("gold_or_confirmation_used") is not False
        or receipt.get("production_fitting") is not False
        or receipt.get("clinical_accuracy_evidence") is not False
        or receipt.get("production_config_sha256", {}).get(segment["arm"])
        != segment["production_config_sha256"]
    ):
        raise ValueError("Engineering receipt is not the actual accepted source/config/inputs")
    measured = receipt.get("measured")
    if not isinstance(measured, dict) or set(measured) != set(t4.MEASURED_KEYS) or (
        funding["measured"] != measured
    ):
        raise ValueError("Funding timings differ from the actual measured engineering session")
    scratch = receipt["resources_before"]["disk"]["/tmp"]["free"]
    sized = plan.production_plan(measured, gpu, scratch)
    segments = [list(s) for s in sized["segments_per_arm"]]
    index = segment["segment"]
    if (
        funding["segments_per_arm"] != segments
        or index >= len(segments)
        or segments[index] != [segment["start_step"], segment["end_step"]]
        or contract["session_timeout_seconds"] != sized["session_timeout_seconds"]
        or segment["output_cap_bytes"] != sized["output_cap_bytes_per_segment"][index]
        or not sized["scratch_fits"]
        or type(funding["gpu_seconds_authorized"]) not in (int, float)
        or funding["gpu_seconds_authorized"] < plan.session_charge(sized["session_timeout_seconds"])
    ):
        raise ValueError("Funding does not match the measured plan for this segment")
    return {"receipt_sha256": funding["engineering"]["receipt_sha256"], "plan": sized}


def verify_previous(remote, contract, digest) -> dict | None:
    """Read-only: the bound previous segment kernel actually completed with this output."""
    segment = contract["segment"]
    previous = segment["previous_output"]
    if previous is None:
        return None
    receipt = completed_gpu_kernel(remote, contract, previous, digest, t4.SEGMENT_RECEIPT,
                                   _previous_inputs(remote, previous, contract))
    prior = receipt.get("contract", {})
    if (
        receipt.get("status") != t4.SEGMENT_STATUS
        or receipt.get("package_manifest_sha256") != contract["package_manifest_sha256"]
        or receipt["publication"]["manifest_sha256"] != previous["publication_sha256"]
        or receipt["publication"]["checkpoint_sha256"] != previous["checkpoint_sha256"]
        or receipt.get("checkpoint_sha256") != previous["checkpoint_sha256"]
        or receipt.get("final_epoch10_evaluated") is not False
        or (prior.get("arm"), prior.get("segment", -2) + 1, prior.get("end_step"))
        != (segment["arm"], segment["segment"], segment["start_step"])
        or prior.get("engineering_receipt_sha256") != segment["engineering_receipt_sha256"]
        or prior.get("production_config_sha256") != segment["production_config_sha256"]
    ):
        raise ValueError("Previous segment output differs from the bound identity")
    return {"kernel_slug": previous["kernel_slug"], "receipt_sha256": previous["receipt_sha256"]}


def _previous_inputs(remote, previous, contract) -> list[str]:
    """The previous segment's own previous-output kernel source, from its pushed metadata."""
    meta = remote.kernel("GetKernel", previous["kernel_slug"], 1)["metadata"]
    shards = {s["slug"] for s in contract["shard_kernels"]}
    extra = [k for k in meta.get("kernelDataSources", []) if k not in shards]
    if len(extra) != (0 if contract["segment"]["segment"] == 1 else 1):
        raise ValueError("Previous segment kernel inputs differ from its chain position")
    return extra


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
    bound_in_source(contract, source)
    request = kernel_request(contract, source)
    plan.require_platform_limits(
        contract["platform_limits"],
        len(request["kernelDataSources"]),
        len(request["datasetDataSources"]),
        (contract["segment"] or {}).get("output_cap_bytes", 0),
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
    production = None
    if contract["mode"] == "segment":
        # Both bindings are verified read-only before any segment intent or SaveKernel.
        production = {"engineering": verify_engineering(remote, contract, digest, gpu),
                      "previous": verify_previous(remote, contract, digest)}
    absence = remote.require_absent_kernel(contract["kernel_slug"])
    proof = {"phase": phase, "inputs": inputs, "gpu_quota": gpu, "funding": funding,
             "production": production, "absence": absence,
             "request_sha256": digest(repr(sorted(request.items())))}
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
