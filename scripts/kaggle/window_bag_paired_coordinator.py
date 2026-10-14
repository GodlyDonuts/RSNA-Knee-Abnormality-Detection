"""Conditional single-coordinator phases for the paired two-T4 production segments.

PREPARED ONLY. Root runs ``run_phase`` inside the canonical Newton coordinator claim with
the canonical account ledger, exactly like ``window_bag_coordinator``. ``preflight`` and
``status`` are read-only; ``push`` performs at most one SaveKernel and never retries an
uncertain outcome. Before any intent this verifies, read-only:

* the actual completed concurrent engineering kernel (``ENGINEERING_PARALLEL_STATUS``,
  two GPUs, devices 0/1, two threads each, shared read-only cache, combined RSS within
  29 GiB), its exact source, image, inputs, package and per-arm production configs;
* funding equal to that session's actual measured timings and the paired plan;
* the fresh conservative allowance covers one session charged for both provisioned T4s;
* the exact completed previous paired kernel output (one shared kernel source, so a
  resumed segment requests 20 shards + 1 previous output within the 21/3 shape).
"""

from __future__ import annotations

from scripts.kaggle import window_bag_coordinator as wc
from scripts.kaggle import window_bag_paired as wp
from scripts.kaggle import window_bag_plan as plan
from scripts.kaggle import window_bag_t4 as t4

MODE = "paired_segment"


def validate_contract(contract: dict) -> dict:
    if (
        not isinstance(contract, dict)
        or set(contract) != wc.CONTRACT_KEYS
        or contract["mode"] != MODE
        or contract["parallel_arms"] is not True
    ):
        raise ValueError("Exact paired two-T4 production contract required")
    # The shard and timeout rules are the coordinator's own; only mode/segment differ.
    wc.validate_contract(
        dict(
            contract,
            mode="engineering",
            parallel_arms=True,
            production_funding=None,
            segment=None,
            session_timeout_seconds=1,
        )
    )
    if (
        not 0
        < contract["session_timeout_seconds"]
        <= (plan.SESSION_LIMIT_SECONDS * plan.SESSION_MARGIN)
    ):
        raise ValueError("Session timeout exceeds the nine-hour margin")
    segment = wp.paired_contract(contract["segment"])
    funding = contract["production_funding"]
    if (
        not isinstance(funding, dict)
        or set(funding) != wc.FUNDING_KEYS
        or not isinstance(funding["engineering"], dict)
        or set(funding["engineering"]) != wc.KERNEL_IDENTITY_KEYS
        or funding["engineering"]["version"] != 1
        or funding["engineering"]["receipt_sha256"] != segment["engineering_receipt_sha256"]
        or not isinstance(funding["authorized_at"], str)
        or not funding["authorized_at"].strip()
    ):
        raise ValueError("Paired segments require exact concurrent-engineering-bound funding")
    if segment["engineering_package_manifest_sha256"] == contract["package_manifest_sha256"]:
        raise ValueError("Paired package must extend, not replace, the engineering package")
    return contract


def verify_engineering(remote, contract, digest, gpu) -> dict:
    """Bind the actual concurrent two-T4 engineering acceptance and its measured timings."""
    funding, segment = contract["production_funding"], contract["segment"]
    identity = funding["engineering"]
    receipt = wc.completed_gpu_kernel(
        remote, contract, identity, digest, t4.ENGINEERING_RECEIPT, []
    )
    source = (
        remote.kernel("GetKernel", identity["kernel_slug"], 1).get("blob", {}).get("source", "")
    )
    if (
        "WINDOW_BAG_MODE = 'engineering'\n" not in source
        or "WINDOW_BAG_PARALLEL = True\n" not in source
        or f"WINDOW_BAG_MANIFEST_SHA256 = {segment['engineering_package_manifest_sha256']!r}\n"
        not in source
    ):
        raise ValueError("Engineering kernel is not the concurrent engineering source")
    arms = receipt.get("arms", {})
    concurrency = receipt.get("concurrency") or {}
    resources = receipt.get("resources_before") or {}
    if (
        receipt.get("status") != t4.ENGINEERING_PARALLEL_STATUS
        or receipt.get("GPUs_used_concurrently") != 2
        or receipt.get("science_git") != t4.SCIENCE_GIT
        or receipt.get("package_manifest_sha256") != segment["engineering_package_manifest_sha256"]
        or (
            receipt.get("fixture_sha256"),
            receipt.get("bag_CPU_sha256"),
            receipt.get("index_sha256"),
        )
        != (t4.FIXTURE_SHA256, t4.BAG_CPU_SHA256, t4.INDEX_SHA256)
        or receipt.get("shard_manifest_sha256") != list(t4.SHARD_MANIFEST_SHA256)
        or set(arms) != set(t4.ARMS)
        or any(
            a.get("complete_actual_partial_final_RNG_optimizer_scaler_and_CSV_recovery") is not True
            for a in arms.values()
        )
        or receipt.get("gold_or_confirmation_used") is not False
        or receipt.get("production_fitting") is not False
        or receipt.get("clinical_accuracy_evidence") is not False
        or receipt.get("production_config_sha256") != segment["production_config_sha256"]
        or concurrency.get("devices") != t4.PARALLEL_DEVICES
        or concurrency.get("threads_per_process") != 2
        or not t4._hex64(concurrency.get("shared_read_only_cache_sha256"))
        or type(concurrency.get("peak_combined_rss_bytes")) is not int
        or not 0 < concurrency["peak_combined_rss_bytes"] <= t4.RAM_CEILING_BYTES
        or resources.get("GPUs_visible") != 2
        or type(resources.get("CPUs")) is not int
        or resources["CPUs"] < t4.PARALLEL_MIN_CPUS
    ):
        raise ValueError("Engineering receipt is not the actual concurrent two-T4 acceptance")
    measured = receipt.get("measured")
    if (
        not isinstance(measured, dict)
        or set(measured) != set(t4.MEASURED_KEYS)
        or (funding["measured"] != measured)
    ):
        raise ValueError("Funding timings differ from the actual concurrent engineering session")
    sized = wp.paired_plan(measured, gpu, resources["disk"]["/tmp"]["free"])
    segments = [list(s) for s in sized["segments_per_arm"]]
    index = segment["segment"]
    if (
        funding["segments_per_arm"] != segments
        or index >= len(segments)
        or segments[index] != [segment["start_step"], segment["end_step"]]
        or contract["session_timeout_seconds"] != sized["session_timeout_seconds"]
        or segment["arm_output_cap_bytes"] != sized["arm_output_cap_bytes_per_segment"][index]
        or segment["output_cap_bytes"] != sized["output_cap_bytes_per_segment"][index]
        or segment["scratch_cap_bytes"] != sized["scratch_cap_bytes_per_segment"][index]
        or not sized["scratch_fits"]
        or not sized["output_caps_fit"]
        or not sized["session_within_9h_margin"]
        or type(funding["gpu_seconds_authorized"]) not in (int, float)
        or funding["gpu_seconds_authorized"] < sized["session_charge_seconds"]
    ):
        raise ValueError("Funding does not match the measured paired plan for this segment")
    return {"receipt_sha256": identity["receipt_sha256"], "plan": sized}


def verify_previous(remote, contract, digest) -> dict | None:
    """Read-only: the one bound previous paired kernel actually completed with both arms."""
    segment = contract["segment"]
    previous = segment["previous_output"]
    if previous is None:
        return None
    identity = {k: previous[k] for k in wc.KERNEL_IDENTITY_KEYS}
    receipt = wc.completed_gpu_kernel(
        remote,
        contract,
        identity,
        digest,
        wp.PAIRED_RECEIPT,
        _previous_inputs(remote, previous, contract),
    )
    source = (
        remote.kernel("GetKernel", previous["kernel_slug"], 1).get("blob", {}).get("source", "")
    )
    if (
        f"WINDOW_BAG_MODE = {MODE!r}\n" not in source
        or f"WINDOW_BAG_MANIFEST_SHA256 = {contract['package_manifest_sha256']!r}\n" not in source
        or receipt.get("package_manifest_sha256") != contract["package_manifest_sha256"]
    ):
        raise ValueError("Previous paired kernel ran a different package")
    wp.check_previous_receipt(receipt, segment)
    return {"kernel_slug": previous["kernel_slug"], "receipt_sha256": previous["receipt_sha256"]}


def _previous_inputs(remote, previous, contract) -> list[str]:
    meta = remote.kernel("GetKernel", previous["kernel_slug"], 1)["metadata"]
    shards = {s["slug"] for s in contract["shard_kernels"]}
    extra = [k for k in wc.repeated(meta, "kernelDataSources") if k not in shards]
    if len(extra) != (0 if contract["segment"]["segment"] == 1 else 1):
        raise ValueError("Previous paired kernel inputs differ from its chain position")
    return extra


def run_phase(phase, contract, source, remote, publication, workflow, journal, digest):
    """``journal`` is the root-owned exclusive-JSON writer inside the coordinator claim."""
    validate_contract(contract)
    if contract["platform_limits"] != plan.admitted_platform_limits() or (
        contract["model_dataset"]["id"],
        contract["capsule_dataset"]["id"],
    ) != (plan.SUPPORTED_DATASET_IDS[0], plan.SUPPORTED_DATASET_IDS[2]):
        raise ValueError("Platform proof/Datasets differ from the actual admitted input shape")
    if digest(source) != contract["source_sha256"]:
        raise ValueError("Prepared notebook source differs from its contract")
    wc.bound_in_source(contract, source)
    request = wc.kernel_request(contract, source)
    # Both arms share the one previous output: at most 21 kernel sources, never 22.
    plan.require_platform_limits(
        contract["platform_limits"],
        len(request["kernelDataSources"]),
        len(request["datasetDataSources"]),
        contract["segment"]["output_cap_bytes"],
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
            if sorted(wc.repeated(meta, name)) != sorted(request[name]):
                raise ValueError("Pushed kernel inputs differ; inspect, never retry")
        return {"status": remote.kernel("GetKernelSessionStatus", contract["kernel_slug"], 1)}
    if phase not in {"preflight", "push"}:
        raise ValueError("Unknown phase")
    inputs = wc.verify_inputs(remote, publication, workflow, contract)
    gpu = remote.call("kernels.KernelsApiService", "GetAcceleratorQuotaStatistics", {})["gpuQuota"]
    funding = {
        "available_seconds": plan.conservative_available(gpu),
        "charge_seconds": plan.session_charge(contract["session_timeout_seconds"]),
    }
    if funding["available_seconds"] < funding["charge_seconds"]:
        raise ValueError("Fresh conservative allowance cannot fund this paired session")
    funding = {
        **funding,
        "provisioned_GPUs_charged": plan.PROVISIONED_T4,
        "GPUs_used_concurrently": 2,
    }
    production = {
        "engineering": verify_engineering(remote, contract, digest, gpu),
        "previous": verify_previous(remote, contract, digest),
    }
    absence = remote.require_absent_kernel(contract["kernel_slug"])
    proof = {
        "phase": phase,
        "inputs": inputs,
        "gpu_quota": gpu,
        "funding": funding,
        "production": production,
        "absence": absence,
        "request_sha256": digest(repr(sorted(request.items()))),
    }
    if phase == "preflight":
        return proof
    journal(
        "kernel-create-intent.json", {**proof, "single_SaveKernel": True, "automatic_retry": False}
    )
    try:
        result = remote.call("kernels.KernelsApiService", "SaveKernel", request)
    except Exception as exc:
        journal(
            "kernel-create-uncertain.json",
            {"error_type": type(exc).__name__, "reconcile_read_only": True},
        )
        raise
    journal("kernel-create-response.json", result)
    if (
        result.get("versionNumber") != 1
        or result.get("ref") != "/code/" + contract["kernel_slug"]
        or result.get("error")
    ):
        raise ValueError("SaveKernel outcome uncertain; reconcile read-only, no retry")
    return {**proof, "kernel_id": result.get("kernelId"), "version": 1}
