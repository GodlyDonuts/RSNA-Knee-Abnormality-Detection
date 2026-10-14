"""Root-side paired two-T4 production: notebook source and conditional coordinator phases.

PREPARED ONLY; writes nothing and calls no API on import. Root runs ``run_phase`` inside the
canonical single-coordinator claim exactly like ``window_bag_coordinator.run_phase``.
``preflight``/``status`` are read-only; ``push`` performs at most one SaveKernel after a fresh
quota, absence, platform-shape and input check. Before any segment intent it binds:

* the actual COMPLETE concurrent two-T4 engineering kernel and its receipt
  (``ENGINEERING_PARALLEL_STATUS``, both GPUs, combined RSS/CPU envelope), whose package is
  exactly the frozen engineering package this production package extends by one runner file;
* funding recomputed from that receipt's concurrent measurements (both provisioned T4s
  charged once per session, 9 h with margin, scratch and saved-output caps); and
* the one shared previous paired kernel (slug/version/source/receipt/both arm outputs).
"""

from __future__ import annotations

import base64
import gzip
import json

from scripts.kaggle import window_bag_coordinator as wc
from scripts.kaggle import window_bag_notebook as nb
from scripts.kaggle import window_bag_paired as wp
from scripts.kaggle import window_bag_plan as plan
from scripts.kaggle import window_bag_t4 as t4

_ENGINEERING_CALL = 'str(code / "scripts/kaggle/window_bag_t4.py"), "--mode", mode, '
_ENGINEERING_TAIL = nb.BRANCH[
    nb.BRANCH.index('        if mode == "engineering":\n') : nb.BRANCH.index(
        "        bounded_process("
    )
]
PAIRED_BRANCH = (
    nb.BRANCH.replace('"WINDOW_BAG_T4_PROBE"', '"WINDOW_BAG_PAIRED_PROBE"')
    .replace(_ENGINEERING_CALL, 'str(code / "scripts/kaggle/window_bag_paired.py"), ')
    .replace(
        _ENGINEERING_TAIL,
        '        (private / "segment.json").write_text(json.dumps(WINDOW_BAG_SEGMENT))\n'
        '        command += ["--segment", str(private / "segment.json")]\n',
    )
)


def production_package() -> tuple[dict, dict]:
    """Frozen engineering package (immutable science + frozen wrapper/plan) plus the runner."""
    files, manifest = nb.immutable_package()
    if manifest["bound_current"] != wp.FROZEN_ENGINEERING_BOUND:
        raise ValueError("Current wrapper/plan differ from the frozen engineering source")
    data = (nb.ROOT / wp.PAIRED_MODULE).read_bytes()
    files[wp.PAIRED_MODULE] = data.decode()
    bound = {**manifest["bound_current"], wp.PAIRED_MODULE: nb.sha_bytes(data)}
    return files, dict(manifest, bound_current=bound)


def compose(entry: str, values: dict) -> str:
    if (
        entry.count(nb.ANCHOR) != 1
        or PAIRED_BRANCH.count("window_bag_paired.py") != 1
        or "--mode" in PAIRED_BRANCH
        or "WINDOW_BAG_T4_PROBE" in PAIRED_BRANCH
    ):
        raise ValueError("Pinned entry anchor/paired branch differs")
    header = "".join(f"{k} = {v!r}\n" for k, v in values.items())
    notebook = header + entry.replace(nb.ANCHOR, PAIRED_BRANCH + nb.ANCHOR)
    compile(notebook, "window-bag-paired-notebook.py", "exec")
    if len(notebook.encode()) >= nb.SOURCE_CAP_BYTES:
        raise ValueError("Private notebook source exceeds its explicit cap")
    return notebook


def build(
    *,
    bundle_manifest_sha256: str,
    capsule: dict,
    fixture: bytes,
    bag_cpu: bytes,
    overlay: bytes,
    segment: dict,
    timeout_seconds: int,
) -> tuple[str, dict]:
    """Return (paired segment notebook source, binding record). Writes nothing, no API."""
    wp.validate_segment(segment)
    if set(capsule) != {"archive_name", "archive_sha256", "arguments"} or set(
        capsule["arguments"]
    ) != {"studies", "artifact-root", "folds", "series"}:
        raise ValueError("Exact render-capsule layout required")
    if not 0 < timeout_seconds <= plan.SESSION_LIMIT_SECONDS * plan.SESSION_MARGIN:
        raise ValueError("Paired session timeout exceeds its bound")
    files, manifest = production_package()
    text = wp.manifest_text(manifest)
    if wp.engineering_manifest_sha256(manifest) != segment["engineering_package_manifest_sha256"]:
        raise ValueError("Segment binds a different engineering package")
    package = gzip.compress(json.dumps({"files": files, "manifest": text}).encode(), mtime=0)
    inputs = nb.private_inputs(fixture, bag_cpu, overlay)
    values = {
        "EXPECTED_MANIFEST_SHA256": bundle_manifest_sha256,
        "WINDOW_BAG_PAIRED_PROBE": True,
        "WINDOW_BAG_MODE": wp.PAIRED_MODE,
        "WINDOW_BAG_PARALLEL": True,
        "WINDOW_BAG_SEGMENT": segment,
        "WINDOW_BAG_TIMEOUT_SECONDS": timeout_seconds,
        "WINDOW_BAG_CAPSULE": capsule,
        "WINDOW_BAG_MANIFEST_SHA256": nb.sha_bytes(text.encode()),
        "WINDOW_BAG_PACKAGE": base64.b64encode(package).decode(),
        "WINDOW_BAG_PRIVATE": {
            k: base64.b64encode(gzip.compress(v, mtime=0)).decode() for k, v in inputs.items()
        },
        "WINDOW_BAG_PRIVATE_SHA256": {k: nb.sha_bytes(v) for k, v in inputs.items()},
    }
    notebook = compose(nb.ENTRY.read_text(), values)
    record = {
        "source_sha256": nb.sha_bytes(notebook.encode()),
        "source_bytes": len(notebook.encode()),
        "science_git": t4.SCIENCE_GIT,
        "package_manifest_sha256": values["WINDOW_BAG_MANIFEST_SHA256"],
        "engineering_package_manifest_sha256": segment["engineering_package_manifest_sha256"],
        "bound_current": manifest["bound_current"],
        "entry_sha256": nb.sha_bytes(nb.ENTRY.read_bytes()),
        "mode": wp.PAIRED_MODE,
        "parallel": True,
        "segment": segment,
    }
    return notebook, record


def validate_contract(contract: dict) -> dict:
    if (
        not isinstance(contract, dict)
        or set(contract) != wc.CONTRACT_KEYS
        or contract["mode"] != wp.PAIRED_MODE
        or contract["parallel_arms"] is not True
    ):
        raise ValueError("Exact paired two-T4 Kaggle contract required")
    shards = contract["shard_kernels"]
    if (
        len(shards) != plan.SHARD_KERNEL_SOURCES
        or [s["shard_index"] for s in shards] != list(range(plan.SHARD_KERNEL_SOURCES))
        or any(s["version"] != 1 for s in shards)
        or [s["shard_manifest_sha256"] for s in shards] != list(t4.SHARD_MANIFEST_SHA256)
        or len({s["slug"] for s in shards}) != plan.SHARD_KERNEL_SOURCES
    ):
        raise ValueError("Exactly the 20 original completed shard kernels, version 1")
    timeout = contract["session_timeout_seconds"]
    if type(timeout) is not int or not 0 < timeout <= plan.SESSION_LIMIT_SECONDS * (
        plan.SESSION_MARGIN
    ):
        raise ValueError("Session timeout exceeds its bound")
    segment = wp.validate_segment(contract["segment"])
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
        raise ValueError("Paired segments require exact engineering-bound funding")
    return contract


def verify_engineering(remote, contract, digest, gpu) -> dict:
    """The actual concurrent two-T4 engineering acceptance and its measured performance."""
    funding, segment = contract["production_funding"], contract["segment"]
    receipt = wc.completed_gpu_kernel(
        remote, contract, funding["engineering"], digest, t4.ENGINEERING_RECEIPT, []
    )
    arms = receipt.get("arms", {})
    concurrency = receipt.get("concurrency") or {}
    before = receipt.get("resources_before") or {}
    if (
        receipt.get("status") != t4.ENGINEERING_PARALLEL_STATUS
        or receipt.get("GPUs_used_concurrently") != len(t4.PARALLEL_DEVICES)
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
    ):
        raise ValueError("Engineering receipt is not the actual concurrent source/config/inputs")
    rss = concurrency.get("peak_combined_rss_bytes")
    if (
        concurrency.get("devices") != t4.PARALLEL_DEVICES
        or concurrency.get("threads_per_process") != 2
        or not t4._hex64(concurrency.get("shared_read_only_cache_sha256"))
        or type(rss) is not int
        or not 0 < rss <= t4.RAM_CEILING_BYTES
        or before.get("GPUs_visible") != len(t4.PARALLEL_DEVICES)
        or type(before.get("CPUs")) is not int
        or before["CPUs"] < t4.PARALLEL_MIN_CPUS
    ):
        raise ValueError("Engineering session did not run both arms within 4 CPUs/29 GiB")
    measured = receipt.get("measured")
    if not isinstance(measured, dict) or funding["measured"] != measured:
        raise ValueError("Funding timings differ from the actual concurrent engineering session")
    sized = wp.paired_plan(measured, gpu, before["disk"]["/tmp"]["free"])
    segments = [list(s) for s in sized["segments_per_arm"]]
    index = segment["segment"]
    if (
        funding["segments_per_arm"] != segments
        or index >= len(segments)
        or segments[index] != [segment["start_step"], segment["end_step"]]
        or contract["session_timeout_seconds"] != sized["session_timeout_seconds"]
        or segment["output_cap_bytes"] != sized["output_cap_bytes_per_segment"][index]
        or not sized["scratch_fits"]
        or not sized["run_scratch_fits"]
        or not sized["output_caps_fit"]
        or type(funding["gpu_seconds_authorized"]) not in (int, float)
        or funding["gpu_seconds_authorized"] < sized["session_charge_seconds"]
    ):
        raise ValueError("Funding does not match the concurrent measured plan for this segment")
    return {"receipt_sha256": funding["engineering"]["receipt_sha256"], "plan": sized}


def verify_previous(remote, contract, digest) -> dict | None:
    """Read-only: the one bound previous paired kernel actually completed with this output."""
    segment = contract["segment"]
    previous = segment["previous_output"]
    if previous is None:
        return None
    receipt = wc.completed_gpu_kernel(
        remote,
        contract,
        previous,
        digest,
        wp.PAIRED_RECEIPT,
        wc._previous_inputs(remote, previous, contract),
    )
    wp.check_previous_receipt(receipt, segment)
    if receipt.get("package_manifest_sha256") != contract["package_manifest_sha256"] or any(
        receipt["arms"][arm]["publication"]["manifest_sha256"]
        != previous["publication_sha256"][arm]
        or receipt["arms"][arm]["checkpoint_sha256"] != previous["checkpoint_sha256"][arm]
        for arm in t4.ARMS
    ):
        raise ValueError("Previous paired output differs from the bound identity")
    return {"kernel_slug": previous["kernel_slug"], "receipt_sha256": previous["receipt_sha256"]}


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
    if "WINDOW_BAG_PAIRED_PROBE = True\n" not in source:
        raise ValueError("Notebook source is not the paired runner")
    request = wc.kernel_request(contract, source)
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
    if plan.seconds(gpu["timeReserved"]) != 0:
        raise ValueError("Another reservation exists; reconcile before a new session")
    # Both provisioned T4s are charged exactly once for the full session timeout.
    funding = {
        "available_seconds": plan.conservative_available(gpu),
        "charge_seconds": plan.session_charge(contract["session_timeout_seconds"]),
        "provisioned_GPUs_charged": plan.PROVISIONED_T4,
        "GPUs_used_concurrently": len(t4.PARALLEL_DEVICES),
    }
    if funding["available_seconds"] < funding["charge_seconds"]:
        raise ValueError("Fresh conservative allowance cannot fund this session")
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
