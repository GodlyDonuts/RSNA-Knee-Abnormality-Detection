"""Paired two-T4 production continuation (issue #55): CPU stand-ins, never GPU acceptance."""

import copy
import json
import os
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_kaggle_window_bag import (
    ENGINEERING_SLUG,
    ISSUE_QUOTA,
    SHARD_SLUGS,
    FakeRemote,
    Publication,
    Workflow,
    contract,
    digest,
    engineering_receipt,
    fake_run,
    gpu_kernel,
    measured,
    notebook_source,
    read_only,
    writable,
)

from scripts.kaggle import window_bag_notebook as nb
from scripts.kaggle import window_bag_paired as wp
from scripts.kaggle import window_bag_paired_coordinator as pc
from scripts.kaggle import window_bag_paired_notebook as pn
from scripts.kaggle import window_bag_plan as plan
from scripts.kaggle import window_bag_t4 as t4

ROOT = Path(__file__).resolve().parents[1]
CHILD = ROOT / "tests/window_bag_session_child.py"
LINUX = pytest.mark.skipif(
    not sys.platform.startswith("linux"),
    reason="Supervisor RSS and reaping checks read Linux /proc",
)
ENG_PKG = "9" * 64
PKG = "7" * 64
CONFIGS = {"mean": "c" * 64, "query": "d" * 64}
BINDINGS = {
    "engineering_receipt_sha256": "e" * 64,
    "engineering_package_manifest_sha256": ENG_PKG,
    "production_config_sha256": CONFIGS,
}
PREVIOUS_KERNEL = {
    "kernel_slug": "me/rsna-window-bag-paired-seg0-v1",
    "version": 1,
    "source_sha256": "f" * 64,
}


def paired_receipt(segment, checkpoint="a", publication="b"):
    arms = {
        arm: {
            "step": segment["end_step"],
            "checkpoint_sha256": (checkpoint + str(i)) * 32,
            "publication": {
                "manifest_sha256": (publication + str(i)) * 32,
                "checkpoint_sha256": (checkpoint + str(i)) * 32,
            },
        }
        for i, arm in enumerate(t4.ARMS)
    }
    return {
        "status": wp.PAIRED_SEGMENT_STATUS,
        "contract": segment,
        "arms": arms,
        "package_manifest_sha256": PKG,
        "final_epoch10_evaluated": False,
        "intermediate_selection": False,
    }


# --- Plan and contract ---------------------------------------------------------------


def test_paired_plan_runs_both_full_schedules_and_charges_both_T4s_once_per_session():
    sized = wp.paired_plan(measured(), ISSUE_QUOTA, 40 * 2**30)
    sequential = plan.production_plan(measured(), ISSUE_QUOTA, 40 * 2**30)
    segments = sized["segments_per_arm"]
    assert segments == sequential["segments_per_arm"]
    assert segments[0][0] == 0 and segments[-1][1] == plan.TOTAL_UPDATES == 2810
    assert sized["sessions"] == len(segments) and sized["paired_segment_runner"]
    assert sized["session_charge_seconds"] == 2 * sized["session_timeout_seconds"]
    assert (
        sized["conservative_charge_seconds"] == sized["sessions"] * sized["session_charge_seconds"]
    )
    assert sized["session_timeout_seconds"] <= 9 * 3600 * 0.8 and sized["session_within_9h_margin"]
    assert sized["output_cap_bytes_per_segment"] == [
        2 * c for c in sized["arm_output_cap_bytes_per_segment"]
    ]
    assert all(
        s > o
        for s, o in zip(
            sized["scratch_cap_bytes_per_segment"], sized["output_cap_bytes_per_segment"]
        )
    )
    assert not sized["scientific_hyperparameters_changed"] and sized["only_final_epoch10_evaluated"]
    starved = wp.paired_plan(measured(), dict(ISSUE_QUOTA, timeUsed="100000s"), 40 * 2**30)
    assert starved["quota_fits_now"] is False and starved["next_session_quota_fits_now"] is False
    assert wp.paired_plan(measured(), ISSUE_QUOTA, 1)["scratch_fits"] is False


def test_paired_chain_binds_one_previous_output_for_both_arms():
    segments = [(0, 2), (2, 4), (4, 6)]
    first = wp.next_paired_segment(None, segments, measured=measured(), **BINDINGS)
    assert (
        first["previous_output"] is None
        and first["output_cap_bytes"] == 2 * first["arm_output_cap_bytes"]
    )
    receipt = paired_receipt(first)
    nxt = wp.next_paired_segment(
        receipt, segments, measured=measured(), previous_kernel=PREVIOUS_KERNEL, **BINDINGS
    )
    previous = nxt["previous_output"]
    assert (nxt["start_step"], previous["kernel_slug"]) == (2, PREVIOUS_KERNEL["kernel_slug"])
    assert previous["receipt_sha256"] == t4.canonical(receipt)
    assert {a: previous["arms"][a]["checkpoint_sha256"] for a in t4.ARMS} == {
        a: receipt["arms"][a]["checkpoint_sha256"] for a in t4.ARMS
    }
    for tamper in ("status", "final", "arm", "step", "config", "engineering_package"):
        bad = copy.deepcopy(receipt)
        if tamper == "status":
            bad["status"] = t4.SEGMENT_STATUS
        elif tamper == "final":
            bad["final_epoch10_evaluated"] = True
        elif tamper == "arm":
            del bad["arms"]["query"]
        elif tamper == "step":
            bad["arms"]["query"]["step"] = 1
        elif tamper == "config":
            bad["contract"] = dict(first, production_config_sha256={**CONFIGS, "query": "0" * 64})
        else:
            bad["contract"] = dict(first, engineering_package_manifest_sha256="0" * 64)
        with pytest.raises(ValueError):
            wp.next_paired_segment(
                bad, segments, measured=measured(), previous_kernel=PREVIOUS_KERNEL, **BINDINGS
            )
    with pytest.raises(ValueError):
        wp.next_paired_segment(
            receipt,
            segments,
            measured=measured(),
            previous_kernel=dict(PREVIOUS_KERNEL, version=2),
            **BINDINGS,
        )


@pytest.mark.parametrize(
    "tamper",
    [
        "arm_cap",
        "total_cap",
        "ceiling",
        "scratch",
        "same_configs",
        "one_config",
        "previous_arm",
        "previous_version",
        "two_previous",
        "first_with_previous",
        "beyond_schedule",
    ],
)
def test_paired_contract_refuses_cap_and_binding_drift(tamper):
    segments = [(0, 2), (2, 4)]
    first = wp.next_paired_segment(None, segments, measured=measured(), **BINDINGS)
    value = wp.next_paired_segment(
        paired_receipt(first),
        segments,
        measured=measured(),
        previous_kernel=PREVIOUS_KERNEL,
        **BINDINGS,
    )
    assert wp.paired_contract(copy.deepcopy(value))
    if tamper == "arm_cap":
        value["arm_output_cap_bytes"] += 1
    elif tamper == "total_cap":
        value["output_cap_bytes"] += 1
    elif tamper == "ceiling":
        value["arm_output_cap_bytes"] = plan.OUTPUT_CAP_CEILING_BYTES
        value["output_cap_bytes"] = 2 * plan.OUTPUT_CAP_CEILING_BYTES
        value["scratch_cap_bytes"] = 5 * plan.OUTPUT_CAP_CEILING_BYTES
    elif tamper == "scratch":
        value["scratch_cap_bytes"] = value["output_cap_bytes"]
    elif tamper == "same_configs":
        value["production_config_sha256"] = {"mean": "c" * 64, "query": "c" * 64}
    elif tamper == "one_config":
        value["production_config_sha256"] = {"mean": "c" * 64}
    elif tamper == "previous_arm":
        del value["previous_output"]["arms"]["query"]
    elif tamper == "previous_version":
        value["previous_output"]["version"] = 2
    elif tamper == "two_previous":
        value["previous_output"]["kernel_slug"] = [PREVIOUS_KERNEL["kernel_slug"]] * 2
    elif tamper == "first_with_previous":
        value["segment"] = 0
        value["start_step"] = 0
    else:
        value["end_step"] = plan.TOTAL_UPDATES + 1
    with pytest.raises(ValueError):
        wp.paired_contract(value)


# --- Coordinator --------------------------------------------------------------------------


def concurrent_receipt():
    receipt = engineering_receipt()
    receipt.update(
        status=t4.ENGINEERING_PARALLEL_STATUS,
        GPUs_used_concurrently=2,
        package_manifest_sha256=ENG_PKG,
        production_config_sha256=dict(CONFIGS),
        concurrency={
            "devices": dict(t4.PARALLEL_DEVICES),
            "threads_per_process": 2,
            "shared_read_only_cache_sha256": "5" * 64,
            "peak_combined_rss_bytes": 20 * 2**30,
            "peak_combined_output_bytes": 2**30,
            "journal_sha256": {a: "6" * 64 for a in t4.ARMS},
        },
        resources_before={"disk": {"/tmp": {"free": 40 * 2**30}}, "GPUs_visible": 2, "CPUs": 4},
    )
    return receipt


def paired_source(segment, manifest=PKG, mode=pc.MODE):
    return notebook_source(mode, segment, manifest=manifest, parallel=True)


def paired_production(segment_index=0, receipt=None):
    receipt = receipt or concurrent_receipt()
    eng_source = notebook_source("engineering", None, manifest=ENG_PKG, parallel=True)
    kernels = {ENGINEERING_SLUG: gpu_kernel(eng_source, t4.ENGINEERING_RECEIPT, receipt)}
    sized = wp.paired_plan(measured(), ISSUE_QUOTA, 40 * 2**30)
    segments = sized["segments_per_arm"]
    bindings = dict(BINDINGS, engineering_receipt_sha256=t4.canonical(receipt))
    segment = wp.next_paired_segment(None, segments, measured=measured(), **bindings)
    for index in range(segment_index):
        slug = f"me/rsna-window-bag-paired-seg{index}-v1"
        source = paired_source(segment)
        previous = paired_receipt(segment, checkpoint=f"{index:x}")
        extra = (
            []
            if segment["previous_output"] is None
            else [segment["previous_output"]["kernel_slug"]]
        )
        kernels[slug] = gpu_kernel(source, wp.PAIRED_RECEIPT, previous, extra)
        segment = wp.next_paired_segment(
            previous,
            segments,
            measured=measured(),
            previous_kernel={"kernel_slug": slug, "version": 1, "source_sha256": digest(source)},
            **bindings,
        )
    funding = {
        "engineering": {
            "kernel_slug": ENGINEERING_SLUG,
            "version": 1,
            "source_sha256": digest(eng_source),
            "receipt_sha256": t4.canonical(receipt),
        },
        "measured": measured(),
        "segments_per_arm": [list(x) for x in segments],
        "gpu_seconds_authorized": sized["session_charge_seconds"],
        "authorized_at": "root explicit funding record",
    }
    value = contract(
        kernel_slug=f"me/rsna-window-bag-paired-seg{segment_index}-v1",
        mode=pc.MODE,
        segment=segment,
        production_funding=funding,
        parallel_arms=True,
        package_manifest_sha256=PKG,
        session_timeout_seconds=sized["session_timeout_seconds"],
    )
    return value, kernels


def paired_phase(name, remote, journal, value, source=None):
    source = source or paired_source(value["segment"], value["package_manifest_sha256"])
    if value["source_sha256"] == "s":
        value = dict(value, source_sha256=digest(source))
    return pc.run_phase(
        name, value, source, remote, Publication, Workflow, lambda n, v: journal.append(n), digest
    )


def test_paired_push_mounts_one_shared_previous_output_and_saves_once_without_retry():
    value, kernels = paired_production(0)
    proof = paired_phase("preflight", FakeRemote(kernels=kernels), [], copy.deepcopy(value))
    assert proof["funding"]["charge_seconds"] == 2 * value["session_timeout_seconds"]
    assert proof["funding"]["GPUs_used_concurrently"] == 2
    assert proof["production"]["previous"] is None
    assert len(pc.wc.kernel_request(value, "x")["kernelDataSources"]) == 20
    value, kernels = paired_production(1)
    request = pc.wc.kernel_request(value, "x")
    previous = value["segment"]["previous_output"]["kernel_slug"]
    assert request["kernelDataSources"] == SHARD_SLUGS + [previous]
    assert len(request["kernelDataSources"]) == plan.SUPPORTED_KERNEL_SOURCES == 21
    assert len(request["datasetDataSources"]) == plan.SUPPORTED_DATASET_SOURCES == 3
    remote, journal = (
        FakeRemote(
            save={"versionNumber": 1, "ref": "/code/" + value["kernel_slug"]}, kernels=kernels
        ),
        [],
    )
    assert paired_phase("push", remote, journal, value)["version"] == 1
    assert journal == ["kernel-create-intent.json", "kernel-create-response.json"]
    assert [c for c in remote.calls if c[0] == "SaveKernel"] == [("SaveKernel", None)]
    assert ("output", previous) in remote.calls and ("output", ENGINEERING_SLUG) in remote.calls
    uncertain, journal = FakeRemote(kernels=kernels), []
    with pytest.raises(RuntimeError):
        paired_phase("push", uncertain, journal, value)
    assert journal == ["kernel-create-intent.json", "kernel-create-uncertain.json"]
    assert [c for c in uncertain.calls if c[0] == "SaveKernel"] == [("SaveKernel", None)]
    value, kernels = paired_production(2)
    assert paired_phase("preflight", FakeRemote(kernels=kernels), [], value)["production"][
        "previous"
    ]


def forge_paired(value, kernels, kind):
    funding, segment = value["production_funding"], value["segment"]
    engineering = kernels[ENGINEERING_SLUG]
    receipt = engineering["outputs"][t4.ENGINEERING_RECEIPT]

    def rebind():
        funding["engineering"]["receipt_sha256"] = t4.canonical(receipt)
        segment["engineering_receipt_sha256"] = t4.canonical(receipt)

    if kind == "sequential_status":
        receipt.update(status=t4.ENGINEERING_STATUS, GPUs_used_concurrently=1)
        rebind()
    elif kind == "one_gpu":
        receipt["GPUs_used_concurrently"] = 1
        rebind()
    elif kind == "devices":
        receipt["concurrency"]["devices"] = {"mean": "0", "query": "0"}
        rebind()
    elif kind == "rss":
        receipt["concurrency"]["peak_combined_rss_bytes"] = t4.RAM_CEILING_BYTES + 1
        rebind()
    elif kind == "cpus":
        receipt["resources_before"]["CPUs"] = 2
        rebind()
    elif kind == "no_concurrency":
        receipt["concurrency"] = None
        rebind()
    elif kind == "receipt_hash":
        funding["engineering"]["receipt_sha256"] = "0" * 64
        segment["engineering_receipt_sha256"] = "0" * 64
    elif kind == "engineering_source":
        funding["engineering"]["source_sha256"] = "1" * 64
    elif kind == "sequential_source":
        engineering["blob"]["source"] = notebook_source("engineering", None, manifest=ENG_PKG)
        funding["engineering"]["source_sha256"] = digest(engineering["blob"]["source"])
    elif kind == "engineering_image":
        engineering["metadata"]["dockerImage"] = "other"
    elif kind == "engineering_package":
        segment["engineering_package_manifest_sha256"] = "2" * 64
    elif kind == "config":
        segment["production_config_sha256"] = {**CONFIGS, "query": "3" * 64}
    elif kind == "measured":
        funding["measured"] = measured(seconds_per_update=1.0)
    elif kind == "segments":
        funding["segments_per_arm"] = [[0, 2810]]
    elif kind == "gpu":
        funding["gpu_seconds_authorized"] = value["session_timeout_seconds"]
    elif kind == "output_cap":
        segment["arm_output_cap_bytes"] += 1
        segment["output_cap_bytes"] += 2
    elif kind == "scratch_cap":
        segment["scratch_cap_bytes"] += 1
    elif kind == "timeout":
        value["session_timeout_seconds"] -= 1
    elif kind == "single_mode":
        value["mode"] = "segment"
    elif kind == "not_parallel":
        value["parallel_arms"] = False
    elif kind == "same_package":
        value["package_manifest_sha256"] = ENG_PKG


@pytest.mark.parametrize(
    "kind",
    [
        "sequential_status",
        "one_gpu",
        "devices",
        "rss",
        "cpus",
        "no_concurrency",
        "receipt_hash",
        "engineering_source",
        "sequential_source",
        "engineering_image",
        "engineering_package",
        "config",
        "measured",
        "segments",
        "gpu",
        "output_cap",
        "scratch_cap",
        "timeout",
        "single_mode",
        "not_parallel",
        "same_package",
    ],
)
def test_paired_funding_and_bindings_are_verified_before_any_intent(kind):
    value, kernels = paired_production(0)
    assert paired_phase("preflight", FakeRemote(kernels=kernels), [], copy.deepcopy(value))
    forge_paired(value, kernels, kind)
    remote, journal = FakeRemote(save={"versionNumber": 1}, kernels=kernels), []
    with pytest.raises(ValueError):
        paired_phase("push", remote, journal, value)
    assert journal == [] and ("SaveKernel", None) not in remote.calls


def test_insufficient_fresh_quota_refuses_dispatch():
    value, kernels = paired_production(1)
    charge = plan.session_charge(value["session_timeout_seconds"])
    used = 108000 - charge + 1
    remote, journal = FakeRemote(dict(ISSUE_QUOTA, timeUsed=f"{used}s"), kernels=kernels), []
    with pytest.raises(ValueError, match="allowance"):
        paired_phase("push", remote, journal, value)
    assert journal == [] and ("SaveKernel", None) not in remote.calls
    reserved = FakeRemote(dict(ISSUE_QUOTA, timeReserved="60000s"), kernels=kernels)
    with pytest.raises(ValueError, match="allowance"):
        paired_phase("preflight", reserved, [], value)


@pytest.mark.parametrize(
    "tamper", ["receipt", "source", "incomplete", "inputs", "checkpoint", "mode", "package"]
)
def test_previous_paired_output_is_read_back_not_trusted(tamper):
    value, kernels = paired_production(1)
    previous = value["segment"]["previous_output"]
    record = kernels[previous["kernel_slug"]]
    if tamper == "receipt":
        record["outputs"][wp.PAIRED_RECEIPT]["arms"]["query"]["step"] = 1
    elif tamper == "source":
        record["blob"]["source"] += "#"
    elif tamper == "incomplete":
        record["status"] = "RUNNING"
    elif tamper == "inputs":
        record["metadata"]["kernelDataSources"] = SHARD_SLUGS[:19]
    elif tamper == "checkpoint":
        previous["arms"]["mean"]["checkpoint_sha256"] = "8" * 64
    elif tamper == "mode":
        record["blob"]["source"] = paired_source(
            record["outputs"][wp.PAIRED_RECEIPT]["contract"], mode="segment"
        )
        previous["source_sha256"] = digest(record["blob"]["source"])
    else:
        record["outputs"][wp.PAIRED_RECEIPT]["package_manifest_sha256"] = "4" * 64
        previous["receipt_sha256"] = t4.canonical(record["outputs"][wp.PAIRED_RECEIPT])
    remote, journal = FakeRemote(save={"versionNumber": 1}, kernels=kernels), []
    with pytest.raises(ValueError):
        paired_phase("push", remote, journal, value)
    assert journal == [] and ("SaveKernel", None) not in remote.calls


def test_status_is_read_only_and_checks_pushed_inputs():
    value, kernels = paired_production(1)
    source = paired_source(value["segment"])
    value["source_sha256"] = digest(source)
    pushed = gpu_kernel(
        source, wp.PAIRED_RECEIPT, {}, [value["segment"]["previous_output"]["kernel_slug"]]
    )
    kernels[value["kernel_slug"]] = pushed
    remote, journal = FakeRemote(kernels=kernels), []
    assert paired_phase("status", remote, journal, value)["status"]["status"] == "COMPLETE"
    pushed["metadata"]["kernelDataSources"] = SHARD_SLUGS
    with pytest.raises(ValueError):
        paired_phase("status", FakeRemote(kernels=kernels), journal, value)
    assert journal == []


# --- Package and notebook -------------------------------------------------------------


def test_paired_package_extends_the_frozen_engineering_package_by_one_file():
    files, engineering = nb.immutable_package()
    paired_files, manifest, linked = pn.paired_package()
    text = json.dumps(engineering, sort_keys=True, indent=2) + "\n"
    assert linked == nb.sha_bytes(text.encode()) == wp.verify_package_link(manifest, linked)
    assert set(paired_files) - set(files) == {wp.PAIRED_SOURCE}
    assert all(paired_files[k] == v for k, v in files.items())
    assert manifest["files"] == engineering["files"] and manifest["science_git"] == t4.SCIENCE_GIT
    with pytest.raises(ValueError):
        wp.verify_package_link(manifest, "0" * 64)
    drifted = copy.deepcopy(manifest)
    drifted["bound_current"]["scripts/kaggle/window_bag_t4.py"] = "0" * 64
    with pytest.raises(ValueError):
        wp.verify_package_link(drifted, linked)
    extra = copy.deepcopy(manifest)
    extra["bound_current"]["scripts/kaggle/other.py"] = "0" * 64
    with pytest.raises(ValueError):
        wp.engineering_manifest_text(extra)


def test_paired_notebook_branch_runs_only_the_paired_segment_on_both_T4s():
    assert "WINDOW_BAG_PAIRED_PROBE" in pn.BRANCH and "WINDOW_BAG_T4_PROBE" not in pn.BRANCH
    assert 'scripts/kaggle/window_bag_paired.py"' in pn.BRANCH
    assert "window_bag_t4.py" not in pn.BRANCH and '"--mode"' not in pn.BRANCH
    assert '"--parallel"' not in pn.BRANCH and '"--receipt"' not in pn.BRANCH
    assert '"0,1" if WINDOW_BAG_PARALLEL is True' in pn.BRANCH
    assert '"--segment"' in pn.BRANCH
    notebook = pn.compose(nb.ENTRY.read_text(), {"WINDOW_BAG_MODE": pc.MODE})
    assert notebook.count(pn.BRANCH) == 1
    segments = [(0, 2), (2, 4)]
    segment = wp.next_paired_segment(None, segments, measured=measured(), **BINDINGS)
    capsule = {
        "archive_name": "a",
        "archive_sha256": "b",
        "arguments": dict.fromkeys(("studies", "artifact-root", "folds", "series"), "x"),
    }
    common = dict(
        bundle_manifest_sha256="x", capsule=capsule, fixture=b"", bag_cpu=b"", overlay=b""
    )
    with pytest.raises(ValueError, match="engineering package"):
        pn.build(segment=segment, timeout_seconds=25920, **common)
    with pytest.raises(ValueError, match="nine-hour"):
        pn.build(segment=segment, timeout_seconds=25921, **common)
    with pytest.raises(ValueError):
        pn.build(segment=dict(segment, arm="mean"), timeout_seconds=25920, **common)


# --- Session-boundary publication / restore ---------------------------------------------


def fake_contract(cap=10**6):
    segment = wp.next_paired_segment(None, [(0, 2), (2, 4)], measured=measured(), **BINDINGS)
    return dict(
        segment, arm_output_cap_bytes=cap, output_cap_bytes=2 * cap, scratch_cap_bytes=4 * cap
    )


def end_session(tmp_path, working, receipt_contract, publication, name="seg0"):
    """Session ends: scratch is gone; only /kaggle/working survives as one kernel input."""
    receipt = {
        "status": wp.PAIRED_SEGMENT_STATUS,
        "contract": receipt_contract,
        "arms": {
            a: {
                "step": receipt_contract["end_step"],
                "checkpoint_sha256": publication["arms"][a]["checkpoint_sha256"],
                "publication": publication["arms"][a],
            }
            for a in t4.ARMS
        },
        "final_epoch10_evaluated": False,
        "intermediate_selection": False,
    }
    wp.write_receipt(working, receipt)
    mount = tmp_path / "input" / f"me-rsna-window-bag-paired-{name}"
    shutil.copytree(working, mount)
    writable(working)
    shutil.rmtree(working)
    read_only(mount)
    return json.loads((mount / wp.PAIRED_RECEIPT).read_text()), mount


def test_paired_publication_is_write_once_capped_and_excludes_cache(tmp_path):
    roots = {}
    for arm in t4.ARMS:
        roots[arm] = tmp_path / "scratch" / arm
        fake_run(roots[arm])
    value = fake_contract()
    with pytest.raises(ValueError, match="cap"):
        wp.publish_paired(roots, tmp_path / "small", fake_contract(cap=10), False)
    assert not (tmp_path / "small" / wp.PAIRED_DIR).exists()
    publication = wp.publish_paired(roots, tmp_path / "working", value, False)
    assert set(publication["arms"]) == set(t4.ARMS)
    assert publication["bytes"] == sum(a["bytes"] for a in publication["arms"].values())
    names = [
        p.relative_to(tmp_path / "working").as_posix()
        for p in (tmp_path / "working").rglob("*")
        if p.is_file()
    ]
    assert names and not any("cache" in n or n.endswith(".npz") for n in names)
    assert {n.split("/")[1] for n in names} == set(t4.ARMS)
    assert all(not os.access(tmp_path / "working" / n, os.W_OK) for n in names)
    with pytest.raises(FileExistsError):
        wp.publish_paired(roots, tmp_path / "working", value, False)


@pytest.mark.parametrize(
    "tamper",
    [
        "member",
        "receipt",
        "arm_checkpoint",
        "publication",
        "chain",
        "absent",
        "duplicate",
        "symlink",
    ],
)
def test_paired_restore_refuses_anything_but_the_bound_mounted_output(tmp_path, tamper):
    roots = {}
    for arm in t4.ARMS:
        roots[arm] = tmp_path / "scratch" / arm
        fake_run(roots[arm])
    value = fake_contract()
    publication = wp.publish_paired(roots, tmp_path / "working", value, False)
    receipt, mount = end_session(tmp_path, tmp_path / "working", value, publication)
    nxt = wp.next_paired_segment(
        receipt, [(0, 2), (2, 4)], measured=measured(), previous_kernel=PREVIOUS_KERNEL, **BINDINGS
    )
    ok = {a: tmp_path / "ok" / a / "run" for a in t4.ARMS}
    restored = wp.restore_previous_paired(tmp_path / "input", nxt, ok)
    assert {a: restored["arms"][a]["checkpoint_sha256"] for a in t4.ARMS} == {
        a: t4.sha(ok[a] / "checkpoint.pt") for a in t4.ARMS
    }
    writable(mount)
    query = mount / wp.PAIRED_DIR / "query" / t4.PUBLICATION_DIR
    if tamper == "member":
        member = query / "run/epoch-checkpoints/epoch-001.pt"
        member.write_bytes(member.read_bytes() + b"x")
    elif tamper == "receipt":
        nxt["previous_output"]["receipt_sha256"] = "0" * 64
    elif tamper == "arm_checkpoint":
        nxt["previous_output"]["arms"]["query"]["checkpoint_sha256"] = "0" * 64
    elif tamper == "publication":
        nxt["previous_output"]["arms"]["mean"]["publication_sha256"] = "0" * 64
    elif tamper == "chain":
        nxt["start_step"] = 3
    elif tamper == "absent":
        shutil.rmtree(mount / wp.PAIRED_DIR / "query")
    elif tamper == "duplicate":
        shutil.copytree(mount, tmp_path / "input/copy")
    else:
        (query / "run/invocation.json").unlink()
        (query / "run/invocation.json").symlink_to(query / "run/run.json")
    with pytest.raises((ValueError, FileNotFoundError)):
        wp.restore_previous_paired(
            tmp_path / "input", nxt, {a: tmp_path / "bad" / a / "run" for a in t4.ARMS}
        )


# --- Actual fresh-process paired sessions through run_paired_segment --------------------


@pytest.fixture
def paired_session(tmp_path, monkeypatch):
    """Patch only the T4/shard admission; the paired runner, Supervisor, trainer processes,
    publication and restore are the real code paths."""
    pytest.importorskip("torch")
    from test_public_triplet_training import build, cohort

    template = tmp_path / "template"
    build(template / "cache", *cohort())
    (template / "series.csv").write_text("SeriesInstanceUID\na\n")
    state = {"hashes": None, "fail": None}
    configs = {arm: {"arm": arm} for arm in t4.ARMS}

    def prepare(args):
        args.package_manifest_sha256 = t4.sha(args.source_manifest)
        empty = SimpleNamespace(study_uid=[])
        return (
            ROOT,
            state["hashes"],
            None,
            None,
            configs,
            {},
            None,
            [],
            [],
            b"",
            {},
            (empty, empty, None, None, None),
        )

    def restore(cache, *_):
        shutil.copytree(template / "cache", cache)
        shutil.copyfile(template / "series.csv", cache.parent / "series.csv")
        return {"members": 0}

    def command(code, config, paths, overlay, cache):
        arm = config.parent.name
        if state["fail"] == arm:
            return [sys.executable, "-c", "import sys, time; time.sleep(1); sys.exit(3)"]
        spec = {
            "root": str(cache.parent),
            "arm": arm,
            "output": "",
            "every": 1,
            "precision": "fp32",
        }
        return [sys.executable, str(CHILD), json.dumps(spec)]

    monkeypatch.setattr(t4, "check_runtime", lambda **_: {"GPUs_visible": 2, "CPUs": 4})
    monkeypatch.setattr(t4, "prepare", prepare)
    monkeypatch.setattr(t4, "production_config", lambda base, spec: base)
    monkeypatch.setattr(t4, "restore", restore)
    monkeypatch.setattr(t4, "trainer_command", command)
    monkeypatch.setattr(
        t4,
        "trainer_env",
        lambda code, device="0": dict(
            os.environ,
            PYTHONPATH=str(ROOT),
            CUDA_VISIBLE_DEVICES=device,
            OMP_NUM_THREADS="2",
            MKL_NUM_THREADS="2",
        ),
    )
    monkeypatch.setattr(plan, "TOTAL_UPDATES", 4)
    manifest = {
        "science_git": t4.SCIENCE_GIT,
        "files": {},
        "bound_current": {n: "0" * 64 for n in (*wp.t4_bound_names(), wp.PAIRED_SOURCE)},
    }
    linked = nb.sha_bytes(wp.engineering_manifest_text(manifest).encode())
    manifest["engineering_package_manifest_sha256"] = linked
    (tmp_path / "source-manifest.json").write_text(json.dumps(manifest))
    bindings = dict(
        BINDINGS,
        engineering_package_manifest_sha256=linked,
        production_config_sha256={a: t4.canonical(configs[a]) for a in t4.ARMS},
    )
    gib = {"checkpoint_bytes": 2**30}

    def run(name, segment, input_root=None):
        session = tmp_path / name
        session.mkdir()
        (session / "segment.json").write_text(json.dumps(segment))
        args = SimpleNamespace(
            source=ROOT,
            source_manifest=tmp_path / "source-manifest.json",
            overlay=None,
            segment=session / "segment.json",
            output=session / "scratch",
            publish_root=session / "working",
            input_root=input_root or tmp_path / "input",
            remaining_seconds=900.0,
            bootstrap_seconds=0.0,
        )
        return wp.run_paired_segment(args), session

    return SimpleNamespace(
        run=run,
        state=state,
        template=template,
        bindings=bindings,
        measured=measured(**gib),
        tmp=tmp_path,
    )


@LINUX
def test_complete_paired_recovery_across_an_actual_published_session_output(paired_session):
    torch = pytest.importorskip("torch")
    from test_competitive_training import _assert_tree_equal

    s = paired_session
    tmp = s.tmp
    full = {}
    for arm in t4.ARMS:
        full[arm] = tmp / "full" / arm
        spec = {
            "root": str(s.template),
            "arm": arm,
            "output": str(full[arm]),
            "every": 1,
            "precision": "fp32",
        }
        assert subprocess_run(spec) == 0
    s.state["hashes"] = torch.load(full["mean"] / "checkpoint.pt", weights_only=True)["binding"][
        "code_sha256"
    ]
    segments = [(0, 2), (2, 4)]
    threads = torch.get_num_threads()
    try:
        first = wp.next_paired_segment(None, segments, measured=s.measured, **s.bindings)
        receipt, one = s.run("session-1", first)
        assert receipt["final_epoch10_evaluated"] is False and receipt["previous_restored"] is None
        assert receipt["GPUs_used_concurrently"] == 2 and receipt["provisioned_GPUs_charged"] == 2
        assert {receipt["arms"][a]["device"] for a in t4.ARMS} == {"0", "1"}
        assert receipt["peak_combined_rss_bytes"] > 0
        assert all(receipt["arms"][a]["prediction_sha256"] is None for a in t4.ARMS)
        # Session 1 ends: its scratch (cache, run dirs) is gone; only its output is mounted.
        writable(one / "scratch")
        shutil.rmtree(one / "scratch")
        mount = tmp / "input/me-rsna-window-bag-paired-seg0"
        shutil.copytree(one / "working", mount)
        read_only(mount)
        mounted = json.loads((mount / wp.PAIRED_RECEIPT).read_text())
        assert t4.canonical(mounted) == t4.canonical(receipt)
        for arm in t4.ARMS:
            published = mount / wp.PAIRED_DIR / arm / t4.PUBLICATION_DIR
            assert t4.sha(published / "run/epoch-checkpoints/epoch-001.pt") == t4.sha(
                full[arm] / "epoch-checkpoints/epoch-001.pt"
            )
            partial = torch.load(published / "run/checkpoint.pt", weights_only=True)
            assert (partial["epoch"], partial["next_update"], partial["step"]) == (1, 0, 2)
        nxt = wp.next_paired_segment(
            mounted, segments, measured=s.measured, previous_kernel=PREVIOUS_KERNEL, **s.bindings
        )
        final, two = s.run("session-2", nxt)
    finally:
        torch.set_num_threads(threads)
    assert final["final_epoch10_evaluated"] is True
    assert final["previous_restored"]["receipt_sha256"] == t4.canonical(receipt)
    for arm in t4.ARMS:
        run = two / "scratch/arms" / arm / "run"
        _assert_tree_equal(
            torch.load(full[arm] / "checkpoint.pt", weights_only=True),
            torch.load(run / "checkpoint.pt", weights_only=True),
        )
        _assert_tree_equal(
            torch.load(full[arm] / "epoch-checkpoints/epoch-002.pt", weights_only=True),
            torch.load(run / "epoch-checkpoints/epoch-002.pt", weights_only=True),
        )
        predictions = (full[arm] / "outer_predictions.csv").read_bytes()
        assert (run / "outer_predictions.csv").read_bytes() == predictions
        assert (
            t4.sha(full[arm] / "outer_predictions.csv") == final["arms"][arm]["prediction_sha256"]
        )
        published = two / "working" / wp.PAIRED_DIR / arm / t4.PUBLICATION_DIR
        assert (published / "run/outer_predictions.csv").read_bytes() == predictions
    assert (
        full["mean"].joinpath("checkpoint.pt").read_bytes()
        != full["query"].joinpath("checkpoint.pt").read_bytes()
    )
    writable(tmp / "input")


def subprocess_run(spec):
    import subprocess

    return subprocess.run(
        [sys.executable, str(CHILD), json.dumps(spec)],
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT)),
        timeout=900,
    ).returncode


@LINUX
def test_one_arm_failure_stops_the_sibling_and_publishes_nothing(paired_session):
    torch = pytest.importorskip("torch")
    s = paired_session
    s.state.update(fail="mean", hashes="unused")
    segment = wp.next_paired_segment(None, [(0, 2), (2, 4)], measured=s.measured, **s.bindings)
    threads = torch.get_num_threads()
    started = time.monotonic()
    try:
        with pytest.raises(ValueError, match="Canonical trainer phase failed"):
            s.run("session-1", segment)
    finally:
        torch.set_num_threads(threads)
    assert time.monotonic() - started < 120
    session = s.tmp / "session-1"
    assert not (session / "working").exists()
    events = {
        a: [
            json.loads(x) for x in (session / f"scratch/{a}-journal.jsonl").read_text().splitlines()
        ]
        for a in t4.ARMS
    }
    assert events["mean"][-1]["event"] == "failed" and events["query"][-1]["event"] == "failed"
    assert [e.get("returncode") for e in events["mean"] if e.get("event") == "exit"] == [3]
    assert events["mean"][0]["device"] == "0" and events["query"][0]["device"] == "1"


@LINUX
def test_wrong_engineering_package_or_config_fails_before_any_trainer(paired_session):
    s = paired_session
    segment = wp.next_paired_segment(None, [(0, 2), (2, 4)], measured=s.measured, **s.bindings)
    with pytest.raises(ValueError, match="engineering package"):
        s.run("a", dict(segment, engineering_package_manifest_sha256="0" * 64))
    with pytest.raises(ValueError, match="configs"):
        s.run("b", dict(segment, production_config_sha256={**CONFIGS}))
    for name in ("a", "b"):
        assert not (s.tmp / name / "scratch").exists()
