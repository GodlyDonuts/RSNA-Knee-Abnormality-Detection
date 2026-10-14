"""Paired two-T4 checkpoint-continuous fixed-10 production (issue #60): code/tests only.

CPU stand-ins (FP32 fixtures) for process supervision and exact recovery; actual FP16/T4
recovery remains a separate required admission. Nothing here calls Kaggle.
"""

import copy
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import pytest
import test_kaggle_window_bag as base

from scripts.kaggle import window_bag_notebook as nb
from scripts.kaggle import window_bag_paired as wp
from scripts.kaggle import window_bag_paired_root as pr
from scripts.kaggle import window_bag_plan as plan
from scripts.kaggle import window_bag_t4 as t4

ROOT = Path(__file__).resolve().parents[1]
CHILD = str(ROOT / "tests/window_bag_session_child.py")
CONFIGS = {"mean": "c" * 64, "query": "d" * 64}
ENGINEERING_PACKAGE = "e" * 64
LINUX = pytest.mark.skipif(
    not Path("/proc/self/stat").exists(),
    reason="Linux /proc process supervision; verified on owning Linux node",
)


def measured(**changes):
    return base.measured(**changes)


# --- Frozen engineering package and notebook source ---------------------------------------


def test_frozen_engineering_wrapper_and_plan_are_byte_unchanged():
    for name, value in wp.FROZEN_ENGINEERING_BOUND.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == value
    assert set(wp.FROZEN_ENGINEERING_BOUND) == set(nb.BOUND_CURRENT)


def test_production_package_is_the_engineering_package_plus_the_runner_only():
    files, manifest = pr.production_package()
    _, engineering = nb.immutable_package()
    expected = hashlib.sha256(wp.manifest_text(engineering).encode()).hexdigest()
    assert wp.engineering_manifest_sha256(manifest) == expected
    assert manifest["files"] == engineering["files"] and manifest["science_git"] == t4.SCIENCE_GIT
    assert set(files) == set(manifest["files"]) | set(manifest["bound_current"])
    assert manifest["bound_current"][wp.PAIRED_MODULE] == t4.sha(ROOT / wp.PAIRED_MODULE)
    for bad in (
        dict(manifest, science_git="0" * 40),
        dict(manifest, bound_current=engineering["bound_current"]),
        dict(manifest, bound_current={**manifest["bound_current"], "x.py": "a" * 64}),
        dict(
            manifest,
            bound_current={
                **manifest["bound_current"],
                "scripts/kaggle/window_bag_t4.py": "0" * 64,
            },
        ),
        dict(manifest, extra=1),
    ):
        with pytest.raises(ValueError):
            wp.engineering_manifest_sha256(bad)


def test_paired_branch_runs_only_the_paired_runner_on_both_devices():
    branch = pr.PAIRED_BRANCH
    assert branch.count("window_bag_paired.py") == 1 and "window_bag_t4.py" not in branch
    assert '"--mode"' not in branch and '"--parallel"' not in branch and '"--receipt"' not in branch
    assert 'globals().get("WINDOW_BAG_PAIRED_PROBE")' in branch
    assert '"0,1" if WINDOW_BAG_PARALLEL is True' in branch and '"--segment"' in branch
    notebook = pr.compose(nb.ENTRY.read_text(), {"WINDOW_BAG_PARALLEL": True})
    assert notebook.count('globals().get("WINDOW_BAG_PAIRED_PROBE")') == 1
    assert 'globals().get("WINDOW_BAG_T4_PROBE")' not in notebook
    with pytest.raises(ValueError):
        pr.compose("no anchor", {})
    segment = first_segment()
    for bad in ({"segment": {}}, {"timeout_seconds": 9 * 3600}, {"capsule": {}}):
        values = (
            dict(
                bundle_manifest_sha256="x",
                capsule={
                    "archive_name": "a",
                    "archive_sha256": "b",
                    "arguments": {k: k for k in ("studies", "artifact-root", "folds", "series")},
                },
                fixture=b"",
                bag_cpu=b"",
                overlay=b"",
                segment=segment,
                timeout_seconds=7200,
            )
            | bad
        )
        with pytest.raises(ValueError):
            pr.build(**values)


# --- Funding recomputed from actual concurrent measurements -------------------------------


def test_paired_plan_charges_both_T4s_once_per_session_from_concurrent_timings():
    sized = wp.paired_plan(measured(), base.ISSUE_QUOTA, 40 * 2**30)
    segments = sized["segments_per_arm"]
    assert segments[0][0] == 0 and segments[-1][1] == plan.TOTAL_UPDATES == 2810
    assert all(a[1] == b[0] for a, b in zip(segments, segments[1:]))
    assert sized["sessions"] == len(segments)
    timeout = sized["session_timeout_seconds"]
    assert timeout <= 9 * 3600 * 0.8
    assert sized["session_charge_seconds"] == 2 * timeout
    assert sized["conservative_charge_seconds"] == len(segments) * 2 * timeout
    assert max(sized["projected_session_seconds"]) <= 9 * 3600 * 0.8
    assert sized["output_cap_bytes_per_segment"] == [
        2 * plan.publication_bytes(measured(), s) for s in segments
    ]
    assert sized["output_caps_fit"] and sized["run_scratch_fits"] and sized["scratch_fits"]
    assert sized["quota_fits_now"] is (
        sized["conservative_charge_seconds"] <= plan.conservative_available(base.ISSUE_QUOTA)
    )
    # Publication/staging time is charged, so never more updates than the projection.
    projected = plan.production_plan(measured(), base.ISSUE_QUOTA, 40 * 2**30, concurrent=True)
    assert segments[0][1] <= projected["segments_per_arm"][0][1]
    slower = wp.paired_plan(measured(seconds_per_update=40.0), base.ISSUE_QUOTA, 40 * 2**30)
    assert slower["sessions"] > sized["sessions"]
    assert wp.paired_plan(measured(), base.ISSUE_QUOTA, 1)["scratch_fits"] is False
    huge = wp.paired_plan(measured(checkpoint_bytes=3 * 2**30), base.ISSUE_QUOTA, 10**12)
    assert not huge["output_caps_fit"] and not huge["run_scratch_fits"]
    poor = dict(base.ISSUE_QUOTA, timeUsed="105000s")
    assert wp.paired_plan(measured(), poor, 40 * 2**30)["next_session_fits_now"] is False
    for bad in (
        {"seconds_per_update": 0},
        {"seconds_per_update": 1e6},
        {"startup_seconds": "1"},
        {"checkpoint_bytes": True},
    ):
        with pytest.raises(ValueError):
            wp.paired_plan(measured(**bad), base.ISSUE_QUOTA, 40 * 2**30)
    with pytest.raises(ValueError):
        wp.paired_plan({"seconds_per_update": 1.0}, base.ISSUE_QUOTA, 40 * 2**30)


# --- One shared previous output per session ------------------------------------------------


BINDINGS = {
    "engineering_receipt_sha256": "a" * 64,
    "engineering_package_manifest_sha256": ENGINEERING_PACKAGE,
    "production_config_sha256": CONFIGS,
}
PREVIOUS_KERNEL = {"kernel_slug": "me/rsna-window-bag-paired-seg0-v1", "version": 1}


def first_segment(segments=((0, 1000), (1000, 2000), (2000, 2810)), **bindings):
    return wp.next_segment(None, list(segments), measured=measured(), **(BINDINGS | bindings))


def paired_receipt(contract, index=0, package="m" * 64):
    arms = {
        arm: {
            "checkpoint_sha256": f"{index}{n}" * 32,
            "publication": {"manifest_sha256": f"{n}{index}" * 32},
            "prediction_sha256": None,
        }
        for n, arm in enumerate(t4.ARMS)
    }
    for record in arms.values():
        record["publication"]["checkpoint_sha256"] = record["checkpoint_sha256"]
    return {
        "status": wp.PAIRED_STATUS,
        "contract": contract,
        "arms": arms,
        "package_manifest_sha256": package,
        "final_epoch10_evaluated": False,
        "intermediate_selection": False,
    }


def test_both_arms_share_one_previous_attachment_and_drift_is_refused():
    segments = [(0, 1000), (1000, 2000), (2000, 2810)]
    first = first_segment(segments)
    assert first["previous_output"] is None and first["output_cap_bytes"] == 2 * (
        4 * measured()["checkpoint_bytes"] + plan.PUBLICATION_SIDECAR_BYTES
    )
    receipt = paired_receipt(first)
    kernel = dict(PREVIOUS_KERNEL, source_sha256="f" * 64)
    second = wp.next_segment(
        receipt, segments, measured=measured(), previous_kernel=kernel, **BINDINGS
    )
    previous = second["previous_output"]
    assert previous["kernel_slug"] == kernel["kernel_slug"] and second["start_step"] == 1000
    assert previous["receipt_sha256"] == t4.canonical(receipt)
    assert (
        set(previous["checkpoint_sha256"])
        == set(previous["publication_sha256"])
        == {
            "mean",
            "query",
        }
    )
    with pytest.raises(ValueError):
        wp.next_segment(receipt, segments, measured=measured(), **BINDINGS)
    tampered = [
        {"status": t4.SEGMENT_STATUS},
        {"final_epoch10_evaluated": True},
        {"intermediate_selection": True},
        {"arms": {"mean": receipt["arms"]["mean"]}},
        {"contract": dict(first, end_step=999)},
    ]
    arm = copy.deepcopy(receipt["arms"])
    arm["query"]["publication"]["checkpoint_sha256"] = "0" * 64
    tampered.append({"arms": arm})
    for change in tampered:
        with pytest.raises((ValueError, KeyError)):
            wp.next_segment(
                receipt | change,
                segments,
                measured=measured(),
                previous_kernel=kernel,
                **BINDINGS,
            )
    for bindings in (
        {"production_config_sha256": {"mean": "c" * 64, "query": "0" * 64}},
        {"engineering_package_manifest_sha256": "0" * 64},
        {"engineering_receipt_sha256": "0" * 64},
    ):
        with pytest.raises(ValueError):
            wp.next_segment(
                receipt,
                segments,
                measured=measured(),
                previous_kernel=kernel,
                **(BINDINGS | bindings),
            )
    for bad in (
        dict(second, previous_output=None),
        dict(second, previous_output=dict(previous, checkpoint_sha256="a" * 64)),
        dict(second, previous_output=dict(previous, version=2)),
        dict(first, end_step=2811),
        dict(first, output_cap_bytes=17 * 2**30),
        dict(first, production_config_sha256={"mean": "c" * 64, "query": "c" * 64}),
        dict(first, arm="mean"),
    ):
        with pytest.raises(ValueError):
            wp.validate_segment(bad)


# --- Root coordinator: actual concurrent engineering, funding, previous kernel ------------


PAIRED_SLUG = "me/rsna-window-bag-paired-seg{}-v1"


def paired_source(segment, manifest="m" * 64):
    return (
        f"WINDOW_BAG_PAIRED_PROBE = True\nWINDOW_BAG_MODE = {wp.PAIRED_MODE!r}\n"
        f"WINDOW_BAG_SEGMENT = {segment!r}\nWINDOW_BAG_MANIFEST_SHA256 = {manifest!r}\n"
        "WINDOW_BAG_PARALLEL = True\nbody\n"
    )


def concurrent_engineering_receipt():
    receipt = base.engineering_receipt()
    receipt |= {
        "status": t4.ENGINEERING_PARALLEL_STATUS,
        "GPUs_used_concurrently": 2,
        "package_manifest_sha256": ENGINEERING_PACKAGE,
        "production_config_sha256": dict(CONFIGS),
        "concurrency": {
            "devices": dict(t4.PARALLEL_DEVICES),
            "threads_per_process": 2,
            "shared_read_only_cache_sha256": "5" * 64,
            "peak_combined_rss_bytes": 20 * 2**30,
            "peak_combined_output_bytes": 2**30,
            "journal_sha256": {"mean": "6" * 64, "query": "7" * 64},
        },
    }
    receipt["resources_before"] |= {"GPUs_visible": 2, "CPUs": 4}
    return receipt


def paired_production(segment_index=0, receipt=None):
    receipt = receipt or concurrent_engineering_receipt()
    eng_source = base.notebook_source("engineering", None, parallel=True)
    kernels = {base.ENGINEERING_SLUG: base.gpu_kernel(eng_source, t4.ENGINEERING_RECEIPT, receipt)}
    sized = wp.paired_plan(receipt["measured"], base.ISSUE_QUOTA, 40 * 2**30)
    segments = sized["segments_per_arm"]
    bindings = BINDINGS | {"engineering_receipt_sha256": t4.canonical(receipt)}
    segment = wp.next_segment(None, segments, measured=receipt["measured"], **bindings)
    for index in range(segment_index):
        source = paired_source(segment)
        previous = paired_receipt(segment, index)
        extra = (
            []
            if segment["previous_output"] is None
            else [segment["previous_output"]["kernel_slug"]]
        )
        kernels[PAIRED_SLUG.format(index)] = base.gpu_kernel(
            source, wp.PAIRED_RECEIPT, previous, extra
        )
        segment = wp.next_segment(
            previous,
            segments,
            measured=receipt["measured"],
            previous_kernel={
                "kernel_slug": PAIRED_SLUG.format(index),
                "version": 1,
                "source_sha256": base.digest(source),
            },
            **bindings,
        )
    funding = {
        "engineering": {
            "kernel_slug": base.ENGINEERING_SLUG,
            "version": 1,
            "source_sha256": base.digest(eng_source),
            "receipt_sha256": t4.canonical(receipt),
        },
        "measured": copy.deepcopy(receipt["measured"]),
        "segments_per_arm": [list(x) for x in segments],
        "gpu_seconds_authorized": sized["session_charge_seconds"],
        "authorized_at": "root explicit funding record",
    }
    value = base.contract(
        kernel_slug=PAIRED_SLUG.format(segment_index),
        mode=wp.PAIRED_MODE,
        segment=segment,
        production_funding=funding,
        session_timeout_seconds=sized["session_timeout_seconds"],
        parallel_arms=True,
    )
    return value, kernels


def paired_phase(name, remote, journal, value, source=None):
    source = source or paired_source(value["segment"])
    value = dict(value, source_sha256=base.digest(source))
    return pr.run_phase(
        name,
        value,
        source,
        remote,
        base.Publication,
        base.Workflow,
        lambda n, v: journal.append(n),
        base.digest,
    )


def test_paired_preflight_is_read_only_and_push_saves_once():
    value, kernels = paired_production(0)
    remote, journal = base.FakeRemote(kernels=kernels), []
    proof = paired_phase("preflight", remote, journal, value)
    timeout = value["session_timeout_seconds"]
    assert proof["funding"]["charge_seconds"] == 2 * timeout
    assert proof["funding"]["GPUs_used_concurrently"] == 2
    assert proof["production"]["previous"] is None and journal == []
    assert ("SaveKernel", None) not in remote.calls
    request = pr.wc.kernel_request(value, "x")
    assert len(request["kernelDataSources"]) == 20 and request["machineShape"] == "NvidiaTeslaT4"
    good = base.FakeRemote(
        save={"versionNumber": 1, "ref": "/code/" + value["kernel_slug"]}, kernels=kernels
    )
    assert paired_phase("push", good, journal, value)["version"] == 1
    assert journal == ["kernel-create-intent.json", "kernel-create-response.json"]
    assert [c for c in good.calls if c[0] == "SaveKernel"] == [("SaveKernel", None)]
    uncertain, journal = base.FakeRemote(kernels=kernels), []
    with pytest.raises(RuntimeError):
        paired_phase("push", uncertain, journal, value)
    assert journal == ["kernel-create-intent.json", "kernel-create-uncertain.json"]


@pytest.mark.parametrize("index", [1, 2])
def test_chained_paired_push_mounts_exactly_one_previous_kernel(index):
    value, kernels = paired_production(index)
    previous = value["segment"]["previous_output"]
    request = pr.wc.kernel_request(value, "x")
    assert request["kernelDataSources"] == base.SHARD_SLUGS + [previous["kernel_slug"]]
    assert len(request["kernelDataSources"]) == plan.SUPPORTED_KERNEL_SOURCES
    assert len(request["datasetDataSources"]) == plan.SUPPORTED_DATASET_SOURCES
    remote = base.FakeRemote(
        save={"versionNumber": 1, "ref": "/code/" + value["kernel_slug"]}, kernels=kernels
    )
    proof = paired_phase("push", remote, [], value)
    assert proof["production"]["previous"]["kernel_slug"] == previous["kernel_slug"]
    assert ("output", previous["kernel_slug"]) in remote.calls


def forge(value, kernels, kind):
    funding, segment = value["production_funding"], value["segment"]
    engineering = kernels[base.ENGINEERING_SLUG]["outputs"][t4.ENGINEERING_RECEIPT]

    def rebind():
        funding["engineering"]["receipt_sha256"] = t4.canonical(engineering)
        segment["engineering_receipt_sha256"] = funding["engineering"]["receipt_sha256"]

    if kind == "sequential_engineering":
        engineering |= {"status": t4.ENGINEERING_STATUS, "GPUs_used_concurrently": 1}
        rebind()
    elif kind == "one_GPU":
        engineering["GPUs_used_concurrently"] = 1
        rebind()
    elif kind == "RSS":
        engineering["concurrency"]["peak_combined_rss_bytes"] = 30 * 2**30
        rebind()
    elif kind == "CPUs":
        engineering["resources_before"]["CPUs"] = 3
        rebind()
    elif kind == "no_concurrency":
        engineering["concurrency"] = None
        rebind()
    elif kind == "engineering_package":
        segment["engineering_package_manifest_sha256"] = "0" * 64
    elif kind == "receipt_hash":
        funding["engineering"]["receipt_sha256"] = "0" * 64
        segment["engineering_receipt_sha256"] = "0" * 64
    elif kind == "engineering_source":
        funding["engineering"]["source_sha256"] = "1" * 64
    elif kind == "measured":
        funding["measured"] = measured(seconds_per_update=1.0)
    elif kind == "sequential_projection":
        sized = plan.production_plan(funding["measured"], base.ISSUE_QUOTA, 40 * 2**30, True)
        funding["segments_per_arm"] = [list(s) for s in sized["segments_per_arm"]]
    elif kind == "gpu":
        funding["gpu_seconds_authorized"] = value["session_timeout_seconds"]
    elif kind == "timeout":
        value["session_timeout_seconds"] -= 1
    elif kind == "output_cap":
        segment["output_cap_bytes"] += 1
    elif kind == "config":
        segment["production_config_sha256"] = {"mean": "c" * 64, "query": "9" * 64}
    elif kind == "datasets":
        value["model_dataset"] = dict(value["model_dataset"], id=1)
    elif kind == "single_arm":
        value["mode"] = "segment"


FORGERIES = [
    "sequential_engineering",
    "one_GPU",
    "RSS",
    "CPUs",
    "no_concurrency",
    "engineering_package",
    "receipt_hash",
    "engineering_source",
    "measured",
    "sequential_projection",
    "gpu",
    "timeout",
    "output_cap",
    "config",
    "datasets",
    "single_arm",
]


@pytest.mark.parametrize("kind", FORGERIES)
def test_forged_binding_or_funding_fails_before_any_paired_intent(kind):
    value, kernels = paired_production(0)
    assert paired_phase("preflight", base.FakeRemote(kernels=kernels), [], copy.deepcopy(value))
    forge(value, kernels, kind)
    remote, journal = base.FakeRemote(save={"versionNumber": 1}, kernels=kernels), []
    with pytest.raises((ValueError, KeyError, TypeError)):
        paired_phase("push", remote, journal, value)
    assert journal == [] and ("SaveKernel", None) not in remote.calls


@pytest.mark.parametrize(
    "quota",
    [
        dict(base.ISSUE_QUOTA, timeUsed="100000s"),
        dict(base.ISSUE_QUOTA, timeReserved="60s"),
        dict(base.ISSUE_QUOTA, minimumTimeAllowed="20000s"),
    ],
)
def test_fresh_allowance_must_fund_both_GPUs_for_the_session(quota):
    value, kernels = paired_production(1)
    remote, journal = base.FakeRemote(quota=quota, save={"versionNumber": 1}, kernels=kernels), []
    with pytest.raises(ValueError):
        paired_phase("push", remote, journal, value)
    assert journal == [] and ("SaveKernel", None) not in remote.calls


@pytest.mark.parametrize(
    "tamper",
    ["receipt", "source", "incomplete", "inputs", "arm_checkpoint", "package", "source_kind"],
)
def test_previous_paired_identity_is_read_back_not_trusted(tamper):
    value, kernels = paired_production(1)
    previous = value["segment"]["previous_output"]
    record = kernels[previous["kernel_slug"]]
    source = None
    if tamper == "receipt":
        record["outputs"][wp.PAIRED_RECEIPT]["arms"]["query"]["checkpoint_sha256"] = "9" * 64
    elif tamper == "source":
        record["blob"]["source"] += "#"
    elif tamper == "incomplete":
        record["status"] = "RUNNING"
    elif tamper == "inputs":
        record["metadata"]["kernelDataSources"] = base.SHARD_SLUGS + ["me/other"]
    elif tamper == "arm_checkpoint":
        previous["checkpoint_sha256"]["mean"] = "8" * 64
    elif tamper == "package":
        record["outputs"][wp.PAIRED_RECEIPT]["package_manifest_sha256"] = "2" * 64
        previous["receipt_sha256"] = t4.canonical(record["outputs"][wp.PAIRED_RECEIPT])
    else:
        source = base.notebook_source("segment", value["segment"], parallel=True)
    remote, journal = base.FakeRemote(save={"versionNumber": 1}, kernels=kernels), []
    with pytest.raises(ValueError):
        paired_phase("push", remote, journal, value, source)
    assert journal == [] and ("SaveKernel", None) not in remote.calls


# --- Write-once paired publication and restore ----------------------------------------------


def fake_arms(scratch):
    arms = {}
    for arm in t4.ARMS:
        root = scratch / arm
        arms[arm] = (root, base.fake_run(root))
        shutil.rmtree(root / "cache")
    return arms


def mount(tmp_path, working, name="input/me-paired-seg0"):
    target = tmp_path / name
    shutil.copytree(working, target)
    shutil.rmtree(working)
    base.read_only(target)
    return target


def test_paired_publication_is_write_once_combined_capped_and_atomic(tmp_path):
    arms = fake_arms(tmp_path / "scratch")
    one = sum(p.stat().st_size for p in t4.publication_files(*arms["mean"], False).values())
    with pytest.raises(ValueError):
        wp.publish(arms, tmp_path / "small", 2 * one - 1, False)
    assert not (tmp_path / "small" / wp.PAIRED_DIR).exists()
    with pytest.raises(ValueError):
        wp.publish({"mean": arms["mean"]}, tmp_path / "one", 10**6, False)
    publications = wp.publish(arms, tmp_path / "working", 10**6, False)
    for arm in t4.ARMS:
        target = tmp_path / "working" / wp.PAIRED_DIR / arm / t4.PUBLICATION_DIR
        assert publications[arm]["manifest_sha256"] == t4.sha(target / t4.PUBLICATION_MANIFEST)
        assert not any("cache" in p.parts for p in target.rglob("*"))
    with pytest.raises(FileExistsError):
        wp.publish(arms, tmp_path / "working", 10**6, False)


def publish_fake(tmp_path, contract):
    arms = fake_arms(tmp_path / "scratch")
    working = tmp_path / "working"
    publications = wp.publish(arms, working, 10**6, False)
    receipt = {
        "status": wp.PAIRED_STATUS,
        "contract": contract,
        "arms": {
            arm: {
                "checkpoint_sha256": publications[arm]["checkpoint_sha256"],
                "publication": publications[arm],
                "prediction_sha256": None,
            }
            for arm in t4.ARMS
        },
        "final_epoch10_evaluated": False,
        "intermediate_selection": False,
    }
    wp.write_receipt(working, receipt)
    shutil.rmtree(tmp_path / "scratch")
    return receipt, mount(tmp_path, working)


def tree_digest(root):
    return {str(p.relative_to(root)): t4.sha(p) for p in sorted(root.rglob("*")) if p.is_file()}


@pytest.mark.parametrize(
    "tamper", [None, "member", "receipt", "arm", "chain", "absent", "duplicate"]
)
def test_restore_stages_both_arms_only_from_the_bound_shared_output(tmp_path, tamper):
    segments = [(0, 2), (2, 4)]
    first = dict(first_segment(segments), output_cap_bytes=10**6)
    receipt, mounted = publish_fake(tmp_path, first)
    kernel = dict(PREVIOUS_KERNEL, source_sha256="f" * 64)
    nxt = wp.next_segment(
        receipt, segments, measured=measured(), previous_kernel=kernel, **BINDINGS
    )
    before = tree_digest(mounted)
    runs = {arm: tmp_path / f"stage/{arm}/run" for arm in t4.ARMS}
    if tamper is None:
        staged = wp.restore_previous(tmp_path / "input", nxt, runs)
        for arm in t4.ARMS:
            assert staged[arm]["checkpoint_sha256"] == t4.sha(runs[arm] / "checkpoint.pt")
        assert tree_digest(mounted) == before
        return
    base.writable(mounted)
    if tamper == "member":
        member = mounted / wp.PAIRED_DIR / "query" / t4.PUBLICATION_DIR / "run/checkpoint.pt"
        member.write_bytes(member.read_bytes() + b"x")
    elif tamper == "receipt":
        nxt["previous_output"]["receipt_sha256"] = "0" * 64
    elif tamper == "arm":
        nxt["previous_output"]["publication_sha256"]["query"] = "0" * 64
    elif tamper == "chain":
        nxt["start_step"] = 3
    elif tamper == "absent":
        shutil.rmtree(mounted)
    else:
        shutil.copytree(mounted, tmp_path / "input/me-paired-copy")
    with pytest.raises(ValueError):
        wp.restore_previous(tmp_path / "input", nxt, runs)


# --- Fresh-process paired sessions on the canonical trainer (CPU FP32 stand-in) -----------


def child(tmp_path, arm, output, **extra):
    spec = {
        "root": str(tmp_path),
        "arm": arm,
        "output": str(output),
        "stop": None,
        "resume": None,
        "every": 1,
        # CPU process fixture; actual FP16/T4 recovery is a separate required admission.
        "precision": "fp32",
    } | extra
    return [sys.executable, CHILD, json.dumps(spec)]


def child_env(arm):
    return dict(
        os.environ,
        PYTHONPATH=str(ROOT),
        CUDA_VISIBLE_DEVICES=t4.PARALLEL_DEVICES[arm],
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
    )


def supervisor(root, **changes):
    root.mkdir(parents=True, exist_ok=True)
    values = dict(poll=0.2, scratch_reserve=0, output_cap=wp.SCRATCH_CAP_BYTES) | changes
    return t4.Supervisor(
        root, time.monotonic() + 1500, {a: root / f"{a}-journal.jsonl" for a in t4.ARMS}, **values
    )


def session_arms(tmp_path, scratch, contract, resume):
    arms = {}
    for arm in t4.ARMS:
        root = scratch / arm
        run = root / "run"
        run.mkdir(parents=True, exist_ok=True)
        (root / "config.json").write_text(arm)
        stop = None if contract["end_step"] == 4 else contract["end_step"] - contract["start_step"]
        arms[arm] = {
            "root": root,
            "run_dir": run,
            "command": child(
                tmp_path, arm, run, stop=stop, resume=str(run / "checkpoint.pt") if resume else None
            ),
            "env": child_env(arm),
        }
    return arms


@LINUX
def test_paired_fresh_process_sessions_publish_restore_and_finish_exactly(tmp_path):
    torch = pytest.importorskip("torch")
    from test_competitive_training import _assert_tree_equal
    from test_public_triplet_training import build, cohort

    build(tmp_path / "cache", *cohort())
    (tmp_path / "series.csv").write_text("SeriesInstanceUID\na\n")
    frozen = t4.freeze_cache(tmp_path / "cache")
    # Uninterrupted references for both arms, run concurrently in fresh processes.
    sup = supervisor(tmp_path / "reference")
    jobs = {}
    for arm in t4.ARMS:
        (tmp_path / "reference" / arm).mkdir()
        jobs[arm] = lambda arm=arm: sup.run(
            arm,
            child(tmp_path, arm, tmp_path / "reference" / arm / "full"),
            tmp_path / "reference" / arm,
            "full",
            0,
            child_env(arm),
            {},
        )
    t4.run_concurrent(jobs, sup)
    full = {
        arm: torch.load(tmp_path / f"reference/{arm}/full/checkpoint.pt", weights_only=True)
        for arm in t4.ARMS
    }
    hashes = full["mean"]["binding"]["code_sha256"]
    assert full["query"]["binding"]["code_sha256"] == hashes
    total = full["mean"]["step"]
    assert total == 4

    # Session 1: both arms stop at the shared epoch-1 boundary and publish write-once.
    segments = [(0, 2), (2, 4)]
    first = dict(first_segment(segments), output_cap_bytes=2**31)
    scratch = tmp_path / "session-1-scratch"
    arms = session_arms(tmp_path, scratch, first, resume=False)
    working = tmp_path / "working"
    results = wp.continue_arms(
        first, arms, supervisor(scratch), working, hashes, tmp_path / "cache", total_updates=total
    )
    receipt = {
        "status": wp.PAIRED_STATUS,
        "contract": first,
        "arms": results,
        "final_epoch10_evaluated": False,
        "intermediate_selection": False,
    }
    wp.write_receipt(working, receipt)
    shutil.rmtree(scratch)
    mounted = mount(tmp_path, working)
    before = tree_digest(mounted)
    for arm in t4.ARMS:
        published = mounted / wp.PAIRED_DIR / arm / t4.PUBLICATION_DIR / "run"
        partial = torch.load(published / "checkpoint.pt", weights_only=True)
        assert (partial["epoch"], partial["next_update"], partial["step"]) == (1, 0, 2)
        _assert_tree_equal(
            torch.load(published / "epoch-checkpoints/epoch-001.pt", weights_only=True),
            torch.load(
                tmp_path / f"reference/{arm}/full/epoch-checkpoints/epoch-001.pt",
                weights_only=True,
            ),
        )
        assert not (published / "outer_predictions.csv").exists()

    # Session 2 (fresh processes, fresh scratch): restore both from the ONE mounted output.
    kernel = dict(PREVIOUS_KERNEL, source_sha256="f" * 64)
    second = wp.next_segment(
        json.loads((mounted / wp.PAIRED_RECEIPT).read_text()),
        segments,
        measured=measured(),
        previous_kernel=kernel,
        **BINDINGS,
    )
    second = dict(second, output_cap_bytes=2**31)
    scratch = tmp_path / "session-2-scratch"
    staged = wp.restore_previous(
        tmp_path / "input", second, {arm: scratch / arm / "run" for arm in t4.ARMS}
    )
    assert {a: staged[a]["checkpoint_sha256"] for a in t4.ARMS} == {
        a: results[a]["checkpoint_sha256"] for a in t4.ARMS
    }
    arms = session_arms(tmp_path, scratch, second, resume=True)
    final = wp.continue_arms(
        second,
        arms,
        supervisor(scratch),
        tmp_path / "working-2",
        hashes,
        tmp_path / "cache",
        total_updates=total,
    )
    assert tree_digest(mounted) == before and t4.cache_digest(tmp_path / "cache") == frozen
    for arm in t4.ARMS:
        run = scratch / arm / "run"
        reference = tmp_path / f"reference/{arm}/full"
        # Complete model/base/AdamW/LR/counters/RNG (and scaler when present) equality.
        _assert_tree_equal(full[arm], torch.load(run / "checkpoint.pt", weights_only=True))
        assert (run / "outer_predictions.csv").read_bytes() == (
            reference / "outer_predictions.csv"
        ).read_bytes()
        assert final[arm]["prediction_sha256"] == t4.sha(reference / "outer_predictions.csv")
        _assert_tree_equal(
            torch.load(reference / "epoch-checkpoints/epoch-002.pt", weights_only=True),
            torch.load(run / "epoch-checkpoints/epoch-002.pt", weights_only=True),
        )
        published = tmp_path / "working-2" / wp.PAIRED_DIR / arm / t4.PUBLICATION_DIR
        assert (published / "run/outer_predictions.csv").is_file()
        assert (published / "run/epoch-checkpoints/epoch-002.pt").is_file()
    base.writable(mounted)
    for path in [tmp_path / "cache", *(tmp_path / "cache").rglob("*")]:
        path.chmod(0o700)


@LINUX
@pytest.mark.parametrize("failure", ["sibling", "boundary", "cap"])
def test_paired_failure_reaps_sibling_publishes_nothing_and_keeps_originals(tmp_path, failure):
    segments = [(0, 2), (2, 4)]
    first = dict(first_segment(segments), output_cap_bytes=10**6)
    receipt, mounted = publish_fake(tmp_path, first)
    before = tree_digest(mounted)
    kernel = dict(PREVIOUS_KERNEL, source_sha256="f" * 64)
    nxt = wp.next_segment(
        receipt, segments, measured=measured(), previous_kernel=kernel, **BINDINGS
    )
    scratch = tmp_path / "session-2-scratch"
    runs = {arm: scratch / arm / "run" for arm in t4.ARMS}
    wp.restore_previous(tmp_path / "input", nxt, runs)
    hang = "import time; time.sleep(300)"
    codes = {
        "sibling": ("import sys, time; time.sleep(1); sys.exit(3)", hang),
        "boundary": ("import sys; sys.exit(75)", "import sys; sys.exit(75)"),
        "cap": (
            f"import time; open({str(runs['mean'] / 'big')!r}, 'wb').write(b'0' * 2**21);"
            " time.sleep(300)",
            hang,
        ),
    }[failure]
    arms = {
        arm: {
            "root": scratch / arm,
            "run_dir": runs[arm],
            "command": [sys.executable, "-c", code],
            "env": child_env(arm),
        }
        for arm, code in zip(t4.ARMS, codes)
    }
    sup = supervisor(scratch, output_cap=2**20 if failure == "cap" else wp.SCRATCH_CAP_BYTES)
    started = time.monotonic()
    with pytest.raises(Exception):  # noqa: B017 - ValueError, or an unloadable staged state
        # "boundary": both exit as interrupted but leave no state at the scheduled step.
        wp.continue_arms(nxt, arms, sup, tmp_path / "working-2", {}, total_updates=4)
    assert time.monotonic() - started < 120 and not sup.live
    assert not (tmp_path / "working-2" / wp.PAIRED_DIR).exists()
    assert tree_digest(mounted) == before
    for arm in t4.ARMS:
        assert (
            t4.sha(runs[arm] / "checkpoint.pt") == nxt["previous_output"]["checkpoint_sha256"][arm]
        )
    base.writable(mounted)
