"""Kaggle T4 path for the whole-window bag (issue #55): engineering only, never accuracy."""

import copy
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.kaggle import window_bag_coordinator as wc
from scripts.kaggle import window_bag_notebook as nb
from scripts.kaggle import window_bag_plan as plan
from scripts.kaggle import window_bag_t4 as t4

ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "initiative/experiments/competitive-v1/source-window-bag-v1"
# Dated root observation in issue #55 (2026-10-14T03:41Z); never a live quota.
ISSUE_QUOTA = {
    "totalTimeAllowed": "162000s",
    "minimumTimeAllowed": "108000s",
    "timeUsed": "2464.964s",
    "timeReserved": "0s",
}


def test_fixed_schedule_matches_the_original_recipe():
    config = json.loads((RECIPE / "mean-r32-config.json").read_text())
    assert (config["epochs"], config["accumulation_steps"], config["warmup_updates"]) == (
        10,
        8,
        281,
    )
    assert (plan.UPDATES_PER_EPOCH, plan.TOTAL_UPDATES) == (281, 2810)
    segments = plan.segment_boundaries(1000)
    assert segments == [(0, 1000), (1000, 2000), (2000, 2810)]
    with pytest.raises(ValueError):
        plan.segment_boundaries(0)


def test_engineering_quota_charges_every_provisioned_T4():
    admitted = plan.admit_engineering(ISSUE_QUOTA)
    assert admitted["available_seconds"] == pytest.approx(105535.036)
    assert admitted["charge_seconds"] == 2 * 7200
    with pytest.raises(ValueError):
        plan.admit_engineering(ISSUE_QUOTA, 7201)
    with pytest.raises(ValueError):
        plan.admit_engineering(dict(ISSUE_QUOTA, timeReserved="1s"))
    with pytest.raises(ValueError):
        plan.admit_engineering(dict(ISSUE_QUOTA, timeUsed="100000s"))


def test_platform_limits_must_be_observed_not_assumed():
    with pytest.raises(ValueError):
        plan.require_platform_limits({}, 20, 3)
    proof = {
        "observed_at": "t",
        "evidence": "actual API response",
        "max_kernel_sources": 19,
        "max_dataset_sources": 10,
        "max_output_bytes": 2**30,
    }
    with pytest.raises(ValueError):
        plan.require_platform_limits(proof, 20, 3)
    assert plan.require_platform_limits(dict(proof, max_kernel_sources=20), 20, 3)
    with pytest.raises(ValueError):
        plan.require_platform_limits(dict(proof, max_kernel_sources=21), 21, 3, 2**30 + 1)


def measured(**changes):
    return {
        "seconds_per_update": 20.0,
        "startup_seconds": 1800.0,
        "checkpoint_seconds": 30.0,
        "final_evaluation_seconds": 3600.0,
        "checkpoint_bytes": 400 * 2**20,
    } | changes


def test_production_plan_keeps_the_recipe_and_reports_quota_honestly():
    result = plan.production_plan(measured(), ISSUE_QUOTA, 40 * 2**30)
    segments = result["segments_per_arm"]
    assert segments[0][0] == 0 and segments[-1][1] == 2810
    assert all(a[1] == b[0] for a, b in zip(segments, segments[1:]))
    assert result["session_timeout_seconds"] <= 9 * 3600 * 0.8
    assert (
        result["only_final_epoch10_evaluated"] and not result["scientific_hyperparameters_changed"]
    )
    assert (
        result["conservative_charge_seconds"]
        == result["sessions"] * 2 * result["session_timeout_seconds"]
    )
    assert result["quota_fits_now"] is (result["conservative_charge_seconds"] <= 105535.036)
    assert plan.production_plan(measured(), ISSUE_QUOTA, 1)["scratch_fits"] is False
    for bad in ({"seconds_per_update": 0}, {"seconds_per_update": 1e6}):
        with pytest.raises(ValueError):
            plan.production_plan(measured(**bad), ISSUE_QUOTA, 40 * 2**30)
    with pytest.raises(ValueError):
        plan.production_plan({"seconds_per_update": 1.0}, ISSUE_QUOTA, 40 * 2**30)


BINDINGS = {"engineering_receipt_sha256": "e" * 64, "production_config_sha256": "c" * 64}
PREVIOUS_KERNEL = {
    "kernel_slug": "me/rsna-window-bag-mean-seg0-v1",
    "version": 1,
    "source_sha256": "f" * 64,
}


def segment_receipt(contract, checkpoint="a" * 64, publication="b" * 64):
    return {
        "status": t4.SEGMENT_STATUS,
        "contract": contract,
        "checkpoint_sha256": checkpoint,
        "publication": {"manifest_sha256": publication, "checkpoint_sha256": checkpoint},
        "package_manifest_sha256": "m" * 64,
        "final_epoch10_evaluated": False,
        "intermediate_selection": False,
    }


def test_segment_chain_binds_the_previous_kernel_output_and_refuses_drift():
    segments = plan.segment_boundaries(1000)
    first = plan.next_segment(None, "mean", segments, measured=measured(), **BINDINGS)
    assert first["previous_output"] is None and (first["start_step"], first["end_step"]) == (
        0,
        1000,
    )
    # 1000 updates complete epochs 1..3: continuation state plus three immutable epochs.
    assert first["output_cap_bytes"] == 4 * measured()["checkpoint_bytes"] + (
        plan.PUBLICATION_SIDECAR_BYTES
    )
    assert t4.segment_contract(first) == first
    receipt = segment_receipt(first)
    with pytest.raises(ValueError):
        plan.next_segment(receipt, "mean", segments, measured=measured(), **BINDINGS)
    second = plan.next_segment(
        receipt, "mean", segments, measured=measured(), previous_kernel=PREVIOUS_KERNEL, **BINDINGS
    )
    assert second["previous_output"] == {
        **PREVIOUS_KERNEL,
        "receipt_sha256": t4.canonical(receipt),
        "publication_sha256": "b" * 64,
        "checkpoint_sha256": "a" * 64,
    }
    assert t4.segment_contract(second) == second and second["start_step"] == 1000
    for tamper in (
        {"final_epoch10_evaluated": True},
        {"status": "other"},
        {"publication": {"manifest_sha256": "b" * 64, "checkpoint_sha256": "0" * 64}},
    ):
        with pytest.raises(ValueError):
            plan.next_segment(
                receipt | tamper,
                "mean",
                segments,
                measured=measured(),
                previous_kernel=PREVIOUS_KERNEL,
                **BINDINGS,
            )
    with pytest.raises(ValueError):
        plan.next_segment(
            receipt,
            "query",
            segments,
            measured=measured(),
            previous_kernel=PREVIOUS_KERNEL,
            **BINDINGS,
        )
    with pytest.raises(ValueError):
        plan.next_segment(
            receipt,
            "mean",
            segments,
            measured=measured(),
            previous_kernel=PREVIOUS_KERNEL,
            **(BINDINGS | {"production_config_sha256": "d" * 64}),
        )
    with pytest.raises(ValueError):
        t4.segment_contract(dict(second, previous_output=None))
    with pytest.raises(ValueError):
        # A free-floating checkpoint SHA without the kernel/version/output identity.
        t4.segment_contract(dict(second, previous_output={"checkpoint_sha256": "a" * 64}))
    with pytest.raises(ValueError):
        t4.segment_contract(dict(first, end_step=2811))


def bag_receipt():
    configs = {a: json.loads((RECIPE / f"{a}-r32-config.json").read_text()) for a in t4.ARMS}
    return {"configs": configs}


def test_original_mean_query_pair_is_the_only_admitted_recipe():
    configs = t4.bag_configs(bag_receipt())
    spec = dict(configs["mean"]["encoder"], weights_path="/m/w", config_path="/m/c")
    values = t4.engineering_config(configs["mean"], spec)
    assert (values["epochs"], values["diagnostic_study_limit"], values["warmup_updates"]) == (
        2,
        16,
        2,
    )
    production = t4.production_config(configs["query"], spec)
    assert {k for k in production if production[k] != configs["query"][k]} == {
        "encoder",
        "min_free_gib",
    }
    for field, value in (("accumulation_steps", 4), ("epochs", 9), ("warmup_updates", 100)):
        changed = bag_receipt()
        changed["configs"]["mean"][field] = value
        changed["configs"]["query"][field] = value
        with pytest.raises(ValueError):
            t4.bag_configs(changed)
    other = bag_receipt()
    other["configs"]["query"]["seed"] = 1
    with pytest.raises(ValueError):
        t4.bag_configs(other)
    with pytest.raises(ValueError):
        t4.mounted_encoder(configs["mean"], dict(spec, weights_sha256="0" * 64))


def test_pinned_shards_match_committed_original_receipts():
    audits = ROOT / "initiative/audits"
    found = {}
    for path in audits.glob("*/shard-0*-verified.json"):
        record = json.loads(path.read_text())
        found[record["shard_index"]] = record["shard_manifest_sha256"]
    first = json.loads(
        (
            audits / "kaggle-public-source-cohort-progress-20261013-v1/"
            "original-shard000-completed.json"
        ).read_text()
    )
    found.setdefault(0, json.dumps(first).split('shard_manifest_sha256": "')[1][:64])
    assert [found[i] for i in range(20)] == list(t4.SHARD_MANIFEST_SHA256)
    cache = json.loads(
        (audits / "public-source-complete-cache-20261013-v1/verified.json").read_text()
    )
    assert (cache["index_sha256"], cache["exact_member_bytes"]) == (
        t4.INDEX_SHA256,
        t4.EXACT_MEMBER_BYTES,
    )
    assert cache["cohort_manifest_sha256"] == t4.COHORT_MANIFEST_SHA256
    assert sum(t4.SHARD_STUDIES) == 3403


def test_numerical_package_is_the_immutable_science_commit():
    files, manifest = nb.immutable_package()
    stage = json.loads(nb.STAGE_MANIFEST.read_text())["source"]
    assert manifest["files"] == stage["files"] and manifest["science_git"] == t4.SCIENCE_GIT
    for name, value in stage["files"].items():
        assert hashlib.sha256(files[name].encode()).hexdigest() == value
    # Current-tree trainer changes after 971d7dac are never shipped as the package.
    assert (
        files["rsna/competitive/train.py"].encode()
        != (ROOT / "rsna/competitive/train.py").read_bytes()
    )
    assert set(manifest["bound_current"]) == set(nb.BOUND_CURRENT)


def test_notebook_compose_inserts_one_branch_and_respects_the_cap():
    files, manifest = nb.immutable_package()
    import base64
    import gzip

    package = base64.b64encode(gzip.compress(json.dumps(files).encode(), mtime=0)).decode()
    notebook = nb.compose(nb.ENTRY.read_text(), {"WINDOW_BAG_PACKAGE": package})
    assert notebook.count('globals().get("WINDOW_BAG_T4_PROBE")') == 1
    # Package plus the documented ~111 KiB compressed overlay stays below 1 MiB.
    assert len(notebook.encode()) + 120 * 1024 < nb.SOURCE_CAP_BYTES
    with pytest.raises(ValueError):
        nb.compose("no anchor", {})
    with pytest.raises(ValueError):
        nb.private_inputs(b"{}", b"{}", b"{}")


SHARD_SLUGS = [f"me/shard{i:03d}" for i in range(20)]
DATASETS = ["me/model", "me/wheels", "me/capsule"]
IMAGE = "image@sha256:x"
ENGINEERING_SLUG = "me/rsna-window-bag-t4-engineering-v1"


def notebook_source(mode, segment, manifest="m" * 64):
    return (
        f"WINDOW_BAG_MODE = {mode!r}\nWINDOW_BAG_SEGMENT = {segment!r}\n"
        f"WINDOW_BAG_MANIFEST_SHA256 = {manifest!r}\nbody\n"
    )


def gpu_kernel(source, receipt_name, receipt, extra=()):
    return {
        "metadata": {
            "isPrivate": True,
            "enableGpu": True,
            "enableInternet": False,
            "machineShape": "NvidiaTeslaT4",
            "dockerImage": IMAGE,
            "datasetDataSources": list(DATASETS),
            "kernelDataSources": SHARD_SLUGS + list(extra),
            "competitionDataSources": [],
        },
        "blob": {"source": source},
        "status": "COMPLETE",
        "outputs": {receipt_name: receipt},
    }


class FakeRemote:
    def __init__(self, quota=ISSUE_QUOTA, save=None, kernels=None):
        self.calls, self.quota, self.save = [], quota, save
        self.kernels = kernels or {}

    def kernel(self, method, slug, version):
        self.calls.append((method, slug))
        record = self.kernels.get(slug)
        if method == "GetKernelSessionStatus":
            return {"status": "COMPLETE" if record is None else record["status"]}
        if record is not None:
            return {"metadata": record["metadata"], "blob": record["blob"]}
        return {"metadata": {"isPrivate": True, "enableGpu": False, "enableInternet": False}}

    def output_json(self, slug, version, name):
        self.calls.append(("output", slug))
        return copy.deepcopy(self.kernels[slug]["outputs"][name])

    def call(self, service, method, body):
        self.calls.append((method, None))
        if method == "GetAcceleratorQuotaStatistics":
            return {"gpuQuota": self.quota}
        if method == "SaveKernel":
            self.request = body
            if self.save is None:
                raise RuntimeError("transport uncertain")
            return self.save
        raise AssertionError(method)

    def require_absent_kernel(self, slug):
        self.calls.append(("absent", slug))
        return {"method": "GetKernel404"}


class Publication:
    @staticmethod
    def metadata(remote, slug):
        ids = {"me/model": 12333543, "me/capsule": 12567458}
        return {"id": ids[slug], "version": 1, "slug": slug, "private": True}


class Workflow:
    @staticmethod
    def verify_shared_wheelhouse(remote, shared):
        return shared


def contract(**changes):
    value = {
        "kernel_slug": ENGINEERING_SLUG,
        "source_sha256": "s",
        "package_manifest_sha256": "m" * 64,
        "mode": "engineering",
        "segment": None,
        "shard_kernels": [
            {"shard_index": i, "slug": SHARD_SLUGS[i], "version": 1, "shard_manifest_sha256": h}
            for i, h in enumerate(t4.SHARD_MANIFEST_SHA256)
        ],
        "model_dataset": {"slug": "me/model", "id": 12333543},
        "wheelhouse": {"dataset_slug": "me/wheels"},
        "capsule_dataset": {"slug": "me/capsule", "id": 12567458},
        "runtime_image": IMAGE,
        "session_timeout_seconds": 7200,
        "platform_limits": {
            "observed_at": "t",
            "evidence": "actual",
            "max_kernel_sources": 21,
            "max_dataset_sources": 3,
            "max_output_bytes": 20 * 2**30,
        },
        "production_funding": None,
    }
    return value | changes


def digest(text):
    return hashlib.sha256(text.encode()).hexdigest()


def phase(name, remote, journal, value=None):
    value = value or contract()
    source = notebook_source(value["mode"], value["segment"])
    if value["source_sha256"] == "s":
        value = dict(value, source_sha256=digest(source))
    return wc.run_phase(
        name, value, source, remote, Publication, Workflow, lambda n, v: journal.append(n), digest
    )


def engineering_receipt():
    arm = {"complete_actual_partial_final_RNG_optimizer_scaler_and_CSV_recovery": True}
    return {
        "status": t4.ENGINEERING_STATUS,
        "science_git": t4.SCIENCE_GIT,
        "package_manifest_sha256": "m" * 64,
        "fixture_sha256": t4.FIXTURE_SHA256,
        "bag_CPU_sha256": t4.BAG_CPU_SHA256,
        "index_sha256": t4.INDEX_SHA256,
        "shard_manifest_sha256": list(t4.SHARD_MANIFEST_SHA256),
        "arms": {"mean": arm, "query": arm},
        "GPUs_used_concurrently": 1,
        "gold_or_confirmation_used": False,
        "production_fitting": False,
        "clinical_accuracy_evidence": False,
        "production_config_sha256": {"mean": "c" * 64, "query": "d" * 64},
        "measured": measured(),
        "resources_before": {"disk": {"/tmp": {"free": 40 * 2**30}}},
    }


def production(segment_index=0):
    """A complete honest production chain: actual engineering kernel and previous segment."""
    receipt = engineering_receipt()
    eng_source = notebook_source("engineering", None)
    kernels = {ENGINEERING_SLUG: gpu_kernel(eng_source, t4.ENGINEERING_RECEIPT, receipt)}
    sized = plan.production_plan(measured(), ISSUE_QUOTA, 40 * 2**30)
    segments = sized["segments_per_arm"]
    bindings = {
        "engineering_receipt_sha256": t4.canonical(receipt),
        "production_config_sha256": "c" * 64,
    }
    segment = plan.next_segment(None, "mean", segments, measured=measured(), **bindings)
    for index in range(segment_index):
        slug = f"me/rsna-window-bag-mean-seg{index}-v1"
        source = notebook_source("segment", segment)
        previous = segment_receipt(segment, checkpoint=f"{index:x}" * 64)
        previous["contract"] = segment
        extra = (
            []
            if segment["previous_output"] is None
            else [segment["previous_output"]["kernel_slug"]]
        )
        kernels[slug] = gpu_kernel(source, t4.SEGMENT_RECEIPT, previous, extra)
        segment = plan.next_segment(
            previous,
            "mean",
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
        "gpu_seconds_authorized": sized["conservative_charge_seconds"],
        "authorized_at": "root explicit funding record",
    }
    value = contract(
        kernel_slug=f"me/rsna-window-bag-mean-seg{segment_index}-v1",
        mode="segment",
        segment=segment,
        production_funding=funding,
        session_timeout_seconds=sized["session_timeout_seconds"],
    )
    return value, kernels


def test_preflight_is_read_only_and_push_saves_once_without_retry():
    remote, journal = FakeRemote(), []
    proof = phase("preflight", remote, journal)
    assert proof["funding"]["charge_seconds"] == 14400 and journal == []
    assert ("SaveKernel", None) not in remote.calls
    request = wc.kernel_request(contract(), "source")
    assert len(request["kernelDataSources"]) == 20 and request["enableInternet"] is False
    assert request["machineShape"] == "NvidiaTeslaT4" and request["sessionTimeoutSeconds"] == 7200
    good = FakeRemote(save={"versionNumber": 1, "ref": "/code/" + ENGINEERING_SLUG})
    assert phase("push", good, journal)["version"] == 1
    assert journal == ["kernel-create-intent.json", "kernel-create-response.json"]
    assert [c for c in good.calls if c[0] == "SaveKernel"] == [("SaveKernel", None)]
    uncertain, journal = FakeRemote(), []
    with pytest.raises(RuntimeError):
        phase("push", uncertain, journal)
    assert journal == ["kernel-create-intent.json", "kernel-create-uncertain.json"]
    assert [c for c in uncertain.calls if c[0] == "SaveKernel"] == [("SaveKernel", None)]


def test_coordinator_refuses_unfunded_drifted_or_unverified_requests():
    with pytest.raises(ValueError):
        phase("preflight", FakeRemote(dict(ISSUE_QUOTA, timeUsed="99000s")), [])
    for bad in (
        {"platform_limits": {}},
        {"session_timeout_seconds": 7201},
        {"mode": "segment"},
        {"shard_kernels": contract()["shard_kernels"][:19]},
    ):
        with pytest.raises(ValueError):
            phase("preflight", FakeRemote(), [], contract(**bad))
    with pytest.raises(ValueError):
        wc.run_phase(
            "push",
            contract(),
            "other",
            FakeRemote(),
            Publication,
            Workflow,
            lambda *_: None,
            lambda s: s,
        )
    # The source must embed exactly the contract's segment and package identity.
    value = contract()
    other = notebook_source("engineering", None, manifest="0" * 64)
    with pytest.raises(ValueError):
        wc.run_phase(
            "preflight",
            dict(value, source_sha256=digest(other)),
            other,
            FakeRemote(),
            Publication,
            Workflow,
            lambda *_: None,
            digest,
        )


def test_resumed_segment_push_mounts_the_verified_previous_kernel_output():
    value, kernels = production(segment_index=1)
    previous = value["segment"]["previous_output"]
    request = wc.kernel_request(value, "source")
    assert request["kernelDataSources"] == SHARD_SLUGS + [previous["kernel_slug"]]
    remote = FakeRemote(
        save={"versionNumber": 1, "ref": "/code/" + value["kernel_slug"]}, kernels=kernels
    )
    journal = []
    proof = phase("push", remote, journal, value)
    assert proof["production"]["previous"]["kernel_slug"] == previous["kernel_slug"]
    assert remote.request["kernelDataSources"][-1] == previous["kernel_slug"]
    assert ("output", previous["kernel_slug"]) in remote.calls
    assert journal == ["kernel-create-intent.json", "kernel-create-response.json"]
    # A third chained segment also verifies the previous kernel's own previous input.
    value, kernels = production(segment_index=2)
    assert phase("preflight", FakeRemote(kernels=kernels), [], value)["production"]["previous"]


@pytest.mark.parametrize("tamper", ["receipt", "source", "incomplete", "inputs", "publication"])
def test_previous_segment_identity_is_read_back_not_trusted(tamper):
    value, kernels = production(segment_index=1)
    slug = value["segment"]["previous_output"]["kernel_slug"]
    record = kernels[slug]
    if tamper == "receipt":
        record["outputs"][t4.SEGMENT_RECEIPT]["checkpoint_sha256"] = "9" * 64
    elif tamper == "source":
        record["blob"]["source"] += "#"
    elif tamper == "incomplete":
        record["status"] = "RUNNING"
    elif tamper == "inputs":
        record["metadata"]["kernelDataSources"] = SHARD_SLUGS[:19]
    else:
        value["segment"]["previous_output"]["publication_sha256"] = "8" * 64
    remote, journal = FakeRemote(save={"versionNumber": 1}, kernels=kernels), []
    with pytest.raises(ValueError):
        phase("push", remote, journal, value)
    assert journal == [] and ("SaveKernel", None) not in remote.calls


def forge(value, kernels, kind):
    funding = value["production_funding"]
    if kind == "nonempty":
        value["production_funding"] = {"approved": True}
    elif kind == "receipt_hash":
        funding["engineering"]["receipt_sha256"] = "0" * 64
        value["segment"]["engineering_receipt_sha256"] = "0" * 64
    elif kind == "measured":
        funding["measured"] = measured(seconds_per_update=1.0)
    elif kind == "segments":
        funding["segments_per_arm"] = [[0, 2810]]
    elif kind == "engineering_status":
        kernels[ENGINEERING_SLUG]["outputs"][t4.ENGINEERING_RECEIPT]["status"] = "other"
        funding["engineering"]["receipt_sha256"] = t4.canonical(
            kernels[ENGINEERING_SLUG]["outputs"][t4.ENGINEERING_RECEIPT]
        )
        value["segment"]["engineering_receipt_sha256"] = funding["engineering"]["receipt_sha256"]
    elif kind == "engineering_source":
        funding["engineering"]["source_sha256"] = "1" * 64
    elif kind == "engineering_image":
        kernels[ENGINEERING_SLUG]["metadata"]["dockerImage"] = "other"
    elif kind == "config":
        value["segment"]["production_config_sha256"] = "d" * 64
    elif kind == "package":
        value["package_manifest_sha256"] = "2" * 64
    elif kind == "gpu":
        funding["gpu_seconds_authorized"] = 1
    elif kind == "output_cap":
        value["segment"]["output_cap_bytes"] += 1


@pytest.mark.parametrize(
    "kind",
    [
        "nonempty",
        "receipt_hash",
        "measured",
        "segments",
        "engineering_status",
        "engineering_source",
        "engineering_image",
        "config",
        "package",
        "gpu",
        "output_cap",
    ],
)
def test_forged_production_funding_fails_before_any_segment_intent(kind):
    value, kernels = production(segment_index=0)
    assert phase("preflight", FakeRemote(kernels=kernels), [], copy.deepcopy(value))["production"]
    forge(value, kernels, kind)
    remote, journal = FakeRemote(save={"versionNumber": 1}, kernels=kernels), []
    with pytest.raises(ValueError):
        if kind == "package":
            source = notebook_source("segment", value["segment"], manifest="2" * 64)
            wc.run_phase(
                "push",
                dict(value, source_sha256=digest(source)),
                source,
                remote,
                Publication,
                Workflow,
                lambda n, v: journal.append(n),
                digest,
            )
        else:
            phase("push", remote, journal, value)
    assert journal == [] and ("SaveKernel", None) not in remote.calls


# --- Mounted original shards -> exact trainer index, using the real producer -----------


@pytest.fixture
def shards(tmp_path, monkeypatch):
    torch = pytest.importorskip("torch")
    del torch
    import test_kaggle_source_render as ksr_test

    from rsna.competitive import kaggle_source_render as ksr
    from rsna.competitive import public_cache_producer as pc

    source = tmp_path / "source"
    fit = ksr_test.write_study(source, "complete_right")
    held = ksr_test.write_study(source, "absent_slots")
    records = {fit[0]: fit[1], held[0]: held[1]}
    s = dict(
        tmp=tmp_path,
        source=source,
        fit=fit[0],
        held=held[0],
        records=records,
        plane_map=fit[2] | held[2],
        studies={u: pc.study_source_sha256(u, r) for u, r in records.items()},
    )
    renderer = pc.Renderer(ksr_test.POLICY, copy.deepcopy(ksr_test.PIXEL_GROUP))
    cohort = ksr_test.cohort_of(s)
    mount = tmp_path / "input"
    built = ksr_test.build_shards(s, renderer, cohort, mount / "notebooks")
    for directory in built:
        (directory.parent / "cohort-manifest.json").write_bytes(pc._json_bytes(cohort))
    merged = tmp_path / "merged"
    ksr_test.make_intent(s, merged)
    ksr.merge_shards(merged, built, cohort)
    pc.finalize(merged, policy=ksr_test.POLICY)
    index = (merged / "public-ordered-triplet-v1/index.json").read_bytes()
    paths = sorted(built, key=lambda p: p.name)
    monkeypatch.setattr(t4, "SHARD_MANIFEST_SHA256", tuple(built[p] for p in paths))
    monkeypatch.setattr(t4, "SHARD_STUDIES", (1, 1))
    monkeypatch.setattr(t4, "COHORT_MANIFEST_SHA256", ksr.canonical_sha256(cohort))
    monkeypatch.setattr(t4, "INDEX_SHA256", hashlib.sha256(index).hexdigest())
    monkeypatch.setattr(t4, "EXACT_MEMBER_BYTES", 2 * pc.member_bytes())
    monkeypatch.setattr(t4, "FITTING_STUDIES", 1)
    monkeypatch.setattr(t4, "OUTER_STUDIES", 1)
    monkeypatch.setattr(t4, "SCRATCH_RESERVE_BYTES", 0)
    (mount / "rsna-knee-abnormality-detection").mkdir()
    return dict(
        s=s,
        mount=mount,
        merged=merged,
        index=index,
        cohort=cohort,
        policy=ksr_test.POLICY,
        group=ksr_test.PIXEL_GROUP,
    )


def test_mounted_shards_rebuild_the_exact_finalized_index_and_restore_members(shards, tmp_path):
    from rsna.competitive import public_input as pi

    found, cohort = t4.locate_shards(shards["mount"])
    manifests = t4.verify_shards(found, cohort)
    index = t4.reconstruct_index(manifests, cohort, shards["policy"], shards["group"])
    assert index == shards["index"]
    cache = tmp_path / "restored"
    fit = shards["s"]["fit"]
    t4.restore(cache, found, manifests, index, [fit])
    members = sorted(os.listdir(cache / pi.CACHE_DIRECTORY / "members"))
    assert members == [f"{fit}.npz"]
    original = (shards["merged"] / pi.CACHE_DIRECTORY / f"members/{fit}.npz").read_bytes()
    assert (cache / pi.CACHE_DIRECTORY / f"members/{fit}.npz").read_bytes() == original
    loaded, sha = pi.load_public_index(cache, shards["policy"], "synthetic")
    assert sha == t4.INDEX_SHA256 and set(loaded["members"]) == set(shards["s"]["studies"])


def test_shard_or_index_drift_is_refused(shards, tmp_path):
    found, cohort = t4.locate_shards(shards["mount"])
    manifests = t4.verify_shards(found, cohort)
    group = copy.deepcopy(shards["group"])
    group[next(iter(group))] = "tampered"
    with pytest.raises(ValueError):
        t4.reconstruct_index(manifests, cohort, shards["policy"], group)
    with pytest.raises(ValueError):
        t4.reconstruct_index(manifests[:1], cohort, shards["policy"], shards["group"])
    member = next((found[0] / "members").iterdir())
    data = bytearray(member.read_bytes())
    data[-1] ^= 0xFF
    member.write_bytes(bytes(data))
    with pytest.raises(ValueError):
        t4.restore(
            tmp_path / "bad",
            found,
            manifests,
            shards["index"],
            [member.name.removesuffix(".npz.gz")],
        )
    duplicate = shards["mount"] / "copy"
    shutil.copytree(found[0], duplicate / "private/shard/shard")
    with pytest.raises(ValueError):
        t4.locate_shards(shards["mount"])


# --- Complete CPU recovery across an actual process/session boundary -------------------


def session(root, arm, output, stop=None, resume=None, every=0):
    spec = {
        "root": str(root),
        "arm": arm,
        "output": str(output),
        "stop": stop,
        "resume": None if resume is None else str(resume),
        "every": every,
    }
    return subprocess.run(
        [sys.executable, str(ROOT / "tests/window_bag_session_child.py"), json.dumps(spec)],
        cwd=ROOT,
        env=dict(os.environ, PYTHONPATH=str(ROOT)),
        timeout=900,
    ).returncode


def read_only(root):
    for path in [root, *root.rglob("*")]:
        path.chmod(stat.S_IRUSR | (stat.S_IXUSR if path.is_dir() else 0))


def writable(root):
    for path in [root, *root.rglob("*")]:
        path.chmod(stat.S_IRWXU)


def fake_run(root):
    """Minimal segment scratch layout (no trainer) for publication/restore refusals."""
    run = root / "run"
    (run / "epoch-checkpoints").mkdir(parents=True)
    for name in t4.PUBLICATION_SIDECARS:
        (root / name).write_text(name)
    for name in t4.PUBLICATION_RUN_FILES:
        (run / name).write_bytes(name.encode() * 10)
    (run / "epoch-checkpoints/epoch-001.pt").write_bytes(b"checkpoint.pt" * 10)
    (run / "epoch-checkpoints/epoch-001.json").write_text(
        json.dumps(
            {"checkpoint_file": "epoch-001.pt", "checkpoint_sha256": t4.sha(run / "checkpoint.pt")}
        )
    )
    (root / "cache").mkdir()
    (root / "cache/member.npz").write_bytes(b"mri")
    return run


def first_contract(arm="mean", cap=10**6):
    return {
        "arm": arm,
        "segment": 0,
        "start_step": 0,
        "end_step": 2,
        "previous_output": None,
        **BINDINGS,
        "output_cap_bytes": cap,
    }


def publish_and_mount(tmp_path, root, run, contract, final=False):
    working = tmp_path / "working"
    publication = t4.publish_segment(root, run, working, contract["output_cap_bytes"], final)
    receipt = {
        "status": t4.SEGMENT_STATUS,
        "contract": contract,
        "checkpoint_sha256": publication["checkpoint_sha256"],
        "publication": publication,
        "final_epoch10_evaluated": final,
        "intermediate_selection": False,
    }
    (working / t4.SEGMENT_RECEIPT).write_text(json.dumps(receipt, sort_keys=True, indent=2))
    # Session ends: scratch is gone; only the notebook output survives as a kernel input.
    shutil.rmtree(root)
    mount = tmp_path / "input/me-rsna-window-bag-seg0"
    shutil.copytree(working, mount)
    shutil.rmtree(working)
    read_only(mount)
    return publication, json.loads((mount / t4.SEGMENT_RECEIPT).read_text()), mount


def test_publication_is_write_once_capped_and_excludes_cache(tmp_path):
    root = tmp_path / "scratch/mean"
    run = fake_run(root)
    with pytest.raises(ValueError):
        t4.publish_segment(root, run, tmp_path / "small", 10, False)
    publication = t4.publish_segment(root, run, tmp_path / "working", 10**6, False)
    target = tmp_path / "working" / t4.PUBLICATION_DIR
    names = {p.relative_to(target).as_posix() for p in target.rglob("*") if p.is_file()}
    assert names == {
        t4.PUBLICATION_MANIFEST,
        *t4.PUBLICATION_SIDECARS,
        *("run/" + n for n in t4.PUBLICATION_RUN_FILES),
        "run/epoch-checkpoints/epoch-001.pt",
        "run/epoch-checkpoints/epoch-001.json",
    }
    assert not any("cache" in n or n.endswith(".npz") for n in names)
    assert publication["manifest_sha256"] == t4.sha(target / t4.PUBLICATION_MANIFEST)
    assert all(not os.access(target / n, os.W_OK) for n in names)
    with pytest.raises(FileExistsError):
        t4.publish_segment(root, run, tmp_path / "working", 10**6, False)
    (run / "epoch-checkpoints/stray.txt").write_text("x")
    with pytest.raises(ValueError):
        t4.publish_segment(root, run, tmp_path / "other", 10**6, False)


@pytest.mark.parametrize("tamper", ["member", "receipt", "publication", "chain", "absent"])
def test_restore_refuses_anything_but_the_bound_mounted_output(tmp_path, tamper):
    root = tmp_path / "scratch/mean"
    run = fake_run(root)
    contract = first_contract()
    _, receipt, mount = publish_and_mount(tmp_path, root, run, contract)
    nxt = plan.next_segment(
        receipt,
        "mean",
        [(0, 2), (2, 4)],
        measured=measured(),
        previous_kernel=PREVIOUS_KERNEL,
        **BINDINGS,
    )
    nxt = dict(nxt, output_cap_bytes=10**6)
    assert t4.restore_previous(tmp_path / "input", nxt, tmp_path / "ok/run")["files"] == 8
    writable(mount)
    if tamper == "member":
        member = mount / t4.PUBLICATION_DIR / "run/epoch-checkpoints/epoch-001.pt"
        member.write_bytes(member.read_bytes() + b"x")
    elif tamper == "receipt":
        nxt["previous_output"]["receipt_sha256"] = "0" * 64
    elif tamper == "publication":
        nxt["previous_output"]["publication_sha256"] = "0" * 64
    elif tamper == "chain":
        nxt["start_step"] = 3
    else:
        shutil.rmtree(mount)
    with pytest.raises(ValueError):
        t4.restore_previous(tmp_path / "input", nxt, tmp_path / "bad/run")


@pytest.mark.parametrize("arm", ["mean", "query"])
def test_complete_recovery_across_an_actual_published_session_output(tmp_path, arm):
    torch = pytest.importorskip("torch")
    from test_competitive_training import _assert_tree_equal

    assert session(tmp_path, arm, tmp_path / "full", every=1) == 0
    # Session 1 stops at the epoch-1 boundary inside its own scratch.
    root = tmp_path / "session-1-scratch" / arm
    run = root / "run"
    assert session(tmp_path, arm, run, stop=2, every=1) == 75
    for name in t4.PUBLICATION_SIDECARS:
        (root / name).write_text(name)
    contract = first_contract(arm, cap=2**30)
    publication, receipt, mount = publish_and_mount(tmp_path, root, run, contract)
    assert not root.exists() and not (tmp_path / "working").exists()
    published = mount / t4.PUBLICATION_DIR
    assert t4.sha(published / "run/epoch-checkpoints/epoch-001.pt") == t4.sha(
        tmp_path / "full/epoch-checkpoints/epoch-001.pt"
    )
    nxt = plan.next_segment(
        receipt,
        arm,
        [(0, 2), (2, 4)],
        measured=measured(),
        previous_kernel=PREVIOUS_KERNEL,
        **BINDINGS,
    )
    second = tmp_path / "session-2-scratch" / arm / "run"
    restored = t4.restore_previous(tmp_path / "input", nxt, second)
    assert restored["checkpoint_sha256"] == publication["checkpoint_sha256"]
    assert session(tmp_path, arm, second, resume=second / "checkpoint.pt", every=1) == 0
    full = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    resumed = torch.load(second / "checkpoint.pt", weights_only=True)
    _assert_tree_equal(full, resumed)
    assert (tmp_path / "full/outer_predictions.csv").read_bytes() == (
        second / "outer_predictions.csv"
    ).read_bytes()
    _assert_tree_equal(
        torch.load(tmp_path / "full/epoch-checkpoints/epoch-002.pt", weights_only=True),
        torch.load(second / "epoch-checkpoints/epoch-002.pt", weights_only=True),
    )
    partial = torch.load(published / "run/checkpoint.pt", weights_only=True)
    assert (partial["epoch"], partial["next_update"], partial["step"]) == (1, 0, 2)
    writable(mount)


def test_measured_performance_comes_only_from_actual_engineering_timings():
    arm = {
        "update_elapsed_seconds": [10.0, 14.0, 21.0, 25.0],
        "phase_seconds": {"full": 45.0},
        "checkpoint_copy_seconds": 2.0,
        "checkpoint_bytes": 1000,
    }
    result = t4.measure(
        {"mean": arm, "query": dict(arm, checkpoint_bytes=1200)},
        {"bytes": t4.EXACT_MEMBER_BYTES // 100},
        3.0,
        50.0,
        20.0,
    )
    assert result["seconds_per_update"] == 10.0 and result["checkpoint_bytes"] == 1200
    assert result["final_evaluation_seconds"] == 20.0 * t4.OUTER_STUDIES / t4.ENGINEERING_STUDIES
    assert result["startup_seconds"] == pytest.approx(50.0 + 20.0 - 3.0 + 300.0, rel=1e-6)
    with pytest.raises(ValueError):
        t4.measure(
            {"mean": dict(arm, update_elapsed_seconds=[10.0])}, {"bytes": 1}, 3.0, 50.0, 20.0
        )
