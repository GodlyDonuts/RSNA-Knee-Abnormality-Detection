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
ISSUE_QUOTA = {"totalTimeAllowed": "162000s", "minimumTimeAllowed": "108000s",
               "timeUsed": "2464.964s", "timeReserved": "0s"}


def test_fixed_schedule_matches_the_original_recipe():
    config = json.loads((RECIPE / "mean-r32-config.json").read_text())
    assert (config["epochs"], config["accumulation_steps"], config["warmup_updates"]) == (10, 8, 281)
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
    proof = {"observed_at": "t", "evidence": "actual API response", "max_kernel_sources": 19,
             "max_dataset_sources": 10}
    with pytest.raises(ValueError):
        plan.require_platform_limits(proof, 20, 3)
    assert plan.require_platform_limits(dict(proof, max_kernel_sources=20), 20, 3)


def measured(**changes):
    return {"seconds_per_update": 20.0, "startup_seconds": 1800.0, "checkpoint_seconds": 30.0,
            "final_evaluation_seconds": 3600.0, "checkpoint_bytes": 400 * 2**20} | changes


def test_production_plan_keeps_the_recipe_and_reports_quota_honestly():
    result = plan.production_plan(measured(), ISSUE_QUOTA, 40 * 2**30)
    segments = result["segments_per_arm"]
    assert segments[0][0] == 0 and segments[-1][1] == 2810
    assert all(a[1] == b[0] for a, b in zip(segments, segments[1:]))
    assert result["session_timeout_seconds"] <= 9 * 3600 * 0.8
    assert result["only_final_epoch10_evaluated"] and not result["scientific_hyperparameters_changed"]
    assert result["conservative_charge_seconds"] == result["sessions"] * 2 * result[
        "session_timeout_seconds"]
    assert result["quota_fits_now"] is (result["conservative_charge_seconds"] <= 105535.036)
    assert plan.production_plan(measured(), ISSUE_QUOTA, 1)["scratch_fits"] is False
    for bad in ({"seconds_per_update": 0}, {"seconds_per_update": 1e6}):
        with pytest.raises(ValueError):
            plan.production_plan(measured(**bad), ISSUE_QUOTA, 40 * 2**30)
    with pytest.raises(ValueError):
        plan.production_plan({"seconds_per_update": 1.0}, ISSUE_QUOTA, 40 * 2**30)


def test_segment_chain_binds_the_previous_checkpoint_and_refuses_drift():
    segments = plan.segment_boundaries(1000)
    first = plan.next_segment(None, "mean", segments)
    assert first == {"arm": "mean", "segment": 0, "start_step": 0, "end_step": 1000,
                     "previous_checkpoint_sha256": None}
    assert t4.segment_contract(first) == first
    receipt = {"status": t4.SEGMENT_STATUS, "contract": first, "checkpoint_sha256": "a" * 64,
               "final_epoch10_evaluated": False, "intermediate_selection": False}
    second = plan.next_segment(receipt, "mean", segments)
    assert second["previous_checkpoint_sha256"] == "a" * 64 and second["start_step"] == 1000
    for tamper in ({"final_epoch10_evaluated": True}, {"status": "other"}):
        with pytest.raises(ValueError):
            plan.next_segment(receipt | tamper, "mean", segments)
    with pytest.raises(ValueError):
        plan.next_segment(receipt, "query", segments)
    with pytest.raises(ValueError):
        t4.segment_contract(dict(second, previous_checkpoint_sha256=None))
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
        2, 16, 2)
    production = t4.production_config(configs["query"], spec)
    assert {k for k in production if production[k] != configs["query"][k]} == {
        "encoder", "min_free_gib"}
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
    first = json.loads((audits / "kaggle-public-source-cohort-progress-20261013-v1/"
                        "original-shard000-completed.json").read_text())
    found.setdefault(0, json.dumps(first).split("shard_manifest_sha256\": \"")[1][:64])
    assert [found[i] for i in range(20)] == list(t4.SHARD_MANIFEST_SHA256)
    cache = json.loads((audits / "public-source-complete-cache-20261013-v1/verified.json").read_text())
    assert (cache["index_sha256"], cache["exact_member_bytes"]) == (
        t4.INDEX_SHA256, t4.EXACT_MEMBER_BYTES)
    assert cache["cohort_manifest_sha256"] == t4.COHORT_MANIFEST_SHA256
    assert sum(t4.SHARD_STUDIES) == 3403


def test_numerical_package_is_the_immutable_science_commit():
    files, manifest = nb.immutable_package()
    stage = json.loads(nb.STAGE_MANIFEST.read_text())["source"]
    assert manifest["files"] == stage["files"] and manifest["science_git"] == t4.SCIENCE_GIT
    for name, value in stage["files"].items():
        assert hashlib.sha256(files[name].encode()).hexdigest() == value
    # Current-tree trainer changes after 971d7dac are never shipped as the package.
    assert files["rsna/competitive/train.py"].encode() != (
        ROOT / "rsna/competitive/train.py").read_bytes()
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


class FakeRemote:
    def __init__(self, quota=ISSUE_QUOTA, save=None):
        self.calls, self.quota, self.save = [], quota, save

    def kernel(self, method, slug, version):
        self.calls.append((method, slug))
        if method == "GetKernelSessionStatus":
            return {"status": "COMPLETE"}
        return {"metadata": {"isPrivate": True, "enableGpu": False, "enableInternet": False}}

    def call(self, service, method, body):
        self.calls.append((method, None))
        if method == "GetAcceleratorQuotaStatistics":
            return {"gpuQuota": self.quota}
        if method == "SaveKernel":
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
        "kernel_slug": "me/rsna-window-bag-t4-engineering-v1",
        "source_sha256": "s",
        "mode": "engineering",
        "shard_kernels": [
            {"shard_index": i, "slug": f"me/shard{i:03d}", "version": 1,
             "shard_manifest_sha256": h}
            for i, h in enumerate(t4.SHARD_MANIFEST_SHA256)
        ],
        "model_dataset": {"slug": "me/model", "id": 12333543},
        "wheelhouse": {"dataset_slug": "me/wheels"},
        "capsule_dataset": {"slug": "me/capsule", "id": 12567458},
        "runtime_image": "image@sha256:x",
        "session_timeout_seconds": 7200,
        "platform_limits": {"observed_at": "t", "evidence": "actual", "max_kernel_sources": 20,
                            "max_dataset_sources": 3},
        "production_funding": None,
    }
    return value | changes


def phase(name, remote, journal, value=None):
    return wc.run_phase(name, value or contract(), "source", remote, Publication, Workflow,
                        lambda n, v: journal.append(n), lambda s: "s")


def test_preflight_is_read_only_and_push_saves_once_without_retry():
    remote, journal = FakeRemote(), []
    proof = phase("preflight", remote, journal)
    assert proof["funding"]["charge_seconds"] == 14400 and journal == []
    assert ("SaveKernel", None) not in remote.calls
    request = wc.kernel_request(contract(), "source")
    assert len(request["kernelDataSources"]) == 20 and request["enableInternet"] is False
    assert request["machineShape"] == "NvidiaTeslaT4" and request["sessionTimeoutSeconds"] == 7200
    good = FakeRemote(save={"versionNumber": 1, "ref": "/code/me/rsna-window-bag-t4-engineering-v1"})
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
        wc.run_phase("push", contract(), "other", FakeRemote(), Publication, Workflow,
                     lambda *_: None, lambda s: s)


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
    s = dict(tmp=tmp_path, source=source, fit=fit[0], held=held[0], records=records,
             plane_map=fit[2] | held[2],
             studies={u: pc.study_source_sha256(u, r) for u, r in records.items()})
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
    return dict(s=s, mount=mount, merged=merged, index=index, cohort=cohort,
                policy=ksr_test.POLICY, group=ksr_test.PIXEL_GROUP)


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
    member.write_bytes(member.read_bytes()[:-1] + b"\0")
    with pytest.raises(ValueError):
        t4.restore(tmp_path / "bad", found, manifests, shards["index"],
                   [member.name.removesuffix(".npz.gz")])
    duplicate = shards["mount"] / "copy"
    shutil.copytree(found[0], duplicate / "private/shard/shard")
    with pytest.raises(ValueError):
        t4.locate_shards(shards["mount"])


# --- Complete CPU recovery across an actual process/session boundary -------------------


def session(root, arm, output, stop=None, resume=None):
    spec = {"root": str(root), "arm": arm, "output": str(output), "stop": stop,
            "resume": None if resume is None else str(resume)}
    return subprocess.run(
        [sys.executable, str(ROOT / "tests/window_bag_session_child.py"), json.dumps(spec)],
        cwd=ROOT, env=dict(os.environ, PYTHONPATH=str(ROOT)), timeout=900,
    ).returncode


@pytest.mark.parametrize("arm", ["mean", "query"])
def test_complete_recovery_across_a_session_boundary(tmp_path, arm):
    torch = pytest.importorskip("torch")
    from test_competitive_training import _assert_tree_equal

    assert session(tmp_path, arm, tmp_path / "full") == 0
    first = tmp_path / "session-1"
    assert session(tmp_path, arm, first, stop=1) == 75
    # Session 1 output becomes an immutable read-only input of session 2.
    mounted = tmp_path / "mounted-session-1"
    shutil.copytree(first, mounted)
    for path in mounted.rglob("*"):
        path.chmod(stat.S_IRUSR | (stat.S_IXUSR if path.is_dir() else 0))
    mounted.chmod(stat.S_IRUSR | stat.S_IXUSR)
    receipt = {"status": t4.SEGMENT_STATUS,
               "contract": {"arm": arm, "segment": 0, "start_step": 0, "end_step": 1,
                            "previous_checkpoint_sha256": None},
               "checkpoint_sha256": t4.sha(mounted / "checkpoint.pt"),
               "final_epoch10_evaluated": False, "intermediate_selection": False}
    nxt = plan.next_segment(receipt, arm, [(0, 1), (1, 4)])
    second = tmp_path / "session-2"
    second.mkdir()
    shutil.copyfile(mounted / "checkpoint.pt", second / "checkpoint.pt")
    assert t4.sha(second / "checkpoint.pt") == nxt["previous_checkpoint_sha256"]
    assert session(tmp_path, arm, second, resume=second / "checkpoint.pt") == 0
    full = torch.load(tmp_path / "full/checkpoint.pt", weights_only=True)
    resumed = torch.load(second / "checkpoint.pt", weights_only=True)
    _assert_tree_equal(full, resumed)
    assert (tmp_path / "full/outer_predictions.csv").read_bytes() == (
        second / "outer_predictions.csv").read_bytes()
    partial = torch.load(mounted / "checkpoint.pt", weights_only=True)
    assert (partial["epoch"], partial["next_update"], partial["step"]) == (0, 1, 1)
    for path in [mounted, *mounted.rglob("*")]:
        path.chmod(stat.S_IRWXU)
