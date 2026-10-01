"""Complete-cohort gates and real reconstructed bytes must survive CPU/GPU handoff."""

import copy
import json
from pathlib import Path

import pytest
import test_native_cache_assembly as prior

from scripts.newton import assemble_native_cache as assembly
from scripts.newton import native384_admission as admission

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def restored(tmp_path):
    root, shard, receipt, report, proofs = prior.bound.__wrapped__(tmp_path)
    record = assembly.bind_shard(shard, receipt, report, proofs)
    destination = tmp_path / "restored"
    assembly.restore_archive(
        root / "cache-side-384.tar",
        receipt["files"]["cache-side-384.tar"],
        record["members"],
        destination,
    )
    return destination, record["members"]


def test_real_native_bytes_survive_cpu_to_gpu_signature_handoff(restored):
    root, members = restored
    signatures = admission.verify_members(root, members)
    assert len(signatures) == 2  # Unit fixture, never a full-cohort acceptance.
    assert signatures == admission.verify_members(root, members, prior=signatures)


@pytest.mark.parametrize("fault", ["bitflip", "extra", "missing", "link", "replaced", "signature"])
def test_corrupted_or_changed_native_cache_cannot_cross_handoff(restored, fault):
    root, members = restored
    signatures = admission.verify_members(root, members)
    path = root / sorted(members)[-1]
    if fault == "bitflip":
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with pytest.raises(ValueError, match="hash_changed"):
            admission.verify_members(root, members)
    elif fault == "extra":
        (root / "cache/extra.npz").write_bytes(b"unexpected payload")
    elif fault == "missing":
        path.unlink()
    elif fault == "link":
        other = root / "same-bytes"
        other.write_bytes(path.read_bytes())
        path.unlink()
        path.symlink_to(other)
    elif fault == "replaced":
        # Identical data at a new inode still invalidates the CPU file proof.
        other = path.with_suffix(".replacement")
        other.write_bytes(path.read_bytes())
        other.replace(path)
    elif fault == "signature":
        signatures["cache/unexpected"] = [1, 2, 3, 4, 5]
    with pytest.raises(ValueError):
        admission.verify_members(root, members, prior=signatures)


@pytest.fixture(scope="module")
def full_registry():
    folder = ROOT / "initiative/experiments/competitive-v1/native-detail-v1/full-cache-shards384-v1"
    plan = json.loads((folder / "manifest.json").read_text())
    registry = {
        "schema_version": 1,
        "status": "all_75_original_source_shards_privately_verified",
        "plan_manifest_sha256": assembly.PLAN_SHA,
        "selection_sha256": plan["selection_sha256"],
        "scientific_source_git": assembly.SCIENCE,
        "source_revision": assembly.SOURCE,
        "repo_id": assembly.REPO,
        "cache_side": 384,
        "study_count": 3403,
        "series_count": 18906,
        "training_uids": plan["training_uids"],
        "outer_uids": plan["outer_uids"],
        "target_values_read": False,
        "full_cache_restored": False,
        "shards": [],
    }
    for n in range(75):
        source = json.loads((folder / f"shard-{n:03d}.json").read_text())
        members = {}
        for row in source["series"]:
            path = "cache/" + row["cache_path"]
            members[path] = {"bytes": 1, "sha256": "0" * 64}
            members[str(Path(path).with_suffix(".json"))] = {"bytes": 1, "sha256": "0" * 64}
        # Synthetic private-receipt placeholders test membership only. They are
        # never restored, published or represented as a real completed cohort.
        registry["shards"].append(
            {"shard": n, "revision": "a" * 40, "series": source["series"], "members": members}
        )
    return registry


def test_full_frozen_registry_accepts_exact_training_outer_and_series_membership(full_registry):
    assert len(admission.validate_registry(full_registry)) == 37812


@pytest.mark.parametrize(
    "fault", ["partial", "side", "train", "duplicate", "path", "floating_revision", "boolean_count"]
)
def test_partial_or_altered_full_registry_cannot_be_admitted(full_registry, fault):
    registry = copy.deepcopy(full_registry)
    if fault == "partial":
        registry["shards"].pop()
    elif fault == "side":
        registry["cache_side"] = 256
    elif fault == "train":
        # Preserve both counts and disjointness; the frozen membership still changed.
        registry["training_uids"][0], registry["outer_uids"][0] = (
            registry["outer_uids"][0],
            registry["training_uids"][0],
        )
    elif fault == "duplicate":
        registry["shards"][1]["series"][0] = registry["shards"][0]["series"][0]
    elif fault == "path":
        registry["shards"][0]["series"][0]["cache_path"] = "../outside.npz"
    elif fault == "floating_revision":
        registry["shards"][0]["revision"] = "main"
    elif fault == "boolean_count":
        registry["schema_version"] = True
    with pytest.raises(ValueError):
        admission.validate_registry(registry)


@pytest.mark.parametrize(
    "state", ["RUNNING|0:0", "PENDING|0:0", "FAILED|2:0", "COMPLETED|1:0", "COMPLETED|0:0"]
)
def test_exact_terminal_cpu_success_is_required(state):
    text = admission.ASSEMBLY_JOB + "|" + state + "\n"
    if state == "COMPLETED|0:0":
        admission.require_terminal_assembly(text)
    else:
        with pytest.raises(ValueError):
            admission.require_terminal_assembly(text)


def test_cpu_preflight_receipt_from_another_job_is_not_terminal_evidence():
    with pytest.raises(ValueError):
        admission.require_terminal_assembly("123|COMPLETED|0:0\n")
    admission.require_terminal_assembly("123|COMPLETED|0:0\n", job="123")
