"""Synthetic acceptance/refusal checks of the separately versioned evc40 target admission.

Engineering tests only: no MRI, overlay, relay cache or runtime is read or written here.
"""

import copy
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.newton import admit_dense_suffix_inputs as dense
from scripts.newton import admit_patch_focus_inputs as admit
from scripts.newton import admit_patch_focus_target_inputs as target
from scripts.newton import probe_public_input_recipe_inputs as gate
from scripts.newton import triplet_cache_relay as triplet

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL = json.loads((ROOT / dense.ORIGINAL_RECEIPT).read_text())
RESTORATION = json.loads((ROOT / target.RESTORATION_AUDIT).read_text())
RELAY = json.loads((ROOT / target.RELAY_AUDIT).read_text())
ENV = {"SLURM_JOB_ID": "917001", "SLURM_CPUS_PER_TASK": "1", "SLURM_JOB_NODELIST": "evc40"}


def declaration(**changes):
    values: dict = {
        "status": target.DECLARATION_STATUS,
        "version": target.VERSION,
        "funded_by": "root",
        "issue": 84,
        "node": "evc40",
        "experiment": admit.EXPERIMENT,
        "owning_root": str(target.ROOT),
        "funding_overlay_sha256": "a" * 64,
        "overlay_restored_sha256": "b" * 64,
        "relay_receipt_sha256": "c" * 64,
        "original_input_receipt_sha256": dense.ORIGINAL_RECEIPT_SHA256,
        "merged_cache_receipt_sha256": dense.MERGED_RECEIPT_SHA256,
        "base_proof_sha256": gate.BASE_PROOF_SHA256,
        "runtime": {"prefix": str(target.RUNTIME)},
        "cap_bytes": 16 * 2**20,
        "existing_liabilities_bytes": 338320442039,
        "reserve_bytes": 20 * 2**30,
        "new_GPU_or_checkpoint_bytes": 0,
        "new_shared_bytes_files": 0,
        "GPU_funded": False,
    }
    values.update(changes)
    return values


def test_pins_are_the_reviewed_restoration_relay_and_original_lineage():
    assert target.OVERLAY == Path("/tmp/rsna-public-label-repeat-inputs-20261012-v1/overlay.json")
    assert (RESTORATION["overlay_sha256"], RESTORATION["bytes"]) == (
        target.OVERLAY_SHA256,
        target.OVERLAY_BYTES,
    )
    assert RESTORATION["CPU_job"] == target.OVERLAY_JOB and RESTORATION["node"] == "evc40"
    assert RESTORATION["funding"]["existing_liabilities_bytes"] == target.LIABILITIES_BYTES
    assert RESTORATION["funding"]["new_control_and_overlay_cap_bytes"] == target.CAP_BYTES
    assert RELAY["merged_receipt_sha256"] == dense.MERGED_RECEIPT_SHA256
    assert (RELAY["files"], RELAY["bytes"]) == (target.CACHE_FILES, target.CACHE_BYTES)
    assert gate.sha(ROOT / dense.ORIGINAL_RECEIPT) == dense.ORIGINAL_RECEIPT_SHA256
    assert ORIGINAL["trainer_inputs"]["overlay"] == str(target.OVERLAY)
    assert set(admit.REQUIRED_SOURCE) < set(target.REQUIRED_SOURCE)
    assert {target.SELF, target.RESTORATION_AUDIT, target.RELAY_AUDIT} <= set(
        target.REQUIRED_SOURCE
    )
    assert {
        "scripts/newton/triplet_cache_relay.py",
        "scripts/newton/native_cache_relay.py",
    } <= set(target.REQUIRED_SOURCE)
    assert all((ROOT / name).is_file() for name in target.REQUIRED_SOURCE if "/" in name)
    target.validate_relay(RELAY, RELAY)
    target.validate_restoration(
        RESTORATION["funding"],
        {k: v for k, v in RESTORATION.items() if k != "funding"},
        RESTORATION,
    )


def test_standalone_evc40_cpu_placement_only():
    assert target.check_placement("evc40", ENV) == "917001"
    bad = [
        ("evc31", {}),
        ("evc40", {"SLURM_JOB_ID": "908454"}),
        ("evc40", {"SLURM_JOB_ID": "899766"}),
        ("evc40", {"SLURM_JOB_ID": "916757"}),
        ("evc40", {"SLURM_JOB_ID": ""}),
        ("evc40", {"SLURM_JOB_ID": "12;3"}),
        ("evc40", {"SLURM_CPUS_PER_TASK": "2"}),
        ("evc40", {"SLURM_JOB_NODELIST": "evc31"}),
        ("evc40", {"SLURM_JOB_GPUS": "0"}),
        ("evc40", {"SLURM_STEP_GPUS": "0"}),
        ("evc40", {"SLURM_GPUS_ON_NODE": "1"}),
        ("evc40", {"CUDA_VISIBLE_DEVICES": "0"}),
    ]
    for node, change in bad:
        with pytest.raises(ValueError, match="standalone"):
            target.check_placement(node, ENV | change)


def test_original_route_is_unchanged_and_still_refuses_evc40(monkeypatch):
    assert gate.NODE == "evc31"
    monkeypatch.setattr(gate.os, "uname", lambda: SimpleNamespace(nodename="evc40"))
    monkeypatch.setenv("SLURM_JOB_ID", "917001")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(ValueError, match="Original owning-node"):
        gate.check_placement()
    with pytest.raises(ValueError, match="Original owning-node"):
        admit.run(target.ROOT)


def test_target_run_refuses_off_target_before_reading_anything(tmp_path):
    with pytest.raises(ValueError, match="standalone"):
        target.run(tmp_path)


def test_runtime_must_equal_the_declared_prepared_environment():
    actual = {"prefix": str(target.RUNTIME), "python": "3.12.3", "torch": "2.8.0"}
    assert target.check_runtime(actual, dict(actual)) == actual
    with pytest.raises(ValueError, match="runtime"):
        target.check_runtime(actual, actual | {"torch": "2.8.1"})
    with pytest.raises(ValueError, match="runtime"):
        target.check_runtime(actual | {"prefix": "/usr"}, actual | {"prefix": "/usr"})
    assert set(target.runtime_identity()) == {"prefix", "python", *target.RUNTIME_PACKAGES}


@pytest.mark.parametrize(
    "change",
    [
        {"cap_bytes": 32 * 2**20},
        {"existing_liabilities_bytes": 1},
        {"reserve_bytes": 20 * 2**30 - 1},
        {"new_GPU_or_checkpoint_bytes": 1},
        {"new_shared_bytes_files": 1},
        {"GPU_funded": True},
        {"node": "evc31"},
        {"owning_root": "/tmp/other"},
        {"experiment": "dense_suffix"},
        {"original_input_receipt_sha256": "0" * 64},
        {"merged_cache_receipt_sha256": "0" * 64},
        {"base_proof_sha256": "0" * 64},
        {"relay_receipt_sha256": "not-a-sha"},
        {"funded_by": "devin"},
        {"extra": 1},
    ],
)
def test_declaration_same_cap_no_gpu_and_original_lineage(change):
    assert target.validate_declaration(declaration())["cap_bytes"] == 16 * 2**20
    with pytest.raises(ValueError):
        target.validate_declaration(declaration(**change))


def test_restoration_proofs_must_match_the_reviewed_receipt():
    funding = copy.deepcopy(RESTORATION["funding"])
    restored = {k: v for k, v in RESTORATION.items() if k != "funding"} | {"private": 1}
    target.validate_restoration(funding, restored, RESTORATION)
    with pytest.raises(ValueError, match="funding"):
        target.validate_restoration(
            funding | {"GPU_checkpoint_funding": True}, restored, RESTORATION
        )
    with pytest.raises(ValueError, match="restoration proof"):
        target.validate_restoration(funding, restored | {"overlay_sha256": "0" * 64}, RESTORATION)
    with pytest.raises(ValueError, match="restoration proof"):
        target.validate_restoration(funding, restored | {"CPU_job": "908454"}, RESTORATION)
    with pytest.raises(ValueError, match="restoration proof"):
        target.validate_restoration(funding, {"overlay_sha256": target.OVERLAY_SHA256}, RESTORATION)
    for missing in ("CPU_job", "original_labels_and_masks_unchanged", "source_node_contacted"):
        incomplete = {k: v for k, v in restored.items() if k != missing}
        with pytest.raises(ValueError, match="restoration proof"):
            target.validate_restoration(funding, incomplete, RESTORATION)


def _before(proof, path):
    """Make the proof older than the latest inode change, whatever the clock granularity."""
    ctime = os.lstat(path).st_ctime_ns
    os.utime(proof, ns=(ctime - 10**9, ctime - 10**9))


def _overlay(tmp_path, monkeypatch, payload=b'{"x":1}\n'):
    path = tmp_path / "inputs" / "overlay.json"
    path.parent.mkdir()
    path.write_bytes(payload)
    path.chmod(0o400)
    proof = tmp_path / "proof.json"
    proof.write_text("{}")
    monkeypatch.setattr(target, "OVERLAY", path)
    monkeypatch.setattr(target, "OVERLAY_SHA256", gate.sha(path))
    monkeypatch.setattr(target, "OVERLAY_BYTES", len(payload))
    return path, proof


def test_restored_overlay_is_inspected_never_written(tmp_path, monkeypatch):
    path, proof = _overlay(tmp_path, monkeypatch)
    before = gate.signature(path)
    result = target.check_overlay(path, os.lstat(proof))
    assert result["mode"] == "0400" and gate.signature(path) == before
    with pytest.raises(ValueError, match="original restored path"):
        target.check_overlay(tmp_path / "proof.json", os.lstat(proof))


def test_overlay_wrong_bytes_mode_links_or_later_change_refused(tmp_path, monkeypatch):
    path, proof = _overlay(tmp_path, monkeypatch)
    monkeypatch.setattr(target, "OVERLAY_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="exact original"):
        target.check_overlay(path, os.lstat(proof))
    monkeypatch.setattr(target, "OVERLAY_SHA256", gate.sha(path))
    path.chmod(0o600)
    with pytest.raises(ValueError, match="changed after"):
        target.check_overlay(path, os.lstat(proof))
    path.chmod(0o400)
    _before(proof, path)
    with pytest.raises(ValueError, match="changed after"):
        target.check_overlay(path, os.lstat(proof))
    os.link(path, tmp_path / "second")
    with pytest.raises(ValueError, match="Linked"):
        target.check_overlay(path, os.lstat(tmp_path / "second"))


def test_symlinked_overlay_or_ancestor_refused(tmp_path, monkeypatch):
    real, proof = _overlay(tmp_path, monkeypatch)
    alias = tmp_path / "alias"
    alias.symlink_to(real.parent)
    monkeypatch.setattr(target, "OVERLAY", alias / "overlay.json")
    with pytest.raises(ValueError, match="Linked"):
        target.check_overlay(alias / "overlay.json", os.lstat(proof))
    with pytest.raises(ValueError, match="Escaped"):
        target._no_links(tmp_path / ".." / "x")
    with pytest.raises(ValueError, match="Escaped"):
        target._no_links(Path("relative/x"))


def test_load_exact_requires_pinned_sha_and_regular_single_link(tmp_path):
    path = tmp_path / "proof.json"
    path.write_text('{"a": 1}')
    assert target.load_exact(path, gate.sha(path))[0] == {"a": 1}
    with pytest.raises(ValueError, match="pinned SHA"):
        target.load_exact(path, "0" * 64)
    (tmp_path / "link").symlink_to(path)
    with pytest.raises(ValueError, match="Linked"):
        target.load_exact(tmp_path / "link", gate.sha(path))
    os.link(path, tmp_path / "hard")
    with pytest.raises(ValueError, match="Linked"):
        target.load_exact(path, gate.sha(path))


@pytest.mark.parametrize(
    "change",
    [
        {"node": "evc31"},
        {"cache": "/tmp/other"},
        {"index_sha256": "0" * 64},
        {"manifest_sha256": "0" * 64},
        {"merged_receipt_sha256": "0" * 64},
        {"CPU_receipt_sha256": "0" * 64},
        {"funding_sha256": "0" * 64},
        {"files": 3403},
        {"bytes": 1},
        {"GPU_used": True},
        {"every_original_byte_size_SHA_count_EOF_layout_verified": False},
    ],
)
def test_wrong_relay_source_or_lineage_refused(change):
    with pytest.raises(ValueError, match="relay"):
        target.validate_relay(RELAY | change, RELAY | change)
    with pytest.raises(ValueError, match="relay"):
        target.validate_relay(RELAY | change, RELAY)


def _cache(tmp_path):
    cache = tmp_path / "merged-cache"
    members = cache / triplet.MEMBERS_DIR
    members.mkdir(parents=True)
    manifest = {}
    for name, payload in ((triplet.INDEX, b"{}"), (f"{triplet.MEMBERS_DIR}/u1.npz", b"abc")):
        path = cache / name
        path.write_bytes(payload)
        path.chmod(0o400)
        manifest[name] = {"bytes": len(payload), "sha256": gate.sha(path)}
    receipt = tmp_path / "target-verified.json"
    receipt.write_text("{}")
    return cache, manifest, receipt


def test_relayed_cache_layout_accepted_without_rehash(tmp_path, monkeypatch):
    cache, manifest, receipt = _cache(tmp_path)
    monkeypatch.setattr(gate, "sha", lambda path: pytest.fail("bulk cache must not be rehashed"))
    proof = target.check_cache(cache, manifest, os.lstat(receipt))
    assert (proof["files"], proof["bytes"]) == (2, 5)


def test_relayed_cache_tampering_refused(tmp_path):
    cache, manifest, receipt = _cache(tmp_path)
    member = cache / triplet.MEMBERS_DIR / "u1.npz"
    escaped = manifest | {"../escape.npz": {"bytes": 1, "sha256": "0" * 64}}
    with pytest.raises(ValueError, match="Escaped"):
        target.check_cache(cache, escaped, os.lstat(receipt))
    (cache / triplet.MEMBERS_DIR / "extra.npz").write_bytes(b"x")
    with pytest.raises(ValueError, match="layout"):
        target.check_cache(cache, manifest, os.lstat(receipt))
    (cache / triplet.MEMBERS_DIR / "extra.npz").unlink()
    wrong = copy.deepcopy(manifest)
    wrong[f"{triplet.MEMBERS_DIR}/u1.npz"]["bytes"] = 4
    with pytest.raises(ValueError, match="size"):
        target.check_cache(cache, wrong, os.lstat(receipt))
    member.chmod(0o600)
    with pytest.raises(ValueError, match="changed after"):
        target.check_cache(cache, manifest, os.lstat(receipt))
    member.chmod(0o400)
    _before(receipt, member)
    with pytest.raises(ValueError, match="changed after"):
        target.check_cache(cache, manifest, os.lstat(receipt))
    receipt.write_text("{}")  # A later receipt would admit; then hard link the member.
    os.link(member, tmp_path / "other")
    with pytest.raises(ValueError, match="Linked"):
        target.check_cache(cache, manifest, os.lstat(receipt))


def test_symlinked_cache_member_refused(tmp_path):
    cache, manifest, receipt = _cache(tmp_path)
    member = cache / triplet.MEMBERS_DIR / "u1.npz"
    real = tmp_path / "real.npz"
    shutil.move(member, real)
    member.symlink_to(real)
    with pytest.raises(ValueError):
        target.check_cache(cache, manifest, os.lstat(receipt))


def test_funded_root_usage_refuses_links_and_counts_every_byte(tmp_path):
    (tmp_path / "code").mkdir()
    (tmp_path / "code" / "a.py").write_bytes(b"x" * 10)
    (tmp_path / "funding-overlay.json").write_bytes(b"y" * 5)
    assert target.root_usage(tmp_path) == 15
    (tmp_path / "code" / "link").symlink_to(tmp_path / "funding-overlay.json")
    with pytest.raises(ValueError, match="Linked"):
        target.root_usage(tmp_path)


def test_same_16mib_cap_and_retained_liabilities_plus_reserve():
    funded = declaration()
    need = target.LIABILITIES_BYTES + 20 * 2**30
    remaining = target.CAP_BYTES - target.OVERLAY_BYTES - 2**20
    budget = target.check_budget(2**20, need + remaining, funded)
    assert budget["charged_bytes"] == target.OVERLAY_BYTES + 2**20
    assert budget["remaining_bytes"] == remaining
    with pytest.raises(ValueError, match="16 MiB"):
        target.check_budget(2**20, need + remaining - 1, funded)
    with pytest.raises(ValueError, match="16 MiB"):
        target.check_budget(target.CAP_BYTES - target.OVERLAY_BYTES - 2**20 + 1, 2**50, funded)


def _release(tmp_path):
    release = tmp_path / "release"
    folds = tmp_path / "folds"
    release.mkdir()
    shutil.copytree(ROOT / "initiative/experiments/competitive-v1/folds", folds)
    for name in ("studies.parquet", "verified_release.json"):
        shutil.copy(ROOT / "initiative/releases/report-supervision-v1" / name, release / name)
    return {"studies": str(release / "studies.parquet"), "folds": str(folds)}


def test_bounded_release_uses_exact_shared_bytes_and_original_binding(tmp_path):
    inputs = _release(tmp_path)
    frame, binding = target.bounded_release(inputs, ORIGINAL)
    assert binding["manifest_sha256"] == ORIGINAL["supervision"]["release_sha256"]
    from rsna.competitive.folds import load_folds

    folds = load_folds(Path(inputs["folds"]), frame, binding)
    assert len(folds) == json.loads((Path(inputs["folds"]) / "folds.json").read_text())["n_studies"]


@pytest.mark.parametrize("name", ["studies", "receipt", "binding", "original"])
def test_bounded_release_wrong_source_refused(tmp_path, name):
    inputs = _release(tmp_path)
    original = ORIGINAL
    if name == "studies":
        Path(inputs["studies"]).write_bytes(Path(inputs["studies"]).read_bytes() + b"\0")
    elif name == "receipt":
        (Path(inputs["studies"]).parent / "verified_release.json").write_text("{}")
    elif name == "binding":
        path = Path(inputs["folds"]) / "folds.json"
        value = json.loads(path.read_text())
        value["release_binding"]["manifest_sha256"] = "0" * 64
        path.write_text(json.dumps(value))
    else:
        original = copy.deepcopy(ORIGINAL)
        original["supervision"]["release_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        target.bounded_release(inputs, original)


def test_wrong_roster_membership_or_order_refused():
    proofs = ORIGINAL["tensor_sha256"]
    roles = {
        "fitting": {k: v for k, v in proofs["public/fitting"].items()},
        "development": {k: v for k, v in proofs["public/development"].items()},
    }
    uids = {"fitting": ORIGINAL["training_16_sha256"], "development": ORIGINAL["outer_16_sha256"]}
    dense.check_replay(roles, uids, ORIGINAL)
    with pytest.raises(ValueError):
        dense.check_replay(roles, uids | {"fitting": "0" * 64}, ORIGINAL)
    swapped = {"fitting": roles["development"], "development": roles["fitting"]}
    with pytest.raises(ValueError):
        dense.check_replay(swapped, uids, ORIGINAL)


def test_base_stat_binding_or_single_target_mount_rehash(tmp_path):
    path = tmp_path / "model.safetensors"
    path.write_bytes(b"weights")
    stats = gate.signature(path)
    record = {"signature": stats, "sha256": gate.sha(path)}
    assert target.base_binding(path, record, stats) == "original_stat_binding"
    other = [stats[0] + 1, *stats[1:]]
    moved = {"signature": other, "sha256": record["sha256"]}
    assert target.base_binding(path, moved, other) == "target_mount_whole_byte_rehash"
    with pytest.raises(ValueError, match="whole-byte"):
        target.base_binding(path, moved | {"sha256": "0" * 64}, other)
    with pytest.raises(ValueError, match="stat binding"):
        target.base_binding(path, record, [stats[0], stats[1] + 1, *stats[2:]])
    with pytest.raises(ValueError, match="stat binding"):
        target.base_binding(path, {"signature": other, "sha256": record["sha256"]}, stats)


def test_target_route_keeps_the_backward_only_cloned_mask_control():
    import pandas as pd

    from rsna.schema import TARGETS

    frame = pd.DataFrame({f"{t}__mask": [1, 1] for t in TARGETS})
    before = frame.copy(deep=True)
    _, mask, receipt = admit.routing_probe(frame)
    pd.testing.assert_frame_equal(frame, before)
    assert receipt["accepted_fitting_masks_changed"] is False and receipt["optimizer_steps"] == 0
    assert mask.sum() == 11 and mask[0, -1] == 0
    assert "admit.routing_probe(fitting)" in (ROOT / target.SELF).read_text()


def test_receipt_is_exclusive_and_aggregate_only(tmp_path):
    path = tmp_path / target.OUTPUT
    dense.write_once(path, {"status": target.STATUS}, ["1.2.3"])
    with pytest.raises(FileExistsError):
        dense.write_once(path, {"status": target.STATUS}, ["1.2.3"])
    with pytest.raises(ValueError, match="case-level"):
        dense.write_once(tmp_path / "other.json", {"uid": "1.2.3"}, ["1.2.3"])
