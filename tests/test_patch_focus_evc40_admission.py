"""Synthetic checks of the separately versioned evc40 patch-focus target admission.

Engineering tests only: no real MRI, relay, GPU or target placement is exercised.
"""

import copy
import json
import os
import time
from pathlib import Path

import pytest
from test_label_source_overlay import _inputs

from scripts.newton import admit_dense_suffix_inputs as dense
from scripts.newton import admit_patch_focus_inputs as admit
from scripts.newton import admit_patch_focus_inputs_evc40 as target
from scripts.newton import probe_public_input_recipe_inputs as gate
from scripts.newton import triplet_cache_relay as relay
from scripts.research.build_label_source_overlay import build

ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30
ORIGINAL_OVERLAY = (
    ROOT / "initiative/experiments/competitive-v1/public-label-source-v1/fold0-overlay.json"
)
STAGE = ROOT / "initiative/experiments/competitive-v1/source-patch-focus-v1"
FINAL_SOURCE = {
    "name": target.FINAL_SOURCE_NAME,
    "csv_sha256": target.ORIGINAL_SOURCE["csv_sha256"],
    "revision": target.ORIGINAL_SOURCE["revision"],
}


def compact(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n").encode()


def synthetic(tmp_path):
    """Original fold-0 and final all-development overlays from the same builder."""
    studies, folds, source = _inputs(tmp_path)
    paths = {scope: tmp_path / f"{scope}.json" for scope in ("outer_fold", "all_development")}
    for scope, path in paths.items():
        build(studies, folds, 0, source, path, "synthetic", "pinned", scope)
    original = json.loads(paths["outer_fold"].read_text())
    final = json.loads(paths["all_development"].read_text())
    original["source"] = dict(target.ORIGINAL_SOURCE)
    final["source"] = dict(FINAL_SOURCE)
    fitting, outer = target.original_rosters(
        studies,
        folds,
        {"release_sha256": gate.sha(studies), "folds_sha256": gate.sha(folds), "fold": 0},
    )
    raw = compact(original)
    pins = {
        "release_sha256": gate.sha(studies),
        "folds_sha256": gate.sha(folds),
        "fitting_uid_sha256": target.roster_sha256(fitting),
        "outer_uid_sha256": target.roster_sha256(outer),
        "development_uid_sha256": target.roster_sha256(fitting + outer),
        "fitting_studies": len(fitting),
        "outer_studies": len(outer),
        "fold": 0,
        "overlay_sha256": target._sha_bytes(raw),
        "overlay_bytes": len(raw),
    }
    return final, fitting, outer, raw, pins


def funding(**changes):
    values: dict = dict(
        status=target.FUNDING_STATUS,
        admission_version=target.VERSION,
        experiment=admit.EXPERIMENT,
        node="evc40",
        funded_by="root",
        original_input_receipt_sha256=dense.ORIGINAL_RECEIPT_SHA256,
        merged_cache_receipt_sha256=dense.MERGED_RECEIPT_SHA256,
        cache_relay_receipt_sha256=target.RELAY_RECEIPT_SHA256,
        original_overlay_sha256=target.ORIGINAL_OVERLAY_SHA256,
        new_CPU_source_and_receipt_bytes=16 * 2**20,
        new_GPU_or_checkpoint_bytes=0,
        new_shared_bytes_files=0,
        prefix_cache_bytes=0,
        GPUs_funded=0,
        GPU_reserved=False,
        existing_liabilities_bytes=338_320_442_039,
        reserve_bytes=20 * GIB,
    )
    values |= changes
    values.setdefault(
        "required_free_bytes",
        values["new_CPU_source_and_receipt_bytes"]
        + values["existing_liabilities_bytes"]
        + values["reserve_bytes"],
    )
    return values


# --- Original evc31 path is unchanged and still refuses evc40 ---------------------------


def test_original_evc31_admission_is_byte_unchanged_and_refuses_evc40(monkeypatch):
    assert gate.sha(ROOT / admit.SELF) == (
        "a925d5a472e5999256e08a581091bd18d3feef3c4a06267c369845be29676c7f"
    )
    assert gate.NODE == "evc31" and "evc40" not in (ROOT / admit.SELF).read_text()
    monkeypatch.setenv("SLURM_JOB_ID", "999001")
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gate.os, "uname", lambda: os.uname_result(("L", "evc40", "", "", "")))
    with pytest.raises(ValueError, match="Original owning-node"):
        gate.check_placement()


def test_target_constants_bind_the_original_lineage():
    assert target.ORIGINAL_OVERLAY_SHA256 == gate.OVERLAY_SHA256 == admit.dense.gate.OVERLAY_SHA256
    assert gate.sha(ORIGINAL_OVERLAY) == target.ORIGINAL_OVERLAY_SHA256
    assert ORIGINAL_OVERLAY.stat().st_size == target.ORIGINAL_OVERLAY_BYTES
    for name, digest in target.PINNED_SOURCE.items():
        assert gate.sha(ROOT / name) == digest
    assert {target.SELF, admit.SELF, admit.WRAPPER, dense.ORIGINAL_RECEIPT} <= set(
        target.REQUIRED_SOURCE
    )
    assert target.PINS["development_uid_sha256"] == relay.COHORT_SHA256
    original = json.loads((ROOT / dense.ORIGINAL_RECEIPT).read_text())
    assert original["supervision"]["source"] == target.ORIGINAL_SOURCE
    assert original["supervision"]["release_sha256"] == target.PINS["release_sha256"]
    assert original["supervision"]["folds_sha256"] == target.PINS["folds_sha256"]


# --- Placement / runtime / funding -------------------------------------------------------

GOOD_ENV = {"SLURM_JOB_ID": "912345", "SLURM_CPUS_PER_TASK": "1"}


def test_new_standalone_evc40_cpu_allocation_accepted():
    assert target.check_target_placement(GOOD_ENV, "evc40") == "912345"


@pytest.mark.parametrize(
    "env,node",
    [
        (GOOD_ENV, "evc31"),
        (GOOD_ENV | {"SLURM_JOB_ID": "908454"}, "evc40"),
        (GOOD_ENV | {"SLURM_JOB_ID": "899766"}, "evc40"),
        ({"SLURM_CPUS_PER_TASK": "1"}, "evc40"),
        (GOOD_ENV | {"SLURM_CPUS_PER_TASK": "2"}, "evc40"),
        ({"SLURM_JOB_ID": "912345"}, "evc40"),
        (GOOD_ENV | {"SLURM_STEP_GPUS": "0"}, "evc40"),
        (GOOD_ENV | {"SLURM_JOB_GPUS": "1"}, "evc40"),
        (GOOD_ENV | {"CUDA_VISIBLE_DEVICES": "0"}, "evc40"),
    ],
)
def test_old_parent_foreign_node_or_gpu_placement_refused(env, node):
    with pytest.raises(ValueError, match="standalone evc40"):
        target.check_target_placement(env, node)


def test_runtime_must_be_the_actual_restored_original_runtime():
    audit = json.loads((ROOT / target.RUNTIME_AUDIT).read_text())
    result = target.check_runtime(audit, target.RUNTIME_PYTHON, "5.6.1", "5.6.1")
    assert result["runtime_archive_sha256"] == target.RUNTIME_ARCHIVE_SHA256
    bad_archive = copy.deepcopy(audit)
    bad_archive["runtime"]["archive_sha256"] = "0" * 64
    for args in (
        (audit, "/usr/bin/python3", "5.6.1", "5.6.1"),
        (audit, target.RUNTIME_PYTHON, "5.7.0", "5.6.1"),
        (bad_archive, target.RUNTIME_PYTHON, "5.6.1", "5.6.1"),
        (audit | {"status": "inferred_compatible"}, target.RUNTIME_PYTHON, "5.6.1", "5.6.1"),
    ):
        with pytest.raises(ValueError):
            target.check_runtime(*args)


def test_funding_is_fresh_16mib_control_with_retained_liabilities_and_floor():
    assert target.validate_funding(funding())["required_free_bytes"] == (
        16 * 2**20 + 338_320_442_039 + 20 * GIB
    )
    for bad in (
        {"existing_liabilities_bytes": 338_320_442_038},
        {"reserve_bytes": 19 * GIB},
        {"new_CPU_source_and_receipt_bytes": 64 * 2**20},
        {"new_GPU_or_checkpoint_bytes": 12 * GIB},
        {"new_shared_bytes_files": 1},
        {"GPUs_funded": 1},
        {"GPU_reserved": True},
        {"node": "evc31"},
        {"experiment": "dense_suffix_capacity_v1"},
        {"cache_relay_receipt_sha256": "0" * 64},
        {"merged_cache_receipt_sha256": "0" * 64},
        {"original_overlay_sha256": target.FINAL_OVERLAY_SHA256},
        {"funded_by": "devin"},
        {"required_free_bytes": 1},
        {"deletion_credit_bytes": 0},
    ):
        with pytest.raises(ValueError):
            target.validate_funding(funding(**bad))
    with pytest.raises(ValueError):
        target.validate_funding({k: v for k, v in funding().items() if k != "status"})


# --- Exact original overlay reconstruction -----------------------------------------------


def test_reconstruction_restores_only_the_original_name_and_is_byte_exact(tmp_path):
    final, fitting, outer, raw, pins = synthetic(tmp_path)
    before = copy.deepcopy(final)
    assert target.reconstruct(final, fitting, outer, pins) == raw
    assert final == before


def test_reconstruction_of_the_tracked_original_overlay_bytes():
    """Real 2247-study bytes: rebuild a final table around them, recover c8fd2e86 exactly."""
    original = json.loads(ORIGINAL_OVERLAY.read_bytes())
    fitting = [s["study_uid"] for s in original["studies"]]
    masked = {t: 0 for t in original["studies"][0]["mask"]}
    outer = [f"9.9.9.{index:04d}" for index in range(3)]
    extra = [{"study_uid": u, "labels": dict.fromkeys(masked, 0.0), "mask": masked} for u in outer]
    studies = sorted([*original["studies"], *extra], key=lambda s: s["study_uid"])
    final = {
        "schema_version": 1,
        "status": "label_source_final_fit_only",
        "source": dict(FINAL_SOURCE),
        "release_sha256": target.PINS["release_sha256"],
        "folds_sha256": target.PINS["folds_sha256"],
        "fit_scope": "all_development",
        "fold": 0,
        "fitting_uid_sha256": target.roster_sha256(fitting + outer),
        "outer_uid_sha256": target.roster_sha256([]),
        "study_count": len(studies),
        "studies": studies,
    }
    final = json.loads(compact(final))
    fake = target.roster_sha256(outer)
    pins = target.PINS | {
        "outer_uid_sha256": fake,
        "outer_studies": len(outer),
        "development_uid_sha256": target.roster_sha256(fitting + outer),
    }
    raw = target.reconstruct(final, fitting, outer, pins)
    real = raw.replace(fake.encode(), target.PINS["outer_uid_sha256"].encode())
    assert target._sha_bytes(real) == target.ORIGINAL_OVERLAY_SHA256
    assert len(real) == target.ORIGINAL_OVERLAY_BYTES


def _edit_source(final, **change):
    final["source"].update(change)


@pytest.mark.parametrize(
    "edit",
    [
        lambda f: _edit_source(f, csv_sha256="0" * 64),
        lambda f: _edit_source(f, revision="main"),
        lambda f: _edit_source(f, name="llm_labels_v5"),
        lambda f: f.update(source={"csv_sha256": FINAL_SOURCE["csv_sha256"]}),
        lambda f: f.update(status="label_source_experiment_only"),
        lambda f: f.update(fit_scope="outer_fold"),
        lambda f: f.update(fold=1),
        lambda f: f.update(release_sha256="0" * 64),
        lambda f: f.update(folds_sha256="0" * 64),
        lambda f: f.update(fitting_uid_sha256="0" * 64),
        lambda f: f.update(outer_uid_sha256="0" * 64),
        lambda f: f.update(study_count=f["study_count"] + 1),
        lambda f: f["studies"].pop(),
        lambda f: f["studies"].append(copy.deepcopy(f["studies"][0])),
        lambda f: f["studies"][0].update(study_uid="uid-foreign"),
        lambda f: f["studies"].reverse(),
        lambda f: f["studies"][0]["mask"].update(ACL=0.5),
        lambda f: f["studies"][0]["mask"].update(ACL=True),
        lambda f: f["studies"][0]["mask"].pop("ACL"),
        lambda f: f["studies"][0].update(confidence=1.0),
        lambda f: f.update(extra=True),
    ],
)
def test_wrong_source_scope_roster_order_or_masks_refused(tmp_path, edit):
    final, fitting, outer, _, pins = synthetic(tmp_path)
    edit(final)
    with pytest.raises(ValueError):
        target.reconstruct(final, fitting, outer, pins)


@pytest.mark.parametrize(
    "rosters",
    [
        lambda f, o: (list(reversed(f)), o),
        lambda f, o: (f[:-1], o + f[-1:]),
        lambda f, o: (f, o[:-1]),
        lambda f, o: (f + o[:1], o[1:]),
        lambda f, o: (f[:-1] + f[:1], o),
    ],
)
def test_wrong_original_roster_order_or_hash_refused(tmp_path, rosters):
    final, fitting, outer, _, pins = synthetic(tmp_path)
    with pytest.raises(ValueError):
        target.reconstruct(final, *rosters(fitting, outer), pins)


def test_shared_study_and_fold_bytes_must_be_the_original(tmp_path):
    studies, folds, _ = _inputs(tmp_path)
    pins = {"release_sha256": gate.sha(studies), "folds_sha256": gate.sha(folds), "fold": 0}
    target.original_rosters(studies, folds, pins)
    for change in ({"release_sha256": "0" * 64}, {"folds_sha256": "0" * 64}):
        with pytest.raises(ValueError, match="study/fold"):
            target.original_rosters(studies, folds, pins | change)
    with pytest.raises(ValueError):
        target.original_rosters(studies, folds)


def test_final_overlay_source_must_be_the_exact_pinned_bytes(tmp_path):
    path = tmp_path / "overlay.json"
    path.write_bytes(b"{}\n")
    assert target.read_final_overlay(path, target._sha_bytes(b"{}\n")) == b"{}\n"
    with pytest.raises(ValueError, match="pinned evc40 path"):
        target.read_final_overlay(path)
    with pytest.raises(ValueError, match="admitted evc40 source"):
        target.read_final_overlay(path, "0" * 64)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        target.read_final_overlay(link, target._sha_bytes(b"{}\n"))


# --- Exclusive capped target write --------------------------------------------------------


def _control(tmp_path) -> Path:
    root = tmp_path / "control"
    root.mkdir()
    return root


def test_exact_overlay_is_written_once_read_only(tmp_path):
    _, _, _, raw, pins = synthetic(tmp_path)
    root = _control(tmp_path)
    path = target.write_overlay(root, raw, {"required_free_bytes": 0}, pins)
    assert path == root / target.OVERLAY_NAME and path.read_bytes() == raw
    assert path.stat().st_mode & 0o777 == 0o400
    (root / target.OVERLAY_RECEIPT).write_text(
        json.dumps({"status": target.OVERLAY_STATUS, "overlay_sha256": pins["overlay_sha256"]})
    )
    assert target.check_overlay(root, pins) == path
    with pytest.raises(ValueError, match="Existing overlay"):
        target.write_overlay(root, raw, {"required_free_bytes": 0}, pins)


def test_non_original_bytes_are_never_written(tmp_path):
    _, _, _, raw, pins = synthetic(tmp_path)
    root = _control(tmp_path)
    for candidate, values in ((raw + b" ", pins), (raw, target.PINS)):
        with pytest.raises(ValueError, match="exact original"):
            target.write_overlay(root, candidate, {"required_free_bytes": 0}, values)
    assert not any(root.iterdir())


def test_escaped_linked_or_existing_destinations_refused(tmp_path):
    _, _, _, raw, pins = synthetic(tmp_path)
    free = {"required_free_bytes": 0}
    real = _control(tmp_path)
    linked = tmp_path / "linked"
    linked.symlink_to(real, target_is_directory=True)
    with pytest.raises(ValueError, match="Linked"):
        target.write_overlay(linked, raw, free, pins)
    with pytest.raises(ValueError, match="Linked"):
        target.write_overlay(linked / "nested", raw, free, pins)
    (real / target.OVERLAY_NAME).symlink_to(tmp_path / "elsewhere.json")
    with pytest.raises(ValueError, match="Existing overlay"):
        target.write_overlay(real, raw, free, pins)
    assert not (tmp_path / "elsewhere.json").exists()
    other = tmp_path / "other"
    other.mkdir()
    (other / target.OVERLAY_RECEIPT).write_text("{}")
    with pytest.raises(ValueError, match="Existing overlay"):
        target.write_overlay(other, raw, free, pins)


def test_control_cap_and_fresh_free_space_are_enforced(tmp_path):
    _, _, _, raw, pins = synthetic(tmp_path)
    root = _control(tmp_path)
    with pytest.raises(ValueError, match="free space"):
        target.write_overlay(root, raw, {"required_free_bytes": 2**62}, pins)
    (root / "code.bin").write_bytes(b"\0" * (target.CONTROL_BYTES - len(raw)))
    with pytest.raises(ValueError, match="control cap"):
        target.write_overlay(root, raw, {"required_free_bytes": 0}, pins)
    assert not (root / target.OVERLAY_NAME).exists()


def test_changed_reconstructed_overlay_refused_at_admission(tmp_path):
    _, _, _, raw, pins = synthetic(tmp_path)
    root = _control(tmp_path)
    with pytest.raises(FileNotFoundError):
        target.check_overlay(root, pins)
    path = target.write_overlay(root, raw, {"required_free_bytes": 0}, pins)
    with pytest.raises(FileNotFoundError):
        target.check_overlay(root, pins)
    (root / target.OVERLAY_RECEIPT).write_text(
        json.dumps({"status": target.OVERLAY_STATUS, "overlay_sha256": pins["overlay_sha256"]})
    )
    os.chmod(path, 0o600)
    with pytest.raises(ValueError, match="absent or changed"):
        target.check_overlay(root, pins)


# --- Actual relay proof instead of the untransferred producer receipt -------------------


def test_actual_relay_receipt_accepted_only_byte_equal_to_its_audit():
    audit = (ROOT / target.RELAY_AUDIT).read_bytes()
    receipt = target.validate_relay_receipt(audit, audit)
    assert receipt["merged_receipt_sha256"] == dense.MERGED_RECEIPT_SHA256
    assert (receipt["files"], receipt["bytes"]) == (3404, 27_668_082_285)
    changed = audit.replace(b'"node": "evc40"', b'"node": "evc41"')
    for raw, copy_ in ((changed, changed), (changed, audit), (audit, changed)):
        with pytest.raises(ValueError, match="verified actual transfer"):
            target.validate_relay_receipt(raw, copy_)


def test_original_producer_receipt_is_never_forged_on_target(tmp_path):
    path = tmp_path / "merged-cache-verified.json"
    target.require_no_forged_merged_receipt(path)
    path.write_text("{}")
    with pytest.raises(ValueError, match="not transferred"):
        target.require_no_forged_merged_receipt(path)
    assert target.require_no_forged_merged_receipt.__defaults__[0] == Path(gate.MERGED_RECEIPT)


def _published(tmp_path):
    cache = tmp_path / "merged-cache"
    members = cache / relay.MEMBERS_DIR
    members.mkdir(parents=True)
    manifest = {}
    for name, payload in (
        (relay.INDEX, b"{}"),
        *((f"{relay.MEMBERS_DIR}/u{i}.npz", b"x" * i) for i in (1, 2)),
    ):
        (cache / name).write_bytes(payload)
        os.chmod(cache / name, 0o400)
        manifest[name] = {"bytes": len(payload), "sha256": target._sha_bytes(payload)}
    time.sleep(0.01)
    receipt = tmp_path / "target-verified.json"
    receipt.write_text("{}")
    return cache, manifest, receipt


def test_original_publication_signatures_accepted_without_rehash(tmp_path):
    cache, manifest, receipt = _published(tmp_path)
    proof = target.check_publication(cache, manifest, receipt)
    assert proof["files"] == 3 and proof["bytes"] == 5 and not proof["member_bytes_rehashed"]


@pytest.mark.parametrize("change", ["mode", "ctime", "link", "extra", "missing", "size", "symlink"])
def test_changed_or_linked_publication_refused(tmp_path, change):
    cache, manifest, receipt = _published(tmp_path)
    member = cache / relay.MEMBERS_DIR / "u1.npz"
    if change == "mode":
        os.chmod(member, 0o600)
    elif change == "ctime":
        time.sleep(0.01)
        os.chmod(member, 0o400)
    elif change == "link":
        os.link(member, tmp_path / "hardlink")
    elif change == "extra":
        (cache / relay.MEMBERS_DIR / "u9.npz").write_bytes(b"")
    elif change == "missing":
        manifest[f"{relay.MEMBERS_DIR}/u3.npz"] = {"bytes": 0, "sha256": "0" * 64}
    elif change == "size":
        manifest[f"{relay.MEMBERS_DIR}/u2.npz"]["bytes"] = 3
    else:
        os.chmod(member, 0o600)
        member.unlink()
        member.symlink_to(cache / relay.INDEX)
    with pytest.raises(ValueError):
        target.check_publication(cache, manifest, receipt)


# --- Reviewable stage template ------------------------------------------------------------


def test_stage_template_matches_the_wrapper_bindings():
    stage = json.loads((STAGE / "cpu-admission-evc40-stage-manifest.json").read_text())
    assert stage["status"] == "proposed_not_staged_not_funded_not_dispatched"
    assert (stage["node"], stage["root"]) == ("evc40", str(target.CONTROL_ROOT))
    assert stage["admission"] == target.SELF and stage["admission_version"] == target.VERSION
    assert stage["original_evc31_admission_unchanged"] == admit.SELF
    assert set(stage["forbidden_allocations"]) == set(target.FORBIDDEN_JOBS)
    pinned = stage["source"]["required_fixed_sha256"]
    assert all(pinned[name] == digest for name, digest in target.PINNED_SOURCE.items())
    expected = funding()
    for key, value in stage["funding.json"].items():
        if key not in {"existing_liabilities_bytes", "required_free_bytes"}:
            assert expected[key] == value, key
    overlay = stage["overlay_reconstruction"]
    assert overlay["final_overlay"] == str(target.FINAL_OVERLAY)
    assert overlay["final_overlay_sha256"] == target.FINAL_OVERLAY_SHA256
    assert overlay["original_overlay_sha256"] == target.ORIGINAL_OVERLAY_SHA256
    assert overlay["original_overlay_bytes"] == target.ORIGINAL_OVERLAY_BYTES
    assert stage["cache"]["relay_receipt_sha256"] == target.RELAY_RECEIPT_SHA256
    assert stage["future_GPU_recovery_after_CPU_acceptance"]["funded"] is False
    assert stage["future_GPU_recovery_after_CPU_acceptance"]["arguments"]["--overlay"] == str(
        target.CONTROL_ROOT / target.OVERLAY_NAME
    )
