from __future__ import annotations

import hashlib
import io
import json
import shutil
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
import pydicom
import pytest
from test_pipeline import dicom_bytes

from rsna import source_recovery
from rsna.audit import audit_cache, inventory_fingerprint
from rsna.preprocess import Source, cache_series
from rsna.source_recovery import recover_source_cache, validate_source_correction


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dicom(study: str, series: str, instance: int, malformed: bool = False) -> bytes:
    ds = pydicom.dcmread(io.BytesIO(dicom_bytes(instance, instance)))
    ds.StudyInstanceUID = study
    ds.SeriesInstanceUID = series
    ds.SOPInstanceUID = ds.file_meta.MediaStorageSOPInstanceUID = f"{series}.{instance}"
    pixels = (np.arange(512, dtype=np.uint16) % 200).reshape(16, 32)
    ds.PixelData = pixels.tobytes()[::2] if malformed else pixels.tobytes()
    output = io.BytesIO()
    ds.save_as(output, enforce_file_format=True)
    return output.getvalue()


def refresh(case: dict, *, inventory: bool = False) -> None:
    patch, receipt = case["plan"], case["publication"]
    if inventory:
        corrected = pd.read_parquet(case["corrected"])
        patch["corrected_inventory_fingerprint_sha256"] = inventory_fingerprint(corrected)
        patch["corrected_inventory"].update(
            sha256=sha(case["corrected"]), bytes=case["corrected"].stat().st_size
        )
        receipt["corrected_inventory_sha256"] = sha(case["corrected"])
    case["patch_path"].write_text(json.dumps(patch))
    receipt["source_correction_plan_sha256"] = sha(case["patch_path"])
    case["receipt_path"].write_text(json.dumps(receipt))


@pytest.fixture
def recovery(tmp_path: Path) -> dict:
    baseline_source, effective, cache = [tmp_path / name for name in ["old", "effective", "cache"]]
    baseline_rows, corrected_rows, changes = [], [], []
    for number in range(1, 4):
        study, series = f"1.2.{number}", f"1.2.{number}.1"
        for instance in [1, 2]:
            path = f"train_series/{study}/{series}/{series}.{instance}.dcm"
            changed = number < 3 and instance == 1
            old = dicom(study, series, instance, malformed=changed)
            original = dicom(study, series, instance)
            old_file = baseline_source / path
            old_file.parent.mkdir(parents=True, exist_ok=True)
            old_file.write_bytes(old)
            before = {
                "split": "train_series",
                "study_uid": study,
                "series_uid": series,
                "sop_uid": f"{series}.{instance}",
                "path": path,
                "sha256": hashlib.sha256(old).hexdigest(),
                "bytes": len(old),
                "storage": "lfs",
            }
            after = {
                **before,
                "sha256": hashlib.sha256(original).hexdigest(),
                "bytes": len(original),
            }
            baseline_rows.append(before)
            corrected_rows.append(after)
            if number < 3:
                local_file = effective / path
                local_file.parent.mkdir(parents=True, exist_ok=True)
                local_file.write_bytes(original)
            if changed:
                official_file = tmp_path / "official" / f"{series}.{instance}.dcm"
                official_file.parent.mkdir(exist_ok=True)
                official_file.write_bytes(original)
                changes.append(
                    {
                        **{
                            key: before[key]
                            for key in ["path", "study_uid", "series_uid", "sop_uid"]
                        },
                        "before": before,
                        "after": after,
                        "official_local_path": str(official_file.relative_to(tmp_path)),
                        "official_crc32": f"{zlib.crc32(original) & 0xFFFFFFFF:08x}",
                        "official_shape": [16, 32],
                        "official_dtype": "uint16",
                    }
                )
        if number == 3:
            cache_series(
                Source(baseline_source),
                [row["path"] for row in baseline_rows[-2:]],
                cache,
                32,
                "a" * 40,
            )
    baseline, corrected = pd.DataFrame(baseline_rows), pd.DataFrame(corrected_rows)
    baseline_path, corrected_path = tmp_path / "baseline.parquet", tmp_path / "corrected.parquet"
    baseline.to_parquet(baseline_path, index=False)
    corrected.to_parquet(corrected_path, index=False)
    evidence = tmp_path / "evidence.json"
    evidence.write_text('{"independent_observations": "preserved"}')
    plan = {
        "schema_version": 1,
        "status": "accepted_exact_authoritative_source_replacement",
        "original_pinned_revision_preserved": True,
        "decoder_changed": False,
        "repo_private_verified": True,
        "agent_id": "/root",
        "independent_agent_id": "/root/reviewer",
        "baseline_source_revision": "a" * 40,
        "publication_parent_revision": "b" * 40,
        "repo_id": "fixture/private",
        "evidence": [
            {"path": evidence.name, "sha256": sha(evidence), "bytes": evidence.stat().st_size}
        ],
        "baseline_inventory_fingerprint_sha256": inventory_fingerprint(baseline),
        "corrected_inventory": {"path": corrected_path.name},
        "changes": changes,
        "n_changed_source_files": 2,
        "n_source_files": 6,
        "n_unchanged_source_files": 4,
        "n_affected_series": 2,
        "n_affected_series_slices": 4,
        "source_bytes_before": int(baseline.bytes.sum()),
        "source_bytes_after": int(corrected.bytes.sum()),
    }
    publication = {
        "schema_version": 1,
        "status": "verified_private_hf_source_correction",
        "private": True,
        "remote_objects_verified": True,
        "repo_id": "fixture/private",
        "baseline_source_revision": "a" * 40,
        "publication_parent_revision": "b" * 40,
        "corrected_source_revision": "c" * 40,
        "changed_source_paths": [change["path"] for change in changes],
        "uploaded_files": [
            {key: change["after"][key] for key in ["path", "sha256", "bytes"]} for change in changes
        ],
    }
    result = {
        "baseline_inventory": baseline_path,
        "baseline_cache": cache,
        "patch_path": tmp_path / "patch.json",
        "receipt_path": tmp_path / "receipt.json",
        "artifact_root": tmp_path,
        "official_source_root": effective,
        "output": tmp_path / "recovered",
        "min_free_gib": 3,
        "plan": plan,
        "publication": publication,
        "corrected": corrected_path,
        "baseline_source": baseline_source,
    }
    refresh(result, inventory=True)
    return result


def kwargs(case: dict) -> dict:
    return {
        key: case[key]
        for key in [
            "baseline_inventory",
            "baseline_cache",
            "patch_path",
            "receipt_path",
            "artifact_root",
            "official_source_root",
            "output",
            "min_free_gib",
        ]
    }


def test_complete_recovery_preserves_originals_and_requires_separate_audit(recovery):
    protected = [
        recovery["baseline_inventory"],
        *recovery["baseline_cache"].rglob("*"),
        *recovery["baseline_source"].rglob("*"),
    ]
    before = {path: sha(path) for path in protected if path.is_file()}
    result = recover_source_cache(**kwargs(recovery))
    assert result["status"] == "cache_recovery_complete_full_audit_required"
    assert (result["n_reused_series"], result["n_fresh_series"], result["n_cache_slices"]) == (
        1,
        2,
        6,
    )
    assert result["full_audit_complete"] is False and result["training_ready"] is False
    assert before == {path: sha(path) for path in before}
    assert sha(recovery["output"] / "inventory_effective.parquet") == sha(recovery["corrected"])
    old = next(recovery["baseline_cache"].rglob("*.npz"))
    new = recovery["output"] / "cache" / old.relative_to(recovery["baseline_cache"])
    assert old.stat().st_ino == new.stat().st_ino
    assert old.with_suffix(".json").stat().st_ino != new.with_suffix(".json").stat().st_ino
    original_meta, new_meta = [
        json.loads(path.with_suffix(".json").read_text()) for path in [old, new]
    ]
    assert new_meta == {**original_meta, "source_revision": "c" * 40}
    lineage = [
        json.loads(line)
        for line in (recovery["output"] / "cache_recovery_lineage.jsonl").read_text().splitlines()
    ]
    assert len(lineage) == 3
    assert lineage[0]["metadata_changed_fields"] == ["source_revision"]
    audit = audit_cache(
        pd.read_parquet(recovery["corrected"]),
        recovery["output"] / "cache",
        "c" * 40,
        recovery["output"] / "separate-audit",
    )
    assert audit["complete"] is True and audit["decoded_slices"] == 6
    # Recovery status itself must never be implicitly promoted by a later audit.
    assert (
        json.loads((recovery["output"] / "source_recovery_status.json").read_text())[
            "full_audit_complete"
        ]
        is False
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("corrected_source_revision", "a" * 40),
        ("corrected_source_revision", "b" * 40),
        ("corrected_source_revision", "main"),
        ("baseline_source_revision", "d" * 40),
        ("remote_objects_verified", False),
        ("private", False),
    ],
)
def test_rejects_wrong_revision_or_unverified_receipt(recovery, field, value):
    recovery["publication"][field] = value
    refresh(recovery)
    with pytest.raises(ValueError):
        recover_source_cache(**kwargs(recovery))
    assert not recovery["output"].exists()


def test_changed_plan_or_evidence_is_rejected(recovery):
    recovery["patch_path"].write_text(recovery["patch_path"].read_text() + " ")
    with pytest.raises(ValueError, match="receipt"):
        recover_source_cache(**kwargs(recovery))
    refresh(recovery)
    (recovery["artifact_root"] / "evidence.json").write_text("changed")
    with pytest.raises(ValueError, match="changed source/cache/evidence"):
        recover_source_cache(**kwargs(recovery))


def test_self_approval_and_duplicate_source_changes_rejected(recovery):
    recovery["plan"]["independent_agent_id"] = "/root"
    refresh(recovery)
    with pytest.raises(ValueError, match="distinct assessor"):
        recover_source_cache(**kwargs(recovery))
    recovery["plan"]["independent_agent_id"] = "/root/reviewer"
    recovery["plan"]["changes"][1] = recovery["plan"]["changes"][0]
    refresh(recovery)
    with pytest.raises(ValueError, match="Duplicate or unknown"):
        recover_source_cache(**kwargs(recovery))


def test_extra_inventory_mutation_rejected_even_with_new_corrected_hash(recovery):
    inventory = pd.read_parquet(recovery["corrected"])
    inventory.loc[5, "sha256"] = "d" * 64
    inventory.to_parquet(recovery["corrected"], index=False)
    refresh(recovery, inventory=True)
    with pytest.raises(ValueError, match="beyond the exact two"):
        recover_source_cache(**kwargs(recovery))


@pytest.mark.parametrize("tree", ["official", "effective"])
def test_authoritative_or_effective_source_hash_mismatch_rejected(recovery, tree):
    path = (
        recovery["artifact_root"] / recovery["plan"]["changes"][0]["official_local_path"]
        if tree == "official"
        else next(recovery["official_source_root"].rglob("*.dcm"))
    )
    content = bytearray(path.read_bytes())
    content[-1] ^= 1
    path.write_bytes(content)
    with pytest.raises(ValueError, match="changed source/cache/evidence"):
        recover_source_cache(**kwargs(recovery))
    assert not recovery["output"].exists()


def test_wrong_dicom_uid_rejected_even_when_source_hashes_are_rebound(recovery):
    change = recovery["plan"]["changes"][0]
    path = recovery["artifact_root"] / change["official_local_path"]
    ds = pydicom.dcmread(path)
    ds.StudyInstanceUID = "9.9"
    ds.save_as(path, enforce_file_format=True)
    change["after"].update(sha256=sha(path), bytes=path.stat().st_size)
    change["official_crc32"] = f"{zlib.crc32(path.read_bytes()) & 0xFFFFFFFF:08x}"
    refresh(recovery)
    with pytest.raises(ValueError, match="incorrect DICOM UIDs"):
        recover_source_cache(**kwargs(recovery))


def test_missing_effective_slice_and_extra_membership_rejected(recovery):
    path = next(recovery["official_source_root"].rglob("*.dcm"))
    content = path.read_bytes()
    path.unlink()
    with pytest.raises(ValueError, match="exactly cover"):
        recover_source_cache(**kwargs(recovery))
    path.write_bytes(content)
    path.with_name("extra.dcm").write_bytes(content)
    with pytest.raises(ValueError, match="exactly cover"):
        recover_source_cache(**kwargs(recovery))


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_revision", "b" * 40),
        ("slice_sha256", ["e" * 64, "e" * 64]),
        ("sop_uids", ["9.9.1", "9.9.2"]),
        ("path_fingerprint", "e" * 64),
        ("study_uid", "9.9"),
    ],
)
def test_reuse_rejects_changed_cache_source_metadata(recovery, field, value):
    path = next(recovery["baseline_cache"].rglob("*.json"))
    metadata = json.loads(path.read_text())
    metadata[field] = value
    path.write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        recover_source_cache(**kwargs(recovery))
    assert not recovery["output"].exists()


def test_same_size_cache_corruption_is_visible_partial_failure(recovery):
    path = next(recovery["baseline_cache"].rglob("*.npz"))
    data = bytearray(path.read_bytes())
    data[-1] ^= 1
    path.write_bytes(data)
    with pytest.raises(ValueError, match="changed source/cache/evidence"):
        recover_source_cache(**kwargs(recovery))
    status = json.loads((recovery["output"] / "source_recovery_status.json").read_text())
    assert status["status"] == "failed_partial_output_preserved"
    assert status["n_reused_series"] == 0 and status["training_ready"] is False


def test_changed_series_is_freshly_decoded_even_if_baseline_has_stale_cache(recovery):
    changed = recovery["plan"]["changes"][0]
    frame = pd.read_parquet(recovery["corrected"])
    paths = frame.loc[frame.series_uid == changed["series_uid"], "path"].tolist()
    old_meta = cache_series(
        Source(recovery["official_source_root"]), paths, recovery["baseline_cache"], 32, "a" * 40
    )
    old = recovery["baseline_cache"] / old_meta["cache_path"]
    old.write_bytes(b"stale cache deliberately unusable")
    original_hash = sha(old)
    result = recover_source_cache(**kwargs(recovery))
    new = recovery["output"] / "cache" / old_meta["cache_path"]
    assert result["n_fresh_series"] == 2 and sha(old) == original_hash
    assert new.stat().st_ino != old.stat().st_ino and sha(new) != sha(old)


def test_decode_failure_preserves_baseline_and_partial_status(recovery, monkeypatch):
    old = next(recovery["baseline_cache"].rglob("*.json"))
    original = sha(old)

    def fail(*args, **kwargs):
        raise ValueError("fresh decode failed")

    monkeypatch.setattr(source_recovery, "cache_series", fail)
    with pytest.raises(ValueError, match="fresh decode failed"):
        recover_source_cache(**kwargs(recovery))
    status = json.loads((recovery["output"] / "source_recovery_status.json").read_text())
    assert status["status"] == "failed_partial_output_preserved"
    assert status["n_reused_series"] == 1 and status["n_fresh_series"] == 0
    assert sha(old) == original
    with pytest.raises(ValueError, match="new directory"):
        recover_source_cache(**kwargs(recovery))


def test_existing_or_protected_output_is_rejected(recovery):
    recovery["output"].mkdir()
    with pytest.raises(ValueError, match="new directory"):
        recover_source_cache(**kwargs(recovery))
    recovery["output"] = recovery["baseline_cache"] / "nested"
    with pytest.raises(ValueError, match="protected"):
        recover_source_cache(**kwargs(recovery))


def test_missing_unchanged_cache_or_orphan_rejected(recovery):
    path = next(recovery["baseline_cache"].rglob("*.npz"))
    path.unlink()
    with pytest.raises(ValueError, match="orphan, extra or missing"):
        recover_source_cache(**kwargs(recovery))
    path.with_suffix(".json").unlink()
    with pytest.raises(ValueError, match="orphan, extra or missing"):
        recover_source_cache(**kwargs(recovery))


def test_preflight_validates_without_creating_output(recovery):
    baseline, corrected, _, receipt, _, affected = validate_source_correction(
        **{
            key: recovery[key]
            for key in [
                "baseline_inventory",
                "patch_path",
                "receipt_path",
                "artifact_root",
                "official_source_root",
            ]
        }
    )
    assert len(baseline) == len(corrected) == 6 and len(affected) == 2
    assert receipt["corrected_source_revision"] == "c" * 40
    assert not recovery["output"].exists()


def test_reserve_failure_does_not_create_output(recovery, monkeypatch):
    usage = shutil.disk_usage(recovery["artifact_root"])
    monkeypatch.setattr(shutil, "disk_usage", lambda path: type(usage)(usage.total, usage.total, 0))
    with pytest.raises(OSError, match="free space"):
        recover_source_cache(**kwargs(recovery))
    assert not recovery["output"].exists()
