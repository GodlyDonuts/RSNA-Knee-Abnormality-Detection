import copy
import hashlib
import json

import pytest

from rsna.competitive import licensed_parent_population as population


@pytest.fixture
def originals(monkeypatch):
    uids = [f"synthetic-{i:04d}" for i in range(3887)]
    uid_sha = hashlib.sha256(json.dumps(uids, separators=(",", ":")).encode()).hexdigest()
    monkeypatch.setattr(population, "POPULATION_SHA", uid_sha)
    probe_rows = [
        {"study_uid": u, "file": hashlib.sha256(u.encode()).hexdigest() + ".npz", "bytes": 1}
        for u in uids[:32]
    ]
    probe = {
        "records": probe_rows,
        "parent_source_sha256": population.PARENT_SOURCE_SHA,
        "codec_sha256": population.CODEC_SHA,
    }
    proofs, shards = [], []
    for i in range(4):
        lo, hi = i * 972, min(3887, (i + 1) * 972)
        count = 32 if i == 0 else 0
        row = {
            "status": "actual_full_pixel_shard_collected_verified",
            "source_sha256": population.SOURCE_SHAS[i],
            "shard": i,
            "studies": hi - lo,
            "new_cases": hi - lo - count,
            "reused_original32": count,
            "global_uid_sha256": uid_sha,
            "empty_studies": 0,
            "decode_errors": 0,
            "all_new_member_pixel_mask_and_both94_input_view_hashes_exact": True,
            "all_fitting_membership_exact": True,
            "original32_retained_no_archive_transfer_or_reencoding": True,
            "index_bytes": 1024,
            "archive_bytes": 2048,
            "index_sha256": "1" * 64,
            "archive_sha256": "2" * 64,
        }
        row["original_Kaggle_receipt"] = {
            "status": "actual_licensed_parent_full_fitting_pixel_shard_complete",
            "shard": i,
            "lo": lo,
            "hi": hi,
            "parent_source_sha256": population.PARENT_SOURCE_SHA,
            "codec_sha256": population.CODEC_SHA,
            "global_uid_sha256": uid_sha,
            "original_full_index_sha256": population.FULL_INDEX_SHA,
            "original_probe_pixel_index_sha256": population.PROBE_SHA,
            "empty_studies": 0,
            "decode_errors": 0,
            **{
                k: row[k]
                for k in ("index_bytes", "archive_bytes", "index_sha256", "archive_sha256")
            },
        }
        proofs.append(row)
        records = [
            {
                **(
                    probe_rows[j]
                    if j < 32
                    else {
                        "study_uid": uids[j],
                        "file": hashlib.sha256(uids[j].encode()).hexdigest() + ".npz",
                        "bytes": 1,
                    }
                ),
                "source_key": "probe32" if j < 32 else "self",
                "has_original_MRI_pixels": True,
            }
            for j in range(lo, hi)
        ]
        shards.append(
            {
                "records": records,
                "shard": i,
                "lo": lo,
                "hi": hi,
                "global_uid_sha256": uid_sha,
                "parent_source_sha256": population.PARENT_SOURCE_SHA,
                "codec_sha256": population.CODEC_SHA,
                "original_full_feature_index_sha256": population.FULL_INDEX_SHA,
                "original_probe_pixel_index_sha256": population.PROBE_SHA,
            }
        )
    return probe, shards, proofs


def test_original32_are_reused_once_and_complete_new_roster_gets_exact_holders(originals):
    probe, shards, proofs = originals
    rows = population.join_records(probe, shards, proofs)
    assert len(rows) == 3887
    assert [r["holder"] for r in rows[:32]] == ["probe32"] * 32
    assert [r["holder"] for r in rows[32:972]] == ["s0"] * 940
    assert rows[972]["holder"] == "s1" and rows[-1]["holder"] == "s3"
    assert "holder" not in shards[0]["records"][0]


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "uncollected",
        "source",
        "error",
        "empty",
        "unverified",
        "digest",
        "oversize",
        "wrong_count",
        "receipt",
    ],
)
def test_full_fitting_cannot_admit_pending_dirty_or_unverified_sources(originals, change):
    _, _, proofs = originals
    if change == "missing":
        proofs.pop()
    elif change == "uncollected":
        proofs[0]["status"] = "RUNNING"
    elif change == "source":
        proofs[0]["source_sha256"] = "0" * 64
    elif change == "error":
        proofs[0]["decode_errors"] = 1
    elif change == "empty":
        proofs[0]["empty_studies"] = 1
    elif change == "unverified":
        proofs[0]["all_fitting_membership_exact"] = False
    elif change == "digest":
        proofs[0]["index_sha256"] = "1" * 63
    elif change == "oversize":
        proofs[0]["archive_bytes"] = 15 * 2**30
    elif change == "wrong_count":
        proofs[0]["new_cases"] -= 1
    else:
        proofs[0]["original_Kaggle_receipt"]["lo"] = 1
    with pytest.raises(ValueError):
        population.validate_proofs(proofs)


@pytest.mark.parametrize(
    "change", ["range", "reused_bytes", "holder", "empty", "failure", "order", "new_identity"]
)
def test_records_cannot_change_original_membership_pixels_or_reuse(originals, change):
    probe, shards, proofs = originals
    if change == "range":
        shards[1]["lo"] += 1
    elif change == "reused_bytes":
        shards[0]["records"][0]["bytes"] += 1
    elif change == "holder":
        shards[0]["records"][0]["source_key"] = "self"
    elif change == "empty":
        shards[1]["records"][0]["has_original_MRI_pixels"] = False
    elif change == "failure":
        shards[1]["records"][0]["preprocessing_error_type"] = "DecodeError"
    elif change == "order":
        shards[1]["records"].reverse()
    else:
        shards[1]["records"][0]["study_uid"] = "unregistered"
    with pytest.raises(ValueError):
        population.join_records(probe, shards, proofs)


def test_locator_prunes_raw_competition_tree_and_rejects_ambiguous_exact_outputs(tmp_path):
    digest = hashlib.sha256(b"bound").hexdigest()
    raw = tmp_path / "competition"
    raw.mkdir()
    (raw / "train.csv").write_text("ignored")
    (raw / "index.private.json").write_bytes(b"bound")
    with pytest.raises(ValueError):
        population.locate(tmp_path, "index.private.json", 5, digest)
    output = tmp_path / "original"
    output.mkdir()
    p = output / "index.private.json"
    p.write_bytes(b"bound")
    assert population.locate(tmp_path, "index.private.json", 5, digest) == p
    other = tmp_path / "duplicate"
    other.mkdir()
    (other / p.name).write_bytes(b"bound")
    with pytest.raises(ValueError):
        population.locate(tmp_path, "index.private.json", 5, digest)


def test_input_receipt_dictionaries_are_not_mutated(originals):
    before = copy.deepcopy(originals)
    population.join_records(*originals)
    assert before == originals


def test_original_Kaggle_admission_does_not_claim_Newton_replica_exists(originals):
    _, _, proofs = originals
    accepted = []
    for i, row in enumerate(proofs):
        receipt = {
            **row["original_Kaggle_receipt"],
            "studies": row["studies"],
            "new_original_MRI_decodes": row["new_cases"],
            "original32_reused": row["reused_original32"],
            "all_new_original_volume_mask_and_both_view_tensors_bit_equal_after_roundtrip": True,
            "original32_reused_without_decode_reencode_or_new_archive_copy": True,
            "reports_clinical_labels_protected_values_read": False,
            "new_weight_download_or_encoder_optimization": False,
        }
        actual = {
            "status": "COMPLETE",
            "version": 1,
            "shard": i,
            "kernel_id": population.KERNEL_IDS[i],
            "source_sha256": population.SOURCE_SHAS[i],
            "receipt": receipt,
        }
        p = population.producer_proof(actual)
        assert p["Newton_replica_collection_or_independent_replay_claim"] is False
        assert p["status"] == "actual_original_Kaggle_pixel_shard_verified"
        accepted.append(p)
        actual["status"] = "RUNNING"
        with pytest.raises(ValueError):
            population.producer_proof(actual)
    population.validate_proofs(accepted)
