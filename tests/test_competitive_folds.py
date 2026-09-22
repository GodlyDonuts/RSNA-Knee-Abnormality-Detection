import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from rsna.competitive.folds import (
    _digest,
    build_folds,
    load_folds,
    validate_overlay,
    write_folds,
)
from rsna.schema import TARGETS


def manifest_fixture():
    rows = []
    for group in range(24):
        for member in range(2):
            row = {
                "study_uid": f"s{group:02}-{member}",
                "leakage_group": f"g{group:02}",
                "split": "train",
            }
            for target in TARGETS:
                row[target] = int(group < 5)
                row[target + "__mask"] = int(group != 23)
            rows.append(row)
    # Nontrain labels deliberately unusable: fold logic must never inspect them.
    row = {"study_uid": "reserved", "leakage_group": "reserved", "split": "weak_validation"}
    for target in TARGETS:
        row[target] = "reserved-value-must-not-be-read"
        row[target + "__mask"] = "reserved-mask-must-not-be-read"
    rows.append(row)
    return pd.DataFrame(rows)


def binding_fixture():
    value = {
        "schema_version": 1,
        "manifest_sha256": "a" * 64,
        "series_sha256": "b" * 64,
        "source_revision": "source",
        "release_receipt_sha256": "c" * 64,
        "semantic_registry_sha256": "d" * 64,
    }
    value["binding_sha256"] = _digest(value)
    return value


def test_deterministic_grouped_rare_coverage_and_row_order():
    manifest = manifest_fixture()
    overlay, summary = build_folds(manifest)
    other, repeat = build_folds(manifest.sample(frac=1, random_state=9))
    pd.testing.assert_frame_equal(overlay, other)
    assert summary == repeat
    assert len(overlay) == 48
    assert overlay.groupby("leakage_group").fold.nunique().eq(1).all()
    assert "reserved" not in set(overlay.study_uid)
    assert all(
        row["positive_components"] >= 1 and row["negative_components"] >= 1
        for row in summary["counts"]
    )
    fracture = [r["positive_components"] for r in summary["counts"] if r["target"] == "Fracture"]
    assert sorted(fracture) == [1, 2, 2]
    assert set(overlay.loc[overlay.leakage_group.eq("g23"), "study_uid"]) == {"s23-0", "s23-1"}


def test_masked_hidden_values_cannot_change_membership_or_fingerprint():
    manifest = manifest_fixture()
    original, summary = build_folds(manifest)
    changed = manifest.copy()
    for target in TARGETS:
        changed.loc[changed[target + "__mask"].eq(0), target] = np.nan
    alternative, other = build_folds(changed)
    pd.testing.assert_frame_equal(original, alternative)
    assert summary == other


def test_full_identity_boundary_checks_without_reserved_label_reads():
    manifest = manifest_fixture()
    manifest.loc[manifest.study_uid.eq("reserved"), "leakage_group"] = "g00"
    with pytest.raises(ValueError, match="crosses accepted split"):
        build_folds(manifest)
    duplicate = pd.concat([manifest_fixture(), manifest_fixture().iloc[:1]])
    with pytest.raises(ValueError, match="duplicate UID"):
        build_folds(duplicate)


def test_write_load_replay_and_no_overwrite(tmp_path):
    manifest, binding = manifest_fixture(), binding_fixture()
    summary = write_folds(tmp_path, manifest, binding)
    loaded = load_folds(tmp_path / "fold.csv", manifest, binding)
    pd.testing.assert_frame_equal(loaded, build_folds(manifest)[0])
    assert summary["source_manifest_sha256"] == binding["manifest_sha256"]
    with pytest.raises(ValueError, match="immutable"):
        write_folds(tmp_path, manifest, binding)


def test_changed_release_binding_rejected(tmp_path):
    manifest, binding = manifest_fixture(), binding_fixture()
    write_folds(tmp_path, manifest, binding)
    changed = dict(binding, manifest_sha256="e" * 64)
    changed["binding_sha256"] = _digest({k: v for k, v in changed.items() if k != "binding_sha256"})
    with pytest.raises(ValueError, match="incompatible release"):
        load_folds(tmp_path, manifest, changed)


@pytest.mark.parametrize("tamper", ["missing", "duplicate", "nontrain", "group", "fold", "score"])
def test_tampering_rejected_even_if_csv_hash_recomputed(tmp_path, tamper):
    manifest, binding = manifest_fixture(), binding_fixture()
    write_folds(tmp_path, manifest, binding)
    frame = pd.read_csv(tmp_path / "fold.csv")
    if tamper == "missing":
        frame = frame.iloc[1:]
    elif tamper == "duplicate":
        frame = pd.concat([frame, frame.iloc[:1]])
    elif tamper == "nontrain":
        frame.loc[0, "study_uid"] = "reserved"
    elif tamper == "group":
        frame.loc[0, "leakage_group"] = "other"
    elif tamper == "fold":
        frame.loc[0, "fold"] = (frame.loc[0, "fold"] + 1) % 3
    else:
        manifest.loc[manifest.study_uid.eq("s00-0"), "Fracture"] = 0
    raw = frame.to_csv(index=False).encode()
    (tmp_path / "fold.csv").write_bytes(raw)
    metadata = json.loads((tmp_path / "folds.json").read_text())
    metadata["fold_csv_sha256"] = hashlib.sha256(raw).hexdigest()
    (tmp_path / "folds.json").write_text(json.dumps(metadata))
    with pytest.raises(ValueError):
        load_folds(tmp_path, manifest, binding)


def test_fewer_rare_components_than_folds_reported_without_fabrication():
    manifest = manifest_fixture()
    manifest.loc[manifest.split.eq("train"), "Fracture"] = 0
    manifest.loc[manifest.study_uid.eq("s00-0"), "Fracture"] = 1
    _, summary = build_folds(manifest)
    assert sorted(
        r["positive_components"] for r in summary["counts"] if r["target"] == "Fracture"
    ) == [0, 0, 1]


def test_fractional_fold_and_mixed_group_rejected():
    manifest = manifest_fixture()
    overlay, _ = build_folds(manifest)
    overlay["fold"] = overlay.fold.astype(float)
    with pytest.raises(ValueError, match="integer"):
        validate_overlay(manifest, overlay)
