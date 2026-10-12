"""Private intake refuses scope drift, unsafe members and malformed pixel envelopes."""

import copy
import struct

import pytest

from scripts.newton.inspect_kneecot_private import REPO, REVISION, nifti_header, validate_manifest


def manifest():
    rows = []
    for n in (1, 2):
        base = f"GJB{n:07d}T"
        for suffix in (".json", "_01.nii"):
            rows.append(
                {
                    "path": f"train_data/{base}/{base}{suffix}",
                    "bytes": 384,
                    "blob_id": "a" * 40,
                    "sha256": "b" * 64,
                }
            )
    return {
        "repo": REPO,
        "revision": REVISION,
        "split": "train_data",
        "owner_agreement_confirmed": True,
        "exams": 2,
        "files": rows,
        "source_bytes": 1536,
    }


def header(endian="<"):
    raw = bytearray(352)
    struct.pack_into(endian + "i", raw, 0, 348)
    struct.pack_into(endian + "8h", raw, 40, 3, 4, 4, 2, 1, 1, 1, 1)
    struct.pack_into(endian + "2h", raw, 70, 2, 8)
    struct.pack_into(endian + "f", raw, 108, 352)
    raw[344:348] = b"n+1\x00"
    return raw


def test_complete_bounded_training_source():
    assert validate_manifest(manifest()) == 1536


@pytest.mark.parametrize(
    "field,value",
    [("split", "test_data"), ("owner_agreement_confirmed", False), ("revision", "main")],
)
def test_no_unapproved_or_unpinned_source(field, value):
    value_manifest = manifest()
    value_manifest[field] = value
    with pytest.raises(ValueError):
        validate_manifest(value_manifest)


@pytest.mark.parametrize(
    "name",
    [
        "../report.json",
        "train_data/GJB0000001T/../../evil.nii",
        "test_data/GJB0000001/GJB0000001_01.nii",
        "$RECYCLE.BIN/private",
    ],
)
def test_unrelated_or_escaping_members_rejected(name):
    value = manifest()
    value["files"][0]["path"] = name
    with pytest.raises(ValueError):
        validate_manifest(value)


def test_duplicate_and_incomplete_exam_rejected():
    value = manifest()
    value["files"][1] = copy.deepcopy(value["files"][0])
    with pytest.raises(ValueError):
        validate_manifest(value)


@pytest.mark.parametrize("endian", ["<", ">"])
def test_header_endianness_and_private_free_text_not_exported(endian):
    raw = header(endian)
    raw[148:160] = b"private-text"
    assert nifti_header(raw, 384) == {"shape": [4, 4, 2], "datatype_code": 2, "bitpix": 8}


def test_truncated_pixels_and_unsupported_header_rejected():
    with pytest.raises(ValueError):
        nifti_header(header(), 383)
    raw = header()
    raw[344:348] = b"ni1\x00"
    with pytest.raises(ValueError):
        nifti_header(raw, 384)


def test_byte_cap_cannot_be_forged():
    value = manifest()
    value["files"][0]["bytes"] = 1024**3 + 1
    value["source_bytes"] = sum(r["bytes"] for r in value["files"])
    with pytest.raises(ValueError):
        validate_manifest(value)
