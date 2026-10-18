"""New packaging regression: identical members still need canonical ZIP order."""

import hashlib
import zipfile

import pytest

from scripts.kaggle.publication import validate_zip


def test_private_correction_transport_preserves_original_but_uses_sorted_members(tmp_path):
    payloads = {
        "corrections.private.npz": b"unchanged fitting-only table",
        "construction-receipt.json": b"{}",
        "bundle-manifest.json": b"{}",
    }
    files = {
        n: {"bytes": len(b), "sha256": hashlib.sha256(b).hexdigest()} for n, b in payloads.items()
    }
    original = tmp_path / "original.zip"
    with zipfile.ZipFile(original, "x", compression=zipfile.ZIP_STORED) as archive:
        for n, b in payloads.items():
            archive.writestr(n, b)
    original_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="ZIP membership changed"):
        validate_zip(original, files)
    corrected = tmp_path / "corrected.zip"
    with zipfile.ZipFile(corrected, "x", compression=zipfile.ZIP_STORED) as archive:
        for n, b in sorted(payloads.items()):
            archive.writestr(n, b)
    validate_zip(corrected, files)
    assert hashlib.sha256(original.read_bytes()).hexdigest() == original_sha
