"""Admission/source drift refusal for a real weighted-engine GPU check."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.newton.run_compact_weighted_recovery import validate_admission, verify_source

ROOT = Path(__file__).resolve().parents[1]


def admission():
    return json.loads(
        (
            ROOT / "initiative/audits/compact-positive-weight-CPU-20261016-v1/accepted.json"
        ).read_text()
    )


def test_actual_full_population_receipt_required():
    validate_admission(admission())
    for field, value in [
        ("source_git", "0" * 40),
        ("fitting_count", 16),
        ("outer_count", 1301),
        ("source_targets_masks_and_folds_unchanged", False),
        ("engineering16_uses_identical_full2247_weights", False),
        ("encoder_optimizer_or_GPU_run", True),
        ("images_decoded", 16),
        ("engine", "native_coatnet_bag12_N3_full_v1"),
    ]:
        with pytest.raises(ValueError):
            validate_admission({**admission(), field: value})
    bad = copy.deepcopy(admission())
    bad["policy"]["positive_weights"][0] += 1
    with pytest.raises(ValueError):
        validate_admission(bad)
    bad = copy.deepcopy(admission())
    bad["policy"]["fitting_count"] = 16
    with pytest.raises(ValueError):
        validate_admission(bad)


def test_source_manifest_refuses_modified_missing_extra_and_linked_modules(tmp_path):
    root = tmp_path / "rsna"
    root.mkdir()
    module = root / "numerical.py"
    original = b"constant = 1\n"
    module.write_bytes(original)
    manifest = {
        "source_git": "f5af4f238929652f9f482c3741ee09d799fa9cc2",
        "archive_sha256": "31be7ae74691682733b1055b01b3041b29d7e4ab147a2b05907632888b994b07",
        "files": {"rsna/numerical.py": hashlib.sha256(original).hexdigest()},
    }
    verify_source(tmp_path, manifest)
    module.write_bytes(b"constant = 2\n")
    with pytest.raises(ValueError):
        verify_source(tmp_path, manifest)
    module.unlink()
    with pytest.raises(ValueError):
        verify_source(tmp_path, manifest)
    module.write_bytes(original)
    extra = root / "unexpected.py"
    extra.write_text("")
    with pytest.raises(ValueError):
        verify_source(tmp_path, manifest)
    extra.unlink()
    extra.symlink_to(module)
    with pytest.raises(ValueError):
        verify_source(tmp_path, manifest)
    extra.unlink()
    extra.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError):
        verify_source(tmp_path, manifest)
