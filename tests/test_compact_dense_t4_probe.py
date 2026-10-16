import json

import pytest
import torch

from rsna.competitive.checkpoint import file_sha256
from rsna.competitive.compact_dense_infer import POLICY
from scripts.kaggle.compact_dense_t4_probe import KIND, admit_component, validate_fixture


def fixture():
    return {
        "fixture_identity": "engineering-dense30-v1",
        "windows": torch.zeros(4, 3, 384, 384),
        "original_window_indices": torch.tensor([0, 2], dtype=torch.int64),
        "inference_policy": dict(POLICY),
    }


def test_only_anonymous_native_float32_windows_with_original_subset_allowed():
    v = fixture()
    w, i = validate_fixture(v)
    assert len(w) == 4 and i.tolist() == [0, 2]
    v["original_study_uid"] = "not-allowed"
    with pytest.raises(ValueError, match="anonymous"):
        validate_fixture(v)


@pytest.mark.parametrize(
    "bad",
    ["dtype", "nan", "bounds", "duplicate", "reverse", "out-of-range", "extra", "wrong_policy"],
)
def test_invalid_or_changed_fixtures_fail_before_GPU(bad):
    v = fixture()
    if bad == "dtype":
        v["windows"] = v["windows"].half()
    elif bad == "nan":
        v["windows"][0, 0, 0, 0] = float("nan")
    elif bad == "bounds":
        v["windows"][0, 0, 0, 0] = 1.01
    elif bad == "duplicate":
        v["original_window_indices"] = torch.tensor([0, 0])
    elif bad == "reverse":
        v["original_window_indices"] = torch.tensor([2, 0])
    elif bad == "out-of-range":
        v["original_window_indices"] = torch.tensor([0, 4])
    elif bad == "extra":
        v["labels"] = torch.zeros(12)
    else:
        v["inference_policy"] = {"arbitrary": "different"}
    with pytest.raises(ValueError):
        validate_fixture(v)


def test_diagnostic_payload_identity_is_not_production_admission(tmp_path):
    names = [
        "checkpoint.pt",
        "snapshot.json",
        "dense-fixture.pt",
        "reference.json",
        "provenance.json",
        "Apache-2.0-LICENSE",
        "initialization-model-card.md",
    ]
    for n in names:
        (tmp_path / n).write_bytes(b"fixture")
    chk = file_sha256(tmp_path / "checkpoint.pt")
    fix = file_sha256(tmp_path / "dense-fixture.pt")
    (tmp_path / "snapshot.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": chk,
                "epoch": 2,
                "optimizer_steps": 4,
                "binding": {"diagnostic_only": True},
            }
        )
    )
    (tmp_path / "reference.json").write_text(
        json.dumps(
            {
                "checkpoint_sha256": chk,
                "fixture_sha256": fix,
                "model_training_source_git": "3803b717b69db6c5e89e11724c4daee15490cbe2",
            }
        )
    )
    value = {
        "kind": KIND,
        "diagnostic_only": True,
        "competition_eligible": False,
        "inference_policy": POLICY,
        "files": {
            n: {"bytes": (tmp_path / n).stat().st_size, "sha256": file_sha256(tmp_path / n)}
            for n in names
        },
    }
    path = tmp_path / "component-manifest.json"
    path.write_text(json.dumps(value))
    assert admit_component(tmp_path, file_sha256(path))["checkpoint_sha256"] == chk
    (tmp_path / "dense-fixture.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="bytes"):
        admit_component(tmp_path, file_sha256(path))
