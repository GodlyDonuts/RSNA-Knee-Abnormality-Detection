import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file
from torch import nn

from rsna.competitive.compact_bag import CompactBagClassifier
from rsna.competitive.compact_train import Settings, fit, predict

PATH = Path(__file__).resolve().parents[1] / "scripts/newton/run_compact_bag_recovery.py"
spec = importlib.util.spec_from_file_location("compact_recovery", PATH)
recovery = importlib.util.module_from_spec(spec)
spec.loader.exec_module(recovery)


def test_complete_decoder_refuses_scalar_types_and_tensor_changes():
    recovery.exact({"x": [torch.ones(3), (1, True, None)]}, {"x": [torch.ones(3), (1, True, None)]})
    for other in [torch.tensor([1.0, 1.0, 1.000001]), torch.ones(3, dtype=torch.float64)]:
        with pytest.raises(ValueError):
            recovery.exact(torch.ones(3), other)
    with pytest.raises(ValueError):
        recovery.exact(True, 1)


def test_actual_numerical_fixture_checks_all_bn_and_prediction_invariants(tmp_path):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:

        class Encoder(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 4, 1)
                self.norms = nn.Sequential(*(nn.BatchNorm2d(4) for _ in range(25)))

            def forward(self, x):
                return self.norms(self.conv(x)).mean((2, 3))

        torch.manual_seed(3)
        model = CompactBagClassifier(Encoder(), dimension=4)
        weights = tmp_path / "synthetic-source.safetensors"
        save_file(model.encoder.state_dict(), weights)
        data = []
        for i in range(16):
            data.append(
                {
                    "study_uid": f"fixture-{i}",
                    "images": torch.rand(2, 16, 1, 4, 4),
                    "series_mask": torch.ones(2, dtype=torch.bool),
                    "slice_mask": torch.ones(2, 16, dtype=torch.bool),
                    "labels": torch.rand(12),
                    "label_mask": torch.ones(12),
                }
            )
        binding = {
            "diagnostic_only": True,
            "outer_uids_sha256": hashlib.sha256(
                json.dumps(sorted(row["study_uid"] for row in data)).encode()
            ).hexdigest(),
        }
        full, resumed = tmp_path / "full", tmp_path / "resumed"
        fit(model, data, full, binding, Settings(epochs=2), device=torch.device("cpu"))
        predict(model, data, torch.device("cpu")).to_csv(
            full / "outer_predictions.csv", index=False
        )
        shutil.copytree(full, resumed)
        result = recovery.compare(full, resumed, weights)
        assert result["BN_counters_checked"] == 25 and result["fitting_bag_forwards"] == 32
        original = torch.load(full / "checkpoint.pt", weights_only=True)
        changed = dict(original)
        changed["model"] = dict(original["model"])
        changed["model"]["encoder.norms.0.num_batches_tracked"] = torch.tensor(33)
        torch.save(changed, resumed / "checkpoint.pt")
        with pytest.raises(ValueError, match="tensor differs"):
            recovery.compare(full, resumed, weights)
        torch.save(original, resumed / "checkpoint.pt")
        p = resumed / "outer_predictions.csv"
        p.write_text(p.read_text() + "\n")
        with pytest.raises(ValueError, match="CSV bytes"):
            recovery.compare(full, resumed, weights)
    finally:
        torch.set_num_threads(old_threads)
