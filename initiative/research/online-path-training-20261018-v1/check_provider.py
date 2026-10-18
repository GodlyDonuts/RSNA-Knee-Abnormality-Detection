"""Test supervision glue, not synthetic MRI accuracy."""

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

D = Path(__file__).resolve().parent
sys.path.insert(0, str(D.parent / "connected-witness-paths-20261017-v1"))
spec = importlib.util.spec_from_file_location(
    "integrated_dense_provider", D / "integrated_provider.py"
)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


@pytest.fixture
def setup(tmp_path, monkeypatch):
    original = {
        "checkpoint": str(tmp_path / "parent/epoch-checkpoints/e.pt"),
        "series_path": str(tmp_path / "series.csv"),
        "native_cache": str(tmp_path),
    }
    (tmp_path / "admitted.json").write_text(json.dumps(original))
    (tmp_path / "parent").mkdir()
    (tmp_path / "parent/run.json").write_text(
        json.dumps({"binding": {"release": {"series_sha256": "exact"}}})
    )
    pd.DataFrame({"dummy": [1]}).to_csv(original["series_path"], index=False)
    sources = {}
    for role in ("fitting", "outer"):
        labels = (
            np.ones((16, 12), np.float32) if role == "fitting" else np.zeros((16, 12), np.float32)
        )
        masks = np.ones((16, 12), bool) if role == "fitting" else np.zeros((16, 12), bool)

        class Source:
            def __len__(self):
                return 16

            def __getitem__(self, i):
                return {
                    "study_uid": self.records[i]["study_uid"],
                    "maps": torch.ones(2, 4),
                    "base_logits": torch.ones(12),
                    "labels": torch.from_numpy(self.labels[i]),
                    "label_mask": torch.from_numpy(self.masks[i]),
                }

        source = Source()
        source.records = [{"study_uid": f"synthetic-{role}-{i}"} for i in range(16)]
        source.labels, source.masks, source.binding = labels, masks, {"role": role}
        sources[role] = source

    class Dataset:
        def __init__(self, frame, *args, **kwargs):
            self.frame = frame

    class Stream:
        def __init__(self, dataset, parent, binding, *, sampling):
            self.dataset, self.sampling = dataset, sampling
            self.role = binding["role"]

        def __getitem__(self, i):
            return sources[self.role][i]

    monkeypatch.setattr(m, "CompetitiveStudyDataset", Dataset)
    monkeypatch.setattr(m, "StreamedFP32PathData", Stream)
    monkeypatch.setattr(
        m, "load_model", lambda *a, **kw: ("frozen", {"epoch": 16, "optimizer_steps": 4496})
    )
    return sources, SimpleNamespace(sha=lambda _: "exact"), tmp_path, Stream


def test_exact_source_rows_and_dense_policy(setup):
    sources, helper, root, _ = setup
    data, parent = m.providers(sources, helper, root)
    assert parent == "frozen"
    for role in sources:
        assert data[role].sampling == "dense"
        frame = data[role].dataset.frame
        assert frame.study_uid.tolist() == [r["study_uid"] for r in sources[role].records]
        assert np.array_equal(frame[list(m.TARGETS)].values, sources[role].labels)
        assert np.array_equal(frame[[t + "__mask" for t in m.TARGETS]].values, sources[role].masks)


@pytest.mark.parametrize("failure", ["membership", "outer", "series", "policy", "maps", "identity"])
def test_refuses_glue_drift(setup, monkeypatch, failure):
    sources, helper, root, Stream = setup
    if failure == "membership":
        del sources["outer"]
    elif failure == "outer":
        sources["outer"].masks[0, 0] = True
    elif failure == "series":
        helper.sha = lambda _: "changed"
    elif failure == "policy":
        monkeypatch.setattr(
            m, "load_model", lambda *a, **kw: (None, {"epoch": 15, "optimizer_steps": 4496})
        )
    else:
        previous = Stream.__getitem__

        def wrong(self, i):
            result = dict(previous(self, i))
            result["maps" if failure == "maps" else "study_uid"] = (
                torch.zeros(2, 4) if failure == "maps" else "different"
            )
            return result

        monkeypatch.setattr(Stream, "__getitem__", wrong)
    with pytest.raises(ValueError):
        m.providers(sources, helper, root)
