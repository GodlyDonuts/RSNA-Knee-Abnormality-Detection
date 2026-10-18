"""Compose accepted dense FP32 MRI inputs with original fitting-only supervision."""

import json
from pathlib import Path

import pandas as pd
import torch
from online_inputs_fp32 import StreamedFP32PathData

from rsna.competitive.compact_infer import load_model
from rsna.competitive.data import CompetitiveStudyDataset
from rsna.schema import TARGETS


def providers(cached, helper, root):
    if set(cached) != {"fitting", "outer"} or any(len(v) != 16 for v in cached.values()):
        raise ValueError("Original16 fitting/16 prediction-only engineering inputs required")
    original = json.loads((root / "admitted.json").read_text())
    parent, policy = load_model(
        Path(original["checkpoint"]),
        "5bec977964231fe67009d7106ad63ec71fdbd9cde5ddeb2cc2883ac59b0a7be5",
        precision="float32",
        device=torch.device("cuda"),
    )
    if policy["epoch"] != 16 or policy["optimizer_steps"] != 4496:
        raise ValueError("Original fixed-final N3 reference required")
    parent_binding = json.loads((Path(original["checkpoint"]).parents[1] / "run.json").read_text())[
        "binding"
    ]
    if helper.sha(original["series_path"]) != parent_binding["release"]["series_sha256"]:
        raise ValueError("Original admitted series metadata changed")
    series = pd.read_csv(original["series_path"])
    data = {}
    for role, source in cached.items():
        frame = pd.DataFrame({"study_uid": [r["study_uid"] for r in source.records]})
        if role == "outer" and torch.as_tensor(source.masks).any():
            raise ValueError("Engineering prediction-only targets must remain unavailable")
        for j, target in enumerate(TARGETS):
            frame[target] = source.labels[:, j]
            frame[target + "__mask"] = source.masks[:, j]
        dataset = CompetitiveStudyDataset(
            frame,
            series,
            Path(original["native_cache"]),
            side=384,
            slices=16,
            max_series=6,
            augment=False,
            soft_targets=False,
        )
        sparse = StreamedFP32PathData(dataset, parent, source.binding, sampling="sparse16")
        # Independent prior acceptance covered all32; this checks the new glue on
        # one original fitting and one outer case in each fresh numerical process.
        actual, expected = sparse[0], source[0]
        for key in ("maps", "base_logits", "labels", "label_mask"):
            if not torch.equal(actual[key].cpu(), expected[key].cpu()):
                raise ValueError("Integrated original sparse input/supervision changed: " + key)
        if actual["study_uid"] != expected["study_uid"]:
            raise ValueError("Engineering source membership changed")
        del actual, expected, sparse
        data[role] = StreamedFP32PathData(dataset, parent, source.binding, sampling="dense")
    return data, parent
