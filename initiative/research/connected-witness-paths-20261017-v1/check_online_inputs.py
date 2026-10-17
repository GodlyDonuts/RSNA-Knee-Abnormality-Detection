"""Synthetic streamed encoder integration, not actual MRI/GPU admission."""

import copy
import hashlib
import io
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from online_inputs import StreamedPathData, rechunk
from torch import nn

from rsna.competitive.compact_bag import CompactBagClassifier
from rsna.competitive.neighbor_input import sample_neighbor_images
from rsna.competitive.spatial_data import OnlineSpatialData
from rsna.data import select_series

torch.set_num_threads(1)


def metadata(count):
    return {
        "n_slices": count,
        "side": 16,
        "ordering": "patient_geometry",
        "sop_uids": [f"synthetic-sop-{j}" for j in range(count)],
        "positions": [[0, 0, 3 * j] for j in range(count)],
        "orientation": [1, 0, 0, 0, 1, 0],
        "original_shape": [count, 16, 16],
        "resized_shape": [16, 16],
        "pixel_spacing": [1, 1],
    }


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 8, 1)
        self.reduce = nn.AdaptiveAvgPool2d(4)
        self.head = nn.Module()
        self.head.global_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, images):
        return self.head.global_pool(self.conv(self.reduce(images))).flatten(1)


class Fixture:
    augment, slices, side, max_series = False, 16, 384, 6
    slice_sampling, series_sampling = "uniform", "selected"
    neighbor_channels = cache_contrast = volume_provider = None
    input_channels = 1
    raw_split = "train_series"

    def __init__(self, root):
        self.cache = root
        self.studies = pd.DataFrame({"study_uid": ["synthetic-study"]})
        self.series, self._metadata, self.volumes = {"synthetic-study": []}, {}, {}
        rng = np.random.default_rng(124)
        for i, count in enumerate((17, 9, 2)):
            sid = f"synthetic-series-{i}"
            meta = metadata(count)
            meta.update(study_uid="synthetic-study", series_uid=sid)
            volume = rng.integers(0, 65536, (count, 16, 16), dtype=np.uint16)
            payload = io.BytesIO()
            np.savez_compressed(payload, volume=volume)
            raw = payload.getvalue()
            meta.update(cache_bytes=len(raw), cache_sha256=hashlib.sha256(raw).hexdigest())
            path = root / "train_series/synthetic-study" / (sid + ".npz")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
            text = json.dumps(meta).encode()
            path.with_suffix(".json").write_bytes(text)
            self._metadata[("synthetic-study", sid)] = (hashlib.sha256(text).hexdigest(), meta)
            self.volumes[sid] = volume
            self.series["synthetic-study"].append(
                {
                    "SeriesInstanceUID": sid,
                    "Fluid_Sensitive": 1,
                    "Fat_Suppression": 0,
                    "Anatomical_Plane": ("Sagittal", "Coronal", "Axial")[i],
                }
            )

    def __len__(self):
        return 1

    def __getitem__(self, index):
        images = torch.zeros(6, 16, 1, 384, 384)
        slice_mask, series_mask = (
            torch.zeros(6, 16, dtype=torch.bool),
            torch.zeros(6, dtype=torch.bool),
        )
        for slot, entry in enumerate(select_series(self.series["synthetic-study"], 6, False)):
            sid = entry["SeriesInstanceUID"]
            _, meta = self._metadata[("synthetic-study", sid)]
            selected = (
                np.linspace(0, meta["n_slices"] - 1, min(16, meta["n_slices"]))
                .round()
                .astype(np.int64)
            )
            images[slot, : len(selected)] = sample_neighbor_images(
                self.volumes[sid], meta, selected, 384, None
            )
            slice_mask[slot, : len(selected)] = True
            series_mask[slot] = True
        return {
            "study_uid": "synthetic-study",
            "images": images,
            "series_mask": series_mask,
            "slice_mask": slice_mask,
            "labels": torch.zeros(12),
            "label_mask": torch.zeros(12),
        }


with tempfile.TemporaryDirectory() as tmp:
    data = Fixture(Path(tmp))
    torch.manual_seed(12)
    parent = CompactBagClassifier(Encoder(), dimension=8).eval().requires_grad_(False)
    original = OnlineSpatialData(
        data, parent, {"synthetic_only": True}, seed=20261017, chunk_size=12
    )
    saved = copy.deepcopy(parent.state_dict())
    expected = original[0]
    sparse = StreamedPathData(data, parent, {"synthetic_only": True}, sampling="sparse16")[0]
    dense_provider = StreamedPathData(data, parent, {"synthetic_only": True}, sampling="dense")
    dense = dense_provider[0]
    assert torch.equal(expected["maps"], sparse["maps"])
    assert expected["plan"].metadata_sha256 == sparse["plan"].source.metadata_sha256
    assert torch.equal(expected["base_logits"], sparse["base_logits"])
    assert torch.equal(dense["base_logits"], sparse["base_logits"])
    assert len(sparse["maps"]) == 21 and len(dense["maps"]) == 22
    assert dense["plan"].accounting["qualified_windows"] == 22
    assert not dense["label_mask"].any() and not dense["maps"].requires_grad
    assert all(torch.equal(saved[k], v) for k, v in parent.state_dict().items())
    assert torch.equal(dense["maps"], dense_provider[0]["maps"])
    parent.train()
    try:
        dense_provider[0]
    except ValueError:
        pass
    else:
        raise AssertionError("Train-mode parent accepted")
    parent.eval()
    path = next((Path(tmp) / "train_series/synthetic-study").glob("*.json"))
    path.write_text(path.read_text() + " ")
    try:
        dense_provider[0]
    except ValueError:
        pass
    else:
        raise AssertionError("Changed metadata accepted")

parts = [torch.arange(n * 3).reshape(n, 3, 1, 1).float() for n in (5, 2, 5, 3)]
assert torch.equal(torch.cat(list(rechunk(parts, 5))), torch.cat(parts))
assert [len(x) for x in rechunk(parts, 5)] == [5, 5, 5]
print(
    json.dumps(
        {
            "status": "synthetic_streamed_frozen_encoder_sparse_reference_and_common_teacher_exact",
            "source_sha256": hashlib.sha256(
                (Path(__file__).parent / "online_inputs.py").read_bytes()
            ).hexdigest(),
            "cross_series_global_chunk_order_exact": True,
            "sparse_full_maps_and_parent_logits_bit_exact": True,
            "same_sparse_teacher_for_dense": True,
            "frozen_parent_all_state_unchanged": True,
            "all_dense_windows_retained": True,
            "train_mode_and_metadata_drift_refused": True,
            "actual_MRI_GPU_runtime_or_training_acceptance": False,
        },
        indent=2,
    )
)
