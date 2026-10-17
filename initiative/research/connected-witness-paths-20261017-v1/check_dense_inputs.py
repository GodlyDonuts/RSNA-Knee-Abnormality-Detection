"""Synthetic coverage, legacy tensor equivalence, chunking and geometry checks."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from dense_inputs import source_plan, tensor_chunks
from variable_source_chains import qualify_variable_chains

from rsna.competitive.neighbor_input import sample_neighbor_images
from rsna.competitive.spatial_reader import build_plan

torch.set_num_threads(1)


def metadata(count):
    return {
        "series_uid": "synthetic-only",
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


checks = 0
generator = np.random.default_rng(1723)
for count in (1, 2, 3, 4, 9, 16, 17, 37, 91):
    meta = metadata(count)
    volume = generator.integers(0, 65536, (count, 16, 16), dtype=np.uint16)
    for sampling in ("sparse16", "dense"):
        graph, windows = source_plan(meta, sampling=sampling, slot=2, metadata_sha256="0" * 64)
        selected = np.asarray(graph["selected_source_indices"])
        for side in (7, 16, 23):
            gray = sample_neighbor_images(volume, meta, selected, side, None)[:, 0]
            reference = (
                torch.stack([gray[j : j + 3] for j in range(len(selected) - 2)])
                if len(selected) > 2
                else torch.empty(0, 3, side, side)
            )
            for chunk in (1, 4, 12):
                parts = list(
                    tensor_chunks(volume, meta, sampling=sampling, side=side, chunk_size=chunk)
                )
                actual = torch.cat([p[2] for p in parts]) if parts else reference
                assert torch.equal(actual, reference), (count, sampling, side, chunk)
                supports = [row for _, rows, _ in parts for row in rows.tolist()]
                assert supports == graph["supports"]
                assert [first for first, _, _ in parts] == list(range(0, len(windows), chunk))
                assert all(len(images) <= chunk for _, _, images in parts)
                checks += 1
        if len(windows):
            plan = build_plan([graph], windows, (8, 4, 4), seed=20261017)
            paths, counts = qualify_variable_chains([graph], windows, plan)
            assert counts["qualified_windows"] == len(windows) and not counts["unsupported_windows"]
            assert sum(len(path.indices) for path in paths) == len(windows)

# Large native discontinuity must break path links, while every image stays.
meta = metadata(37)
for j in range(19, 37):
    meta["positions"][j][2] += 100
graph, windows = source_plan(meta, sampling="dense", slot=0, metadata_sha256="a" * 64)
assert len(windows) == 35
for field in ("continuity_edges", "repeat_edges"):
    for left, right in graph[field]:
        centers = [graph["supports"][k][1] for k in (left, right)]
        assert not centers[0] < 19 <= centers[1]

# Short inputs reproduce exact sparse windows; >16 includes changed context.
for count in (3, 16, 17, 37):
    graphs = [
        source_plan(metadata(count), sampling=s, slot=0, metadata_sha256="b" * 64)[0]
        for s in ("sparse16", "dense")
    ]
    assert (graphs[0]["supports"] == graphs[1]["supports"]) == (count <= 16)

refused = 0
for mutate in (
    "float_volume",
    "wrong_shape",
    "zero_chunk",
    "bad_sampling",
    "duplicate_SOP",
    "bad_order",
):
    meta = metadata(9)
    volume = np.zeros((9, 16, 16), dtype=np.uint16)
    options = {"sampling": "dense", "chunk_size": 4}
    if mutate == "float_volume":
        volume = volume.astype(np.float32)
    elif mutate == "wrong_shape":
        volume = volume[:-1]
    elif mutate == "zero_chunk":
        options["chunk_size"] = 0
    elif mutate == "bad_sampling":
        options["sampling"] = "unspecified"
    elif mutate == "duplicate_SOP":
        meta["sop_uids"][1] = meta["sop_uids"][0]
    else:
        meta["positions"][1] = copy.deepcopy(meta["positions"][0])
    try:
        if mutate in {"duplicate_SOP", "bad_order"}:
            source_plan(meta, sampling="dense", slot=0, metadata_sha256="c" * 64)
        else:
            list(tensor_chunks(volume, meta, **options))
    except ValueError:
        refused += 1
    else:
        raise AssertionError(mutate)
assert refused == 6
print(
    json.dumps(
        {
            "status": "synthetic_sparse_dense_original_normalization_chunk_and_source_checks_passed",
            "source_sha256": hashlib.sha256(
                (Path(__file__).parent / "dense_inputs.py").read_bytes()
            ).hexdigest(),
            "exact_chunk_reference_checks": checks,
            "source_gap_keeps_images_and_breaks_physical_edges": True,
            "variable_path_accounting_all_synthetic_windows": True,
            "refusals": refused,
            "actual_MRI_or_encoder_or_accuracy_acceptance": False,
        },
        indent=2,
    )
)
