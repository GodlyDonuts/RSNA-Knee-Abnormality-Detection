"""Synthetic source/geometry contract checks, without clinical records."""

import copy
from dataclasses import replace

import numpy as np
import torch
from source_chains import qualify_source_chains

from rsna.competitive.spatial_reader import build_plan


def fixture(shear=0.0):
    selected = list(range(9))
    supports = [selected[j : j + 3] for j in range(7)]
    graph = {
        "slot": 0,
        "series_uid": "synthetic-acquisition",
        "selected_source_indices": selected,
        "supports": supports,
        "centers_mm": [3.0 * (j + 1) for j in range(7)],
        "repeat_edges": [[j, j + 3] for j in range(4)],
        "repeat_gaps_mm": [9.0] * 4,
        "geometry_known": True,
        "native_original_shape": [9, 32, 32],
        "native_resized_shape": [32, 32],
        "native_cache_side": 32,
        "pixel_spacing": [1.0, 1.0],
        "image_orientation_patient": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0],
        "selected_image_positions_patient": [[shear * j, 0.0, 3.0 * j] for j in range(9)],
        "full_frame_letterbox_align_corners_false": True,
        "input_augmentation_or_mirror": False,
    }
    windows = np.array([(0, j) for j in range(7)], dtype=np.int64)
    plan = build_plan([graph], windows, (8, 4, 4), seed=20261017)
    return graph, windows, plan


torch.set_num_threads(1)
graph, windows, plan = fixture()
paths, counts = qualify_source_chains([graph], windows, plan)
assert [x.indices for x in paths] == [(0, 3, 6), (1, 4), (2, 5)]
assert counts["qualified_windows"] == 7 and counts["qualified_repeat_edges"] == 4
assert counts["unsupported_windows"] == 0

# Source gap breaks cause independent chain starts; they cannot be bridged.
split = copy.deepcopy(graph)
split["repeat_edges"] = split["repeat_edges"][1:]
split["repeat_gaps_mm"] = split["repeat_gaps_mm"][1:]
split_plan = build_plan([split], windows, (8, 4, 4), seed=20261017)
paths, counts = qualify_source_chains([split], windows, split_plan)
assert [x.indices for x in paths] == [(0,), (1, 4), (2, 5), (3, 6)]
assert counts["focal_singleton_chains"] == 1

# No transition is manufactured when original geometry has no repeat links.
focal = copy.deepcopy(graph)
focal["repeat_edges"], focal["repeat_gaps_mm"] = [], []
focal_plan = build_plan([focal], windows, (8, 4, 4), seed=20261017)
paths, counts = qualify_source_chains([focal], windows, focal_plan)
assert len(paths) == counts["focal_singleton_chains"] == 7
assert counts["qualified_repeat_edges"] == 0

# Unsupported directed motion is accounted for, never repaired into a pass.
shifted, shifted_windows, shifted_plan = fixture(shear=1.0)
paths, counts = qualify_source_chains([shifted], shifted_windows, shifted_plan)
assert not paths and counts["unsupported_windows"] == 7
assert counts["unsupported_chain_reasons"]["nonreversible_or_invalid_joint_not_supported"] == 3

# Missing original operators are not silently replaced by an identity prior.
empty = replace(plan, groups=tuple(g for g in plan.groups if g.kind != "repeat"))
paths, counts = qualify_source_chains([graph], windows, empty)
assert not paths and counts["unsupported_windows"] == 7

refusals = 0
for field, value in (
    ("repeat_gaps_mm", [0.0] * 4),
    ("repeat_edges", [[0, 3], [1, 4], [2, 5], [0, 3]]),
    ("repeat_edges", [[0, 3], [1, 4], [2, 5], [3, 5]]),
    ("selected_source_indices", [0, 1, 2, 2, 4, 5, 6, 7, 8]),
):
    bad = copy.deepcopy(graph)
    bad[field] = value
    try:
        qualify_source_chains([bad], windows, plan)
    except ValueError:
        refusals += 1
    else:
        raise AssertionError("Invalid source/geometry accepted")
assert refusals == 4
print(
    "Source-chain partition, gap/focal handling, directed/missing-operator accounting and4refusals passed"
)
