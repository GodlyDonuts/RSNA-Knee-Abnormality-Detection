"""Geometry-only complete source-roster priors, never sampled-window priors."""

from __future__ import annotations

import math

import torch

from rsna.competitive.predictive.density import Supports


def build_supports(
    region_frame, region_tile, acquisition, ordinal, *, classes=12, dtype=torch.float32
):
    """Five declared scales; equal family mass and equal within-family mass.

    Same-frame tiles and within-acquisition ordinal adjacency are renderer-traced
    neighborhoods. They are not cross-acquisition registration or independent
    clinical lesion supports. Full acquisition alternatives cover diffuse signal.
    """
    if (
        region_frame.device.type != "cpu"
        or region_frame.ndim != 1
        or region_frame.dtype != torch.int64
        or region_tile.shape != (len(region_frame), 2)
        or region_tile.dtype != torch.int64
        or acquisition.ndim != 1
        or ordinal.shape != acquisition.shape
        or acquisition.dtype != torch.int64
        or ordinal.dtype != torch.int64
        or min(len(region_frame), len(acquisition), classes) < 1
        or (region_frame < 0).any()
        or (region_frame >= len(acquisition)).any()
        or (region_tile < 0).any()
        or (region_tile > 3).any()
        or dtype not in (torch.float32, torch.float64)
    ):
        raise ValueError(
            "Complete canonical raw-frame4x4 region roster and acquisition ordinals required"
        )
    frames = {}
    for j, (f, tile) in enumerate(zip(region_frame.tolist(), region_tile.tolist())):
        key = tuple(tile)
        d = frames.setdefault(f, {})
        if key in d:
            raise ValueError("A source region must not be duplicated")
        d[key] = j
    if any(set(row) != {(i, j) for i in range(4) for j in range(4)} for row in frames.values()):
        raise ValueError("Keep every unique source frame region, not an observed-only subset")
    families = [[[j] for j in range(len(region_frame))], [], [], [], []]
    for row in frames.values():
        families[1].extend(
            [
                [row[(y + dy, x + dx)] for dy in range(2) for dx in range(2)]
                for y in range(3)
                for x in range(3)
            ]
        )
        families[2].append(list(row.values()))
    by_acq = {}
    for f in frames:
        by_acq.setdefault(int(acquisition[f]), []).append(f)
    for group in by_acq.values():
        group.sort(key=lambda f: int(ordinal[f]))
        if len({int(ordinal[f]) for f in group}) != len(group):
            raise ValueError("Ambiguous canonical source ordering")
        families[4].append([j for f in group for j in frames[f].values()])
        for start in range(len(group) - 2):
            triple = group[start : start + 3]
            if [int(ordinal[f]) for f in triple] != list(
                range(int(ordinal[triple[0]]), int(ordinal[triple[0]]) + 3)
            ):
                continue
            families[3].extend(
                [[frames[f][(y, x)] for f in triple] for y in range(4) for x in range(4)]
            )
    families = [f for f in families if f]
    indices = []
    ptr = [0]
    weights = []
    for family in families:
        for row in family:
            indices.extend(sorted(set(row)))
            ptr.append(len(indices))
            weights.append(-math.log(len(families)) - math.log(len(family)))
    prior = torch.tensor(weights, dtype=dtype)[None].expand(classes, -1).clone()
    supports = Supports(
        torch.tensor(ptr, dtype=torch.int64),
        torch.tensor(indices, dtype=torch.int64),
        prior,
        len(region_frame),
    )
    supports.validate(torch.device("cpu"))
    return supports


def coordinates(study, ordinal):
    """Eight pixel-independent source coordinates, without identities or statistics."""
    acq = study.acquisition
    if ordinal.shape != acq.shape or ordinal.dtype != torch.int64 or (ordinal < 0).any():
        raise ValueError("Original within-acquisition ordinal metadata required")
    count = max(1, len(acq.unique()) - 1)
    # Normalize acquisition index by its metadata rank, not raw UID/hash values.
    ranks = {int(a): rank for rank, a in enumerate(sorted(acq.unique().tolist()))}
    values = []
    height, width = study.context_frames.shape[1:]
    for i in range(len(acq)):
        n = max(1, int(ordinal[acq == acq[i]].max()))
        values.append(
            [
                ranks[int(acq[i])] / count,
                float(ordinal[i]) / n,
                0.5,
                0.5,
                1.0,
                1.0,
                height / 1024,
                width / 1024,
            ]
        )
    tokens = torch.tensor(values, dtype=torch.float32, device=acq.device)
    regions = tokens[study.region_frame].clone()
    regions[:, 2:4] = (study.region_tile.float() + 0.5) / 4
    regions[:, 4:6] = 0.25
    return tokens, regions
