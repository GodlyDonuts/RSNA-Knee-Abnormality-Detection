"""Render the original selected contexts without cloning unused source frames.

Every original frame is validated and every canonical target region and support
is retained. Only the intermediate masked image volume is materialized lazily.
This is an isolated serving candidate; no live training source is modified.
"""

from __future__ import annotations

from dataclasses import replace

import torch
from torch.nn import functional as F

from rsna.competitive.predictive.cases import RawGroup
from rsna.competitive.predictive.supports import build_supports


def _admit(groups, side, halo, context_cap, classes):
    if (
        not isinstance(groups, (list, tuple))
        or not groups
        or any(type(v) is not int or v < 1 for v in (side, context_cap, classes))
        or type(halo) is not int
        or halo < 0
    ):
        raise ValueError("Original groups and positive rendering configuration required")
    aliases = set()
    plans = []
    for g in groups:
        if not isinstance(g, RawGroup):
            raise ValueError("Original RawGroup required")
        frames = g.frames
        if (
            not isinstance(frames, torch.Tensor)
            or frames.device.type != "cpu"
            or frames.dtype != torch.float32
            or frames.ndim != 3
            or frames.requires_grad
            or len(frames) < 1
            or min(frames.shape[1:]) < 32
            or min(frames.shape[1:]) // 4 <= 2 * halo
            or not torch.isfinite(frames).all()
            or (frames < 0).any()
            or (frames > 1).any()
        ):
            raise ValueError("Complete finite original metadata-scaled raw frames required")
        for t in (g.aliases, g.acquisition, g.ordinal):
            if (
                not isinstance(t, torch.Tensor)
                or t.device.type != "cpu"
                or t.dtype != torch.int64
                or t.shape != (len(frames),)
                or (t < 0).any()
            ):
                raise ValueError("Complete original integer source metadata required")
        keys = g.aliases.tolist()
        first = {}
        for index, alias in enumerate(keys):
            if alias in first:
                if not torch.equal(frames[index], frames[first[alias]]):
                    raise ValueError("Alias frames disagree")
            else:
                first[alias] = index
        if aliases.intersection(first):
            raise ValueError("Exact aliases must stay within one matrix group")
        aliases.update(first)
        d = g.descriptors
        if (
            not isinstance(d, torch.Tensor)
            or d.device.type != "cpu"
            or d.dtype != torch.float32
            or d.shape != (16 * len(first), 64)
            or d.requires_grad
            or not torch.isfinite(d).all()
        ):
            raise ValueError("Complete admitted exact detached descriptor cache required")
        plans.append(
            {
                "group": g,
                "first": list(first.values()),
                "canonical": [first[key] for key in keys],
                "acq": g.acquisition.tolist(),
                "ord": g.ordinal.tolist(),
            }
        )
    return plans


def _images(plan, neighbors, pattern, *, halo, side):
    """Same masked pixels/halo/interpolation, including cross-acquisition aliases."""
    g = plan["group"]
    _, height, width = g.frames.shape
    # The original checkerboard mask has only two geometry patterns. Aliases
    # always inherit their first occurrence's pattern, even if their own
    # acquisition/ordinal metadata differs.
    masks = torch.zeros((2, height, width), dtype=torch.bool)
    for parity in (0, 1):
        for row in range(4):
            for col in range(4):
                if (row + col + parity) % 2 == pattern:
                    y0, y1 = row * height // 4, (row + 1) * height // 4
                    x0, x1 = col * width // 4, (col + 1) * width // 4
                    masks[
                        parity,
                        max(0, y0 - halo) : min(height, y1 + halo),
                        max(0, x0 - halo) : min(width, x1 + halo),
                    ] = True
    rendered = []
    # One original three-channel window at a time bounds temporary source
    # pixels irrespective of the number of selected windows or source slices.
    for triple in neighbors:
        masked = g.frames[torch.tensor(triple)]
        for channel, i in enumerate(triple):
            representative = plan["canonical"][i]
            parity = (plan["ord"][representative] // 4 + plan["acq"][representative]) % 2
            masked[channel].masked_fill_(masks[parity], 0.5)
        rendered.append(
            F.interpolate(masked[None], size=(side, side), mode="bilinear", align_corners=False)
        )
    return torch.cat(rendered).contiguous()


def assemble(groups, *, side=320, halo=8, context_cap=94, classes=12):
    plans = _admit(groups, side, halo, context_cap, classes)
    roster = sorted(
        (p["acq"][i], p["ord"][i], group_id, i)
        for group_id, p in enumerate(plans)
        for i in p["first"]
    )
    if len({x[:2] for x in roster}) != len(roster):
        raise ValueError("Unique source acquisition ordinals required across matrix groups")
    selected = (
        torch.linspace(0, len(roster) - 1, min(len(roster), context_cap)).round().long().tolist()
    )
    centers = {(roster[i][2], roster[i][3]) for i in selected}
    all_acq = sorted({r[0] for r in roster})
    ranks = {a: i / max(1, len(all_acq) - 1) for i, a in enumerate(all_acq)}
    max_ord = {a: max(1, max(r[1] for r in roster if r[0] == a)) for a in all_acq}
    token_meta, region_meta, targets, region_frame, tiles, acq, orders = ([] for _ in range(7))
    observed = [[], []]
    frame_offset = 0
    for group_id, p in enumerate(plans):
        g = p["group"]
        n, height, width = g.frames.shape
        metadata = torch.tensor(
            [
                [
                    ranks[p["acq"][i]],
                    float(p["ord"][i]) / max_ord[p["acq"][i]],
                    0.5,
                    0.5,
                    1.0,
                    1.0,
                    height / 1024,
                    width / 1024,
                ]
                for i in p["canonical"]
            ],
            dtype=torch.float32,
        )
        rf = torch.tensor([i for i in p["first"] for _ in range(16)], dtype=torch.int64)
        rt = torch.tensor(
            [(y, x) for _ in p["first"] for y in range(4) for x in range(4)], dtype=torch.int64
        )
        region = metadata[rf].clone()
        region[:, 2:4] = (rt.float() + 0.5) / 4
        region[:, 4:6] = 0.25
        chosen = [i for i in range(n) if (group_id, i) in centers]
        neighbors = []
        for i in chosen:
            same = [j for j in range(n) if p["acq"][j] == p["acq"][i]]
            same.sort(key=lambda j: p["ord"][j])
            index = same.index(i)
            neighbors.append([same[max(0, index - 1)], i, same[min(len(same) - 1, index + 1)]])
        p["neighbors"] = neighbors
        if chosen:
            token_meta.append(metadata[chosen])
        region_meta.append(region)
        targets.append(g.descriptors)
        for pattern in (0, 1):
            observed[pattern].append(
                torch.tensor(
                    [
                        (y + x + p["ord"][i] // 4 + p["acq"][i]) % 2 == pattern
                        for i in p["first"]
                        for y in range(4)
                        for x in range(4)
                    ],
                    dtype=torch.bool,
                )
            )
        region_frame.append(rf + frame_offset)
        tiles.append(rt)
        acq.append(g.acquisition)
        orders.append(g.ordinal)
        frame_offset += n
    supports = build_supports(
        torch.cat(region_frame),
        torch.cat(tiles),
        torch.cat(acq),
        torch.cat(orders),
        classes=classes,
    )
    result = []
    for pattern in (0, 1):
        result.append(
            {
                "images": torch.cat(
                    [
                        _images(p, p["neighbors"], pattern, halo=halo, side=side)
                        for p in plans
                        if p["neighbors"]
                    ]
                ),
                "token_metadata": torch.cat(token_meta),
                "region_metadata": torch.cat(region_meta),
                "targets": torch.cat(targets),
                "observed": torch.cat(observed[pattern]),
                "supports": replace(
                    supports,
                    indptr=supports.indptr.clone(),
                    regions=supports.regions.clone(),
                    log_prior=supports.log_prior.clone(),
                ),
            }
        )
    if not torch.equal(result[0]["observed"], ~result[1]["observed"]) or not torch.equal(
        result[0]["targets"], result[1]["targets"]
    ):
        raise ValueError("Complete complementary masks and identical target roster required")
    return result
