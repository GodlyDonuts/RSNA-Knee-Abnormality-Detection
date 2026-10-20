"""Variable-matrix raw source assembly with one complete study support prior.

The caller admits original DICOM provenance and label-independent source order.
Context centers may be budgeted; target/support rosters are never truncated or
renormalized to the centers. This is a new renderer, not old-cache input parity.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from rsna.competitive.predictive.inputs import masked_study, triplets
from rsna.competitive.predictive.supports import build_supports


@dataclass(frozen=True)
class RawGroup:
    frames: torch.Tensor
    aliases: torch.Tensor
    acquisition: torch.Tensor
    ordinal: torch.Tensor
    descriptors: torch.Tensor | None = None


def assemble(groups, *, side=320, halo=8, context_cap=94, classes=12):
    """Two complementary views; sparse supports cover all unique source regions.

    Encoding metadata scaling must already have been applied to original raw
    matrices. This function intentionally has no teacher/logit/label argument.
    No augmentation, image-statistics scaler or cross-acquisition registration.
    """
    if not groups or type(context_cap) is not int or context_cap < 1:
        raise ValueError("Original variable-matrix source groups and context budget required")
    roster = []
    aliases = set()
    for group_id, g in enumerate(groups):
        if not isinstance(g, RawGroup) or g.frames.device.type != "cpu":
            raise ValueError("Admitted raw CPU source groups required")
        local = set(g.aliases.tolist())
        if aliases & local:
            raise ValueError("Exact aliases must be grouped together before masking")
        aliases |= local
        first = {}
        for i, alias in enumerate(g.aliases.tolist()):
            first.setdefault(alias, i)
        for i in first.values():
            roster.append((int(g.acquisition[i]), int(g.ordinal[i]), group_id, i))
    roster.sort()
    if len({x[:2] for x in roster}) != len(roster):
        raise ValueError("Unique source acquisition ordinals required across matrix groups")
    selected = (
        torch.linspace(0, len(roster) - 1, min(len(roster), context_cap)).round().long().tolist()
    )
    centers = {(roster[i][2], roster[i][3]) for i in selected}
    all_acq = sorted({r[0] for r in roster})
    ranks = {a: i / max(1, len(all_acq) - 1) for i, a in enumerate(all_acq)}
    max_ord = {a: max(1, max(r[1] for r in roster if r[0] == a)) for a in all_acq}
    results = []
    for pattern in (0, 1):
        images, token_meta, region_meta, targets, observed, region_frame, tiles, acq, orders = (
            [] for _ in range(9)
        )
        frame_offset = 0
        for group_id, g in enumerate(groups):
            view = masked_study(
                g.frames,
                g.aliases,
                g.acquisition,
                g.ordinal,
                pattern=pattern,
                halo=halo,
                cached_descriptors=g.descriptors,
            )
            n, height, width = g.frames.shape
            metadata = torch.tensor(
                [
                    [
                        ranks[int(g.acquisition[i])],
                        float(g.ordinal[i]) / max_ord[int(g.acquisition[i])],
                        0.5,
                        0.5,
                        1.0,
                        1.0,
                        height / 1024,
                        width / 1024,
                    ]
                    for i in range(n)
                ],
                dtype=torch.float32,
            )
            region = metadata[view.region_frame].clone()
            region[:, 2:4] = (view.region_tile.float() + 0.5) / 4
            region[:, 4:6] = 0.25
            chosen = [i for i in range(n) if (group_id, i) in centers]
            if chosen:
                neighbors = []
                for i in chosen:
                    same = [j for j in range(n) if g.acquisition[j] == g.acquisition[i]]
                    same.sort(key=lambda j: int(g.ordinal[j]))
                    p = same.index(i)
                    neighbors.append([same[max(0, p - 1)], i, same[min(len(same) - 1, p + 1)]])
                images.append(
                    triplets(view, torch.tensor(chosen), torch.tensor(neighbors), side=side)
                )
                token_meta.append(metadata[chosen])
            region_meta.append(region)
            targets.append(view.descriptors)
            observed.append(view.observed)
            region_frame.append(view.region_frame + frame_offset)
            tiles.append(view.region_tile)
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
        results.append(
            {
                "images": torch.cat(images),
                "token_metadata": torch.cat(token_meta),
                "region_metadata": torch.cat(region_meta),
                "targets": torch.cat(targets),
                "observed": torch.cat(observed),
                "supports": supports,
            }
        )
    if not torch.equal(results[0]["observed"], ~results[1]["observed"]) or not torch.equal(
        results[0]["targets"], results[1]["targets"]
    ):
        raise ValueError("Complete complementary masks and identical target roster required")
    return results
