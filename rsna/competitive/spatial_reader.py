"""Full-map joint evidence on nominal within-acquisition coordinates.

No registration, local receptive-field guarantee or lesion annotation is
inferred. Cross-series links are deliberately absent. The matched null keeps
the same pixels, source supports, observed-token coverage and both weighted
marginals; it intentionally randomizes the positive geometric edges.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import Tensor, nn

from rsna.competitive.spatial_correspondence import randomized_correspondence_null

VERSION = "N3_full_spatial_nominal_joint_reader_v1"


@dataclass(frozen=True)
class RelationGroup:
    first: np.ndarray
    second: np.ndarray
    tokens: np.ndarray
    aligned: Tensor
    control: Tensor
    kind: str


@dataclass(frozen=True)
class SpatialPlan:
    windows: int
    channels: int
    height: int
    width: int
    valid_tokens: tuple[np.ndarray, ...]
    groups: tuple[RelationGroup, ...]
    metadata_sha256: str


def _seed(seed: int, identity: str, kind: str) -> int:
    raw = json.dumps([seed, identity, kind, VERSION], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")


def build_plan(graphs: list[dict], windows: np.ndarray, shape: tuple[int, int, int], *, seed: int):
    """Source-bound nominal grid; does not turn a token into a local lesion.

    Grid centers describe the input letterbox. Encoder attention makes their
    actual receptive fields nonlocal. Sigma is one grid step, not a clinical
    distance cutoff. Original-slice-disjoint repeat edges stay within a series.
    """
    channels, height, width = shape
    if (
        type(seed) is not int
        or min(channels, height, width) < 1
        or windows.ndim != 2
        or windows.shape[1] != 2
        or not np.issubdtype(windows.dtype, np.integer)
        or not len(windows)
        or len(set(map(tuple, windows.tolist()))) != len(windows)
    ):
        raise ValueError("Distinct original windows, shape and explicit seed required")
    valid: list[np.ndarray | None] = [None] * len(windows)
    groups = []
    for graph in graphs:
        slot = graph["slot"]
        selected = np.asarray(graph["selected_source_indices"], dtype=np.int64)
        supports = np.asarray(graph["supports"], dtype=np.int64).reshape(-1, 3)
        expected = np.array([selected[j : j + 3] for j in range(max(0, len(selected) - 2))])
        expected = expected.reshape(-1, 3)
        take = np.flatnonzero(windows[:, 0] == slot)
        if (
            len(selected) < 1
            or (selected < 0).any()
            or (np.diff(selected) <= 0).any()
            or not np.array_equal(supports, expected)
            or not np.array_equal(windows[take, 1], np.arange(len(supports)))
        ):
            raise ValueError("Stored windows and original-slice supports differ")
        if not len(take):
            continue
        side = graph["native_cache_side"]
        resized = np.asarray(graph["native_resized_shape"], dtype=np.int64)
        original = np.asarray(graph["native_original_shape"], dtype=np.int64)
        if (
            type(side) is not int
            or side < 1
            or resized.shape != (2,)
            or original.shape != (3,)
            or (original <= 0).any()
            or (resized <= 0).any()
            or (resized > side).any()
            or graph.get("input_augmentation_or_mirror") is not False
            or graph.get("full_frame_letterbox_align_corners_false") is not True
            or (selected >= original[0]).any()
        ):
            raise ValueError("Unchanged native full-frame letterbox geometry required")
        top, left = (side - resized) // 2
        # Pixel-center convention for the nominal feature grid in input space.
        yy, xx = np.meshgrid(
            (np.arange(height) + 0.5) * side / height - 0.5,
            (np.arange(width) + 0.5) * side / width - 0.5,
            indexing="ij",
        )
        keep = (
            (yy >= top - 0.5)
            & (yy < top + resized[0] - 0.5)
            & (xx >= left - 0.5)
            & (xx < left + resized[1] - 0.5)
        )
        tokens = np.flatnonzero(keep.ravel())
        if not len(tokens):
            raise ValueError("Letterbox contains no represented nominal image token")
        for index in take:
            if valid[index] is not None:
                raise ValueError("Duplicate source-series slot")
            valid[index] = tokens
        if not graph["geometry_known"]:
            continue
        spacing = np.asarray(graph["pixel_spacing"], dtype=float)
        if spacing.size == 0:
            # Known slice centers do not imply known in-plane pixel spacing.
            continue
        orientation = np.asarray(graph["image_orientation_patient"], dtype=float)
        positions = np.asarray(graph["selected_image_positions_patient"], dtype=float)
        if (
            spacing.shape != (2,)
            or not np.isfinite(spacing).all()
            or (spacing <= 0).any()
            or orientation.shape != (6,)
            or positions.shape != (len(selected), 3)
            or not np.isfinite(positions).all()
            or not np.isfinite(orientation).all()
        ):
            raise ValueError("Claimed physical geometry lacks original finite source fields")
        basis = orientation.reshape(2, 3)
        if not np.allclose(basis @ basis.T, np.eye(2), atol=1e-3, rtol=0):
            raise ValueError("Orthonormal patient in-plane basis required")
        # DICOM column indices use spacing[1], row indices spacing[0].
        y = ((yy - top + 0.5) * original[1] / resized[0] - 0.5) * spacing[0]
        x = ((xx - left + 0.5) * original[2] / resized[1] - 0.5) * spacing[1]
        xy = np.stack((x.ravel()[tokens], y.ravel()[tokens]), axis=1)
        sigma = math.sqrt(
            (
                (side / width * original[2] / resized[1] * spacing[1]) ** 2
                + (side / height * original[1] / resized[0] * spacing[0]) ** 2
            )
            / 2
        )
        repeat = np.asarray(graph["repeat_edges"], dtype=np.int64).reshape(-1, 2)
        if (
            (repeat < 0).any()
            or (repeat >= len(take)).any()
            or any(set(supports[i]) & set(supports[j]) for i, j in repeat)
        ):
            raise ValueError("Longitudinal pair must have disjoint ORIGINAL source support")
        families: dict[tuple, list[tuple[int, int]]] = {
            (0.0, 0.0, "local"): [(i, i) for i in range(len(take))]
        }
        for first, second in repeat:
            delta = (positions[second + 1] - positions[first + 1]) @ basis.T
            # Cache-equivalent sub-micron header rounding is not registration.
            dx, dy = np.round(delta, 3).tolist()
            families.setdefault((dx, dy, "repeat"), []).append((int(first), int(second)))
        for (dx, dy, kind), edges in families.items():
            distances = ((xy[:, None] - xy[None] - [dx, dy]) ** 2).sum(-1)
            support = distances <= (2.5 * sigma) ** 2
            if kind == "local":
                np.fill_diagonal(support, False)
            kernel = np.where(support, np.exp(-distances / (2 * sigma**2)), 0)
            if kernel.sum() <= 0:
                continue
            matrix = torch.tensor(kernel / kernel.sum(), dtype=torch.float32)
            identity = graph["series_uid"] + json.dumps([dx, dy])
            allowed = torch.ones_like(matrix, dtype=torch.bool)
            if kind == "local":
                allowed.fill_diagonal_(False)
            control, receipt = randomized_correspondence_null(
                matrix, allowed, seed=_seed(seed, identity, kind)
            )
            if not receipt["testable"]:
                continue  # Both arms abstain; global image evidence stays available.
            edges_array = np.asarray(edges, dtype=np.int64)
            groups.append(
                RelationGroup(
                    take[edges_array[:, 0]], take[edges_array[:, 1]], tokens, matrix, control, kind
                )
            )
    if any(x is None for x in valid):
        raise ValueError("Every original window must have exactly one source-series plan")
    digest = hashlib.sha256(json.dumps(graphs, sort_keys=True).encode()).hexdigest()
    return SpatialPlan(
        len(windows),
        channels,
        height,
        width,
        tuple(x for x in valid if x is not None),
        tuple(groups),
        digest,
    )


class SpatialReader(nn.Module):
    """One learned projection sees FULL maps; shared image and relational paths.

    No hard logit ceiling: a common detached-teacher objective regularizes both
    arms. Output weights start exactly zero. The base encoder is never stored
    inside this trainable module.
    """

    def __init__(self, channels=1024, width=64, *, seed=20261017):
        super().__init__()
        if type(channels) is not int or type(width) is not int or min(channels, width) < 1:
            raise ValueError("Positive spatial reader dimensions required")
        self.channels, self.width = channels, width
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.project = nn.Sequential(
                nn.LayerNorm(channels), nn.Linear(channels, width), nn.GELU()
            )
            self.evidence = nn.Linear(width, 12)
            self.output = nn.Parameter(torch.zeros(12, 12))
            self.bias = nn.Parameter(torch.zeros(12))

    def statistics(self, maps: Tensor, plan: SpatialPlan, *, arm: str) -> Tensor:
        if (
            arm not in {"aligned", "null"}
            or maps.shape != (plan.windows, plan.channels, plan.height, plan.width)
            or plan.channels != self.channels
            or maps.dtype != torch.float32
            or not torch.isfinite(maps).all()
        ):
            raise ValueError("Finite complete FP32 maps and bound spatial plan required")
        flat = maps.flatten(2).transpose(1, 2)
        evidence = self.evidence(self.project(flat))
        kept = torch.cat(
            [
                evidence[i, torch.as_tensor(tokens, device=maps.device)]
                for i, tokens in enumerate(plan.valid_tokens)
            ]
        )
        values = [
            kept.mean(0),
            kept.std(0, unbiased=False),
            torch.logsumexp(kept, 0) - math.log(len(kept)),
            kept.amax(0),
        ]
        covariance: dict[str, list[Tensor]] = {"local": [], "repeat": []}
        for group in plan.groups:
            first = torch.as_tensor(group.first, device=maps.device)
            second = torch.as_tensor(group.second, device=maps.device)
            tokens = torch.as_tensor(group.tokens, device=maps.device)
            pi = (group.aligned if arm == "aligned" else group.control).to(maps.device)
            a, b = pi.sum(1), pi.sum(0)
            e1, e2 = evidence[first][:, tokens], evidence[second][:, tokens]
            c1 = e1 - torch.einsum("p,wpc->wc", a, e1)[:, None]
            c2 = e2 - torch.einsum("p,wpc->wc", b, e2)[:, None]
            numerator = torch.einsum("wpc,pq,wqc->wc", c1, pi, c2)
            va = torch.einsum("p,wpc->wc", a, c1.square())
            vb = torch.einsum("p,wpc->wc", b, c2.square())
            denominator = (va * vb).clamp_min(1e-8).sqrt()
            covariance[group.kind].append(numerator / denominator)
        zero = kept.sum(0) * 0
        counts = []
        for kind in ("local", "repeat"):
            rows = covariance[kind]
            joined = torch.cat(rows) if rows else None
            values.extend([joined.mean(0), joined.amax(0)] if joined is not None else [zero, zero])
            counts.append(0 if joined is None else len(joined))
        nuisance = [
            math.log1p(plan.windows),
            math.log1p(len(kept)),
            *[math.log1p(n) for n in counts],
        ]
        return torch.stack(values + [kept.new_full((12,), x) for x in nuisance], dim=1)

    def forward(self, maps: Tensor, base_logits: Tensor, plan: SpatialPlan, *, arm: str):
        if (
            base_logits.shape != (12,)
            or base_logits.dtype != torch.float32
            or base_logits.device != maps.device
            or not torch.isfinite(base_logits).all()
        ):
            raise ValueError("Twelve original finite FP32 base logits required")
        stats = self.statistics(maps, plan, arm=arm)
        correction = (stats * self.output).sum(1) + self.bias
        return base_logits.detach() + correction, correction
