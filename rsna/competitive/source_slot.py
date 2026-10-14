"""Source-matched Pilkwang v15 slot classifier for one solo DINOv3 student (issue #47).

Reproduces the original v15 classifier/training interface on the admitted uint8
``[6,12,336,336]`` public rows (pilkwang/rsna-knee-baseline-v1 v15, by Pilkwang; private
research only, redistribution licensing unconfirmed). Original semantics, read from the
exact source (SHA ``b32a9155…``):

* training: one three-slice group ``g`` of four non-overlapping groups
  (``take_group``: slices ``3g..3g+2``) per ``BATCH_STUDIES=8`` batch, shared by every
  study and slot of that batch, then ``augment`` (rigid jitter + intensity on uint8);
* per slot ``Model.forward``: ``x/255``, ImageNet normalization, encoder,
  ``cls_mean`` = ``[CLS, mean(patch tokens)]``; ``SlotHead`` per-target query attention
  over the six slots with absent slots masked to ``-1e4`` (``prior=False``);
* submission (``predict_member``): ten overlapping windows ``0..9``, mean of
  ``sigmoid(logits)`` (``TTA_POOL="prob"``).

DINOv2 -> DINOv3 bridge: CLS is DINOv3's final-normalized CLS (``pooler_output``) and the
patch mean excludes the four register tokens (the encoder's ``forward_features``).
Deterministic differences, declared rather than hidden: the group is derived from
``(seed, epoch, update)`` and augmentation draws from a per-study generator instead of
the global RNG (same distributions, resume-exact, device independent); absent slots are
never encoded (their attention weight is exactly zero in the original).
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from rsna.competitive import public_input
from rsna.schema import TARGETS

KIND = "source_matched_slot_v1"
# tools/research/public_teacher_evidence/teacher_review.json: SlotHead/Model nodes.
REFERENCE_NODE_TEXT_SHA256 = "4288d80afb58289511b14037ddf29e97dd04459955d6b8bca739303f50a9648c"
# tools/research/source_matched_slot/review.json: take_group/augment nodes.
TRAINING_NODE_TEXT_SHA256 = "48a1b397288001d491a2aefc40279789c7b47c0ac1215739984d0963a2201409"
POOL = "dinov3_cls_mean_registers_excluded"
TRAINING_WINDOWS = "source_take_group_one_per_update_v1"
EVALUATION_WINDOWS = "source_overlap10_probability_mean_v1"
AUGMENT = "source_rigid_intensity_v1"
AUGMENTATIONS = {AUGMENT, "none"}
SLOTS, SLICES, GROUP = public_input.SLOTS, public_input.SLICES, 3
N_GROUP = SLICES // GROUP
TRAINING_VIEWS = tuple(group * GROUP for group in range(N_GROUP))  # (0, 3, 6, 9)
EVALUATION_VIEWS = tuple(range(SLICES - GROUP + 1))  # original window_starts(12, 3)
AUG_ROT_DEG, AUG_SCALE, AUG_SHIFT, AUG_INTENSITY = 8.0, 0.08, 0.05, 0.10
STRATEGY_FIXED = {
    "kind": KIND,
    "source_sha256": public_input.SOURCE_SHA256,
    "reference_node_text_sha256": REFERENCE_NODE_TEXT_SHA256,
    "training_node_text_sha256": TRAINING_NODE_TEXT_SHA256,
    "pool": POOL,
    "prior": False,
    "hidden": 256,
    "dropout": 0.2,
    "training_windows": TRAINING_WINDOWS,
    "evaluation_windows": EVALUATION_WINDOWS,
}
STRATEGY_KEYS = set(STRATEGY_FIXED) | {"augmentation"}


def validate_strategy(strategy: object) -> dict:
    """Exact versioned policy; only the augmentation switch may be ``none``."""
    if (
        not isinstance(strategy, Mapping)
        or set(strategy) != STRATEGY_KEYS
        or any(
            type(strategy[key]) is not type(value) or strategy[key] != value
            for key, value in STRATEGY_FIXED.items()
        )
        or not isinstance(strategy["augmentation"], str)
        or strategy["augmentation"] not in AUGMENTATIONS
    ):
        raise ValueError("Exact source-matched slot classifier strategy required")
    return dict(strategy)


class SourceSlotHead(nn.Module):
    """Original v15 ``SlotHead`` (``prior=False``): same parameters, names, init order."""

    def __init__(self, dim: int, n_slot: int, n_out: int, hidden: int = 256, p: float = 0.2):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)
        self.query = nn.Parameter(torch.randn(n_out, hidden) * 0.02)
        self.drop = nn.Dropout(p)
        self.out = nn.Linear(hidden, n_out)
        self.hidden = hidden

    def forward(self, x: Tensor, mask: Tensor) -> Tensor:
        h = self.proj(x) + self.slot_emb
        att = torch.einsum("bsh,oh->bos", h, self.query) / self.hidden**0.5
        att = att.masked_fill(mask.unsqueeze(1) < 0.5, -1e4).softmax(-1)
        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias


class SourceSlotStudyNet(nn.Module):
    """One DINOv3 encoder plus the source slot head; inputs are canonical public windows.

    ``forward`` scores exactly one selected window ``[B,6,1,3,H,W]`` (training);
    ``window_probabilities`` averages ``sigmoid`` over the ten original windows.
    Sequence features and slice geometry are accepted for interface parity but unused,
    as in the original head (slot identity is the learned ``slot_emb``).
    """

    def __init__(
        self,
        encoder: nn.Module,
        embedding_dim: int,
        strategy: Mapping,
        encode_chunk_size: int = 64,
        input_channels: int = 3,
    ):
        super().__init__()
        self.strategy = validate_strategy(strategy)
        if input_channels != 3 or not callable(getattr(encoder, "forward_features", None)):
            raise ValueError("Source slot classifier needs the three-channel spatial encoder")
        if type(embedding_dim) is not int or embedding_dim < 1 or encode_chunk_size < 1:
            raise ValueError("Invalid source slot model dimensions")
        self.encoder = encoder
        self.embedding_dim = embedding_dim
        self.encode_chunk_size = encode_chunk_size
        self.input_channels = input_channels
        self.head = SourceSlotHead(
            2 * embedding_dim,
            SLOTS,
            len(TARGETS),
            hidden=self.strategy["hidden"],
            p=self.strategy["dropout"],
        )

    def slot_features(self, window: Tensor, series_mask: Tensor) -> Tensor:
        """``[B,6,3,H,W]`` in [0,1] -> ``[B,6,2D]`` cls_mean; absent slots stay zero."""
        valid = window[series_mask]
        forward_features = getattr(self.encoder, "forward_features")
        parts = [forward_features(part) for part in valid.split(self.encode_chunk_size)]
        cls = torch.cat([part[0] for part in parts])
        patches = torch.cat([part[1] for part in parts])
        if (
            cls.shape != (len(valid), self.embedding_dim)
            or patches.ndim != 4
            or patches.shape[0] != len(valid)
            or patches.shape[-1] != self.embedding_dim
        ):
            raise ValueError("Encoder must return [N,D] CLS and [N,h,w,D] register-free patches")
        features = torch.cat([cls, patches.flatten(1, 2).mean(1)], dim=1)
        full = features.new_zeros((*series_mask.shape, features.shape[-1]))
        full[series_mask] = features
        return full

    def window_logits(self, window: Tensor, series_mask: Tensor) -> Tensor:
        return self.head(self.slot_features(window, series_mask), series_mask.float())

    def _check(self, images, features, series_mask, slice_mask, positions, views: int) -> None:
        b = images.shape[0] if images.ndim == 6 else -1
        if (
            images.ndim != 6
            or tuple(images.shape[1:4]) != (SLOTS, views, self.input_channels)
            or not images.is_floating_point()
            or series_mask.shape != (b, SLOTS)
            or series_mask.dtype != torch.bool
            or slice_mask.shape != (b, SLOTS, views)
            or features.shape != (b, SLOTS, 5)
            or positions.shape != (b, SLOTS, views, 2)
        ):
            raise ValueError("Source slot inputs must be [B,6,views,3,H,W] public windows")
        if not series_mask.any(dim=1).all():
            raise ValueError("Every study needs an acquired source slot")
        if not torch.equal(slice_mask, series_mask[:, :, None].expand_as(slice_mask)):
            raise ValueError("Source slot windows follow the slot mask exactly")
        valid = images[series_mask]
        if not torch.isfinite(valid).all() or (valid < 0).any() or (valid > 1).any():
            raise ValueError("Source slot windows must be finite values in [0,1]")
        if images[~series_mask].count_nonzero():
            raise ValueError("Absent source slots must remain zero")

    def forward(self, images, features, series_mask, slice_mask, positions) -> Tensor:
        self._check(images, features, series_mask, slice_mask, positions, 1)
        return self.window_logits(images[:, :, 0], series_mask)

    def window_probabilities(self, images, features, series_mask, slice_mask, positions):
        """Original ``predict_member``: ``sum_w sigmoid(z_w.float()) / 10``, same order."""
        self._check(images, features, series_mask, slice_mask, positions, len(EVALUATION_VIEWS))
        total: Tensor | None = None
        for view in EVALUATION_VIEWS:
            value = torch.sigmoid(self.window_logits(images[:, :, view], series_mask).float())
            total = value if total is None else total + value
        assert total is not None
        return total / len(EVALUATION_VIEWS)


def _digest_int(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


def training_group(seed: int, epoch: int, update: int) -> int:
    """One uniform group per optimizer update (eight studies), independent of global RNG."""
    if any(type(value) is not int or value < 0 for value in (seed, epoch, update)):
        raise ValueError("Window selection needs nonnegative integer counters")
    return _digest_int(f"{KIND}:group:{seed}:{epoch}:{update}") % N_GROUP


def augmentation_generator(seed: int, epoch: int, study_uid: str) -> torch.Generator:
    if not isinstance(study_uid, str) or not study_uid:
        raise ValueError("Augmentation draws are bound to a study identity")
    return torch.Generator().manual_seed(
        _digest_int(f"{KIND}:augment:{seed}:{epoch}:{study_uid}") % (2**63 - 1)
    )


def source_augment(window: Tensor, generator: torch.Generator) -> Tensor:
    """Original ``augment`` on one study ``[S,3,H,W]`` of exact ``k/255`` values.

    Draw order and arithmetic follow the source; pixels round-trip through uint8 levels
    (source caches uint8, ``.to(imgs.dtype)`` truncates) and return as ``k/255`` floats.
    Runs on CPU so identical draws give identical pixels on any accelerator.
    """
    if window.ndim != 4 or window.device.type != "cpu" or not window.is_floating_point():
        raise ValueError("Source augmentation takes one CPU study [S,3,H,W]")
    levels = (window.float() * 255).round()
    if not torch.allclose(levels / 255, window.float(), rtol=0, atol=1e-6):
        raise ValueError("Source augmentation requires exact uint8 levels /255")
    x = levels
    n = x.shape[0]
    rot = (torch.rand(n, generator=generator) - 0.5) * 2 * (AUG_ROT_DEG * math.pi / 180)
    sc = 1.0 + torch.rand(n, generator=generator) * AUG_SCALE
    tx = (torch.rand(n, generator=generator) - 0.5) * 2 * AUG_SHIFT
    ty = (torch.rand(n, generator=generator) - 0.5) * 2 * AUG_SHIFT
    cos, sin = torch.cos(rot) / sc, torch.sin(rot) / sc
    theta = torch.zeros(n, 2, 3, dtype=torch.float32)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 0, 2] = cos, -sin, tx
    theta[:, 1, 0], theta[:, 1, 1], theta[:, 1, 2] = sin, cos, ty
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    scale = 1.0 + (torch.rand(n, 1, 1, 1, generator=generator) - 0.5) * 2 * AUG_INTENSITY
    x = (x * scale).clamp(0, 255)
    return x.to(torch.uint8).float().div(255).to(window.dtype)


def training_inputs(
    batch: dict, device: torch.device, strategy: Mapping, *, seed: int, epoch: int, update: int
) -> tuple[Tensor, ...]:
    """Select the update's source group (view ``3g`` of the ten) and augment per study."""
    strategy = validate_strategy(strategy)
    images = batch["images"]
    if images.ndim != 6 or images.shape[2] != len(EVALUATION_VIEWS):
        raise ValueError("Source slot training reads the ten-view public row")
    view = TRAINING_VIEWS[training_group(seed, epoch, update)]
    window = images[:, :, view : view + 1]
    if strategy["augmentation"] == AUGMENT:
        series_mask = batch["series_mask"]
        rows = []
        for index, uid in enumerate(batch["study_uid"]):
            augmented = source_augment(
                window[index, :, 0].cpu(), augmentation_generator(seed, epoch, uid)
            )
            # Absent rows stay exactly zero (the original warps zero rows to zero).
            rows.append(torch.where(series_mask[index, :, None, None, None].cpu(), augmented, 0))
        window = torch.stack(rows)[:, :, None]
    return (
        window.to(device, non_blocking=True),
        batch["features"].to(device, non_blocking=True),
        batch["series_mask"].to(device, non_blocking=True),
        batch["slice_mask"][:, :, view : view + 1].to(device, non_blocking=True),
        batch["positions"][:, :, view : view + 1].to(device, non_blocking=True),
    )


def source_row_windows(images: Tensor) -> Tensor:
    """Deployment: one rendered uint8 row ``[6,12,336,336]`` -> ``[6,10,3,336,336]`` /255.

    Identical to the hashed ``dino_adapter.study_inputs`` view construction; the encoder
    applies ImageNet normalization once.
    """
    if images.dtype != torch.uint8 or tuple(images.shape) != (SLOTS, SLICES, 336, 336):
        raise ValueError("Exact source-rendered uint8 six-slot twelve-slice336 row required")
    return (
        torch.stack([images[:, start : start + GROUP] for start in EVALUATION_VIEWS], dim=1)
        .float()
        .div(255)
    )


@torch.inference_mode()
def deployment_probabilities(
    model: SourceSlotStudyNet,
    images: Tensor,
    slot_mask: Tensor,
    device: torch.device,
    precision: str = "fp16",
) -> Tensor:
    """MRI-only student inference for one study: rendered uint8 row + slot mask -> [12].

    No labels, reports, teacher or committee. The caller writes ``submission.csv``.
    """
    if precision not in {"fp32", "fp16", "bf16"}:
        raise ValueError("Explicit supported student inference precision required")
    if slot_mask.dtype != torch.bool or tuple(slot_mask.shape) != (SLOTS,):
        raise ValueError("Boolean six-slot acquisition mask required")
    if not slot_mask.any() or images[~slot_mask].count_nonzero():
        raise ValueError("Acquired slots required; absent slots must be zero")
    windows = source_row_windows(images)[None].to(device)
    series_mask = slot_mask[None].to(device)
    slice_mask = series_mask[:, :, None].expand(1, SLOTS, len(EVALUATION_VIEWS))
    features = torch.zeros(1, SLOTS, 5, device=device)
    positions = torch.zeros(1, SLOTS, len(EVALUATION_VIEWS), 2, device=device)
    model.eval()
    with torch.autocast(
        device.type,
        dtype=torch.float16 if precision == "fp16" else torch.bfloat16,
        enabled=precision != "fp32",
    ):
        probabilities = model.window_probabilities(
            windows, features, series_mask, slice_mask, positions
        )
    return probabilities[0].float().cpu()
