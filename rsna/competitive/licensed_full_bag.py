"""Memory-bounded whole-study encoding for the privately supplied parent class.

Compose ``class WholeStudy(CheckpointedEncode, OriginalRaptor): ...`` on authorized
storage. Load its exact original full state before learning. The original class
and source globals remain untouched. No external implementation or weights here.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

VERSION = "whole94_window_MIL_checkpointed_encoder8_v1"


class CheckpointedEncode:
    """Recompute training activations; preserve the entire study's attention bag.

    Nonreentrant checkpointing preserves RNG and the autocast context during
    recomputation and supports ordinary DDP. Frozen BN statistics are mandatory
    so a recomputed chunk cannot update running buffers twice. Inference delegates
    to the unchanged parent encoder, retaining its original chunking/precision.
    """

    training: bool
    backbone: nn.Module
    checkpoint_chunk = 8

    def encode(self, x):
        if not self.training or not torch.is_grad_enabled():
            return getattr(super(), "encode")(x)
        if x.ndim != 5 or min(x.shape[:2]) < 1:
            raise ValueError("Nonempty complete B,K,C,H,W study bag required")
        if any(
            isinstance(m, nn.modules.batchnorm._BatchNorm) and m.training
            for m in self.backbone.modules()
        ):
            raise ValueError("Freeze parent BN running statistics before recomputation")
        flat = x.flatten(0, 1)
        features = [
            checkpoint(self.backbone, chunk, use_reentrant=False, preserve_rng_state=True)
            for chunk in flat.split(self.checkpoint_chunk)
        ]
        return torch.cat(features).view(x.shape[0], x.shape[1], -1)
