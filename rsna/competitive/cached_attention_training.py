"""New matched deterministic-head training path for cached full-context changes.

No parameters/buffers are added to the licensed parent. Missing log proposals
always take the original full-bag inference path. A distinct trainer must bind
actual moments, source, modes and recovery; this module dispatches nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from rsna.competitive.cached_attention_control import linear_readout
from rsna.competitive.licensed_importance import ProposalCases
from rsna.competitive.licensed_importance import Recipe as BaseRecipe
from rsna.competitive.licensed_importance import sample as sample

VERSION = "cached_attention_first_order_full_context_matched_v1"


@dataclass(frozen=True)
class Recipe(BaseRecipe):
    role: str = "influence"
    variant: str = "linear"

    def __post_init__(self):
        super().__post_init__()
        if self.role != "influence" or self.variant not in ("reference", "linear"):
            raise ValueError("Matched influence proposals and reference/linear variants required")


class ReferenceCases(ProposalCases):
    """Original pixel/target reader plus source-bound full teacher score/vote arrays."""

    def __init__(self, original, proposals, reference_uids, scores, votes, reference_sha256):
        super().__init__(original, proposals)
        if (
            list(reference_uids) != self.uids
            or not isinstance(reference_sha256, str)
            or len(reference_sha256) != 64
            or any(c not in "0123456789abcdef" for c in reference_sha256)
            or any(not isinstance(x, torch.Tensor) for x in (scores, votes))
            or scores.shape != (len(self), 2, 94, 12)
            or votes.shape != scores.shape
            or any(
                x.dtype != torch.float32
                or x.device.type != "cpu"
                or x.requires_grad
                or not torch.isfinite(x).all()
                for x in (scores, votes)
            )
        ):
            raise ValueError(
                "Exact fitting roster and finite detached original two-view full94 moments required"
            )
        self.reference_sha256 = reference_sha256
        self._scores, self._votes = scores, votes

    def case(self, index, positions, mirror):
        result = super().case(index, positions, mirror)
        result["attention_reference"] = {
            "scores": self._scores[index, int(mirror)].contiguous(),
            "votes": self._votes[index, int(mirror)].contiguous(),
            "indices": torch.tensor(positions, dtype=torch.int64),
        }
        return result


class CachedChangeReadout:
    norm: nn.Module
    att: nn.Module
    clsW: torch.Tensor
    clsb: torch.Tensor
    cached_attention_variant: str

    def forward(self, images, log_proposal=None, reference=None):
        if log_proposal is None:
            if reference is not None:
                raise ValueError("Full original inference cannot consume a teacher reference cache")
            return getattr(super(), "forward")(images)
        if (
            self.training
            or getattr(self, "cached_attention_variant", None) not in ("reference", "linear")
            or not isinstance(reference, dict)
            or set(reference) != {"scores", "votes", "indices"}
            or images.ndim != 5
            or images.shape[1] != 16
            or log_proposal.shape != images.shape[:2]
            or log_proposal.dtype != torch.float32
            or log_proposal.requires_grad
            or not torch.isfinite(log_proposal).all()
            or (log_proposal > 0).any()
        ):
            raise ValueError(
                "Explicit deterministic matched mode and source-bound selected16 inputs required"
            )
        encoded = getattr(self, "encode")(images)
        # Both new variants share this FP32 deterministic head policy. It differs
        # from older active protocols and must not be called their causal control.
        with torch.autocast(device_type=images.device.type, enabled=False):
            h = self.norm(encoded.float())
            scores = self.att(h)
            votes = torch.einsum("bkd,td->bkt", h, self.clsW)
            r, v, indices = (reference[k] for k in ("scores", "votes", "indices"))
            if (
                r.shape != (len(images), 94, 12)
                or v.shape != r.shape
                or indices.shape != images.shape[:2]
                or indices.dtype != torch.int64
                or any(x.device != images.device for x in (r, v, indices, log_proposal))
                or any(
                    x.dtype != torch.float32 or x.requires_grad or not torch.isfinite(x).all()
                    for x in (r, v)
                )
                or (indices < 0).any()
                or (indices >= 94).any()
            ):
                raise ValueError(
                    "Exact detached full94 scores/votes and duplicate-preserving selected indices required"
                )
            if self.cached_attention_variant == "reference":
                weights = (scores - log_proposal.unsqueeze(-1)).softmax(1)
                return (weights * votes).sum(1) + self.clsb
            selected = indices.unsqueeze(-1).expand_as(scores)
            return linear_readout(
                r.softmax(1),
                v,
                indices,
                log_proposal,
                scores - r.gather(1, selected),
                votes - v.gather(1, selected),
                self.clsb,
            )
