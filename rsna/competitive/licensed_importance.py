"""Fixed teacher-guided observations for matched full-encoder training.

The proposal was screened against a frozen teacher, not clinical truth. A
self-normalized finite-bag estimate can be biased. Inference still delegates to
the original full94 readout. No patient record or external model body is included.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch

from rsna.competitive.licensed_parent_train import Recipe as OriginalRecipe

VERSION = "licensed_full_encoder_uniform_or_teacher_influence_IID16_v1"
UNIFORM_FLOOR = 0.25 / 94


@dataclass(frozen=True)
class Recipe(OriginalRecipe):
    windows: int = 16
    role: str = "uniform"

    def __post_init__(self):
        super().__post_init__()
        if self.windows != 16 or self.role not in ("uniform", "influence"):
            raise ValueError("Matched fixed16 uniform/influence observation recipe required")


def validate_proposal(value):
    if (
        not isinstance(value, torch.Tensor)
        or value.shape != (94,)
        or value.dtype != torch.float64
        or value.device.type != "cpu"
        or value.requires_grad
        or not torch.isfinite(value).all()
        or (value < UNIFORM_FLOOR - 1e-14).any()
        or abs(float(value.sum()) - 1) > 1e-12
    ):
        raise ValueError("Fixed normalized CPU FP64 94-window proposal with25%floor required")


def sample(uid, recipe, epoch, proposal):
    """Common uniform variates and anatomical view; duplicates are valid IID draws.

    The seed excludes role so both arms see the same study order/view and use
    the same uniform variates. Sampling never changes any global RNG stream.
    """
    if (
        not isinstance(recipe, Recipe)
        or not isinstance(uid, str)
        or not uid
        or type(epoch) is not int
        or not 0 <= epoch < recipe.epochs
    ):
        raise ValueError("Exact case, recipe and zero-based epoch required")
    validate_proposal(proposal)
    q = proposal if recipe.role == "influence" else torch.full((94,), 1 / 94, dtype=torch.float64)
    seed = hashlib.sha256(f"{VERSION}/{recipe.seed}/{epoch}/{uid}".encode()).digest()
    g = torch.Generator().manual_seed(int.from_bytes(seed[:8], "little") % 2**63)
    u = torch.rand(recipe.windows, generator=g, dtype=torch.float64)
    selected = torch.searchsorted(q.cumsum(0), u, right=True).clamp_max(93)
    mirror = bool(torch.randint(0, 2, (), generator=g).item())
    # Log probabilities are constants. Correcting by -log(q) avoids selecting
    # the same influential window and then counting its proposal twice.
    return selected.tolist(), mirror, q[selected].log().float()


class ProposalCases:
    """Wrap the unchanged unique-window reader while preserving IID duplicates."""

    def __init__(self, original, proposals):
        if (
            not hasattr(original, "uids")
            or not len(original)
            or len(original.uids) != len(original)
            or len(set(original.uids)) != len(original)
            or not isinstance(proposals, torch.Tensor)
            or proposals.shape != (len(original), 94)
        ):
            raise ValueError("One fixed aligned proposal for every original fitting study required")
        for row in proposals:
            validate_proposal(row)
        self.original = original
        self.uids = list(original.uids)
        self._proposals = proposals.clone()
        self.proposal_sha256 = hashlib.sha256(self._proposals.numpy().tobytes()).hexdigest()

    def __len__(self):
        return len(self.uids)

    def proposal(self, index):
        return self._proposals[index].clone()

    def case(self, index, positions, mirror):
        if (
            type(index) is not int
            or not 0 <= index < len(self)
            or type(mirror) is not bool
            or not isinstance(positions, list)
            or len(positions) != 16
            or any(type(p) is not int or not 0 <= p < 94 for p in positions)
        ):
            raise ValueError("Exactly16 original IID window indices required")
        unique = list(dict.fromkeys(positions))
        reverse = {p: i for i, p in enumerate(unique)}
        result = dict(self.original.case(index, unique, mirror))
        images = result["images"]
        if not isinstance(images, torch.Tensor) or images.shape[0] != len(unique):
            raise ValueError("Complete original unique-window images required")
        result["images"] = images[[reverse[p] for p in positions]].contiguous()
        return result


class ImportanceReadout:
    """Compose before CheckpointedEncode and the privately supplied parent class.

    No parameter/buffer is added. Missing log proposals take the original full
    inference path. During sampled training, center the log proposal (a softmax
    invariant), subtract it from attention logits, then normalize over the IID
    draws. This corrects sampling frequency; it does not make the ratio unbiased.
    """

    norm: torch.nn.Module
    att: torch.nn.Module
    clsW: torch.Tensor
    clsb: torch.Tensor

    def forward(self, images, log_proposal=None):
        if log_proposal is None:
            return getattr(super(), "forward")(images)
        if (
            not isinstance(log_proposal, torch.Tensor)
            or log_proposal.dtype != torch.float32
            or images.ndim != 5
            or log_proposal.shape != images.shape[:2]
            or log_proposal.requires_grad
            or not torch.isfinite(log_proposal).all()
            or (log_proposal > 0).any()
        ):
            raise ValueError("One finite constant log proposal for every sampled window required")
        h = self.norm(getattr(self, "encode")(images))
        scores = self.att(h)
        centered = log_proposal - log_proposal.mean(1, keepdim=True)
        weights = (scores.float() - centered.float().unsqueeze(-1)).softmax(1).to(h.dtype)
        pooled = torch.einsum("bkn,bkf->bnf", weights, h)
        return (pooled * self.clsW).sum(-1) + self.clsb
