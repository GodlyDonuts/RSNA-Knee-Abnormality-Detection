"""End-to-end modules for matched B/C/C0, with no inherited classifier shortcut.

A supplied full trainable image backbone is shared in architecture/initial state
between arms. Predictors see only source-masked images and acquisition/region
coordinates. Actual targets enter the evidence/readout AFTER prediction. This
module does not admit a backbone, loader, experiment, GPU or competition entry.
"""

from __future__ import annotations

import copy
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from rsna.competitive.predictive.density import evidence


class Context(nn.Module):
    def __init__(
        self,
        encoder,
        feature_width,
        *,
        hidden=128,
        metadata_width=8,
        classes=12,
        descriptor_width=64,
    ):
        super().__init__()
        if hidden % 4 or min(hidden, metadata_width, classes, descriptor_width, feature_width) < 1:
            raise ValueError("Explicit common image/decoder dimensions required")
        if (
            not isinstance(encoder, nn.Module)
            or not list(encoder.parameters())
            or not all(p.requires_grad for p in encoder.parameters())
        ):
            raise ValueError(
                "The new research arm requires the full image encoder to remain trainable"
            )
        self.encoder = encoder
        self.feature_width = feature_width
        self.hidden = hidden
        self.classes = classes
        self.descriptor_width = descriptor_width
        self.project = nn.Linear(feature_width, hidden)
        self.position = nn.Sequential(
            nn.Linear(metadata_width, hidden), nn.GELU(), nn.Linear(hidden, hidden)
        )
        self.attention = nn.MultiheadAttention(hidden, 4, batch_first=True, dropout=0)
        self.norm = nn.LayerNorm(hidden)
        self.prior = nn.Linear(hidden, classes)

    def train(self, mode=True):
        super().train(mode)
        for layer in self.encoder.modules():
            if isinstance(layer, nn.modules.batchnorm._BatchNorm):
                layer.eval()
        return self

    def condition(self, images, token_metadata, region_metadata):
        if (
            images.ndim != 4
            or images.shape[1] != 3
            or images.shape[0] != len(token_metadata)
            or token_metadata.ndim != 2
            or region_metadata.ndim != 2
            or token_metadata.shape[1] != region_metadata.shape[1]
            or not torch.isfinite(images).all()
            or not torch.isfinite(token_metadata).all()
            or not torch.isfinite(region_metadata).all()
        ):
            raise ValueError(
                "Source-masked image tokens and finite metadata-only coordinates required"
            )
        if self.training and torch.is_grad_enabled():
            encoded = torch.cat(
                [
                    checkpoint(self.encoder, c, use_reentrant=False, preserve_rng_state=True)
                    for c in images.split(8)
                ]
            )
        else:
            encoded = torch.cat([self.encoder(c) for c in images.split(8)])
        if encoded.shape != (len(images), self.feature_width):
            raise ValueError(
                "Supplied image backbone must expose its full image features, not class probabilities"
            )
        with torch.autocast(device_type=images.device.type, enabled=False):
            memory = self.project(encoded.float()) + self.position(token_metadata.float())
            query = self.position(region_metadata.float())
            current, _ = self.attention(query[None], memory[None], memory[None], need_weights=False)
            h = self.norm(query + current[0])
            return h, self.prior(memory.mean(0))


class Predictive(Context):
    """C and C0 share this exact graph; only density-loss coefficient differs."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.negative = nn.Linear(self.hidden, self.classes * self.descriptor_width)
        self.shift = nn.Linear(self.hidden, self.classes * self.descriptor_width)
        nn.init.zeros_(self.shift.weight)
        nn.init.zeros_(self.shift.bias)
        self.register_buffer("fixed_log_variance", torch.zeros(self.classes, self.descriptor_width))

    def predict_hypotheses(self, images, token_metadata, region_metadata):
        h, a = self.condition(images, token_metadata, region_metadata)
        with torch.autocast(device_type=images.device.type, enabled=False):
            shape = (len(h), self.classes, self.descriptor_width)
            mu = self.negative(h).reshape(shape).permute(1, 0, 2)
            delta = self.shift(h).reshape(shape).permute(1, 0, 2)
            return mu, delta, a

    def forward(self, images, token_metadata, region_metadata, targets, observed, supports):
        mu, delta, a = self.predict_hypotheses(images, token_metadata, region_metadata)
        with torch.autocast(device_type=images.device.type, enabled=False):
            e = evidence(
                targets[None],
                mu[None],
                delta[None],
                self.fixed_log_variance,
                supports,
                observed[None],
            )
        return {"context_logits": a, "evidence": e, "logits": a + e.log_bayes_factor[0]}


class ReconstructionControl(Context):
    """B sees the same observed target descriptors as C, plus generic prediction."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reconstruct = nn.Linear(self.hidden, self.descriptor_width)
        self.target_project = nn.Linear(self.descriptor_width, self.hidden)
        self.vote = nn.Linear(self.hidden, self.classes)
        self.weight = nn.Linear(self.hidden, self.classes)

    def forward(self, images, token_metadata, region_metadata, targets, observed):
        h, a = self.condition(images, token_metadata, region_metadata)
        if (
            targets.shape != (len(h), self.descriptor_width)
            or targets.requires_grad
            or observed.shape != (len(h),)
            or observed.dtype != torch.bool
            or not torch.isfinite(targets[observed]).all()
        ):
            raise ValueError("The control receives actual observed detached descriptors too")
        with torch.autocast(device_type=images.device.type, enabled=False):
            reconstruction = self.reconstruct(h)
            if observed.any():
                z = h[observed] + self.target_project(targets[observed])
                logits = a + (self.weight(z).softmax(0) * self.vote(z)).sum(0)
                nll = (
                    0.5
                    * (
                        (targets[observed] - reconstruction[observed]).square() + 1.8378770664093453
                    ).mean()
                )
            else:
                logits = a
                nll = reconstruction.sum() * 0
        return {"logits": logits, "generic_density_loss": nll, "context_logits": a}


def control_two_mask_loss(outputs, targets, label_mask, *, density_weight=0.1):
    if (
        len(outputs) != 2
        or targets.ndim != 2
        or len(targets) != 1
        or label_mask.shape != targets.shape
        or label_mask.dtype != torch.bool
        or not math.isfinite(density_weight)
        or density_weight < 0
    ):
        raise ValueError("Two complete contexts and unchanged one-study weak labels/masks required")
    if (
        not torch.isfinite(targets[label_mask]).all()
        or ((targets[label_mask] < 0) | (targets[label_mask] > 1)).any()
    ):
        raise ValueError("Observed weak probabilities must remain finite and within0to1")
    q = torch.where(label_mask, targets.detach(), 0)
    z = torch.stack([x["logits"] for x in outputs])[None]
    logp = torch.logsumexp(F.logsigmoid(z), 1) - 0.6931471805599453
    log1p = torch.logsumexp(F.logsigmoid(-z), 1) - 0.6931471805599453
    exam = (-(q * logp + (1 - q) * log1p)).masked_fill(
        ~label_mask, 0
    ).sum() / label_mask.sum().clamp_min(1)
    density = torch.stack([x["generic_density_loss"] for x in outputs]).mean()
    return exam + density_weight * density, {
        "exam_loss": exam,
        "density_loss": density,
        "probabilities": logp.exp(),
    }


def matched_modules(encoder, feature_width, *, seed=20261020, **kwargs):
    """Prepare B/C with identical shared state, without consuming global RNG.

    This allocates two image encoders. A production wrapper should instantiate
    them sequentially or on separate admitted GPUs, not duplicate T4 inference.
    No acceptance or accuracy follows from identical initialization alone.
    """
    device_list = sorted({p.device.index for p in encoder.parameters() if p.is_cuda})
    with torch.random.fork_rng(devices=device_list):
        torch.random.default_generator.manual_seed(seed)
        B = ReconstructionControl(copy.deepcopy(encoder), feature_width, **kwargs)
        torch.random.default_generator.manual_seed(seed)
        C = Predictive(copy.deepcopy(encoder), feature_width, **kwargs)
    left, right = B.state_dict(), C.state_dict()
    shared = set(left) & set(right)
    if not all(torch.equal(left[n], right[n]) for n in shared):
        raise ValueError("Shared encoder/context initialization differs")
    return (
        B,
        C,
        {
            "same_shared_tensor_values": True,
            "shared_tensor_names": sorted(shared),
            "B_parameters": sum(p.numel() for p in B.parameters()),
            "C_parameters": sum(p.numel() for p in C.parameters()),
            "actual_MRI_GPU_acceptance_or_accuracy": False,
        },
    )
