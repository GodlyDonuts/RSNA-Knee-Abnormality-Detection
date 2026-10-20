"""Passive train-only summaries from the two already-computed C forward outputs."""

import math

import torch
from torch.nn import functional as F


class Monitor:
    """No extra model call, backward, optimizer, RNG draw, or modified output."""

    def __init__(self, model, *, selected=(1, 24, 146, 200), epoch_steps=486):
        self.selected = frozenset(selected) | frozenset(epoch_steps * n for n in range(1, 7))
        self.step = None
        self.labels = self.mask = None
        self.outputs = []
        self.rows = []
        self.hook = model.register_forward_hook(self._observe)

    def begin_case(self, step, labels, mask):
        if self.outputs or self.labels is not None:
            raise ValueError("Previous selected study did not produce exactly two contexts")
        if self.step is not None and self.step != step:
            raise ValueError("Selected update was not finalized")
        if step not in self.selected:
            return
        self.step = step
        self.labels, self.mask = labels.detach(), mask.detach()

    def _observe(self, _module, _args, output):
        if self.labels is None:
            return
        # Detach and copy only twelve logits/BFs; never retain training graphs.
        self.outputs.append((output["context_logits"].detach().float().cpu(),
                             output["evidence"].log_bayes_factor[0].detach().float().cpu()))
        if len(self.outputs) == 2:
            with torch.no_grad():
                a = torch.stack([pair[0] for pair in self.outputs])
                bf = torch.stack([pair[1] for pair in self.outputs])
                mask = self.mask.cpu().bool()
                q = torch.where(mask, self.labels.cpu().float(), 0)

                def bce(z):
                    logp = torch.logsumexp(F.logsigmoid(z), 0)[None] - math.log(2)
                    log1p = torch.logsumexp(F.logsigmoid(-z), 0)[None] - math.log(2)
                    return (-(q * logp + (1 - q) * log1p)).masked_fill(
                        ~mask, 0
                    ).sum() / mask.sum().clamp_min(1)

                posterior, prior = (a + bf).sigmoid().mean(0), a.sigmoid().mean(0)
                full_bce, prior_bce = bce(a + bf), bce(a)
                values = {
                    "BCE_full": full_bce.item(), "BCE_context_only": prior_bce.item(),
                    "BCE_full_minus_context": (full_bce - prior_bce).item(),
                    "mean_absolute_posterior_shift": (posterior - prior).abs().mean().item(),
                    "BF_mean_square": bf.square().mean().item(),
                    "BF_absolute_max": bf.abs().max().item(),
                    "posterior_saturation_fraction":
                        ((posterior <= 1e-4) | (posterior >= 1 - 1e-4)).float().mean().item(),
                    "observed_target_count": int(mask.sum().item()),
                }
                if not all(math.isfinite(v) for v in values.values()):
                    raise ValueError("Nonfinite passive evidence diagnostic")
                self.rows.append(values)
            self.outputs.clear()
            self.labels = self.mask = None
        # Returning None preserves the original forward output exactly.

    def finish_update(self, step, studies):
        if step not in self.selected:
            if self.rows or self.outputs or self.labels is not None:
                raise ValueError("Unexpected telemetry outside selected update")
            return {}
        if self.step != step or len(self.rows) != studies or self.outputs or self.labels is not None:
            raise ValueError("Selected telemetry must cover the complete original update")
        mean_keys = ("BCE_full", "BCE_context_only", "BCE_full_minus_context",
                     "mean_absolute_posterior_shift", "posterior_saturation_fraction")
        result = {key: sum(r[key] for r in self.rows) / studies for key in mean_keys}
        result.update(
            BF_RMS=math.sqrt(sum(r["BF_mean_square"] for r in self.rows) / studies),
            BF_absolute_max=max(r["BF_absolute_max"] for r in self.rows),
            studies=studies, contexts=2 * studies,
            observed_targets=sum(r["observed_target_count"] for r in self.rows),
            before_current_optimizer_update=True, training_only=True,
            extra_model_forwards=0, extra_backwards=0, accuracy_or_encoder_gradient_claim=False,
        )
        self.rows.clear()
        self.step = None
        return {"C100_passive_evidence": result}

    def close(self):
        if self.rows or self.outputs or self.labels is not None:
            raise ValueError("Unfinished passive telemetry")
        self.hook.remove()
