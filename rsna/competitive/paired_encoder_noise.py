"""Streaming paired gradient error against an independently sampled dense reference.

All parameters participate; no projection onto a convenient head subset. This
measures a numerical estimator, not clinical accuracy or proven gradient bias.
The caller owns actual encoder/FP16/dropout/source/input and RNG admission.
"""

from __future__ import annotations

import torch


class GradientReference:
    """Welford mean and trace variance; retain O(parameters), never all draws."""

    def __init__(self):
        self.count = 0
        self.mean: dict[str, torch.Tensor] = {}
        self.m2: dict[str, torch.Tensor] = {}

    def add(self, gradients):
        _validate(gradients, self.mean or None)
        if self.count == 0:
            self.mean = {n: g.double().clone() for n, g in gradients.items()}
            self.m2 = {
                n: torch.zeros((), dtype=torch.float64, device=g.device)
                for n, g in gradients.items()
            }
        else:
            for name, value in gradients.items():
                delta = value.double() - self.mean[name]
                self.mean[name].add_(delta / (self.count + 1))
                self.m2[name].add_((delta * (value.double() - self.mean[name])).sum())
        self.count += 1

    def standard_error_squared(self):
        if self.count < 2:
            raise ValueError("At least two independent dense stochastic draws required")
        return {n: float(m2 / (self.count * (self.count - 1))) for n, m2 in self.m2.items()}


def _validate(gradients, reference=None):
    if (
        not isinstance(gradients, dict)
        or not gradients
        or (reference is not None and gradients.keys() != reference.keys())
        or any(
            not isinstance(n, str)
            or not n
            or not isinstance(g, torch.Tensor)
            or g.dtype != torch.float32
            or g.requires_grad
            or not torch.isfinite(g).all()
            or (
                reference is not None
                and (g.shape != reference[n].shape or g.device != reference[n].device)
            )
            for n, g in gradients.items()
        )
    ):
        raise ValueError("Complete matching finite detached unscaled FP32 gradients required")


class PairedGradientError:
    """Matched draws and frozen reference; expose reference noise rather than hide it."""

    def __init__(self, reference):
        if not isinstance(reference, GradientReference) or reference.count < 2:
            raise ValueError("Actual independently sampled full-bag reference required")
        self.reference_count = reference.count
        self.reference = {n: g.clone() for n, g in reference.mean.items()}
        self.reference_se = reference.standard_error_squared()
        self.count = 0
        self.sums = {
            arm: {n: torch.zeros_like(g) for n, g in self.reference.items()}
            for arm in ("ordinary", "controlled")
        }
        self.errors = {arm: {n: 0.0 for n in self.reference} for arm in self.sums}

    def add(self, ordinary, controlled):
        # Validate both arms before changing any statistic.
        for gradients in (ordinary, controlled):
            _validate(gradients, self.reference)
        for arm, gradients in (("ordinary", ordinary), ("controlled", controlled)):
            for name, value in gradients.items():
                g = value.double()
                self.errors[arm][name] += float((g - self.reference[name]).square().sum())
                self.sums[arm][name].add_(g)
        self.count += 1

    def result(self):
        if self.count < 2:
            raise ValueError("At least two paired sampled gradient draws required")
        groups = {
            "all": list(self.reference),
            "encoder": [n for n in self.reference if n.startswith("backbone.")],
            "head": [n for n in self.reference if not n.startswith("backbone.")],
        }
        out = {}
        for group, names in groups.items():
            if not names:
                continue
            se = sum(self.reference_se[n] for n in names)
            values = {}
            for arm in self.sums:
                mse = sum(self.errors[arm][n] for n in names) / self.count
                mean_error = sum(
                    float((self.sums[arm][n] / self.count - self.reference[n]).square().sum())
                    for n in names
                )
                values[arm] = {
                    "MSE_against_estimated_dense_gradient": mse,
                    "empirical_mean_gradient_error": mean_error,
                    "MSE_minus_estimated_reference_noise": mse - se,
                }
            base, new = values["ordinary"], values["controlled"]

            def ratio(a, b):
                return a / b if b > 0 else None

            out[group] = {
                "parameters": sum(self.reference[n].numel() for n in names),
                "reference_mean_squared_standard_error": se,
                **values,
                "MSE_ratio": ratio(
                    new["MSE_against_estimated_dense_gradient"],
                    base["MSE_against_estimated_dense_gradient"],
                ),
                "empirical_mean_error_ratio": ratio(
                    new["empirical_mean_gradient_error"], base["empirical_mean_gradient_error"]
                ),
                "reference_noise_fraction_of_baseline_MSE": ratio(
                    se, base["MSE_against_estimated_dense_gradient"]
                ),
                "reference_noise_subtraction_is_not_clipped": True,
            }
        return {
            "dense_stochastic_draws": self.reference_count,
            "paired_draws": self.count,
            "groups": out,
            "gradient_bias_proven": False,
            "actual_GPU_precision_dropout_runtime_or_accuracy_accepted": False,
        }
