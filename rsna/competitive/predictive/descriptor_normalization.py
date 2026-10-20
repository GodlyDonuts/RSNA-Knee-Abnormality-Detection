"""Standalone fitting-only descriptor scaling candidate; no pipeline integration.

Each complete fitting study has equal weight, and each canonical region within
that study has equal weight. This estimates marginal coordinate variation, not
conditional noise covariance. Scaling does not remove the density loss's
1/(64*N_observed) discrimination-gradient dilution or establish accuracy.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Mapping

import torch

WIDTH = 64
STD_FLOOR = 1e-3  # Predeclared numerical cap: inverse scale <=1000; no validation fitting.
SCHEMA = "study_balanced_canonical_descriptor_normalizer_v1"
ALGORITHM = "equal_study_population_FP64_within_plus_between_variance"
BINDING_KEYS = frozenset({
    "split", "studies", "fitting_roster_sha256", "source_manifest_sha256",
    "descriptor_source_sha256",
})
MAX_ARTIFACT_BYTES = 16384


def _require(value, message):
    if not value:
        raise ValueError(message)


def _hash(value):
    return isinstance(value, str) and len(value) == 64 and not set(value) - set("0123456789abcdef")


def _binding(value):
    _require(isinstance(value, Mapping) and set(value) == BINDING_KEYS,
             "Exact fitting roster/source binding required")
    result = dict(value)
    _require(result["split"] == "fitting" and type(result["studies"]) is int
             and result["studies"] > 0, "Only a declared complete fitting cohort")
    _require(all(_hash(result[k]) for k in BINDING_KEYS - {"split", "studies"}),
             "Complete fitting/source receipt hashes required")
    return result


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _vector(value):
    _require(isinstance(value, (list, tuple)) and len(value) == WIDTH
             and all(type(x) is float and math.isfinite(x) for x in value),
             "Exactly64 finite FP64 coordinate values required")
    return tuple(value)


@dataclass(frozen=True)
class FrozenDescriptorNormalizer:
    """Immutable scalars/tuples, with private canonical serialization and explicit admission."""

    # Binding is stored as immutable sorted pairs rather than a mutable dict.
    binding_items: tuple
    regions: int
    per_study_receipts_sha256: str
    mean: tuple[float, ...]
    variance: tuple[float, ...]
    scale: tuple[float, ...]

    def __post_init__(self):
        binding = _binding(dict(self.binding_items))
        _require(self.binding_items == tuple(sorted(binding.items())), "Immutable canonical binding")
        _require(type(self.regions) is int and self.regions >= 16 * binding["studies"]
                 and self.regions % 16 == 0 and _hash(self.per_study_receipts_sha256),
                 "Complete canonical region and per-study receipt counts required")
        for value in (self.mean, self.variance, self.scale):
            _require(type(value) is tuple, "Frozen coordinate vectors must be immutable tuples")
            _vector(value)
        _require(all(v >= 0 for v in self.variance), "Nonnegative population variances required")
        _require(self.scale == tuple(max(math.sqrt(v), STD_FLOOR) for v in self.variance),
                 "Frozen scale must use the predeclared floor without retuning")
        # The existing Gaussian implementation admits fixed |log variance| <=20.
        _require(all(abs(2 * math.log(s)) <= 20 for s in self.scale),
                 "Descriptor scale is incompatible with existing finite Gaussian variance bounds")

    @property
    def binding(self):
        return dict(self.binding_items)

    def to_bytes(self):
        body = _canonical({
            "schema": SCHEMA, "algorithm": ALGORITHM, "std_floor": STD_FLOOR,
            "binding": self.binding, "regions": self.regions,
            "per_study_receipts_sha256": self.per_study_receipts_sha256,
            "mean": self.mean, "variance": self.variance, "scale": self.scale,
        })
        _require(len(body) <= MAX_ARTIFACT_BYTES, "Bounded private normalizer artifact required")
        return body

    @classmethod
    def from_bytes(cls, body, *, expected_sha256, expected_binding):
        _require(type(body) is bytes and len(body) <= MAX_ARTIFACT_BYTES
                 and _hash(expected_sha256)
                 and hashlib.sha256(body).hexdigest() == expected_sha256,
                 "Exact independently recorded normalizer SHA required")
        expected = _binding(expected_binding)

        def unique(pairs):
            result = {}
            for key, value in pairs:
                _require(key not in result, "Duplicate serialized normalizer field")
                result[key] = value
            return result

        def invalid_constant(_value):
            raise ValueError("Nonfinite serialized normalizer")

        data = json.loads(body, object_pairs_hook=unique, parse_constant=invalid_constant)
        _require(isinstance(data, dict) and set(data) == {
            "schema", "algorithm", "std_floor", "binding", "regions",
            "per_study_receipts_sha256", "mean", "variance", "scale",
        }, "Exact serialized artifact schema required")
        _require(data["schema"] == SCHEMA and data["algorithm"] == ALGORITHM
                 and type(data["std_floor"]) is float and data["std_floor"] == STD_FLOOR
                 and _binding(data["binding"]) == expected,
                 "Fitting cohort, source or fixed algorithm differs")
        result = cls(tuple(sorted(expected.items())), data["regions"],
                     data["per_study_receipts_sha256"], _vector(data["mean"]),
                     _vector(data["variance"]), _vector(data["scale"]))
        _require(result.to_bytes() == body, "Canonical exact artifact bytes required")
        return result

    def _vectors(self, like):
        return (torch.tensor(self.mean, dtype=like.dtype, device=like.device),
                torch.tensor(self.scale, dtype=like.dtype, device=like.device))

    def transform_targets(self, targets, observed=None):
        """Transform only observed FP32 targets; preserve censored values and caller masks."""
        _require(isinstance(targets, torch.Tensor) and targets.ndim >= 2
                 and targets.shape[-1] == WIDTH and targets.dtype == torch.float32
                 and not targets.requires_grad, "Detached FP32 descriptor targets required")
        if observed is None:
            observed = torch.ones(targets.shape[:-1], dtype=torch.bool, device=targets.device)
        _require(isinstance(observed, torch.Tensor) and observed.shape == targets.shape[:-1]
                 and observed.dtype == torch.bool and observed.device == targets.device,
                 "Original aligned censoring mask required")
        _require(bool(torch.isfinite(targets[observed]).all()), "Nonfinite observed descriptors")
        mean, scale = self._vectors(targets)
        result = targets.clone()
        result[observed] = (targets[observed] - mean) / scale
        _require(bool(torch.isfinite(result[observed]).all()), "Transformed observed overflow")
        return result

    def inverse_hypotheses(self, normalized_mean, normalized_shift):
        """Differentiable inverse: raw mu=m+s*mu_z; raw delta=s*delta_z."""
        _require(isinstance(normalized_mean, torch.Tensor)
                 and isinstance(normalized_shift, torch.Tensor)
                 and normalized_mean.ndim >= 2 and normalized_mean.shape[-1] == WIDTH
                 and normalized_shift.shape == normalized_mean.shape
                 and normalized_mean.dtype in (torch.float32, torch.float64)
                 and normalized_shift.dtype == normalized_mean.dtype
                 and normalized_shift.device == normalized_mean.device
                 and bool(torch.isfinite(normalized_mean).all())
                 and bool(torch.isfinite(normalized_shift).all()),
                 "Aligned finite normalized mean/shift tensors required")
        mean, scale = self._vectors(normalized_mean)
        raw_mean, raw_shift = mean + scale * normalized_mean, scale * normalized_shift
        _require(bool(torch.isfinite(raw_mean).all()) and bool(torch.isfinite(raw_shift).all()),
                 "Raw hypothesis overflow")
        return raw_mean, raw_shift

    def raw_log_variance(self, *, device=None, dtype=torch.float32):
        _require(dtype in (torch.float32, torch.float64), "FP32/64 Gaussian arithmetic required")
        return 2 * torch.tensor(self.scale, dtype=dtype, device=device).log()

    def summary(self):
        """Aggregate-only report. Coordinate vectors stay in the private artifact."""
        return {
            "schema": SCHEMA, "studies": self.binding["studies"], "regions": self.regions,
            "artifact_sha256": hashlib.sha256(self.to_bytes()).hexdigest(),
            "std_floor": STD_FLOOR,
            "floored_coordinates": sum(v < STD_FLOOR**2 for v in self.variance),
            "mean_minimum": min(self.mean), "mean_maximum": max(self.mean),
            "variance_minimum": min(self.variance), "variance_maximum": max(self.variance),
            "scale_minimum": min(self.scale), "scale_maximum": max(self.scale),
            "conditional_covariance_or_accuracy_claim": False,
        }


class StudyBalancedMoments:
    """Streaming across studies, never across labels or held-out data.

    The caller proves that each row is one canonical region using the frozen
    source/cache receipt. Ordinals must enumerate its COMPLETE region roster
    once, including all16 tiles per canonical source frame. Supply one context's
    complete targets, not both duplicate views or only the observed half.
    """

    def __init__(self, binding):
        self._binding = _binding(binding)
        self._studies = self._regions = 0
        self._mean = torch.zeros(WIDTH, dtype=torch.float64)
        self._between = self._mean.clone()
        self._within = self._mean.clone()
        self._chain = hashlib.sha256(b"fitting_descriptor_study_receipts_v1").digest()
        self._final = None

    def add_study(self, ordinal, descriptors, canonical_region_ordinals, *, source_receipt_sha256):
        _require(self._final is None and type(ordinal) is int and ordinal == self._studies
                 and ordinal < self._binding["studies"], "Complete fitting study order exactly once")
        _require(isinstance(descriptors, torch.Tensor) and descriptors.ndim == 2
                 and descriptors.shape[1] == WIDTH and len(descriptors) > 0
                 and len(descriptors) % 16 == 0 and descriptors.dtype == torch.float32
                 and descriptors.device.type == "cpu" and not descriptors.requires_grad
                 and bool(torch.isfinite(descriptors).all()), "Finite complete canonical FP32[N,64] study")
        _require(isinstance(canonical_region_ordinals, torch.Tensor)
                 and canonical_region_ordinals.dtype == torch.int64
                 and canonical_region_ordinals.device.type == "cpu"
                 and torch.equal(canonical_region_ordinals, torch.arange(len(descriptors))),
                 "Every canonical region ordinal must appear once in original order")
        _require(_hash(source_receipt_sha256), "Exact original study source receipt required")
        # Center within each study before accumulation, avoiding E[x²]-E[x]² cancellation.
        x = descriptors.double()
        mean = x.mean(0)
        within = (x - mean).square().mean(0)
        count = self._studies + 1
        delta = mean - self._mean
        next_mean = self._mean + delta / count
        next_between = self._between + delta * (mean - next_mean)
        next_within = self._within + (within - self._within) / count
        _require(all(bool(torch.isfinite(v).all()) for v in
                     (next_mean, next_between, next_within)), "FP64 moment overflow")
        receipt = _canonical({
            "ordinal": ordinal, "regions": len(descriptors), "source_receipt_sha256": source_receipt_sha256,
            "canonical_FP32_descriptor_sha256":
                hashlib.sha256(descriptors.contiguous().numpy().tobytes()).hexdigest(),
        })
        self._chain = hashlib.sha256(self._chain + receipt).digest()
        self._mean, self._between, self._within = next_mean, next_between, next_within
        self._studies = count
        self._regions += len(descriptors)

    def finalize(self):
        _require(self._studies == self._binding["studies"], "Complete declared fitting roster required")
        if self._final is None:
            variance = self._within + self._between / self._studies
            _require(bool(torch.isfinite(variance).all()) and bool((variance >= 0).all()),
                     "Finite nonnegative FP64 population variance required")
            values = tuple(variance.tolist())
            self._final = FrozenDescriptorNormalizer(
                tuple(sorted(self._binding.items())), self._regions, self._chain.hex(),
                tuple(self._mean.tolist()), values,
                tuple(max(math.sqrt(v), STD_FLOOR) for v in values))
        return self._final
