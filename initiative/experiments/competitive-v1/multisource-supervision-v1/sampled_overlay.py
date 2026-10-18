"""Label-only adapter to the existing IID16 optimizer; no new training loop."""

from __future__ import annotations

import hashlib
import json

import torch

from rsna.competitive.licensed_importance import ProposalCases, Recipe
from rsna.competitive.licensed_importance_train import fit as original_fit
from rsna.competitive.licensed_parent_train import PARENT_SHA

VERSION = "fitting_safe_common_family_mean_v1"
ORIGINAL_FILE = "48b4485f03b680161b0d2e1a22c73100ec53591bce93e5a2c8826aeec3eb9285"
POPULATION = "0485d1b1df7af2b8ef10207bfa37545e2c65f7dc68f47a254f2af0c475de7f42"
# Filled only from the actual immutable materialization receipt, never from a
# guessed name. An unset pin refuses production while allowing synthetic tests.
PRODUCTION_TARGET = "65d3730702a098c8c819396dba6122902a4f3720dde9b2cb3f9ffd7ac5eca3b5"
PRODUCTION_ANNOTATIONS = "54f1c94943bd7049b3e3da79fdcffb1f7083ecb373a64b9e2642aa2fe9230df9"

PRODUCTION_LABELS = "f88ee60e27b1673f3fa7934107c91cc8aef2e810b7cedd34aa725f6470571ddd"
PRODUCTION_ORIGINAL_LABELS = "8ce962aeef9b4ab5e0de65145e875e9caffeddc8fddd66065692f21ca9637444"
PRODUCTION_MASK = "e80fba0e4d304d013b49895ebe3d3201972a0afd158e95fc296b8c222685f9a1"
PRODUCTION_TEACHER = "d9853c7745161807705087bef505f1ceb488f2138264dea3902388662fe37566"


def sha(tensor):
    return hashlib.sha256(tensor.detach().contiguous().numpy().tobytes()).hexdigest()


class ConsensusLabels:
    def __init__(self, original, labels, eligible, policy, *, diagnostic_only):
        n = len(original)
        if (
            type(diagnostic_only) is not bool
            or not n
            or not isinstance(policy, dict)
            or len(original.uids) != n
            or len(set(original.uids)) != n
        ):
            raise ValueError("Original unique fitting roster and explicit scope required")
        for value, shape, dtype in (
            (original.labels, (n, 12), torch.float32),
            (labels, (n, 12), torch.float32),
            (original.mask, (n, 12), torch.bool),
            (eligible, (n, 12), torch.bool),
            (original.teacher, (n, 2, 12), torch.float32),
        ):
            if (
                not isinstance(value, torch.Tensor)
                or value.dtype != dtype
                or value.shape != shape
                or value.device.type != "cpu"
                or value.requires_grad
            ):
                raise ValueError("Exact detached original CPU tensor types/shapes required")
        if (
            not torch.isfinite(original.labels).all()
            or not torch.isfinite(labels).all()
            or not torch.isfinite(original.teacher).all()
            or ((labels < 0) | (labels > 1)).any()
            or ((original.labels < 0) | (original.labels > 1)).any()
            or (eligible & ~original.mask).any()
            or not torch.equal(labels[~eligible], original.labels[~eligible])
        ):
            raise ValueError("Only eligible original supervised label values may change")
        population = hashlib.sha256(
            json.dumps(list(original.uids), separators=(",", ":")).encode()
        ).hexdigest()
        role = policy.get("role")
        expected = {
            "version": VERSION,
            "diagnostic_only": diagnostic_only,
            "parent_checkpoint_sha256": PARENT_SHA,
            "original_target_file_sha256": ORIGINAL_FILE,
            "population": n,
            "population_sha256": population,
            "original_labels_sha256": sha(original.labels),
            "labels_sha256": sha(labels),
            "mask_sha256": sha(original.mask),
            "teacher_sha256": sha(original.teacher),
            "eligible_sha256": sha(eligible),
        }
        if (
            role not in ("unchanged_reference", "safe_family_mean")
            or any(policy.get(k) != value for k, value in expected.items())
            or set(policy)
            != set(expected) | {"role", "target_file_sha256", "annotation_file_sha256"}
        ):
            raise ValueError("Exact source/array/role and resume identity required")
        for key in ("target_file_sha256", "annotation_file_sha256"):
            value = policy[key]
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(c not in "0123456789abcdef" for c in value)
            ):
                raise ValueError("Immutable source artifact digests required")
        if role == "unchanged_reference" and (
            not torch.equal(labels, original.labels)
            or policy["target_file_sha256"] != ORIGINAL_FILE
        ):
            raise ValueError("Reference must preserve every target exactly")
        if not diagnostic_only and (
            n != 3887
            or population != POPULATION
            or not original.mask.all()
            or PRODUCTION_TARGET is None
            or PRODUCTION_ANNOTATIONS is None
            or policy["annotation_file_sha256"] != PRODUCTION_ANNOTATIONS
            or policy["original_labels_sha256"] != PRODUCTION_ORIGINAL_LABELS
            or policy["mask_sha256"] != PRODUCTION_MASK
            or policy["teacher_sha256"] != PRODUCTION_TEACHER
            or role == "safe_family_mean"
            and policy["labels_sha256"] != PRODUCTION_LABELS
            or role == "safe_family_mean"
            and policy["target_file_sha256"] != PRODUCTION_TARGET
            or role == "safe_family_mean"
            and int((labels != original.labels).sum()) != 35418
        ):
            raise ValueError("Actual complete pinned fitting-only production artifacts required")
        self.original, self.uids = original, list(original.uids)
        self._old, self._labels = original.labels.clone(), labels.clone()
        self._mask, self._teacher = original.mask.clone(), original.teacher.clone()
        self._policy = json.loads(json.dumps(policy, sort_keys=True))

    def __len__(self):
        return len(self.uids)

    @property
    def binding(self):
        return json.loads(json.dumps(self._policy))

    def case(self, index, positions, mirror):
        if (
            type(index) is not int
            or not 0 <= index < len(self)
            or type(mirror) is not bool
            or list(self.original.uids) != self.uids
        ):
            raise ValueError("Original admitted fitting index/order/view required")
        row = self.original.case(index, positions, mirror)
        if (
            set(row) != {"images", "labels", "mask", "parent_logits"}
            or not torch.equal(row["labels"], self._old[index])
            or not torch.equal(row["mask"], self._mask[index])
            or not torch.equal(row["parent_logits"], self._teacher[index, int(mirror)])
        ):
            raise ValueError("Original mask, labels or teacher changed after admission")
        return {**row, "labels": self._labels[index].clone()}


def fit(model, data, output, binding, recipe, **kwargs):
    if (
        not isinstance(recipe, Recipe)
        or recipe.role != "uniform"
        or recipe.source_weight != 0.25
        or not isinstance(data, ProposalCases)
        or not isinstance(data.original, ConsensusLabels)
        or not isinstance(binding, dict)
        or "multisource_supervision" in binding
        or binding.get("diagnostic_only") is not data.original.binding["diagnostic_only"]
        or not binding["diagnostic_only"]
        and recipe != Recipe()
    ):
        raise ValueError(
            "Unchanged uniform16 recipe, source-parent fraction and explicit role required"
        )
    return original_fit(
        model,
        data,
        output,
        {**binding, "multisource_supervision": data.original.binding},
        recipe,
        **kwargs,
    )
