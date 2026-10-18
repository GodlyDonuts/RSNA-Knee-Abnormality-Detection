"""Synthetic scientific checks; never reads clinical labels or private rosters."""

import json
from copy import deepcopy

import numpy as np
import torch
from multisource_overlay import construct
from torch.nn.functional import binary_cross_entropy_with_logits as bce


def fixture():
    baseline = np.array([[1, 0.5], [0, 0.3], [0.5, 1]], dtype=np.float32)
    values = np.stack([baseline, 1 - baseline], axis=2)
    states = np.full(values.shape, "probability", dtype="U16")
    states[0, 1, 0] = "unmentioned"  # legacy .5 must remain untouched
    states[1, 0, 1] = "uncertain"
    values[1, 0, 1] = np.nan  # no placeholder vote or NaN propagation
    return dict(
        baseline=baseline,
        baseline_mask=np.array([[True, True], [True, True], [False, True]]),
        values=values,
        states=states,
        source_ids=("baseline", "second"),
        parents={"baseline": (), "second": ()},
        canonical_by_target=(("baseline", "second"),) * 2,
        baseline_source="baseline",
        admitted=np.ones((2, 2), dtype=bool),
        fitting_ids=("synthetic-fit-a", "synthetic-fit-b", "synthetic-fit-c"),
        fitting_components=("component-a", "component-b", "component-c"),
        excluded_ids=frozenset({"synthetic-eval"}),
        excluded_components=frozenset({"eval-component"}),
        accepted_override_mask=np.array([[False, False], [False, True], [False, False]]),
    )


def run():
    args = fixture()
    original = deepcopy(args)
    result = construct(**args)
    assert np.array_equal(result.mask, args["baseline_mask"])
    assert result.targets[0, 0] == 0.5 and result.targets[2, 1] == 0.5
    assert result.changed.sum() == 2
    assert result.targets[0, 1] == args["baseline"][0, 1]  # legacy unmentioned
    assert result.targets[1, 0] == 0  # other source abstains
    assert result.targets[1, 1] == args["baseline"][1, 1]  # accepted override
    assert result.targets[2, 0] == args["baseline"][2, 0]  # masked fallback
    for key in ("baseline", "baseline_mask", "values", "states", "admitted"):
        assert (
            np.array_equal(args[key], original[key], equal_nan=True)
            if key != "states"
            else np.array_equal(args[key], original[key])
        )
    single = deepcopy(args)
    single["canonical_by_target"] = (("baseline",),) * 2
    identity = construct(**single)
    assert np.array_equal(identity.targets, args["baseline"])
    assert not identity.changed.any()
    # Fixed logits: exact source-BCE mixture versus mixed-target BCE, with
    # unchanged per-target effective-batch denominator and detached teacher.
    z = torch.tensor(
        [[-3.0, 0.0], [1.0, 6.0], [0.7, -0.2]], dtype=torch.float64, requires_grad=True
    )
    w = torch.from_numpy(result.effective_weights)
    raw = torch.from_numpy(np.where(w.numpy() > 0, args["values"], 0).astype(np.float64))
    mixed = (raw * w).sum(2)
    explicit = sum(w[:, :, s] * bce(z, raw[:, :, s], reduction="none") for s in range(2))
    compact = bce(z, mixed, reduction="none")
    use = torch.from_numpy(result.eligible)
    old = torch.from_numpy(args["baseline"].astype(np.float64))
    explicit = torch.where(use, explicit, bce(z, old, reduction="none"))
    compact = torch.where(use, compact, bce(z, old, reduction="none"))
    mask = torch.from_numpy(result.mask)
    teacher = torch.full_like(z, 0.23)

    def loss(cells):
        anchored = 0.25 * cells + 0.75 * bce(z, teacher, reduction="none")
        return ((anchored * mask).sum(0) / mask.sum(0)).mean()

    a, b = loss(explicit), loss(compact)
    assert torch.allclose(a, b, atol=1e-14, rtol=1e-14)
    assert torch.allclose(
        torch.autograd.grad(a, z, retain_graph=True)[0],
        torch.autograd.grad(b, z)[0],
        atol=1e-14,
        rtol=1e-14,
    )
    # Probability .5 is meaningful when explicitly canonical, never globally masked.
    half = deepcopy(args)
    half["states"][:] = "probability"
    half["values"][:] = 0.5
    half["baseline"][:] = 0.5
    assert construct(**half).eligible.sum() == 4
    negatives = []
    for name, mutate in (
        ("evaluation identity", lambda x: x.update(excluded_ids=frozenset({"synthetic-fit-a"}))),
        (
            "evaluation component",
            lambda x: x.update(excluded_components=frozenset({"component-b"})),
        ),
        ("mirror vote", lambda x: x["parents"].update(second=("baseline",))),
        (
            "overlapping blend",
            lambda x: x["parents"].update(baseline=("shared",), second=("shared",), shared=()),
        ),
        ("unknown ancestry", lambda x: x["parents"].update(second=("unknown",))),
        ("cycle", lambda x: x["parents"].update(second=("second",))),
        ("unadmitted source", lambda x: x["admitted"].__setitem__((0, 1), False)),
        ("unmapped state", lambda x: x["states"].__setitem__((0, 0, 1), "severity")),
        ("out of range", lambda x: x["values"].__setitem__((0, 0, 1), 1.2)),
        ("changed baseline", lambda x: x["values"].__setitem__((0, 0, 0), 0.8)),
        ("negative weight", lambda x: x.update(weights=-np.ones((2, 2)))),
        ("duplicated source", lambda x: x.update(source_ids=("baseline", "baseline"))),
    ):
        bad = deepcopy(args)
        mutate(bad)
        try:
            construct(**bad)
        except ValueError:
            negatives.append(name)
        else:
            raise AssertionError(name)
    return {
        "status": "synthetic_target_constructor_accepted_no_source_or_training_admission",
        "checks": [
            "single-source identity",
            "missing sentinel abstention",
            "legacy fallback",
            "mask and override identity",
            "no input mutation",
            "meaningful half probability",
            "anchored per-target BCE loss and gradient equality",
        ],
        "refused": negatives,
        "synthetic_audit": result.audit(("synthetic-target-a", "synthetic-target-b")),
        "actual_training_cells_changed": None,
        "new_model_trained": False,
    }


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
