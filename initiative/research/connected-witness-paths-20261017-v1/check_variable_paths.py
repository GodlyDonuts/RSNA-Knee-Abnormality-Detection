"""Exhaustive physical/control enumeration and gradients; not clinical tests."""

import hashlib
import itertools
import json
from pathlib import Path

import prototype
import torch
import variable_paths as p


def exhaustive(unary, joints, starts, gaps, kappa, arm):
    transitions = joints / joints.sum(2, keepdim=True)
    measures = p.conditional_marginals(transitions, starts)
    scores = []
    for diagnosis in range(unary.shape[2]):
        terms = []
        for first in range(len(unary)):
            for last in range(first, len(unary)):
                for path in itertools.product(range(unary.shape[1]), repeat=last - first + 1):
                    mass = starts[first, path[0]] / len(unary)
                    for t in range(first, last):
                        c = torch.sigmoid(kappa[diagnosis] - (gaps[t] / p.REFERENCE_GAP_MM).log())
                        probability = (
                            transitions[t, path[t - first], path[t - first + 1]]
                            if arm == "physical"
                            else measures[first, t + 1, path[t - first + 1]]
                        )
                        mass = mass * c * probability
                    if last < len(unary) - 1:
                        mass = mass * (
                            1
                            - torch.sigmoid(
                                kappa[diagnosis] - (gaps[last] / p.REFERENCE_GAP_MM).log()
                            )
                        )
                    if mass > 0:
                        terms.append(
                            mass.log()
                            + sum(
                                unary[t, path[t - first], diagnosis] for t in range(first, last + 1)
                            )
                        )
        scores.append(torch.logsumexp(torch.stack(terms), dim=0))
    return torch.stack(scores)


torch.set_num_threads(1)
torch.manual_seed(671)
dtype = torch.float64
unary = torch.randn(3, 2, 2, dtype=dtype) * 0.3
joints = torch.tensor([[[0.7, 0.3], [0.2, 0.8]], [[0.05, 0.95], [0.6, 0.4]]], dtype=dtype)
starts = torch.tensor([[0.3, 0.7], [0.8, 0.2], [0.6, 0.4]], dtype=dtype)
gaps = torch.tensor([3.0, 15.0], dtype=dtype)
kappa = torch.tensor([-0.4, 0.8], dtype=dtype)
for arm in ("physical", "independent"):
    assert torch.allclose(
        p.log_partition(unary, joints, starts, gaps, kappa, arm=arm),
        exhaustive(unary, joints, starts, gaps, kappa, arm),
        atol=1e-12,
        rtol=0,
    )
    assert torch.autograd.gradcheck(
        lambda u, k: p.witness_score(u, joints, starts, gaps, k, arm=arm),
        (unary.clone().requires_grad_(), kappa.clone().requires_grad_()),
        atol=1e-5,
        rtol=1e-4,
    )
    for prior in (-80.0, 0.0, 80.0):
        u = torch.zeros_like(unary, requires_grad=True)
        k = torch.full((2,), prior, dtype=dtype, requires_grad=True)
        score = p.witness_score(u, joints, starts, gaps, k, arm=arm)
        assert torch.equal(score, torch.zeros_like(score))
        score.sum().backward()
        assert torch.isfinite(u.grad).all() and torch.isfinite(k.grad).all()

# Enumerate every prefix to verify ALL start-conditioned token marginals.
transitions = joints / joints.sum(2, keepdim=True)
measures = p.conditional_marginals(transitions, starts)
for first in range(3):
    for last in range(first, 3):
        physical = torch.zeros(2, dtype=dtype)
        independent = torch.zeros(2, dtype=dtype)
        for path in itertools.product(range(2), repeat=last - first + 1):
            a = b = starts[first, path[0]]
            for t in range(first, last):
                a = a * transitions[t, path[t - first], path[t - first + 1]]
                b = b * measures[first, t + 1, path[t - first + 1]]
            physical[path[-1]] += a
            independent[path[-1]] += b
        assert torch.allclose(physical, measures[first, last], atol=1e-12, rtol=0)
        assert torch.allclose(independent, physical, atol=1e-12, rtol=0)

# The old stationary/reversible case is an exact special case, not a new recipe.
joint = torch.tensor([[0.4, 0.1], [0.1, 0.4]], dtype=dtype)
repeated = joint[None].repeat(2, 1, 1)
stationary_starts = joint.sum(1)[None].repeat(3, 1)
for arm, old_joint in (("physical", joint), ("independent", prototype.factorized_joint(joint))):
    new = p.witness_score(unary, repeated, stationary_starts, gaps, kappa, arm=arm)
    old = prototype.witness_score(unary, old_joint, gaps, kappa)
    assert torch.allclose(new, old, atol=1e-12, rtol=0)

# Focal and FP32 paths, including actual zero outgoing entries, stay finite.
for arm in ("physical", "independent"):
    single = torch.randn(1, 2, 2, dtype=dtype, requires_grad=True)
    focal_kappa = kappa.clone().requires_grad_()
    p.witness_score(single, joints[:0], starts[:1], gaps[:0], focal_kappa, arm=arm).sum().backward()
    assert torch.isfinite(single.grad).all() and torch.isfinite(focal_kappa.grad).all()
    u = (unary.float() * 1000).requires_grad_()
    k = torch.tensor([-80.0, 80.0], requires_grad=True)
    sparse = joints.float()
    sparse[0, 0] = torch.tensor([1.0, 0.0])
    score = p.witness_score(u, sparse, starts.float(), gaps.float(), k, arm=arm)
    score.sum().backward()
    assert (
        torch.isfinite(score).all()
        and torch.isfinite(u.grad).all()
        and torch.isfinite(k.grad).all()
    )

refused = 0
for j, s, g in (
    (torch.zeros_like(joints), starts, gaps),
    (joints, starts * 0, gaps),
    (joints, starts, gaps * 0),
    (joints * float("nan"), starts, gaps),
    (joints.requires_grad_(), starts, gaps),
):
    try:
        p.witness_score(unary, j, s, g, kappa, arm="physical")
    except ValueError:
        refused += 1
    else:
        raise AssertionError("Invalid source prior accepted")
assert refused == 5
D = Path(__file__).resolve().parent
result = {
    "status": "varying_directed_path_and_exact_start_conditioned_marginal_control_math_checks_passed",
    "source_sha256": hashlib.sha256((D / "variable_paths.py").read_bytes()).hexdigest(),
    "exhaustive_physical_and_independent_path_partitions_match": True,
    "both_unary_and_kappa_gradchecks_passed": True,
    "all_start_conditioned_single_position_marginals_enumerated_equal": True,
    "old_stationary_case_both_arms_reproduced": True,
    "zero_evidence_extreme_priors_focal_FP32_sparse_transition_gradients_finite": True,
    "invalid_source_prior_refusals": refused,
    "actual_MRI_input_GPU_training_or_accuracy_acceptance": False,
    "complexity_limit": "Reference conditional null has quadratic sequence-length storage/computation; real MRI/dense throughput is unmeasured.",
}
(D / "variable-mathematical-checks.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result))
