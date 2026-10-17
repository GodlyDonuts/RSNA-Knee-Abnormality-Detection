"""Exhaustive latent-path and gradient checks, not MRI accuracy tests."""

import hashlib
import importlib.util
import itertools
import json
import sys
from pathlib import Path

import torch

D = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location(
    "connected_witness_mathematical_prototype", D / "prototype.py"
)
p = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = p
spec.loader.exec_module(p)
torch.set_num_threads(1)
torch.manual_seed(643)
dtype = torch.float64


def exhaustive(unary, joint, gaps, logits):
    joint, marginal = p._joint(joint)
    transition = joint / marginal[:, None]
    values = []
    for d in range(unary.shape[2]):
        terms = []
        for start in range(len(unary)):
            for stop in range(start, len(unary)):
                for path in itertools.product(range(unary.shape[1]), repeat=stop - start + 1):
                    mass = marginal[path[0]] / len(unary)
                    for t in range(start, stop):
                        c = torch.sigmoid(logits[d] - torch.log(gaps[t] / p.REFERENCE_GAP_MM))
                        mass = mass * c * transition[path[t - start], path[t - start + 1]]
                    if stop < len(unary) - 1:
                        c = torch.sigmoid(logits[d] - torch.log(gaps[stop] / p.REFERENCE_GAP_MM))
                        mass = mass * (1 - c)
                    if mass > 0:
                        evidence = sum(unary[t, path[t - start], d] for t in range(start, stop + 1))
                        terms.append(mass.log() + evidence)
        values.append(torch.logsumexp(torch.stack(terms), dim=0))
    return torch.stack(values)


joint = torch.tensor([[0.4, 0.1], [0.1, 0.4]], dtype=dtype)
unary = torch.randn(3, 2, 2, dtype=dtype) * 0.4
gaps = torch.tensor([2.0, 17.0], dtype=dtype)
logits = torch.tensor([-0.3, 0.6], dtype=dtype)
expected = exhaustive(unary, joint, gaps, logits)
assert torch.allclose(
    p.log_path_partition(unary, joint, gaps, logits), expected, atol=1e-12, rtol=0
)
assert torch.allclose(
    p.log_path_partition(torch.zeros_like(unary), joint, gaps, logits),
    torch.zeros(2, dtype=dtype),
    atol=1e-12,
    rtol=0,
)
assert torch.autograd.gradcheck(
    lambda u, k: p.witness_score(u, joint, gaps, k),
    (unary.clone().requires_grad_(), logits.clone().requires_grad_()),
    atol=1e-5,
    rtol=1e-4,
)
for length in (1, 2, 8):
    for k in (-80.0, 0.0, 80.0):
        field = torch.zeros(length, 2, 2, dtype=dtype, requires_grad=True)
        parameter = torch.full((2,), k, dtype=dtype, requires_grad=True)
        delta = torch.full((length - 1,), 3.0, dtype=dtype)
        score = p.witness_score(field, joint, delta, parameter)
        assert torch.equal(score, torch.zeros_like(score))
        score.sum().backward()
        assert torch.isfinite(field.grad).all() and torch.isfinite(parameter.grad).all()
# Strong evidence and extreme continuation remain finite in log space.
field = torch.tensor([[[1000.0, -1000.0], [-1000.0, 1000.0]]] * 3, dtype=dtype, requires_grad=True)
k = torch.tensor([80.0, -80.0], dtype=dtype, requires_grad=True)
score = p.witness_score(field, joint, gaps, k)
score.sum().backward()
assert (
    torch.isfinite(score).all()
    and torch.isfinite(field.grad).all()
    and torch.isfinite(k.grad).all()
)
# Reversibility/stationary marginal is shared by physical and factorized priors.
null = p.factorized_joint(joint)
assert torch.allclose(null.sum(1), joint.sum(1), atol=1e-12, rtol=0)
perm = torch.tensor([1, 0])
assert torch.allclose(
    p.witness_score(unary[:, perm], joint[perm][:, perm], gaps, logits),
    p.witness_score(unary, joint, gaps, logits),
    atol=1e-12,
    rtol=0,
)
# Nonuniform token mass is important: a uniform test alone cannot establish
# preservation of letterbox/edge-weighted stationary marginals.
weighted_joint = torch.tensor(
    [[0.30, 0.02, 0.03], [0.02, 0.10, 0.05], [0.03, 0.05, 0.40]], dtype=dtype
)
weighted_field = torch.randn(3, 3, 2, dtype=dtype) * 0.4
weighted_null = p.factorized_joint(weighted_joint)
weighted_marginal = weighted_joint.sum(1)
assert torch.allclose(weighted_null.sum(1), weighted_marginal, atol=1e-12, rtol=0)
assert torch.allclose(
    weighted_marginal @ (weighted_joint / weighted_marginal[:, None]),
    weighted_marginal,
    atol=1e-12,
    rtol=0,
)
assert torch.allclose(
    p.log_path_partition(weighted_field, weighted_joint, gaps, logits),
    exhaustive(weighted_field, weighted_joint, gaps, logits),
    atol=1e-12,
    rtol=0,
)
fp32_field = torch.randn(16, 3, 12, dtype=torch.float32, requires_grad=True)
fp32_kappa = torch.linspace(-8, 8, 12, dtype=torch.float32).requires_grad_()
fp32_gaps = torch.linspace(1, 24, 15, dtype=torch.float32)
fp32_score = p.witness_score(fp32_field, weighted_joint.float(), fp32_gaps, fp32_kappa)
fp32_score.sum().backward()
assert (
    torch.isfinite(fp32_score).all()
    and torch.isfinite(fp32_field.grad).all()
    and torch.isfinite(fp32_kappa.grad).all()
)
# A toy equal-marginal/equal-pair-dot summary collision. NOT a collision proof
# for the actual trained CNN or every statistic of the existing MRI reader.
a = torch.tensor([[2, 2, -2, -2], [2, -2, 2, -2], [2, -2, -2, 2]], dtype=dtype)[..., None]
b = torch.tensor([[2, 2, -2, -2], [2, -2, 2, -2], [-2, 2, 2, -2]], dtype=dtype)[..., None]


def pair_summary(x):
    y = x[:, :, 0]
    pairs = (y[:-1] * y[1:]).mean(1)
    return torch.stack(
        [
            y.mean(),
            y.std(unbiased=False),
            torch.logsumexp(y.flatten(), 0),
            y.max(),
            pairs.mean(),
            pairs.max(),
        ]
    )


assert torch.equal(pair_summary(a), pair_summary(b))
physical = 0.97 * torch.eye(4, dtype=dtype) + 0.03 * torch.ones(4, 4, dtype=dtype) / 4
independent = p.factorized_joint(physical)
delta = torch.full((2,), p.REFERENCE_GAP_MM, dtype=dtype)
prior = torch.zeros(1, dtype=dtype)
physical_a = p.witness_score(a, physical, delta, prior).item()
physical_b = p.witness_score(b, physical, delta, prior).item()
null_a = p.witness_score(a, independent, delta, prior).item()
null_b = p.witness_score(b, independent, delta, prior).item()
assert physical_a > physical_b and abs(null_a - null_b) < 1e-12
changed_gap = p.witness_score(a, physical, delta * 4, prior).item()
assert abs(changed_gap - physical_a) > 1e-4
zero_mix = p.acquisition_mixture(
    [(torch.zeros_like(a), physical, delta), (torch.zeros_like(a[:1]), physical, delta[:0])], prior
)
assert torch.allclose(zero_mix, torch.zeros_like(zero_mix), atol=1e-12, rtol=0)
refusals = 0
cases = [
    (unary, joint, torch.tensor([0.0, 1.0], dtype=dtype), logits),
    (unary, joint, torch.tensor([float("nan"), 1.0], dtype=dtype), logits),
    (unary, torch.tensor([[0.4, 0.2], [0.1, 0.3]], dtype=dtype), gaps, logits),
    (unary, torch.zeros(2, 2, dtype=dtype), gaps, logits),
    (unary, joint, gaps, logits[:1]),
    (unary * float("nan"), joint, gaps, logits),
]
for args in cases:
    try:
        p.witness_score(*args)
    except ValueError:
        refusals += 1
    else:
        raise AssertionError("Invalid path input accepted")
assert refusals == 6
out = {
    "status": "connected_witness_mathematical_prototype_checks_passed",
    "prototype_sha256": hashlib.sha256((D / "prototype.py").read_bytes()).hexdigest(),
    "exhaustive_all_paths_T3_P2_D2_log_partition_matches": True,
    "unary_and_continuation_double_precision_gradcheck_passed": True,
    "zero_evidence_exact_baseline_lengths_1_2_8_and_logits_minus80_zero_plus80": True,
    "strong_evidence_and_extreme_prior_outputs_and_gradients_finite": True,
    "stationary_token_marginal_control_and_coordinate_permutation_passed": True,
    "nonuniform_stationary_marginal_and_exhaustive_partition_passed": True,
    "FP32_16_position_12_target_outputs_and_gradients_finite": True,
    "toy_equal_histogram_and_pair_summary_collision": {
        "physical_coherent": physical_a,
        "physical_switching": physical_b,
        "factorized_coherent": null_a,
        "factorized_switching": null_b,
        "is_actual_MRI_or_existing_trained_CNN_collision": False,
    },
    "absolute_gap_changes_model_score_in_toy": True,
    "overlapping_chain_mixture_zero_evidence_normalization_passed": True,
    "invalid_input_refusals": refusals,
    "MRI_input_qualification_real_trainer_recovery_or_clinical_accuracy_passed": False,
    "new_GPU_or_model_training": False,
}
(D / "mathematical-checks.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out))
