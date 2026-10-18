# Finding-stratified MRI observations at the unchanged mixture density

Question: does removing random variation in *which finding proposal supplies a
window* improve sixteen-window fidelity to the frozen full94 teacher? Current
influence training averages twelve normalized finding-sensitivity proposals with
25% uniform coverage, then samples sixteen times independently from that mixture.
The candidate takes one independent draw from each finding proposal plus four
uniform draws. Both retain the **same expected window counts and exact original
mixture density**. No additional images, labels, teacher changes or encoder passes.
This is a new finding-proposal comparison, not the historical uniform-versus-
stratified slice-position experiment. Existing live training remains untouched.

The connection is deterministic-allocation multiple importance sampling, widely
used in Monte Carlo rendering: [Veach and Guibas, 1995](https://doi.org/10.1145/218380.218498)
and [Veach's thesis, chapter 9](https://graphics.stanford.edu/papers/veach_thesis/thesis.pdf).
The application here is a testable coverage mechanism, not a novel theorem.

For twelve component distributions p_c and uniform u, the unchanged density is
q=(sum_c p_c+4u)/16. Every observation uses the **mixture q** in the original
inverse-proposal attention correction, including observations drawn from a
specific component. Using that component's density in the correction would
change the estimator and is not this comparison. Degenerate zero-sensitivity
findings retain the original uniform fallback. Duplicate indices remain legal.

For any fixed window set A, the probability of missing it is
(1-q(A))^16 for IID and product_c(1-p_c(A))*(1-u(A))^4 for fixed counts.
The latter cannot exceed the former by the arithmetic/geometric mean inequality.
This is set-coverage dominance, **not** identification of a lesion. Similarly,
fixed allocation removes between-component variance for unnormalized linear
importance estimators. The attention readout is a ratio followed by a sigmoid:
its bias/MSE and a learned model's disease performance have no such guarantee.

## Predeclared bounded assessment

Reuse the original whole-byte-admitted 3887 fitting-only two-view encoder feature
cases, original eight-tensor MIL head and original proposal table on evc33. Do not
read reports, source labels, Gold or confirmation. Use all12 findings,128 draws
per study/role, base seed20261018 plus sorted study index, common random uniforms,
and the original25% uniform mixture. Report every target's teacher-probability
MSE/bias, unique-window counts, and exact no-hit probability for its smallest
80%-component-mass support. That set is an estimator diagnostic, not clinical
anatomy. Verify the candidate density against every original stored proposal.
The FP64 moment target is reported with its discrepancy from the original FP32
readout; neither is an independent clinical label.

Nine focused checks pass: original-density identity, zero-sensitivity fallback,
independent inverse-CDF reference, exact all-subset miss dominance, an enumerated
Bernoulli-count variance calculation, deterministic all-target benchmark, and
invalid uniform/changed-density refusals. No training recipe or core module is
modified. Mathematical checks are not MRI acceptance or improved Kaggle accuracy.

The one CPU/zero GPU assessor may use a fresh private2MiB subdirectory inside the
original32MiB importance-control envelope after fresh original/family/reader
liabilities and shared20GiB/5000-file reserves. No new shared files, feature copy,
checkpoint, model fit or submission. Existing intent/root means inspect read-only,
never resubmit. Only aggregate results/hashes leave the owning node.

A favorable fidelity result can justify considering a separate exact-recovery
sampler comparison against completed influence0.949; it does not authorize an
accuracy claim or alter the healthy family fit. A negative result is preserved
and the mechanism is not expanded merely because its coverage theorem is true.
