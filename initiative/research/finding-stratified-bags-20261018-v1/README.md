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

## Actual original CPU result and trainer integration

CPU953779 completed the3887-study comparison in34.53seconds. The candidate
reduced mean teacher-probability MSE by8.6365%, with lower MSE on all12targets.
Every generated mixture density exactly equalled its original stored proposal.
The largest FP32-versus-FP64 full-teacher probability discrepancy was1.8921e-7.
Mean unique windows increased14.3098 to14.4480. These are fitting-population,
frozen-teacher fidelity measurements, not held-out AUC or clinical improvements.

| Finding | Teacher-probability MSE reduction |
| --- | ---: |
| ACL | 6.23% |
| MCL | 6.22% |
| Medial Meniscus | 8.89% |
| Lateral Meniscus | 9.18% |
| Medial OA | 7.33% |
| Lateral OA | 6.88% |
| PF OA | 6.29% |
| Effusion | 5.96% |
| Synovitis | 6.32% |
| Baker's | 12.43% |
| Contusion | 12.94% |
| Fracture | 11.44% |

The separate `fit.py` closes over the unchanged actual optimizer engine; no
original module globals are modified. Its IID role reproduces the old complete
numerical state/predictions, and both roles recover full/stop1/resume including
AdamW/LR/cursors/all-RNG/epochs and predictions in synthetic MRI-shaped checks.
A singleton final batch, changed-component/role resume refusals and unchanged
global original engine are checked. The real-MRI diagnostic compiler preserves
all original numerical modules and reuses original pixels/features/weights and
old diagnostic output for complete old-control replay. It has no production or
entry authority. Prepared compilation is not actual GPU acceptance.

A separate capacity closure also recognizes the newer family and distinct serving
ledger keys; the old shared helper only knew older campaign prefixes. Three
zero/one/two-live-actor checks prevent silently ignoring the current family fit,
without modifying any frozen live controller. Unregistered/manual sessions remain
outside that registry; an uncertain push must stop rather than retry.

Actual CPU953915 compiled the derived MRI recovery source7a4e7db6 on the
original owning node. Every original numerical module is byte unchanged. It
uses the existing32-case MRI/features and original accepted recovery output,
with no new Dataset or weight transfer. The bounded diagnostic can use one
otherwise-free two-T4 session for at most1800seconds after expanded known-family
capacity and all remaining original family-work funding checks. It compares
complete old influence-control state/predictions, then full/stop1/resume for
both allocation roles. No production or competition model is accepted by this
engineering decision; full fitting/storage/deployment require separate review.

Actual diagnostic139349540v1/source7a4e7db6 was pushed ONCE viaCPU953953,
finite owner22857. Expanded capacity actually identified the healthy family E2
notebook as the one occupied lane. Fresh85118VMseconds covered51600seconds of
all original remaining family work plus this1800s diagnostic. No new Dataset,
weights or MRI transfer occurred. This is a push, not actual numerical recovery
acceptance; follow the original owner and never retry an uncertain mutation.

Independent preparation may materialize the already-tested fitting-only
3887x94x12FP64 component distributions privately, in a distinct80MiB envelope
covering the tensor, exact fitting roster and one future upload archive together.
Fresh funding includes all original reader/family liabilities and no deletion
credit. This does not fund classifier checkpoints, a Dataset creation, model
fitting or serving. No case rows or arrays enter Git/source mirrors.
