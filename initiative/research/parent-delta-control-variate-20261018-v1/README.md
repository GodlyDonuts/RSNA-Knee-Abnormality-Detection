# Estimate the change from the parent, not the whole study again

Hypothesis: the current16-window student is penalized against a full94-window
teacher even at identical parameters because its sampled pooling differs. Use
common-random-number control variates to separate this sampling noise from an
actual change in the classifier. This is Monte Carlo variance reduction applied
to finding-specific MIL, not a claim of a new general estimator or clinical gain.
Both live full fits and evaluation owners remain unchanged.

Two formulas are predeclared, using the SAME IID16/q draws and frozen parent
moments, with no full-student normalizer available to the estimator:

1. Raw moments: N=N0+mean((a_theta*v_theta-a0*v0)/q),
   D=1+mean((a_theta-a0)/q), then N/D. Here a0 is parent normalized attention
   and a_theta=a0*exp(student_score-parent_score). This reproduces the dense
   parent at zero drift; enumerated expected BCE gradients also equal the dense
   gradient there. N and D are individually unbiased; their ratio is NOT unbiased
   after drift. D can become negative. No clipping or silent fallback is allowed.
2. Stable logit residual: full_parent_logit + sampled_student_logit minus
   sampled_parent_logit, on the exact same windows. Both sampled normalizers
   stay positive. Initial forward noise cancels, but the sampled student gradient
   retains finite-bag bias. It is not the unbiased gradient claim of formula1.

The first CPU screen uses all3887 original fitting-only cached teacher features,
64paired IID16draws and a FIXED synthetic drift grid0/.05/.2/.5/1. Vote drift is
scaled by each parent's finding-wise spread; attention-score drift is Gaussian.
This is simulated drift of actual MRI-derived moments, NOT a trained student's
measured trajectory, held-out disease evidence, or justification to submit.
Read no clinical labels/reports/Gold/confirmation. Preserve all failures, all12
findings, exact seeds, denominator tails, and conditioning. Formula1 MSE must be
explicitly conditional on positive-denominator draws; formula2 reports every draw.

Eleven focused checks cover algebra, reference ratios, negative denominators,
input refusal, deterministic grid, exact parent forward and expected gradient at
initialization, and gradient bias after drift. Tiny floating-point error at zero
drift is recorded, not called bit equality. Source wrapper uses oneCPU/zeroGPU,
4GiBRAM/15minutes,2MiB inside the original32MiB node-control envelope. It reuses
original cached-feature/head/input proofs in place; no array/weight/pixel transfer
or checkpoint obligation. Actual fresh reserve must pass before any write.

Next decision: inspect actual all-study approximation/stability results before
implementing a bounded differentiable trainer. No training or serving change is
admitted by this screen; actual learned-student/recovery/runtime and matched
fixed-final disease-ranking evidence would still be required.
