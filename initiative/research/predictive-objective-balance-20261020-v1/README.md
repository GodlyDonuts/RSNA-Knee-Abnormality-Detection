# Actual C24 objective-balance measurement

CPU **981494 completed 0:0** on evc27, October 20, 16:30:21–16:31:33 UTC.
The program measured 68.400 seconds and 2,830,712,832 bytes peak RSS using
two CPUs and zero GPUs. It read the retained original C checkpoint at update 24,
science `39e315c65546aa5205518713bc27349ec63e7538`, and the predeclared fitting
ordinals 0 and 1 from its twelve-case engineering cohort, with both complete
complementary contexts. It performed no optimizer steps or remote file writes;
checkpoint/model state was unchanged.

[Actual measurements](actual-result.json) and the
[bound interpretation](actual-interpretation.json) preserve the result.
The measurement receipt SHA256 is
`6f10fc5a21e8149731e1600088e601b97c16be42b0a9e944baa05c67f6920e2a`.

## What was measured

The density gradient includes its **0.1 coefficient exactly once**. The table
reports `100 × ||0.1 ∇density|| / ||∇exam BCE||`; these are gradient ratios,
not loss ratios or estimated parameter-update ratios.

| Gradient scope | Fitting ordinal 0 | Fitting ordinal 1 |
| --- | ---: | ---: |
| All nonencoder parameters | 0.01024% | 0.01400% |
| Finding-shift head | 0.000540% | 0.000404% |
| Negative-mean head | 0.05985% | 0.07334% |
| Context representation `h` | 0.07924% | 0.19561% |

The density term has no direct prior-head gradient. All-head objective-gradient
cosines were 0.03687 and 0.01646; per-head values are retained in the receipt.
The independently computed total gradient matched the vector sum of the two
objectives, with maximum relative L2 discrepancy below `8.91e-7` across groups.

Evidence was active: removing the log Bayes factor changed the mean posterior
by **0.025159** and **0.020489**. Its effect on the fitting BCE had opposite
signs: full minus context-only BCE was **+0.001033** and **−0.000951**.
Neither averaged posterior had saturated targets at the declared
`p ≤ 1e-4` or `p ≥ 1−1e-4` thresholds. Density above the fixed unit-Gaussian
constant `0.9189385332` was 0.00033484 and 0.00026207.

## Interpretation and limits

At this checkpoint and on these two fitting cases, the measured density
gradient is much smaller than the exam gradient. This supports testing
objective balance in a **separately declared experiment**. It does not establish
an optimal coefficient, a collapse, or C/C0 irrelevance. Evidence already
changes the posterior; the mixed BCE direction does not establish a benefit.

This was CPU FP32 evaluation with detached encoder features. It measured
nonencoder and context-representation gradients before clipping, and measured
neither encoder gradients nor an Adam update. Adam can amplify small
coordinatewise signals; the density term also supplies negative-mean-head
signal absent from exam loss at neutral initialization. Two fitting cases at
an early engineering checkpoint provide no held-out, clinical, first-200
training, full-model, or competition-accuracy evidence. Image summary statistics
include mask fill and do not measure unmasked MRI contrast.

The ongoing model pipeline and frozen B/C/C0 recipe remain unchanged. No
training, new checkpoint, protected-data read, model selection, source download,
or Kaggle operation was performed for this diagnostic.
