# One exploratory density-balance intervention

**Prepared; no GPU dispatch or model result yet.** This is a separate engineering
experiment. The original full B/C/C0 owners, recipes and inputs remain unchanged.

The [actual C24 measurement](../predictive-objective-balance-20261020-v1/README.md)
found that weighted density gradients were only 0.0102–0.0140% of exam gradients
across nonencoder parameters on two fixed fitting cases. We therefore predeclare
one intervention: **C100 uses density coefficient 100 instead of 0.1**. Its graph,
likelihood, descriptors, initialization, training order and optimizer stay the
same. This tests whether appreciable negative-reference and contextual
reconstruction supervision changes learning. It does not establish a new
architecture, an optimal coefficient, or an accuracy improvement.

At the measured C0.1 checkpoint, linear rescaling would put the auxiliary/exam
gradient ratio near 0.10–0.14 across heads, 0.60–0.73 in the negative-mean head,
0.79–1.96 at decoder representation `h`, and 0.004–0.005 in the finding-shift head.
These are local raw-gradient projections, not predictions of Adam updates or
encoder-wide balance. The separate context-prior path bypasses `h`. Starting
from the same fresh initialization is necessary; continuing the measured C24
checkpoint would confound the experiment.

## Bounded first experiment

Reuse the original, already accepted evc27 contexts: twelve fitting training
studies and four disjoint fitting engineering studies. Use the original
`39e315c65546aa5205518713bc27349ec63e7538` numerical modules, two complementary
contexts, effective batch eight with the four-study tail, 24 optimizer updates,
2,916-update schedule horizon, head-only updates 1–3, full encoder from update 4,
seed 20261020, encoder/head LR 3e-5, clip 1 and BF16 GPU forwards.

One ordinary H100, two CPUs and a 30-minute bound cover fresh full, stop-after-one,
and fresh-process resume runs. Compare the complete decoded model, AdamW,
recipe/binding, counters, all RNG states and history, plus identical four-by-twelve
prediction bytes. Those engineering predictions must not select a recipe or
enter the competition. Reuse existing original inputs in place; make no new MRI
acquisition or transfer.

Nonfinite objectives, gradients, evidence or predictions, a wrong update/recipe
binding, failed recovery equality, or exceeded storage caps are hard failures.
Retain the actual original failure without an automatic retry. Review the full
exam-loss trace, density above its Gaussian constant, evidence ranges and
clipping before any further admission. A warning is a final six-update mean exam
loss above twice the original C mean, or any exam loss above twice the original C
maximum (0.8219250441), or maximum absolute Bayes factor above ten times the
original C maximum (1.6598255634). These are predeclared engineering review
thresholds, not clinical rejection thresholds or a reason to mutate the live
controls.

Raw total loss contains roughly **91.9 of fixed Gaussian constant** at this
coefficient. That constant has zero gradient and must not be mistaken for an
optimization failure. Twenty-four updates reach only about 4.94e-6 LR; passing
does not establish stability at peak LR or over a full fit.

## Decision after the pilot

There is no automatic full-training promotion. Review actual acceptance,
numerical warnings, source and fresh resource/deadline feasibility. If admitted,
declare a distinct six-epoch C100 run at the same full initialization, 3,887
studies, order, effective batch and 2,916 updates as C0.1/C0/B. Reuse the completed
C0 context placement when available instead of creating another large transfer.
Separately fund its checkpoints and publication; do not consume the primary
predictive deployment reservation or delay the original experiment.

The principal new scientific comparison would be fixed-final C100 minus C0.1.
C0 and B remain useful controls, with all original limitations: the inherited
parent overlaps the weak holdout, Gold58 influenced parent selection, and
aggregate Kaggle scores do not yield paired clinical uncertainty. No tiny
engineering metric, best-epoch choice or coefficient search may be reported as
independent accuracy.
