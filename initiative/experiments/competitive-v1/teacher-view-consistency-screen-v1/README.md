# Does the teacher disagree with its own anatomical mirror?

The actual parent averages two anatomical-view probabilities during inference.
Current adaptation instead anchors each training view to its own cached teacher
logits. Hypothesis: substantial teacher view disagreement could make probability-
consensus distillation a better invariance target. This is not yet a training
recipe, an accuracy gain, or a new invention.

Read only the original admitted 3,887-fitting-study UID roster and paired teacher
logits in the existing compact NPZ. Never access its source labels/masks, reports,
images, protected clinical values or model weights. Verify the original roster,
array identity and existing whole-input proof; publish aggregates only. No new
MRI inference, transfer, Dataset or GPU is required.

Report all twelve targets: mean and quantiles of absolute probability disagreement;
fractions above 0.05/0.10; strict crossing of 0.5; and all-pairs ordering reversals,
with tied pairs and undefined conditional rates explicit. Per-view MSE to the
probability consensus is a teacher-fidelity quantity, not clinical error.

Predeclared practical triage, **not statistical or clinical acceptance**:

- If every target's mean gap is below 0.005 and 99th percentile below 0.05, treat
  this as a weak opportunity and do not fund another GPU arm from this screen.
- If any target has over 10% of cases with a probability gap above 0.10, or over
  10% strict-pair ordering discordance, retain it as a nontrivial mechanism to
  investigate after current actual results and fresh resource review.
- Otherwise preserve the result as intermediate; do not manufacture a win.

A later matched candidate would change only the teacher target to
`0.5 * (sigmoid(plain_logits) + sigmoid(mirror_logits))`, retaining the original
source term, masks, model, windows/draws, initialization, optimizer and fixed
endpoint. Averaging logits is a different mechanism. A consensus may remove
useful view-specific gradients or preserve shared teacher errors; no improvement
is assumed. Current training must remain unchanged. No GPU experiment is admitted
by this screen, and the exposed teacher provides no independent internal OOF.

Implementation: [screen_teacher_views.py](../../../../scripts/newton/screen_teacher_views.py).
Four focused numerical tests cover tied/identical views, shifted probabilities
without ordering changes, complete order reversal and nonfinite rejection.
