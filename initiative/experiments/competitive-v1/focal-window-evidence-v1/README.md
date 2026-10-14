# Focal window evidence without treating anatomy as uncertainty (issue #37)

Prepared mechanism and precommitted matched protocol; **not fitted, not wired into the
trainer, no GPU job, no new teacher collection, no storage spent**. Synthetic CPU tests
check arithmetic and contracts only; they demonstrate no accuracy.

## Problem

The issue #35 aggregate teacher weight is `clamp(|2p-1| - (max_k p_k - min_k p_k), 0, 1)`
over the ten windows. A tear visible in one window and absent in nine gives a large
spread and therefore weight 0, so the term discards exactly the focal cases we want.
Window disagreement here is *anatomical* (different slices), not model uncertainty.

## Window identity

`window_starts(12, 3)` in the exact source returns `range(10)`. Window `k` is rendered
cache slices `k, k+1, k+2` of every acquired slot (after the renderer's own ordering and
right-sagittal reversal). The teacher's `k`-th logit is one study-wide `Model.forward` of
that window across all slots; `dino_adapter.study_inputs` gives the student the same
`[6,10,3,...]` windows with position = center slice `k+1`. Windows are depth-band indices
inside each slot's 0.2-0.8 band, so the same `k` in different planes is not the same
physical location. `test_window_identity_matches_exact_source_and_student_adapter`
checks all of this against the reviewed source namespace.

## Mechanism (`rsna/competitive/focal_window.py`)

`FocalWindowPool` replaces only the slice/series attention head (identical 70,668
parameters at hidden 128):

1. per window and disease, gated attention over valid slots (disease query) -> window
   feature -> window logit `u = 6 tanh(raw/6)`; the student counterpart of the teacher's
   window logit;
2. per disease, gated attention over valid windows mixed with a fixed uniform floor
   `beta = 0.75 softmax + 0.25/K`; study logit `z = sum_k beta_k u_k`.

Bounds: `z` is a convex combination, so `|z| < 6` and diffuse equal evidence stays at its
own value (no noisy-OR amplification). One window weighs at most `1-0.25+0.025 = 0.775`
and at least `0.025`, so a focal window can dominate a learned selection but cannot alone
make a study certain; under uniform selection a spurious extreme window contributes at
most `6/K` to `z`. Zero-initialized queries make the initial pooling a plain mean. No teacher input
exists in `forward`.

Window-level supervision: `window_teacher_targets` uses `sigmoid(teacher window logit)`
as each window's soft target and its own decisiveness `|2p-1|` as weight; across-window
spread never lowers it. `window_teacher_loss` uses the label loss's effective-batch
normalization; `combined_loss(..., 0)` returns the label loss object without evaluating
the teacher term. Absent studies get zero weight; slot-mask mismatch refuses.

`window_variability` reports `anatomical_window_range` (across windows) separately from
`transformation_spread` (across `R>=2` predictions of the *same* window). #35 holds one
prediction per window, so the latter is `None`; neither is used as a weight.

Adversarial tests: one focal positive, diffuse weak positives, one 1e6 outlier window,
NaN-poisoned empty slots/windows, all-empty study, absent/mismatched teacher rows.

## Matched protocol

Precommitted in [design.json](design.json). Arms on the unchanged RSNA-only public-triplet
fixed-final recipe (same seed, data, labels/masks/folds, encoder, schedule, evaluation):

| Arm | Pooling | Teacher |
| --- | --- | --- |
| A | existing attention | none |
| B | existing attention | #35 aggregate, w=0.5 |
| C | focal window | none (D at w=0) |
| D | focal window | window-level, w=0.5 |

A and B are the #35 design arms and are reused if fitted identically; C and D are the new
fits. Primary: D-A paired macro AUC (existing bootstrap). Gatekept secondary: D-B.
Descriptive: C-A, D-C, ACL/menisci, window-weight collapse, drop-top-window sensitivity.
Fixed final epoch only; no best-epoch, gold, confirmation or label-driven setting.

Failure criteria: engineering stops (test failure, C != D at w=0, binding/roster/slot
mismatch, any evaluation-study teacher row, non-finite); harm (primary upper bound < 0);
no established gain (interval includes 0, keep ordinary); not teacher-attributable (D-C
includes 0); focal hypothesis unsupported (no focal-target direction, or weights
collapse to the single-window bound for most study-targets).

## Costs

- Parameters: +0 versus arm A (head swap, 70,668 each); encoder unchanged.
- Memory: extra activations about `B*K*T*H` = 15,360 floats per study; negligible
  next to 60 DINOv3-L views.
- Compute: encoder FLOPs identical; window loss adds 120 BCE terms per study.
- Teacher evidence: 2,247 x 10 x 12 float32 = 1.08 MB, already within #35's 8 MiB
  artifact; nothing new stored. Fits: 2 new fixed-10 runs (C, D) if A/B are reused,
  otherwise 4; checkpoint bytes per fit as arm A.

## Limits

Teacher training roster unknown: no clean teacher/student validation, eligibility,
clinical truth or teacher-accuracy claim. Public leaderboard scores do not make the
teacher trustworthy. The teacher's window logits were trained to be averaged, not as
standalone window predictions. Trainer integration needs a separate reviewed change.
