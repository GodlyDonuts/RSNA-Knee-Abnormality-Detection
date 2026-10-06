# Consistency-EMA (mean-teacher) arm — experiment design

Status: design only — no code, no run config yet. Blocked on a GPU slot
alongside every other arm; written so a future session can implement it
without re-deriving the reasoning.

## What it is

Mean-teacher semi-supervised consistency on the masked cells. The v2 label
manifest leaves ~35–90% of cells masked per target (Synovitis worst at
10.6% coverage). The v2.1 adjudication program recovered positives on the
severity targets, but thousands of cells remain masked by policy
(hedged-only, indeterminate, traumatic-not-degenerative). Consistency-EMA
lets those cells still shape the model *without* assigning them labels:

- **Student**: the model being trained on supervised cells (masked BCE as
  today).
- **Teacher**: an EMA of the student (`teacher = α·teacher + (1−α)·student`,
  α≈0.99–0.999), updated after each optimizer step, no gradients.
- **Consistency loss**: on masked cells, penalize disagreement between
  student and teacher predictions under *different* views — e.g.
  `KL(student_aug || teacher_clean)` or symmetric MSE on logits. The model
  must be invariant to the augmentation perturbation, which forces it to
  commit to internally consistent structure in unlabeled data.

## Why it fits this competition specifically

1. **The masked mass is the dataset's untapped signal.** The v2.1 program
   decided 7,911 of ~8,300 contested severity cells; 460 stay masked plus
   every non-severity uncertain cell. On Synovitis, ~89% of cells are
   masked — consistency is the only way that mass contributes.
2. **Complementary to the label-source arms.** v21/v21pn/v21hc answer
   "which decided labels are right"; consistency answers "what do we do
   with the cells no protocol can decide". They compose: run
   consistency-EMA on top of whichever supervision arm wins.
3. **Natural augmentation asymmetry already exists.** The recipe
   normalizes laterality and applies study-level augments; the teacher
   can see the canonical/clean view while the student sees the augmented
   view — exactly the mean-teacher asymmetry, no extra machinery.

## Design parameters (to be fixed before implementation)

- `ema_alpha`: 0.999 (raise to 0.9995 if updates are <1k/epoch)
- `consistency_weight`: ramp 0 → λ over first ~20% of steps (λ≈1.0 for
  MSE-on-logits; tune on fold-0 only)
- Teacher input: clean (no aug) vs weak-aug — default: weak-aug, since
  deterministic mirroring already defines a canonical view
- Consistency applies to **masked cells only** by default; a variant
  applying it corpus-wide is a second arm

## Implementation sketch (existing seams)

- `TrainConfig.masked_cell` already gates a masked-cell auxiliary loss
  (`masked_cell_entropy` in `rsna/competitive/loss.py`, ESA arm);
  consistency-EMA adds `kind: "ema_consistency"` with fields
  `alpha`, `weight`, `ramp_fraction`, `teacher_view`.
- `CompetitiveStudyNet` needs an EMA shadow — copy-on-init, `no_grad`
  update hook after `optimizer.step()`; state-dict must round-trip
  through checkpoint save/load (smooth-witness already proves complete
  learned-state recovery — teacher params go in the same state).
- Eval keeps using the **student** for checkpoints/selection; teacher
  is training-only machinery.

## Predicted effect and falsification

- Prediction: +small mean AUC over the matched supervision arm, largest
  on Synovitis/MCL (highest masked mass). If the paired comparison on the
  1,156-study dev set shows no movement, masked mass is not the
  bottleneck — close this arm and keep masked-cell handling at
  entropy-max or pure masking.

## Interaction notes

- Run the label-source ablation FIRST: consistency-EMA composes with
  whichever supervision wins; running it before the ablation spends GPU
  on an arm whose baseline may be superseded.
- Do not combine with `graded-u05` — soft labels and consistency target
  the same cells through different mechanisms; run them as alternatives.
