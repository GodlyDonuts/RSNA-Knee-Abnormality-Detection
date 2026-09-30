# DINOv3-L plan after the first full fold

**September 30 next controlled capacity experiment:** the matched sampled-input
[rank-32/rank-64 protocol](../initiative/experiments/competitive-v1/adapter-rank-capacity-v1/)
is prepared and validated against complete production graphs. Rank 64 doubles
encoder adapter capacity to 20,447,232 parameters while retaining 120 residual
modules, the frozen original encoder and the same data/input/schedule recipe.
A real exact-recovery pilot is required after one of the two current GPU jobs
finishes; nothing is dispatched or reserved for rank 64. A higher rank is a
testable capacity hypothesis, not a promised score improvement.

Whole-series original-base/delta restoration passed an exact private offline
T4 six-MRI full-versus-delta comparison. The three public MRI studies took
96.26 seconds and three profiled large studies 186.35 seconds with no fallback.
This clears the reconstruction gate; production accuracy and representative
hidden-scale inference remain separate. Whole-series control epochs 1–2 and
all-development epochs 1–7 are privately fully verified. Both fits continue.

**September 30 actual competition result:** sampled A32 fold-0 epoch 10 scored
**0.769** in successful exact Kaggle submission **56697342**, versus the earlier
solo frozen DINOv3 best 0.638. Its internal 0.8620940794 is a different measure
on report-supervised development data. The broader recipe changed several
settings, so this is not an isolated estimate of adapter rank's effect.
All-development A32 final fitting and the same-fold whole-series control remain
live; completed primary epochs 1–7 and control epochs 1–2 are privately verified.
Use the requested all-data final checkpoint after deployment checks, then judge
whole-series fixed-final accuracy against the matched sampled control. Reserved
confirmation and gold remain untouched. The owner permits worthwhile checkpoint
submissions; use evidence and exact offline acceptance within fresh live allowance
rather than filling five slots with redundant early snapshots.

**September 29 completed adapter update:** A32 fold-0 production completed ten
epochs successfully. Its fixed-final internal macro AUC is **0.8620940794**,
versus **0.7783102649** on the identical original fold-0 holdout; every target
improved on this split. All ten private checkpoints are fully download/hash
verified. See the [complete result audit](../initiative/audits/dinov3-adapter-a32-fold0-result-v1/).
No A32 Kaggle score exists. Recipe changes include precision, head learning rate
and scheduling, so the prepared matched frozen control is still useful. Native
input detail and explicit slice interaction remain untrained future experiments.
Fresh storage acceptance is required before additional full runs.

**September 29 adapter direction:** The owner's newer request is to keep original
pretrained weights frozen while learning substantial internal adapters and
improving use of each MRI series. The current
[adapter and supervision plan](dinov3-adapter-and-supervision-plan.md) supersedes
the historical next-experiment order below. It begins with a controlled rank-32
adapter comparison, then separately tests native detail, sequential coverage and
qualified label confidence. Public source comparison has started and exposes
dependent label tables and differing missingness semantics; it is not a new
training-label release. The subsequent completed A32 experiment is recorded above.

**September 29 public-leaderboard update:** Five matched fold-2 global-CLS
checkpoints scored 0.611/0.628/0.637/0.638/0.635 at epochs 2/4/6/8/10.
Epoch 8 is the best observed public score; epoch 10 has already plateaued.
The scored OSS-copy control is 0.942. Hidden code execution now succeeds. The low rank-based score could involve
model quality, labels, domain shift or fallback-heavy hidden inference; it
does not diagnose one cause. Prioritize target-label
checks and a common-fold OSS-ensemble comparison before another later-epoch
submission; then test bounded encoder adaptation or higher-resolution spatial
modeling. Do not use public scores to fit a blend, touch reserved sets, or
claim an isolated DINOv3-versus-DINOv2 result.
[Live receipts and analysis](../initiative/audits/kaggle-checkpoint-comparison-v1/).

**September 24 three-fold update:** All three fixed-final ten-epoch DINOv3-L
fold runs are complete, and all 30 epoch checkpoints are privately preserved.
The predeclared positive-negative-pair-weighted outer-fold macro AUC is
**0.7522657161782244** on 3,403 accepted observed-training studies; the
secondary pooled OOF macro is **0.725385883462049**. MCL has only seven
observed positives across the three folds and 0.498 pair-weighted AUC;
lateral meniscus has 0.689. See the [exact three-fold evaluation](../initiative/audits/dinov3-threefold-oof-v1/).
These are internal report-label scores, not Kaggle scores. Complete the
hidden-scale deployment diagnostic and then test the predeclared patch-token
attention challenger against this frozen baseline.

The frozen DINOv3-L global-embedding baseline completed ten epochs on fold 0
using two H100s and four CPUs. It made 2,810 optimizer updates in 8,014.1
seconds (4.45 allocated H100-hours) and scored **0.7783102649 macro AUC**
against twelve accepted report-derived targets on 1,156 outer-fold studies.
The final epoch was selected by the fixed schedule, not by those labels.
The result, exact checkpoint and predictions are preserved in the
[run audit](../initiative/audits/epoch1-baseline-and-epoch-preservation-v1/).
This is one development fold. It is not a Kaggle score, a clinical validation,
or evidence that ten epochs is the best stopping point.

The lowest target AUCs were MCL **0.5973** and lateral meniscus **0.6472**.
The MCL result is fragile: the entire 3,904-study development pool has only
seven accepted MCL-positive leakage components, and fold 0 has two. Removing
either fold-0 positive moves MCL AUC to **0.4514–0.7431**. Lateral meniscus
has 149 observed positive and 821 negative outer-fold studies, so its weaker
ranking is a more actionable test of whether global slice embeddings lose
focal meniscal detail. The [independent score recomputation](../initiative/audits/post-fold0-score-analysis-v1/)
preserves all twelve observed-cell counts and source hashes. High AUCs for synovitis **0.9504** and fracture **0.8494**
must likewise be interpreted with their sparse opposing classes; they do not
justify shrinking those tasks or inventing labels.

The current Kaggle notebooks use MRI only, an exact private offline wheelhouse,
and a version-bound public-example receipt. Epochs 1, 2, 4, 6 and 10 were
submitted once; Kaggle later marked all five hidden reruns complete with
unhandled notebook errors and zero output bytes, so none received a score. Epoch 10's
exact version-1 notebook passed its receipt and was submitted as entry 56485356.
Epoch 8's original notebook slug remains uncertain; a distinct exact-source
private version-1 notebook completed its MRI-only public-example receipt and
must not be submitted in version 1. A three-study public
execution checks the package but cannot establish hidden-test speed or AUC.
The [failure audit](../initiative/audits/kaggle-hidden-rerun-failure-v1/) records
the exact API outcomes and an unconfirmed three-row roster hypothesis. Correct
and validate hidden-size MRI-only inference before spending another slot.
Do not choose a new training recipe from an unscored or failed code submission.

## Next bounded experiment

1. **Complete the monitoring sequence and diagnose deployment first.** Record
   exact notebook version, Kaggle submission ID, final status, public score and
   hidden-test runtime for each scheduled epoch. Respect the live shared account
   allowance. If a hidden rerun fails or exceeds the nine-hour limit, fix and
   remeasure that exact MRI-only inference path before using leaderboard deltas
   to judge the training recipe. Keep the three-row public CSV out of submissions.
2. **Measure the same DINOv3-L recipe across all three folds.** Run folds 1 and
   2 with the unchanged accepted release, component-aware memberships, weight
   revision, input policy, seed, optimizer steps and fixed final epoch. Start
   with one two-H100 node/four CPUs, then use a second only for useful independent
   work after its pilot is sound. On fold-0 elapsed time, two more ten-epoch
   folds would cost about 8.9 H100-hours; queueing, cache locality and other
   nodes may change wall time. Report each target's positive/negative component
   counts and within-fold AUC, the predeclared pair-weighted fold aggregate and
   pooled OOF macro separately. This is the minimum evidence for a baseline
   ensemble and for judging any architecture change.
3. **Challenge global embeddings with spatial evidence.** Keep DINOv3-L and the
   same cached images, folds, labels and optimization budget. Compare the current
   global slice vector with registered patch-token pooling plus target-specific
   slice/series attention. The primary hypothesis is better lateral-meniscus
   ranking without broad target collapse. Patch tokens are a controlled
   architectural test; upsampling the 256-pixel cache is not native detail.
   If patch pooling helps, prepare a separate provenance-checked native-DICOM
   transform and test actual higher-resolution crops as a second change.
4. **Test limited encoder adaptation only after the baseline comparison.** A
   bounded last-block or low-rank DINOv3-L fine-tune may adapt natural-image
   features to MRI. Match data, folds, steps, augmentations and inference budget
   against the frozen model. Choose learning rate and stopping inside each
   training fold or predeclare one fixed schedule; never select on the outer
   fold, the 441-study confirmation set, the 58 official studies, or repeated
   public-leaderboard peeks. Track class exposure, especially the five fracture
   and seven MCL positive components, and reject a gain carried by one component.

Promote a challenger only after repeatable within-fold improvement across the
three folds, no consistent supported-target collapse, and a verified Kaggle
offline runtime. Compare an equal-cost frozen DINOv3-L fold ensemble before
adding an expensive branch. If clinical supervision remains the bottleneck for
MCL, Baker's or fracture, a blinded reader pilot is more defensible than
aggressive pseudo-labeling; no such new labels are claimed or included here.
The current 441-study weak-validation partition remains reserved for one
predeclared confirmation after the shortlist and blend rule are frozen.
