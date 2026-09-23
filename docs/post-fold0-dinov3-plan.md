# DINOv3-L plan after the first full fold

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
