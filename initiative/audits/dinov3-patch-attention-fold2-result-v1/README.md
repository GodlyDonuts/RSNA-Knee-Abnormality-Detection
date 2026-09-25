# Patch-attention fold-2 fixed-final result

Newton Slurm job **848958** completed `0:0` after ten epochs and **2,860 finite
optimizer updates** on 2,281 accepted training studies. The epoch-10 checkpoint
SHA-256 is `e9bdd7d535e4d4bc3061343167dcebdeda014910d4897250dfb82b30d695bcaf`.
All ten immutable fold-2 checkpoints were individually published exactly once
to private `GodlyDonuts/RSNA-Data` and fully re-downloaded/hash verified; see
the [epoch receipts](../dinov3-patch-attention-fold2-launch-v1/).

The predeclared final checkpoint scored **0.7197381600 macro ROC AUC** across
all twelve accepted report-supervision targets on 1,122 outer-fold-2 studies.
The matched global-CLS fold-2 result was **0.7590158140**, a patch-attention
difference of **−0.0392776540**. Lateral meniscus was 0.7495561569 versus
0.7529948635, a difference of −0.0034387066. These are internal held-out
results, **not Kaggle scores**. The outer labels did not choose a checkpoint.

The [exact result](result.json), [predictions](outer_predictions.csv),
[run binding](run.json) and [Slurm terminal state](slurm-terminal.txt) were
copied from the completed job. [Independent verification](independent-verification.json)
checks their hashes, exact 1,122-study membership, accepted release and frozen
fold overlay, target masks, all twelve AUCs, matched CLS scores, and checkpoint
identity. No reserved confirmation or gold study entered this evaluation.

The [three-fold summary](threefold-summary.json) reweights each final-epoch
target AUC by that fold's observed positive-negative pairs, the predeclared
primary aggregation. Patch attention reaches **0.7360400218** versus
**0.7522657162** for global CLS across 3,403 outer-fold predictions: a
**−0.0162256943** macro difference. Lateral meniscus improves by
**+0.0185916049** in this aggregate, but the macro and fold-2 evidence do not
support replacing the global head wholesale. The aggregate derives fold-0's
score from its archived result rather than replaying its prediction CSV.
These numbers should guide a new, separately validated experiment, not be
treated as a hidden Kaggle result.
