# Fixed final-epoch three-fold DINOv3-L baseline

The official `rsna.competitive.evaluation` command combined the three
independently completed, fixed-epoch outer-fold predictions for seed 20260922.
The accepted release and frozen component-aware fold overlay were validated
before scoring. The fold-0 and fold-2 prediction CSVs came from their
documented completion audits. The fold-1 CSV was copied read-only from Newton
`/home/sa305415/rsna-knee/runs/dinov3-l16-production-fold1-b74060b-v2/outer_predictions.csv`;
its SHA-256 `6fa7c7ded2a5bd7ffdeca72a55a33c781a7d7d2f0a9420c9ff31dba301045554`
matches the immutable [fold-1 result](../dinov3-fold1-completion-fold2-production-v1/fold1-result.json).
The [evaluation record](output/evaluation.json) binds all three input hashes,
release, fold files and code revisions. The [canonical predictions](output/oof_predictions.csv)
contain exactly 3,403 distinct accepted observed-training studies and no
reserved confirmation or gold rows.

The predeclared primary metric weights each fold's target AUC by that fold's
number of observed positive-negative study pairs, then averages all twelve
targets. It is **0.7522657161782244 macro AUC**. The separately reported
pooled OOF macro is **0.725385883462049**; independently trained fold scores
have different scales, so pooled cross-fold comparisons are secondary. These
are internal report-supervision results, **not Kaggle scores**.

The strongest actionable gap is MCL, **0.498 pair-weighted AUC** from only
seven observed positives across the three outer folds (2, 3, 2). Lateral
meniscus is **0.689**, with 149, 149 and 148 observed positives by fold.
Fracture has only 2, 1 and 2 observed positives, making its **0.776** aggregate
fragile. Synovitis has just five observed negatives per fold; its variation
from 0.950 to 0.540 to 0.798 illustrates the same concern. Per-target fold
counts and AUCs are preserved in [metrics.json](output/metrics.json).

This baseline does not select an epoch, fit a blend, or justify promoting a
new architecture. The next controlled test in the
[post-fold-0 plan](../../../docs/post-fold0-dinov3-plan.md) is target-specific
patch-token/slice attention using the same accepted data and folds, with
lateral-meniscus ranking as the main hypothesis. Hidden Kaggle deployment
must also be repaired before a leaderboard comparison is possible.
