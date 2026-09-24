# Fold-2 fixed-final training result

Newton production job **844305** and its CPU odd-epoch watcher **844306**
completed `0:0` on September 24, 2026. The unchanged official DINOv3-L
training recipe completed ten epochs and **2,860 optimizer updates** on
2,281 accepted outer-fold-2 training studies. The epoch-10 checkpoint SHA-256
is `5e400e82c4b50e94a27efcb250fc86d911b2191b6ed04792276fe51d9e093734`.
All ten immutable fold-2 checkpoints are privately published and individually
download/hash verified in the [epoch preservation audit](../dinov3-fold2-epoch-preservation-v1/).

The fixed final checkpoint reached **0.7590158139992688 macro ROC AUC** across
all twelve report-supervision targets on 1,122 accepted outer-fold-2 holdout
studies. [result.json](result.json) contains the per-target scores and measured
coverage. [outer_predictions.csv](outer_predictions.csv) was copied from the
completed job and its SHA-256 matches `predictions_sha256` in the result.
[run.json](run.json) preserves the release binding and explicit exclusion of
the reserved confirmation and gold sets. This is an internal holdout result,
**not a Kaggle score**. The fold-0 and fold-1 scores use different holdouts;
they should not be compared as checkpoint improvements on a common test set.

The completion result, checkpoint preservation receipts, and private
Kaggle diagnostic were committed to GitHub main at `a0011e7`. All 24 small
files in that milestone were mirrored to private `GodlyDonuts/RSNA-Data`
revision `cc472ccbeecd0f40494d14cb2219ea10fc432f41` and fully
download/hash verified; see [the mirror receipt](hf-milestone-verified.json).
