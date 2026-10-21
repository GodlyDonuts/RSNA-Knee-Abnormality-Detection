# Fixed-final C100 readout diagnostic

Newton C100's actual score is **0.692** despite the lowest fitting loss among
the four original arms. The causal explanation is unresolved. This bounded
diagnostic compares the exact scored final checkpoint's context probabilities
with its complete density-corrected probabilities on 128 label-blind,
hash-selected **fitting** studies. Both complementary contexts and FP16
inference match the served readout. No parameters or original inputs change.

The report measures masked soft-label BCE, per-target probability spread and
density movement. It does not binarize weak labels, estimate hidden performance,
select an epoch, or claim independent accuracy. Predictions remain private on
the original owning node; only aggregate statistics and provenance are exported.
Four focused local tests cover selection, masked soft BCE, missing targets,
boundary predictions and nonfinite rejection. These tests are not actual GPU
execution. Actual dispatch and completion receipts must establish execution.

Single-use coordinator `dispatch_once.py` stages on one CPU, then requests one
H100/two CPUs on ordinary physical-two-GPU evc27, with a 25-minute cap and
03:00 UTC October22 deadline. It does not displace other jobs. Existing original
model/runtime/cache are reused; fresh quota must retain all prior liabilities
plus 16 MiB private node-local output and the standing reserves. No shared
model/storage growth, new dataset, training, or Kaggle mutation is authorized
by this diagnostic. Original Colab publisher and consumer8603 continue unchanged.

## Actual result

Original GPU1002912 completed0:0 in56seconds; readout itself took42.16seconds.
CPU1002981 collected the unchanged sanitized receipt, bound to exact model,
source, fitting roster and selected subset hashes. Context-only BCE0.588540
versus full0.559755 (difference -0.028785), with improvements in all12 targets.
Mean absolute probability shift0.070665; all predictions finite and variable.
Per-case arrays stay private on evc27. [Original receipt](actual-complete.json).

This rules out neither generalization failure nor a harmful effect on hidden
ranking. It does show that dropping the density term cannot be justified by
claiming it worsened these fitting predictions. The next candidate needs an
independent reason to outperform, not another fitting-loss comparison. Keep
the scored0.692 regression, no ensemble promotion, and original Colab handoff.

CPU1003000 then analyzed the already saved private arrays, without another model
forward or new case/label access. Across targets, context-versus-full Spearman
correlation is **0.978724–0.997608**; only **1.75–5.14%** of non-tied study pairs
reverse order. The per-target mean shift explains **87.48–98.73%** of the mean
squared log-Bayes factor, and predictions move downward for96.88–100% of cases.
This supports describing the correction as predominantly a calibration shift
on this fitting sample. It does not establish hidden-set behavior or prove
which earlier modeling choice caused the low score. Lower fitting BCE alone
is therefore a poor promotion criterion for this branch.
[Actual ranking analysis](actual-ranking-analysis.json), [exact observer](rank_analysis.py).

The earlier contrast pilot984784 already compared normalization on12 fitting
studies and did not establish an improvement; do not repeat that experiment.
If continuing this failure investigation, isolate the fixed parent readout
from the fresh context classifier and their actual input contracts. That is
a next-step proposal, not a dispatched job or a justified new competition
candidate. Do not promote context-only/dense96 merely because this model
has a low score.
