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
