# Exact cached-input execution for the predictive B/C/C0 experiment

Original GPU978305 completed all B/C24-update/full-stop-resume comparisons on12
fitting studies with4 disjoint engineering prediction cases. Complete learned,
AdamW, history, counters, RNG and CSV equality passed. Each uncached full phase
took438seconds/96study exposures, peaking5.89GB. This is engineering evidence;
it does not establish a clinical or competition improvement. C's early loss and
large head gradients remain negative evidence, including the original fractional
warmup overshoot. The original job/source/checkpoints are preserved unchanged.

This distinct bounded job measures the actual cached execution path, corrected
fractional warmup and one batched device-to-host gradient-diagnostic transfer.
The latter retains original per-tensor FP64 reductions and ordered Python sums;
no optimizer gradient, clipping, architecture or objective changes. Contexts are
exact FP32 copies of the16 accepted inputs, transferred privately once from33to27
with independent whole-byte SHA and exact original study/label/mask joins. The
existing parent/runtime and labels on27 are reused without copy.

B/C/C0 each use12fitting studies,2epochs/fourupdates, effective8 including the
smaller4study batch, head-only1, encoder/headLR3e-5; C0 alone has density weight0.
Each requires full/stop1/resume complete state and4x12CSV equality. This is the
first actual real-MRI C0 engine check. It also reports CPU load and full loop
seconds. Four engineering cases cannot select a scientific recipe or model.

New node27 scopes512MiB contexts,12GiB engineering snapshots,16MiB controls retain
all older obligations and20GiB reserve. Shared2MiB/3logs require fresh quota.
One finite coordinator, oneCPU private transfer, at most one1H100/2CPU/20min GPU
job after original978305actual completion and input acceptance. No production,
Kaggle upload or competition submission. Existing intents require read-only
reconciliation. Full B/C should run on independent ordinary H100s in parallel
once full inputs, this actual execution measurement, and serving are admitted.
