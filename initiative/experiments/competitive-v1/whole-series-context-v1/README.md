# Whole-series DINOv3 A32 context experiment

September 30, 2026. This is a separate candidate from the running all-development
fit and the submitted original A32 epoch 10. Accepted supervision, per-target
masks, source weights, leakage folds, reserved confirmation and gold stay fixed.
No teacher or ensemble is introduced. The sole MRI classifier still uses the
official DINOv3-L with all 120 rank-32 residual projections.

The implemented loader can encode every cached source slice from every accepted
series. It pads varying study/series depths with explicit masks rather than
repeating slices or silently imposing the old six-series/16-slice caps. Encoding
remains chunked; no giant pixel montage or full-volume self-attention is used.
Inputs remain 224 from the verified 256 cache: native detail has not been restored.

A 66,433-parameter message module mixes preceding/current/following compressed
slice features inside each series. It uses signed physical gaps when known and
ordinal offsets otherwise; it does not invent millimeters or registration across
sequences. The output starts at zero residual. Shared parameters and CPU RNG
are identical to the same-seed control at initialization. Target-specific slice
and series pooling still applies supervision once per study, not per slice.

Compare three fixed fold-0 candidates: the existing sampled A32 reference,
all-slice/all-series A32 without context, and the same full-coverage recipe with
neighboring context. The context/control pair has identical image coverage,
shared initial weights, seed, optimizer schedule and accepted masks. This keeps
coverage and sequence mixing distinguishable. Additional outer folds and a
predeclared confirmation evaluation are required before a broad accuracy claim.
Rank 64, native resolution, public labels and teachers remain separate changes.

The prepared two-epoch/16-study pilot is diagnostic only. It must demonstrate
finite real optimizer progress, all adapters and context learning, immutable
base tensors, strict checkpoint reload, exact interrupted/resumed state and
predictions, and safe largest-examination memory/throughput before production.
The prepared ten-epoch control and challenger are NOT dispatched jobs. Recheck
Slurm and projected byte/file quota before any launch; never exceed two GPU
nodes. Full production snapshots cannot be funded by the present storage
projection without another verified retention plan.

The real pilot may keep full checkpoints temporarily on owned job-local disk for
recovery comparison, then preserve a separately typed, hash-bound diagnostic
adapter/context delta and optimizer state on private storage. That compact file
is not a competition model or the production trainer's resume format. Preserve
the official base revision, license and exact source/config/hash receipts.
