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

## Exact recovery with smaller production checkpoints

An additional opt-in `frozen_base_delta` format stores every learned tensor,
complete optimizer state, progress counters and per-rank RNG. It omits only the
immutable pretrained tensors and records each omitted tensor's name, shape,
dtype and SHA256, together with pinned encoder provenance. Saving refuses any
changed frozen tensor or partition. Recovery requires the original verified
weights/config and rejects any different base, missing/extra learned tensor,
incorrect dtype/shape or nonfinite learned value before loading the model.
The existing all-development run and submitted A32 checkpoint remain unchanged.

The new `*-delta-config.json` variants preserve the scientific recipes. The
same-seed full-storage reference config permits a bounded Newton comparison of
full versus delta storage and interrupted/resumed delta recovery. Both context
and no-context CPU recovery checks cover optimizer, RNG, weights and predictions;
actual Newton acceptance is still required before production. The previous
compact diagnostic file remains a separate, non-resumable artifact.

Schema-2 delta inference requires an explicit checksummed original-weight spec.
The current Kaggle packager deliberately rejects these checkpoints until its
separate offline deployment path is accepted. The diagnostic pilot cannot be
submitted. Full-coverage production still needs a fresh projected quota check,
including all ten immutable delta snapshots, one live recovery checkpoint and
the atomic-write transient, alongside the primary fit's remaining full files.
