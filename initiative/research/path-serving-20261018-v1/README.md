# Streamed inference for the new coherent-path model

Prepared while the original integrated recovery955582 and full aligned/null
jobs955711/955712 await H100 allocation. This is new deployment work, not an
old-data/model audit, another reproduction, measured GPU throughput or an
accuracy result. It does not change any queued scientific source or submit a
Kaggle notebook/entry. Training need not wait for this module.

`serving.py` exports only the final68560-parameter path head from an actual
3epoch/843update full checkpoint plus its matching final receipt. The complete
optimizer, clinical rows, targets, predictions and MRI pixels are excluded. It
retains exact original parent, scientific/worker source, integrated recovery,
full checkpoint/terminal hashes and FP32dense policy. The unchanged full N3E16
CoAtNet parent must be supplied separately; this is neither a DINO export nor a
replacement for the R6ensemble. Source/license provenance travels with the full
scientific bundle and parent, not a new licensing assertion in this head.

`PathInference.predict_csv` consumes the same StreamedFP32PathData provider as
new training. It holds one study's maps at a time and writes sigmoid probabilities
without rounding or thresholds. Finite13column output, unique study IDs and
complete successful iteration are required before publication as submission.csv.
A timeout/failure retains only the clearly named partial file. Existing outputs
are never replaced. Deadline means remaining inference time; a future notebook
must subtract model setup/preprocessing from its total9hour kernel budget.
The deadline is not evidence that the model will finish within that budget.

For twoT4s, construct one parent/provider/PathInference per independent process
on cuda:0 and cuda:1. `ShardedProvider(rank=0/1)` divides whole studies by index,
with no change to within-study math or image exposure. `merge_shards` requires
both complete lanes, exact original test.csv study membership, no duplicates or
extras, and reorders rows into original test order. No DataParallel window
reshaping, precision switch, sparse replacement or budget fallback is added.
A final single-model head must be chosen from actual prespecified held-out
results, not selected automatically by this code.

Eleven focused synthetic wrapper tests passed: final-head-only export, correct
schema/order/unrounded probabilities, checkpoint and parent identity, no
incomplete/deadline/duplicate publication, immutable output, and two-lane byte
identity with the single-lane synthetic CSV. These do not test MRI accuracy or
T4runtime. Ruff passes for these two new files only; no old test suite rerun.

Still needed before any actual deployment: successful original training and
fixed-final analysis, actual full-source bundle, competition test-DICOM-to-cache
adapter integration, actual twoT4offline runtime/memory/whole-cohort feasibility,
and fresh allowance/one-time exact notebook submission handling. No Dataset,
notebook, transfer, GPU reservation or competition submission was made here.
