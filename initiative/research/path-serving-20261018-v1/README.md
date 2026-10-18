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
fixed-final analysis, actual full-source bundle, actual complete MRI/provider/reader integration on twoT4s, offline runtime/
memory/whole-cohort feasibility,
and fresh allowance/one-time exact notebook submission handling. No Dataset,
notebook, transfer, GPU reservation or competition submission was made here.

## Competition-image adapter now implemented

competition_input.py reads explicit StudyInstanceUIDs from test.csv and only
the five image-metadata columns from test_series.csv. Reports, clinical labels
and sample_submission.csv do not enter. It invokes the existing single-decode
series/slice recovery policy into a fresh per-study native384 cache, then feeds
that study to the unchanged StreamedFP32PathData. Successful encoding discards
only this new disposable cache; original DICOMs and cluster caches are never
changed. A failed new cache is retained for diagnosis. Every decoded alternative
counts against the explicit2GiB owned-cache limit and3GiB free-disk reserve.
Two GPU processes use independent owned scratch roots and ShardedProvider.
The new adapter itself does not spawn workers, reserve hardware or enterKaggle.

Five new adapter tests passed, including actual original decoder/native-cache/
dataset wiring on synthetic DICOMs with only the encoder-provider stubbed. Other
cases cover pre-decode capacity refusal, immutable scratch, required test.csv
and safe DICOM identities. This is new glue acceptance, not real MRI/path
inference or T4time/VRAM acceptance. Both full scientific jobs stay unchanged.

## Executable two-T4 runner

`run_two_t4.py` now composes the complete new serving path: actual test metadata,
existing native384decode/recovery, one exact parent+head per T4, private per-study
scratch, complete lane prediction, original-order13column submission.csv merge.
Use it from the exact source bundle with its existing offline dependencies:

```sh
python run_two_t4.py --competition /kaggle/input/rsna-knee-abnormality-detection \
  --work /kaggle/working/path-serving-v1 \
  --output /kaggle/working/submission.csv \
  --parent /path/to/exact-N3E16.pt --parent-sha256 EXACT_PARENT_SHA \
  --head /path/to/actual-fixed-final-head.pt --head-sha256 EXACT_HEAD_SHA \
  --max-seconds 28800
```

This is usage, not an executed Kaggle request; final path-head weights do not
exist until actual training completes. The scientific/core source tree layout
must be preserved in the bundle. Optional --vectorized-scoring defaultsOFF and
still needs actual target-device equivalence/runtime evidence.

The parent counts image paths (never pixels/clinical values) to estimate per-study
work, assigning largest estimated studies first to the currently lighter lane.
This improves estimated load balance over fixed alternating study order without
filtering images or changing per-study operations. Counts are estimates, not a
measured makespan guarantee. Both GPUs keep a complete model; memory is not
pooled. With only one test study, one lane is sufficient; actual twoT4hardware
is still required by this target runner.

All child processes use spawn, one TorchCPUthread and explicit cuda:rank. An
absolute deadline is shared from launcher startup, including child/model setup;
it is not reset per lane/study. Default8hours leaves notebook-installation and
other overhead headroom; the future notebook must account for those phases too.
A failed/timed-out lane stops only this launcher's owned children and cannot
publish partial predictions as submission.csv. Existing source assets/work/output
cannot be overwritten. Raw competition images are never modified.

Future actual run receipts record per-GPU peak allocated/reserved memory, total
VRAM, per-process OSpeakRSS and decode/cache statistics. Sum of individual process
RSSpeaks is a conservative bound for these processes, not a measurement of all
Kaggle auxiliary processes;2GiB RAM headroom is required before final merge.
No actual T4execution or nine-hour whole-notebook admission is claimed yet.

Fourteen focused NEW tests pass for scheduling/coverage, shared deadlines,
owned-process shutdown, immutable asset paths, subset ordering, metadata-only
cost discovery and budget rejection. The CLIhelp runs without touching GPUs.
These tests are engineering acceptance only; old MRI/model recovery and full
repository test suites were not replayed. No GPU job or competition entry added.

## T4 memory and throughput experiment control

The runner now exposes `--encoder-chunk-size` (default12, explicit range1–96).
It reaches both GPU providers and is recorded in their input binding and final
statistics. This changes encoder execution batches only: the same selected
series, dense three-slice windows, FP32 weights and sparse16 parent are retained.
No automatic OOM retry drops windows or silently changes batch size. Actual
encoder time is recorded after CUDA synchronization, separately from decoding.

Larger values are experimental until measured on the actual T4s against12 for
full-study logits, finite outputs, peak memory and total wall time. Floating
point batch-dependent kernels can change results, so no exact-equivalence or
speed claim follows from the wrapper. The training jobs and default serving
recipe remain unchanged. The25 focused adapter/launcher cases pass, including
new32-window propagation and rejection before scratch creation. No old clinical
cohort replay, GPU dispatch, notebook push or competition submission occurred.

## Encoder batch handoff correction

The experimental runner previously accepted `--encoder-chunk-size32`, but the final prediction wrapper still required12, so that requested throughput experiment would fail before study reads. The wrapper now accepts an explicitly bound execution size1–96, defaults12, requires the provider to match that size, and records it in launch and prediction receipts. The exported scientific reference policy remains12; larger execution batches remain candidates requiring actual T4 logits/memory/time comparison.

Seven new regression cases cover a32-window provider through actual prediction/output, pre-read rejection of mismatched execution, and invalid sizes. All43 focused serving/input/runner tests and scoped Ruff pass. The synthetic CSV comparison uses precomputed maps; it proves only plumbing, not actual encoder batch equivalence or speed. No live training source, GPU job or Kaggle version was changed.
