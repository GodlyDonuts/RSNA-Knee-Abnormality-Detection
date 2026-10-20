# Predictive T4 benchmark with retained measurements

**Pushed once: kernel 139802342 version 1; authenticated UI at 10:41 UTC shows Queued (API unreported)**, `godlydonuts/rsna-predictive-t4-v2-5d32589edb9e`, source SHA `5d32589edb9efe7f68de536bc01a1d2e50a0da555e9c5b10734cf4351aafd737`. [Actual push receipt](actual-push.json). No competition entry was made and no terminal benchmark result exists yet. The original v1 source, version and receipts are preserved.

The actual pre-push check found source-zero E3 **139786921 v1 COMPLETE**, so its unfinished-training allowance was retired. Free quota was **5,376.85 VM seconds**, above the required **4,200** (1,200 for this probe plus 3,000 retained for final source-zero serving). This dated receipt does not authorize another GPU session.

The authenticated original [v1 diagnosis](../predictive-T4-throughput-20261020-v1/actual-cancelled-ui-diagnosis-v2.json)
shows that both GPU futures, finite FP16/FP32 checks, synthetic nonzero-evidence
checks and memory assertions had completed by about 600.7 seconds. Its
550-second assertion ran before saving the measured receipt, and the original
600-second session ended at 608.8 seconds. No actual throughput numbers survived
that ordering. This successor changes measurement retention and the bounded
session allowance; it does not substitute a different model or call v1 accepted.

`benchmark.py.txt` preserves the two public-MRI masked contexts, side 320,
24 context windows, complete target/support rosters, exact inherited CoAtNet
encoder and fresh predictive heads. It times each T4 processing all three public
studies independently. The six timed study/device executions cover only three
unique public studies. Fresh zero-shift probability comparisons exercise the
context prior; a separate deterministic synthetic nonzero-shift check exercises
the evidence path and restores the original zero shift.

The successor writes atomic, finite JSON and flushes aggregate log records after:

- imports and model loading;
- each study's decoding and context rendering;
- each device's warmup and timed FP16 measurement;
- each zero-shift and synthetic nonzero-shift precision check.

The private working root is `predictive-T4-engineering-v2`. Its `progress.json`
is also exported as `predictive-T4-progress-v2.json`. The final measured receipt
is atomically written to both `actual.json` and `predictive-T4-benchmark-v2.json`
**before** runtime classification. Classification is separately exported as
`predictive-T4-classification-v2.json`. A result exceeding the original
550-second target is `measured_slow`; it is retained without an elapsed-time
assertion. Device errors retain sanitized exception types and all earlier
measurements. A session killed before completion still has its last complete
progress file. No MRI identifiers, arrays or prediction values are included.

The new session cap is **1,200 seconds**. The canonical coordinator takes fresh
account capacity/quota and exact original source-zero v1 observations before a
push. Required free VM time is **4,800 seconds** while source-zero is unfinished
or unknown: 1,200 for this probe, 3,000 reserved for final source-zero qualification,
and 600 for unfinished source-zero work. Only an actual `COMPLETE` state for exact
kernel 139786921 version 1 retires the latter 600 seconds, leaving a 4,200-second
minimum. A fresh read also requires original probe 139796378 v1 to be terminal.
Neither a dated receipt nor silence grants capacity.

The new root, hash-derived `rsna-predictive-t4-v2-...` slug, version-1 intent and
`licensed-importance-full-fit:predictive-T4-v2:...` canonical key are distinct
from v1. Existing intents require read-only reconciliation. The coordinator
reuses Dataset **12431568 version 1**, exact original source/weight/license hashes,
the original pinned Docker image and private offline two-T4 execution. It creates
no new Dataset, pays no credits, writes no `submission.csv`, and makes no entry.
The original 4 MiB control allocation remains charged alongside a distinct 4 MiB
successor allowance. The owner must review fresh current obligations before
running any later phase. Prepare and push are already complete and must not be repeated.

`run_once.py.txt` provides prepare, inspect, push and status phases through the
canonical Newton claim/ledger. Its status phase attempts to collect retained
progress, measurements and classification from terminal complete, failed or
cancelled versions; missing output is reported rather than called successful.

Focused local tests exercise concurrent atomic progress, invalid-number
containment, measured-before-classified ordering, slow/incomplete outcomes,
timed-before-precision event order, exact source-zero budget observations and
distinct offline version/resource bindings. These CPU tests are not an actual
T4 run. Even a completed public v2 measurement cannot establish trained-model
precision, hidden-size runtime or clinical accuracy.
