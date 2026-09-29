# MRI input recovery, unchanged DINOv3 probabilities

The owner authorized fixing the remaining deployment gaps and making one new
code submission on September 28. The prior entry 56550973 remains failed with
zero output and no score. Its hidden traceback is unavailable. The tied-position
failure reproduced in [comparison-v2](../kaggle-successful-submission-comparison-v2/)
justifies this correction; it does not establish the hidden root cause or
promise hidden success.

## Corrected data boundary

The deployment wrapper reads each source slice through the unchanged pinned
DICOM decoder. Expected decoding/data exceptions exclude that actual source
slice and record its path and exception. The remaining real slices enter the
original cache transform; no neighbor pixels, synthetic slices or positions
are invented. Filesystem, transport and unexpected runtime exceptions propagate.
All-source-invalid series are rejected so a healthy alternate can be selected.

After cache preparation, the wrapper calls the original model dataset's
`_positions` validation before accepting a series. A tied-position series is
rejected at this boundary and a healthy same-plane alternate is tried, instead
of letting the later model constructor abort the entire chunk. Cache integrity
errors remain fatal. Bad series caches are removed during normal chunk cleanup.

A later Python input-handling failure can be isolated by splitting a chunk into
smaller study groups in distinct attempt directories, ending at one study.
Isolation requires an MRI data-module traceback and an expected data exception;
model/source/receipt corruption, CUDA failures, signal termination and filesystem
errors stop the run. Each failed attempt retains its logs. A wholly unusable
study receives the existing explicit 0.5 fallback; unrelated studies retain their
model probabilities and test.csv order. A run without any verified MRI-backed
model prediction cannot publish an all-fallback submission. Final output is
validated and then atomically renamed to `submission.csv`.

The exact training runtime, checkpoint, FP32 inference, slice sampling, physical
features and sigmoid operation are unchanged. This is not rank fusion, decimal
rounding, calibration or a newly trained model. The new notebook enables the
three explicit recovery flags; default legacy wrapper behavior remains available
for historical replay.

## Verification

All 1,036 repository tests, Ruff lint/format, mypy for 35 source files, archive
verification and complete v2 coverage passed. The 42 targeted tests passed, including all existing chunking regressions and
new real DICOM/model-route tests. The latter verify:

- Duplicate positions select a healthy alternate before dataset construction.
- One corrupt source slice preserves the real healthy SOP membership.
- Healthy source cache bytes exactly match the original transform; recovering
  two healthy slices produces identical cache metadata/bytes to directly caching
  those same original files.
- Real CPU model inference completes on duplicate-position and corrupt-slice
  fixtures with finite, non-neutral raw sigmoid outputs.
- A later single-study input failure cannot discard unrelated predictions;
  output identity, order and unchanged example probabilities are verified.
- All-study failures, CUDA/integrity errors and corrupt successful receipts
  cannot produce a misleading fallback-only or unverified submission.

Run from the repository root with the pinned pydicom dependency available:

```bash
python -m pytest -q tests/test_kaggle_chunked_inference.py tests/test_kaggle_input_recovery.py
```

These tests are diagnostic fixtures, never competition weights or submissions.
The historical comparison script binds the old wrapper hash; replay that script
from its original revision rather than assuming it targets this new wrapper.

## Candidate and deployment gate

Use the unchanged global-CLS fold-2 scheduled epoch-8 checkpoint SHA
`eb280864bd1c2f896210ce360e1ae36d59649c854a9051523c2454a34d08c7b1`,
training Git `b74060b98f01bff1b924070ccba5ce44dd002f40`, and shared private
cp312/cu130 wheelhouse. Build an exact projection with
`--preserve-training-code` so every `rsna/**/*.py` byte remains the training
revision. The new distinct candidate slugs are
`godlydonuts/rsna-knee-f2e008-inputguard-eb280864bd1c` and
`godlydonuts/rsna-knee-epoch8-f2-inputguard-eb280864bd1c`.

At initial preparation, Newton DNS failed and Stokes SSH timed out. No new
Dataset, notebook, code submission or local ledger was created. Restore UCF
network access, inspect actual jobs/quota and the canonical Newton ledger, then
reconcile failed 56550973 before a distinct corrected publication. Use only
`/home/sa305415/rsna-knee/kaggle/account-ledger.sqlite3` and the one coordinator;
preserve the old failed operation, all earlier uncertain slugs and their history.

Prepare the new bundle on CPU-only Newton storage, fully verify the distinct
private Dataset, push the exact private offline T4 notebook once, and check its
actual public MRI-only receipt and unchanged healthy predictions. Only after
those pass and fresh authoritative live allowance permits, submit that code
version once with output name `submission.csv`. The public three-row CSV must
never be uploaded as a competition submission. Report actual hidden status/score
without inferring it from public execution.
