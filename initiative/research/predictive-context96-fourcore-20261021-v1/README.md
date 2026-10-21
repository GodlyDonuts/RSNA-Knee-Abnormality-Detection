# Dense96 four-core serving comparison

Prepared October 21, 2026. This is a distinct successor to the held dense96
public notebook; the original remains unchanged and unsubmitted.

The measured failure to address is runtime: its last public preparation projected
9.44 hours at 1,300 studies/two lanes with setup and tail allowance. The separate
actual CPU thread experiment found 19.95% less warm preparation with two intra-op
threads per lane and <=1.1921e-7 image differences. That does not establish a
trained numerical comparison or a cold runtime gain.

This source preserves the exact C100 final model, all 96 encoder centers per
context, complete descriptors/supports, original decoder and fitting provenance.
On the exact three-study public roster it prepares the two-thread candidate
**first**, then a one-thread reference after joining the first pool. It compares
all context fields, actual trained FP32 and FP16 predictions, and cross-device
predictions. Actual private paired arrays are retained in Kaggle outputs and
independently checked in owning-node RAM. No images or individual predictions
are collected to the Mac/Git. Hidden execution has two threads per lane and no
reference pass. Public qualification keeps strict memory/precision limits.

The old diagnostic only saved hashes and aggregate differences, so its exact
prediction discrepancy cannot be recovered from those files. This new paired
comparison does not invent an explanation or retroactively accept that version.

The public watchdog is 480 seconds, plus 120 seconds startup reservation. The
single coordinator requires fresh >=2,520 free GPU-VM seconds, preserving the
original Colab final deployment's 1,920 seconds. It cannot submit a competition
entry. Existing intents are never retried. Runtime acceptance is a separate
measured decision; no accuracy improvement is assumed.

Four focused tests passed: compilation through the real source layers preserves
model/provenance, actual tensor preparation rejects field corruption, the
coordinator preserves quota/one-shot behavior, and independent paired-array
checking rejects numerical disagreement even when the file hashes agree.

Prepared storage scope: 32 MiB new node-local, 1 MiB shared, zero new shared
files, after prior 518 GiB+144 MiB node and 125 MiB/370-file shared liabilities.
The actual fresh CPU preflight must retain 20 GiB and 5,000 shared files. No
Dataset/model copy, label modification or new training job is involved.

**No GPU execution or competition entry at source freeze.** Read subsequent
original `actual-*.json` receipts before acting. Dispatch through `run_once.py`
with the exact source commit; phases are prepare, push and read-only status.
