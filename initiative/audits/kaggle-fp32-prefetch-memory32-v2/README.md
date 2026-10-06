# Complete notebook/workload RSS check, prepared only

The original prefetch benchmark omitted both preparation workers and the
resource tracker from its RAM sum. The accepted Linux reporter now accounts
for those processes before shutdown. The **notebook launcher also remains
alive during inference**, so v2 adds its own lifetime RSS peak to the six
workload-process peaks, including its bootstrap peak. Independent peaks are
summed conservatively. Eleven tests check arithmetic, omitted processes,
inconsistent records, RAM excess and total wall time including bootstrap.

This is a **prepared private diagnostic**, not a pushed notebook, competition
submission, actual MRI/T4 memory acceptance or accuracy improvement. It uses
the same32 fitting-only studies, original0.804 checkpoint, original saved
1300-study FP32 reference, native384 preprocessing and unchanged numerical
model/inference modules. Every32x12 value must remain exactly equal. No fresh
Dataset or repeated reference inference is needed.

The new two-T4 notebook has a30-minute cap including bootstrap and must account
for the notebook plus all six live workload processes. It measures process
RSS, not total cgroup/page-cache usage. Full hidden9h acceptance remains
unproven. The memory-only source compiled with12 pinned modules,211547 bytes,
SHAc8e63bd416240602d58441a61656ea05684b74cb4ff36cb21982cf925e50521f.
Original scalar/model/preprocessing sources remain byte-identical; changed
modules are the performance wrapper, preparation PID observer and two memory
reporters. The actual Linux877784 six-process receipt/source remains bound.

Both client and canonical Newton coordinator require original private
137329288/v1 to COMPLETE with all1300 predictions exact, finite and no
fallbacks before any new cloud intent. The server rechecks exact source and
actual result, then fresh byte/file/GPU quota, original jobs, private Dataset
versions and canonical account ledger. The actual local push guard refused
the current RUNNING prerequisite **before creating a dispatch intent or
launching any SSH/API/GPU work**. See actual-local-guard-refusal.json.

Local six-process-only v1 was compiled but never pushed; it is superseded by
v2's notebook-inclusive report. Its original preparation is retained. The
first v2 local generator stopped before writing a payload because an inactive
CPU branch shared the generic timeout/return anchor; the corrected generator
binds the reference/performance branch explicitly. No remote/model failure
or scientific restart occurred. A streaming log observer later timed out;
the authoritative kernel status was rechecked, not inferred terminal.

Use only the prepared single coordinator in ignored
artifacts/kaggle-fp32-prefetch-memory32-v2 after actual prerequisite success.
Existing intents/results require read-only reconciliation; no automatic
retry. The healthy875858 fit and queued877134 comparison stay unchanged.
