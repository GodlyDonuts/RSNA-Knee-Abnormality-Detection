# Large-study preparation stress test

Independent private CPU-only Kaggle engineering run, no model/labels, GPU
allocation, Dataset upload, clinical prediction or submission. Healthy original
Newton and Colab fitting and their final handoffs remain unchanged.

Inventory accessible training-series directories for at most300seconds. Select
the union of four largest frame counts and four largest estimated FP32 pixel
volumes (first original matrix header per series). This estimate is not a proof
of the true maximum and selection reads no reports, clinical labels or scores.
No protected cohort is used for clinical/model selection; these are strictly
source-size engineering observations.

Run original and prospective preparation in distinct CPU children on each real
selected study. Each child gets at most360seconds and is stopped at an observed
24GiB RSS bound; parent lifetime is bounded at2400seconds. Complete context
bytes/metadata/supports are hashed only in RAM for equality; no pixel signature,
UID, source path, image or tensor is emitted. Child errors are sanitized and
retained as failures, including partial coverage. Actual memory caps and
timeouts are not converted into success.

The new preparation is exactly the separately committed selected-frame,
descriptor-batch and original-first resilient decoder. No live science or
serving owner changes. Original-first ordering can favor the successor through
filesystem caching; timings are diagnostic, not controlled causal throughput
or hidden-runtime proof. CPU host differences and absence of the learned GPU
model are explicit. Any discovered failure must inform a distinct correction
with original artifacts preserved.

Funding: at most32MiB new node-local coordinator/control/receipt storage, no new
shared artifacts or model checkpoints. Fresh evc27 free bytes must retain the
conservative518GiB other scopes plus20GiB floor. Shared123MiB/370future files
and20GiB/5000file reserve remain. Same canonical ledger/claim serializes the
one private CPU notebook push; an existing intent requires read-only
reconciliation, never repeat a notebook version on uncertainty.

Actually pushed once by zero-GPU CPU999559: private notebook **140052135v1**, `godlydonuts/rsna-predictive-largecpu-fd60a32fac50`, exact source `fd60a32fac5019bba7e2cf89ad639f9b69ddd0d8123d61d7efb10fd82e563145`. Read-only CPU999570 confirmed RUNNING, private/offline/GPU disabled. Four focused checks passed, including actual child stdin/import transport and bounded timeout failure. No clinical or hidden inference result yet. [Actual start](actual-started.json).

## Actual completion

Exact private notebook140052135v1 completed in1184.2seconds; original measured
experiment1158.85seconds. The300second inventory covered **2,830/4,407**
accessible studies, and selected **seven** by the declared size-only rule.
Every original and successor preparation completed, and all seven complete
contexts were **bitwise equal**. The largest estimated pixel case peaked at
6,654,279,680bytes original versus4,737,122,304bytes successor. Aggregate
preparation was 517.852→281.579seconds, but original-first cache ordering
confounds a causal speed claim. These are fresh CPU children, not a long-lived
GPU submission or hidden-data test. Both original and corrected preparation
succeeded, so this does not identify the hidden exception.
[Original complete receipt](actual-complete.json), [bounded analysis](analysis.json).
