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
