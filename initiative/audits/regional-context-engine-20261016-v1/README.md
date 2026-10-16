# Matched spatial-information readout: new typed engine

The two arms use the SAME1,334,796-parameter8192-input/128-width/two-layer graph
and identical initialization. Control input is eight copies of the coherent1024
global vector; candidate is that global vector plus seven regional summaries,
all1024channels each. Same parent logits, labels/masks, centers/slots/depth/present,
shuffle seed, CPUFP32deterministic AdamW, schedule, batch16/two views and bounded
0.5logit correction.25%source/75%parent anchoring; full recipe10epochs/2430updates
on3887allowed fitting studies. The historical smaller1024head is a reference,
not the matched input-information control. No CNN parameters are fitted here.

The numerical loop is unchanged released_context_train.py SHA
5addea8212501ba31021c739543a7fa627cf70066ad439fa0889ba5c5b3fa59f.
New regional reader strictly accepts same-forward FP32global/logits plus FP16
regional arrays, original metadata, immutable archive/member bytes and original
fitting-only roster hashes. Role is bound into complete checkpoint state, so a
control checkpoint cannot resume as a regional fit. The new CLI enforces8192/
128/12/two-layer graph, actual imported source admission and exact full recipe.
The original1024serving/exporter stays unchanged and rejects incompatible recipes.

Local targeted38tests/Ruff/three-fileMypy cover exact representations, initial
parameter identity, missing labels, full/stop1/resume complete state and identical
predictions, uneven batches, altered bytes/schema/metadata/roles and deep-state
verification failures. Original CPU937207 stopped at the initial verification, before any optimizer
phase/checkpoint. Read-only937257 proved the zero-start logits were bit-equal to
the parent with zero correction; the unbatched sigmoid oracle differed5.96e-8
from the actual batched predictor. The matched-batch oracle is bit-exact. This
was a verifier shape-dependent CPUrounding failure, not a model/data failure.
Original code, input admission and failure are preserved; no training was repeated.
The distinct corrected admission keeps exact equality and all numerical/data/
model policies unchanged. It is not yet dispatched.
Before full fitting, run six fresh processes: full/stop1/resume for each arm,
32actual cached fitting studies/twoepochs/fourupdates. Require exact complete
model/AdamW/LR/counter/order/allRNG/history and32x12NPYbytes, actual stop1,
nonzero learning, identical initial non-binding training states and shared teacher/
labels/metadata. No engineering scores select a model or enter Kaggle.

The existing512MiBhead/recovery envelope is retained. Engineering writes only
six final/recovery checkpoints (112MiBsub-bound), without redundant engineering
epoch copies. Every actual production epoch still gets an immutable snapshot;
full roles need actual checkpoint-size/retained-liability review before dispatch.
Actual complete3887coherent feature collection, production admission/preservation,
new typed final exporter and realT4serving are separate remaining work.

Read-only CPU937278 additionally checked every original3887cached presence vector:
zero studies lack a present center; minimum60of94. Thus the active full feature
pass's center-presence precondition holds on the previously verified cohort.
No new image inference, transfer, labels or metrics were used for this check.
