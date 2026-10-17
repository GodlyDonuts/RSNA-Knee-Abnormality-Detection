# Original S6 classifier and finite offline reproduction owner

Actual CPU **942136** completed the classifier reference at **04:29 UTC**.
All **three original public MRIs / 1,586 windows** scored with finite
3×1×12 logits, zero fallback and probability range **0.04317–0.95459**.
Elapsed **355.95 s**, peak child RSS **5.83 GB**. The original statistic named
`gpu_s` is **CPU time in this run**, not evidence of a GPU execution.
[Actual reference and source/model/input hashes](actual-CPU-classifier.json).

Only four exact AST device/report expressions were replaced: the CUDA device
once, capability reporting once, and device-name reporting twice. The unchanged
model/feature/pooling consumed the already fully verified original MRI tensors.
[Actual worker](CPU-worker.py.txt) and [program](CPU-program.py.txt).
This is **CPU FP32 reference acceptance**, not T4, hidden-scale or clinical
acceptance. Original H100 FP32/FP16 fixture probe **942047 remains pending**;
never duplicate it. Its original worker/source/output are unchanged.

## Unchanged-source Kaggle reproduction is now owned

CPU **942176** staged the exact original author source and pinned source/weight
mounts, with **no new Dataset/weight upload or GPU push**.
[Actual stage](actual-offline-stage.json). Six local capacity-schema cases cover
RUNNING/QUEUED/missing/COMPLETE/ERROR/CANCELLED and preserve the original main lane.
[Checks](capacity-schema-six-cases.json).

Finite owner **PID16380 / exec14034** uses [this immutable phase coordinator](phase-coordinator.py.txt)
and [this finite loop](finite-owner.py.txt), 12-hour bound, at most two distinct
public GPU runs, **zero competition entries**. It shares the existing capacity
family with explicit S6 lineage, author epoch16 and project epoch null. Neither
the capacity helper nor original training owners/numerical source changed.

- Qualification: `godlydonuts/rsna-s6-original-t4-check-6e87350d5ae0`, 30 min;
  never submission-eligible.
- Separate offline competition-capable notebook:
  `godlydonuts/rsna-s6-original-offline-6e87350d5ae0`, nine-hour limit; only after
  actual qualification and fresh complete-future budget admission.

At this recorded check neither GPU notebook was pushed: both live T4 training
lanes were occupied and conservative all-future VM funding was insufficient.
[Actual capacity/budget wait](actual-capacity-and-budget-wait.json). No mutation
intent or GPU reservation was created by that wait. The finite owner continues.
Do not restart it, edit its phase program in flight or mistake a prepared slug
for an actual notebook. No S6 project score or teacher complementarity result
exists yet; **0.950 remains attributed to the public author**.

Original full inference executes both CoAtNet and S6 even at W_A=0/W_B=1;
include both in runtime estimates. Source notebook138285808v2/SHA6e87350d,
weight Dataset12589750v3/modelSHA0c7297d5 and source/anatomy Dataset12357722v1
remain unchanged. Restricted bodies, weights, pixels and case predictions stay
private; only source/hash/provenance and aggregate acceptance are mirrored.
