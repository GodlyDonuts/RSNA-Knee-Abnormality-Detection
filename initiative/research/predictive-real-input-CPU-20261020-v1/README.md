# Retained MRI CPU preparation measurement

This bounded measurement compares original rendering with the already prepared
descriptor-once serving helper on the original 16 fitting engineering studies.
It uses their existing lossless integer cache and original whole-byte/stat proofs,
removes cached descriptors in RAM, and computes both complete complementary views.
Every image, target descriptor, mask, metadata field and support prior must remain
bitwise equal. Baseline/optimized order alternates by study.

One two-CPU, zero-GPU, 8-GiB, ten-minute allocation on owning evc33 reads existing
files; it writes no remote files, downloads no data and reads no diagnosis labels.
Two workers use one Torch thread each. Model fitting, GPUs, Kaggle, raw DICOM
decoding and clinical evaluation are outside this measurement. Its speed ratio is
for rendering under this shared CPU workload, not a hidden-run time guarantee.

`run_once.py` binds committed program/helper bytes, prepared science44cc and the
original16 index. A durable exclusive local intent prevents repeating an uncertain
dispatch. Only the aggregate terminal receipt goes into Git; raw inputs and private
index/source metadata remain on their owning node. Source files for healthy jobs
are not modified. This measurement does not replace trained T4 qualification.

## Actual result

CPU **981081 COMPLETED0:0**, October 20 15:19:21–15:22:59UTC. All 16 original
studies passed complete bitwise equality. Paired per-study rendering sums were
**217.376s original versus150.031s optimized**, a **1.4489× ratio / 30.98% reduction**.
The full paired measurement, including cache loading and equality checks, took
214.243seconds wall time with two workers. Peak process RSS was 6,193,881,088bytes.
The maximum per-study render times were 34.101s and 27.545s respectively.

[Original aggregate receipt](actual-result.json) binds source 2ed496a5, the frozen
serving helper and original scientific module hashes. Slurm terminal is also
retained in [the independent subsequent observation](../predictive-cache-http-recovery-20261020-v1/actual-followup-progress.json).
Neither timing sum is the total pipeline wall time. This one alternated-order pass
provides no clinical or Kaggle runtime result and causes no live source change.
