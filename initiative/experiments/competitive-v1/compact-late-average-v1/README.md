# Fixed late-parameter average from PaperIII

Before observing derived/final-model scores, fix equal averaging of learnedFP32
parameters from original N3epochs12/14/16. Accumulate in CPUfloat64 and castonce.
Retain ALL epoch16buffers exactly, including BN means/variances/counters. No
recalibration, no averaging integer counters, no different sources/cohorts/runs.
This is a separately identified derived model, not the original scheduled epoch16;
canonical original serving must reject its new engine identity. The unchanged
encoder/head graph adds zero inference parameters. Test on the same1156 weak-label
development cases and record all12 targets, paired component uncertainty, runtime
and failure rate. Do not promote from synthetic tests or choose epochs afterward.

Implementation has13 focused synthetic tests, not an actual averaged model or GPU
result. Wait for actual original14/16/private preservation and separately fund
CPUconstruction/evaluation/serving. Keep the original fixed16 training unchanged.

Actual CPU stage933572 prepared the frozen source85d9939b and builder; exact
stage receipt is authoritative for its returned job ID. CPU933573 was submitted
ONCE under afterok:932869, oneCPU/zeroGPU/8GiB/20minutes. It also requires original
E16 private download/hash verification before creating any output. No average
exists while the parent is running. A parent failure cancels the dependent job;
never infer averaging success or repeat the dispatch. Node source/output1GiB is
separately funded; no new pixels/shared growth/competition submission.
