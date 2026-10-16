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
