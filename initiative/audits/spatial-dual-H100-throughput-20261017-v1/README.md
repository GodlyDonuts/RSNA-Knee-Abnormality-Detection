# Frozen original MRI encoding: bounded one-versus-two H100 test

The two full spatial-reader jobs950679/950680 stay unchanged on evc40. This
independent15-minute normalQOS job requests both H100s on **physical2-GPU evc33**,
where exact original full native MRI/cache/runtime/parent and32-study fixture
already reside. No4-/8-GPU node, new data transfer, model fit or optimizer is used.
It may wait for existing allocated GPUs; never duplicate its queued job.

The unchanged original provider encodes the same32 complete engineering studies.
One parent is loaded by its canonical292MB identity check and cloned to GPU1;
both full parameter/buffer fingerprints must match before and after. Two separate
provider instances avoid concurrent hooks or metadata-cache mutation on one
model. Timed order is single/dual/dual/single. Every pass requires complete maps,
base logits, all-masked targets and source geometry/operators to be bit equal to
the original fixture. Cached reference arrays are opened once, retained privately
in bounded RAM and reused across the timed passes; no feature cache is copied or
written. Actual CUDA contexts for both assigned H100s are mandatory.

Report all four timings, per-device peak memory and actual resident memory.
Warm fixture timing and four allocated CPUs cannot establish2247-study throughput
or pure isolated GPU scaling. This does not admit a parallel production trainer,
larger MRI sampling policy, clinical accuracy or competition serving. A useful
speed result would motivate a separate exact-recovery-qualified provider for
richer MRI coverage; it cannot silently change healthy fitting.

Node33-only16MiB benchmark/cache subcap comes from its existing120MiB engineering
allowance, retaining104MiB and every old control. Original source/epoch/full-fit
budgets remain. Existing node33 retired8GiB floor is reused, never subtracted
again; actual fresh reserve is checked on admission and GPU startup. No new
shared bytes/files. [Exact protocol](protocol.json). An actual dispatch receipt,
then actual terminal/result, are required before claiming execution or speed.

Actual CPU950938 verified32original cases/168source metadata entries and the
unchanged118-module source/runtime before submitting **950940 once** on evc33.
It is PENDING Resources/zero runtime at21:39UTC. Existing full950679/950680 stay
RUNNING on evc40; no dispatch, restart or source change affects them. Original
node33 engineering control paid29847bytes;104MiB remains for it and16MiB funds
this benchmark, inside its original120MiB. [Actual dispatch](actual-dispatch.json).
Finite read-only observer87933 is live. [Original launch](actual-observer-started.json).
GPU visibility is left to [Slurm GRES assignment](https://slurm.schedmd.com/gres.html);
the worker requires both allocated H100 contexts and never overrides visibility.
