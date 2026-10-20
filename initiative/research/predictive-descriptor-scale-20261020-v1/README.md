# Fitting-only descriptor scale: separate engineering candidate

CPU **982960** completed the one-time collection from exactly 12 fitting studies
and 45,360 unique canonical regions; all four engineering evaluation studies were
excluded. Measured coordinate standard deviations ranged from
**0.0009531507 to 0.2062999973**. One coordinate used the predeclared 0.001 floor,
so frozen scales range from 0.001 to 0.2062999973, with maximum inverse scale 1,000.
The collector took 0.195 seconds, read no images, clinical labels or models, and
performed zero optimizer steps. Exact private artifact readback and binding passed;
see the [actual result](actual-result.json).

The 4,795-byte private artifact has SHA-256
`e0b57e18bebe7ccc16aff260b05cd02f46be7333a8ac03d7778bb7dc81aae142`.
This measures descriptor variation; it does not establish a training or accuracy
improvement.

The actual C24 diagnostic shows small heterogeneous descriptor coordinates against
fixed unit Gaussian variance. A separate fixed coordinate scale is worth measuring,
but **cannot remove the loss's structural `1/(64*N_observed)` dilution of
diagnosis-specific density gradients relative to exam BCE**. No current training
or serving module imports this new normalizer.

The standalone module gives each fitting study equal weight and each canonical
region within it equal weight. It accumulates within- and between-study population
variance in FP64, freezes `scale=max(SD, 0.001)`, and applies
`z=(descriptor-mean)/scale`. The floor is declared before collection, caps inverse
scale at1,000, and is not selected on validation performance. Marginal coordinate
variance is not conditional noise covariance. It does not fix image contrast.

The one-time CPU collector admitted exactly the original first12 engineering
training cases from GPU978709. It excluded its four evaluation cases and read
only original context manifests, complete descriptor targets, masks and region
counts. The two masks' identical targets are counted once, using the previously
accepted canonical alias-closed region roster. No image or clinical-label array,
model or checkpoint is loaded. Full source/input digests remain bound.

The frozen vectors stay private on evc27 at
`/tmp/rsna-predictive-descriptor-scale-20261020-v1/normalizer.private.json`.
Only aggregate ranges/counts and hashes may enter Git. The new node-local
allowance is1MiB/three small files; zero new shared files/bytes, oneCPU/zeroGPU.
Fresh local reserve includes every existing C0/C100 obligation and the20GiB floor.
No fit, transfer, competition admission or new GPU follows from collection.

An eventual full-data normalizer needs its own **complete3,887-study fitting**
artifact and matched new initialization. The12-case artifact cannot qualify it.
Changing descriptor scale while fixing normalized Gaussian covariance to identity
changes the model's raw-space covariance and gradients; it is an experiment, not
a claim of numerical equivalence to the existing unit-variance model. Shared
Jacobian terms cancel in the likelihood ratio. Tests cover that equality, exact
censoring/null-evidence invariants, study weighting and strict artifact binding.

The primary B/C/C0 and queued C100 experiments remain unchanged. No clinical
accuracy, competition gain or generative-gradient correction is established here.
