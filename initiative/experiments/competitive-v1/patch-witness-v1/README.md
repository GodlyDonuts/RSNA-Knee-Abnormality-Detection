# Signed patch evidence beside global MRI context

October3 owner direction: pursue mechanisms rather than more undirected rank,
resolution or epoch scaling; keep GitHub ready for external deep research.
See [the complete research brief](../../../../docs/research-brief.md).

## Mechanism and matched control

Current adapted production pools global slice embeddings. This optional branch
scores local DINO patch features by target before aggregating, keeps top4 patches
per slice/top2 genuine slices per series/strongest genuine series, and adds that
signed evidence to the unchanged global-context logits. Target loss stays at
study level; no patch/slice inherits a diagnosis. Both arms extract the same
patch tokens from the same encoder views, so the control pays the same forward
budget. The exact policy differs only in weight0/control versus1/candidate.

A separately seeded branch leaves the shared initialization/RNG stream unchanged.
The local output is zero initialized: both arms start with identical complete
learned states and zero-weight logits. Padding, including NaN padding, never
enters the encoder or the evidence pool. Whole-exam inputs and context stay
present. No extra crop, encoder, external teacher or proposed anatomical alignment.

This is a literature-informed experimental operator, NOT a novelty claim. Small
lesion dilution is studied by [Attention2Minority](https://pubmed.ncbi.nlm.nih.gov/37890421/)
in histopathology; that result does not establish an MRI benefit. Our architecture
already has target-specific slice/series attention. This branch changes the level
at which signed classification evidence is preserved, not just attention naming.

## Predeclared experiment and rejection

First accept actual official-weight native384 MRI full/stop1/resume engineering
on an allocated H100, with identical original base, initial state, MRI roster,
all decoded optimizer/RNG/counters and predictions. Inspect actual gradient
strength/finite learning and retention of context; synthetic success is not this
acceptance. Stage no full fit until that actual pass and fresh funding.

Then use the exact2247-fitting/1156-disjoint-outer fold0 cohort, accepted original
masks, strict uniform16-slice/max6-series/native384/rank32 recipe, effective8 and
281updates per epoch. Candidate/control share head seed, branch seed, source,
initial state and encoder observations. A fixed3-epoch/843-update screen can
reject a nonpromising mechanism quickly; its endpoint is chosen BEFORE fitting,
not by best epoch. If funded/accepted, subsequent full10-epoch contrast and a
matched second seed/fold determine promotion. Record the actual encoder image/
patch budget and elapsed GPU time. No protected gold/confirmation tuning.

Report all12 original masked outer metrics, support and paired whole-component
uncertainty including undefined draws. Sparse labels limit these conclusions.
A tiny engineering metric does not select weights or show clinical benefit.
Potential failure: artifact maxima, duplicated views, early top-k lock-in, loss
of diffuse/extent signals and scanner shortcuts. Padding tests do not test these
clinical hypotheses. Add predeclared missing-series/unrelated-background stress
probes and review training-only selected regions before interpreting scores.
A matched random-evidence selector or cross-view corroboration is a later
separate contrast, not secretly combined with this one.

The31-cell Effusion overlay is a separate supervision experiment; it is not enabled
here. Mixing changed supervision and aggregation would prevent attribution.
Stanford source-derived weights are not used. No competition deployment requested.

## Actual implementation status

`rsna/competitive/witness.py` implements and validates the finite policy.
`CompetitiveStudyNet`, canonical TrainConfig/binding and strict inference restore
support the optional branch. Absent witness policy preserves old numerical
branches/config serialization; exact historical source remains required for old
checkpoints. Tests cover equal initial states/RNG/logits, finite gradients,
zero masked-image gradient, variable real counts, NaN padding, local evidence
not diluted by appended low evidence, strict state restoration and BF16.
Both zero/one arms exercise the actual canonical trainer on synthetic studies:
full/stop1/resume and supervisor-requested recovery, complete model/AdamW/LR/
counters/RNG equality and exact outer CSV bytes. These are engineering tests.

No actual real-MRI GPU recovery, staged source, new Slurm job, funded production
checkpoint/publication or model accuracy gain is claimed by this protocol.
Existing healthy production C9 libraries/configs were not modified remotely.
