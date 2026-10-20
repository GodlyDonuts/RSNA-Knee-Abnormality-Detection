# Owner-directed primary research branch: predictive MRI evidence

The owner supplied `knee_predictive_evidence_research.zip`, matching separate
`RESEARCH_PLAN.md`, and explicitly requested following the proposal. Original
files are preserved byte-exact under `reference/`; the independent local run of
its18mathematical tests is `actual-reference-tests.json`. This is a user-proposed
research synthesis, not a proven globally novel or clinically effective method.

**Decision:** make the matched B/C end-to-end experiment the next primary new
training branch: diagnosis-independent masked reconstruction versus diagnosis-
conditioned descriptor prediction with complete-support marginalization. No old
parent distillation or parent-logit shortcut in these new models. Preserve the
healthy source-zero/ranking fits, existing final evaluators and baseline. Do not
extend cached-attention into full production ahead of this branch; original
queued engineering977100 remains unchanged. Existing masks/folds/weak labels
stay fixed. C0 is the required subsequent identical-architecture density ablation.

Implementing now: source-level alias-closed masking before triplet rendering,
metadata-only pixel scaling, fixed64D intensity/DCT/gradient/texture targets,
sparse complete-roster support aggregation and the BCE of the averaged two-mask
posterior. The supplied prototype is a dense mathematical oracle, not the input
loader or production trainer. Direct controls must receive actual observed
descriptors, not be handicapped with context alone. There is no new trained model
or accuracy result yet.

The old uint8 selected cache is **not admitted as raw input** for this branch:
actual original-reader AST facts include a percentile over `allpx`. That is a
possible target-normalization path. The old DINO native cache also uses series
min/max. The new first input pilot therefore reuses the original67raw DICOMs
already privately retained on evc33, uses only encoding-header bounds to scale
pixels, constructs raw-source masks/halo/alias closure, then renders. No source
re-download, accepted-input overwrite, reports or protected data. Descriptor
sufficiency and the distribution shift of this fixed scaler must be measured.

Authenticated GetCompetition CPU977416 independently confirms the deadline:
**2026-10-22T23:59:00Z**. Keep the proposal's latest final-launch target of
October22at12:00UTC, preserving a qualified baseline and nine-hour allowance.
Full B/C funding/throughput/T4 admission are still required; the source's12–16
H100hours per arm is an unmeasured estimate. Never substitute toy/frozen metrics
for the promised end-to-end comparison or silently shrink only one arm.

Primary sources independently checked:
- https://arxiv.org/abs/2111.06377 (masked autoencoders)
- https://arxiv.org/abs/2603.19122 (2026 generative classification/order effects)
Neither establishes this proposal's MRI benefit. Full references and failure
criteria remain in the supplied plan. Protected confirmation/gold remain unused.

## Implementation and actual evidence

Actual CPU977456 passed the original67-DICOM raw intervention test:1072regions,
536observed/mask, three acquisitions/two matrix groups, no target-pixel influence
on masked frames or overlapping triplets. See the adjacent raw-pilot actual.json.
No encoder or diagnosis labels ran in that pilot.

`rsna/competitive/predictive/{inputs,cases,supports,density,model,train}.py` now
implements source masking, full-study variable-matrix assembly, fixed64D targets,
sparse geometry priors, trainable B/C heads and separate AdamW B/C/C0 loop. The
encoder-window budget never drops descriptor/support regions. Two contexts share
one original label/mask and are probability-averaged. The new direct B control
also receives actual observed descriptors. Head-only warmup is explicit in the
recipe, not a permanently frozen encoder. Private immutable checkpoints retain
complete model/optimizer/cursor/history/all RNG for exact restart.

Ten CPU checks passed, including both source interventions and complete new-loop
full/stop/resume equality in B/C/C0. These synthetic tests do not establish real
MRI gradients, descriptor sufficiency, H100/T4 acceptance, accuracy or throughput.
No new predictive GPU/full run dispatched. Raw full-source admission, immutable
parent/base/runtime bindings, actual tiny-MRI overfit/gradient/recovery, resource
funding and earlyT4 timing remain prerequisites. The old unsafe selected cache
cannot substitute for the source-masked input path. Healthy earlier fits continue.
