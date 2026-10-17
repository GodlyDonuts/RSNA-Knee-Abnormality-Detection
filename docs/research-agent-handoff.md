# Research-agent handoff

**Verified October17,2026,10:55UTC. Target0.97; actual best0.950.** Read
[current state](current-state.md), [research brief](research-brief.md) and
[execution handoff](codex-continuation.md) before proposing or dispatching work.
Implement directly: Devin is unavailable. Do not equate audit counts or passing
engineering checks with accuracy.

## Current facts a proposal must account for

- S6 exact offline entry**57300990 scored0.950**, COMPLETE/noerror. It reproduces
  external weights/inference, not our source training. nartaaCoAtNet remains0.949.
- Our independently MRI-trained **N3E16 scored0.894**, after0.821/0.869/0.890.
  It trains all73,136,096parameters from ImageNet, not a frozen tiny DINO head.
- Context/regional/full-field variants each scored0.948. Whole94E1 scored0.949.
  No public improvement over its unchanged parent is established.
- Whole94 **finished all1458updates**; all3epochs privately fully verified.
  FinalE3 serving/entry are owned but unscored. Uniform16E3 is RUNNING;
  influence16final remains owned. Never duplicate those originals.
- Expanded ownN3 completed16epochs/5184updates/allprivate verified; weak-final
  macro0.91743158 versus0.92092535 has no established gain. It changes both
  data and updates, so is not a data-only causal comparison.
- Public0.958 reference ran all8unchanged cells in one separate interactive
  twoT4session and passed final3×13CSV/roster/finite/hash checks. VM stopped.
  Our saved139199164v1 is still source-only, with no committed offline version
  or competition entry. Do not report0.958 as our measured score.
- Its **R6 input must remain Datasetv1**: v2 replaced it with R7 weights under
  identical filenames. Original scored code/input/runtime bindings matter.
- S6v5 source supervision differs from ours across all3887fit studies;
  Fracture target rankcorrelation0.51446. A label-only uniformIID16 contrast has
  actual full-array/private-Dataset/storage admission and32CPU engineering
  checks, but no GPU recovery/fit/score. Neither label source is clinical truth.
- Fresh10:54UTC allowance5used/0available. Existing GPU/entry/publication owners
  retain ownership. Old unstarted equal-rank27596 and R7owner31309 are retired
  without a push/entry; their source/assets remain.
- Released encoders saw our internal cohort/Gold58. Head-only CV on their
  features is not independent whole-model OOF. We have not read protected
  clinical values to tune these experiments.

## Exact recipes and assets

| Asset | Actual contract / evidence |
|---|---|
| OwnN3 reference | Gitdcfff319cb7bd7bbae679a20da2b1b9268e8d680;2247fit/1156weakdev;384pixels/up to6series/16selected slices per series;12training windows;16epochs4496updates; [actual E16score](../initiative/audits/compact-production-e016-kaggle-20261016-v1/actual-scored-canonical.json) |
| Released CoAtNet | 96slices in5anatomical slots/94triples/140mm crop;384prep/320execution/anatomicalmirrorTTA; author reports4349RSNA+2399OAI knees/3tasks, blended labels and averaged snapshots; [actual source contract](research-watch/additional/rsna-public-training-contract-update-2026-10-16.md) |
| Our whole94adaptation |3887fit/all94/all73136096parameters/effective8/3epochs1458updates;25%source75%detached-parent loss;LR3e-6/3e-5; fixedBNbuffers; [actual final](../initiative/audits/whole94-fixed3-final-20261017-v1/README.md) |
| Sampled matched control/contrast |Uniform16 vs influence16 IID replacement/common RNG;25%uniform proposal floor/inverse-proposal attention; originalfull94inference; [actual recovery/production](../initiative/audits/licensed-importance-full-comparison-20261017-v1/README.md) |
| NewS6source-label arm |Same uniform16recipe/3887/init/draws/views/masks/teacher/1458updates, source labels only differ; [protocol and32checks](../initiative/experiments/competitive-v1/s6-source-label-contrast-v1/README.md), [actual data/storage](../initiative/audits/s6-source-label-admission-20261017-v1/README.md) |
| Released S6 |Dataset12589750v5source;4263fit/16epochs/64uniformwindows/physical3.5mm/denseBCE/noEMA/SWA/fixedE16; noOAI;86missing studies are author's local disk deletions; [source audit](../initiative/audits/s6-training-release-20261017-v1/README.md) |
| Exact0.958 reference |CoAtNet384+224/R6fourfoldphysical/state/fluid stack/threefold2.5Dreader;8cellsSHA300b34417bc1...; originalR6bundlefe055406fad6a12f; [actual qualification](../initiative/audits/public0958-reference-20261017-v1/interactive-v1/README.md) |

The released CoAtNet parentSHA is
`7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef`,
Dataset12431568v1. Competition adaptation permission/attribution is retained.
Private pixels/checkpoints/UIDs/labels/teacher arrays must not be added to Git.
Current cache already covers3887fitting studies; do not download it again.
Stanford source-task results are not RSNA-transfer evidence. KneeCoT/Stanford
source-derived weights are not part of current eligible scored models.

## Original evidence map: avoid rediscovery

| Question | Actual evidence |
|---|---|
| Neighboring slice channels | [Fixed-final comparison](../initiative/audits/compact-source-neighbor-fixed-final-20261016-v1/README.md): delta−0.00572762,conditional95%[−0.02522802,+0.01337630] |
| Positive BCE weighting | [Original comparison](../initiative/audits/compact-positive-weight-fixed-final-20261016-v1/README.md): delta+0.00298799,interval[−0.00999470,+0.01656811] |
| Late averaging | [Raw-serving comparison](../initiative/audits/compact-average-raw-outer-comparison-20261016-v1/README.md): delta+0.00031564,interval[−0.00214034,+0.00293081] |
| Expanded fitting population | [Fixed-final evidence](../initiative/audits/compact-expanded2592-fixed-final-20261017-v1/README.md): delta−0.00349377,interval[−0.01331711,+0.00607252] |
| Spatial/context additions | [Regional deployment](../initiative/audits/regional-context-deployment-20261016-v1/README.md), [context terminal](../initiative/audits/released-context-deployment-20261016-v1/actual-scored-recovered.json): actual0.948 |
| Full inference field | [Actual0.948](../initiative/audits/licensed-native-field-20261017-v1/actual-scored-result.json); does not test retraining at a different field |
| Selection_Tier | [All1156study counterfactual](../initiative/audits/compact-serving-tier-counterfactual-20261016-v1/README.md): zero changed selected-series choices |
| Source confidence | [Matched controls](../initiative/audits/source-confidence-context-20261016-v1/README.md): both full10epoch fits complete; no established scored gain |
| Pair ranking | [Completed full fit/qualification](../initiative/audits/source-ranking-context-20261016-v1/README.md); no new competition score claimed |
| Influence sampling | [Protocol](../initiative/experiments/competitive-v1/licensed-importance-encoder-v1/README.md): frozen teacher MSE−45.6%, not disease accuracy or measured speedup |
| Crossing-series boundaries | [Fitting-only screen](../initiative/audits/frozen-series-boundary-screen-20261017-v1/README.md): low attention; fewer pair changes than matched interiors |
| Cross-lineage confidence/rank | [Actual fitting8 result](../initiative/audits/s6-fitting8-cross-lineage-20261017-v1/README.md): raw confidence gaps exceed rank gaps; not clinical error |
| Source-label coverage semantics | [Corrected historical matched DINO](research-handoff-20261012-corrected.md): original0.87867/full-public0.89978; values-only and coverage-only did not explain total gain |

Rare-target support is sparse: N3component bootstrap2836defined/1164undefined.
Conditional intervals are not unconditional certainty. Expensive LLM labels are
not automatically correct; silence/uncertainty remain masked. Preserve original
reports/judgments/accepted corrections. Do not optimize against protected labels.

## Next useful work and falsification

1. Obtain one exact committed offline0.958version under fresh capacity/funding,
   then its actual hidden result under live allowance. No diagnostic model or
   public3rowCSV submission. Do not change the reference before reproduction.
2. Finish observing owned finalE3/scout/CPU32/sampled outcomes. Negative results
   should change priorities, not trigger a duplicate run.
3. Admit actual real-MRI source-label GPU recovery and fit the fixed treatment
   only after full state/prediction recovery and all-future funding pass. It
   tests labels, not a claim that either supervision source is ground truth.
4. Use measured residual failures for any new mechanism: missing localized
   evidence, cross-view geometry, state/extent supervision, teacher-error
   anchoring or study ordering. Tie each idea to the nearest completed control,
   exact changes, exposure/splits, measured cost and a falsification criterion.
   More parameters/windows/epochs without that argument are not a frontier claim.

A useful candidate must earn its result on appropriate matched evidence. A
fixed-final gain should survive a matched seed/fold or suitable external check;
released-model exposure limits internal causal claims. Do not promise first place.

Full previous narrative/source/evidence remains in
[the preceding Git snapshot](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/blob/ca90fa5e347310dbbb35765e5dcc5c1b07e04910/docs/research-agent-handoff.md),
[research archive](research-watch/README.md), [research index](research-watch/INDEX.md)
and [original initiative receipts](../initiative/audits/). Those dated job states
are historical; the current execution handoff owns live next actions.
