# Research-agent handoff

**Prepared October 17, 2026; live reconciliation through 04:56 UTC.** Start here when
proposing new work. [Current state](current-state.md) is the score/job authority;
[research brief](research-brief.md) explains mechanisms and uncertainty;
[execution handoff](codex-continuation.md) records active owners and exact identities.

## The facts a new proposal must account for

- **Best available public score: 0.949**, from nartaa's released MRI-trained
  CoAtNet with unchanged inference. We reproduced its inference, not its training.
- **Best independently MRI-trained model: our N3 CoAtNet epoch 16, 0.894.**
  Earlier own epochs scored 0.821 / 0.869 / 0.890. N3 trains every backbone/head
  parameter; it is not the historical frozen DINO model or a tiny adapter.
- Our sequence-context addition and full-384 inference field each scored **0.948**.
  Neither establishes an improvement over the released 0.949 parent.
- Full-parent adaptation is running on Kaggle; expanded own-model fitting is
  running on Newton. Neither has a final/new competition score. The matched
  importance-sampling production comparison has completed/private-verified uniformE1;
  influenceE1 139154664v1 is now actually running.
- The earlier **two-slot** observation preceded the actual learnedE1 entry
  **57294905** (COMPLETE/**0.949**, noerror); refresh authoritative allowance. Regional
  submission 57291182 actually scored **0.948**; entry owners already cover
  learned-parent epoch 1 and fixed-final epoch 3. Inspect
  those originals before proposing a conflicting submission.
- The external parent's development used this internal cohort/Gold58. Its
  derivatives have **no independent internal OOF** on that cohort. Our protected
  clinical values remain unused. A weak-label metric is not clinical truth.

The fresh [live leaderboard check](../initiative/audits/github-current-state-refresh-20261017-v2/actual-public-leaderboard.json)
shows first place **0.969** versus our **0.949**. Shared 0.950 solos / 0.956 blends are useful
reproduction targets, not proof we have reached the frontier. Current leaders'
methods and private scores remain unknown.

## Exact recipes and assets

| Asset | Actual contract | Source / evidence |
|---|---|---|
| Our N3 reference | CoAtNet RMLP2, 73,136,096 trainable parameters; finding-specific MIL; 2,247 fit / 1,156 weak dev; 384 pixels; up to six series / 16 selected slices per series; 12 training windows; 16 epochs / 4,496 updates | Scientific Git `dcfff319cb7bd7bbae679a20da2b1b9268e8d680`; [actual E16 result](../initiative/audits/compact-production-e016-kaggle-20261016-v1/actual-scored-canonical.json) |
| Released parent | Same 73.1M architecture; 96 slices in five slots, 94 triples; 140 mm crop; 384 preparation / 320 execution; mirror TTA; per-target ranks | [Public source/training-contract review](research-watch/additional/rsna-public-training-contract-update-2026-10-16.md), [exact admission](../initiative/audits/released0949-baseline-20261016-v1/source-manifest.json) |
| Current full-parent fit | 3,887 studies; all 94 windows; all encoder/head parameters; three epochs / 1,458 updates; effective eight; 25% source / 75% detached-parent anchoring | [Original actual production](../initiative/audits/licensed-parent-whole94-production-20261016-v1/README.md); scientific Git `c68d8752bd6b692babe971f08ca4b4590aa35db5` |
| Expanded own N3 | 2,592 fit / unchanged 1,156 outer; 16 epochs / 5,184 updates; original recipe | [Original production](../initiative/audits/compact-expanded2592-production-20261016-v1/README.md); scientific Git `a1ca238526dad75b2a5cea46edba44fd95c71938` |
| Matched sampled full-parent roles | Uniform-16 versus teacher-influence-16 IID draws; common random variates and exact starting model/RNG; inverse-proposal readout; full-94 inference | [Protocol](../initiative/experiments/competitive-v1/licensed-importance-encoder-v1/README.md), [actual full-input/GPU/storage admission](../initiative/audits/licensed-importance-full-comparison-20261017-v1/README.md) |
| Current pixel cache | All 3,887 fitting studies; 32 reused plus 3,855 newly decoded; zero decode/empty cases; four original lossless archives and replica verified | Whole-parent production audit; private node-local assets, not a new download request |

The released checkpoint SHA is
`7e5315dad125b99fc65b340b3de41de628e9be51ff5835355dd61c86472244ef`, Dataset
12431568 v1. Its author reports 4,349 RSNA studies, blended supervision, 2,399 OAI
knees for three tasks and selected snapshot averaging. We did not reproduce that
complete recipe or acquire controlled OAI data. Permission for competition
use/adaptation/noncommercial redistribution is retained with attribution. A
source-trained parent and an ImageNet-initialized challenger are different claims.

Public scores also mix fitting cohort, initialization, source labels, preprocessing,
training observations, optimization and serving. The gap is real, but none of
those differences has been isolated as its sole cause.

## Evidence map: avoid rediscovering completed negatives

| Question / mechanism | What we actually learned | Original evidence |
|---|---|---|
| Do physical neighboring slices help N3? | Fixed-final weak macro 0.91519773 versus 0.92092535; delta -0.00572762, conditional 95% [-0.02522802, +0.01337630]. No primary promotion | [Neighbor comparison](../initiative/audits/compact-source-neighbor-fixed-final-20261016-v1/README.md) |
| Does positive BCE weighting help? | 0.92391334; delta +0.00298799, interval [-0.00999470, +0.01656811]. Not an established primary gain | [Weighting comparison](../initiative/audits/compact-positive-weight-fixed-final-20261016-v1/README.md) |
| Does late parameter averaging help? | Raw-serving 0.92100408 versus raw E16 0.92068844; delta +0.00031564, interval [-0.00214034, +0.00293081] | [Raw-serving averaging](../initiative/audits/compact-average-raw-outer-comparison-20261016-v1/README.md) |
| Is spatial pooling already tested? | Matched regional/global-repeat heads completed all ten epochs and all 20 checkpoints are privately verified. Regional entry 57291182 actually scored **0.948**, with no established gain | [Spatial deployment](../initiative/audits/regional-context-deployment-20261016-v1/README.md) |
| Is supervision-confidence weighting already tested? | Both full ten-epoch fits completed; the uniform control reproduces original complete state/predictions. No new scored gain is established | [Confidence contrast](../initiative/audits/source-confidence-context-20261016-v1/README.md) |
| Is a pair-ranking head already built? | Full 3,887-study / 10-epoch / 2,430-update fitting and private preservation completed, with actual two-T4 qualification. No competition score is claimed | [Ranking contrast](../initiative/audits/source-ranking-context-20261016-v1/README.md) |
| Is adding context enough? | The head changes inter-study ordering, but actual public score is 0.948 versus 0.949. Architectural mechanism is not a gain | [Context result](../initiative/audits/released-context-deployment-20261016-v1/actual-scored-recovered.json) |
| Does retaining the whole inference field help? | Same released weights, full 384 field: actual public 0.948. This does not test retraining at a different field | [Field result](../initiative/audits/licensed-native-field-20261017-v1/actual-scored-result.json) |
| Did `Selection_Tier` alone select a different outer cohort input? | Dropping the field changed zero selected-series choices in all 1,156 outer studies. Complete rendering equivalence remains separate | [Exact counterfactual](../initiative/audits/compact-serving-tier-counterfactual-20261016-v1/README.md) |
| Does influence sampling see the teacher more efficiently? | Full-cohort teacher-approximation MSE fell 45.6%, all targets improving. No disease labels/new CNN were used; this is not an accuracy or throughput result | [Influence protocol/admission](../initiative/audits/licensed-importance-full-comparison-20261017-v1/README.md) |
| Are source labels/coverage interchangeable? | Historical matched DINO: original 0.87867060; full-public 0.89978406; original masks/public values 0.85368128; coverage-only 0.86930231. Neither follow-up alone explains the full-public gain | [Corrected October 12 analysis](research-handoff-20261012-corrected.md), with exact original fixed-final audit paths |
| Was uniform/stratified sampling already tried? | Earlier DINO contrast was near-null: +0.0009107, conditional interval [-0.0090501, +0.0099533]. It is not the current full-encoder influence experiment | [Historical full snapshot](codex-continuation.history-20261017-0146.md) and original sampling audits |
| Are Stanford source-task scores RSNA progress? | Stanford ACL/meniscus/abnormality tasks were separately fitted; source-task validation is not demonstrated RSNA transfer. Stanford/KneeCoT are not part of these current eligible scored weights | [Current provenance limits](current-state.md), private-source audits under `initiative/audits/` |

The N3 component-bootstrap comparisons have 2,836 defined / 1,164 undefined draws.
Rare-target support is extremely small. Source-label coverage and rubric/severity
differences matter; expensive LLM annotation does not make every label ground
truth. Original reports, accepted reviews/corrections and masks remain preserved;
new image reviews are advisory until accepted with provenance. Do not silently
replace training labels or use protected evaluation to choose their semantics.

## October 17 owner research program

Read the [complete new program and assessment](../initiative/research/owner-deadline-program-20261017-v1/README.md).
The first concrete action was source-only admission of S6 ConvNeXt:
138285808 v2 and Dataset 12589750 v3 manifests are pinned. Its author's reported
0.950 is not our reproduction. The original 200,185,003-byte checkpoint is now
privately downloaded once and SHA/header verified. Dependencies/source/anatomy,
CPU MRI tensors and CPU classifier reference now pass. Actual T4/runtime/output
acceptance and a distinct S6 entry remain undone. [Actual checkpoint proof](../initiative/audits/s6-checkpoint-admission-20261017-v1/README.md). Original numerical training continues unchanged; only terminal serving/entry3
observer dependencies were replaced. [Current originals and new sole owners](codex-continuation.md).

The program's recommendation to obtain strict five-fold OOF by fitting heads on
released encoder features is not valid when those encoders saw the evaluated
studies. State source exposure explicitly. The assessment also separates genuine
geometry/protocol mechanisms from completed context/regional controls, and treats
dual-teacher agreement as a hypothesis rather than truth. Do not promise the
paper's speculative gains or estimated GPU-hour budgets.

## Highest-value research questions still open

These are questions for new mechanisms, not a declaration that more experiments
have already run or that the next mechanism will win.

1. **Can adaptation correct parent mistakes without simply copying them?** The
   current loss anchors 75% to the parent. Propose a training-only uncertainty,
   disagreement or evidence-consistency mechanism that changes the gradient when
   source/teacher disagree, with an unchanged-input/compute control. Explain how
   it avoids reinforcing confident teacher errors and protected-label leakage.
2. **Which observations improve learning per second?** The current influence
   comparison isolates the sampling proposal at fixed draws and updates. Test
   its failure modes: stale influence, small lesions never sampled, cross-view
   inconsistency and finite-bag ratio bias. Compare measured end-to-end time and
   actual scores; 94/16 is only a window-count ratio.
3. **What finding-specific information is lost by pooling?** Localized tears and
   broader extent/severity may need different evidence aggregation. Tie a new
   extent/localization mechanism to a measured failure and the completed
   context/regional/global-repeat controls. Do not rename an unregistered
   physical coordinate system as cross-series anatomical registration.
4. **Can the objective improve study ordering directly?** AUC rewards ordering.
   An inference-only strictly monotone per-target transform cannot improve AUC
   without changing ties. A useful ranking/distillation objective must change
   meaningful pair ordering without leaking dev labels. A completed pair-ranking
   head is already in the archive; inspect its qualification and current entry
   status before proposing it as new.
5. **How much of the gap is source exposure rather than architecture?** Design
   an informative comparison of initialization, source supervision and fitting
   scope. The external author used more/different data; external 0.949 versus
   own 0.894 is not a controlled backbone comparison. Expanded N3 changes compute
   as well as data, so do not call it data-only evidence.

For each recommendation, return: the measured failure; mechanism; closest existing
control; exact change; training/evaluation cohort and exposure; GPU/runtime cost;
expected falsification signal; and which concrete result would change the decision.
A bounded useful experiment beats another long list of untested architectures.

## Where the full findings live

- [Current synthesis](research-brief.md) and [actual state](current-state.md).
- [Complete research collection](research-watch/README.md), [file index](research-watch/INDEX.md)
  and [original manifest](research-watch/MANIFEST.json). Papers/proposals retain
  their dates. Their historical “next” instructions are not live authority.
- [October 16 source-contract correction](research-watch/additional/rsna-public-training-contract-update-2026-10-16.md),
  which supersedes earlier guesses about the strong public notebook's training.
- [Reconciled historical experiment ledger](research-watch/dossier/rsna-research-ledger-20261014/experiment-ledger-reconciled.md)
  and [family coverage](research-watch/dossier/rsna-research-ledger-20261014/experiment-family-coverage.md).
  Audit-directory counts are not counts of successful experiments or accuracy gains.
- [Original audit directory](../initiative/audits/) and [experiment protocols](../initiative/experiments/competitive-v1/).
  Prefer original terminal/prediction/checkpoint receipts over old status prose.
- [Source/report archive and indexes](../initiative/README.md). Historical report
  releases differ from today's fitting cohorts and public-soft supervision.

The new primary public S6 source and live leaderboard were actually checked.
Broader software QA is recorded in [the current refresh audit](../initiative/audits/github-current-state-refresh-20261017-v2/verification.json);
do not infer full-suite success before its actual terminal receipt. The sampled serving preparation has synthetic checks only and still lacks
actual sampled-model GPU/public-MRI acceptance. Separately, learned whole94E1
actually passed32MRI/twoT4 qualification with zero probability difference; its
public run actually passed and its exact code entry57294905 is accepted once;
hidden terminal is **COMPLETE/0.949**, matching the parent without a gain. Current recovery/capacity/screen/
serving checks total80focused cases, Ruff6/mypy3; no clinical gain is inferred. Tests and state recovery establish operability, not accuracy.

## Open PRs and collaboration

[Fresh open-PR metadata](../initiative/audits/github-research-handoff-20261017-v1/open-pull-requests.json)
lists #77, #71, #67, #66, #63, #61, #58, #27, #6, #4 and #3. They are open proposals
or historical branches, not evidence of a deployed experiment; that metadata
check is not a new code review. Some topics overlap later main implementations.
Check source identities and actual main behavior before reviving one. **Devin is
unavailable; implement directly and do not make execution depend on Devin.**

Live training and finite owners must not be duplicated. Do not overwrite their
scientific recipe or restart a GPU job because an observer disconnects. Reuse
verified caches/epochs, retain exact source/weights/rights/runtime proofs and
negative results, and fund actual storage/compute before a new dispatch. Keep
private pixels, clinical records, individual predictions and restricted-source
weights out of Git. Publish sanitized aggregate evidence and code; reference the
existing private artifact proofs when large assets are required.

## New measured opportunity screen

The fitting-only teacher-view screen found mean view gaps1.29–2.71probability
points and strict-pair ordering disagreement2.75–7.41%. The predeclared triage is
**intermediate**, not a new training promotion. Clinical labels were not read,
so those ordering pairs are not positive-negative AUC errors. Current target
definitions stay unchanged. [All-target result and protocol](../initiative/audits/teacher-view-consistency-screen-20261017-v1/README.md).

Whole94 epoch1 is complete and privately verified; epoch2 resumes the exact
original worker/model/optimizer/RNG after a metadata-order-only compiler repair.
The first sampled uniform source was separately reproducible byte-for-byte.
Kaggle's actual two-session cap now has admission/expiring inference priority;
old capacity-rejected private identity is quarantined, never retried. These are
execution facts, not accuracy improvements. Read the current execution handoff
before taking ownership of any live model or planning a conflicting GPU push.

The [earlier private mirror](../initiative/audits/hf-whole94-e1-capacity-screen-20261017-v1/verified.json)
fully verifies source60ed92d3 at HFda34234c/all102smallobjects. This includes the
original new paper, critical assessment and exact S6 checkpoint/source receipts.
Current guide receipt bookkeeping can be later than that mirrored source.

## Later actual engineering and training evidence

Whole94 E2 last visible722/1458 remains an older detailed log observation.
Uniform16 E1 actually completed486updates and all5objects are privately fully
verified; influenceE1 is RUNNING. Expanded N3 original938284 is at2675/5184 finite updates,epoch9, with eight
completed privately fully verified epochs; this supersedes2166/epoch7. Original patch929938 actually
completed0:0 with exact complete-state/prediction full/stop1/resume recovery;
it is not a clinical result or a new production dispatch.
[Actual later receipts](../initiative/audits/research-progress-20261017-0401-v1/README.md).

S6 source/anatomy now passes all293 original hashes; unchanged CPU B_PROBE
passes with strict full classifier loading and a finite anatomy forward. Distinct
private CPU-only Kaggle input gate139151928v1 completed on all three public MRIs
with1586windows and no decode/anchor failures; original inputs are now privately
fully verified on Newton. CPU942136 then passed the full classifier reference on all3/1586windows with
zero fallback. H100 classifier probe942047 remains queued, not scored; finite
T4 reproduction owner16380 is waiting actual capacity/all-future budget. It has
zero competition entries and has not pushed an S6 GPU notebook at this check.
[Exact source/runtime/gate proof](../initiative/audits/s6-source-tree-admission-20261017-v1/README.md).
Original full regression terminated successfully:4537passed/8skipped/exit0;
no numerical source changed. Do not mistake that for clinical improvement.

Earlier distinct sourceb9dabf26 is now privately fully verified at HF0917971d,
all40objects/268191bytes; current receipt bookkeeping follows that source.
[Exact private proof](../initiative/audits/hf-s6-original-mri-20261017-v1/verified.json).
UniformE1 is now terminal/private verified at HFc19961db; influenceE1 is active.
[Original results and sole final-serving/entry owners](../initiative/audits/research-progress-20261017-0448-v1/README.md).


## New evidence against a boundary-only explanation

[Actual all3887 label-free screen](../initiative/audits/frozen-series-boundary-screen-20261017-v1/README.md)
compared existing within-series removal to a same-count interior control.
Boundary windows already receive only0.35–1.54%meanattention and their removal
reverses0.074–0.494%strict study pairs, versus0.321–1.511%for interior windows.
This deprioritizes another boundary-only training arm, not proof of accuracy.
The existing within-series and edge-repeat notebooks are public-MRI accepted
but unscored; do not recreate them. An improved source-DICOM representation or
S6/CoAtNet complementary-error mechanism must address information this readout
screen did not test, with actual runtime and source-exposure limits stated.


Latest distinct private preservation: source**5b55c961** is fully verified at
HF**feb3100aa7b418db9bfed48cc869c710288d1e38**, all46smallobjects/207863bytes,
CPU942389/canonical shared lock/full re-download-size-SHA. This includes the
actual learnedE1 **0.949**, new boundary/S6 CPU results, original epoch proofs and
sole final-serving owners. No private pixels/clinical arrays/features/weights.
[Exact one-time mirror](../initiative/audits/hf-parent0949-boundary-s6cpu-20261017-v1/README.md).
Never repeat this or earlier prefixes; this receipt bookkeeping is later than
that mirrored source.
