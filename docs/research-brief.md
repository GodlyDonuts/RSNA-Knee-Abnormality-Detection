# Research brief — October 16, 2026

Read [current state](current-state.md) first. It contains timestamped scores,
current run phases, the actual training recipe, population counts and deployment
status. This brief records conclusions; it is not another execution log.

## What the evidence supports

Fresh authenticated [public leaderboard](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/leaderboard)
reference on October16 is **0.968** at first place. Our licensed0.949 baseline
is still0.019 below that public reference; own0.894 is0.074 below. Neither is
a winning result or a private-leaderboard guarantee. Our context head actually scored0.948, below unchanged0.949; no gain established.
The full384field entry57289392 is pending/unknown, not a score. Full-model
fine-tuning is healthy at293/1458updates; E1/E3 comparisons are installed.

The attributed author-trained model scored **0.949** on Kaggle. Our own CoAtNet
epoch 16 scored **0.894**, following epochs 4/8/12 at **0.821/0.869/0.890** and
prior own DINO best **0.872**. The final increment is +0.004, compared with
+0.048 and +0.021 earlier; continued improvement is not guaranteed. These competition
results remain distinct from the completed fixed-16 internal macro AUC of
**0.920925**, independently replayed on 1,156 weak-label holdout studies.

That internal endpoint has only two MCL/fracture positives and three Baker
positives. It is not a clinical score or evidence of parity with the external
competition result. Exact per-target support and scored receipts are linked in
current state.

The new N3 model fully trains a 73M-parameter CoAtNet. It is not the historical
frozen-encoder/small-head recipe. It already uses dense public soft fitting labels,
384-pixel inputs, finding-specific MIL, 12 training windows and a fixed 16 epochs.
The remaining differences from the released reference include population,
observation/rendering, loss, optimization, late-state averaging and inference.
Their combined score gap does not identify any one causal ingredient.

Current N3 triples are consecutive **selected slots**, not guaranteed consecutive
source slices. The source-neighbor challenger preserves original centers and all
other registered settings. Its full fit and a separate positive-weight full fit
were submitted after actual H100 recovery. Neighbor932957 is now COMPLETED;
all16epochs are privately verified. Actual paired fixed-final macro0.91519773versus
original0.92092535, delta-0.00572762, conditional95%[-0.02522802,+0.01337630],
2,836defined/1,164undefined whole-component draws. No established primary gain;
keep original N3. Lateral-meniscus/contusion secondary improvements need matched
replication and do not justify post-hoc target splicing. Narrower adjacency versus
wider anatomical context is a plausible tradeoff, not a proved explanation.
Weighted933001COMPLETED, all16privateverified. Actual independent fixed-final
macro0.92391334vs0.92092535, delta+.00298799, conditional95%[-.00999470,+.01656811].
No established primary gain; retainN3E16. MCLpoint improvement has only2positives,
with exploratory LM/OA regressions. No posthoc target splicing or weighted Kaggle score.
Public one-study gradient accumulation also means mixed-study BatchNorm is not a
known missing ingredient. [Full paired evidence](../initiative/audits/compact-source-neighbor-fixed-final-20261016-v1/README.md).

Additional 345 fitting-only caches and private preservation are complete.
Expanded 2,592-study inputs passed actual zero-GPU admission933324 with original
image/label/mask tensor parity; full production938284was dispatched once after actual H100recovery933382 and is
PENDINGPriority; no expanded full-fit result yet. The old 3,904/441 release counts describe a historical release;
current correspondence-filtered capacities are 2,592 fitting / 3,887 train /
4,327 potential frozen-final. Do not equate these with the active 2,247-study fit.

## Findings that should guide the remaining work

- Held-out weak-label AUC measures agreement with weak supervision. It is not a
  substitute for clinical truth or a numerical prediction of Kaggle performance.
- Higher rank, resolution and more parameters have not by themselves established
  a robust gain. Completed negative/null comparisons remain in the linked history.
- Physical source adjacency and denser coverage are different interventions.
  A neutral adjacency result would not rule out a coverage benefit.
- Cached rewindowing cannot restore spatial detail removed before caching.
  Raw physical cropping, crop-after-resize and intensity normalization are separate.
- Our read-only 31-second training sample showed bursty GPU utilization (22.4%
  mean); it did not isolate a cause. Source reads/decompresses whole NPZ volumes.
  Checkpointing occurs at epoch/stop boundaries, **not every update**.
- A public model trained on all RSNA studies cannot provide independent OOF
  predictions for our held-out cohort. Do not fit blend weights on such leakage.
- Source-trained Stanford results are separate three-task research, not an RSNA
  transfer result. Neither diagnostic recovery nor cache completion establishes
  accuracy. No model is described as clinically perfect.

## Active experiments and decision rules

The original fixed N3 fit and predeclared E4/E8/E12/E16 submission curve are complete;
all four exact hidden outcomes are scored; never repeat those submissions.
Compare source-neighbor and positive-weight fits separately against the same
2,247/1,156 baseline, fixed seed and 4,496-update schedule. Require per-target
results, paired whole-component uncertainty, runtime and failure accounting.
A promising result needs replication where time permits; no best-epoch picking
on protected cohorts.

Admit expanded fitting inputs without changing the outer roster. Freeze the
selected recipe before a larger final refit. Record that more examples at the same
number of passes also cost more updates. Do not label such a result data-only.

Investigate dense inference, explicit anatomical metadata, exact anatomical TTA
and late-epoch parameter averaging as separate candidates. BN buffer policy and
training-serving geometry must be explicit. OrthoFoundation/domain adaptation
remain research directions until actual checkpoint/input compatibility and
appropriate use are established; upstream task scores are not competition gains.

## Evidence and historical findings

- [Actual current runs and score receipts](current-state.md)
- [Exact public recipe correction](research-watch/additional/rsna-public-training-contract-update-2026-10-16.md)
- [Paper 2 assessment and feedback](../initiative/audits/research-paper2-assessment-20261016-v1/README.md)
- [Neighbor full-fit admission](../initiative/audits/compact-source-neighbor-full-fit-20261016-v1/README.md)
- [Positive-weight full-fit admission](../initiative/audits/compact-positive-weight-full-fit-20261016-v1/README.md)
- [All additional-cache preservation](../initiative/audits/additional345-cache-terminal-20261016-v1/README.md)
- [Historical research brief, including earlier results and negatives](research-brief.history-20261016.md)

Historical findings are preserved, not erased. Dated statements such as “pending,”
“no score” or “not implemented” in that history do not describe the current state.

The registered equal learned-parameter average ofE12/E14/E16 is now constructed
and privately fully verified, with all savedE16buffers retained. Actual CPU MRI
serving acceptance and the actual paired two-T4 evaluation passed; its point
gain is small and not established by the uncertainty analysis. The first worker's
nonpersistent-buffer check failed before output; corrected verification leaves
selection and numerical averaging unchanged. No derived gain is claimed.

The fixed average and E16 were evaluated together on the same raw-serving
outer MRI windows in Kaggle, with clinical labels retained on the cluster. This
separates an averaging gain from a changed serving reference; neither is assumed.
A metadata-only replay already rules out Selection_Tier alone changing the held-out
series selection (zero changes/1,156 studies), but other metadata/rendering and
BF16-versus-FP16 differences remain. The preregistered comparison and negatives
are in [the original audit](../initiative/audits/compact-average-raw-outer-comparison-20261016-v1/README.md).

A fitting-only CPU profile compares materializing all windows versus only a fixed
12-window subset. Code inspection confirms that **production N3 already selects
before materialization**, so this is not a new N3 optimization. Selected tensors and global RNG stayed exact;
window materialization summed 2.1092→0.9034 seconds over 16 studies, with another
5.3107 seconds spent loading. It is a counterfactual substep saving, not a new training speed/accuracy
gain; no live trainer was changed. The neighbor dataset still eagerly constructs
all donor windows, a separate future optimization. [Original measurement](../initiative/audits/compact-selected-window-cost-20261016-v1/README.md).

A new licensed inference experiment targets eight cross-series triples in the
attributed 0.949 model. Primary edge-repeat retains all centers; secondary
within-series removes the mixed windows. Both exact private v1 candidates
completed public MRI/CSV acceptance without changing learned weights; neither
has a competition submission/hidden score yet. This is an
unproven mechanism and external adaptation, not an own-training improvement or
independent internal validation. [Protocol and interpretation](../initiative/experiments/competitive-v1/released-series-boundary-v1/README.md).

The context mechanism now has a complete accepted3,887study/two-view feature cache. Its exact fixed10epoch/2430update FP32engine completed in430.943seconds and all10checkpoints/finalcomponent are privately download/hash verified. The model relates all94windows through a bounded zero-start attention correction while retaining the strong licensed parent. The cache cost3,519seconds on twoT4s after832savedrecords were reused; later head experiments need no repeatedCNN encoding. RAM exhaustion in the original unbounded producer and the omitted failed-notebook dependency are preserved negatives; bounding both task submission and consumption fixed the actual extraction path. Actual final-head precision on bothT4s and the publicMRI/CSV path also passed, but no new context clinical/Kaggle result or independent internal OOF exists.

A complementary idea addresses small-bag training exposure using importance sampling inspired by Monte Carlo estimators. Against the strong teacher's fullMILreadout, fixed influence sampling with inverse-proposal correction reduced approximation MSE42.0%on32engineering cases and45.6%onall3,887fitting cases, with everytarget improved. This consumed no disease labels/newCNN/GPU in the fullrepeat. [Actual fullcohort result](../initiative/audits/licensed-MIL-sampling-20261016-v1/README.md).

This is evidence of better teacher fidelity per16-windowbag, not better disease classification. The teacher may be wrong, finite self-normalized estimates can be biased, and computing its influence requires the complete teacher feature bag. It suggests a cache-guided training contrast; it does not establish a cheaper inference scheme or justify untested CNN promotion. A matched16-window uniform/influence full-encoder engine and exact recovery compiler are now implemented and locally checked. Actual GPU recovery and full-production funding remain prerequisites; no such full CNNfit is dispatched. Preserve this distinction in further plans and reports. [Fixed protocol](../initiative/experiments/competitive-v1/licensed-importance-encoder-v1/README.md).

The registered averaging result remains null: raw-servingE16 0.92068844 versus fixedaverage0.92100408, delta+.00031564 with whole-component conditional95%[-.00214034,.00293081]. Keep E16 as own-model default; no independent evidence supports an averaging improvement.

A post-fit label-free check found the context adaptation changes0.935%–3.829%of strict study-pair orders (3,887fitting cases), rather than only shifting class calibration. This establishes behavior, not correctness or a clinical gain. The expanded2,592study route now has actual H100recovery933382accepted and one original full-fit938284dispatched/PENDINGPriority. Source/data/scientific recipe are unchanged; this is not completed training or a data-only causal result.

## Spatial information hypothesis

The original frozen architecture reduces each1,024×10×10map to a global vector.
Distinct synthetic focal/uniform maps can have the same global mean. This motivates
retaining seven fixed spatial summaries, all1,024channels each, alongside the
original global features. It does not establish disease localization, novelty or
a causal explanation of the score gap.

Actual32MRI/both-T4engineering acceptance139066640/v1 passed, with original
observer features/logits bit-equal. Cross-run global-feature differences caused
the first probe's refusal, despite small parent probability drift; root numerical
cause remains unproven. The corrected representation caches global, regional and
parent outputs from one forward, never mixes old/new embeddings. The full3,887-study feature pass139068442/v1 COMPLETED in4,382seconds, reusing32
accepted originals. CPU937898 recovered the interrupted archive using only missing suffixes and
accepted every member/whole SHA/complete cohort. Both fixed10fits COMPLETED31:41/31:55, same graph/init and2,430updates. All20
immutable epochs are privately fully verified; no spatial accuracy result yet.
[Original failure](../initiative/audits/regional-feature-probe-20261016-v1/README.md),
[actual32acceptance](../initiative/audits/regional-feature-probe-20261016-v2/README.md),
[full experiment](../initiative/audits/regional-feature-full-20261016-v1/README.md).

Verified private preservation: sourcee2fd3911/all51smallobjects235,995bytes at
HF9d593e484d0fde760381e12a7c606e2f84fb32a4, fully re-downloaded/size/SHAverified
once under the canonical lock byCPU936730. No private arrays/images/identities/
weights in the small mirror. [Exact receipt](../initiative/audits/hf-regional-expanded-20261016-v1/verified.json).
Do not repeat this prefix or earlier verified milestones.

CPU936870 verified the actual final context head on old/new globals from32MRIcases:
max mean-TTA probability drift0.0001293 (parent0.0001238), within the existing
0.003precision bound. No newCNN/GPU/labels/fitting; cause of cross-run feature drift
and full-population bound remain unproven. Context was submitted once and actually scored0.948, below unchanged parent0.949. [Actual targeted check](../initiative/audits/context-cross-cache-numerics-20261016-v1/README.md).
Neighbor fixed-final sourcea44f476a and10smallobjects are private-HF fully verified
ONCE at fd2b776e0bc977cdac4dc7730d2e894d1bfbfb32 (CPU936842).
[Verified preservation](../initiative/audits/hf-neighbor-fixed-final-20261016-v1/verified.json).

The matched spatial-head engine now has actual six-process CPU937321 acceptance,
not only38local tests: each arm full/stop1/resume, complete numerical state and
32x12prediction bytes exact, both with nonzero learning. Same1,334,796parameter/
8192-input graph and initial values isolate feature content:8coherent global
copies versus global+7regional summaries. It reuses unchanged5addea82FP32training.
This is engineering acceptance, not disease accuracy. Full3887fixed10fits937942/937975are COMPLETE/all20epochs privately verified;
no clinical or competition improvement is yet established. The original pre-update CPU rounding verifier
failure and corrected exact same-batch check are preserved.
[Actual recovery](../initiative/audits/regional-context-engine-20261016-v2/README.md).

The separate typed fixed-final spatial exporter/readout passes33focused tests
with old serving rejection preserved. It retains exact role/feature representation,
registered immutable source/cache/recovery/permission and strips private training
payload. Actual final exports and twoT4/publicMRI acceptance have now passed
for both roles; neither has a hidden score. Software/runtime acceptance is not
evidence of accuracy.

Verified latest small preservation: source6c06674d/all49objects236002bytes,
privateHFc6e848a5e21090b2cf3b462680bf8166a064aee0. Full re-download/size/SHA
under the canonical lock byCPU938041; no private arrays/images/identities/weights.
[Exact receipt](../initiative/audits/hf-regional-actual-full-head-20261016-v1/verified.json).
Never repeat this prefix or preceding verified mirrors.


Both actual final spatial components have now passed twoT4precision against32MRI
feature references (max1.192e-7) and exact3study public MRI/CSV execution in26–28s,
with zero fallback/empty studies. Private combined Dataset12639590v1 is fully
verified. Neither spatial head has a hidden score. The distinct regional-only owner87869
is live for02:00UTC after actual prior outcomes/fresh allowance. The original
context scored0.948; old spatial/confidence owners stopped before any entry.
[Actual deployment](../initiative/audits/regional-context-deployment-20261016-v1/README.md).

A distinct next supervision hypothesis remains untested: the provided source
documents0.5as unaddressed and recommends confidence2*abs(p−0.5). Current positive-
class weighting is a different intervention and did not establish a primary gain.
A matched source-confidence/teacher-anchoring contrast could test whether report
uncertainty should reduce source influence while retaining the same original
labels/masks/teacher/graph/compute. Preserve fixed current recipes and expose any
normalization/gradient-scale change; this observation does not justify label repair, protected-cohort tuning or
a claim of improved accuracy. See the original
[source-lineage dossier](research-watch/dossier/rsna-single-model-evidence-2026-10-14.md).

Latest verified small milestone: source16dc318e/all59objects242,393bytes at
privateHF56671edc96e98252a5183f6cfaefcb955916017a, fully re-downloaded/size/SHA
under the canonical lock byCPU938456. No private arrays/images/identities/weights.
[Exact receipt](../initiative/audits/hf-spatial-T4-weighted-final-20261016-v1/verified.json).
Never repeat this or earlier verified prefixes.

## Current source-confidence test: actual endpoint

The completed matched3887-study contrast isolates source influence while keeping
model/data/compute fixed. Actual seven-process recovery and full original-uniform
state/prediction-byte equality passed. Both10-epoch finals/all20epochs are privately
verified. Only28/46644current cells are exactly.5; this rules out that subset alone
as a broad explanation of the gap. Source-recommended fractional confidence changes
mean influence.25→.19512 and Synovitis→.13581. Strict fitting pair-order changes
.633%–2.472%prove a distinct predictor, not a gain. No independent localOOF is
claimed; actual Kaggle evidence is still required. [Original receipts](../initiative/audits/source-confidence-context-20261016-v1/README.md).

The next larger direction starts from the licensed MRI-trained.949 parent and
first proves its exact compressed pixel/input path. This can test task-specific
initialization without accidentally changing rendering. It is a prepared protocol,
not a new cache/GPU fit or inherited score. [Protocol](../initiative/experiments/competitive-v1/licensed-parent-end-to-end-v1/README.md).

## Current exact-parent whole-study adaptation

The stronger direction starts from the attributed MRI-trained parent that actually
scored0.949; our own ImageNet-initialized N3 best remains0.894. No new leaderboard
or clinical gain has yet been established by this adaptation.

Actual32MRI input gate139099964v1/CPU938918 passed: exact original96x384uint8
volumes/masks and both94-window FP32 tensors,326.80MBarchive,149.16seconds,
1.90GBRSS. It preserves the scored model's inputs without additional quality loss;
it does not preserve every raw DICOM slice. Two secondary slow-reader cases also
matched exactly. [Original proof](../initiative/audits/licensed-parent-pixels32-20261016-v1/README.md).

Both real twoT4 numerical recoveries are now COMPLETE:

| Path | Exact notebook | Actual result | Peak allocated/GPU |
|---|---:|---|---:|
| Original12sampled-window training |139104577v1| Full/stop1/resume exact |7.604GB|
| Whole94-window checkpointed training |139105348v1| Full/stop1/resume exact |5.973GB|

Both used16fitting+16disjoint engineering MRI cases,2epochs/4updates,
all73,136,096parameters trainable, complete model/base/AdamW/LR/cursors/allrankRNG/
GradScaler equality and identical16engineering two-view prediction bytes. Parent
BN buffers stayed unchanged. These prove engineering behavior, not accuracy.
The whole94path consumed416.87seconds for all three process flows; its full arm
168.15seconds includes engineering evaluation and cannot be treated as pure
training time. [12window receipt](../initiative/audits/licensed-parent-trainer-recovery-20261016-v1/actual-twoT4-accepted.json),
[whole-study source/proofs](../initiative/audits/licensed-parent-whole94-recovery-20261016-v1/README.md).

Whole-study MIL keeps all94original windows in each bag, checkpoint-recomputing
8-window encoder chunks to bound activation memory. Every encoder/head parameter
still trains. Inference uses the unchanged parent encoder and94two-view path.
The full production recipe is now running:3epochs/3887eligible fitting studies/
1458updates/effective8, low encoder/headLR3e-6/3e-5, AdamW/10%warmup-cosine/clip1,
.25original source+.75view-specific full-parent teacher, frozen parent BN buffers.
Protected Gold/confirmation remain unused. No independent localOOF is possible
for this exposed parent. CPU939476 staged the unchanged production sourcec68d8752;
CPU939655 joined all3887ordered fitting UIDs to the exact accepted target bundle.
Original **139111324v1**,godlydonuts/rsna-parent-whole94-full-e1-50b5a0d9c138,
was pushed ONCE and is actually RUNNING. Authenticated live logs show BOTH Tesla
T4s, all73,136,096parameters trainable and **293finite optimizer updates** in epoch1 (latest visible log)
at23:10UTC. No epoch is complete and no new accuracy/competition score exists.
[Actual live optimizer proof](../initiative/audits/licensed-parent-whole94-production-20261016-v1/actual-live-training-UI-43.json).

Live finiteowner80039/exec6555 continues up to3checkpoint-continuous seven-hour
sessions at486/972/1458steps, preserving one complete LR/optimizer/RNG schedule.
All four original private CPU cache notebooks139103622/139103665/139103730/
139103781 v1 are COMPLETE:972/972/972/971studies, zero decode errors/empty cases,
38,870,763,520new archive bytes plus326,799,360reused original32bytes. All3887
ordered fitting studies and both94-window input-view hashes passed. The trainer
mounts those immutable originals read-only without a second40GBworking tree.
Independent Newton backup/replay is now complete for ALL4shards via
CPU939550/939676/939777/939861, each original archive/member/both-view hash
verified once. Collector80537 completed successfully; do not restart it or repeat
any original producer or transfer. The complete private registry SHA is8cc5c194.
[Actual four sources/funding](../initiative/audits/licensed-parent-pixels-full-20261016-v1/README.md).

CPU939271 compacted all3887original labels/masks and both teacher-logit views
bit-exactly into483656bytes. CPU939291 created ONE private training Dataset
**12641441v1**,godlydonuts/rsna-parent-fit-targets-48b4485f03b6, all5objects fully
download/size/SHAverified. No encoder or optimizer ran in that preparation. The
private UID/label/logit arrays stay on cluster/Kaggle, not Git/smallHF mirrors.
[Full input bridge](../initiative/audits/licensed-parent-production-inputs-20261016-v1/README.md).

Original numerical/input engineering checks:52passed,Ruff/mypy clean, including fresh
2rank CPU recovery, exact whole94gradient/RNG parity and input/target roundtrips.
The new loader also passed actualCPU939362: all32original MRI cases/64views,
all94window inputs, source labels/masks and both teacher views matched bit-exactly
in37.01seconds, reading directly inside the original tar without a second pixel
tree. Full production additionally passed all4actual original Kaggle producer
proofs, exact weight/source/label bindings, accepted whole94GPU receipt and fresh
resource state. Its fixed-schedule sessions retain the original LR/optimizer
schedule across the three segments.

Latest exact-parent source and actual engineering/input receipts are privately
fully verified at HFa879e194d2646971b0b9238ce588abce64c2d03f:71small objects,
419747bytes, CPU939371/canonical shared lock/source9adce769. No MRI/UID/label/
teacher arrays/prediction rows/parent source body/weights. Do not repeat this or
prior verified prefixes. [Original proof](../initiative/audits/hf-licensed-parent-whole94-20261016-v1/verified.json).

Original-byte readiness and Newton backup are now separate: the full original
Kaggle receipts cover every newly encoded case and both94-window tensor hashes,
complete roster, zero decoding errors/empty cases, original source and whole-byte
hashes. The production wrapper joins all3887indices to the exact target bundle
before any GPU push; its loader also checks every accessed immutable member and
rendered view. The original readiness preceded the independently completed Newton replica.
That independent backup is now complete: all3887cases/4shards were independently
replayed once; original collector80537 is terminalsuccess. It cannot hold the
submission lock.

Production preservation watcher80311/exec52344 is LIVE: each actual completed
session triggers oneCPU/zeroGPU collection, independent full snapshot/model/
AdamW/history/two-rank RNG/scaler checks and private HF publication with complete
re-download/size/SHA/license/provenance. No production epoch exists yet. The new
full-population/session/epoch/range paths pass48focused checks plus7existing
loader checks, Ruff/mypy. These are engineering checks, not an accuracy gain.
[Exact full-production source, stage and ownership](../initiative/audits/licensed-parent-whole94-production-20261016-v1/README.md).

Latest full-production control/source milestone: sourcebe105c27, privateHF
b08293736906370bfc050aef5509c721a457e703,35objects/247805bytes fully re-downloaded/
size/SHAverified under the shared lock byCPU939577. No MRI/identities/labels/
teachers/prediction rows/parent body/weights mirrored. Never repeat this or old
prefixes. [Original preservation receipt](../initiative/audits/hf-licensed-parent-full-production-20261016-v1/verified.json).


## Exact learned-epoch serving and single comparison owners

CPU939778 actually staged the separate inference controls from source095d3a7d.
Live82392/exec1021 waits for the first actual completed, independently preserved
full epoch, then checks exact trained-wrapper/original-class state,32MRI cases,
all94two-view tensors and probabilities on bothT4s. It subsequently qualifies the
exact private offline public MRI/CSV notebook, reusing the immutable original
training output directly; no new model Dataset or repeated weight upload. No
serving GPU has yet been pushed because epoch1 is still training. Three actual
epochs/six notebooks maximum, quota waits reserve no GPU and create no intent.

Live82509 is a separate finite ONE-entry epoch1 owner. It waits for actual
private complete-model preservation and bothT4/publicMRI/CSV acceptance, the
original context comparison's actual scored outcome, no unresolved entries and
fresh authoritative daily allowance. It sends the exact code notebook ONCE,
then observes its actual hidden outcome. Maximum one entry; no three-row CSV
upload, diagnostic model or uncertain-mutation retry. The old head-entry owners are terminal; the distinct regional-only owner is
live. Separate fixed-final E3 owner89000 can send one exact final entry only
after completed/preserved/qualified E3 and the actual E1 score. Inspect original
intents/results before action.
[Actual stage, owners and source](../initiative/audits/licensed-parent-epoch-serving-20261016-v1/README.md).


Actual full-model training and learned-epoch inference/entry milestone source
63952698 is privately fully re-downloaded/size/SHA verified at HF
5c7d47eb24b32ff07d342622733c303fadf06634:30objects/217900bytes,CPU939848,
canonical shared lock. No MRI/identifiers/labels/teachers/prediction rows/parent
body/weights mirrored. Never repeat this or earlier prefixes.
[Verified preservation](../initiative/audits/hf-licensed-parent-live-training-serving-20261016-v1/verified.json).


## New study-pair objective: complete fixed-final model

OriginalCPU939891 COMPLETED0:0/8:13 on evc33. Its actual32MRI-feature seven-process
recovery and uniform frozen-engine numerical equality passed. The distinct ranking
candidate completed all3887studies/10epochs/2430finite updates in417.42seconds,
402956parameters; no CNN was re-encoded or fitted. The original full uniform fit
was reused, not repeated. Half of25%source BCE becomes study-pair logistic risk
on the actual mean-of-two-view probabilities; the75%teacher contribution remains.
Same graph/features/labels/masks/seed/order/update budget. No clinical or
independent internal accuracy claim follows from that completed training.

All TEN immutable epochs were privately fully re-downloaded/size/SHA verified
ONCE at HF9f27825ef383b30bf5c77486e09ada62165be991:40objects/55,403,596bytes.
Final checkpoint e0fc7c891d2d9f00ad9fb37aa4de27c239e1ef6f41c849e5cea77fde9e768a05.
ActualCPU939970 accepted strict fixed-final export, matching the trained graph
with zero probability error on the original32MRI feature cases. The1,626,295byte
component6e0894d90343 is separately private/full verified at
HFca34866401e2e68a87dd8b1f0dcf517d0d14856b:19objects/1,723,620bytes.
Never repeat this fit, either publication, acquisition or full uniform control.

Private Dataset12642673v1,godlydonuts/rsna-ranking-head-e10-6e0894d90343, is fully
6object download/hash verified. Exact private offline precision139119567v1,
godlydonuts/rsna-ranking-precision-6e0894d90343, actually COMPLETED on BOTH T4s:
all32MRI feature cases, maximum CPU/GPU probability error1.1920928955078125e-7,
no new CNN encodings. This is engineering acceptance, not disease accuracy.

Original84212 was stopped ONLY while idle in its confirmed pre-public quota wait;
no public notebook intent/push existed. A SaveKernel response alone does not
prove an entire future timeout is reserved by account stats. Distinct finite
public-only owner84548/exec49246 therefore retains7hours for EVERY actual-unfinished
original full-model epoch plus1hour reserve before its optional9hour public push.
It reuses the exact original Dataset/qualified version/model/source; maximum
ONE public notebook,zero newDatasets/precision pushes/competition entries.
No mutation intent on quota waits; never restart the old84212 or repeat qualified
139119567. Public MRI/CSV and a ranking competition result do not yet exist.
29focused math/recovery/export/compiler checks and Ruff/mypy6sources passed. [Protocol and primary papers](../initiative/experiments/competitive-v1/source-ranking-context-v1/README.md),
[actual original proofs/owners/funding](../initiative/audits/source-ranking-context-20261016-v1/README.md).

The original full CoAtNet139111324v1 independently continues on bothT4s,293finite
updates last verified in epoch1 (293/1458total;60.3%of epoch1). All four original lossless pixel caches are now
also independently collected/replayed on Newton:3887studies,3.204MBprivate
registry SHA8cc5c194d4795f6ab574c28b2cf8eed510c5f92fc6b5baff63049d16323fb1df.
Original collector80537 is TERMINALsuccess; never restart it or repeat transfers.
Total archive bytes39,197,562,880including the unchanged original32. Zero decode
errors/empty cases; all pixels/masks/two-view94-window hashes match.

The rank experiment actually funded all prior607095306179required bytes plus
new256MiBcheckpoint/publication+32MiBcontrols, leaving25.494GB after liabilities.
Raw retained checkpoints138,599,668bytes stay private. No deletion credit or
shared growth. Main training stays unchanged. Old head-entry observation failures are preserved;
choose future ranking entries only after actual public acceptance, relevant
outcomes and fresh daily allowance.


Latest distinct small milestone sourcea6c70b657110 is privately fully re-downloaded/
size/SHA verified at HF117b0b51042e7b54412a3b8b46cf20e38c224320:62objects300619bytes,
CPU940547/canonical shared lock. It preserves full384T4/MRI/entry proofs, actual
context0.948, healthy293-update full fit and finalE3 entry control. No MRI/UIDs/
labels/features/weights/individual prediction rows; never repeat this or prior
verified mirrorHFa9e66639. Latest node33funding607422461891required bytes includes
8MiBnew mirror controls and16MiBfield controls; no originals deleted.
[Exact verified preservation](../initiative/audits/hf-licensed-field-T4-context-score-20261017-v1/verified.json).


## Verified full-field input mechanism, October17

ActualCPU940198 COMPLETED0:0 on the original owning node:32admitted MRI studies,
64views reproduced original320 input hashes exactly. Their unchanged central320
values are bit-equal within full384 tensors, which retain44%more original image
area.69.73%of sampled outer-band pixels were nonzero; this measures image content,
not lesion coverage or clinical accuracy. No model/labels/GPU/new pixel archive.
Actual privateT4 qualification139126353v1 COMPLETED: all learned parameter
values identical at320/384, both32-case inputs exact, cross-GPU384outputs equal,
control error0.00024855, peak1.934GB/device. Exact privateoffline public139128447v1
COMPLETED in23.372s/3studies/zero pair failures/13columns470CSVbytes, fully verified.
The same exact source21439b154382 was submittedONCE as57289392 at00:50UTC; actual
hidden status/score remain unreported at00:53UTC. Owner88168 observes it. The
public ceiling is7h, retaining all unfinished main7h sessions plus1h reserve;
measured throughput estimate6925.6s is not a hidden runtime guarantee.
No384 accuracy gain exists. Original main full-model training stays unchanged. [Actual proof and limits](../initiative/audits/licensed-native-field-20261017-v1/README.md).


## Current outcome and next comparisons

Context57287572 actually scored **0.948**, below unchanged attributed parent0.949;
no improvement established, keep the parent as reference. Its failed read-only
observer and two dependent waiting entry owners are preserved as operational
failures, not ML/submission failures. New exact regional-only recovery uses the
original accepted v1 model/source and fresh account allowance from02:00UTC;
global-repeat/confidence remain unsubmitted, not active owners. Full-field384
public139128447v1 has actual MRI/CSV acceptance and its ONE entry57289392 is
pending/unknown. Main139111324v1 has293/1458finite updates in epoch1; training
80039/preservation80311/serving82392/E1-entry82509 remain healthy.
Distinct E3-entry89000/exec38366 is LIVE, waiting actual fixed-final private
preservation, bothT4/public acceptance, E1 actual score and fresh allowance.
Today's plan is context,field,regional,E1,E3; only context/field are submitted.
There is no guarantee unready candidates finish within the daily window, and no
unqualified duplicate will fill a slot. No scientific recipe changed. [Actual score/recovery](../initiative/audits/released-context-deployment-20261016-v1/README.md),
[field proof and exact public source](../initiative/audits/licensed-native-field-20261017-v1/README.md).

## Matched efficient full-encoder sampling — current engineering work

The original all3887label-free screen found45.6%lower teacher-readout approximation
MSE with16influence-guided draws; this is not disease accuracy. The new matched
uniform/influence full-encoder engine is implemented,46focused numerical/recovery/
compiler checks pass, and original whole94 source stays byte-identical.
CPU940638 staged scientific sourcebe837f9b and funded32MiB controls above carried
607422461891bytes, preserving25.29GB after all liabilities; future full-production
checkpoint funding is not included. ONE private offline twoT4 diagnostic139130771v1
was pushed from exact sourcea01dd9643355, observed by live90613/exec24374. It
requires actual both-role full/stop1/resume equality on16fitting+16disjoint
engineering cases; no GPU acceptance, clinical gain or new full fit is claimed yet.
CPU940658 separately materialized the complete fixed3887x94proposal table from
original features in21.97seconds, zero labels/newCNN/transfers, private2.923MB
SHAe812a9f90cd2b3db92d9acafd28688d5ee6fe16fc188d6517c2bab565f79205e. No Dataset
or slot used. Original main training and today's five-entry plan are unchanged.
[Protocol and papers](../initiative/experiments/competitive-v1/licensed-importance-encoder-v1/README.md),
[actual stage, push and CPU input receipt](../initiative/audits/licensed-importance-encoder-20261017-v1/README.md).
