# Current project state

**Verified October17,2026,18:50UTC. Target0.97; actual best0.950.** Our target is **0.97**. Best measured
project public AUC remains **0.950**. The target has not been reached.

| Candidate | Submission | Public AUC | What was demonstrated |
|---|---:|---:|---|
| Released S6, exact offline inference | **57300990** | **0.950** | External weights and inference reproduced; COMPLETE, no error |
| Released nartaa CoAtNet | 57258866 | 0.949 | External weights reproduced |
| Our whole94 parent adaptation, epoch 1 | 57294905 | 0.949 | No improvement over unchanged parent established |
| Independent N3, epochs 4/8/12/16 | 57256756/57259376/57262392/57265916 | 0.821/0.869/0.890/**0.894** | Own MRI training from general ImageNet initialization |
| Context / full-field / regional additions | 57287572/57289392/57291182 | 0.948 each | No demonstrated public gain |
| Earlier own DINO | 57182737/57182547 | 0.872 | Historical own-model reference |

Public AUC is not private-final performance or clinical accuracy.
[Score receipts](../initiative/audits/s6-actual0950-result-20261017-v1/README.md).

## Compute authorization, October17

The owner explicitly authorized using as many accessible Newton **H100** GPUs
as useful, superseding the former two-node cap. **Physical4-GPU and8-GPU nodes
are prohibited even when requesting only one GPU.** Admit only physical1/2-GPU
H100 nodes; scheduler/account limits and storage/data requirements still apply.
Parallelize distinct work, reuse completed shards and preserve healthy jobs.
[Fresh inventory and exact scope](../initiative/audits/h100-capacity-expansion-20261017-v1/actual-inventory-and-policy.json).
The fresh inventory found one momentarily idle usable H100 on evc46;
remaining non-reserved eligible H100s were allocated. Existing native inputs
and runtime are owning-node assets on evc33; fresh remote staging is required
before moving an inference/trainer to another node. The first
spatial scout949626was actually PREEMPTED after91seconds, preserving26/32studies
and1912windows. No full scout acceptance or new accuracy is claimed.

## Latest execution, October 17, 18:50 UTC

Best measured public AUC remains **0.950**, reproduced S6 entry57300990;
own independently MRI-trained N3E16 is **0.894**. Target0.97 is unachieved.
No new score is claimed. Full-CNN label training and all three immutable epochs
completed and are privately fully verified (final HF56581aab…); never repeat.
Original scoring owners78386/10613 and read-only reference observer25736 remain
live. Uniform public139269448v1 and exact reference script139271768v2 still have
no actual public acceptance/hidden score. Preserve their original intents.
Latest authenticated allowance17:50UTC was5used/0available; refresh before entries.

The [new owner frontier memo](../initiative/research/frontier-GRAIL-20261017-v1/README.md)
is preserved verbatim and assessed against actual source evidence. Next research
priority is **criterion-relational residual learning**, with an aligned/sham
control sharing original images, encoder, targets, parameter graph, histogram,
metadata and compute. The zero-initialized branch now implements distinct-source
corroboration, physical-gap-weighted continuity/variation and common nuisance
inputs. It is implemented/local-tested, **not trained or accuracy-validated**.
Do not silently turn Gold58 into meta-training or claim released-encoder head CV
is independent whole-model OOF. Full one-header patient audit already covers4410
studies/no repeated IDs; it does not prove true-patient independence.

**Actual CPU949320:** all3887 fitting studies/four fixed-final heads/both weak
sources/all12targets quantified achieved response in41.24seconds. None of5832
updates clipped (maximum norm0.135618). In S6-label/full94, median Fracture
probability movement0.0750 yielded77/660 pointwise-correctable opposed pairs
reordered, with55 new inversions. These are dependent weak-label fitting pairs,
not clinical errors or a generalization gain. This rules out clipping in these
runs and distinguishes loss-permitted correction from achieved discrimination.
[Complete results and limits](../initiative/audits/fitting-achieved-response-20261017-v1/README.md).
Never repeat the completed diagnostic or old head fits/publications.

**Actual CPU949445:** unchanged N3 rosters2247fit/1156development and all selected
source metadata have usable within-series geometry.162651/84333original windows
and127734/66237disjoint-support repeat edges are bound without pixel or clinical
target reads. Their median separation is about20mm: this is NOT literal adjacent-
image clinical evidence. All17671selected series lack cached FrameOfReferenceUID
and SliceThickness. Cross-plane registration and lesion volume remain unproved.
Original frozen spatial-feature scout949626was PREEMPTED after26/32studies;
CPU949734fully verified all26files/1,912windows and actual1024x12x12maps.
Distinct normalQOSrecovery949735is PENDING; it reuses26completed forwards and
encodes only six missing cases. Full32study scout acceptance is absent;
require its actual source/roster/base-logit/device/throughput acceptance before
one full cache pass or aligned/sham fitting. Reuse original N3E16/native bytes.

The [latest spatial-evidence proposal](../initiative/research/frontier-spatial-evidence-20261017-v1/README.md)
changes the scout: retain actual N3 **pre-pooling spatial maps**, not merely
window embeddings. The actual regional operator has a reproduced location
collision; the new centered co-support and weighted-marginal/exact-support null
have nine focused tests. These establish representation/math, not MRI gain.
The bounded32-study H100scout completed26cases before Slurm preemption;
normalQOSrecovery949735is pending and reuses those26verified forwards. It records
actual dimensions, original letterbox/source geometry, base-logit fidelity and
cost. No full spatial cache or fit is admitted. Cross-acquisition geometry is
unqualified; cached frame/thickness fields are absent. Existing longitudinal
GRAIL work remains separate and cannot stand in for the spatial experiment.

**Actual CPU949671:** fixed original N3E16predictions on1156development
studies decomposed exactly across12manufacturers/42manufacturer-model domains.
Macro0.9209253491221031reconstructs to<1e-12. Within-model ACL.9540vs overall.9549
and medial-meniscus.9458vs.9484show no broad collapse under this grouping;
lateralOA.7981vs.8349and effusion.9042vs.9159each have only9positive exams.
Only2.3–10%of target pairs are within manufacturer/model. This is descriptive,
not a causal scanner diagnosis or new validation. [Original results/limits](../initiative/audits/N3-fixed-source-domain-20261017-v1/README.md).

**Original GPU942047 actually completed0:0** and passed the three-MRI S6 H100
FP32/FP16 fixture:1586windows, maximum probability gap0.0003156333. This replaces
old pending snapshots; it does not qualify GPU anatomy preprocessing, a trainer
or hidden-scale serving. [Exact original terminal](../initiative/audits/s6-source-tree-admission-20261017-v1/actual-original-H100-accepted.json).
No new GPU was used for its collection; do not repeat it. All healthy source
recipes, accepted masks/folds, original checkpoints and protected cohorts stay
unchanged. All46focused new cases and115-module mypy passed; current v2coverage is89/89.
Initial fullsuite had4660pass/4fail/6skip: one child lackedPYTHONPATH and
three bound source edits made during replay. The same four isolated cases now
pass; a new complete suite freezes and checks all numerical sources. Global
1327lint/57format debt is unchanged. Numerical source and actual diagnostics are published on main; distinct
private small-milestone verification is pending. Full-source QA is still running.

## Completed CPU research: labels × whole-study coverage

**Actual 2026-10-17 15:44 UTC.** Original CPU recovery948113passed all four
roles with exact full/stop1/resume learned/AdamW/LR/cursor/all-RNG/history/epoch
states and16×12prediction-byte equality. The fixed numerical source stayed
unchanged after the original zero-runtime scheduler-wrapper failure948089.
Distinct corrected owner93414then completed all four full3887fitting runs in
CPU948122: **3epochs/1458finite updates each**,279832learned readout
parameters, original frozen CNN features. Source labels (original/S6) and training
coverage (16IID/all94) are the two factors; order/views/teacher/masks/init and
optimizer remain common. CPUFP32/zero dropout are shared across these four,
so comparison to the live full-CNN family is not encoder-only causal evidence.

All **12immutable epochs** are privately preserved once at
**HFc4cba4c210542307436a3bdc89034f5f99633ce4**, CPU948191; all60objects/14984974bytes
were fully re-downloaded/size/SHA verified with source/licence/parent/runtime/
roster proofs. No encoder weights, MRI/feature/label rows or prediction arrays
were copied into this publication. No fitting loss or teacher agreement is an
accuracy result; released-encoder exposure precludes independent whole-model OOF.
Protected clinical cohorts remain unused. No new Kaggle score exists.

Numerical/control sourceeed22c54is separately privately verified at
HFebffa540e0f0f033b4acfe7324ab9b713e3da7e9 (14objects/94480bytes,CPU948112).
The separate256MiBnode33cap retains original liabilities. Owner93414stopped after the first publication path error; standalone corrected
publisher completed after original-commit absence and licence reconciliation.
Never repeat either recovery/full-fitting phase or12epoch publication. Existing
full-encoder61037and matched-final scoring78386remain separate original owners.
[Exact protocol](../initiative/experiments/competitive-v1/cached-full-study-factorial-v1/README.md)
and [actual recovery/full/private proofs](../initiative/audits/cached-full-study-factorial-20261017-v1/README.md).

## Full-encoder campaign: fixed-final training and preservation complete

The **source-label comparison** completed training all 73,136,096 parameters on the
same 3,887 fitting studies as the completed uniform control. Source labels are
the only scientific change: images, masks, teacher, sampling, optimizer and the
fixed three-epoch/1,458-update recipe remain matched. Neither label source is
clinical ground truth.

- **139234107 v1** completed epoch 1: **486 finite updates** on two T4s.
  Exact source/runtime/inputs and the completed model/optimizer receipt passed.
- Its actual GPU recovery **139230981 v1** passed complete old-control replay,
  both-label full/stop/resume state equality and prediction-byte equality.
- **60358** finished successfully. **61037** independently verified and privately preserved epoch 1
  at **HF33d992cfa8157d8b6fd3cc1a547b80838b874b44** (CPU947422; all five objects
  re-downloaded/size/SHA verified), and resumed epoch 2 once as **139245643 v1**, now COMPLETE at972cumulative updates and privately verified at HFf5a74128…/CPU948213.
  Exact source and private offline T4 metadata and independent complete-state/private verification passed. Original E3 is running as139256441v1. The fixed-final model is now complete and privately verified viaCPU948773/
  HF56581aab…; its actual accuracy result remains pending. Scientific source remains
  `990c8ded` throughout the campaign.

[Actual recovery, training and continuation](../initiative/audits/097-label-campaign-continuation-20261017-v1/README.md).

## Completed models awaiting scoring

Whole94, uniform16 and influence16 each completed three epochs and 1,458 updates;
all nine immutable checkpoints are privately fully download/size/SHA verified.
Do not repeat their training or preservation.

Uniform/influence final inference sources are now prepared by **CPU946894**;
**CPU946946** validated both actual preserved snapshots through their serving
adapter. They retain original full94, two-view inference and reuse immutable
notebook weights. Uniform qualification **139267646v1** is now actually RUNNING; no accepted
qualification/public output or new competition score exists yet. The label final
model/private proof also completed, and CPU948784 prepared its serving source.
Original scorer78386 continues these comparisons once each actual check passes.
[Exact scoring preparation](../initiative/audits/matched-final-scoring-preparation-20261017-v1/README.md).

The stronger public reference **139199164 v2** remains Queued on its actual
logs page; an absent raw API status decodes QUEUED in the official SDK. It is unqualified and unscored. Its separate
interactive run passed, which is not a hidden result. Retain R6 Dataset v1;
v2 contains different R7 weights under the same filenames. Never repeat Save,
submit source-only v1, or upload the public three-row CSV.

## Other actual outcomes

The CPU32 scout **139205901 v2** is ERROR and observer40709 finished. Its
classifier actually scored all 32 studies without fallbacks in 9,618.9 seconds;
the outer wrapper then failed a final retained-byte/RSS assertion. Resource
acceptance remains unproven; no repeat run was launched. Original private
outputs were preserved, and only aggregate failure/cost evidence was collected.

H100 pilot **942047** remains pending with zero runtime on evc33, where both GPUs
were allocated. Immediate evc40 device-only request947064 could not allocate;
no new CUDA result exists. Healthy Kaggle training remains untouched.

## Resources and handoff

Authenticated competition allowance at **15:53 UTC**: **5 used, 0 available**. Refresh it
before any entry. Single canonical ledger/claim remain on Newton; credentials
stay in ignored0600.env/RAM/stdin, and large artifacts stay private.

The new serving controls use 4 MiB of the existing node33 future allowance,
leaving **4,128,768 other future bytes** within the unchanged 32 MiB envelope.
Preserve the unpaid part of that 4 MiB allocation as well as other liabilities;
deduct paid bytes once. Original 20 GiB and 5,000-file reserves remain.

Source **f25a17d7** is privately mirrored at
**HF276963b335043f552d8f3e8cfd59c8afe92247a3**, all **29 objects / 152,794 bytes**
fully downloaded/size/SHA verified by CPU947499. It preserves the actual epoch-1
milestone and scoring follow-through. Original scoring owner **78386** is live,
waiting for the final treatment/private proof. It can qualify the three matched
finals and enter ready candidates promptly under fresh limits, observing each
actual hidden outcome before another entry. Do not duplicate either owner or
any verified private publication.
[Preservation receipt](../initiative/audits/hf-matched-final-scoring-ready-20261017-v1/README.md).

Read the [research brief](research-brief.md), [execution handoff](codex-continuation.md)
and [research-agent handoff](research-agent-handoff.md). Accepted masks/folds and
protected gold/confirmation remain unchanged. Released encoders saw internal
studies; head CV on their features is not independent whole-model OOF. Negative
results and original evidence remain in the research archive.

CPU947337staged one new matched-final scoring coordinator; actualCPU947372
refused the unfinished treatment before anyGPUintent. It will wait original
61037final/private proof, then qualify/public-run the uniform, label and influence
E3models and enter only exact accepted versions under fresh allowance. A ready
entry need not wait for unrelated qualification; actual previous hidden score
is required before the next entry. Maximum6GPU pushes/3entries,20hour finite
owner **78386now armed/live**, waiting original61037final/private proof. [Exact coordinator](../initiative/audits/matched-final-scoring-coordinator-20261017-v1/README.md).

**14:21UTC actual epoch1 terminal:**139234107v1 COMPLETE,486finite updates,
modelSHAe78824b3…/292872115bytes; original60358finished. Existing61037completed full-state/private verification at
HF33d992cfa8157d8b6fd3cc1a547b80838b874b44 viaCPU947422/all5objects verified,
E2subsequentlyCOMPLETED972updates at15:46UTC; private verification later passed viaCPU948213/HFf5a74128….
Do not repeat epoch1publication.
No fixed-final or new score yet. [Actual epoch records](../initiative/audits/s6-label-full-epochs-20261017-v1/README.md).

Latest distinct source**f25a17d7** is privately preserved at
**HF276963b335043f552d8f3e8cfd59c8afe92247a3**,29objects/152794bytes fully
re-downloaded/size/SHA verified viaCPU947499. Original scoring owner**78386**
is actually live, immutable sources/private pin accepted, waiting actual
labelE3/private proof. Maximum6GPU pushes/3code entries; no model/weightDataset
copies. Do not restart owner61037or78386or repeat either private publication.
[Exact new preservation](../initiative/audits/hf-matched-final-scoring-ready-20261017-v1/README.md).
