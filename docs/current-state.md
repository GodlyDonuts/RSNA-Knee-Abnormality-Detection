# Current RSNA Knee research state

Updated October 20, 2026, 05:37 UTC. Goal active; first place and 0.97 remain unachieved. Original receipts and live service state supersede this snapshot.

## Actual competition results

| Model | Kaggle score | Entry |
| --- | --- | --- |
| External R6 reproduction | **0.958** | 57333150 |
| Released S6 reproduction | 0.950 | 57300990 |
| Own whole94 full encoder | 0.949 | 57336190 |
| Own influence IID16 full encoder | 0.949 | 57332456 |
| Own paired / higher-LR adaptations | 0.949 / 0.949 | 57392062 / 57394237 |
| Own uniform / source-label adaptations | 0.947 / 0.947 | 57329796 / 57331631 |
| R6 with own paired member | 0.957 | 57397292 |
| New cross-lineage supervision, epoch1 / final epoch3 | 0.948 / **0.949** | 57418131 / **57422143** |

No original improvement over the public reproduction is established. Engineering checks, training loss and more parameters are not accuracy. Public teacher/base exposure is not independent OOF evidence.

## Active full-model experiments

| Experiment | Actual state at this update | Next action / sole owner |
| --- | --- | --- |
| Changed cross-lineage teacher, fixed three epochs | All 1,458 updates COMPLETE; all three checkpoints privately verified. Final code entry **57422143** accepted once; hidden **COMPLETE/scored0.949**, no error. E1 scored **0.948**. | Original evaluator **21960** ended normally after both outcomes. Never repeat entry or model publication. |
| Matched two-view pointwise versus teacher-ranking | Original H100 **975297 RUNNING on evc48**, unchanged source7f75fd46. Pointwise epoch2/972 updates completed and privately verified throughCPU976879. Epoch3 in progress; ranking arm follows, no ranking result. | Preservation owner **27713** handles each distinct completed epoch once. Do not alter healthy training or its reader. |
| Full source-zero paired-view fitting | **Kaggle139742882 v1 RUNNING**, first epoch of the predeclared three-epoch /1,458-update fit. Sourcebd680ac0, science509db5b9. This is production fitting on all3,887 studies, not the completed diagnostic. | Owner **32102** follows first epoch, independent private verification, then fresh quota/capacity admission before each of two possible resume sessions. It creates no competition entry. |

All three use the full **73,136,096-parameter CoAtNet**, effective eight-study
batch and original accepted fitting cohort. Source-zero removes the source-label
loss contribution while retaining the original parent teacher, masks, proposals,
order and learning rates. It uses two anatomical views and the actual accepted
sequential/DDP engine14f4b234. The H100 comparison uses the separately accepted
joint one-rank engine8384b578. Their backend/RNG grouping differs, so their score
difference will not be a strict source-weight-only causal estimate. Protected
gold and confirmation remain unused. None has established a new accuracy gain.

### Cross-teacher final checkpoint and submission

Original E3 kernel139719974v1 completed1,458 updates; epoch3 fitting took
4,344.73seconds. Final model292,873,139bytes SHA
`b968653621f9d35f92957fbe9c6ce5a7d1944a0379407a581bb8cf13b1817067`.
CPU976299 independently verified model/AdamW/contiguous-history/two-rank RNG/
scaler and all six private objects by full re-download/size/SHA at
**HF25c8cada7265f7c33f9551d1d0bc6511517bebc5**. E1/E2 likewise already privately
verified; no checkpoint or fit is to be repeated. Compilation owner25232 finished.

Final qualification **139739395v1 COMPLETE**, exact
`godlydonuts/rsna-s6-target-e3-qual-c7eae107bea9`. Exact public inference
`godlydonuts/rsna-s6-target-e3-offline-6276abeb0298` v1 completed, then original
entry **57422143** was accepted once. The public three-row CSV was not uploaded.
Authenticated outcome is now **COMPLETE/scored0.949**, no error. It ties the
existing0.949 solo adaptation and does not establish a new supervision gain.
Original evaluator21960 completed both predeclared outcomes and exited normally;
never restart it. Public execution alone was not the evidence of this score.

Correct runtime:
`gcr.io/kaggle-private-byod/python@sha256:2757e0c7d1e0a9cb43da657b97e223c321a98f5014bdf64f44f2f6b083ad2b2f`.
Original failed/unpinned versions and E3 draft reconciliation remain preserved;
never submit or restart them. Exact receipts and lineage:
[S6 production route](../initiative/research/S6-Kaggle-production-route-20261019-v1/).

### Source-zero full fitting: actual dispatch and limits

Private notebook **`godlydonuts/rsna-zero-full-e1-bd680ac064af` v1**, kernel
**139742882**, sourceSHA
`bd680ac064af9810e039633bf26b020c2219709a55989bb6053e379fcaf2e9c3`.
Actual service status is RUNNING. The official API live-log request returned HTTP404, but read-only inspection of
the authenticated Kaggle Logs tab now showed **at least213 optimizer updates**:
loss0.5441966, gradient norm1.695083, two views/eight studies, epoch1, scale1024.
The exact v1/script360284180 is bound by the original authenticated source receipt.
No completed epoch or entire history is claimed. Original API observation loss
was not a training failure or permission to restart.

New models/publication scope is4GiB on evc48 plus2MiB controls,96KiB shared ledger
allowance and zero permanent shared files. Existing liabilities and20GiB/
5,000-file reserves remain funded; no deletion credit. At first push the fresh
free budget was20,442.60 GPU-VM seconds. Each new session is capped6,000seconds
and checks fresh free budget/capacity;3,000seconds remain reserved for fixed-final
serving. All future sessions are not yet fully funded. No paid credits.

Sole finite follower32102/source466a0a88 permits at most two later GPU sessions,
requires each preceding immutable epoch's private independent verification
before resume, and never retries uncertain mutations. It submits no competition
entry. Source/compiler/worker/publisher/funding/push receipts:
[source-zero full T4](../initiative/research/source-zero-full-T4-20261020-v1/).

Initial CPU976504 staged the exact source successfully but the outer wrapper
misclassified SystemExit(0). Read-only976536 reconciled existing bytes and absence
of push intent; staging was not repeated. Original failed wrapper receipt stays
preserved. The separately accepted worker/engine were not changed by that fix.

## Completed prerequisites — do not repeat

- Actual two-T4 diagnostic **139736683v1 COMPLETE/accepted**:17 fitting engineering
  studies,15 disjoint engineering predictions, six finite updates, complete
  full/stop1/resume model/AdamW/LR/cursors/all-rank RNG/scaler/history/snapshots and
 15x12 two-view full94 CSV bytes equal. Original73M model, initiala75cb18d on both
 T4s;383.207seconds and5,866,004,480-byte peak perGPU. Engineering acceptance only.
- Actual CPU input probe **139738563v1 COMPLETE/accepted**:32 fitting studies/
 64views, bit-identical selected images/targets/CPU RNG. Original84.7192seconds
 versus selected26.1145seconds, **3.24414× CPU case-loading speed**. This is not
 GPU or end-to-end training throughput, nor accuracy. The accepted reader is in
 the new source-zero full fit, never injected into healthy975297.
- Private integrity Dataset **12722964v1**,
 `godlydonuts/rsna-window-integrity-85c2fbb342c4`, already fully downloaded/size/SHA
 verified once. All3,887 studies ×2views ×94windows bound by the original private
 23,384,320-byte manifest. No array/signatures in Git/Mac. Uncertain first creation
 was reconciled read-only, not repeated. Original CPU failure/finalizer receipts
 remain preserved.
- H100975954 was cancelled only while held/PENDING/00:00:00 after the actual T4
 acceptance. Neither a H100 recovery pass nor a model failure. Never revive it.
- Original H100975297 pointwise epoch1 privately verified once throughCPU976356,
 six objects at **HF87b7ef555f1205e3c38f823aedbaa5cf5827a880**, modelSHA
 `6dad5a3ce5c1a800f0291373ac96b7d311c4b1f183babb2361407a59514c7fac`.
 Owner27713 handles only the remaining five epochs; cancelled975129 never revived.

The source-zero hypothesis follows a fitting-only **full94 parent-logit**
derivative measurement: source/teacher term norm ratio1.9622 and38,453/93,130
nonzero components opposed. These are not sampled-window student logits or
encoder gradients; they do not prove labels are wrong or identify the cause of
low scores. The new experiment tests the hypothesis rather than treating it as
an established explanation.

## Preservation and evidence

Small source milestonea619666c is privately mirrored at
**HFb50caa1ad45839c7f9c31980787805da1c2bf6d6**, CPU976315:44objects/191,323bytes
independently downloaded/size/SHA verified under the canonical shared lock.
Its snapshot predates later CPU/GPU terminal acceptances and full-source-zero
production dispatch. Never repeat its prefix or claim those later receipts were
included. Subsequent selected source56747336 and46 small objects/209,448bytes are now
privately verified at **HF381ae720b6498227c4991f20d130d9f634c175e0**, CPU976734.
This distinct milestone includes full source-zero dispatch/final serving controls,
actual input/numerical acceptance and final cross-teacher0.949; every object was
independently downloaded/size/SHA checked under the canonical lock. No MRI/IDs/
private arrays/weights were included. Never repeat its prefix. Models remain
separately preserved by their original publishers.

Historical exact sources, failures, numerical checks and older model lineages
remain in the linked original audit directories and Git history. This current
page reports actual state rather than accumulating conflicting dated snapshots.
Relevant scoped22 tests passed for the new full-fit compiler/publication bindings;
tests establish engineering behavior, not clinical or competition accuracy.

[Execution continuation](codex-continuation.md) contains original controls,
source pins, storage funding and recovery bindings. Never duplicate a live owner,
job, Dataset, notebook push, checkpoint publication or competition entry.


## Final source-zero serving preparation

A distinct fixed-final-only inference compiler and provenance validator are
implemented in `initiative/research/source-zero-final-serving-20261020-v1`.
The original full94 two-view inference math stays unchanged. The route requires
actual1,458-update source-zero completion, all six independently private-verified
objects and the owning-node publication pointer before any CPU compilation.
The account adapter retains canonical claim/ledger, exact version/source/runtime,
free budget, two-lane capacity, actual both-T4 qualification, public CSV checks
and fresh competition allowance. Maximum future entries:one fixed-final model.
This preparation is not actual source-zero T4 qualification or a submitted model.


Actual account-control CPU976721 staged the new sources once on evc33 from
93e19967. Sole finite evaluation owner **33801** is alive, waiting without any
compilation intent; max30h/two serving GPU sessions/one final-only code entry.
Never restart it. Original fitting32102 and H100 preservation27713 stay separate.
All48 scoped compiler/provenance/account regression checks passed. This does not
establish actual source-zero GPU qualification or clinical improvement.


## New mechanism: cached full-context attention changes

Instead of normalizing only over16 sampled windows, the research prototype uses
cached full94 teacher attention/votes and estimates the student's change from
that reference. This adapts control-variate and local linearization ideas to the
existing MIL graph; it is not a claim that these ideas are newly invented.

- Original ratio estimator **CPU976806 FAILED** its gate:2invalid denominators
 at synthetic drift.5. It stays rejected despite lower conditional MSE.
- Separate linearized version **CPU976828 PASSED** all predeclared levels and
 all12targets on32cached fitting MRI feature records,128IID16draws/two views.
 No clinical labels or actual model updates. All-draw approximation MSE ratios
 to original SNI were **.001779/.027942/.162722** at synthetic drift.05/.2/.5;
 no invalid outputs. This is neither clinical accuracy nor measured student drift.
- The linearized estimator has exact-reference logits/expected first gradients,
 but finite drift adds Taylor error. No random denominator, clipping or hidden
 fallback. Actual dropout/precision/reference staleness after real learning remain
 unaccepted. No live training or original inference was modified.
- First original CPU976785 refused before numerical work due historical funding/
 old control headroom. Read-only976787 proved no screen state. Distinct current
 controls explicitly admitted256KiB within the existing8MiB on33 and preserved
 another2MiB for original future fit/serving; no original deletion/funding credit.

[Mechanism, primary papers, all negative/positive evidence and agent decision](../initiative/research/cached-attention-control-20261020-v1/README.md).
Next is a separately funded actual-encoder diagnostic with matched deterministic
mode, exact moment alignment, complete recovery and gradient/drift measurements;
no full model/GPU dispatch or submission exists for this mechanism yet.


The selected new mechanism sources/negative ratio result/positive linearized CPU
result and sanitized source-zero progress receipt are now privately preserved:
**HFaa111c78483a5f0b79ff9b2d56be187fdd470294**, CPU976868/source4eb1a39a,
20objects/187,637bytes. Every object independently re-downloaded/size/SHA under
canonical lock; no pixels, reports, private arrays, individual clinical records
or learned weights. Never repeat this mirror. This does not change the lack of
an actual encoder/clinical result for the new mechanism.


Latest actual production preservation: original975297 completed pointwise
**epoch2/972updates**. CPU976879 privately published/independently downloaded/
size/SHA verified all six objects at **HF755515ba5576fcadff371273d056370338ac6d3e**.
Model292,867,251bytes SHA3e8bf39d37137785574018bbb8d85461eb44a94920c8b64b73b774b6546d7d3d.
Never repeat pointwise1/2. Owner27713 now waits for pointwise3, then teacher_rank1/2/3.
The four remaining checkpoints and fixed-final accuracy comparison are unobserved.
This epoch2 receipt postdates the small-source mirror4eb1a39a; its model/provenance
are separately privately preserved at the revision above.


## October20 — actual full reference cache and real-encoder dispatch

The original private cache was materialized and transferred exactly once by
CPU976992/976991: all3,887 fitting studies, two views,94windows,12targets;
71,148,396bytes, SHAee28ba94592f29fdd56f50d65350a1e6c8be2ba6d2f893e740b9ccfd46d31695.
Exact byte/SHA/EOF, finite FP32 arrays and ordered roster passed. Original FP32
head probability reduction gap was1.7881393432617188e-7; original cached-FP16
probability gap was0.00029981136322021484. No arrays/UIDs reached Mac/Git/HF.
Reuse this cache; never rebuild/retransfer it without an actual new requirement.

New actual encoder implementation/source9e76db5d uses deterministic eval mode
WITH gradients in both matched arms and an FP32 readout. Its explicit encoder
checkpointing override preserves activation checkpointing despite eval mode;
all encoder/head parameters remain trainable. Full94 inference remains original
and requires no reference cache at test time. Twenty-nine scoped CPU checks
passed; these are not H100 acceptance or clinical evidence.

CPU977068 admitted exact cache/source/runtime/original full-pixel bindings and
separately funded128MiB cache/8MiB controls/12GiB engineering-plus-runtime cap.
CPU977099 then submitted new GPU977100 ONCE on physical two-GPU evc48, requesting
one H100/two CPUs/32GiB/30minutes. At the actual queue observation it was
PENDING(Priority); no optimizer progress or GPU acceptance is claimed.

Scope: reference/linear arms,17 fitting engineering studies,15 disjoint fitting
prediction studies, two epochs/six updates each, fresh full/stop1/resume
processes. Requires complete decoded model/AdamW/LR/counter/all-RNG/scaler/history
and snapshot equality, exact full94 two-view15x12 prediction CSV bytes, finite
updates and actual reference drift. Engineering results cannot select a clinical
winner or be submitted. Full production fitting is NOT funded/dispatched here.

Original root `/tmp/rsna-cached-attention-encoder-20261020-v1` on evc48;
inspect original977100 and original GPU-run/dispatch/phase receipts read-only.
Never repeat submit_gpu or restart on observation loss. Original975297 and local
owners27713/32102/33801 were verified live, left unchanged. Source-zero139742882
was still RUNNING via its live original follower; final serving waits for final
verified fitting. No new score exists. Goal remains active/unachieved.

[Original new source, cache, funding and dispatch receipts](../initiative/research/cached-attention-encoder-20261020-v1/README.md).
