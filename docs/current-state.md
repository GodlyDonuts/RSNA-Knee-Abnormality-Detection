# Current RSNA Knee state

Updated **October 21, 2026, 17:21 UTC**. This replaces stale operational snapshots. Historical experiments and original receipts remain under `initiative/research/` and Git history; no original scientific evidence was removed.

## October 21, 17:21 UTC deployment update

Original Newton consumer3332 stopped after a successful qualification compile:
its inherited parser expected an older status name. The existing private
Dataset12762386v1 and exact compiled source were reconciled read-only; no
creation/upload/compile was repeated. Distinct finite continuation **8547**,
source4297039c, has pushed exact private qualification **140062137v1** once:
`godlydonuts/rsna-predictive-c100-qualification-e6-fe364a3c8a9b`.
Authenticated observation reports RUNNING. Actual two-T4 acceptance and competition entry are still outstanding.

Only the waiting local Colab consumer3333 was explicitly stopped before any
handoff/account phase. Corrected consumer **8603**, same source4297039c, reads
the same original handoff and protects Newton's budget. The actual Colab A100
training and queued publisher remain untouched; latest observed step2240/2916.
Never restart3332/3333 or repeat the already-pushed qualification.
[Exact continuation and receipt](../initiative/research/predictive-final-status-recovery-20261021-v1/README.md).

Existing passive Newton logs show a nonzero diagnosis-conditioned contribution
(final sampled logBF RMS0.3729, mean posterior shift7.54percentage points), with
mixed fitting BCE differences across the ten sampled updates. The final seven
fitting studies show−0.03783BCE, but this is not held-out accuracy or proof of a
gain. [Original aggregate evidence](../initiative/research/predictive-C100-passive-final-20261021-v1/README.md).

## Results we actually have

| Candidate | Actual competition result | Interpretation |
| --- | --- | --- |
| External R6 ensemble reproduction | **0.958** | Best scored system; not an original model or a from-scratch reproduction |
| Our earlier adaptations | **0.949** | Best own adaptation; no demonstrated gain over R6 |
| Source-zero, pointwise, ranking | **0.948 each** | Completed entries; no established improvement |
| Predictive B epoch6,57482618 | **Failed; no score** | Hidden notebook exception, zero output bytes |
| Predictive C epoch6,57482739 | **Failed; no score** | Hidden notebook exception, zero output bytes |
| Predictive C0 epoch6,57482858 | **Pending/unknown** | Still unreported at latest authenticated observation |
| Newton C100 | **Six epochs completed; no score** | Final checkpoint privately preserved and handed to deployment |
| Independent Colab C100 | **Training; no score** | Different seed; not an accuracy gain yet |

B/C/C0 each completed six epochs/2,916 updates; all18 immutable checkpoints are privately verified. Their actual two-T4 and three-study public runs passed, but that did not establish hidden robustness. Kaggle supplies a generic exception, not a stack trace. [Actual B/C outcomes](../initiative/research/predictive-hidden-input-audit-20261021-v1/actual-hidden-outcomes.json); [original serving and entries](../initiative/research/predictive-final-dataset-continuation-20261021-v1/README.md).

## What is training now

**Newton C100:** original GPU995351 COMPLETED0:0 on ordinary evc27 at16:51:31UTC, after6h52m12s. All six epochs/2,916 updates and23,322 study exposures completed. Source8f1e8f2b/bindingf0d73e6c,3,887 fitting studies. C100 increases the density objective coefficient from0.1 to100 after measured fitting-only gradients showed severe dilution. It does not prove the new weighting improves diagnosis. The original model/optimizer/recipe were not restarted or changed. All six checkpoints are privately preserved. [Actual final evidence](../initiative/research/predictive-C100-private-backup-20261021-v1/actual-final-preserved.json).

**Independent Colab C100:** A10040GB, seed20261021, same3,887 fitting studies and six-epoch budget. Latest actual visible output: **epoch5/update2,240**, finite encoder gradient10.0596/head0.9171 and real encoder updates. This is a **73,271,124-parameter CoAtNet/predictive-evidence model**, not DINOv3. Its different seed/runtime is not a perfectly matched causal experiment. [Actual optimizer receipt](../initiative/research/colab-C100-epoch2-recovery-20261021-v1/actual-progress-2240.json). Colab reports264.99remaining resource units and approximately5.3units/hour; one A100 session remains active.

Colab's first recovery hit a Kaggle HTTP429 during publication after epoch2. The complete update972 checkpoint was retained and restored exactly. Recovery-v2 does no network work during training; a distinct queued publisher will preserve epochs2–6 and the terminal, with at least900seconds between Dataset creates. Epoch1 is already privately verified. Replayed973–975/possible unlogged976 are disclosed. Original ten-hour fitting bound and300-resource-unit authorization remain. [Exact recovery and publication](../initiative/research/colab-C100-epoch2-recovery-20261021-v1/README.md).

## What was fixed operationally

HF storage rejected Newton epoch4’s commit with HTTP400 requiring recharge; billing was unchanged. Epochs1–3 retain their verified HF proofs. Original backup1000318 created private Kaggle Dataset12762120v1 once for epochs4–6, then stopped on an immediate GetDatasetStatusHTTP403. A fresh read found the same Dataset READY. One-CPU1000389 fully downloaded and size/SHA verified all16 original objects, reconciled the retained original claim, and completed the explicit Kaggle proofs. No upload, creation, export or fitting was repeated. Original failure history is preserved. [Actual final and reconciliation](../initiative/research/predictive-C100-private-backup-20261021-v1/actual-final-preserved.json).

The old final handoffs86015/95310 stopped before new account phases. Distinct finite continuations **8547/Newton** and **8603/Colab**, source4297039c, are live: Newton has accepted its original completed, fully preserved final; Colab is still training. They will use the new exact faster/resilient serving path and independently pass trained two-T4 qualification, exact public source/output acceptance and fresh quota before one code entry each. Colab reuses its existing final Dataset and waits for Newton's canonical entry. [Newton handoff](../initiative/research/predictive-fast-newton-deployment-20261021-v1/actual-started.json); [Colab handoff](../initiative/research/predictive-fast-colab-deployment-20261021-v1/actual-started.json).

## Hidden failure investigation and serving evidence

The CPU-only private audit140048037v1 completed in720.255seconds. **203,001 headers from1,087 of4,407 accessible training studies passed** the tested original metadata assumptions; all three public full decoders passed. Coverage is partial and does not test hidden images, all pixel bodies, model numerics or long-run memory. It weakens the metadata hypothesis; the root cause remains unproven. [Actual audit](../initiative/research/predictive-hidden-input-audit-20261021-v1/actual-complete.json).

The successor keeps original successful decoding and tensors. For rejected studies it can recover usable original MRI from certain optional-metadata irregularities, counting omissions/repairs; all-unusable studies still fail. A numerically failed FP16 forward is recomputed at FP32 using the same model/context. Strict memory/precision qualification remains, but a valid completed competition CSV is not discarded solely for exceeding the tighter qualification memory threshold. Physical Kaggle limits remain. There are no constant predictions or alternate-model fallbacks. These are tested prospective corrections, **not a confirmed explanation or hidden success**. [Exact changes and limitations](../initiative/research/predictive-inference-resilience-20261021-v1/README.md).

The separate selected-frame CPU experiment140041310v1 preserved all original context bytes in five trials. Warm preparation mean31.8422→22.9442seconds (**27.94% less**), assembly17.3616→4.7069seconds. Descriptor batching also preserved original bytes. Public three-study CPU timing is not hidden-scale acceptance. Both new trained qualifications must compare complete original/optimized contexts and FP16/FP32 predictions. [Actual preparation result](../initiative/research/predictive-selected-frame-preparation-20261021-v1/README.md).

The separate private **large-study CPU stress140052135v1 completed**: all seven studies selected by frame count/pixel-volume from2,830inspected studies passed both original and successor preparation, with bitwise-equal complete contexts. Largest pixel-volume case peakRSS fell6.65→4.74GB. Original-first/cache effects confound the observed aggregate timing reduction45.63%; fresh child processes do not test long-lived allocator or GPU behavior. Both paths passed, leaving the earlier hidden failure unexplained. [Actual result](../initiative/research/predictive-large-study-CPU-20261021-v1/README.md).

The separate four-core CPU experiment140055931v1 also completed. Balanced warm preparation27.0233→21.6329seconds (19.95% less) using two threads per lane; maximum image difference1.1920928955078125e-7 and every other context field exact. This is a promising contingency. Current final consumers retain their existing path; adoption would need an actual trained prediction comparison and justified available GPU capacity. No model, labels, GPU quota or competition entry was used by this experiment. [Original receipt and decision](../initiative/research/predictive-thread-budget-CPU-20261021-v1/README.md).

The apparently flat density loss is dominated by a fixed Gaussian normalization constant: at weight100,91.89385 of the reported92.50460 contributes zero gradient. At update1860 the nonconstant objective is about0.61075. This corrects interpretation; no loss or healthy recipe was changed, and loss scale is not clinical accuracy or a measure of gradient contribution. [Source-bound calculation](../initiative/research/colab-C100-epoch2-recovery-20261021-v1/actual-progress-1860-objective.json).

The four-core contingency now has a separate implemented compiler layer and two
passing focused tests. It retains original one-thread contexts for comparison
against the same trained model and requires FP32 probability difference<=1e-5
before deployment. No owner, model, notebook or account phase changed; actual
trained acceptance remains outstanding. [Prepared contingency](../initiative/research/predictive-thread-budget-serving-20261021-v1/README.md).

A fresh authenticated read at16:48UTC still found B/Cfailed, C0pending/unknown,
and two slots. The original canonical rows were inspected read-only. A preceding
CPU observer used the wrong ledger-state literal and stopped; this was an
observer error, not a new training or submission failure. [Original receipt](../initiative/research/predictive-hidden-input-audit-20261021-v1/actual-hidden-outcomes-1648.json).

## Decisions and limitations

- B directly classifies with a diagnosis-independent reconstruction objective; C predicts withheld MRI descriptors conditioned on diagnosis; C0 is the no-density-loss ablation. All inherited this branch's public CoAtNet encoder and train it after the first three head-only updates. The principal original comparisons were fixed epoch6 C−B/C−C0; hidden failures currently prevent those comparisons.
- G100's separated generative-gradient pilot was **rejected for full training**: final20 fitting exam loss0.867789 versus C1000.288520, total gradient61.39 versus2.99, clipping100% versus85%. This is bounded optimization evidence, not clinical accuracy or a general rejection of generative learning. [Original results](../initiative/research/predictive-matched-pilot-results-20261020-v1/README.md).
- Visible contrast normalization showed no consistent fitting advantage in its matched pilot; its full run was deferred. The standalone optimized input work remains preserved. [Decision](../initiative/research/predictive-matched-pilot-results-20261020-v1/contrast-agent-decision.json).
- The inherited parent overlaps the weak development cohort; historical Gold58 selection also limits independence. Training loss, engineering examples and aggregate leaderboard results cannot establish paired clinical confidence intervals. Protected data are not newly tuned in this branch.
- The prepared C+R6 blend is conditional on a qualifying scored C result and separate runtime funding; no combined entry exists. The best0.958 baseline remains preserved.

## Deadline and resources

Competition ends **October22 23:59UTC**. Latest intended full-scored launch is **October22 noonUTC**, allowing the nine-hour hidden run. At15:55UTC authenticated Kaggle reported **two remaining daily slots and3,056.1426641 free GPU-VM seconds**. Each later mutation rechecks live capacity; no paid credits. B/C/C0 outcome owner81657 stopped on a JSONDecodeError. A separate strictly read-only canonical-ledger/authenticated query at16:22 confirmed B/Cfailed and C0stillpending; it did not restart the old consumer or submit anything. [Actual query](../initiative/research/predictive-hidden-input-audit-20261021-v1/actual-hidden-outcomes-1622.json).

Current node free666,825,506,816bytes and shared free22,872,809,472bytes/5,371files passed explicit liabilities and20GiB/5,000-file floors. Shared future scope123MiB/370files reflects six proved original GPU logs already counted physically. Large artifacts remain private. Verified source and receipt changes are committed and synchronized to GitHub; **not privately HF-mirrored**, since HF storage is exhausted. Latest small private source mirror remains v7.

See [execution continuation](codex-continuation.md) for exact owners, journals and next actions. Old stopped owners must never be restarted.
