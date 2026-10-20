# RSNA Knee Abnormality Detection

**October 20, 2026: best verified Kaggle score is 0.958, from an attributed
external R6 ensemble reproduction. Target 0.97 is not achieved.** Current encoder
adaptation uses a **73M CoAtNet**, with earlier DINOv3 research retained separately.

Use [current measured state](docs/current-state.md) for results and unfinished
work, and [execution continuation](docs/codex-continuation.md) for original live
owners and exact operational bindings. Historical dispatch files are audit
artifacts, not instructions to restart completed or uncertain operations.

## Verified results and model lineage

| Result | What produced it | Interpretation |
| --- | --- | --- |
| Kaggle **0.958**, submission57333150 | Exact released R6 multi-branch ensemble, version2 | External reproduction; not a newly trained solo model or original 0.958 gain. [Actual hidden outcome](initiative/audits/exact-R6-entry-20261018-v1/actual-hidden-terminal.json). |
| Kaggle **0.950** | Released S6 weights/inference | External reproduction. |
| Kaggle **0.949** | Released CoAtNet parent | Reference for the corresponding adaptation experiments. |
| Kaggle **0.949** influence versus **0.947** uniform16 | Matched full-encoder,3epoch/1458update adaptation on3887 fitting studies | An observed sampler comparison; influence ties the unchanged parent's displayed0.949. It is not a new project best. |
| Kaggle **0.894** | Our ImageNet-initialized, MRI-trained N3 CoAtNet | Separate own-training lineage; not a controlled causal comparison with the released models above. |

The R6score is an actual completed competition result, superseding the old
README's0.950best and pending-R6 statements. Internal weak-label AUCs below are
separate endpoints, not Kaggle scores or clinical validation. Released models
can have upstream data/calibration exposure; head-level cross-fitting cannot
make their complete encoders independent OOF models.

## Active full-model work

Three distinct experiments are moving: the fixed-final cross-teacher model has
completed all1,458 updates and was submitted once as57422143 (hidden score still
unknown); the matched pointwise/ranking comparison is training on H100975297;
and full source-zero paired-view fitting is RUNNING on Kaggle139742882v1.
Each fits all73Mparameters on3,887 MRI studies. The source-zero run uses an
actual byte-equivalent selected-input path measured3.244× faster for CPU case
loading; no GPU-throughput or accuracy improvement is inferred from that.
The latest actual scores still show no original gain over the0.958 reproduction.
See [current state](docs/current-state.md) for exact owners, private preservation
and remaining work.

## Completed spatial-reader experiment

The new dense spatial reader completed both fixed3epoch/843update arms, with
all six checkpoints privately preserved. It trained68,560reader parameters over
a frozen N3encoder; it was **not end-to-end backbone adaptation**.

On the same1156weak-label development studies:

| Reader | Internal macro AUC |
| --- | --- |
| Aligned spatial paths | 0.9212176612 |
| Marginal null paths | 0.9205166745 |
| Simultaneous unchanged FP32 parent | 0.9205878230 |

Aligned minus null is+0.00070099, conditional95%interval
[-0.00092435,+0.00236799]. Aligned minus parent is+0.00062984, interval
[-0.00429430,+0.00486056]. There were1186undefined macro draws out of4000;
intervals condition on defined draws and this fitted seed/fold. Most lateral-
meniscus improvement was shared with the null. **No established overall or
alignment-specific gain: stop expanding this unchanged version.** No target
cherry-picking or new Kaggle promotion is supported.
[Actual result, limits and decision](initiative/research/path-fixed-final-analysis-recovery-20261018-v2/result.md).

## Active learning experiments

- **Encoder learning rate:** same3887-study/full73Mencoder recipe; encoderLR
  3e-6→3e-5, headLR stays3e-5. Original epoch1 is complete/privateverified;
  epoch2 is accepted, with no final score yet.
- **Finding/family supervision:** full3epoch fits are complete and privately
  preserved. Neither has a final hidden competition score recorded; execution stage is in
  current-state receipts. Training/public-example completion is not hidden acceptance.
- **Paired-logit correction:** production-objective frozen-feature diagnostic
  reduced head-gradient MSE88.48% on16fitting studies/64draws. This used CPUFP32,
  no dropout and both views averaged; it is not full-encoder or accuracy proof.
  Original real-encoder recovery and combined gradient/cost measurement remain
  under their existing finite owners.
- **H100 alternative:** a new CPU-only check confirms retained32-study source
  descriptors and RAM availability. Actual tensors/source/runtime and a bounded
  GPU wrapper remain prerequisites. No H100 measurement has run from this route.

[Encoder learning rate](initiative/research/encoder-lr-comparison-20261018-v1/README.md),
[combined measurement](initiative/research/paired-encoder-gradient-noise-v2-20261018-v1/README.md),
[prospective decision](initiative/research/paired-probe-promotion-20261018-v1/README.md),
[H100 route availability](initiative/research/paired-encoder-H100-memory-probe-20261018-v1/README.md).

The prospective correction decision requires lower complete/encoder gradient
and prediction error, no increased empirical mean-gradient error, acceptable
reference precision and at most10%measured optimizer-step overhead before
considering a separately funded matched full fit. It does not prove unbiasedness
or admit an accuracy gain. A correction-input Dataset creation returned503;
its exact uncertain intent is preserved and subsequent checks are read-only.

## Supervision and MRI review

Accepted training masks/labels and their source provenance stay explicit.
Current report-derived weak targets are not adjudicated MRI truth. A claimed
whole-cohort image-label export has not been established by the current admitted
pipeline. Protected evaluation data remains separate; no reviewer-confidence or
agreement count proves perfect labels.

Daenerys has completed eight sequential fitting-study reviews. Study9's two
original reviews are collected/source-bound to all342slices/ten series; its
requested14additional original views are delivered and reviewing. One target
changed to uncertain after the report; accepted labels remain unchanged.
Original images/reports/reviews/identities/native references stay private.
[Review evidence](initiative/audits/daenerys-multimodal-study-review-v1/ninth-original-reviews-and-requested14-views-20261018.json).

## Research and operational evidence

- [Current state](docs/current-state.md): canonical results and current limits.
- [Research brief](docs/research-brief.md): earlier mechanisms and findings;
  current-state receipts supersede obsolete phase statements.
- [Research-agent handoff](docs/research-agent-handoff.md): evidence and questions.
- [Execution continuation](docs/codex-continuation.md): exact owners and controls.
- [Creative research plan](docs/creative-research-plan.md): standing research approach.
- [Frozen report protocol](protocols/report_weak_v2.md): supervision definitions.
- [Previous README preserved verbatim](docs/README.history-20261018.md): historical
  source, including its obsolete scores/phase states; original links were relative
  to repository root. [Byte provenance](docs/README.history-20261018.provenance.json).

Keep large/private artifacts on authorized cluster/HF storage, with exact source,
weight, license, input, runtime and checkpoint provenance. Preserve negative
results and original failure receipts. Never duplicate an uncertain job, upload
or entry. Software checks and checkpoint preservation establish operability;
only actual evaluation supports a model-performance claim.
