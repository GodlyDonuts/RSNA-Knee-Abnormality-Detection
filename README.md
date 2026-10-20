# RSNA Knee Abnormality Detection

Updated October 20, 2026. The predictive-evidence B/C/C0 engineering runs have
passed; full-cohort input preparation is running, and full predictive fitting
has not started. The completed pointwise and ranking models are privately
preserved. Pointwise has passed two-T4 qualification and its public notebook is
queued. See [current evidence](docs/current-state.md).

**October 20, 2026: best verified Kaggle score is 0.958, from an attributed
external R6 ensemble reproduction. Target 0.97 is not achieved.** Current encoder
adaptation uses a **73M CoAtNet**, with earlier DINOv3 research retained separately.

Use [current measured state](docs/current-state.md) for results and unfinished
work, and [execution continuation](docs/codex-continuation.md) for original live
owners and exact operational bindings. Historical dispatch files are audit
artifacts, not instructions to restart completed or uncertain operations.

## New primary research: predictive evidence

Following the owner's supplied research, we are implementing a matched comparison
between direct classification plus generic reconstruction and finding-conditioned
prediction of withheld MRI descriptors. The source-level masking test passed on
67 original DICOMs: withheld-pixel changes cannot alter context inputs. The three
matched arms have a trainable encoder, complete region priors and a tested
recovery loop. C0 uses C's architecture without the density loss.
**No trained predictive-model result exists yet.** See the
[proposal, decisions and exact limits](initiative/research/predictive-evidence-proposal-20261020-v1/README.md)
and [current implementation/next gates](docs/current-state.md).

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

The pointwise/ranking comparison completed all three epochs per arm on
H100 job975297, with all six checkpoints privately verified. Pointwise final
qualification139818636v1 passed all32 studies on both T4s; its exact public
notebook139819877v1 is queued, with no competition entry or score yet. Ranking's
final model is ready, but its deployment is unadmitted while the remaining
Kaggle GPU budget is reserved for the predictive comparison.

Cross-teacher fixed-final entry57422143 scored **0.949**. Source-zero fixed-final
entry57435296 scored **0.948 without an error**. Both fit all73Mparameters on
3,887 MRI studies; neither improves the external baseline.

The primary next experiment is the matched six-epoch B/C/C0 predictive-evidence
comparison. Its accepted recipe fixes2,916 updates and identical fitting labels,
masks, order and shared initialization. Original raw/context producers
979664/979665 remain healthy; at13:32UTC they had prepared842/840 of3,887 studies.
Training and deployment controllers wait for complete inputs and actual final
checkpoints. No predictive full-model score exists.

Cached-attention diagnostic977100 completed its engineering recovery. It is
retained as research evidence; it has not been promoted to a full production fit.

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

## Earlier learning experiments

- **Encoder learning rate:** same3887-study/full73Mencoder recipe; encoderLR
  3e-6→3e-5, headLR stays3e-5. The final higher-LR adaptation scored0.949,
  without an improvement over the corresponding parent.
- **Finding/family supervision:** full3epoch fits are complete and privately
  preserved. Original audit receipts retain their outcomes and limits.
- **Paired-logit correction:** production-objective frozen-feature diagnostic
  reduced head-gradient MSE88.48% on16fitting studies/64draws. This used CPUFP32,
  no dropout and both views averaged; it is not full-encoder or accuracy proof.
  Subsequent real-encoder work and completed paired adaptations are documented
  in the original audits and current results; this diagnostic is not an active
  dispatch instruction.

[Encoder learning rate](initiative/research/encoder-lr-comparison-20261018-v1/README.md),
[combined measurement](initiative/research/paired-encoder-gradient-noise-v2-20261018-v1/README.md),
[prospective decision](initiative/research/paired-probe-promotion-20261018-v1/README.md),
[H100 route availability](initiative/research/paired-encoder-H100-memory-probe-20261018-v1/README.md).

The original prospective correction decision required lower complete/encoder gradient
and prediction error, no increased empirical mean-gradient error, acceptable
reference precision and at most10%measured optimizer-step overhead before
considering a matched full fit. It does not prove unbiasedness or an accuracy
gain. Historical uncertain mutations remain preserved; dated audit instructions
must not be used to repeat them.

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
