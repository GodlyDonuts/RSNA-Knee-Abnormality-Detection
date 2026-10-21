# RSNA Knee Abnormality Detection

Updated **October 21, 2026, 18:04 UTC**. Newton's new C100 predictive-evidence
model completed six epochs on all3,887 fitting studies and is submitted as
**57491555**; its hidden score is still unreported. The independent Colab A100
model is in epoch6 at2,520/2,916updates. All six Newton checkpoints are privately
preserved; Colab has five local immutable epochs, with paced private preservation
queued after fitting. [Current evidence](docs/current-state.md).

**Best verified Kaggle score remains0.958**, from an attributed external R6
ensemble reproduction; own released-parent adaptations remain0.949. Target0.97
is not achieved. Current predictive training uses a **73M CoAtNet**; earlier
DINOv3 research is retained separately. No predictive accuracy gain is established.

Use [current measured state](docs/current-state.md) for results and unfinished
work, and [execution continuation](docs/codex-continuation.md) for original live
owners and exact operational bindings. Historical dispatch files are audit
artifacts, not instructions to restart completed or uncertain operations.

## New primary research: predictive evidence

The matched B/C/C0 predictive-evidence arms each completed six epochs and all
18checkpoints are privately verified. B and C failed in hidden execution with
no score; C0 has now also failed with the same generic hidden exception. C100 increases the density objective weight
after fitting-only gradients exposed dilution; its completed Newton final is
now submitted through a distinct faster, more resilient serving path.
Colab is completing an independently seeded C100 run. These are trained models,
not a claim that the predictive mechanism improves accuracy.

A new complete-cohort audit found that24 encoder centers cover43.37% of slices
per study on average, versus97.32% at96. All-slice fixed descriptors already
existed. The same final model passed a96-center, two-T4 diagnostic in137seconds,
with original24 predictions reproduced exactly. Probability movement is not
accuracy. A distinct96-center public-output candidate is being prepared;
[actual coverage and T4 evidence](initiative/research/predictive-context96-T4-20261021-v1/README.md)
record runtime-tail limitations and the pending outcome decision.

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

Newton C100 original GPU995351 finished6epochs/2,916updates on3,887studies.
Private final Dataset12762386v1, actual qualification140062137v1 and public
notebook140062846v1 bind the exact final model. Code entry57491555 was accepted
once; no hidden score exists yet. [Actual entry and preservation](initiative/research/predictive-C100-final-entry-20261021-v1/README.md).

Independent Colab C100 runs on one A10040GB, seed20261021. Latest visible output
is epoch6/update2,520, with finite gradients and actual encoder updates. It has
259.25of the authorized300resource units remaining; its original training cell
and queued publisher are intact. No subagents or Devin are active.

Pointwise, ranking and source-zero finals each scored0.948. Cross-teacher scored
0.949. They do not improve the external0.958baseline. B/C hidden failures remain
failures/no scores, and public execution is not a hidden result. The current
C100 serving changes address measured preparation costs and numerical/metadata
failure possibilities; the original hidden root cause remains unproven.

See [current state](docs/current-state.md) and [continuation](docs/codex-continuation.md)
for actual owners, exact receipts and remaining work. Do not restart historical
completed or uncertain jobs from this README.

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
