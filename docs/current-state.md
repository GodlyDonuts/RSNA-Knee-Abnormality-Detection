# Current measured state

Updated October 18, 2026. Actual original receipts and live job states
supersede older narrative snapshots. A test, preserved checkpoint or engineering
acceptance is not an accuracy gain.

## Results

- **Best actual Kaggle score: 0.958**, submission **57333150**, exact R6 version 2.
  This is an attributed external reproduction, not an original training gain.
  [Actual hidden outcome](../initiative/audits/exact-R6-entry-20261018-v1/actual-hidden-terminal.json).
- The original 73M CoAtNet influence adaptation scored **0.949**, tying its
  unchanged parent's displayed score. Our own ImageNet-initialized N3 scored
  **0.894**. These are different model lineages, not a controlled architecture test.
- No new accuracy gain is established for the higher encoder learning rate,
  finding/family supervision, dense spatial reader or paired-logit correction.
  Current full-encoder adaptation is **CoAtNet, not DINOv3**.

## Experiments currently moving

| Experiment | Actual state and sole owner | Next evidence |
| --- | --- | --- |
| Dense spatial aligned versus null | Aligned original **955711 COMPLETED 0:0**, 843 updates; null **955712 RUNNING**, 843/843 updates; final evaluation active. CPU owner **48097** follows the same jobs. | All six epochs privately verified; compare both fixed-final predictions on the same 1,156 development studies, all 12 targets, with paired component uncertainty. |
| Encoder learning rate | Same original epoch 2 **139415974 v1**, status unreported; finite fitting owner **66083**. Epoch 1 completed 486 updates and is privately verified. | Finish fixed three epochs with encoder LR 3e-5 versus original 3e-6; head LR stays 3e-5. Final scoring owner **66215** waits that completion. |
| Finding-stratified fitting | All three epochs/1,458 updates and private preservation complete. Exact 32-study two-T4 qualification passed. | Owner **66217** waits actual capacity to execute the original zero-version draft once, then a fresh-allowance competition entry. |
| Family supervision | Full fit, qualification and public run complete. | Owner **43079** waits fresh competition allowance; no hidden entry or score yet. |
| Paired-logit actual encoder recovery | Original **139417155 v1**, actual UI queued/two T4, followed only by owner **66085**. | Actual complete optimizer/RNG/scaler recovery and identical inference. Do not repeat the accepted notebook. |
| Combined encoder noise and update cost | Exact revised source CPU-staged **958642**; sole owner **71891** waits the original recovery. Old unsubmitted owner 68344 retired after canonical reconciliation. | One bounded GPU experiment, zero entries: complete-gradient noise plus reversed-order ordinary/corrected optimizer cost. No GPU result yet. |

The current ordinary and corrected production template use the same 3,887 fitting
studies, original labels/masks, teacher, initialization, 16 sampled windows and
full-94 inference. The full corrected template is prepared, **not a dispatched
production fit**. [Combined probe](../initiative/research/paired-encoder-gradient-noise-v2-20261018-v1/README.md),
[production template](../initiative/research/paired-logit-full-template-20261018-v1/README.md),
[spatial experiment](../initiative/research/path-full-training-20261018-v1/README.md).

## Strongest new mechanism evidence and its limit

On 16 fitting studies and 64 matched draws, paired-logit correction reduced
**head-gradient MSE by 88.48%** using the actual masked source/teacher objective.
That used frozen parent features, the adapted head, CPU FP32, no dropout and both
views averaged. It is **not** full-encoder, production single-view FP16 or accuracy
evidence. The GPU experiment measures the missing conditions before a full fit.
[Actual production-objective decision](../initiative/research/paired-logit-head-gradient-20261018-v1/production-objective-decision.json).

## Data review

Eight sequential fitting-study candidate reviews are collected. The eighth's
14 additional native views changed no target statuses; uncertainty remains.
Study nine is now delivered once and acknowledged: **342 original slices across
10 series on 62 native-resolution sheets**, with a separate exact report.
All 64 hosted assets match original size/SHA and access. Image-only review is
requested before report access; independent underlying tool chronology remains
uncollected. Accepted labels/masks are unchanged. Images, reports, original
reviews and native references remain private.
[Sanitized handoff](../initiative/audits/daenerys-multimodal-study-review-v1/eighth-addendum-and-ninth-delivery-20261018.json).

## Compute and execution

Original healthy owners remain in control; never duplicate their jobs, pushes or
publications. At 14:38 UTC both evc33 H100s were allocated: original spatial null
and another user's job. Its new CPU inventory found 605,061,308,416 free bytes;
that does not create funding for a new checkpoint/runtime obligation. The queued
Kaggle experiment is not an observed training failure. Physical 4/8-GPU nodes
remain prohibited by the owner; useful accessible physical 1/2-GPU nodes are
permitted. Fresh resources and exact original receipts govern new work.

Private source preservation through source f3492774 is fully verified at
HF db9feeb2bf1e76ed0422d5af91afb2227c00aa05. The preceding current-state/review milestone through source62f2674e is also
fully privately verified at HF0d1752b642c309b895617db55cc030de01b67c84.
The newer decision implementation through sourcef6fed9cf is fully privately
verified at HFeff049a068fb3be01195fa7e04844c2cb12dedfc (15small objects). No model or clinical data is
included in a source milestone.

[Execution continuation](codex-continuation.md) carries exact operational handles.
[Preserved preceding state](current-state.history-20261018-1438.md) is historical,
including its obsolete pending states and former owner PIDs.
