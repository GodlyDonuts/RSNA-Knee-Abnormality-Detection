# Source-confidence anchoring: prospective contrast

The report-label creator defines p=.5 as “not addressed” and recommends
`c=2*abs(p-.5)`. Our accepted sequence head gives every observed source cell
the same source influence. This is a testable supervision difference, not a
confirmed bug or a promised accuracy improvement. It differs from the completed
positive-class-weighting CNN experiment. [Source and exact-byte lineage](../../../../docs/research-watch/additional/rsna-steven-v6-provenance-addendum-2026-10-14.md).

Freeze the original global-feature graph, data, seed, initialization and compute:
3,887 fitting studies, 402,956 head parameters, two views, 16 studies/update,
10 epochs/2,430 updates; CPU FP32 AdamW3e-4/WD.01, one-epoch warmup/cosine,
clip1, bounded±.5-logit correction. The licensed MRI parent stays frozen.
No new labels/masks, raw-image inference, protected values or CNN fitting.

Control remains `t=.25*y+.75*teacher` on observed cells. Candidate uses
`alpha=.25*c; t=alpha*y+(1-alpha)*teacher`; missing cells use detached teacher
alone. Average BCE over all cells exactly as before. There is **no confidence-sum
normalization**: changing source-gradient scale and returning its unused influence
to the teacher is the declared intervention. Binary source endpoints reduce to
the old objective; p=.5 returns entirely to teacher. Fractional confidence is an
author convention, not calibrated clinical certainty; suppressing it could discard
useful signal or amplify extraction bias. Retain this negative possibility.

Before full fitting, require seven fresh actual32-MRI CPU processes: full/stop1/
resume for both arms plus the frozen original engine. Require complete model,
AdamW/LR/permutation/counters/all RNG/history and prediction-byte equality on
recovery, plus numerical equality of uniform with the original. Keep both new
versions distinct; existing accepted notebooks/controllers stay immutable.
The actual full control must additionally match the original complete3,887-study
fit numerically before interpreting any candidate comparison.

Preserve all immutable epochs privately once with exact source, base, permission,
runtime and input provenance. Independently fund at most256MiB of checkpoints/
single-file publication verification plus32MiB controls, on top of every prior
liability/reserve; reuse existing feature bytes without copying. No deletion credit.

The externally trained parent saw our local outer/Gold cohort upstream. Therefore
no independent internal OOF/clinical accuracy is claimed. A new head is not scored
until its own actual Kaggle result; software recovery or training loss cannot
promote it. Existing next-window entry owners are unchanged. No diagnostic model
or public three-row CSV may be submitted.

Status update20:23UTC: actual CPU938522 completed both full10-epoch fits and
all20immutable epochs are privately full-download/hash verified. Full uniform
numerical state/predictions equal the original control. No new score.
[Actual results and negative source-semantics finding](../../../audits/source-confidence-context-20261016-v1/README.md).
