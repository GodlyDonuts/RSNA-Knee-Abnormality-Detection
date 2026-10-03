# Owner-supplied research program: critical review and execution decision

October 3, 2026. [The original supplied text](original.txt) is preserved verbatim
with [byte provenance](provenance.json). Its title and confidence are the author's;
this project does not promise first place. This review separates primary-source
support, our measured evidence, and proposed experiments. Start with the
[current research brief](../../../docs/research-brief.md) for original results.

## What changes now

Supervision is the first experimental priority. Seven usable MCL positives, five
fracture positives, nine Baker's positives and fifteen synovitis negatives across
the eligible development cohort are a measured learning and evaluation bottleneck.
A strong internal score on these selected labels can coexist with weak transfer
to the competition. This is a plausible explanation, not an established sole cause
of the public gap. More rank/resolution has not established a repeatable gain.

The reviewed Effusion proposal is the smallest ready intervention: restore31
specific source-supported fitting cells in a separate experimental arm, going
from16 to47 fitting positives. It tests loss of useful report supervision while
keeping the encoder, views, targets, membership and evaluation fixed. It cannot
by itself account for a0.132 macro-AUC gap or validate all discarded labels.
The trainer now admits this proposal explicitly in both control/candidate arms,
binds it to checkpoints/resume and applies it before diagnostic subsetting. See
[the experiment](../../experiments/competitive-v1/effusion-review-contrast-v1/README.md).
No GPU fitting or improved accuracy is claimed by this integration.

Next build fitting-only broad concepts/partial orders from explicit source
findings, alongside strict supervision. Treat it as a different experiment.
Maintain the distinction between pathology being present, severity exceeding the
competition threshold, certainty and omission. Auxiliary report targets are not
newly certified MRI labels. Start with supported target-specific comparisons;
do not turn every omitted finding into a negative or invent a universal severity
scale across diseases. Keep missingness separate from confidence. Nonrandom
report omission makes off-the-shelf positive-unlabeled assumptions questionable.

Physical crop and genuine neighboring slices are the next independent input
contrast. First establish DICOM spacing/coordinate handling, true within-series
neighbors and fitting/inference parity. Existing neighbor3 code uses slice
embeddings; it is not the same intervention as putting adjacent raw slices into
three image channels. A fixed plane-aware64-image stack also changes coverage.
Hold either input or supervision fixed when comparing. Do not sweep130/150/170mm
and select a public best, or change crop/coverage/encoder/labels in one trial.

Anatomy-conditioned evidence retrieval is a later solo-model mechanism, not an
immediate large architecture build. Soft cross-plane correspondence must respect
physical geometry, acquisition differences and uncertainty; planes cannot simply
be treated as registered volumes. Existing signed patch-witness implementation
is retained as an independent alternative, but its GPU fitting is no longer the
first priority. Earlier frozen patch-attention regression remains relevant.

## Primary-source checks, October3

| Report claim | Checked source | What can be concluded |
| --- | --- | --- |
| Twelve-target macro AUC and strict imaging criteria | [Official overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview), [host label description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343) | The metric and target definitions are supported; the host describes independent specialist readings with adjudication. |
| A released strong single CoAtNet uses dense supervision and different inputs | [Author's model card](https://www.kaggle.com/datasets/dreaddevelopment/raptor-knee-widedense) | The author reports0.924 public and0.9167 on58 held-out labeled studies, with soft report targets, a plane-specific64-image stack,140mm spacing-based crop and adjacent-slice channels. These are author-reported results, not our reproduction. Joint recipe differences cannot identify a causal winner. |
| OrthoFoundation has a released knee-specialized DINO-L model | [Authors' repository](https://github.com/ytrsk/OrthoFoundation-MSK-5M) | A checkpoint is listed. The README distinguishes a1.25M-image manuscript cohort from an expanded5M-image released checkpoint initialized from DINOv3-L. No explicit license was shown in the inspected root/README; compatible weights, source rights and overlap remain unadmitted. |
| KneePreM pretrained a3D model on OAI | [Authors' preprint](https://arxiv.org/abs/2609.31461) | The abstract supports19,011 series/4,791 participants and a3D U-Net masked-autoencoder. It is not a drop-in DINO backbone or an RSNA score. |
| OrthoDiffusion has anatomy/diagnostic tasks | [Authors' preprint](https://arxiv.org/abs/2602.20752), [repository](https://github.com/lt-0123/OrthoDiffusion) | These are relevant primary leads, not proof of transferable anatomy labels or compatible released weights. No acquisition/training admission follows from the title. |

The separate host discussion733826 rendered without readable comment text in this
check. The exact alleged report-versus-image-authority answer was therefore not
independently verified here; the readable host overview supports the stricter
clinical criteria and reader process. Keep those two evidence levels distinct.
The claimed current leaderboard ceiling, anecdotal competitor scores and public/
private split were not independently established in this review. No decision
requires taking those estimates as facts. Original supplied claims remain intact.

## Corrections and boundaries

- The Effusion review establishes report support, not MRI consensus positives.
  Calling it a smoking gun for model failure is stronger than our evidence.
- Accepted masks are selective. They are not useless: strict explicit evidence
  remains the anchor for controlled broader-supervision experiments.
- Report omission cannot imply an ordering below explicit negative or definite
  positive. Indeterminate findings cannot automatically be assigned0.5,0.8 or
  a clinically meaningful probability. Learn/test mappings using fitting evidence;
  never calibrate them against protected evaluation or public scores.
- The58 official evaluation studies and protected confirmation are not a tuning
  set. Historical engineering/earlier evaluation exposure must not be relabeled
  as permission for new selection. Do not construct reviewed gold by reading
  protected rows during this research.
- A0.90 public threshold is aspirational, not an engineering acceptance test or
  prerequisite to informative matched work. The released baseline provides a
  useful reference without proving it is reproducible under our labels and folds.
- Report/text privilege can be training-only. Exact MRI-only inference must
  reject report-dependent deployment. A general knee-text embedding is not a
  substitute for target-specific evidence or source-level validation.
- Keep the owner's solo-model objective. An ensemble section in supplied research
  does not silently authorize a switch to ensemble/public-score optimization.
- External weights and private Stanford adapters require exact provenance,
  compatible rights, source overlap review and a matched extra-compute control.
  The existing official base remains pinned; do not download bulk candidates
  merely because a README reports strong results.

## Execution order and falsification

1. Admit actual-MRI recovery for the now-integrated31-cell paired experiment under
   fresh resources. Fit the same source, initial state, uniform views, update budget
   and outer masks. Effusion is its declared primary target; report all12 and paired
   whole-component uncertainty/undefined draws. A null result narrows this specific
   omission explanation, not the entire dense-supervision hypothesis.
2. Build a separately reviewed fitting-only concept/partial-order admission. Test
   strict BCE versus strict BCE plus auxiliary ranking; do not discard the strict
   anchor or change images simultaneously. Source semantics and gradient scales
   precede a full fit. Unreviewed categorical levels are not clinical annotations.
3. Test physical observation with the same accepted supervision and encoder.
   Capture paired input/parity proofs, protocol/plane failures and compute; require
   held-out performance and a second seed/fold before promoting a promising gain.
4. Develop anatomy-conditioned retrieval or knee-specific initialization only after
   the simpler contrasts establish which information is actually missing.

Kaggle stays paused under the latest owner direction. No new remote job, bulk
transfer, external model admission, source annotation, checkpoint publication or
competition mutation is implied by this document. Existing completed checkpoints
and original scientific sources remain preserved.
