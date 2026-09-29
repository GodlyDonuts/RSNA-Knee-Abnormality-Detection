# Public teacher models and whole-examination context

September 29, 2026. Static investigation of the owner's three public notebook
links; no downloaded code was executed, weights downloaded, MRI inference run,
labels changed, GPU job launched, or Kaggle submission made.

## What the sources actually establish

Public Kaggle `api/v1/kernels/pull/{owner}/{slug}` returned each notebook's JSON
source and metadata. [sources.json](sources.json) records the returned identities,
versions, dependencies, source byte lengths and SHA256 values. Original responses,
sources and line-numbered cell extracts remain in ignored
`artifacts/public-teacher-notebooks-v1/`; third-party code is not republished in
this milestone. These are captured public sources, not a verification that each
is the exact source of an advertised leaderboard score.

| Notebook | API version | Static findings |
| --- | --- | --- |
| [pjmathematician/rsna-knee-d4-lite](https://www.kaggle.com/code/pjmathematician/rsna-knee-d4-lite) | 1 | Multi-branch ensemble, three-slice DINO windows, per-target series attention and rank fusion |
| [maverickss26/rsna-knee-restructured-version-3](https://www.kaggle.com/code/maverickss26/rsna-knee-restructured-version-3) | 1 | Same 14 named Dataset dependencies and DINO prefix hash; additional presentation/runtime changes |
| [amanatar/rsna-knee-abnormality-detection](https://www.kaggle.com/code/amanatar/rsna-knee-abnormality-detection) | 6 | Same common dependencies and prefix hash; explicitly attaches the first notebook as a kernel dependency |

All three attach DINOv2-small, the same two training notebooks, and 14 common
named Datasets. One Dataset is an OpenCV wheel, so this is **not** a count of 14
independent models. Static code shares a pinned six-block DINO prefix SHA256
`0a55b893bde971c864ea5aeea443f075b17c084d13aa36f30bd1c7863f655cf9`.
This is strong evidence of shared ancestry, not a full weight download or proof
that every branch executes or all final predictions are identical. Distinct
notebook names do not constitute independent label votes.

In the first source's extracted cells:

- Lines 638–675 configure three image channels, six series slots, image sizes
  224/336, and a backbone-tail adaptation setting of six blocks.
- Lines 1381–1424 reshape each series' three-slice window into a three-channel
  DINOv2 input, combine CLS/patch summaries, and construct a per-target slot head.
  The constructor supports unfreezing the final six blocks. Exact optimization
  history still requires each checkpoint's training manifest.
- Lines 1975–2050 enumerate overlapping windows over the selected slice cache,
  infer each window, and combine window probabilities/logits by target. Adjacent
  cached slices need not be adjacent original DICOM slices: cache selection matters.
- The ensemble combines member ranks; its later `_ov_rank` and final output
  likewise rank predictions within each target column. With three distinct values,
  percentile ranks are 1/3, 2/3 and 1. Those are cohort-relative ranks, not disease
  probabilities. A monotone rank transform preserves an individual target's AUC
  absent tie/precision changes; combining ranks across models is a separate step.

The second source's returned metadata enables internet, despite the competition's
offline requirement. We have not bound that snapshot to a particular scored
submission. Notebook descriptions' score claims are not live score verification.
Specific training-study membership, label ancestry, validation exposure,
calibration and licenses of attached weights remain unresolved for teacher use.

## Teacher protocol

Use public models to identify useful disagreements and potentially distill visual
knowledge. They are fallible observers, not replacement ground truth. A 0.94 mean
ROC AUC is neither 94% diagnostic accuracy nor a per-row reliability estimate.

1. **Admit by ancestry and eligibility.** Inspect licenses, competition eligibility,
   training UID/leakage-component membership, label parents and gold/public-LB
   tuning. Start with one representative per distinct checkpoint/training family,
   rather than counting three copies of the same ensemble. Architectural diversity
   is helpful but does not establish independent training evidence.
2. **Control contamination.** For auditing training rows, collect genuinely
   out-of-training predictions where possible. An ensemble of all five folds is
   usually in-sample for four members on a training row. Its confidence is not
   independent evidence. Unknown-membership teachers may propose a review queue,
   but cannot automatically relabel or support claims of clean held-out accuracy.
3. **Respect the student's outer boundary.** Every teacher, calibration fit and
   distillation target for a student outer fold must exclude that outer fold's
   leakage components, plus confirmation and official evaluation cases. Produce
   inner out-of-fold training predictions entirely inside that outer training
   pool. A globally generated OOF table is not sufficient: teachers for other rows
   may have trained on the student's outer fold.
4. **Retain raw evidence.** Save per-family pre-rank logits/probabilities, exact
   checkpoint/input hashes, missing-series and fallback counts, and disagreement
   under valid image views. Raw sigmoid outputs also need train-only calibration
   before interpreting confidence. Preserve final ranks separately; do not pass
   percentile 1.0 to BCE as a certain positive or reconstruct confidence from a
   three-study CSV. Rank-only sources could support a separate pairwise ranking
   loss, after the same membership checks.
5. **Review per study and target.** Compare independent-family predictions with
   source-report evidence and the frozen target definitions. Prioritize explicit
   conflicts and rare targets; sample agreement cases too, since teachers can
   share errors. Preserve original labels and masks. Report review cannot settle
   every image/report disagreement; retain uncertainty without qualified image
   adjudication. No model vote converts an unmentioned finding into a negative.
6. **Distill separately.** If membership and provenance pass, test a bounded
   teacher loss alongside observed-label loss. Any pseudo-supervision on masked
   cells is an explicitly versioned experimental overlay, not a change to the
   accepted masks. Compare observed-only versus added-teacher supervision on
   unchanged outer labels, with per-target support and paired uncertainty.

Freeze teacher selection and mixing on inner validation, not tiny public-score
differences. The outer folds are already development data; the reserved 441-study
partition remains confirmation-only under the existing protocol. No teacher in
this audit is yet admitted for distillation.

## Our model already sees multiple slices

The production configuration selects up to 16 slices from each of six series.
`CompetitiveStudyNet.forward` in `rsna/competitive/model.py` encodes valid slices
independently, adds physical positions, then uses target-specific attention across
slices and series. The global-CLS baseline is not a single-slice classifier.
Its limitations include sparse coverage and compressing each slice before
explicit neighboring-slice interaction. Positional features do not by themselves
provide the rich local depth context proposed below.

Proposed comparison after the isolated A32 adapter experiment:

- Keep broad examination coverage and native-derived detail controls separate.
- Cheap context control: three physically ordered neighboring slices as channels,
  then pool windows across each series. Three selected distant planes are not
  equivalent to neighboring context. Preserve spacing and missingness metadata.
- Stronger candidate: adapted DINO patch features → compressed regional tokens →
  a residual local depth-attention/convolution module → target-specific slice
  pooling → cross-series attention → twelve examination outputs. Initialize the
  new residual path to preserve the reference and measure its actual contribution.
- Within-series depth uses physical position/spacing. Sagittal, coronal and axial
  sequences are complementary views, not adjacent frames. Avoid pixelwise fusion
  without registration, and never assign a study-positive label to every slice.
- At inference process all eligible slices in bounded chunks or explicitly
  benchmark a coverage budget. Chunking does not require loading the entire MRI
  into GPU memory. Preserve deterministic broad coverage alongside selected detail.

[MRNet](https://stanfordmlgroup.github.io/projects/mrnet/) is a primary precedent
for a 2D slice encoder plus examination-level aggregation.
[TimeSformer](https://arxiv.org/abs/2102.05095) motivates separating in-plane and
cross-frame interactions; MRI depth is a proposed adaptation, not a claim that
its video results establish MRI gains. The
[official DINOv3 card](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
documents the image/patch feature extractor. Teacher-to-student transfer follows
the general [distillation approach](https://arxiv.org/abs/1503.02531), subject to
the contamination and calibration restrictions above.

The [adapter and supervision plan](../../../docs/dinov3-adapter-and-supervision-plan.md)
incorporates these changes. Implementation, clean teacher predictions, measured
depth gains and any new training result remain future work.
