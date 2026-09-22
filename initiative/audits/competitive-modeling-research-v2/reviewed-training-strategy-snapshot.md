# Competitive training strategy

Prepared September 22, 2026 against the accepted `report-supervision-v1` data.
Root synthesized the primary research and the preserved [independent recommendation](../initiative/audits/competitive-modeling-research-v2/memo.md).
This is an experiment plan, not a trained solution or a promised leaderboard rank.

The recommended first competitive model is a pretrained image encoder with
target-specific attention over slices and MRI series. Start with a small model,
measure whether our supervision helps, then spend compute on better spatial detail
and complementary representations. Our likely advantage is the ability to identify
reliable supervision and explain uncertainty. The main risk is that the strict
labels are too sparse and selectively mentioned to teach some abnormalities well.

## The evidence changes the priorities

The [readiness report](data-readiness-report.md) gives exact counts and hashes.
There are 3,403 training studies with any accepted target, containing 21,044
observed cells. Training has only seven MCL, nine Baker's and five fracture
positives, while synovitis has only 15 negatives. Three weak-validation targets
have one positive each. These are not adequate samples for finely tuning twelve
independent specialists. A shared representation, low-capacity heads and careful
validation deserve priority over a very large network.

The final small CPU run proves loading, optimization, masking, resume and prediction
work. It did not establish a useful diagnostic score. No experiment below has yet
been run, and the proposed time/memory budgets are estimates to replace with actual
measurements on the chosen GPU.

Kaggle uses mean AUC across twelve targets. Reports are absent from approximately
1,300 hidden test studies. Inference is offline with a nine-hour limit. Five daily
submissions and two final selections are allowed; the final deadline is October 22.
See the [current primary-source contract](competition-contract-20260922.md). Equal
target weighting makes the rare conditions important, but it does not make their
tiny validation samples trustworthy.

## First model and the main hypothesis

**Hypothesis:** transferable image features plus anatomy- and sequence-aware pooling
will learn more from our limited, partially observed labels than an end-to-end
large 3D model trained from scratch.

Use one shared ConvNeXt-Tiny encoder initialized with the official ImageNet weights.
Train a small pooling head with the encoder frozen, then unfreeze its final stages.
The official implementation provides a roughly 29-million-parameter checkpoint;
pin its exact version, license and weight hash before use. Generic image features
are a starting point, not validated knee-MRI expertise.
[Torchvision ConvNeXt-Tiny](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html)

For each study, sample up to six retained series with plane coverage and the accepted
series preferences. Begin with 16 slice positions per series and a 224–256-pixel
field of view (224 or 252 for a DINOv2 variant, matching its 14-pixel patches). Feed a single slice repeated across RGB channels as the reference
input; test three adjacent slices as a separate 2.5D variant. Preserve geometry
ordering, aspect ratio and the complete knee field. Use normalized slice position,
plane, fluid sensitivity and fat suppression as separate features. Do not treat
fluid sensitivity as equivalent to fat suppression.

Pool slices into series features, then series into twelve target-specific study
features. Start with gated attention and a max-pooling comparator; use missing-series
masks and modest series dropout. Target-specific attention can choose different
evidence for meniscus, cartilage and marrow findings without twelve full encoders.
Do not interpret attention maps as proof of lesion localization. MRNet demonstrates
that pretrained 2D slice features with series pooling can work for knee MRI; the
attention-MIL paper provides a learnable pooling mechanism. Neither establishes
performance for our twelve-target rubric.
[MRNet](https://journals.plos.org/plosmedicine/article?id=10.1371/journal.pmed.1002699),
[attention-based MIL](https://arxiv.org/abs/1802.04712)

Start with the existing per-target normalized masked BCE. Unknown and held targets
remain absent from the supervised loss. Share the encoder across tasks and keep the
rare-target heads small. Use AdamW, conservative weight decay, a short head warm-up,
then a lower encoder learning rate. Initial search: head rates 1e-3/3e-4, encoder
rates 1e-5/3e-5, at most 10–20 epochs, with gradient accumulation to an effective
batch of eight studies. These are proposed search points, not tuned defaults.

Compare uniform study sampling against group-balanced sampling first, then modest
rare-class sampling with exposure
caps. Record how often each of the five fracture positives is reused. Avoid enormous
inverse-frequency weights that let one ambiguous example dominate. Test a square-root class weight capped at five, normalized within target, only after the unweighted baseline; do not simultaneously
change sampling, loss, encoder and augmentation. Synovitis needs negative evidence,
not another generic positive oversampler. Masked cells never become negatives.

## Alternatives, ranked by expected information gained

| Approach | Why test it | Main risk / decision |
| --- | --- | --- |
| Frozen DINOv2-S/14 features + the same pooling head | Cheap comparison of self-supervised versus supervised initialization; reusable feature cache | Natural-image transfer may miss MRI contrast; compare matched slices, splits and heads |
| Fine-tuned ConvNeXt 2.5D | Adds local through-plane continuity without isotropic 3D resampling | Adjacent channels differ from RGB pretraining; geometry gaps and variable spacing matter |
| DINOv2-B or DINOv3-S/B | Stronger dense features, potentially useful for cartilage and small structures | Extra memory/runtime; DINOv3 has distinct access and redistribution terms; verify exact license first |
| Train-fold MRI self-supervision | Uses unlabeled training images without inventing diagnoses | Must exclude evaluation images; benefit on only a few thousand studies is uncertain |
| Image/report contrastive auxiliary training | Uses information beyond twelve discrete labels while inference remains image-only | Language/style shortcuts and correspondence errors; only verified training pairs, explicit auxiliary task |
| Low-resolution 3D series encoder | Tests whether spatial continuity adds complementary evidence | Thick anisotropic slices, memory, resampling loss and pretraining mismatch |
| Native-resolution anatomy crops | May reveal fine cartilage defects or fracture lines hidden at 256 pixels | Current cache cannot recover detail by upsampling; requires a new native-source transform and validated crops |

DINOv2's official repository supplies several sizes and Apache-2.0 weights.
DINOv3 offers newer ViT and ConvNeXt backbones under its own license; it is an
optional candidate, not automatically a better knee model. The initial plan can
run without gated weights.
[DINOv2 paper](https://arxiv.org/abs/2304.07193),
[official models and license](https://github.com/facebookresearch/dinov2),
[DINOv3 models](https://github.com/facebookresearch/dinov3),
[DINOv3 license](https://github.com/facebookresearch/dinov3/blob/main/LICENSE.md)

Masked reconstruction is a plausible MRI pretraining objective, while Swin UNETR
demonstrates 3D medical pretraining principally on CT segmentation. Those results
support experiments, not an assumption that CT weights solve knee classification.
ConVIRT motivates image/report representation learning; its findings are from
other medical-image tasks. If used here, remove administrative text, balance
languages, preserve multilingual clinical meaning and prevent near-identical reports
from acting as false contrastive negatives. Exclude mismatched report/image pairs.
[MAE](https://arxiv.org/abs/2111.06377),
[Swin UNETR](https://arxiv.org/abs/2111.14791),
[ConVIRT](https://arxiv.org/abs/2010.00747)

Do not start with automatic pseudo-labeling of every masked target. Confidence
from five fracture positives is not reliable evidence. A later teacher/student
experiment may use training-fold-only out-of-fold predictions, agreement across
models and conservative confidence limits. Keep its supervision in a separate
versioned experiment policy. Report holds and explicit negatives remain authoritative;
never overwrite the accepted release. FixMatch motivates consistency training, but
its standard benchmark successes do not validate our rare multilabel setting.
[FixMatch](https://arxiv.org/abs/2001.07685)

MRNet is a possible external pretraining resource: it contains 1,370 knee exams,
but its ACL/meniscus/abnormality labels are not our full severity rubric. Audit
access, terms and overlap before including it. OAI, fastMRI and other knee datasets
are candidates for a separate access audit, not approved dependencies in this plan.
KneeCoT is excluded under the current host ruling. No external data was downloaded
or purchased for this investigation.
[MRNet dataset](https://stanfordmlgroup.github.io/competitions/mrnet/),
[host external-data clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965),
[KneeCoT ruling](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109)

## Evaluation that can reject our own theory

Freeze the accepted 441-study weak-validation split for comparable development
results. Always report its per-target class counts and AUCs alongside macro AUC.
It measures agreement with report-derived supervision, not independent image truth.
Do not select a model because it ranks a single MCL or fracture example correctly.

Within the 3,904-study training pool, create three deterministic group-aware inner
folds, stratified on observed targets when feasible, before any model runs. Existing
leakage groups stay indivisible; official gold and its quarantine stay excluded.
Publish the fold manifest and validate it as a new experiment artifact without
changing the accepted base release. Evaluate pooled out-of-fold predictions, show
fold results, and use paired group bootstrap intervals for model differences.
Single-class resamples are undefined and must be counted, not converted to 0.5.
With only five fracture-positive groups, even these intervals remain fragile.

Pre-register a small candidate list and two development seeds. A change advances
when it improves pooled out-of-fold macro AUC across the seeds, does not cause
a consistent target collapse, and fits the inference budget. A positive paired
95% interval is strong evidence; if intervals overlap zero, record an inconclusive
result and prefer the simpler model unless its prediction diversity improves a
predefined ensemble. Avoid claiming a specific 0.001 improvement is significant.

The 58 official studies were used in legacy calibration. Keep them out of training,
early stopping, loss-weight tuning, thresholds, architecture and ensemble selection.
Freeze a small shortlist first, then use at most one predeclared diagnostic comparison
to identify catastrophic failures, disclosing the historical exposure. Do not call
this an unbiased estimate. The public leaderboard is another noisy, repeatedly
observed development signal; reserve the final choice before the deadline.

Our cleaning hypothesis requires a controlled comparison. First compare the full
accepted supervision with a reduced-supervision subset that excludes all corrected cells,
using identical image inputs, masks on evaluation, training steps and seeds.
Then compare equal-observed-cell subsamples so a change is not merely due to label
volume. Report per-target coverage and sensitivity to removing rare examples.
These measure the value of accepted corrections/reuse. They do not by themselves
prove every semantic hold was beneficial. An unfiltered historical-label comparison
would require a separately reviewed research policy and must never load legacy labels
through or bypass the production release gate. No held or uncertain claim is silently
reintroduced for an ablation.

If reliable model selection remains impossible, a small separately budgeted, blinded
expert review of development images may have more value than another large model.
That would need its own consent/access arrangements and label protocol; it is not
represented as completed, free, or already authorized spending.

## Centers and domain shift

The dataset is international, but report language is not an institution identifier.
Our 19 report-style proxies, twelve language groups and 51 scanner cohorts are
useful for error analysis, not named-center ground truth. Compare coverage,
observability and performance by proxy where both classes exist; report unsupported
cells as unavailable. A report-style holdout tests shift in labeling conventions,
while scanner/protocol holdouts test imaging shift; they answer different questions.

Do not use report language or inferred center as a prediction input when reports
are unavailable at inference. Start with the official sequence attributes. A
metadata-only control can expose site/sequence shortcuts; assess gains from scanner
features under held-out cohort stress tests before allowing them. Keep small cohorts
pooled for training rather than fitting a separate network per guessed institution.

## Ordered experiments and stopping decisions

| Order | Experiment | Main comparison | Advance only if |
| --- | --- | --- | --- |
| 0 | GPU/loader/offline packaging pilot | 100 representative training studies, fixed data release, one epoch; timed native-DICOM inference | Finite gradients/predictions, reproducible resume tolerance, measured memory and realistic throughput |
| 1 | Frozen ConvNeXt head and frozen DINOv2 head | Same input schedule and pooling | At least one exceeds a constant-score baseline on sufficiently supported targets; no identity leakage |
| 2 | Fine-tune the better small encoder | Frozen versus final-stage unfreezing | Repeatable group-CV benefit per additional GPU-hour |
| 3 | 2D versus adjacent-slice 2.5D; max versus target attention | Change one factor at a time | Benefit survives matched seeds and cohort stress checks |
| 4 | Accepted correction/reuse and class-balancing ablations | Fixed architecture, equalized exposure where possible | Improvement is not driven solely by one rare validation example |
| 5 | One representation experiment | MRI self-supervision or verified image/report auxiliary objective | Improves the frozen downstream protocol enough to repay added compute |
| 6 | One spatial-detail/diversity experiment | Native crops or a small 3D model | Complementary errors improve held-out ensemble predictions |
| 7 | Final retrains and compact ensemble | Preselected models/seeds; no gold fitting | End-to-end offline test rehearsal fits the time and memory limits |

Stop expanding the search if the first four steps consume the compute cap without
repeatable improvement. Diagnose supervision, resolution, input selection and domain
shift before scaling parameters. A rare target cannot be rescued simply by repeating
its few positive examples for hundreds of epochs.

## Compute, storage and delivery budget

The current 8-GB Apple M1 is suitable for checks and small diagnostics. Plan actual
competitive training on a 16–24-GB GPU, with 24 GB preferred for fine-tuning.
Use mixed precision, one or two studies per microbatch, gradient accumulation and
activation checkpointing. Merely chunking slices does not release all retained
training activations; measure peak allocation with the full study graph.

The full existing cache is 82.6 GB plus metadata. A GPU workspace needs roughly
120–160 GB for that cache, code, weights and checkpoints. A raw-source rebuild needs
substantially more: the 569.8-GB original collection plus cache, temporary data and
a 20-GiB free floor. Do not duplicate it on this nearly full laptop. Frozen float16
384-dimensional features for all 819,635 slices require about 0.63 GB before
indices; patch-token caches are much larger. New 384-pixel inputs require native
reprocessing to add detail, not enlargement of the existing 256-pixel cache.

These are planning bands, not benchmarks:

| Path | Proposed scope | GPU-hour allocation | Illustrative compute cost |
| --- | --- | ---: | ---: |
| Free Kaggle | Frozen features, one small fine-tune, few controlled comparisons | Fit within the account's actual available quota; do not assume a guaranteed weekly amount | $0 direct GPU rental, subject to availability |
| Measured starter | Pilot 4h, feature extraction 8h, fine-tuning/ablations 48h, inference/retries 12h | 72h on a 24-GB GPU | About $53 at $0.74/h, plus storage |
| Broader search | Group folds/seeds, one auxiliary pretrain, one complementary architecture, final ensemble | 250–400h on the same assumed device | About $185–296 at $0.74/h, plus storage |
| Larger-memory option | Only if the pilot shows memory is the limiting factor | 72h example on an 80-GB A100 | About $114 at $1.59/h, plus storage |

The quoted example rates were visible on Runpod's pricing page on September 22:
RTX 4090 24 GB at $0.74/hour and A100 PCIe 80 GB at $1.59/hour. Standard network
storage below 1 TB was $0.07/GB/month, so 150 GB is about $10.50/month. These are
illustrations, not guaranteed availability or an endorsed paid order; taxes and
other provider charges can differ. No compute has been purchased.
[Provider pricing](https://www.runpod.io/pricing)

Replace estimates after the pilot using measured seconds per study, optimizer
step, epoch, decode and complete inference. For example, 20 epochs over 3,403
supervised studies at 0.5 measured seconds/study is about 9.5 hours before validation;
at 2 seconds/study it is about 37.8 hours. This sensitivity is why we should benchmark
before committing to many folds. Account for wall-clock session caps and persist
checkpoints between free sessions.

For every run, save code commit, release and series hashes, exact pretrained weight
hash, fold/seed, selected-study identity, normalization, augmentation, loss/sampling
parameters, optimizer/RNG state, package versions, hardware, peak memory, timing,
per-target metrics and out-of-fold predictions. Keep private data and credentials
out of model artifacts intended for public release. Download and license-audit all
dependencies before offline submission; never rely on a live weight download.

## Ensemble and submission decisions

Begin with an equal-weight blend of two or three models with distinct evidence:
for example a fine-tuned ConvNeXt 2.5D model and a DINO feature model, optionally a
small 3D or crop model if it adds out-of-fold benefit. Test a coarse blend grid on
group-CV predictions; shrink toward equal weights. Avoid twelve freely fitted
target-specific blending weights on the tiny rare-class validation sets.

Per-target positive-temperature scaling alone preserves ranking and therefore
does not improve that model's AUC. It may change a probability blend, so compare
simple probability and rank blends on fixed out-of-fold predictions before adding
calibration. Threshold optimization is irrelevant to this scoring metric. Add
only anatomically valid, measured TTA; avoid transformations that swap medial and
lateral semantics or erase subtle findings.

Rehearse from raw DICOMs with internet disabled, including codec imports, model
loading, decoding, preprocessing and CSV checks. Do not extrapolate the three sample
studies alone: use a larger representative training subset and high-slice-count
studies to estimate the tail. For 1,300 studies, nine hours permits roughly 24.9
seconds/study including overhead. Aim for six hours (about 16.6 seconds/study) to
leave headroom, and measure the final ensemble as a whole. Define missing-series
and decoding-failure handling before submission; failures must be surfaced in logs.

Use a predeclared shortlist and an experiment log instead of exhausting five daily
submissions. Reserve one final slot for the strongest robust ensemble and the other
for a simpler independently competitive model only if validation supports it.
Freeze the main solution several days before October 22 and use the remaining time
for offline runtime and reproducibility checks. No high Kaggle score has yet been
measured for this project.
