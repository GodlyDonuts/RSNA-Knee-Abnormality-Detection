# Competitive training strategy

Prepared September 22, 2026 against the accepted `report-supervision-v1` data.
Root synthesized the primary research and the preserved [independent recommendation](../initiative/audits/competitive-modeling-research-v2/memo.md).
This is an experiment plan, not a trained solution or a promised leaderboard rank.

**Recommendation:** use DINOv2-B/14 as the reference, challenge it directly with
DINOv3-B/16 and a convolutional encoder, then invest in knee-specific representation
learning, native image detail and a complementary 3D branch. Select the final model
by a frozen evaluation protocol, not by popularity or parameter count. H100 access
makes this comparison practical; it does not establish which family wins.

The owner has access to UCF Newton's shared GPU cluster, with opportunistically
available two-GPU nodes and inter-node InfiniBand, plus substantial CPU capacity.
This supersedes the earlier 24-GB rental-oriented plan. The original plan and its
reviews remain archived as historical evidence. Our current data bottleneck is
independent, reliable supervision for rare targets; our final deployment bottleneck
remains Kaggle's offline nine-hour limit.

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

Use the official DINOv2-B/14 register checkpoint as the reference backbone,
with the register variant fixed before experiments. Compare DINOv3-B/16 and a
ConvNeXt-Tiny/ImageNet control through the same small target-specific pooling head.
The two DINO base models are approximately the same parameter scale, but different
patch sizes mean different token counts: common 224-pixel input controls image
coverage, not FLOPs. Report both matched-data/update and fixed-GPU-hour comparisons.
Start with frozen features, then allow the same limited fine-tuning search for both
DINO candidates; a frozen loss alone does not reject a model's ability to adapt.

Use global embeddings first, followed by a controlled local patch-token pooling
comparison. Global pooling may hide small focal abnormalities. Exclude register and
padding tokens from spatial pooling correctly. Pin the exact weight revision,
preprocessing, license and hash. None of these published natural-image results is a
measured knee-MRI result for our data.
[DINOv2 official models](https://github.com/facebookresearch/dinov2),
[DINOv3 official models](https://github.com/facebookresearch/dinov3),
[ConvNeXt reference](https://docs.pytorch.org/vision/stable/models/generated/torchvision.models.convnext_tiny.html)

For each study, sample up to six retained series with plane coverage and the accepted
series preferences. Begin with 16 slice positions per series and a common 224-pixel
field of view for the controlled encoder screen. Feed a single slice repeated across RGB channels as the reference
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

## The competitors we should actually test

| Candidate | Role and hypothesis | Promotion condition |
| --- | --- | --- |
| DINOv2-B/14 | Strong reproducible reference with sequence-aware target pooling | Remains the reference unless another family wins the declared comparison |
| DINOv3-B/16, then L/16 | Newer dense visual features may transfer better to small lesions | Base wins fairly before a large model gets a major allocation; exact license and redistribution checked |
| ConvNeXt-Tiny, then Base or EfficientNetV2 | Different inductive bias and potentially complementary errors | Independent CV benefit or improvement in a held-out blend, rather than diversity by name |
| Train-fold MRI-adapted DINO | Continued self-supervision on knee images may improve domain transfer | Beats the same initialization without adaptation under an unchanged downstream protocol |
| Knee-specific pretrained encoder | Potential shortcut to domain expertise | Public exact weights, upstream terms, pretraining provenance and overlap checks pass before use |
| Anisotropy-aware 3D series encoder | Preserves through-plane morphology that sparse slice pooling can miss | Same anatomical coverage; controlled random versus pretrained initialization; acceptable inference cost |
| Native-detail / anatomy-aware branch | Finer cartilage, meniscus and marrow evidence than the existing cache permits | A new native-derived transform improves held-out results; enlarging the 256 cache is not extra information |

A competitor's public account describes an EfficientNetV2/DINOv2 2.5D ensemble,
but reports thresholded accuracy on 58 studies rather than a competition AUC or
leaderboard result. It establishes an alternative implementation exists, not that
DINOv2 is optimal or that our pipeline is already better.
[Author's report](https://www.kaszub.ski/work/rsna-kaggle)

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

Two newer knee-specific checkpoints warrant an early eligibility and frozen-feature
check. **OrthoFoundation-L** continues DINOv3-L pretraining on knee radiographs and
MRI, with a public checkpoint in the author repository; its project MIT notice does
not remove upstream DINOv3 obligations. **OrthoDiffusion** offers three orientation
specific 3D checkpoints, with knee-domain pretraining on 15,948 exams. Its source
cohort excludes some surgical/hardware and acute complex-fracture cases, so domain
relevance is not the same as coverage of our competition targets. Public weight
availability is verified; pretraining overlap, exact release terms and competition
eligibility remain unverified. Neither is an approved training dependency yet.
[OrthoFoundation paper](https://arxiv.org/abs/2601.18250),
[author checkpoint](https://github.com/ytrsk/OrthoFoundation),
[OrthoDiffusion paper](https://arxiv.org/abs/2602.20752),
[author code](https://github.com/lt-0123/OrthoDiffusion),
[checkpoint listing](https://huggingface.co/lanstat0123/orthodiffusion/tree/main)

If those checks fail, retain fully reproducible alternatives: continued MRI
self-supervision from the eligible DINO initialization and a small 3D encoder.
A conditional second-line screen can include SigLIP2, BiomedCLIP/MedSigLIP and VoCo,
as described in the [independent hardware-aware research](../initiative/audits/h100-model-strategy-research-v1/memo.md).
Natural-image/text, medical figure/text, and CT pretraining answer different
transfer questions. Do not run every model combination. The independent researcher
favors starting with DINO Large; root chooses the matched Base screen first to
reduce time-to-evidence, then scales both families if the measured gains justify it.
The 6.7B DINOv3 teacher is a later frozen/distillation option, not the starting model.

For MRI adaptation, compare the unchanged initialization against continued
DINO/iBOT-style teacher-student learning and one masked-reconstruction control,
with the same eligible images and a bounded update budget. Use spatially overlapping
same-series crops; arbitrary slices and unregistered sequences are not automatically
matching anatomical views. Monitor collapse and compare downstream ranking through
the unchanged evaluation protocol. Cache and train an adaptation separately for
each fold's training pool, excluding its evaluation images even without labels.
Only expand to cross-sequence consistency when geometry supports correspondence.
[Limited-data knee SSL precedent](https://arxiv.org/abs/2208.13923),
[task-dependent MRI pretraining findings](https://arxiv.org/abs/2604.11679)

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

Keep the accepted split unchanged. Use the 3,904 training studies for development
through three deterministic, group-aware folds. Before model runs, freeze the
membership, masks and positive/negative counts in studies **and leakage components**.
These components are conservative leakage units, not proven patient identities.
The rare training counts were checked: five fracture-positive components, seven
MCL-positive, nine Baker-positive, and fifteen synovitis-negative components.
[Aggregate counts](../initiative/audits/h100-evaluation-group-counts-v1/counts.json)

Choose learning rates, stopping epochs and blending inside each fold's training
portion, or use a fixed schedule declared before its outer evaluation. Never pick
the checkpoint on the fold used to report its performance. Restrict the initial
candidate list and search budgets; give finalists two matched seeds. More seeds
measure training variability and do not create independent patients.

Reserve the 441-study weak-validation set for one predeclared confirmation after
freezing the shortlist, input policy, optimizer schedule and ensemble rule. It
measures report agreement and is not independent image ground truth. Its rare-target
counts remain too small to establish a close winner. If its results cause further
changes, disclose that it became development data; do not continue calling it a
locked confirmation. The earlier diagnostic smoke predictions are disclosed and
must not be used to select the competitive models.

Report all twelve per-target AUCs and their counts. Pooling out-of-fold scores can
create misleading cross-fold rankings when separately trained models have different
score scales. Report within-fold AUCs and a predeclared within-fold aggregate that
weights each fold by its observed positive-negative pair count, then averages the
twelve targets equally. Show pooled OOF macro AUC separately, and investigate
disagreement rather than choosing the favorable summary. Do not use held-out labels
to normalize score scales. If any target is undefined, say the complete twelve-target
macro is unavailable; a supported-target summary must have a different name and a
fixed target set chosen from counts before seeing results.

Use paired leakage-component bootstrap intervals within folds, preserving fold and
seed identities. They measure resampling uncertainty conditional on fitted models,
not all model-search uncertainty. Count undefined resamples; never replace them
with AUC 0.5. Report seed variability separately. Leave-one-positive-component-out
sensitivity identifies fragile rare-target results, not extra confirmation data.
Promote candidates with repeatable supported-target gains, no consistent target
collapse and acceptable inference cost. Inconclusive comparisons favor the simpler
model unless a predeclared held-out ensemble test supports the alternative.

The 58 official studies were used in legacy calibration. Exclude them and their
quarantine group from supervised training, MRI adaptation, stopping, calibration,
architecture and blend selection. After freezing the model, one predeclared failure
diagnostic may be reported with its historical exposure disclosed; it is not an
unbiased selection score. No threshold or blend fitting follows it. The public
leaderboard is also a noisy development signal, not a substitute for this protocol.

All train-fold self-supervision, learned crop selection, normalization statistics,
report contrastive learning and pseudo-label teachers obey the same exclusions.
Unchanged public frozen encoders may extract validation features for inference;
no statistics or weights may be learned jointly across the partitions.

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

If qualified image readers are available, first run a separately versioned labeling
pilot. Allocate review effort between a probability-sampled blinded development
audit, targeted training acquisition and an independent second reader/adjudication.
Prioritize distinct-component fracture/MCL/Baker positives and synovitis negatives;
include normal controls and uncertain boundaries. Keep outcome-enriched acquisition
separate from the random audit, recording how cases were sampled. Train-only model
disagreement can select cases to review but cannot supply the diagnosis. Define
present, absent and indeterminate against the exact competition rubric, preserve
unresolved claims and originals, and measure yield before committing a large review
budget. No paid review or new clinical labels are represented as completed.

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

## Experiment sequence on Newton

These are bounded allocations to replace with pilot measurements, not predicted
training times or a claim on the account's entire GPU allowance.

| Stage | Work and decision | Initial H100 GPU-hour ceiling |
| --- | --- | ---: |
| 0 | GPU, loader, distributed masking, checkpoint and offline-inference pilots; freeze experiment folds | 24 |
| 1 | Matched frozen and short adaptation screen: DINOv2-B, DINOv3-B, ConvNeXt control; identical inputs and head | 96 |
| 2 | Top two candidates, three folds and two seeds, with inner stopping or fixed schedule; correction and exposure controls | 300 |
| 3 | MRI adaptation and one native-detail or 3D branch; conditional knee-pretrained comparator if eligible | 300 |
| 4 | Selected final refits, coarse ensemble or distillation and submission rehearsal | 180 |
| Reserve | Failures, checkpoint recovery and decisive reruns | 100 |
| Total proposal | A gated 1,000 GPU-hour campaign, reducible to a smaller screen if allocation is limited | 1,000 |

Do not launch the entire matrix at once. Freeze a stage's hypotheses and advance
only after its evidence review. If the measured fold cost exceeds a stage's ceiling,
reduce the preregistered candidate count or request a revised allocation; never label
an incomplete comparison as passed. A smaller 200-hour campaign would prioritize the
pilot, a matched two-family screen, one complete finalist comparison and deployment
checks if measured costs permit, dropping speculative pretraining and large-model
searches. A complete comparison takes priority over a broad unfinished matrix.

A run's GPU-hour cost is allocated GPU count multiplied by measured elapsed hours.
Three folds times two seeds train about four full-training-pool equivalents at the
same epoch budget, before validation and inner search. Pilot each actual model;
H100 memory capacity and theoretical FLOPs do not predict MRI loader throughput.
At two continuously allocated GPUs, 1,000 GPU-hours consumes about 20.8 days;
at eight it consumes about 5.2 days, before queueing. These are arithmetic scenarios,
not availability promises. If access arrives late, shorten the campaign to protect
the final deployment window.

Aim to finish pilots and the encoder screen in the first three working days after
access; complete robust finalist comparisons in the following week; run only the
most promising representation/detail branch in the next week. Freeze architecture
and preprocessing by October 15, weights and ensemble by October 18, and use the
remaining days for offline rehearsals and recovery before October 22 at 23:59 UTC.
Check the official timeline again before scheduling. Do not spend the deadline on
an unfinished foundation-model experiment.

## Cluster execution and implementation

UCF documents 29 dual-H100 80-GB PCIe nodes and additional eight-H100 nodes alongside
V100 nodes. The actual allocation may differ. Verify assigned GPU type/memory,
node topology, partitions, walltime, preemption, account balance and storage quota
on access. The advertised 2,000 GPU-hours/month is a shared faculty allocation,
not evidence that all those hours are available to us. No new cloud rental is
needed in this plan.
[Newton resources](https://arcc.ist.ucf.edu/index.php/resources/newton/about-newton),
[UCF scheduler and allocations](https://arcc.ist.ucf.edu/docs/scheduler/)

Prefer independent fold, seed or ablation jobs on available two-GPU nodes. Benchmark
one-GPU jobs against intra-node distributed training with the same effective batch,
optimizer steps and target exposure. Two independent jobs may finish more useful
experiments than one distributed job. Use cross-node InfiniBand only after measuring
a bottleneck and scaling efficiency, principally for larger MRI pretraining. It
does not combine memory automatically or guarantee faster training.

The existing accepted model is a small diagnostic CNN. Pretrained adapters,
physical-position features, accumulation, the competitive pretrained/distributed
path and a 3D encoder
are **implementation work to follow this plan**, not already validated capabilities.
The diagnostic implementation already contains a distributed path; this is not
evidence that the proposed pretrained H100 path has passed acceptance.
Build separate experiment entrypoints consuming the verified release and a hashed
fold overlay. Keep the production release and original supervision immutable.

For masked distributed loss, aggregate per-target observed-loss sums and counts
over the entire effective batch and all ranks, accounting for distributed gradient
averaging. Averaging independently normalized microbatch losses is not equivalent.
Before scaling, compare the same deterministic effective batch with a single-process
reference and test all-masked targets, uneven rank counts, padding and resume.
Use BF16 only after finite-gradient and numerical checks. Validate restart with the
same world size; a different world size is a separately declared restart mode with
a tested sampler and optimizer policy, not an automatic exact-resume guarantee.

Use the CPU cluster for DICOM decoding, geometry checks, deterministic image shards,
feature indices and evaluation. Cache immutable data on node-local scratch where
available, while durable source, checkpoints and logs live in approved shared storage.
Shard by series/study to avoid millions of small file operations. Never run large
preprocessing on login nodes. Each shard and feature cache binds source hashes,
transform, series/slice selection, encoder weights and feature dtype. Fine-tuned
encoders invalidate frozen-feature caches.

The existing cache is 82.6 GB; original DICOMs are 569.8 GB. A native-derived 384
cache has a rough pixel-count planning scale of 186 GB and 448 about 253 GB
if volume and encoding were unchanged; actual padding and compression will differ.
Request roughly 1–2 TB of approved scratch for source plus selected derivatives,
features and checkpoints, then measure quota and real sizes before writing. Store
only selected resolutions, not every experimental derivative. A full 819,635-slice
float16 768-dimensional global-feature cache is about 1.26 GB; dense patch tokens
are hundreds of times larger and should be pooled or generated selectively.

Save atomic checkpoints before the scheduler walltime or preemption signal, with
model/optimizer, sampler and random state, precision, fold identity and input hashes.
Use bounded job arrays and resume logs; avoid untracked retries that silently become
extra seeds. The laptop remains an orchestration and audit machine, not the place
to duplicate native-resolution datasets.

For every run, save code commit, release and series hashes, exact pretrained weight
hash, fold/seed, selected-study identity, normalization, augmentation, loss/sampling
parameters, optimizer/RNG state, package versions, hardware, peak memory, timing,
per-target metrics and out-of-fold predictions. Keep private data and credentials
out of model artifacts intended for public release. Download and license-audit all
dependencies before offline submission; never rely on a live weight download.

## Ensemble, distillation and submission decisions

Begin with an equal-weight blend of two or three models with distinct evidence:
for example the winning DINO model and a convolutional, 3D or native-detail model
that adds held-out benefit. Fit a coarse global blend only inside training folds,
then evaluate it on outer-fold predictions not used to fit its weights; shrink
toward equal weights. Avoid twelve freely fitted
target-specific blending weights on the tiny rare-class validation sets.

Per-target positive-temperature scaling alone preserves ranking and therefore
does not improve that model's AUC. It may change a probability blend, so compare
simple probability and rank blends on fixed out-of-fold predictions before adding
calibration. Threshold optimization is irrelevant to this scoring metric. Add
only anatomically valid, measured TTA; avoid transformations that swap medial and
lateral semantics or erase subtle findings.

A large teacher ensemble can train on H100s while a smaller student is deployed.
Distillation is optional: use only fold-eligible training images and teacher outputs.
Task-logit distillation must obey the final per-target masks and whole-study
supervision exclusions. Any auxiliary policy that changes that scope needs its own
explicit experiment review; it cannot bypass a hold through a soft target. Feature
distillation on technically eligible training images is a separate representation
objective. Never treat teacher confidence as new diagnostic truth. Retain the student only if
it preserves held-out ranking and reduces measured inference cost. No hidden-test
adaptation or report access is assumed.

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
