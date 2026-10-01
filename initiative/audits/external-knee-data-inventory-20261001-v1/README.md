# External knee MRI inventory, October 1, 2026

The owner asked whether independent data outside RSNA could improve the solo
DINOv3 model. This is a source and supervision audit, not a new training release.
Only public fastMRI+ annotation metadata, README and repository LICENSE were
downloaded; no external MRI images, k-space, reports or model weights were fetched.
Accepted RSNA labels, masks, cohorts and folds remain unchanged. No new Slurm job
or Kaggle submission was made for this investigation.

## Candidate cohorts

| Source | Published scale and unit | Useful supervision | Admission status |
| --- | --- | --- | --- |
| [MRNet](https://stanfordmlgroup.github.io/competitions/mrnet/) | 1,370 examinations; public training split 1,130 and tuning split 120; hidden test split 120 | Report-derived ACL tear, generic meniscal tear and overall abnormality | Official download now routes through Stanford Redivis. Dataset-specific terms, actual accessible splits and overlap remain unverified. |
| [fastMRI](https://fastmri.med.nyu.edu/) with [fastMRI+](https://www.nature.com/articles/s41597-022-01255-z) | NYU advertises more than 1,500 fully sampled knee MRI acquisitions and DICOMs from 10,000 clinical knee MRIs. fastMRI+ reports annotations on 1,172 single-coronal-series examinations from the raw cohort. These counts are not additive. | Spatial pathology boxes, ligament categories, meniscus findings, cartilage loss and effusion | Public annotations inspected and pinned; underlying image access, permitted competition use and overlap not admitted. NYU requires a research/educational use agreement and restricts redistribution. The annotation repository's license does not grant rights to the MRI images. |
| [Osteoarthritis Initiative](https://www.niams.nih.gov/grants-funding/funded-research/osteoarthritis-initiative) | 4,796 participants with repeated longitudinal imaging visits, not 4,796 independent scans at every visit | OA-focused imaging analyses, anatomy and potentially unlabeled adaptation | Official NIH page links to NDA. Exact access route and dataset-specific terms still unverified; general NDA controlled-access policies do not establish the OAI-specific requirements. |
| [SKM-TEA](https://aimi.stanford.edu/datasets/skm-tea-knee-mri) | 155 quantitative knee scans; 86 training, 33 validation and 36 test | Tissue segmentation and 3D boxes for 16 pathology categories | Useful anatomy/localization candidate; qDESS differs from routine RSNA sequences. Dataset-specific terms and overlap remain unverified. |

[Stanford AIMI](https://aimi.stanford.edu/data) describes free noncommercial research
access subject to each dataset's terms. This general statement is not a
dataset-specific acceptance receipt or permission to republish images/derivatives.
Preserve attribution; verify whether trained weights can be distributed before
using a candidate in a winning-model pipeline.

The [September 22 competition contract](../../../docs/competition-contract-20260922.md)
records the organizer's external-data accessibility/cost conditions and exclusion
of KneeCoT. The October 1 web reader returned no readable rules/discussion text;
this investigation does not claim a fresh rendered host ruling. Keep KneeCoT
excluded and recheck host conditions before admitting any external cohort.

## Actual fastMRI+ annotation inspection

Official repository revision: `67ed9a68ee33d3197f93893e8125688470c4a577`.
[Metadata receipt](fastmri-plus-metadata.json) records the byte size and SHA256
of each of four fully downloaded public objects. Raw copies stay in ignored
`artifacts/external-knee-research-20261001-v1/`.

The pinned knee CSV contains 16,167 annotation rows: 16,154 spatial annotation
rows and 13 study-level rows, all of the latter labeled artifact. These are not
16,167 patients or thirteen study-level disease classes. There are 974 distinct
file IDs with at least one annotation, and 22 label strings including artifact.
The reviewed-file list has 1,173 nonempty CSV records beginning `file1000000`,
while the paper reports 1,172 examinations. Treating the first identifier as a
header produces 1,172 data rows. Its interpretation remains unresolved; no
training count is accepted from that file until resolved against image metadata.

Useful positive annotation file counts include ACL high-grade sprain 101,
ACL low-grade sprain 153, MCL high-grade sprain 4, MCL low/moderate-grade sprain
121, meniscus tear 663, effusion 142 and the combined fracture/contusion/dislocation
category 119. These are distinct file IDs within each category, not verified
independent patients, additive categories or accepted RSNA positives.

The [authors' README](https://github.com/microsoft/fastmri-plus/tree/67ed9a68ee33d3197f93893e8125688470c4a577)
warns that annotations from one reader and one view are not exhaustive ground
truth. It also documents an up/down flip when creating annotation DICOMs;
bounding boxes must undergo the matching coordinate transform when loading
reconstructions. The paper describes the limitations of a single coronal series.

## Label policy and experiment sequence

1. Admit a source only after source/version/license/access and permitted weight
   distribution checks. Never use an unofficial mirror to evade source terms.
   A mirror of existing RSNA scans adds storage, not independent examples.
2. Establish deidentified overlap and grouping using only methods permitted by
   each source's terms. Do not identify people or link clinical identities.
   fastMRI's terms restrict linkage with other datasets; do not assume a
   cross-dataset matching workflow is allowed. Resolve this before any overlap
   check or admission requiring such comparison. If independence cannot be
   established within permitted methods, do not claim a leakage-clean cohort.
3. Keep repeated visits, both knees and related acquisitions in the same leakage
   group. Do not mix a source's evaluation partition into auxiliary training by
   default. Retain the frozen RSNA development, confirmation and official
   evaluation boundaries, including for unlabeled adaptation.
4. Freeze a separate source-label mapping. Generic meniscus tear does not mean
   both medial and lateral meniscus are positive. A combined fracture/contusion/
   dislocation finding cannot supervise either fracture or contusion separately.
   Cartilage loss is not automatically compartment-specific OA, periarticular
   cysts are not automatically Baker's cysts, and unannotated targets remain
   masked. Align ACL/MCL severity with the organizer's actual rubric before
   direct target supervision.
5. First candidate, conditional on admission: train the same solo DINOv3 encoder
   adapters with a separate auxiliary head for compatible external findings,
   or with a bounded MRI adaptation objective when no compatible labels exist.
   Preserve the original base weights. Spatial supervision can train a separate
   auxiliary localization loss without equating attention with diagnosis.
   Remove auxiliary heads at MRI-only RSNA inference.
6. Fine-tune on RSNA with the same input recipe, seed, split and fixed training
   schedule as its control. Report per-target ranking and leakage-component
   uncertainty, and the added compute/update budget. Compare against a
   compute-matched RSNA-only continuation so improvement is not attributed to
   external data solely because more optimizer updates were taken.
7. Promote only after measured held-out benefit. Larger quantities or confident
   report extraction alone do not establish better MRI labels or a higher score.

Priority: MRNet for direct multisequence task relevance; fastMRI+ for spatial
supervision if image access/use is compatible; OAI for a larger OA/domain-adaptation
experiment. SKM-TEA is a smaller localization resource. None is currently an
approved training dependency. This keeps the ongoing native384 cache assembly
and rank64 score reconciliation unchanged.

## Fresh KneeCoT ruling after the owner's link

The owner supplied the KneeCoT Hugging Face page later on October 1. Its public
card advertises 25,000 volumes; that is an unverified author count, not admitted
new patients. The rendered Kaggle discussion was then read directly in Chrome.
The competition host's exclusion remained explicit, with no reversal in the two
visible comments. [The small receipt](kneecot-ruling-recheck.json) records the
source, host badge, brief exact ruling and read-only scope. No gated files or
access agreement were accepted. Exclude auxiliary training and derived weights
as well as direct training. The [creative research plan](../../../docs/creative-research-plan.md)
describes conditional alternatives and experiments that separate more compute,
additional anatomy exposure and compatible disease supervision.

## Owner-authorized signed-in MRNet review

The owner supplied the Stanford Redivis link and authorized using the signed-in
session. Data access and both requirements were visibly complete; the existing
research agreement was approved. Reading it did not accept or resubmit anything.
It limits use to personal noncommercial research, restricts redistribution and
derivative works. Compatibility with the competition's public weight release is
unresolved; data access alone is not that permission. No personal application
answers, account credentials or raw MRI/label data were archived.

The version1.0 file indexes contain3,390 training and360 validation files,
consistent with three views for1,130/120 examinations. Each corresponding label
table has only1,129/119 rows. The actual train-acl variable names are `_0000`
and `_0`, strongly suggesting a headerless first record became the header.
This is not a confirmed repair: obtain the original CSV bytes, interpret explicit
no-header columns and validate exact membership across all six label files.
Do not invent a missing case or its label from the column names. The [access and
schema receipt](mrnet-redivis-access-review.json) preserves observed facts and
the unresolved admission gates.

The owner subsequently said KneeCoT's newer permission was on Discord and agreed
it could be left unused. No newer organizer text was supplied or found in the
checked public sources. Leave it excluded for now rather than repeatedly asking
the owner or quietly treating the conflicting statement as a verified ruling.
