# Decisions, corrections, and unresolved work

## Current corrective run

The registry now contains 333 active corrections. Cartilage003 adds 47 after
root's 38 full-source rereads / 114 frozen OA judgments and explicit independent
acceptance of the second reader's exact proposals. Thirty positives and twelve
negatives become uncertain; five source quotations gain needed context. All 286
prior correction objects and the three previous retirements remain unchanged.

All 114 final root states match inspection, and 54 identical-report groups / 204
reports have zero conflicts. Four root reconsiderations preserve the distinction
between lesion location and qualifying extent, and between focal normal/below-
threshold findings and whole-compartment coverage. The unchanged rubric permits
qualitative broad extent and does not require two measured axes.

The latest verified publication contains 286 corrections at Git `76d2c274` /
private HF `543c2d84`. The 333 milestone needs its own verified publication receipt.
All three global semantic QA gates remain pending. See
[adjudications.md](adjudications.md). The expanded mission also requires a
competitive training research and experiment plan after data readiness; see
[mission-and-training-plan.md](mission-and-training-plan.md).

The required correspondence registry enforces 18 whole-study report-supervision
exclusions. The latest is 069:10, accepted independently from its full RIGHT
report and conflicting literal L header fields; no new image interpretation is
claimed. Three other source/header cases retain their current supervision,
subject to semantic QA. Twenty-four unresolved cases already have all twelve
report targets masked and require an exact final-export zero-mask check.
All original clinical artifacts and the prior seventeen exclusions are preserved.

The reviewed sampling preference retains every original series and gives lower
priority to three alternatives in two studies. The loader preserves plane
coverage and existing ordering/RNG behavior for all unaffected studies. See
[the policy, complete table and validation](series-preferences.md). This preference
does not certify equivalent coverage or alter clinical labels. The combined
462-test suite, Ruff and mypy pass; global semantic QA remains unfinished.

A required source-bound report-similarity registry now augments the exact-report/pixel leakage graph before gold quarantine. The first accepted edge covers038:44/045:15, whose only difference is a redundant full stop and space. No source text, clinical annotation, patient or hospital identity is inferred or changed. See [the policy and validation](report-similarity.md).

The owner authorized direct main integration and a monorepo on 2026-09-20.
Use `protocols/report_weak_v2.md`, `protocols/agent_tasks.md`, and
`initiative/reviews-v2/`; the previous rubric was too permissive about severity.
The current decoder/cache version is `series-u16-v3`. Fresh authenticated Kaggle
metadata now matches the archived source byte for byte. All 4,410 sampled study
headers have distinct pseudonymous PatientIDs, which does not prove real patient
disjointness. Read `docs/codex-continuation.md` for current progress and gates.
Both GitHub and Hugging Face are verified private destinations for preparation
checkpoints. Preserve restricted access, immutable source revisions and verified
publication receipts. Historical access and publication notes below describe the
previous run; the current evidence is in the continuation guide.

## Objective and scope

The owner wants a dependable RSNA Knee Abnormality Detection training package
and asked for each training study's report to be individually read and checked.
They authorized sustained work, parallel agents, and incremental commits.
GitHub must contain enough data, context, results, and procedures for another
session to continue. The repository is private; reports and suitable image,
cache, and checkpoint artifacts are permitted.

## Earlier approaches and retracted claims

Earlier label exports combined public report-derived scores, NLI outputs,
rule-based extraction, and calibration on the 58 official studies. The
`pseudo`, `custom`, and `manual` filenames do not establish human or per-report
agent review. In particular, the older `manual` export was an automated
rule/NLI procedure, despite the misleading name.

An audit found **zero non-gold positives for ACL, MCL, lateral OA, and PF OA**
in the legacy hard `rsna_manual_labels.csv`. Do not train a large model with it.
The previous training-ready claim was incorrect: loading and a small decode
sample had passed, but label validity, duplicate isolation, and full image
coverage had not been established. The old study split also did not isolate all
official gold and duplicate groups. These artifacts are retained under
`initiative/legacy/` for reproducibility, with their limitations visible.

## Replacement protocol

Every one of 4,407 reports receives an annotator and a separate reviewer, blinded
to official target columns. The workflow batches reports deterministically and
keeps up to six agents active, starting new work as slots become free.

The target order is fixed in `rsna/schema.py`. Each target receives `positive`,
`negative`, `uncertain`, or `not_mentioned`. Positive and negative decisions
need explicit evidence; every non-missing state requires a literal contiguous
report quote. Silence never becomes negative. Uncertain and unmentioned
targets are masked during training. Ambiguity is retained in notes.

Ligament identities, meniscus tear versus degeneration, OA compartment,
effusion versus synovitis, and contusion versus nonspecific marrow edema must
be distinguished. The original rubric is archived verbatim. Agents must read
reports individually rather than replace the task with regex, public scores,
or an NLI model.

## Leakage and evaluation

All series belonging to a study stay together. Normalized duplicate reports and
identical nonconstant decoded pixels join connected leakage groups. Official
gold stays in `gold_evaluation`; weak studies connected to those groups are
quarantined. Train/test pixel duplicates block release. Study identifiers alone
do not prove patient disjointness.

The 58 official studies are not used to calibrate replacement labels. Gold
comparison follows blinded annotation/review. Weak validation evaluates
agreement with report-derived supervision. A target needs both classes for AUC;
macro AUC is unavailable unless every target is evaluable. Repeated gold
inspection or rubric tuning must be declared rather than called unbiased.

## Image processing and training

The corrected source inventory and `series-u16-v3` cache have passed the full
819,635-slice audit. Two invalid mirror encodings were replaced with exact official
files in a separately preserved source revision. The cache verifies source hashes and UIDs,
decodes rescale/monochrome conventions, orders by geometry, preserves physical
aspect, and uses per-volume min/max scaling with uint16 storage. Geometry
warnings are preserved; cache pairs are atomic and checksummed.

The model is an auditable multi-series/multi-plane baseline. Masked BCE is
normalized per target across all distributed ranks. Checkpoints bind manifests,
targets, preprocessing, and training settings and preserve optimizer/scaler/RNG
state. Exact CPU epoch-boundary resume has regression coverage. A small baseline
is not a validated winning architecture.

## Access and publication

The original Kaggle-to-HF transfer completed. The source inventory verifies
819,635 DICOMs plus the original metadata files against the original archive
index. HF is used because the Kaggle download credential previously returned
HTTP 401. HF rate limits required authenticated LFS reads and reset-aware retry.
Source counts and immutable revisions are archived; no credential is needed to
understand the work.

Github holds actual reports, metadata, batch inputs/outputs, provenance, audits,
and suitable model/data artifacts. Large files are partitioned or compressed
where appropriate; full image/cache collections retain immutable external
references. Small release artifacts should be mirrored into both repositories.
Historical outputs and in-progress drafts are not a corrected release.

## Remaining acceptance gates

1. All 4,407 reports have two archived review passes with validated evidence and
   IDs. There are 333 active accepted target corrections, each with exact
   original/source bindings and preserved prior artifacts. Resolve the remaining cartilage, bone terminology, and qualified-negation/
   duplicate semantic gates against the exact final artifacts and corrections.
2. Recheck zero report loss masks for all 24 administratively unresolved
   correspondence cases after final adjudication. Incorporate the complete image
   audit, accepted exclusions and verified duplicate edges into final isolation.
3. Generate disjoint corrected manifests, coverage, distributions, and gold
   comparison without training or calibrating on the gold evaluation set.
4. Complete real-data smoke training, exact resume, gold inference, and exact
   sample-submission inference.
5. Record configurations, hashes, measured timings, warnings, and limitations.
6. Publish immutable corrected artifacts, verify downloaded hashes, and update
   GitHub/HF references and release status.

There is no claim that all images have been diagnostically reread by a
radiologist, that report labels are optimal, that the hidden test set is
available, or that winning performance has been demonstrated.
