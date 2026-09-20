# Decisions, corrections, and unresolved work

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

The source inventory is complete and source-verified. Full decoding is separate
and still must finish. The `series-u16-v2` cache verifies source hashes and UIDs,
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

1. Finish and archive every independent review; validate all evidence and IDs.
2. Finish the full cache and technical image audit; resolve failures/duplicates.
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
