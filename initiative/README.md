# Complete initiative record

The [October 17 owner program and critical assessment](research/owner-deadline-program-20261017-v1/README.md)
and [actual S6 source-only admission](audits/s6-public-source-20261017-v1/README.md)
are available for new research agents. The [original S6 checkpoint](audits/s6-checkpoint-admission-20261017-v1/README.md)
is now privately downloaded once and SHA/header verified. The reported S6 score has not yet been
reproduced by this project.

**Current state is maintained in the [current-state guide](../docs/current-state.md),
[research brief](../docs/research-brief.md) and [research-agent handoff](../docs/research-agent-handoff.md).**
As of October 17, 2026, 03:20 UTC, our own N3 epoch 16 scored 0.894; the attributed
external nartaa baseline scored 0.949. Full-parent epoch1 is complete and privately verified, epoch2 has resumed,
and expanded-own-model training is running into epoch5. The first matched
uniform16 GPU session is also running; no new score is claimed. Use those current guides and original receipts for live status.

## Historical releases and training narrative

The narrative below is retained as historical evidence. Its “latest,” “next,”
queue, quota and pending statements describe their original dates, not current
execution. The 0.854 October 13 result was superseded by later own-model results;
it is not today's best score.

This folder contains actual data and evidence, not only links to a previous
session. A new agent can recover the report-review work and source inventory
from a clone of this repository.

The September 29 DINOv3 encoder-adapter fold-0 run completed with **0.8621 internal
macro AUC**, versus 0.7783 for the earlier model on the identical held-out pool.
All ten checkpoints are privately download/hash verified. See the
[A32 result audit](audits/dinov3-adapter-a32-fold0-result-v1/); this is not a Kaggle score.

**The local report-supervision release has passed data and runtime acceptance.**
Use [the readiness report](../docs/data-readiness-report.md), the final
[manifest and receipt](releases/report-supervision-v1/), and the
[training strategy](../docs/training-strategy.md). There are 3,904 training studies,
441 weak-validation studies, 58 official evaluation studies and four quarantined
studies. All 569 accepted corrections and original judgments remain preserved.
The three semantic QA gates are resolved against the exact accepted conservative
supervision ledger; unresolved clinical claims remain masked.

The complete image audit and actual final-input optimizer, exact CPU resume,
58-study evaluation prediction and three-study template checks passed. Rare targets
still have limited evidence, and no competitive score or complete diagnostic image
reread is claimed. The 24 unresolved correspondence cases comprise 23 masked
non-gold studies and one official evaluation-only study; all 18 registered report
exclusions are masked.

The accepted data/code release is Git `f41879a3`. Its [private Hugging Face release](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data/tree/a177562d5f49458e287752b94f085dcb061f6495/releases/report-supervision-v1) is pinned to `a177562d5f49458e287752b94f085dcb061f6495`; all 19 direct-download objects passed full byte verification. The [publication record](provenance/report-supervision-v1-publication.json) links the separate complete-tree recovery archive and both restoration proofs.
The [expanded mission](../docs/mission-and-training-plan.md) includes both data
readiness and a competitive training plan.

The current run uses the frozen [v2 rubric](../protocols/report_weak_v2.md) and
[agent procedure](../protocols/agent_tasks.md). The `reviews/` outputs and index
preserve the historical v1 run; its blinded `inputs/` are reused unchanged.
Verified milestones go directly to main under the owner's explicit authorization.

The next experiment uses DINOv3-L on Newton. The [competitive training handoff](../docs/competitive-training.md)
and [DINOv3 recipe](../configs/competitive/dinov3-campaign.json) describe the new
training-only folds, two-GPU runtime, and authorized two-epoch Kaggle cadence.
The pinned weights, pilot cache, exact Newton source and offline CPU environment
are verified. Earlier GPU attempts exposed device visibility, checkpoint-key
and missing Python-header failures. The corrected adapter and private headers
then passed CPU loading and exact runtime checks. [Newton job 839321](audits/dinov3-h100-pilot-v6/)
completed a real two-H100 DINOv3-L diagnostic: 100 training studies, two epochs,
26 optimizer updates and exact decoded checkpoint state and outer predictions
after interruption/resume. Its uninterrupted trainer elapsed time was 109.06
seconds including prediction. The [full-cache transfer](audits/full-cache-newton-stage-v1/)
is verified at 48,772 files and 82.83 GB. The [actual two-H100 Slurm
stop and separate resume](audits/newton-recovery-production-v1/) passed: a
checkpoint was preserved after three updates, the resume completed all 26,
and one-CPU comparison found equal final checkpoint payloads and outer
predictions against the uninterrupted pilot. The corrected ten-epoch fold-0
production job 840562 reached 22 epoch-0 optimizer updates on 2,247 eligible
optimization studies, then stalled in CUDA synchronization and ended without a
checkpoint after the verified stop grace expired. The [replacement audit](audits/newton-production-stall-v1/)
records its failure and the single v6 job 840578 dispatch on evc40. That job
reproduced the identical update-22 CUDA stall and also ended without a
checkpoint. The [repeatability audit](audits/newton-repeatable-stall-v1/)
records the evidence: CPU-only job 840735 extracted the exact update-23 cache
members but failed an image-range assertion; CPU-only job 840800 isolated 20
interpolated pixels at 1.000000238 in one rank-1 study. The loader now clips
this FP32 roundoff. CPU job 840938 verified the exact eight-study fix;
bounded H100 job 840991 crossed the previous stall and verified an update-26
checkpoint. One-CPU job 841091 decoded and preserved that checkpoint, then
the first two-H100 continuation 841094 failed at evc27's CUDA preflight before
training. The single retry 841107 allocated evc28 with the same verified
checkpoint. The [fix audit](audits/newton-interpolation-fix-v1/) preserves the pinned source,
checkpoint proof and dispatch. Continuation 841107 completed all ten full-data
epochs and final outer-fold prediction (`COMPLETED 0:0`); each immutable checkpoint
is in the private Hugging Face
dataset with full download verification. The separately replayed epoch-1
checkpoint passed CPU proof. Its fixed-final-epoch fold-0 macro AUC was
0.7783102649 across twelve accepted report-derived targets on 1,156 development
studies; this is neither a Kaggle score nor a three-fold estimate. The first
completed-epoch private Kaggle Dataset
passed full byte verification, and its exact-source MRI-only T4 notebook
completed after a recorded slug migration. It was submitted once as competition
entry 56482890; its hidden rerun later failed without a score. The
epoch-2 private Dataset also passed full byte verification and its version-1
notebook completed MRI-only inference; it was submitted once as entry 56483451
but its hidden rerun failed. The epoch-6 notebook also completed MRI-only inference and
was submitted once as entry 56483731; its hidden rerun failed. The epoch-4 notebook
also completed MRI-only inference and was submitted once as entry 56483949
but its hidden rerun failed. Epoch 8's private Dataset is byte-verified, but its single
notebook push is uncertain after a Kaggle server error and awaits read-only
reconciliation. The epoch-10 private Dataset appeared later and passed full
download/hash verification. An exact-search guard separated the epoch-8
unnamed placeholder from the absent epoch-10 slug; one CPU-only recovery pushed
the unchanged private offline epoch-10 notebook as version 1. Its MRI-only
public-example receipt passed and that version was submitted once as Kaggle
entry 56485356; its hidden rerun failed. The account's five daily slots are now
used. A separate exact-source epoch-8 notebook identity
was pushed as private offline version 1 without copying Dataset bytes; its
three-study MRI-only public-example receipt passed. Its version-1 notebook is
held from submission pending a corrected full-roster version. Kaggle's exact
[failure audit](audits/kaggle-hidden-rerun-failure-v1/) shows all five submissions
`COMPLETE` with zero output bytes and unhandled notebook errors; none scored.
The cause is still under investigation. See the
[epoch-1 Kaggle audit](audits/kaggle-epoch1-baseline-v1/),
[epoch-2 audit](audits/kaggle-epoch2-v1/), [epoch-4 audit](audits/kaggle-epoch4-v1/),
[epoch-6 audit](audits/kaggle-epoch6-v1/), [epoch-8 audit](audits/kaggle-epoch8-v1/),
[epoch-10 audit](audits/kaggle-epoch10-v1/),
and current [continuation guide](../docs/codex-continuation.md).
The [post-fold-0 audit mirror](audits/hf-post-fold0-milestone-v1/) preserves the
plan, recomputed metrics and Kaggle receipts in the private Hugging Face dataset
with full re-download verification.
The accepted data release is unchanged. See the [deployment evidence](audits/dinov3-deployment-v1/).

## Contents

| Location | Contents |
| --- | --- |
| `source/train.csv` | All 4,407 original reports, study identities, and official label columns |
| `source/train_series.csv` | All 24,371 training-series metadata rows |
| `source/test*.csv`, `source/sample_submission.csv` | Released three-study test metadata and exact submission template |
| `source/inventory/` | Every one of the 819,635 DICOM inventory records, in ordinary Git-tracked Parquet parts |
| `source/zip_files.txt.gz` | Complete original archive member-name index, losslessly compressed |
| `source/dicom_inventory.json` | Verified source counts, uniqueness results, and source revision |
| `reviews/inputs/` | All 89 deterministic batches of original reports, with gold columns excluded |
| `reviews/drafts/` | Original annotator outputs, one JSONL per completed batch |
| `reviews/reviewed/` | Original independent-review outputs, added as they complete |
| `reviews/provenance.json` | Agent/session/workflow IDs, study counts, completion notes, paths, and SHA-256 hashes |
| `reviews/index.json` | Exact batch coverage, evidence-validation results, and unfinished stages |
| `reviews-v2/` | Current drafts, independent reviews, source-first prereads, bound provenance, and progress index |
| `audits/` | Measured code/data validation and explicitly partial progress snapshots |
| `provenance/` | Source revisions, original workflow, original published checksums, external artifact catalog |
| `legacy/` | Earlier label exports, intermediate NLI/rule outputs, transfer scripts, and historical reference code |
| `manifest.json` | SHA-256 and byte size of every archived file except this manifest itself |

The inventory is partitioned into at most 100,000 rows per file so every file
fits GitHub's regular Git file-size limit. Concatenating the parts through
`pandas.read_parquet("initiative/source/inventory")` reproduces the complete
original table; the original table and the partitioned round trip were checked
for exact equality. The original single-file hash is recorded separately.

```bash
python -m rsna.archive
# Current v2 completion gate; deliberately fails while either pass is missing:
python -m rsna.review_progress --inputs initiative/reviews/inputs --root initiative/reviews-v2 --rubric protocols/report_weak_v2.md --require-complete
```

The default check allows an explicitly recorded incomplete archive; it does not
certify a release. It detects missing/changed files, changed source inputs,
unrecorded batches, and drift between the review index and actual outputs.
Unsupported evidence is retained as an invalid original artifact and blocks
completion rather than being silently repaired or discarded.

## Working agreements

The owner requested that GitHub hold everything useful for another session:
reports and other data are permitted in this private repository. Suitable
DICOM samples, caches, checkpoints, and small experiment outputs can also be
tracked. The complete original image collection is 569,754,741,408 bytes;
full image/cache/model collections need external storage with immutable
references and checksums. See the [external artifact catalog](provenance/external-artifacts.json).
Credentials and signed download URLs remain outside Git history.

Actual completed artifacts are pushed incrementally. The archive does not
pretend that unfinished agent outputs exist. A new session should check the
workflow for newly completed batches and archive them using
[the maintenance procedure](../docs/archive-maintenance.md).

## Navigation

- [Current continuation guide](../docs/codex-continuation.md)
- [Decisions, corrections, and limitations](../docs/decisions.md)
- [Report review protocol](../docs/report-review.md)
- [Release gates](../docs/releases.md)
- [Historical artifacts and their limitations](legacy/README.md)
- [Main repository](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection/tree/main)
- [Hugging Face data](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data)
