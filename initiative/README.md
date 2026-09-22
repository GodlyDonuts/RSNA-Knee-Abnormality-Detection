# Complete initiative record

This folder contains actual data and evidence, not only links to a previous
session. A new agent can recover the report-review work and source inventory
from a clone of this repository.

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
are verified. The first GPU pilot stopped before training because its allocated
GPUs were not correctly available to the process. Basic CUDA checks passed on
another Newton node, and the exact offline Kaggle T4 runtime also passed. The
recovery-pilot retry then exposed a strict weight-key compatibility mismatch
before training; that fix and the full-cache transfer continue. No competitive
DINOv3 result has been measured yet. The
accepted data release is unchanged. See the [deployment evidence](audits/dinov3-deployment-v1/).

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
