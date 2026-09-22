# Complete initiative record

This folder contains actual data and evidence, not only links to a previous
session. A new agent can recover the report-review work and source inventory
from a clone of this repository.

**The corrected training release is unfinished.** Read
[`reviews-v2/index.json`](reviews-v2/index.json) for current per-batch availability and
validation. Drafts awaiting independent review must not be promoted as final
labels. Review completion alone does not establish image audit or training
readiness.
The current registry contains **362 active accepted target corrections** and
18 enforced report-supervision exclusions. Cartilage005 adds four: two negatives
become uncertain, and two quotations gain supporting context. All 358 earlier
correction objects and the three earlier retirements remain unchanged. Both
readers and root attention adjudication have finished the priority cartilage set;
three global semantic QA gates and final leakage, mask, manifest and training
checks remain open. Twenty-four unresolved correspondence cases must retain zero
report-loss masks.

The most recent verified private publication contains 358 corrections and 8,955
restorable Git files at Git `3d3b135f` / HF `33d5abd3`. The local 362 milestone needs
its own verified publication receipt. After data readiness, the owner's expanded
mission requires competitive training research and an experiment plan; see
[mission-and-training-plan.md](../docs/mission-and-training-plan.md).

The current run uses the frozen [v2 rubric](../protocols/report_weak_v2.md) and
[agent procedure](../protocols/agent_tasks.md). The `reviews/` outputs and index
preserve the historical v1 run; its blinded `inputs/` are reused unchanged.
Verified milestones go directly to main under the owner's explicit authorization.

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
