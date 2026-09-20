# Data and code release procedure

GitHub contains executable code, tests, and procedures. Hugging Face
[`GodlyDonuts/RSNA-Data`](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data)
contains the original competition data and versioned derived artifacts.

## Required release contents

Use a new `releases/<version>/` directory in HF for each release:

- `studies.parquet`, `study_labels.csv`, `report_annotations.parquet`,
  `label_audit.json`;
- source `train.csv`, `train_series.csv`, `test_series.csv`,
  `sample_submission.csv`, and original `zip_files.txt`, or pinned references;
- `dicom_inventory.parquet`, `dicom_inventory.json`, `dicom_audit.json`,
  `series_cache_index.parquet`;
- original annotation and reviewed JSONL files, rubric/script revision, and
  workflow result metadata;
- preprocessing cache shards and checksums if published, otherwise exact commands
  and measured disk requirements to rebuild;
- smoke-run metrics, checkpoint/resume verification, sample submission validation,
  measured runtime, and machine/dependency details;
- `release.json`: code commit, source revision, target order, label policy, seed,
  cache version/resolution, artifact hashes and sizes, validation status, and
  known limitations.

Never include credentials, authorization headers, temporary signed URLs, or
Devin-private attachment links in the public release. Upload the actual annotation
files and retain public HF paths; retain internal workflow metadata privately.

## Acceptance gates

1. Confirm exact inventory path equality with the original archive, expected split
   counts, unique SOP/path identities, and source byte hashes.
2. Complete decoding and verify every cache against source inventory and its own
   checksum. Resolve failures; record geometry/spacing warnings without pretending
   the images have been diagnostically reviewed.
3. Complete both annotation passes for all reports and validate all evidence.
4. Check report/image duplicate groups, official-gold isolation, target state
   distributions, masked coverage, and gold comparison. Investigate class collapse.
   Unknowns remain unknown even when coverage is low.
5. Run tests, lint, and type checks. Exercise real-data optimizer updates,
   checkpoint resume, gold prediction, and three-study sample submission offline.
6. Publish versioned artifacts and their checksums to HF. Verify downloads from the
   resulting immutable commit before updating dataset-card examples.
7. Update GitHub/HF documentation with reciprocal links, measured validation
   results, and limitations. Pin code and data revisions in every experiment.

The corrected study manifest supersedes legacy image-row metadata. Keep legacy
files for historical reproducibility with a clear deprecation notice; do not
silently relabel them as the corrected release. Current Hugging Face `load_dataset`
default configurations may still point at legacy metadata until promotion.

## Scope of readiness

A successful release establishes technical pipeline integrity and auditable weak
labels. It does not guarantee optimal labels, a winning leaderboard score, patient
disjointness from study IDs alone, or fitness for clinical use. Large-scale GPU
training, architecture selection, and full Kaggle runtime validation remain
separate measured experiments.
