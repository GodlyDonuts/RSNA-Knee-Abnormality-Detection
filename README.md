# RSNA Knee Abnormality Detection

Reproducible report annotation, DICOM auditing, study-level training, and offline
Kaggle inference for the [RSNA competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection).

**Code and procedures live here. Data, labels, inventories, and release artifacts
live in [GodlyDonuts/RSNA-Data](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data).**
Do not put DICOMs, reports, credentials, caches, or checkpoints in GitHub.

## Status

The original mirror contains 4,407 training studies (58 with complete official
labels), 819,078 training DICOMs, and 557 DICOMs from the three released sample test
studies. The competition's hidden test studies are not in this mirror.

**Do not use the legacy hard labels for a large training run.** The non-gold
`rsna_manual_labels.csv` has zero positives for ACL, MCL, lateral OA, and PF OA.
Earlier `rsna_pseudo_labels*`, `rsna_custom_labels*`, `rsna_manual_labels*`, and
image-row metadata exports are historical experimental artifacts, not the corrected
release. Their calibration also used the 58 gold studies. These legacy results
must not be presented as independent evaluation of those studies.

The replacement release requires every report to be read and independently
reviewed, all DICOMs to pass the technical audit, and a real-data training/inference
smoke run. The implementation and synthetic regression tests are available here;
release validation is still in progress. Check the versioned HF release's audit
files before starting a costly training run. This repository does not claim a
competitive model score or a diagnostic rereading of every image.

## Setup

Python 3.10 or newer; Python 3.10 is used in CI.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install pip==25.0.1
# CPU development / CI. For GPU training install the matching CUDA build instead.
python -m pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[dev]"
ruff check .
ruff format --check .
mypy rsna
python -m pytest -q
```

The runtime includes `pydicom` and JPEG/JPEG2000 decoders. No report language model,
API call, pretrained-weight download, or internet access is needed during training
or inference. The report workflow itself uses Devin and is a separate preparation
stage.

## Data contract

Targets, in exact order:

```
ACL, MCL, Medial Meniscus, Lateral Meniscus, Medial OA, Lateral OA,
PF OA, Effusion, Synovitis, Baker's, Contusion, Fracture
```

`studies.parquet` has one row per study: UID, split, report, duplicate groups, twelve
labels, per-target `__mask`, `__state`, `__evidence`, and `__provenance` columns.

| Report state | Training value | Mask |
| --- | ---: | ---: |
| positive | 1 | 1 |
| negative (explicit absence) | 0 | 1 |
| uncertain | placeholder 0 | 0 |
| not_mentioned | placeholder 0 | 0 |

Always multiply the loss by the mask. A stored zero with mask zero is **not a
negative diagnosis**. These categorical states are not calibrated probabilities.
Official gold values are preserved exactly, with mask one.

Report duplicates (NFKC, case-fold, whitespace normalization) and nonconstant
identical-pixel duplicates form connected study groups. A SHA-256 of each group's
smallest study UID assigns approximately 90% to `train` and 10% to
`weak_validation`. All 58 official studies go to `gold_evaluation`; non-gold
members of their duplicate groups go to `quarantine`. Train/test pixel duplicates
block label release. Study IDs do not establish patient-disjoint cohorts.

Weak validation measures agreement with reviewed reports, not diagnostic accuracy.
Keep gold evaluation separate from training, threshold fitting, and hyperparameter
selection; report any repeated use honestly. Per-target ROC-AUC is unavailable when
only one class is observed. Official macro ROC-AUC is reported only when all twelve
targets are evaluable.

## Preparation

Use a pinned source revision. The initial source inventory used:

```
52f995f4568e36bfa5fbbaf69a89af41226b4f5d
```

Keep the HF clone outside this repository. A pointer-only clone avoids downloading
the entire 569.8 GB source:

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/datasets/GodlyDonuts/RSNA-Data "$HOME/rsna-hf-repo"
python -m rsna.inventory \
  --repo "$HOME/rsna-hf-repo" \
  --revision 52f995f4568e36bfa5fbbaf69a89af41226b4f5d \
  --metadata-root "$HOME/rsna-hf-repo" \
  --original-paths data/zip_files.txt --output data/inventory/dicom_inventory.parquet
python -m rsna.preprocess \
  --manifest data/inventory/dicom_inventory.parquet --output data/cache256 \
  --source-revision 52f995f4568e36bfa5fbbaf69a89af41226b4f5d
python -m rsna.audit \
  --inventory data/inventory/dicom_inventory.parquet --cache data/cache256 \
  --revision 52f995f4568e36bfa5fbbaf69a89af41226b4f5d --output data/audit
```

`zip_files.txt` is the original Kaggle ZIP member-name inventory, one path per line;
obtain it from the validated HF release or independently enumerate the source ZIP.
Without it, the original-archive path equality check cannot run.
The inventory records SHA-256, byte size, storage type, and UID/path identity for
every DICOM. HF downloads use LFS object batches with verified size and hashes;
the 144 directly stored Git blobs use pinned resolve URLs. Public reads need no
HF token, but `HF_TOKEN` supports authenticated Hub rate limits and private access.
Optional direct Kaggle access uses `KAGGLE_API_TOKEN`; local unpacked
input uses `--root /path/to/competition`.

Caching checks UID consistency, rescale parameters, monochrome interpretation,
physical slice geometry, transfer syntax, pixel finiteness, and dimensions. It
resizes with preserved physical aspect ratio and center padding, scales by the
volume's minimum and maximum, and stores normalized uint16 volumes. Geometry ordering
falls back to unique InstanceNumber only when geometry is incomplete; warnings
remain in metadata. Geometry is used for ordering, not full physical-space
resampling or canonical anatomical reorientation.

Each cache has an atomic `.npz`/`.json` pair with source revision, path fingerprint,
transform version, content checksum, source slice hashes, and decoded pixel hashes.
Reruns reuse verified pairs; incompatible or corrupt pairs are rejected. Move an
orphaned pair aside and rerun that series after investigating the interruption.
`progress.jsonl` records successes and failures. Failed series are never silently
included in a release. Reserve sufficient storage; caching stops below 3 GiB free.

For the annotation stage, see [the review procedure](docs/report-review.md). After
all batches and the DICOM audit complete:

```bash
python -m rsna.labels --train data/train.csv \
  --reviews data/reviews --drafts data/drafts \
  --dicom-audit data/audit/dicom_audit.json --output data/release
```

The command requires exact study coverage and verbatim evidence quotes and writes
`studies.parquet`, `study_labels.csv`, `report_annotations.parquet`, and
`label_audit.json`. Keep the draft and reviewed JSONL files for provenance.

## Train and evaluate

The reference model encodes slices with a small shared 2D CNN, then aggregates
slices and series using attention and MRI sequence metadata. It selects series by
plane before filling with additional fluid-sensitive sequences. This is an
auditable baseline, not a tuned competition-winning architecture. Do not interpret
smoke-run metrics as evidence of model quality.

```bash
python -m rsna.train --studies data/release/studies.parquet \
  --series data/train_series.csv --cache data/cache256 \
  --output runs/baseline --epochs 20 --device cuda \
  --side 224 --slices 16 --max-series 6 --batch-size 2 --workers 4
python -m rsna.train --studies data/release/studies.parquet \
  --series data/train_series.csv --cache data/cache256 \
  --output runs/baseline --epochs 30 --device cuda \
  --side 224 --slices 16 --max-series 6 --batch-size 2 --workers 4 \
  --resume runs/baseline/checkpoint.pt
python -m rsna.infer --checkpoint runs/baseline/checkpoint.pt \
  --studies data/release/studies.parquet --split gold_evaluation \
  --series data/train_series.csv --cache data/cache256 \
  --output runs/baseline/gold_predictions.csv --device cuda
```

Training uses masked BCE, AdamW, AMP on CUDA, gradient clipping, deterministic
epoch seeds, and study-level sampling. All-unknown studies are excluded from
supervised training. Checkpoints include optimizer/scaler states, seed, RNG
states, epoch/step, target order, manifest/series hashes, and preprocessing
configuration. Resume rejects changed manifests/settings or incomplete smoke
epochs. Exact epoch-boundary continuation is tested on CPU; GPU/distributed
reproducibility and throughput require measurement on the target hardware.

For multiple GPUs, run the same command with
`torchrun --standalone --nproc-per-node=4 -m rsna.train ...`.
Rank zero writes checkpoints and evaluates the full weak-validation split.
Masked target counts are reduced across ranks so missing labels do not change
the per-target weighting between single-process and distributed training.

## Offline Kaggle inference

Prepare an offline wheel bundle in the Kaggle notebook's input dataset if its
installed dependencies do not match this package. Install from that bundle with
`pip install --no-index --find-links /kaggle/input/your-wheels ...`. Include this
package and the checkpoint as notebook inputs; preflight all imports offline.

```bash
python -m rsna.infer --checkpoint /kaggle/input/your-model/checkpoint.pt \
  --series /kaggle/input/rsna-knee-abnormality-detection/test_series.csv \
  --template /kaggle/input/rsna-knee-abnormality-detection/sample_submission.csv \
  --dicom-root /kaggle/input/rsna-knee-abnormality-detection \
  --cache /kaggle/working/cache --output /kaggle/working/submission.csv \
  --source-revision kaggle-hidden-test --device cuda
```

Do not pass training reports to inference. With a template, the command predicts
every listed study and verifies exact columns, unique IDs, row order, finite
values, and probabilities in [0, 1]. A prebuilt matching cache can be supplied
without `--dicom-root`. The metrics sidecar records preprocessing and inference
timings. The released three-study sample cannot establish hidden-test performance.

The competition requires internet disabled and a nine-hour notebook runtime.
Measure the complete notebook, including preprocessing and loading, on Kaggle
hardware before submission. Local CPU timings are not a Kaggle runtime guarantee.

## Releases

See [release procedure](docs/releases.md). Pin **both** the GitHub code commit and
HF data revision in experiment records. Never replace a versioned data release in
place or promote an incomplete review/audit to the default training dataset.
