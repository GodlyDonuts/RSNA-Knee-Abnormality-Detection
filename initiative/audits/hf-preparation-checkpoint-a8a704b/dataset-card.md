---
configs:
- config_name: default
  data_files:
  - split: train
    path: train_metadata.parquet
  - split: validation
    path: val_metadata.parquet
  - split: test
    path: test_metadata.parquet
  - split: gold
    path: gold_metadata.parquet
  default: true
---

# RSNA Knee Abnormality Detection Dataset

## Complete technical image audit — September 21, 2026

The newest private [preparation checkpoint](preparation-checkpoints/a8a704bc3913b193ce1f3359773a829db4a63f4e) preserves GitHub `a8a704bc3913b193ce1f3359773a829db4a63f4e` with 2,657 independently reviewed reports, 31,884 target decisions and 200 passing tests. It remains **incomplete and not training-ready**.

The corrected source is pinned at `0b5e410288232f570c69ba4f44723356e6ae6ab7`. All819,635 slices in24,386 series passed the complete technical cache audit with zero warnings. One cross-study nonconstant pixel link is retained for leakage isolation, with no overlap against the three released test studies. This is not a diagnostic reread or a hidden-test claim.

[Full image audit and input-selection evidence](audits/full-image-recovery-v1) includes per-series cache lineage, the complete audit/index, and exact training/test selections. The training table preserves all4,407 studies while excluding33 fully equivalent model slots, leaving24,338 selected series; all15 released test series remain. The original source files and baseline are preserved.

Full report review, global semantic/correspondence adjudication, remaining technical candidate assessment, the final leakage-isolated manifest and final training/resume/inference checks remain required. Older sections below describe historical checkpoints.

## Verified preparation checkpoint — September21,2026

The private [preparation checkpoint](preparation-checkpoints/31e4ff8d456e37fffc16fcd09f37ca613d2d7c02) preserves GitHub commit `31e4ff8d456e37fffc16fcd09f37ca613d2d7c02` after152 tests, style/type checks and archive validation passed. It contains2,457 reports with two complete source readings and29,484 explicit target decisions. It is **incomplete and not training-ready**; full report coverage, semantic/correspondence adjudication, complete pixel/leakage audit and final training/resume/inference checks remain required.

The full image pass decoded819,561 slices and exposed two nonconforming mirror files in two series. Their8-bit payloads contradicted unchanged16-bit DICOM headers. Exact authenticated Kaggle originals were obtained and independently compared; the two source paths in this commit restore those authoritative originals. No guessed pixel reconstruction or decoder exception was used. [Source correction evidence and effective inventory](audits/source-corrections/official-pixel-representation-v1) preserve exact old/new hashes, official-file verification and both independent assessments. The original baseline remains pinned at `52f995f4568e36bfa5fbbaf69a89af41226b4f5d`.

A fresh official ZIP index covers all819,635 image paths; its only file-size differences are those two known files. This is a full path/size check, not a claim that every mirror file has been compared byte-for-byte with Kaggle. All74 sources in the affected series match official ZIP size and CRC after substituting the two originals. Rebuilding those series and auditing the complete corrected cache remain pending.

## Current status: replacement labels under review

**Do not start a large training run with the legacy hard labels or default
image-row metadata.** The non-gold rows in `rsna_manual_labels.csv` contain zero
positives for ACL, MCL, lateral OA, and PF OA. The former "training-ready" claim was
incorrect. Preserving the 58 official rows exactly was not independent validation:
they were also used for calibration, and the old train/validation exports include
gold studies.

Code, tests, the full annotation rubric, training, and offline Kaggle inference
now live in [GodlyDonuts/RSNA-Knee-Abnormality-Detection](https://github.com/GodlyDonuts/RSNA-Knee-Abnormality-Detection).
Verified changes are integrated directly into the repository’s `main` branch.
Hugging Face remains the home for the original data and versioned derived artifacts.

The replacement process reads every report and independently reviews each batch,
retains verbatim evidence, distinguishes positive/negative/uncertain/not-mentioned
states, and masks unknowns during training. It also isolates gold evaluation,
groups report/image duplicates, and audits DICOM decoding and source checksums.
**That full review and audit are still in progress; corrected labels have not yet
been released.** This is report annotation and technical image validation, not a
diagnostic rereading of all images.

The [verified source inventory](https://huggingface.co/datasets/GodlyDonuts/RSNA-Data/tree/86df08e5be2149aef0dc855f0ebffd376e164a19/audits/source-inventory-52f995f)
is available with the original ZIP path list, all DICOM object hashes/sizes,
study/series coverage counts, and artifact checksums. All 819,635 DICOM paths match
the original archive index and study/series CSVs. This verifies source coverage;
full pixel decoding and corrected report labels remain in progress.

The original competition files remain available. Everything below documents the
**deprecated historical exports**, including the current `load_dataset` default.
Do not treat those exports as the upcoming corrected release.

## Original data and historical exports

This repository contains the RSNA knee abnormality detection competition data.

* **Source:** Kaggle competition `rsna-knee-abnormality-detection`
* **Hugging Face dataset:** `GodlyDonuts/RSNA-Data`
* **Images:** 819,078 training DICOMs across 4,407 studies, plus 557 test DICOMs across 3 sample test studies.
* **Labels:** 12-class knee abnormality labels. 58 studies have competition-supplied gold labels; the remaining 4,349 studies have labels derived from their radiologist reports using an inspectable rule + sentence-level multilingual NLI pipeline, calibrated on the 58 gold rows. Both hard (0/1) and soft (probability) labels are provided.

## Legacy splits

| Split | Studies | Images | File | Notes |
|-------|-----------|--------|------|-------|
| train | 3,505 | 657,334 | `train_metadata.parquet` | 80% of studies, pseudo + gold labels |
| validation | 902 | 161,744 | `val_metadata.parquet` | 20% of studies, pseudo + gold labels |
| test | 3 | 557 | `test_metadata.parquet` | Released sample test studies, no labels |
| gold | 58 | 10,528 | `gold_metadata.parquet` | Competition-supplied expert labels only |

The historical train/validation split is deterministic and study-level: a study is assigned by `md5(study_uid) % 10`, with 0-7 as train and 8-9 as validation. It does not isolate the gold cohort from training or group duplicated reports across studies.

## Metadata columns

Each metadata row corresponds to one DICOM instance:

| Column | Description |
|--------|-------------|
| `study_uid` | `StudyInstanceUID` |
| `series_uid` | `SeriesInstanceUID` |
| `sop_uid` | `SOPInstanceUID` |
| `path` | DICOM path inside the repo (e.g., `train_series/<study>/<series>/<sop>.dcm`) |
| `plane` | `Anatomical_Plane` (`Sagittal`, `Coronal`, `Axial`, etc.) |
| `fluid_sensitive` | Series-level fluid-sensitive flag (`0`/`1`, `-1` if unknown) |
| `fat_suppression` | Series-level fat-suppression flag (`0`/`1`, `-1` if unknown) |
| `report` | Radiologist report text (train/validation only) |
| `ACL` ... `Fracture` | Hard 0/1 labels for the 12 target findings |
| `ACL_soft` ... `Fracture_soft` | Calibrated soft probabilities for the 12 target findings |

The label columns are: `ACL`, `MCL`, `Medial Meniscus`, `Lateral Meniscus`, `Medial OA`, `Lateral OA`, `PF OA`, `Effusion`, `Synovitis`, `Baker's`, `Contusion`, `Fracture`.

### Label distribution (train split, image-level)

| Finding | Positives | Prevalence | Mean soft |
|---------|----------:|-----------:|----------:|
| ACL | 3,609 | 0.55% | 0.408 |
| MCL | 832 | 0.13% | 0.152 |
| Medial Meniscus | 3,852 | 0.59% | 0.445 |
| Lateral Meniscus | 73,210 | 11.14% | 0.376 |
| Medial OA | 51,363 | 7.81% | 0.243 |
| Lateral OA | 1,075 | 0.16% | 0.177 |
| PF OA | 2,846 | 0.43% | 0.362 |
| Effusion | 632,499 | 96.22% | 0.586 |
| Synovitis | 72,735 | 11.07% | 0.431 |
| Baker's | 68,876 | 10.48% | 0.207 |
| Contusion | 23,554 | 3.58% | 0.283 |
| Fracture | 36,012 | 5.48% | 0.230 |

## Historical metadata access with `datasets`

Load metadata only (fast, no DICOM decoding):

```python
from datasets import load_dataset

dataset = load_dataset("GodlyDonuts/RSNA-Data", split="train", streaming=True)
for example in dataset:
    print(example["study_uid"], example["ACL"])
    break
```

## Legacy image-level DataLoader (deprecated)

The helper `rsna_dataset.py` in this repo streams and decodes DICOMs on-the-fly:

```python
import torchvision.transforms as T
from rsna_dataset import RSNAStreamingDataset, TARGETS, collate_fn

transform = T.Compose([T.Resize((512, 512)), T.ToTensor()])

ds = RSNAStreamingDataset(split="train", transform=transform, label_mode="hard")

loader = torch.utils.data.DataLoader(
    ds, batch_size=8, num_workers=0, collate_fn=collate_fn
)

for images, labels in loader:
    print(images.shape, labels.shape)  # [B, 1, 512, 512], [B, 12]
    break
```

`label_mode="soft"` returns the calibrated probabilities instead of thresholded hard labels.

## DICOM decoding details

* Modality rescale (`RescaleSlope`/`RescaleIntercept`) is applied when present.
* If `WindowCenter`/`WindowWidth` are present, the linear VOI transform is applied and clipped to 0-255.
* If no window is present, min-max scaling to 0-255 is used.
* `PhotometricInterpretation == MONOCHROME1` is inverted so that low pixel values are black.
* `pylibjpeg` and `pydicom` handlers are used for compressed transfer syntaxes.

## Files in this repository

* `train_metadata.parquet`, `val_metadata.parquet`, `test_metadata.parquet` — image-level metadata and labels.
* `rsna_dataset.py` — DICOM decoder and `RSNAStreamingDataset` / `DataLoader` helper.
* `train_example.py` — minimal runnable training example.
* `train_series/`, `test_series/` — raw DICOM files.
* `train.csv`, `train_series.csv`, `test.csv`, `test_series.csv` — original metadata from the competition.
* `rsna_manual_labels.csv`, `rsna_manual_labels_soft.csv` — study-level hard and soft labels.

## Label generation summary

The 58 gold-labelled studies define the 12-class schema. For the remaining 4,349 training studies we used an inspectable multilingual rule teacher (`report_teacher.py`) plus a sentence-level multilingual `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` NLI model. The raw scores were calibrated with a `LogisticRegression(C=0.5)` on the 58 gold rows, and the gold rows override the weak targets. Spot checks and the zero-gold-mismatch check confirm the 58 expert labels are preserved exactly.

## Important caveats

* The legacy labels are **unreliable pseudo-labels** derived from reports. They are not verified image labels or a validated training release.
* The test split contains only the three studies released with the competition test folder. The hidden Kaggle test set is not included.
* Repository access was verified private on2026-09-21. Keep competition data and preparation checkpoints restricted in accordance with the competition terms.
