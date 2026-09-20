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

This repository contains the RSNA 2023 knee abnormality detection data prepared for model training.

* **Source:** Kaggle competition `rsna-knee-abnormality-detection`
* **Hugging Face dataset:** `GodlyDonuts/RSNA-Data`
* **Images:** 819,078 training DICOMs across 4,407 studies, plus 557 test DICOMs across 3 sample test studies.
* **Labels:** 12-class knee abnormality labels. 58 studies have competition-supplied gold labels; the remaining 4,349 studies have labels derived from their radiologist reports using an inspectable rule + sentence-level multilingual NLI pipeline, calibrated on the 58 gold rows. Both hard (0/1) and soft (probability) labels are provided.

## Splits

| Split | Studies | Images | File | Notes |
|-------|-----------|--------|------|-------|
| train | 3,505 | 657,334 | `train_metadata.parquet` | 80% of studies, pseudo + gold labels |
| validation | 902 | 161,744 | `val_metadata.parquet` | 20% of studies, pseudo + gold labels |
| test | 3 | 557 | `test_metadata.parquet` | Released sample test studies, no labels |
| gold | 58 | 10,528 | `gold_metadata.parquet` | Competition-supplied expert labels only |

The train/validation split is deterministic and study-level: a study is assigned by `md5(study_uid) % 10`, with 0-7 as train and 8-9 as validation. This prevents study-level leakage.

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

## Quick start with `datasets`

Load metadata only (fast, no DICOM decoding):

```python
from datasets import load_dataset

dataset = load_dataset("GodlyDonuts/RSNA-Data", split="train", streaming=True)
for example in dataset:
    print(example["study_uid"], example["ACL"])
    break
```

## Decode images and build a PyTorch DataLoader

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

* The labels are **pseudo-labels** derived from radiologist reports, not from direct image inspection. They are strongest for common, clearly described findings and weakest for rare, subtle, or ambiguous findings.
* The test split contains only the three studies released with the competition test folder. The hidden Kaggle test set is not included.
* This dataset is public because of Hugging Face free-plan storage constraints. Review Kaggle's competition terms before redistributing.
