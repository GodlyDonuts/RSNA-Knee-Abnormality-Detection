# Adapter and public-supervision investigation, September 29

The owner requested a substantial adapter with original DINOv3 weights frozen,
better use of MRI series, and comparison against public labels. This milestone
contains an executable **descriptive** source audit and an experiment design. It
contains no new model, training-label release, GPU dispatch, or submission.

## Measured source findings

Seven Hugging Face repositories were inspected through public metadata APIs.
Six expose image caches/features without separate label CSVs in their file lists;
this does not establish what is inside their archives. `Shio-Koube/knee3`, pinned
to `373f4e1daf36984fa48bf3fc8fd9883fe17cb584`, exposes five label tables downloaded
and size/SHA checked. The frozen [source catalog](sources.json) records revisions,
file hashes, access status, and candidate provenance. No image archives were fetched.

Comparison uses only **2,603 accepted training studies outside fold 0**. The
outer-fold, 441 confirmation studies, 58 official evaluation studies, and quarantine
are excluded before numeric comparison. Source files can contain those rows, but
no statistics or tuning use their values. Labels and masks remain unchanged.
The [comparison](comparison.json) records every denominator and per-target count.

- `frankenstein_labels.csv` and `train_labels_raw.csv` are exactly equal in
  **9 of 12 columns across all 2,599 shared selected studies**: 75% of values.
  These are not independent votes on those targets. The remaining differences
  do not establish independent generation either.
- `frankenstein_labels.csv` has **1,239 selected cells outside [0,1]**, reaching
  **1.2745**. It cannot be passed directly as a BCE probability target. No
  automatic clipping or inferred calibration was applied.
- At a descriptive 0.5 split, the v4 blend agrees with our observed medial
  meniscus labels in **1,907/1,914 (99.63%)** and lateral meniscus in
  **1,859/1,862 (99.84%)** cells. For effusion it calls **727/1,257** of our
  observed negatives above 0.5. This identifies a target-specific disagreement;
  it does not determine which source matches MRI truth. Thresholds here are
  not calibrated confidence estimates, and exact 0.5 values are excluded.
- The Hugging Face v4 blend is **byte-identical** to the official download from
  `stevenleehans/rsna-knee-llm-report-labels` version 6. That author's data card
  describes blending an earlier extraction with lixin73's extraction and
  making gold-guided changes, including an effusion proxy for missing synovitis.
  It is useful for diagnosis of differences, but cannot be treated as an
  independent source of gold validation or automatically imported supervision.
- `yunusgmsoy/rsna-knee-llm-report-labels` version 3 was separately downloaded
  and hashed. Its card declares Gemini-only extraction, **0.05 for unmentioned**,
  **0.5 for uncertain**, and numeric severity categories. Neither 0.05 nor
  distance from 0.5 is reliable evidence of diagnostic confidence here.
  Its row-level agreement has not been included in the five-table comparison.

These findings support comparing sources **per target, with provenance and explicit
missingness**. Intersecting every table would preferentially keep easy/common
examples and can remove the rare cases we need. Source agreement is not accuracy.

## Reproduce

Download only the five exact public objects listed in `sources.json` into an
ignored folder, and save its `download_receipts` as `download-receipts.json` there.
Then, from the repository root:

```sh
PYTHONPATH=. artifacts/qa-env/bin/python \
  initiative/audits/adapter-source-research-v1/compare_sources.py \
  --downloads artifacts/adapter-source-research-v1 \
  --output /tmp/rsna-source-comparison-new.json
```

The script verifies accepted manifest/fold/file hashes, one row per UID, canonical
column ordering, and training-only membership. It refuses to overwrite its output
and never modifies any training artifact. Recompute equality was checked against
an independently written first-pass calculation.

## Proposed next experiment

See [DINOv3 adapter and supervision plan](../../../docs/dinov3-adapter-and-supervision-plan.md).
The first intervention is a rank-32 adapter inside the visual encoder, evaluated
with the existing labels and inputs. Higher-resolution native inputs, sequential
slice modeling, and source-derived confidence are subsequent isolated comparisons.
The rank-64 candidate and coarse-to-fine whole-study design are proposed experiments,
not claims of measured improvement.

## Sources

- [HF source tree](https://huggingface.co/datasets/Shio-Koube/knee3/tree/373f4e1daf36984fa48bf3fc8fd9883fe17cb584)
- [Original v4 source](https://www.kaggle.com/datasets/stevenleehans/rsna-knee-llm-report-labels)
- [Gemini extraction source](https://www.kaggle.com/datasets/yunusgmsoy/rsna-knee-llm-report-labels)
- [Rank-stabilized LoRA](https://arxiv.org/abs/2312.03732)
- [AdaptFormer](https://arxiv.org/abs/2205.13535)
- [Official DINOv3 model card](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m)
- [Competition code constraints](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview)

Author claims about public-label accuracy were not independently reproduced and
were not used to rank sources. Dataset cards and downloads were read through
public APIs; no local reports or credentials were sent to those services.
