# RSNA Knee data contract — checked 2026-09-20

This is a primary-source audit. Kaggle's rendered pages were read directly in the browser where search indexing omitted content. No credentials or private services were accessed. Exact local counts and hashes belong in the independent inventory audit.

## Files and inference contract

`train.csv` contains one study UID, its multilingual report, and twelve binary target columns. Only a small subset has supplied target values. Missing target cells are unobserved labels, not negatives. `train_series.csv` connects study and series UIDs and supplies `Fluid_Sensitive`, `Fat_Suppression`, and `Anatomical_Plane`. The two binary flags must be treated separately; the data page explicitly warns that correlation does not establish equivalence. Planes are Sagittal, Coronal, or Axial.

DICOM layout is `{train|test}_series/<StudyInstanceUID>/<SeriesInstanceUID>/<SOPInstanceUID>.dcm`. Images vary in orientation, spatial resolution, intensity, and transfer syntax; Explicit/Implicit VR Little Endian, JPEG Lossless, and JPEG 2000 are documented. The page lists 86 retained metadata tags.

Public test files contain three example studies. Scoring replaces them with approximately 1,300 studies, matching series metadata and DICOMs. **No test Report column is provided.** Training, public-test, and final-test prevalence need not match. [Official data description](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/data)

## What the supplied labels mean

Host Po-Hao Chen confirms that the supplied reference labels were made from images, independently of reports. They take precedence when report text disagrees. A supplied zero means the finding was annotated absent under the rubric. Bilateral reports/studies occur; organizers reviewed them and adjusted the released text or DICOM metadata to enable disambiguation. Clinical-report disagreement is expected because the reference process uses multiple readers and stricter image-based criteria. **Do not “correct” supplied gold labels from report text.** [Host clarification](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733826)

The official target-specific severity rubric and a versioned extraction protocol are summarized in `work/report_protocol_v2.md`. The authoritative rubric is the pinned host post [Knee Abnormality Detection AI Challenge Overview](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733343). The checked text contains no blanket postoperative/graft rule, no effusion-to-synovitis implication, and no blanket rule accepting every insufficiency or osteochondral lesion. Do not invent those as organizer instructions.

## External data, hosted label extraction, and redistribution

Commercial LLM/API processing of competition reports is expressly permitted under the accessibility and minimal-cost conditions. The host says such processing alone is not prohibited private sharing. Straightforward registration or click-through access generally satisfies accessibility; institution-specific approvals, negotiated agreements, IRB approval, or lengthy credentialing may not. Noncommercial dataset licenses are not automatically excluded merely because prizes exist; dataset owners' agreements still apply. [Host ruling and follow-up](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)

**KneeCoT is explicitly prohibited** because its institutional agreement creates unequal access; the host reserves possible reconsideration if access terms change. No superseding ruling was observed. [KneeCoT ruling](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/734109)

Rules §2.4 restrict redistribution of competition data to nonparticipants and reference RSNA's MIRA license. §2.6 allows accessible external data/tools. §3.4.b prohibits using hand-labeled/human-predicted validation or test records in submissions. §2.8 requires reproducible training/inference code, weights, environment details, and public Kaggle weights for winners. The named winner license is CC-BY-NC 4.0; nearby boilerplate also contains incompatible “no commercial-use restriction” wording, so do not treat this audit as resolving that textual inconsistency. [Current rules](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/rules)

## Scoring and execution

Score is the mean of twelve per-target ROC AUCs. Submit one probability per target per study in the official target order; the file is `submission.csv`. Kaggle inference notebooks have internet disabled and a nine-hour CPU or GPU limit. [Evaluation and code requirements](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/overview/evaluation)

## Correction-search limits

The current pinned discussions and targeted Kaggle discussion searches for updates/corrections were inspected. No organizer-issued downloadable-data replacement notice was found. This does **not** prove no correction exists. The confirmed corrections relevant to this pipeline are conceptual: use the published severity rubric, preserve separate series flags, treat report/gold disagreement as expected, and handle bilateral studies. A fresh authenticated metadata/hash comparison remains necessary to establish whether the archived source files match the current Kaggle release.
